import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

import gymnasium as gym
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.running_mean_std import RunningMeanStd

# tu entorno cookie
import minigrid
from cookie_env.envs import CornerEnv

log_dir = "./tb_logs/"
os.makedirs(log_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def conv_output_size(model_conv, c, h, w, device):
    """Run a dummy forward to compute flattened size after convs"""
    with torch.no_grad():
        inp = torch.zeros(1, c, h, w, device=device)
        out = model_conv(inp)
        return int(np.prod(out.shape[1:]))

def preprocess_obs(obs, device):
    """
    Acepta obs en:
      - numpy array (N, H, W, C) uint8 o float
      - torch tensor (N, H, W, C) o (N, C, H, W)
    Devuelve tensor torch (N, C, H, W), float32 en [0,1], en device.
    """
    if isinstance(obs, np.ndarray):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
    elif isinstance(obs, torch.Tensor):
        obs_t = obs.to(device=device, dtype=torch.float32)
    else:
        raise TypeError("obs must be numpy array or torch tensor")

    if obs_t.ndim != 4:
        raise ValueError(f"Expected obs with 4 dims (N,H,W,C) or (N,C,H,W), got {obs_t.shape}")

    # Caso común: (N, H, W, C)
    if obs_t.shape[-1] in (1, 3):  # HWC
        obs_t = obs_t.permute(0, 3, 1, 2)  # -> (N, C, H, W)
    obs_t = obs_t / 255.0

    return obs_t  # torch.Tensor (N, C, H, W)

class RNDModel(nn.Module):
    def __init__(self, obs_shape, feature_dim=512, device="cpu"):
        super().__init__()
        print("RND device:", device)
        c,h,w = obs_shape
        self.device = device

        self._target_conv = nn.Sequential(
            nn.Conv2d(c, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        ).to(device)
        flat_size = conv_output_size(self._target_conv, c, h, w, device)

        self._target_fc = nn.Sequential(
            nn.Linear(flat_size, feature_dim)
        ).to(device)
        
        self._predictor_conv = nn.Sequential(
            nn.Conv2d(c, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        ).to(device)
        self._predictor_fc = nn.Sequential(
            nn.Linear(flat_size, feature_dim)
        ).to(device)

        self.target = nn.Sequential(self._target_conv, self._target_fc)
        self.predictor = nn.Sequential(self._predictor_conv, self._predictor_fc)

        for p in self.target.parameters():
            p.requires_grad = False

        self.opt = torch.optim.Adam(self.predictor.parameters(), lr=1e-4)

    def _forward_features(self, obs):
        """Return (pred, target) given obs (B,C,H,W) tensor"""
        pred = self.predictor(obs)
        with torch.no_grad():
            targ = self.target(obs)
        return pred, targ

    def compute_intrinsic_reward(self, obs):
        """
        obs: torch tensor (B, C, H, W) float in [0,1]
        returns: numpy array (B,) of MSE per-sample
        """
        pred, targ = self._forward_features(obs)
        # MSE per sample across feature dim
        mse = F.mse_loss(pred, targ, reduction="none")
        # mse shape (B, feature_dim) -> mean over features
        per_sample = mse.mean(dim=1)
        return per_sample.detach()

    def train_step(self, obs):
        """
        obs: torch tensor (B, C, H, W) float in [0,1]
        Performs one optimizer step on predictor.
        Returns loss float.
        """
        pred, targ = self.predictor(obs), self.target(obs).detach()
        loss = F.mse_loss(pred, targ)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return loss.item()

class RNDVecEnv(VecEnvWrapper):
    def __init__(
        self, 
        venv, 
        rnd_model, 
        device, 
        intrinsic_coef=0.005, 
        gamma=0.99, 
        eps=1e-8
    ):
        super().__init__(venv)
        self.rnd = rnd_model
        self.device = device
        self.beta = intrinsic_coef
        self.gamma = gamma
        self.eps = eps
        
        # Running return normalization state
        self.ret_rms = RunningMeanStd(shape=())
        self.running_returns = np.zeros(self.num_envs, dtype=np.float64)

    def step_wait(self):
        # 1. Obtenemos las variables originales (obs es NumPy array)
        obs, rewards, dones, infos = self.venv.step_wait()  # obs: (N, H, W, C)
        
        # 2. Preprocesamos solo para calcular RND
        obs_t = preprocess_obs(obs, self.device)
        with torch.no_grad():
            intrinsic_t = self.rnd.compute_intrinsic_reward(obs_t)  # torch tensor (N,)
        intrinsic = intrinsic_t.cpu().numpy().astype(np.float64)

        # 3. Normalizamos y combinamos la recompensa
        self.running_returns = self.running_returns * self.gamma + intrinsic
        self.ret_rms.update(self.running_returns)
        norm_intrinsic = intrinsic / (np.sqrt(self.ret_rms.var + self.eps))

        # add to external rewards
        rewards = rewards.astype(np.float64) + self.beta * norm_intrinsic

        # 4. Registramos los logs y reseteamos retornos si hay "done"
        for i in range(len(infos)):
            infos[i]["intrinsic_reward"] = float(norm_intrinsic[i])
            infos[i]["raw_intrinsic"] = float(intrinsic[i])

            # if episode resets, reset running returns for that env
            if dones[i].any() if isinstance(dones[i], (list, np.ndarray)) else dones[i]:
                # many VecEnv implementations return boolean or array; handle usual booleans
                self.running_returns[i] = 0.0

        return obs, rewards, dones, infos

    def reset(self):
        obs = self.venv.reset()
        self.last_obs = obs
        return obs


class RNDTrainCallback(BaseCallback):
    def __init__(self, rnd_model: RNDModel, device, batch_size=128):
        super().__init__()
        self.rnd = rnd_model
        self.device = device
        self.batch_size = batch_size
        self.obs_buffer = []

    def _on_step(self) -> bool:
        """
        Train predictor using the latest new_obs (batch) from the rollouts.
        """
        # Escribir intrinsic y extrinsic reward
        infos = self.locals["infos"]
        intr_rewards = [info["intrinsic_reward"] for info in infos]
        avg_intr = np.mean(intr_rewards)
        self.logger.record("rollout/intrinsic_reward_avg", avg_intr)
        raw_intr = [info["raw_intrinsic"] for info in infos]
        self.logger.record("rollout/intrinsic_reward_raw", np.mean(raw_intr))
        
        obs = self.locals["new_obs"]
        self.obs_buffer.append(obs)
        
        num_envs = obs.shape[0]
        total_samples = len(self.obs_buffer) * num_envs
        if total_samples >= self.batch_size:
            # Concatenamos todo en un solo array de NumPy (B, H, W, C)
            batch_obs = np.concatenate(self.obs_buffer, axis=0)
            
            # Preprocesamos al tensor (B, C, H, W) en la GPU/CPU
            obs_t = preprocess_obs(batch_obs, self.device)
            
            # Entrenamos la red predictora
            _ = self.rnd.train_step(obs_t)
            
            # Limpiamos el buffer para el siguiente lote
            self.obs_buffer = []
        
        
        return True
            

def make_env():
    # Esta función crea un solo ambiente
    def _init():
        env = gym.make("CornerEnv-v0", render_mode="rgb_array", size=55)
        env = RGBImgPartialObsWrapper(env, tile_size=32)
        env = ImgObsWrapper(env)
        env = Monitor(env)
        return env
    return _init

def show_env():
    env = DummyVecEnv([make_env()])
    obs = env.reset()[0]
    print(f"Forma de la observación (H, W, C): {obs.shape}")

    # 1. Creamos el entorno con tus wrappers
    init_fn = make_env()
    env = init_fn()
    obs, info = env.reset()
    global_view = env.unwrapped.render()
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(obs)
    ax[0].set_title(f"Obs Agente (CNN Input)\nShape: {obs.shape}")
    ax[0].axis('off')
    ax[1].imshow(global_view)
    ax[1].set_title("Vista Global del Entorno")
    ax[1].axis('off')
    plt.tight_layout()
    plt.show()
    env.close()

def pretrain_RND(rnd_model: RNDModel, base_env: DummyVecEnv, device, pre_train_steps=20000):
    print("Iniciando pre-entrenamiento RND (acciones aleatorias)...")
    obs = base_env.reset()
    for step in range(pre_train_steps):
        actions = np.array([base_env.action_space.sample() for _ in range(base_env.num_envs)])
        new_obs, rewards, dones, infos = base_env.step(actions)
        
        obs_t = preprocess_obs(new_obs, device)
        rnd_model.train_step(obs_t)
        
        if step % 100 == 0:
            print(f"Pre-train RND: {step}/{pre_train_steps}")
    print("Pre-train RND finalizado.")


def train(args):
    num_envs = args.envs
    base_env = DummyVecEnv([make_env() for _ in range(num_envs)])  
    
    # --- Visualización de la primera imagen ---
    show_env()
    
    if args.no_rnd:
        print("\n===== ENTRENANDO SIN RND =====")        
        rnd_model = None
        callback = None
        env = base_env
    else:
        print("\n===== ENTRENANDO CON RND =====")
        # create RND model with obs shape (C,H,W)
        # base_env.observation_space.shape -> (H, W, C)
        h, w, c = base_env.observation_space.shape
        obs_shape = (c, h, w)
        rnd_model = RNDModel(obs_shape, feature_dim=512, device=device).to(device)

        # pre-train predictor on random policy using base_env (without wrapper)
        pretrain_RND(rnd_model, base_env, device, pre_train_steps=args.pretrain)

        # wrap the env AFTER pretraining so we add intrinsic reward to PPO
        env = RNDVecEnv(base_env, rnd_model, device, intrinsic_coef=args.intrinsic_coef, gamma=args.gamma)

        # create callback to keep training predictor during PPO training
        callback = RNDTrainCallback(rnd_model, device)


    """
    model = RecurrentPPO(
        "CnnLstmPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.0003,
        n_steps=128,
        device="auto",
        tensorboard_log=log_dir,
    )
    """
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=128,
        device="auto",
        tensorboard_log=log_dir,
    )
            
    model.learn(
        total_timesteps=args.steps,
        callback=callback,
    )
        
    name = "ppo" if args.no_rnd else "ppo_rnd"
    model.save(os.path.join(log_dir, name))
    print("Modelo guardado con éxito.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--no-rnd", action="store_true",
                        help="Desactivar RND intrinsic reward")
    parser.add_argument("--intrinsic-coef", type=float, default=0.005,
                        help="Peso del reward intrinseco")
    parser.add_argument("--envs", type=int, default=4,
                        help="Numero de ambientes paralelos")
    parser.add_argument("--steps", type=int, default=1_000_000,
                        help="Total timesteps")

    parser.add_argument("--pretrain", type=int, default=2000,
                        help="Pasos de pre-entrenamiento RND (acciones aleatorias)")
    parser.add_argument("--gamma", type=float, default=0.99,
                        help="Gamma para discounted return de intrinsic RMS")
    
    args = parser.parse_args()
    train(args)