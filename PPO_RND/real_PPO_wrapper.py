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


class RNDModel(nn.Module):
    def __init__(self, obs_shape, feature_dim=512):
        super().__init__()

        c,h,w = obs_shape
        self.device = device

        self._target_conv = nn.Sequential(
            nn.Conv2d(c, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        flat_size = conv_output_size(self._target_conv, c, h, w, device)

        self._target_fc = nn.Sequential(
            nn.Linear(flat_size, feature_dim)
        )
        
        self._predictor_conv = nn.Sequential(
            nn.Conv2d(c, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten()
        )
        self._predictor_fc = nn.Sequential(
            nn.Linear(flat_size, feature_dim)
        )

        self.target = nn.Sequential(self._target_conv, self._target_fc).to(device)
        self.predictor = nn.Sequential(self._predictor_conv, self._predictor_fc).to(device)

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

    def _normalize_obs(self, obs_np):
        # obs_np expected HWC per env -> convert to tensor CHW and scale to [0,1]
        obs_t = torch.tensor(obs_np, dtype=torch.float32, device=self.device)  # (N, H, W, C)
        if obs_t.ndim == 4:
            # to (N, C, H, W)
            obs_t = obs_t.permute(0, 3, 1, 2)
        obs_t = obs_t / 255.0
        return obs_t

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()  # obs: (N, H, W, C)
        # compute intrinsic reward
        obs_t = self._normalize_obs(obs)
        with torch.no_grad():
            intrinsic_t = self.rnd.compute_intrinsic_reward(obs_t)  # torch tensor (N,)
        intrinsic = intrinsic_t.cpu().numpy().astype(np.float64)

        # update discounted running returns for normalization
        self.running_returns = self.running_returns * self.gamma + intrinsic
        # update RMS
        self.ret_rms.update(self.running_returns)
        # normalize intrinsic by RMS of discounted returns
        norm_intrinsic = intrinsic / (np.sqrt(self.ret_rms.var + self.eps))

        # add to external rewards
        rewards = rewards.astype(np.float64) + self.beta * norm_intrinsic

        # logging to infos
        for i in range(len(infos)):
            infos[i]["intrinsic_reward"] = float(norm_intrinsic[i])
            infos[i]["raw_intrinsic"] = float(intrinsic[i])

            # if episode resets, reset running returns for that env
            if dones[i].any() if isinstance(dones[i], (list, np.ndarray)) else dones[i]:
                # many VecEnv implementations return boolean or array; handle usual booleans
                self.running_returns[i] = 0.0

        return obs, rewards, dones, infos

class RNDTrainCallback(BaseCallback):
    def __init__(self, rnd_model: RNDModel, device):
        super().__init__()
        self.rnd = rnd_model
        self.device = device

    def _on_step(self) -> bool:
        """
        Train predictor using the latest new_obs (batch) from the rollouts.
        """
        new_obs = self.locals.get("new_obs")
        if new_obs is None:
            return True

        # new_obs shape (N, H, W, C)
        obs_t = torch.tensor(new_obs, dtype=torch.float32, device=self.device)
        if obs_t.ndim == 4:
            obs_t = obs_t.permute(0, 3, 1, 2)
        obs_t = obs_t / 255.0
        _ = self.rnd.train_step(obs_t)
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
    
    plt.figure(figsize=(5,5))
    plt.imshow(obs)
    plt.title("Vista Parcial RGB del Agente")
    plt.axis('off')
    plt.show()
    env.close()

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

def pre_train_RND(rnd_model: RNDModel, base_env: DummyVecEnv, device, pre_train_steps=20000):
    print("Iniciando pre-entrenamiento RND (acciones aleatorias)...")
    # reset running env
    obs = base_env.reset()
    for step in range(pre_train_steps):
        # sample random actions per env
        actions = np.array([base_env.action_space.sample() for _ in range(base_env.num_envs)])
        new_obs, rewards, dones, infos = base_env.step(actions)
        # prepare obs tensor for train
        obs_t = torch.tensor(new_obs, dtype=torch.float32, device=device)
        if obs_t.ndim == 4:
            obs_t = obs_t.permute(0, 3, 1, 2)
        obs_t = obs_t / 255.0
        rnd_model.train_step(obs_t)
        if step % 1000 == 0:
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
        pre_train_RND(rnd_model, base_env, device, pre_train_steps=args.pretrain)

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
        n_steps=128,      # Tamaño de la ventana de experiencia
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
                        help="Activar RND intrinsic reward")
    parser.add_argument("--intrinsic-coef", type=float, default=0.005,
                        help="Peso del reward intrinseco")
    parser.add_argument("--envs", type=int, default=4,
                        help="Numero de ambientes paralelos")
    parser.add_argument("--steps", type=int, default=1_000_000,
                        help="Total timesteps")
    args = parser.parse_args()
    train(args)