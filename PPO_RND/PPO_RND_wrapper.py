import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from stable_baselines3.common.vec_env import DummyVecEnv, VecEnvWrapper
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.running_mean_std import RunningMeanStd


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
        Solo para logging.
        """
        infos = self.locals["infos"]
        intr_rewards = [info.get("intrinsic_reward", 0) for info in infos]
        self.logger.record("rollout/intrinsic_reward_avg", np.mean(intr_rewards))
        
        raw_intr = [info.get("raw_intrinsic", 0) for info in infos]
        self.logger.record("rollout/intrinsic_reward_raw", np.mean(raw_intr))
        return True

    def _on_rollout_end(self) -> None:
        """
        Aquí ocurre la optimización (n_opt) como dice el paper.
        """
        # 1. Obtenemos los hiperparámetros de PPO
        n_epochs = self.model.n_epochs  # Este es el 'n_opt' del paper
        batch_size = self.batch_size or self.model.batch_size
        
        # 2. Extraemos todas las observaciones del buffer de PPO
        # (N_STEPS * N_ENVS, H, W, C)
        obs_buffer = self.model.rollout_buffer.observations
        
        # Aplanamos los ejes de envs y steps para tener un pool de muestras
        shape = obs_buffer.shape
        flat_obs = obs_buffer.reshape((-1,) + shape[2:]) 
        
        total_samples = flat_obs.shape[0]
        
        # 3. Ciclo de optimización: 'for _ in range(n_opt)'
        indices = np.arange(total_samples)
        
        for _ in range(n_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, total_samples, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                
                # Extraemos el batch Bi del paper
                batch_obs = flat_obs[batch_idx]
                
                # Preprocesamos y entrenamos la red predictora (Distillation loss)
                obs_t = preprocess_obs(batch_obs, self.device)
                loss = self.rnd.train_step(obs_t)
                
        self.logger.record("train/rnd_loss", loss)

class RNDTrainCallback(BaseCallback):
    def __init__(self, rnd_model: RNDModel, device, batch_size=None):
        super().__init__()
        self.rnd = rnd_model
        self.device = device
        self.batch_size = batch_size
        self.rnd_update_count = 0  # Contador de minibatches procesados por RND

    def _on_step(self) -> bool:
        """
        Logging de recompensas durante la recolección.
        """
        infos = self.locals["infos"]
        # Usamos .get() por seguridad si alguna info no trae la llave
        intr_rewards = [info["intrinsic_reward"] for info in infos]
        raw_intr = [info["raw_intrinsic"] for info in infos]
        
        self.logger.record("rollout/intrinsic_reward_avg", np.mean(intr_rewards))
        self.logger.record("rollout/intrinsic_reward_raw", np.mean(raw_intr))
        return True

    def _on_rollout_end(self) -> None:
        """
        Sincronización exacta con el ciclo de optimización de PPO.
        """
        #  Leer parámetros del paper directamente de PPO
        n_epochs = self.model.n_epochs
        batch_size = self.batch_size or self.model.batch_size 
        
        # Extraer y preparar datos 
        # El buffer tiene forma (n_steps, n_envs, H, W, C)
        obs_buffer = self.model.rollout_buffer.observations
        
        # Aplanamos para tener (N * K) muestras totales
        flat_obs = obs_buffer.reshape((-1,) + obs_buffer.shape[2:])
        total_samples = flat_obs.shape[0]
        
        indices = np.arange(total_samples)
        last_loss = 0
        for _ in range(n_epochs):
            np.random.shuffle(indices)
            for start in range(0, total_samples, batch_size):
                end = start + batch_size
                batch_idx = indices[start:end]
                batch_obs = flat_obs[batch_idx]
                obs_t = preprocess_obs(batch_obs, self.device)
                last_loss = self.rnd.train_step(obs_t)
                
                # Aumentamos contador de actualizaciones de red (minibatches)
                self.rnd_update_count += 1
        
        #  Logs de actualizaciones y pérdidas 
        self.logger.record("train/ppo_update_cycles", self.model._n_updates)
        self.logger.record("train/rnd_total_gradient_steps", self.rnd_update_count)
        self.logger.record("train/rnd_loss", last_loss)

        # Log informativo de qué estamos usando
        if self.model._n_updates == 1:
            print(f"RND configurado: n_epochs={n_epochs}, batch_size={batch_size}, total_samples={total_samples}")


def pretrain_RND(rnd_model: RNDModel, base_env: DummyVecEnv, device, pre_train_steps=20000):
    print("Iniciando pre-entrenamiento RND (acciones aleatorias)...")
    obs = base_env.reset()
    for step in range(pre_train_steps):
        actions = np.array([base_env.action_space.sample() for _ in range(base_env.num_envs)])
        new_obs, rewards, dones, infos = base_env.step(actions)
        obs_t = preprocess_obs(new_obs, device)
        # rnd_model.train_step(obs_t)
        
        if step % 100 == 0:
            print(f"Pre-train RND: {step}/{pre_train_steps}")
    print("Pre-train RND finalizado.")