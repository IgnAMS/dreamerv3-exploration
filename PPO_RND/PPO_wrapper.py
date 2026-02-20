import gymnasium as gym
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from sb3_contrib import RecurrentPPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import matplotlib.pyplot as plt
import os
import argparse


log_dir = "./tb_logs/"
os.makedirs(log_dir, exist_ok=True)

# Cookie env
import minigrid
from cookie_env.envs import CornerEnv

# RND
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.running_mean_std import RunningMeanStd

class RNDCallback(BaseCallback):
    def __init__(self, env, verbose=0, lr=1e-4, weight_intrinsic=0.1):
        super().__init__(verbose)
        self.intrinsic_reward_std = RunningMeanStd(shape=())
        self.weight_intrinsic = weight_intrinsic
        self.optimizer = None
        self.predictor = None
        self.target = None
        self.lr = lr
        self.env = env

        self.cum_ext = np.zeros(4, dtype=np.float64)
        self.is_initialized = False
        self._setup_rnd()

    def _setup_rnd(self):
        if self.is_initialized:
            return # Salir si ya existe
            
        print("--- Inicializando Redes RND ---")
        obs_shape = self.env.observation_space.shape
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.target = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(obs_shape), 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        ).to(device)

        self.predictor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(np.prod(obs_shape), 256),
            nn.ReLU(),
            nn.Linear(256, 128)
        ).to(device)

        self.optimizer = optim.Adam(self.predictor.parameters(), lr=self.lr)
        self.is_initialized = True


    def _on_training_start(self) -> None:
        self._setup_rnd()
        self.target.to(self.model.device)
        self.predictor.to(self.model.device)

    def _on_step(self) -> bool:
        # 1. Obtener la observación actual
        obs = torch.FloatTensor(self.locals["new_obs"]).to(self.model.device)

        # 2. Calcular error de RND (Intrinsic Reward)
        with torch.no_grad():
            target_out = self.target(obs)
        predict_out = self.predictor(obs)
        
        # Error cuadrático medio por cada ambiente en paralelo
        intrinsic_reward = torch.pow(target_out - predict_out, 2).mean(dim=1).detach().cpu().numpy()
        
        # Testing
        extrinsic = np.array(self.locals["rewards"], dtype=np.float64)
        self.cum_ext += extrinsic
        
        mean_intrinsic_reward = np.mean(intrinsic_reward)
        self.logger.record("train/intrinsic_reward_mean", mean_intrinsic_reward)
        self.logger.record("train/intrinsic_reward_var", self.intrinsic_reward_std.var)
        self.logger.record("rollout/raw_extrinsic_reward", np.mean(self.locals["rewards"]))
        self.logger.record("rollout/raw_extrinsic_step_mean", np.mean(extrinsic))
        
        # 3. Normalizar la recompensa intrínseca usando RunningMeanStd
        self.intrinsic_reward_std.update(intrinsic_reward)
        intrinsic_reward /= np.sqrt(self.intrinsic_reward_std.var + 1e-8)

        # 4. Inyectar la recompensa al agente
        rewards = self.locals["rewards"].astype(np.float64)
        rewards += intrinsic_reward * self.weight_intrinsic
        self.locals["rewards"] = rewards

        # detectar episodios que terminaron y loggear la suma extrínseca por episodio
        infos = self.locals.get("infos", [])
        for i, info in enumerate(infos):
            if info is None:
                continue
            ep = info.get("episode")  # Monitor añade esto cuando termina un episodio
            if ep is not None:
                # ep['r'] normalmente es la suma de rewards que Monitor guardó
                self.logger.record("rollout/extrinsic_episode_return", self.cum_ext[i])
                # reset del acumulador para ese env
                self.cum_ext[i] = 0.0

        # 5. Actualizar la red Predictor
        loss = torch.pow(target_out.detach() - predict_out, 2).mean()
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return True

    def update_predictor(self, obs_numpy, device):
        """Método para entrenar el RND sin afectar al agente"""
        obs = torch.FloatTensor(obs_numpy).to(device)
        
        # Obtenemos las salidas para calcular el loss
        with torch.no_grad():
            target_out = self.target(obs)
        predict_out = self.predictor(obs)
        
        # Actualizamos el RunningMeanStd de las observaciones si lo usas, 
        # o simplemente el predictor
        loss = torch.pow(target_out.detach() - predict_out, 2).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        # También actualizamos el RMS de la recompensa para que empiece estable
        with torch.no_grad():
            intrinsic_reward = torch.pow(target_out - predict_out, 2).mean(dim=1).cpu().numpy()
            self.intrinsic_reward_std.update(intrinsic_reward)
            

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
    env = SubprocVecEnv([make_env()])
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


def pre_train_RND(rnd_callback, env, device, pre_train_steps=20000):
    print("Iniciando fase de pre-entrenamiento del RND (movimientos al azar)...")
    obs = env.reset()

    for step in range(pre_train_steps):
        # 1. Acciones totalmente al azar
        actions = np.array([env.action_space.sample() for _ in range(env.num_envs)])
        
        # 2. Ejecutar paso en el env
        new_obs, rewards, dones, infos = env.step(actions)
        
        # 3. Entrenar el RND manualmente (llamando al método del callback)
        # Necesitamos pasarle las observaciones al callback para que actualice el predictor
        rnd_callback.update_predictor(new_obs, device)
        
        if step % 1000 == 0:
            print(f"Pre-train RND: {step}/{pre_train_steps}")

    print("RND pre-entrenado. Iniciando entrenamiento del agente...")
    


def train(args):
    num_envs = args.envs
    env = DummyVecEnv([make_env() for _ in range(num_envs)])  
    # --- Visualización de la primera imagen ---
    show_env()
    # ------------------------------------------

    # 3. Configurar RecurrentPPO
    # Nota: SB3 reordena automáticamente los canales de (H,W,C) a (C,H,W) internamente
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
    if args.no_rnd:
        print("\n===== ENTRENANDO SIN RND =====")
        model.learn(
            total_timesteps=args.steps,
        )
    else:
        rnd_callback = RNDCallback(env, weight_intrinsic=args.intrinsic_coef)
        print("\n===== ENTRENANDO CON RND =====")
        print("Pre-training predictor...")
        pre_train_RND(rnd_callback, env, model.device)
        model.learn(
            total_timesteps=args.steps,
            callback=rnd_callback,
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