import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

import gymnasium as gym
from minigrid.wrappers import RGBImgPartialObsWrapper, ImgObsWrapper
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import copy
import torch
import random
import pandas as pd
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob

# entorno cookie
import minigrid
from cookie_env.envs import CornerEnv

from PPO_RND.PPO_RND_wrapper import (
    RNDModel,
    pretrain_RND,
    RNDVecEnv,
    RNDTrainCallback
)


log_dir = "./tb_logs/"
os.makedirs(log_dir, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def set_seed(seed):
    """Fija todas las semillas para reproducibilidad"""
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def make_env(args):
    # Esta función crea un solo ambiente
    def _init():
        env = gym.make(args.env, render_mode="rgb_array", size=args.env_size)
        env = RGBImgPartialObsWrapper(env, tile_size=32)
        env = ImgObsWrapper(env)
        env = Monitor(env)
        return env
    return _init

def show_env(args):
    # 1. Creamos el entorno con tus wrappers
    init_fn = make_env(args)
    env = init_fn()
    obs, info = env.reset()
    print(f"Forma de la observación (H, W, C): {obs.shape}")
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


def train(args, current_seed):
    set_seed(current_seed)
    num_envs = args.envs
    base_env = DummyVecEnv([make_env(args) for _ in range(num_envs)])  
    
    rnd_name = "RND_Enabled" if not args.no_rnd else "RND_Disabled"
    env_name = args.env
    env_size = args.env_size
    run_id = f"{rnd_name}_{env_name}_{env_size}/seed_{current_seed}"    
    current_log_dir = os.path.join(args.log_dir, run_id)
    
    # --- Visualización de la primera imagen ---
    show_env(args)
    
    if args.no_rnd:
        print("\n===== ENTRENANDO SIN RND =====")        
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

    model = PPO(
        "CnnPolicy",
        env,
        verbose=0,
        learning_rate=3e-4,
        n_steps=128,
        device="auto",
        tensorboard_log=args.log_dir,
    )
            
    model.learn(
        total_timesteps=args.steps,
        callback=callback,
        tb_log_name=run_id
    )
        
    model.save(os.path.join(current_log_dir, "model"))
    env.close()

def extract_data_from_tb(log_dir, tag="rollout/ep_rew_mean"):
    """Extrae datos de los archivos de TensorBoard a un DataFrame de Pandas"""
    data_list = []
    # Buscamos todos los archivos tfevents en las subcarpetas
    event_files = glob.glob(f"{log_dir}/**/events.out.tfevents.*", recursive=True)
    
    for event_file in event_files:
        # Identificar configuración y semilla por el path
        parts = event_file.split(os.sep)
        # Ajusta esto según cómo guardes las carpetas: tb_logs/RND_Enabled/seed_1/...
        config = parts[-3] 
        seed = parts[-2]
        
        ea = EventAccumulator(event_file)
        ea.Reload()
        
        if tag in ea.Tags()['scalars']:
            for event in ea.Scalars(tag):
                data_list.append({
                    "step": event.step,
                    "reward": event.value,
                    "config": config,
                    "seed": seed
                })
    return pd.DataFrame(data_list)

def plot_comparison(log_dir):
    """Genera el gráfico comparativo de Extrinsic Reward"""
    print("\nGenerando gráfico comparativo...")
    df = extract_data_from_tb(log_dir)
    
    if df.empty:
        print("No se encontraron datos para graficar. Asegúrate de que TensorBoard haya guardado eventos.")
        return

    plt.figure(figsize=(10, 6))
    sns.set_theme(style="darkgrid")
    
    # El lineplot de seaborn promedia automáticamente las semillas y dibuja la desviación estándar
    sns.lineplot(data=df, x="step", y="reward", hue="config", ci="sd")
    
    plt.title("RND vs No-RND: Rendimiento en CornerEnv (Extrinsic Reward)")
    plt.xlabel("Pasos de entrenamiento")
    plt.ylabel("Recompensa Media del Episodio")
    plt.legend(title="Configuración")
    
    plot_path = os.path.join(log_dir, "comparativa_rendimiento.png")
    plt.savefig(plot_path)
    plt.show()
    print(f"Gráfico guardado en: {plot_path}")


def main(args):
    os.makedirs(args.log_dir, exist_ok=True)
    if args.single_train:
        train(args, args.seed)
    else:
        for seed in range(1, 5):
            # RND
            args_rnd = copy.deepcopy(args)
            args_rnd.no_rnd = False
            train(args_rnd, seed)
            
            # Sin RND
            args_no_rnd = copy.deepcopy(args)
            args_no_rnd.no_rnd = True
            train(args_no_rnd, seed)
        plot_comparison(args.log_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--env", type=str, default="CornerEnv-v0")
    parser.add_argument("--env-size", type=int, default=55)
    parser.add_argument("--single-train", action="store_true", help="Solo correr una config")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=str, default="./tb_logs/")
    
    parser.add_argument("--no-rnd", action="store_true", help="Desactivar RND")
    parser.add_argument("--intrinsic-coef", type=float, default=0.005)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1_000_000)
    parser.add_argument("--pretrain", type=int, default=2000)
    parser.add_argument("--gamma", type=float, default=0.99)
    args = parser.parse_args()
    
    
    main(args)