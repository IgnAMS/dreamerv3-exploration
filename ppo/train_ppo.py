import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

# Importación local directa de los archivos del entorno y el callback
from middlegrid_env import MiddleGridEnv, CornerEnv
from heatmap import HeatmapCallback


def run_training(
    *,
    grid_size,
    n_envs,
    total_timesteps,
    heatmap_save_freq,
    n_steps,
    batch_size,
    learning_rate,
    log_dir,
    make_env,
    gamma,
    **kwargs
):
    """
    Configura y entrena el agente PPO con el entorno MiddleGrid y el Callback.
    """
    print("Entrenando con:", grid_size, n_envs, log_dir)
    print("kwargs:", kwargs)
    # 1. Crear directorios
    os.makedirs(log_dir, exist_ok=True)
    print(f"Los logs y los mapas de calor se guardarán en: {log_dir}")

    # 2. Crear entorno vectorizado (SB3 prefiere VEC_ENVS)
    env = make_vec_env(
        make_env, 
        n_envs=n_envs, 
        env_kwargs={'size': grid_size},
        vec_env_cls=DummyVecEnv, 
        monitor_dir=log_dir
    )

    # 3. Inicializar el Callback
    heatmap_callback = HeatmapCallback(
        size=grid_size, 
        save_freq=heatmap_save_freq // n_envs,
        verbose=1,
        # log_dir=LOG_DIR,
    )

    # 4. Inicializar el modelo PPO
    model = PPO(
        "CnnPolicy",
        env,
        learning_rate=learning_rate,
        n_steps=n_steps,  
        batch_size=batch_size,
        n_epochs=10,
        gamma=gamma,
        verbose=1,               
        tensorboard_log=log_dir,
        device="auto"
    )

    # 5. Entrenar el modelo
    print("--- Comenzando el Entrenamiento PPO ---")
    model.learn(
        total_timesteps=total_timesteps,
        callback=heatmap_callback,
        reset_num_timesteps=True
    )
    print("--- Entrenamiento Finalizado ---")

    # 6. Guardar el modelo final
    model.save(os.path.join(log_dir, "ppo_middle_grid_final.zip"))
    heatmap_callback.save_heatmap()

def probar_multiples_parametros():
    
    """
    Default PPO:
        learning_rate=0.0003, 
        n_steps=2048, 
        batch_size=64, 
        n_epochs=10, 
        gamma=0.99, 
        gae_lambda=0.95, 
        clip_range=0.2, 
        clip_range_vf=None, 
        normalize_advantage=True, 
        ent_coef=0.0, 
        vf_coef=0.5, 
        max_grad_norm=0.5, 
    """
    
    experimentos = [
        {
            "grid_size": 22, "n_envs": 8, "total_timesteps": 600000,
            "heatmap_save_freq": 1000,
            "batch_size": 64,
            "n_steps": 2048,
            "learning_rate": 1e-4, 
            "gamma": 0.997,
            "exp_description": f"600k_8_envs_n_step_{2048}",
            "make_env": CornerEnv, "env_name": "cornerenv"
        },
        {
            "grid_size": 22, "n_envs": 8, "total_timesteps": 600000,
            "heatmap_save_freq": 1000,
            "batch_size": 64,
            "n_steps": 2048 // 128,
            "learning_rate": 1e-4, 
            "gamma": 0.997,
            "exp_description": f"600k_8_envs_n_step_{2048 // 128}",
            "make_env": CornerEnv, "env_name": "cornerenv"
        },
        {
            "grid_size": 22, "n_envs": 128, "total_timesteps": 600000,
            "heatmap_save_freq": 1000,
            "batch_size": 64,
            "n_steps": 2048 // 128,
            "learning_rate": 1e-4, 
            "gamma": 0.997,
            "exp_description": f"600k_128_envs_n_step_{2048 // 128}",
            "make_env": CornerEnv, "env_name": "cornerenv"
        },
    ]
    for exp in experimentos:
        logdir = f"./ppo_logs/cornerenv22_run3/{exp['exp_description']}"
        run_training(
            log_dir=logdir,
            **exp,
        )


if __name__ == "__main__":
    probar_multiples_parametros()