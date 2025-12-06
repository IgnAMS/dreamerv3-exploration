import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env

# Importación local directa de los archivos del entorno y el callback
from middlegrid_env import MiddleGridEnv 
from heatmap import HeatmapCallback

# --- Configuración ---
GRID_SIZE = 55
N_ENVS = 4                      # Entornos paralelos
TOTAL_TIMESTEPS = 500000        # Número total de pasos de entrenamiento
HEATMAP_SAVE_FREQ = 1024       # Generar un mapa de calor cada 25,000 pasos
LOG_DIR = "./ppo_logs_middle_grid" # Directorio para logs y mapas de calor

def run_training():
    """
    Configura y entrena el agente PPO con el entorno MiddleGrid y el Callback.
    """
    # 1. Crear directorios
    os.makedirs(LOG_DIR, exist_ok=True)
    print(f"Los logs y los mapas de calor se guardarán en: {LOG_DIR}")

    # 2. Crear entorno vectorizado (SB3 prefiere VEC_ENVS)
    env = make_vec_env(
        MiddleGridEnv, 
        n_envs=N_ENVS, 
        env_kwargs={'size': GRID_SIZE},
        vec_env_cls=DummyVecEnv, 
        monitor_dir=LOG_DIR
    )

    # 3. Inicializar el Callback
    heatmap_callback = HeatmapCallback(
        size=GRID_SIZE, 
        save_freq=HEATMAP_SAVE_FREQ,
        verbose=1,
        log_dir=LOG_DIR,
    )

    # 4. Inicializar el modelo PPO
    model = PPO(
        "CnnPolicy",             # Usamos la política CNN por defecto, adecuada para 96x96x3
        env,
        learning_rate=3e-4,
        n_steps=2048 // N_ENVS,  
        batch_size=64,
        n_epochs=10,
        gamma=0.999,             
        verbose=1,               
        tensorboard_log=LOG_DIR,
        device="auto"
    )

    # 5. Entrenar el modelo
    print("--- Comenzando el Entrenamiento PPO ---")
    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=heatmap_callback,
        reset_num_timesteps=True
    )
    print("--- Entrenamiento Finalizado ---")

    # 6. Guardar el modelo final
    model.save(os.path.join(LOG_DIR, "ppo_middle_grid_final.zip"))
    
    # 7. Generar el mapa de calor final al terminar
    heatmap_callback.generate_heatmap()


if __name__ == "__main__":
    run_training()