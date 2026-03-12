
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pandas as pd
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import glob

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

def extract_episode_count(log_dir):
    data = []
    event_files = glob.glob(
        f"{log_dir}/**/events.out.tfevents.*",
        recursive=True
    )
    for event_file in event_files:
        parts = event_file.split(os.sep)
        config = parts[-3]
        seed = parts[-2]
        ea = EventAccumulator(event_file)
        ea.Reload()
        if (
            "rollout/ep_len_mean" in ea.Tags()['scalars']
            and "time/total_timesteps" in ea.Tags()['scalars']
        ):

            lens = ea.Scalars("rollout/ep_len_mean")
            steps = ea.Scalars("time/total_timesteps")

            for l, s in zip(lens, steps):
                approx_eps = s.value / l.value

                data.append({
                    "step": s.step,
                    "episodes": approx_eps,
                    "config": config,
                    "seed": seed
                })

    return pd.DataFrame(data)


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

def plot_episode_length(log_dir):
    df = extract_data_from_tb(
        log_dir,
        "rollout/ep_len_mean"
    )
    df = df.rename(columns={"reward": "avg_step"})
    fig, axes = plt.subplots(
        1, 2,
        figsize=(16, 6),
        sharex=True
    )

    # ===== Gráfico completo =====
    sns.lineplot(
        data=df,
        x="step",
        y="avg_step",
        hue="config",
        errorbar="sd",
        ax=axes[0]
    )
    axes[0].set_title("Average Steps per Episode (Full)")
    axes[0].set_xlabel("Training Step")
    axes[0].set_ylabel("Episode Length")

    # ===== Zoom =====
    sns.lineplot(
        data=df,
        x="step",
        y="avg_step",
        hue="config",
        errorbar="sd",
        ax=axes[1],
        legend=False
    )
    axes[1].set_ylim(0, 300)
    axes[1].set_title("Zoom: Episode Length [0, 300]")
    axes[1].set_xlabel("Training Step")

    plt.tight_layout()
    plt.savefig(
        os.path.join(log_dir, "episode_length.png")
    )
    plt.show()


def parse_run_config(df):
    """Extrae rnd, env y size desde el nombre de config"""
    def split_config(cfg):
        parts = cfg.split("_")
        rnd = parts[1]           # Enabled / Disabled
        env = parts[2]           # CornerEnv-v0
        size = parts[3] if len(parts) > 3 else "default"
        return pd.Series([rnd, env, size])

    df[["rnd", "env", "size"]] = df["config"].apply(split_config)

    return df

def plot_comparison_grouped(log_dir):
    print("\nGenerando gráficos comparativos...")
    df = extract_data_from_tb(log_dir)
    if df.empty:
        print("No hay datos.")
        return
    
    df = parse_run_config(df)
    sns.set_theme(style="darkgrid")
    # iterar por environment y size
    for (env, size), df_group in df.groupby(["env", "size"]):
        plt.figure(figsize=(10,6))
        sns.lineplot(
            data=df_group,
            x="step",
            y="reward",
            hue="rnd",
            errorbar="sd"
        )
        plt.title(f"{env} | size={size} | Extrinsic Reward")
        plt.xlabel("Training Steps")
        plt.ylabel("Episode Reward")
        filename = f"reward_{env}_{size}.png"
        plt.savefig(os.path.join(log_dir, filename))
        plt.close()

        print("Guardado:", filename)