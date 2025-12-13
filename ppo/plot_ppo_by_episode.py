import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings

def process_folder(folder: Path, rolling_window: int = 1):
    print(f"\nProcesando carpeta: {folder}")
    monitor_files = sorted(folder.rglob("*.monitor.csv"))
    if not monitor_files:
        print("  > No se encontraron archivos '*.monitor.csv' en esta carpeta.")
        return

    per_file_avgs = []
    per_file_names = []
    all_episode_returns = []
    gamma = 0.997
    for f in monitor_files:
        df = pd.read_csv(f, comment='#')
        if df.empty:
            continue

        rewards = df['r'].astype(float).to_numpy()
        lengths = df["l"].astype(int).to_numpy()
        returns = (gamma ** (lengths - 1)) * rewards
        
        per_file_avgs.append(np.nanmean(returns))
        per_file_names.append(f.name)
        all_episode_returns.extend(returns.tolist())

    if not per_file_avgs:
        print("  > No hay datos de reward válidos para plotear.")
        return

    # Crear carpeta de salida plots
    out_dir = folder / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Bar chart: promedio por archivo
    plt.figure(figsize=(10, 5))
    x = np.arange(len(per_file_avgs))
    plt.bar(x, per_file_avgs, color='C0')
    plt.xticks(x, per_file_names, rotation=45, ha='right', fontsize=8)
    plt.xlabel("Archivo monitor")
    plt.ylabel("Reward promedio (episodic sum)")
    plt.title(f"{folder.name} — Reward promedio por archivo")
    plt.tight_layout()
    file1 = out_dir / "per_file_avg_reward.png"
    plt.savefig(file1)
    plt.close()
    print(f"  > Guardado: {file1}")

    # 2) Episode-level plot con moving average
    returns_arr = np.array(all_episode_returns, dtype=float)
    episodes = np.arange(len(returns_arr))
    
    # compute moving average (centered = False)
    window = max(1, rolling_window)
    if len(returns_arr) < window:
        window = 1
        ma = pd.Series(returns_arr).rolling(window=1).mean().to_numpy()
    else:
        ma = pd.Series(returns_arr).rolling(window=window, min_periods=1).mean().to_numpy()

    plt.figure(figsize=(12, 6))
    plt.scatter(episodes, returns_arr, s=6, alpha=0.25, label="return (episodios)")
    plt.plot(episodes, ma, color='C1', lw=2, label=f"MA (window={window})")
    plt.xlabel("Episodio")
    plt.ylabel("Return (episodic sum)")
    plt.title(f"{folder.name} — Retorno por episodio (MA window={window})")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    file2 = out_dir / "episodes_moving_avg.png"
    plt.savefig(file2)
    plt.close()
    print(f"  > Guardado: {file2}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", type=str, required=True,
                        help="Directorio padre que contiene subcarpetas (ej: /ppo_logs/cornerenv_22)")
    parser.add_argument("--rolling-window", type=int, default=50,
                        help="Ventana para la media movil en el plot por episodio.")
    args = parser.parse_args()

    base = Path(args.base_dir).expanduser().resolve()
    if not base.exists():
        raise SystemExit(f"Directorio no existe: {base}")

    # enumerar subcarpetas (solo primer nivel)
    subfolders = [p for p in sorted(base.iterdir()) if p.is_dir()]

    # si la carpeta base contiene directamente monitor files sin subcarpetas, tratarla también
    if not subfolders:
        print(f"No hay subcarpetas en {base}, procesando {base} directamente.")
        process_folder(base, rolling_window=args.rolling_window)
        return

    for sub in subfolders:
        process_folder(sub, rolling_window=args.rolling_window)

if __name__ == "__main__":
    main()
    # python3 ppo/plot_ppo_by_episode.py --base-dir ./ppo_logs/cornerenv_22