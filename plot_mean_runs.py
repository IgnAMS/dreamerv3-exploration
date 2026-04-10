import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuración
BASE = Path("~/logdir/dreamer").expanduser()
EXPERIMENTS = {
    'POL multigoalfixed10': ('POL_multigoalfixed10/size12m', '#0022ff'),
    'POL HER hergoalfixed10': ('POL_HER_hergoalfixed10_k4/size12m', '#ff4400'),
#     'POL multigoal9': ('POL_multigoal9/size12m', '#00aa44'),
}
RUNS = ['01', '02', '03', '04', '05']

outdir = Path("plots") / "comparison"
outdir.mkdir(parents=True, exist_ok=True)

def load_run(logdir):
    path = logdir / "scores.jsonl"
    if not path.exists():
        return None, None
    records = []
    for line in path.read_text().strip().split("\n"):
        try:
            records.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    data = sorted(
        [(r["step"], r["episode/score"])
         for r in records if "step" in r and "episode/score" in r],
        key=lambda x: x[0])
    if not data:
        return None, None
    steps, scores = zip(*data)
    return np.array(steps), np.array(scores)

def interpolate_to_common(all_steps, all_scores, n_points=200):
    """Interpola cada run a una grilla común de steps."""
    max_step = min(s[-1] for s in all_steps)
    grid = np.linspace(0, max_step, n_points)
    interp = [np.interp(grid, s, sc) for s, sc in zip(all_steps, all_scores)]
    return grid, np.array(interp)

def moving_avg(x, w):
    return np.convolve(x, np.ones(w) / w, mode='valid')

fig, ax = plt.subplots(figsize=(10, 5))

for label, (rel_path, color) in EXPERIMENTS.items():
    all_steps, all_scores = [], []
    for run in RUNS:
        logdir = BASE / rel_path / run
        steps, scores = load_run(logdir)
        if steps is not None:
            all_steps.append(steps)
            all_scores.append(scores)

    if not all_steps:
        print(f"No data for {label}")
        continue

    grid, matrix = interpolate_to_common(all_steps, all_scores)
    mean   = matrix.mean(axis=0)
    std    = matrix.std(axis=0)

    # Suavizado
    w = max(1, len(grid) // 30)
    grid_s = grid[w-1:]
    mean_s = moving_avg(mean, w)
    std_s  = moving_avg(std,  w)

    ax.fill_between(grid_s, mean_s - std_s, mean_s + std_s,
                    alpha=0.15, color=color)
    ax.plot(grid_s, mean_s, color=color, linewidth=2.0, label=f'{label} (n={len(all_steps)})')

ax.set_xlabel("Steps")
ax.set_ylabel("Episode score")
ax.set_title("Comparación de experimentos (media ± std, 5 runs)")
ax.grid(color='#eeeeee')
ax.legend()
fig.tight_layout()
fig.savefig(outdir / "comparison.png", dpi=150)
print("Saved:", outdir / "comparison.png")