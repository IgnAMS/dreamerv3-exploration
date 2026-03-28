import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

logdir = Path("~/logdir/dreamer/POL_multigoal9/size12m/03").expanduser()
outdir = Path("plots") / f"single_run_POL_multigoal9_size12m_03"
outdir.mkdir(parents=True, exist_ok=True)

# Cargar scores.jsonl
records = []
for line in (logdir / "scores.jsonl").read_text().strip().split("\n"):
    try:
        records.append(json.loads(line))
    except json.JSONDecodeError:
        continue

data = []
for r in records:
    if "step" in r and "episode/score" in r:
        data.append((r["step"], r["episode/score"]))

data.sort(key=lambda x: x[0])
steps, scores = zip(*data)

# Binning suave (promedio móvil)
window = max(1, len(scores) // 30)
def moving_avg(x, w):
    return np.convolve(x, np.ones(w)/w, mode='valid')

steps_smooth  = steps[window-1:]
scores_smooth = moving_avg(scores, window)

# Plot
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(steps, scores, alpha=0.2, color='#0022ff', linewidth=0.8)
ax.plot(steps_smooth, scores_smooth, color='#0022ff', linewidth=1.8, label='HER_OBS size12m')
ax.set_xlabel("Steps")
ax.set_ylabel("Episode score")
ax.set_title("POL_multigoal9 / size12m / 03")
ax.grid(color='#eeeeee')
ax.legend()
fig.tight_layout()
fig.savefig(outdir / "performance.png", dpi=150)
print("Saved:", outdir / "performance.png")
