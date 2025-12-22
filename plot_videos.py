import json
import os
import numpy as np
import imageio.v2 as imageio

METRICS_PATH = "/home/iamonardes/logdir/dreamer/cookiepedrodeterministic18x29/size12m/06/metrics.jsonl"
OUTDIR = "videos_out"

os.makedirs(OUTDIR, exist_ok=True)

def is_video(x):
    if not isinstance(x, list):
        return False
    arr = np.array(x)
    return (
        arr.dtype == np.uint8
        and arr.ndim == 4        # [T, H, W, C]
        and arr.shape[-1] in (1, 3, 4)
    )

with open(METRICS_PATH, "r") as f:
    for idx, line in enumerate(f):
        entry = json.loads(line)
        step = entry.get("step", idx)

        for key, value in entry.items():
            if not key.startswith("openloop/"):
                continue

            try:
                arr = np.array(value, dtype=np.uint8)
            except Exception:
                continue

            if arr.ndim != 4:
                continue

            # Nombre limpio
            name = key.replace("/", "_")
            outfile = os.path.join(
                OUTDIR, f"{name}_step{step}.mp4"
            )

            print(f"Saving {outfile}  shape={arr.shape}")

            writer = imageio.get_writer(outfile, fps=10)
            for frame in arr:
                writer.append_data(frame)
            writer.close()
