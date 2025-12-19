import os
import yaml
import numpy as np
from PIL import Image
import orbax.checkpoint as ocp
import jax
import jax.numpy as jnp
from embodied.envs.new_minigrid import CookiePedro

print("UNO\n\n")

CKPT_DIR = "/home/iamonardes/logdir/dreamer/cookiepedrodeterministic18x29/size12m/05/ckpt/20251219T110803F946099"
CONFIG_YAML = "/home/iamonardes/logdir/dreamer/cookiepedrodeterministic18x29/size12m/05/config.yaml"
OUT_PNG = "reconstruction.png"
print("DOS\n\n")

with open(CONFIG_YAML, "r") as f:
    config = yaml.safe_load(f)
print("TRES\n\n")

handler = ocp.PyTreeCheckpointHandler()
mgr = ocp.CheckpointManager(
    CKPT_DIR,
    handler,
    options=ocp.CheckpointManagerOptions(max_to_keep=5)
)
print("CUATRO\n\n")

step = mgr.latest_step()
if step is None:
    raise RuntimeError("No checkpoint steps found in " + CKPT_DIR)
print("CINCO\n\n")

restored = mgr.restore(step)
print("SEIS\n\n")

print("Checkpoint restored; top-level keys:", restored.keys())
params = restored.get("model_params") or restored.get("params") or restored.get("model")
print("example param keys:", list(params.keys())[:10])


# python3 -m dreamerv3.seeing_with_dreamer