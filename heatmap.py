#!/usr/bin/env python3
"""
Recolecta heatmap de visitas en MiniGrid usando embodied.Driver (tu embodied/core/driver.py).

- Usa SimpleGrid en embodied/envs/minigrid.py
- Usa Heatmap.increase (que espera tran['info']['agent_pos'])
- Ejecuta episodios con política aleatoria
- Guarda y muestra heatmap overlay si el renderer entrega frames RGB
"""
import time
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

# Import your SimpleGrid and Driver
from embodied.envs.minigrid import SimpleGrid
from embodied.core.driver import Driver

# --- Heatmap class you provided ---
class Heatmap:
  def __init__(self):
        self.heatmap = defaultdict(int)
  def increase(self, tran, env_index, **kwargs):
    if "agent_pos" in tran:
        x, y = map(int, tran["agent_pos"])
        # convert (x,y) -> (row,col)=(y,x) for indexing
        self.heatmap[(int(y), int(x))] += 1

# --- plotting helper (same as earlier) ---
def plot_heatmap_from_counts(counts_dict, sample_frame=None, smooth_sigma=1.0, cmap="hot", alpha=0.6, show=True, savepath=None):
    if len(counts_dict) == 0:
        print("No visits recorded.")
        return
    rows = [k[0] for k in counts_dict.keys()]
    cols = [k[1] for k in counts_dict.keys()]
    H = max(rows) + 1
    W = max(cols) + 1
    counts = np.zeros((H, W), dtype=float)
    for (r, c), v in counts_dict.items():
        counts[r, c] = v
    try:
        from scipy.ndimage import gaussian_filter
        if smooth_sigma > 0:
            counts = gaussian_filter(counts, sigma=smooth_sigma)
    except Exception:
        pass
    maxv = counts.max()
    counts_norm = counts / maxv if maxv > 0 else counts
    plt.figure(figsize=(6, 6))
    if sample_frame is None:
        plt.imshow(counts_norm, cmap=cmap)
        plt.colorbar()
    else:
        ph, pw = sample_frame.shape[0], sample_frame.shape[1]
        block_h = max(1, ph // H)
        block_w = max(1, pw // W)
        heat_pixels = np.kron(counts_norm, np.ones((block_h, block_w)))
        heat_pixels = heat_pixels[:ph, :pw]
        plt.imshow(sample_frame)
        plt.imshow(heat_pixels, cmap=cmap, alpha=alpha, extent=(0, pw, ph, 0))
        plt.colorbar()
    plt.axis("off")
    if savepath:
        plt.savefig(savepath, bbox_inches="tight", pad_inches=0)
    if show:
        plt.show()

# --- Small random agent compatible with Driver._step(policy, ...) ---
class RandAgent:
    def init_policy(self, length=1):
        # driver.reset expects a callable that can be used to init policy state.
        # We return None as initial carry.
        return None

    def policy(self, carry, obs, **kwargs):
        # obs is a dict with stacked arrays along axis 0; get length n
        try:
            first_k = next(iter(obs.keys()))
            n = obs[first_k].shape[0]
        except Exception:
            n = 1
        actions = np.random.randint(0, 5, size=(n,))  # MiniGrid has 5 actions
        acts = {"action": actions}
        outs = {}
        return carry, acts, outs

# ----------------- main -----------------
def main():
    GRID_TASK = "5x5"
    PIXEL_SIZE = (80, 80)
    TILE_SIZE = 16
    EPISODES = 60
    STEPS_PER_CALL = 10   # driver(policy, steps=STEPS_PER_CALL)
    PARALLEL = False

    print("Creating envs factory list (fns)...")
    # fns is a list of callables making envs; here we create a single env factory
    def make_env_fn(idx=0):
        return SimpleGrid(task=GRID_TASK, size=PIXEL_SIZE, rgb_img_obs='full', tile_size=TILE_SIZE)
    fns = [make_env_fn]  # single env. If you want multiple envs, put more factories here.

    print("Instancing Driver...")
    driver = Driver(fns, parallel=PARALLEL)

    agent = RandAgent()
    policy_callable = lambda *args, **k: agent.policy(*args, **k)

    # counters
    step_counter = {"n": 0}
    episode_counter = {"n": 0}

    # heatmap
    heat = Heatmap()

    # Try to capture a sample frame for overlay
    sample_frame = None

    # Register callbacks on driver
    # increment step
    def step_inc(tran, worker):
        step_counter["n"] += 1
    driver.on_step(step_inc)

    # increment episode count when is_last True
    def episode_inc(tran, worker):
        if tran.get("is_last", False):
            episode_counter["n"] += 1
    driver.on_step(episode_inc)

    # replay.add / logfn etc omitted — we only want heatmap
    # Register heatmap callback using your Heatmap.increase
    # Note: Heatmap.increase expects (tran, env_index, **kwargs)
    driver.on_step(lambda tran, worker: heat.increase(tran, worker))

    # Reset driver with initial policy state (as in train)
    print("Resetting driver with agent.init_policy ...")
    driver.reset(agent.init_policy)

    print(f"Running driver loop until {EPISODES} episodes recorded...")
    start_time = time.time()
    # Keep calling driver(policy, steps=STEPS_PER_CALL) until we collect EPISODES
    while episode_counter["n"] < EPISODES:
        # call driver with policy; driver will call our callbacks
        driver(policy_callable, steps=STEPS_PER_CALL)
        # safety guard: avoid infinite loop
        if step_counter["n"] > EPISODES * 1000:
            print("Too many steps, aborting.")
            break

    elapsed = time.time() - start_time
    print(f"Finished. Episodes: {episode_counter['n']}, Steps: {step_counter['n']}, Time: {elapsed:.2f}s")

    # Plot heatmap
    frame = None
    # try to get a render frame from one of the envs
    try:
        env0 = driver.envs[0]
        # try env0.render
        try:
            frame = env0.render()
        except TypeError:
            frame = env0.render("rgb_array")
    except Exception:
        frame = None

    print("Total visits recorded:", sum(heat.heatmap.values()))
    plot_heatmap_from_counts(heat.heatmap, sample_frame=frame, smooth_sigma=1.0, cmap="hot", alpha=0.6, show=True, savepath="heatmap_overlay.png")
    print("Saved heatmap_overlay.png")

if __name__ == "__main__":
    main()
