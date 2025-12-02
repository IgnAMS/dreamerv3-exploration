import os
import pickle
import glob
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from matplotlib.widgets import Slider

# ------------ CONFIG ------------
LOGDIR = Path.home() / "logdir" / "dreamer" / "minigrid30" / "size12m" / "01"
OUT_ANIM = Path("log_exploration_gpu_30x30_200k.gif") 
CMAP = "inferno"
SMOOTH = 0.0   # gaussian sigma (0 = no smoothing). Requiere scipy.ndimage if >0
FPS = 4
# ---------------------------------

def find_heatmap_files(logdir):
    files = sorted(glob.glob(str(logdir / "heatmap_*.pkl")))
    # sort by the integer in filename
    def keyfn(p):
        m = re.search(r"heatmap_(\d+)\.pkl$", p)
        return int(m.group(1)) if m else 0
    files = sorted(files, key=keyfn)
    return files

def load_heatmap_file(p):
    with open(p, "rb") as f:
        d = pickle.load(f)
    # expect dict {(r,c): count}
    return d

def accumulate_heatmaps(files, assume_xy=False, grid_size=None):
    """
    Carga dicts y devuelve acumulados normalizados.

    Args:
      files: lista de paths a heatmap_*.pkl (ordenados).
      assume_xy: si True, interpreta las llaves del dict como (x,y) -> las convierte a (row=y, col=x).
      grid_size: si no None, fuerza el tamaño (grid_size x grid_size).
    Returns:
      accum_list, H, W
    """
    dicts = []
    re_step = re.compile(r"heatmap_(\d+)\.pkl$")
    for pth in files:
        m = re_step.search(pth)
        step = int(m.group(1)) if m else None
        d = load_heatmap_file(pth)
        dicts.append(d)

    # si no hay dicts
    if not dicts:
        return [], 0, 0

    # recopilar todos los índices para inferir rango
    all_coords = []
    for d in dicts:
        for (r, c) in d.keys():
            if assume_xy:
                x, y = r, c
                row, col = int(y), int(x)
            else:
                row, col = int(r), int(c)
            all_coords.append((row, col))

    rows = [rc[0] for rc in all_coords]
    cols = [rc[1] for rc in all_coords]

    min_r = min(rows)
    min_c = min(cols)
    max_r = max(rows)
    max_c = max(cols)

    # Si el usuario fuerza grid_size, úsalo. Sino inferir del rango (normalizado)
    if grid_size is not None:
        H = grid_size
        W = grid_size
        # en este caso asumimos que el offset es 0..grid_size-1; pero si min_r>0,
        # hacemos un shift compensatorio al mapear.
    else:
        H = max_r - min_r + 1
        W = max_c - min_c + 1

    # Acumular con shift (restar min_r/min_c)
    accum_list = []
    accum = np.zeros((H, W), dtype=float)
    for d in dicts:
        for (r, c), v in d.items():
            if assume_xy:
                x, y = r, c
                row, col = int(y), int(x)
            else:
                row, col = int(r), int(c)
            # shift to zero-based
            rs = row - min_r
            cs = col - min_c
            # if a forced grid_size is given, we can optionally map coordinates into it
            if 0 <= rs < H and 0 <= cs < W:
                accum[rs, cs] = v
            else:
                # ignore out-of-range coords (print for debugging)
                # print("coord out of inferred grid:", (row, col), "-> shifted", (rs, cs), " HxW", (H, W))
                pass
        accum_list.append(accum.copy())
    return accum_list, H, W


def maybe_smooth(arr, sigma):
    if sigma and sigma > 0:
        from scipy.ndimage import gaussian_filter
        return gaussian_filter(arr, sigma=sigma)    
    return arr

def make_animation(accum_list, sample_frame=None, outpath=OUT_ANIM, cmap=CMAP, fps=FPS, smooth_sigma=SMOOTH):
    fig, ax = plt.subplots(figsize=(5,5))
    plt.tight_layout()
    fig.subplots_adjust(right=0.80)

    # choose vmax for color scaling (global max across frames)
    # vmax = max(a.max() for a in accum_list) or 1.0
    first = maybe_smooth(accum_list[0], smooth_sigma)
    first_log = np.log1p(first)
    
    im = None

    if sample_frame is None:
        im = ax.imshow(first_log, cmap=cmap, vmin=0, vmax=first_log.max() or 1.0, origin="lower")
    else:
        ax.imshow(sample_frame)
        im = ax.imshow(first_log, cmap=cmap, alpha=0.6, vmin=0, vmax=first_log.max() or 1.0, origin="lower")
    
    cb = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("log1p(visits)")
    title = ax.set_title(f"steps={accum_list[0].sum():.0f}")

    def update(i):
        data = maybe_smooth(accum_list[i], smooth_sigma)
        log_data = np.log1p(data)
        im.set_data(log_data)
        
        frame_max = float(log_data.max() or 1.0)
        im.set_clim(0, frame_max)
        
        cb.update_normal(im)
        
        title.set_text(f"steps={accum_list[i].sum():.0f}")
        return (im,)

    anim = animation.FuncAnimation(fig, update, frames=len(accum_list), blit=True, interval=1000/fps)

    # Save
    outpath = Path(outpath)
    if outpath.suffix == ".gif":
        anim.save(str(outpath), writer='pillow', fps=fps)
        print(f"Saved GIF to {outpath}")
    else:
        # try ffmpeg for mp4
        try:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=fps, metadata=dict(artist='me'), bitrate=1800)
            anim.save(str(outpath), writer=writer)
            print(f"Saved MP4 to {outpath}")
        except Exception as e:
            print("Could not save MP4 (ffmpeg missing?). Trying GIF fallback.")
            try:
                anim.save(str(outpath.with_suffix(".gif")), writer='pillow', fps=fps)
                print("Saved GIF fallback")
            except Exception as e2:
                print("Failed to save animation:", e, e2)
    plt.close(fig)

def interactive_view(accum_list, sample_frame=None, cmap=CMAP, smooth_sigma=SMOOTH):
    fig, ax = plt.subplots(figsize=(5,5))
    fig.subplots_adjust(right=0.80, bottom=0.15)
    first = maybe_smooth(accum_list[0], smooth_sigma)
    first_log = np.log1p(first)
    
    if sample_frame is None:
        img = ax.imshow(first_log, cmap=cmap, origin="lower", vmax=first_log.max() or 1.0)
    else:
        ax.imshow(sample_frame)
        img = ax.imshow(first_log, cmap=cmap, alpha=0.6, origin="lower", vmax=first_log.max() or 1.0)
    
    cb = plt.colorbar(img, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("log1p(visits)")
    
    title = ax.set_title(f"step={accum_list[0].sum():.0f})")
    axcolor = 'lightgoldenrodyellow'
    axslider = plt.axes([0.15, 0.05, 0.7, 0.03], facecolor=axcolor)
    slider = Slider(axslider, 'frame', 0, len(accum_list)-1, valinit=0, valstep=1)

    def update(val):
        i = int(val)
        data = maybe_smooth(accum_list[i], smooth_sigma)
        log_data = np.log1p(data) 
        img.set_data(log_data)
        
        vmax = float(log_data.max() or 1.0)
        img.set_clim(0, vmax)
        cb.update_normal(img)

        title.set_text(f"step={accum_list[i].sum():.0f})")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()

def main():
    files = find_heatmap_files(LOGDIR)
    if not files:
        print("No heatmap files found in", LOGDIR)
        return
    print("Found files:", files)
    accum_list, H, W = accumulate_heatmaps(files)
    print(f"Grid size inferred: {H} x {W}, frames: {len(accum_list)}")

    # generate animation file
    print("Generating animation...")
    make_animation(accum_list, sample_frame=None, outpath=OUT_ANIM, cmap=CMAP, fps=FPS, smooth_sigma=SMOOTH)

    # show interactive viewer
    print("Opening interactive viewer (close the window to finish).")
    interactive_view(accum_list, sample_frame=None, cmap=CMAP, smooth_sigma=SMOOTH)

if __name__ == "__main__":
    main()
