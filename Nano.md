# Nano.md — Comandos útiles

Referencia rápida para correr, monitorear y graficar experimentos de este repo.

---

## 0. Setup

```bash
# Entorno de DreamerV3 (JAX)
python3.11 -m venv .venv && source .venv/bin/activate
pip install -U -r requirements.txt        # incluye jax[cuda12]==0.4.33

# Entorno separado para PPO + RND (PyTorch / SB3)
python3.11 -m venv .venvPPO && source .venvPPO/bin/activate
pip install -U -r PPO_RND/requirements.txt
```

Los scripts `run_*.sh` asumen las rutas del servidor:
- `.venv`  → `/home/iamonardes/dreamerv3-exploration/.venv` (Dreamer, JAX)
- `.venvPPO` → `/home/iamonardes/dreamerv3-exploration/.venvPPO` (PPO+RND)

Todos envuelven el comando en `xvfb-run` (MiniGrid necesita display virtual) y
fijan `XLA_PYTHON_CLIENT_PREALLOCATE=false`, `OMP/MKL_NUM_THREADS=4`.

| Script | Para qué |
|---|---|
| `run_gpu.sh` | Dreamer en GPU (`JAX_PLATFORM_NAME=gpu`) |
| `run_cpu.sh` | Dreamer en CPU (`JAX_PLATFORM_NAME=cpu`) |
| `run_create_image.sh` | CPU forzado + `CUDA_VISIBLE_DEVICES=""` — para scripts de análisis/render |
| `run_gpu_ppo.sh` | PPO con SB3 (`ppo/train_ppo.py`) |
| `run_ppo_rnd.sh` | PPO + RND (usa `.venvPPO`) |

---

## 1. Task spooler (`ts`) — cola de trabajos en el servidor

```bash
ts                      # ver todos mis procesos en cola
ts -G 1 bash run_gpu.sh ...   # encolar pidiendo 1 GPU
ts -t {job_id}          # ver la cola de output actual (tail en vivo)
ts -c {job_id}          # ver TODO el output del job
ts -k {job_id}          # matar un job
ts -r {job_id}          # sacar un job de la cola
CUDA_VISIBLE_DEVICES=1 ts -G 1 bash run_gpu.sh ...   # forzar la segunda GPU
```

---

## 2. Entrenar DreamerV3

Forma general:

```bash
ts -G 1 bash run_gpu.sh python3 dreamerv3/main.py \
  --logdir ~/logdir/dreamer/{experimento}/{size}/{nn} \
  --configs {env_config} {size_config} \
  --task {suite}_{task} \
  --run.steps 1000000 \
  --run.save_every 42
```

### Prueba rápida en CPU
```bash
ts bash run_cpu.sh python3 dreamerv3/main.py \
  --logdir ~/logdir/dreamer/minigrid/size1m/01 \
  --configs minigrid size1m --run.steps 100 --jax.platform cpu
```

### Debug local (red mínima, todo en CPU)
```bash
python3 dreamerv3/main.py --logdir /tmp/dbg --configs minigrid debug --run.steps 200
```

### Runs en GPU
```bash
# MiniGrid Empty, goal al medio
ts -G 1 bash run_gpu.sh python3 dreamerv3/main.py \
  --logdir ~/logdir/dreamer/minigrid55/size12m/01 \
  --configs minigrid size12m --task minigrid_55x55 --run.steps 1000000

# Corridor
ts -G 1 bash run_gpu.sh python3 dreamerv3/main.py \
  --logdir ~/logdir/dreamer/corridor32/size12m/01 \
  --configs corridor size12m --task corridor_32 --run.steps 1000000

# TwoRooms
ts -G 1 bash run_gpu.sh python3 dreamerv3/main.py \
  --logdir ~/logdir/dreamer/tworooms18x29/size12m/01 \
  --configs tworooms size12m --task tworooms_18x29 --run.steps 1000000

# Cookie determinista
ts -G 1 bash run_gpu.sh python3 dreamerv3/main.py \
  --logdir ~/logdir/dreamer/cookiepedrodeterministic18x29/size12m/07 \
  --configs cookiepedrodeterministic size12m \
  --task cookiepedrodeterministic_18x29 --run.steps 1000000 --run.save_every 42

# Cookie sin respawn, observación completa (el que sí aprendió)
ts -G 1 bash run_gpu.sh python3 dreamerv3/main.py \
  --logdir ~/logdir/dreamer/cookiepedrofullfixed18x29/size12m/03 \
  --configs cookiepedrofullfixed size12m \
  --task cookiepedrofullfixed_18x29 --run.steps 1000000 --run.save_every 42
```

> ⚠️ **Siempre pasar `--task` explícito para los `cookiepedro*`.** En
> `configs.yaml` esos bloques traen `task: cookieenv_18x29`, y el suite
> (`cookieenv`) no existe en el dispatch de `dreamerv3/main.py:231`. El suite es
> lo que va antes del primer `_`, así que el task debe ser
> `cookiepedrofull_18x29`, `cookiepedrodeterministic_18x29`, etc.

### Configs disponibles (`dreamerv3/configs.yaml`)

- **Envs propios:** `minigrid`, `cookiepedro`, `cookiepedroonehot`,
  `cookiepedrodeterministic`, `cookiepedrotwocookies`, `cookiepedrofull`,
  `cookiepedrofullfixed`, `corridor`, `tworooms`
- **Envs upstream:** `crafter`, `atari`, `atari100k`, `dmlab`, `procgen`,
  `dmc_proprio`, `dmc_vision`, `bsuite`, `loconav`, `minecraft`
- **Tamaños:** `size1m`, `size12m`, `size25m`, `size50m`, `size100m`,
  `size200m`, `size400m`
- **Extra:** `debug` (red mínima + CPU), `multicpu`

Mapeo suite → clase en `dreamerv3/main.py:231-243`; los envs viven en
`embodied/envs/new_minigrid.py`.

### Flags que se usan seguido

```bash
--run.steps 1e6           # pasos de entorno
--run.save_every 42       # cada cuántos segundos checkpointea (42 = muy seguido, para heatmaps)
--run.train_ratio 256     # gradient steps por step de entorno
--run.envs 1              # nº de envs paralelos
--run.script train        # train | train_eval | eval_only | parallel | parallel_env | parallel_envs | parallel_replay
--run.from_checkpoint /ruta/ckpt
--jax.platform cpu        # forzar CPU
--batch_size 1            # para descartar OOM cuando hay errores CUDA
--seed 1
```

**Reanudar un run:** volver a lanzar el mismo comando con el mismo `--logdir`.
Si sale `Too many leaves for PyTreeDef`, el checkpoint no calza con la config
actual (típicamente un `--logdir` reusado por error).

### Solo evaluación
```bash
ts -G 1 bash run_gpu.sh python3 dreamerv3/main.py \
  --logdir ~/logdir/dreamer/cookiepedrofullfixed18x29/size12m/03 \
  --configs cookiepedrofullfixed size12m --task cookiepedrofullfixed_18x29 \
  --run.script eval_only
```

---

## 3. Visualizar resultados

### Scope viewer (visor nativo de DreamerV3)
```bash
pip install -U scope
python -m scope.viewer --basedir ~/logdir --port 8000
```

### TensorBoard (runs de PPO / PPO+RND)
```bash
tensorboard --logdir ./tb_logs
tensorboard --logdir ./ppo_logs
```

### Traerse logs del servidor
```bash
scp -r iamonardes@barto.ing.uc.cl:/home/iamonardes/logdir/dreamer/tworooms18x29 ./logdir/
```

### Curvas de score (`plot.py`, estilo paper)
```bash
python3 plot.py --indirs ~/logdir/dreamer --outdir plots_nano \
  --pattern '**/scores.jsonl' --tasks '.*' --methods '.*' --bins 30
```
Flags: `--xkeys`, `--ykeys` (default `episode/score`), `--binsize`, `--xlim`,
`--ylim`, `--cols`, `--agg`, `--todf salida.json.gz`, `--latest_seed_only`.
Los baselines publicados están en `scores/*.json.gz` y `baselines.yaml`.

### Heatmaps de exploración (GIF animado)
Estos scripts tienen las rutas **hardcodeadas arriba del archivo** — editar
`LOGDIR` y `OUT_ANIM` antes de correr:

```bash
python3 plot_heatmap.py        # escala lineal      (plot_heatmap.py:13-17)
python3 plot_log_heatmap.py    # escala logarítmica (plot_log_heatmap.py:14-20)
python3 new_plot_heatmap.py    # versión más nueva  (new_plot_heatmap.py:13-17)
```
Salidas en `plots_nano/*.gif`. Constantes: `LOGDIR`, `OUT_ANIM`, `CMAP`,
`SMOOTH` (sigma gaussiano), `FPS`.

### Heatmap de un agente random (overlay sobre el frame)
```bash
python3 heatmap.py     # edita GRID_TASK / EPISODES en heatmap.py:90 → heatmap_overlay.png
```

### Exportar videos openloop desde `metrics.jsonl`
```bash
python3 plot_videos.py   # editar METRICS_PATH arriba → videos_out/*.mp4
```
Los videos ya generados están en `reports/openloop_*.mp4`.

### Inspeccionar un env a mano (pygame, teclado)
```bash
python3 probando.py      # editar GRID_SIZE / clase de env en probando.py:99
python3 using_minigrid_env.py
```

---

## 4. Análisis del world model ("ver soñar" al agente)

Reconstruye observaciones vs. imaginación desde un checkpoint entrenado.
Editar `LOGDIR` arriba del archivo (`seeing_with_dreamer.py:33`,
`seeing_with_dreamer_2.py:31`), que apunta al run con su `config.yaml` y `ckpt`.

```bash
ts bash run_create_image.sh python -m dreamerv3.seeing_with_dreamer
ts bash run_create_image.sh python -m dreamerv3.seeing_with_dreamer_2
```

Lectura del video resultante (ver `bitacora investigacion/29_12.md`): 6 columnas
= batches; fila 1 = observación real, fila 2 = decode del estado estocástico,
fila 3 = diferencia. Borde verde = `decode(z_t)` (posterior), borde rojo =
`decode(ẑ_t)` (imaginado).

---

## 5. PPO (Stable-Baselines3)

```bash
# Barrido de hiperparámetros definido en ppo/train_ppo.py (lista `experimentos`)
ts -G 1 bash run_gpu_ppo.sh
# equivalente local:
python3 ppo/train_ppo.py
```
Logs → `ppo_logs/{exp_name}/`. Envs en `ppo/middlegrid_env.py`
(`MiddleGridEnv`, `CornerEnv`), heatmaps vía `ppo/heatmap.py` (callback).

```bash
# Curvas de PPO por episodio
python3 ppo/plot_ppo_by_episode.py --base-dir ./ppo_logs/cornerenv22_run3 --rolling-window 50
```

### ICM (curiosity) sobre PPO — ⚠️ NO FUNCIONAL, no hay comando que sirva

Los archivos `ppo/ICM_*.py` quedaron a medio portar (último commit: *"feature:
trying to recreate ICM"*). No intentes correrlos sin arreglarlos antes. Lo que
está roto, concretamente:

- **No hay entrypoint.** `ppo/ICM_train.py` no tiene `if __name__ == "__main__"`
  ni `main()`; solo define `train_loop()` y `evaluate()`. Correrlo no entrena nada.
- **Dependencias que no están instaladas ni en ningún `requirements.txt`:**
  `a2c_ppo_acktr` (de `ikostrikov/pytorch-a2c-ppo-acktr-gail`) en `ICM_train.py`
  e `ICM_PPO.py`, y `baselines` (OpenAI baselines, ≠ `stable_baselines3`) en
  `ICM_envs.py`.
- **`np.int` en `ICM_utils.py:8`** — removido en NumPy ≥1.24, y `PPO_RND/requirements.txt`
  fija `numpy==1.26.4`. Revienta al instanciar `CuriosityStatistics`.
- **Imports planos** (`from ICM_utils import ...`, `from PPO_CONFIG import *`):
  solo resuelven con el cwd *dentro* de `ppo/`, no desde la raíz del repo.
- **`ICM_PPO.py` corre todo a nivel de módulo** (crea envs y entrena con solo
  importarlo), con `cuda:1` hardcodeado y `env_name = "corner_55"`, un id que no
  está registrado en gym clásico.
- Usa `gym` clásico, mientras el resto del repo ya está en `gymnasium`.

Si se retoma: lo mínimo es envolver `ICM_PPO.py` en un `main()` con argparse,
reemplazar `a2c_ppo_acktr`/`baselines` por SB3 (como ya hace `PPO_RND/`), y
cambiar `np.int` → `int`.

---

## 6. PPO + RND

```bash
# Comparativa completa: RND on/off × seeds 1..4, y grafica al final
ts -G 1 bash run_ppo_rnd.sh python -m PPO_RND.main \
  --env CornerEnv-v0 --env-size 55 --steps 1000000 --log-dir ./tb_logs/

# Un solo entrenamiento
ts -G 1 bash run_ppo_rnd.sh python -m PPO_RND.main \
  --single-train --seed 42 --env-size 102 --intrinsic-coef 0.005 --envs 4 --steps 1000000

# Sin RND (baseline)
ts -G 1 bash run_ppo_rnd.sh python -m PPO_RND.main --single-train --no-rnd --env-size 102
```

Flags de `PPO_RND/main.py:198-211`: `--env`, `--env-size`, `--single-train`,
`--seed`, `--log-dir`, `--no-rnd`, `--intrinsic-coef`, `--envs`, `--steps`,
`--pretrain`, `--gamma`. Tamaños de red en `PPO_RND/configs.py`
(`tiny` / `small` / `large`). Logs → `tb_logs/RND_{Enabled,Disabled}_{env}_{size}/`.

Este es el camino que sí quedó andando (a diferencia de ICM), pero con dos
detalles a tener en cuenta:

- **Usar `python -m PPO_RND.main`, no `python PPO_RND/main.py`.** `main.py` hace
  `from PPO_RND.PPO_RND_wrapper import ...` (import absoluto), y al correr el
  archivo directo `sys.path[0]` es `PPO_RND/`, no la raíz del repo → `ModuleNotFoundError`.
- **`run_ppo_rnd.sh:15` apunta a un archivo que no existe:** `PPO_RND/PPO_wrapper.py`
  (el real es `PPO_RND_wrapper.py`). Por eso `bash run_ppo_rnd.sh` *sin argumentos*
  falla. Con argumentos funciona igual, porque el script ejecuta `"$@"` y nunca
  llega a usar `$SCRIPT`.

---

## 7. Docker

```bash
docker build -f Dockerfile -t img .
docker run -it --rm -v ~/logdir/docker:/logdir img \
  python dreamerv3/main.py --logdir /logdir/{timestamp} --configs minigrid size12m
```

---

## 8. Traspaso de resultados y claves

```bash
# Bajar un logdir completo del servidor
scp -r iamonardes@barto.ing.uc.cl:/home/iamonardes/logdir/dreamer/{run} ./logdir/{run}/

# Bajar solo los scores (liviano)
scp iamonardes@barto.ing.uc.cl:'/home/iamonardes/logdir/dreamer/**/scores.jsonl' ./
```

### Claves de deploy de git (en el servidor)
```bash
/home/iamonardes/.ssh/deploy_minigrid       # privada
/home/iamonardes/.ssh/deploy_minigrid_pub   # pública
```

---

## 9. Notas rápidas

- Errores de CUDA: casi siempre la causa real está más arriba en el log (OOM o
  mismatch JAX/CUDA). Probar `--batch_size 1` para descartar OOM.
- Las métricas escalares quedan en `{logdir}/scores.jsonl` y `{logdir}/metrics.jsonl`.
- `--run.save_every 42` genera checkpoints muy seguido: útil para armar los GIF
  de heatmap, pesado en disco para runs largos.
