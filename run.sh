#!/bin/bash
set -euo pipefail

# --- Ajusta la ruta de tu virtualenv aquí si hace falta ---
# Por ejemplo: /home/iamonardes/.venvs/dreamer o relative path ./virtualenv
VENV="/home/iamonardes/dreamerv3-exploration/.venv"
if [ -d "$VENV" ]; then
  # activar venv
  # shellcheck disable=SC1090
  . "${VENV}/bin/activate"
fi

echo "Running on user: $(whoami)"
echo "Python: $(which python) ($(python --version 2>&1))"
echo "PIP freeze (subset):"
pip freeze | grep -E "nvidia|jax|jaxlib" || true
echo

echo "GPUs (nvidia-smi):"
nvidia-smi --query-gpu=gpu_name,memory.total,driver_version --format=csv || true
echo

# Recomendaciones para JAX/XLA (evitan prealloc y reducen ruido)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export TF_CPP_MIN_LOG_LEVEL=3

# Forzar JAX a GPU y evitar prealloc
export JAX_PLATFORM_NAME=cpu
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80
export TF_CPP_MIN_LOG_LEVEL=3
# limitar threads CPU
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4


# Finalmente ejecuta el comando pasado como argumentos envueltos en xvfb-run
# (Dreamer usa gym/envs que a veces requieren una pantalla virtual — si no la necesitas,
#  puedes reemplazar la línea siguiente por: exec "$@")
xvfb-run -a -s '-screen 0 1024x768x24 -ac +extension GLX +render -noreset' "$@"
