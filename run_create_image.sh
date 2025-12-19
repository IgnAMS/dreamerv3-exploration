#!/bin/bash
set -euo pipefail

########################################
# Virtualenv
########################################
VENV="/home/iamonardes/dreamerv3-exploration/.venv"
if [ -d "$VENV" ]; then
  # shellcheck disable=SC1090
  source "${VENV}/bin/activate"
fi

########################################
# Info básica
########################################
echo "Running on user: $(whoami)"
echo "Python: $(which python) ($(python --version 2>&1))"
echo

echo "Relevant pip packages:"
pip freeze | grep -E "jax|jaxlib|cuda|nvidia" || true
echo

########################################
# GPU info (solo informativo)
########################################
echo "nvidia-smi:"
nvidia-smi || true
echo

########################################
# JAX / XLA SAFE CPU SETUP
########################################

# 🚨 IMPORTANTE: esto evita que JAX intente tocar CUDA
export JAX_PLATFORM_NAME=cpu
export CUDA_VISIBLE_DEVICES=""

# Evita prealloc agresivo (aunque sea CPU)
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.80

# Reduce ruido
export TF_CPP_MIN_LOG_LEVEL=3

# Limitar threads CPU (muy recomendado)
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

########################################
# Debug extra (opcional)
########################################
echo "JAX platform forced to: $JAX_PLATFORM_NAME"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo

########################################
# Ejecutar comando
########################################

# Si NO necesitas pantalla virtual, usa esta línea:
# exec "$@"

# Si alguna dependencia de gym intenta usar display:
exec xvfb-run -a -s "-screen 0 1024x768x24 -ac +extension GLX +render -noreset" "$@"
# chmod +x run_create_image.sh
# ts bash run_create_image.sh python -m dreamerv3.seeing_with_dreamer