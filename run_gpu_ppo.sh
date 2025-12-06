#!/bin/bash
set -euo pipefail

# --- Ajusta la ruta de tu virtualenv aquí si hace falta ---
VENV="/home/iamonardes/dreamerv3-exploration/.venv"
if [ -d "$VENV" ]; then
  # activar venv
  # shellcheck disable=SC1090
  . "${VENV}/bin/activate"
fi

echo "Running PPO Training Script"
echo "============================="
echo "User: $(whoami)"
echo "Python: $(which python) ($(python --version 2>&1))"

# Muestra el estado de la GPU (útil para verificar que PyTorch la detectará)
echo "GPUs (nvidia-smi):"
nvidia-smi --query-gpu=gpu_name,memory.total,driver_version --format=csv || true
echo

# --- Variables de Entorno para Stable Baselines3 (PyTorch) ---
# PyTorch detecta CUDA automáticamente, no necesitamos JAX/XLA específicos.
# Solo limitamos threads de CPU (buena práctica para evitar sobrecarga)
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
# Opcional: Establecer el nivel de log para PyTorch/Python
export TF_CPP_MIN_LOG_LEVEL=3

# --- Ejecución del Script de PPO ---
# Ejecutamos directamente el script de entrenamiento PPO.
# Asumo que el script 'train_ppo.py' está en la ruta 'pop/train_ppo.py'
echo "Ejecutando: python pop/train_ppo.py"
python pop/train_ppo.py