#!/bin/bash
set -euo pipefail

########################################
# Configuración
########################################

# Ruta absoluta al proyecto (ajústala si es necesario)
PROJECT_ROOT="/home/iamonardes/dreamerv3-exploration"

# Virtualenv
VENV="${PROJECT_ROOT}/.venvPPO"

# Script a ejecutar
SCRIPT="${PROJECT_ROOT}/PPO_RND/ppo_rnd.py"

########################################
# Activación del entorno
########################################

if [ -d "$VENV" ]; then
  echo "[INFO] Activando virtualenv: $VENV"
  # shellcheck disable=SC1090
  . "${VENV}/bin/activate"
else
  echo "[WARN] Virtualenv no encontrado en $VENV"
fi

########################################
# Info de ejecución
########################################

echo "====================================="
echo "Running PPO + RND Training"
echo "====================================="
echo "User      : $(whoami)"
echo "Host      : $(hostname)"
echo "Python    : $(which python)"
python --version
echo "Working dir: $(pwd)"
pip freeze | grep -E "nvidia|jax|jaxlib" || true
echo

########################################
# GPU info (opcional)
########################################

if command -v nvidia-smi &> /dev/null; then
  echo "GPUs (nvidia-smi):"
  nvidia-smi --query-gpu=gpu_name,memory.total,driver_version --format=csv
else
  echo "[INFO] nvidia-smi no disponible"
fi
echo

########################################
# Variables de entorno (PyTorch)
########################################

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TF_CPP_MIN_LOG_LEVEL=3
export CUDA_DEVICE_ORDER=PCI_BUS_ID

########################################
# Ejecución
########################################

echo "Ejecutando: python ${SCRIPT}"
echo "-------------------------------------"

python "${SCRIPT}"

echo
echo "====================================="
echo "Entrenamiento finalizado"
echo "====================================="
# ts -G 1 bash run_ppo_rnd.sh