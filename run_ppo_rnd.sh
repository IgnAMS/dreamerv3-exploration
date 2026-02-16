#!/bin/bash
set -euo pipefail

########################################
# Configuración
########################################

# Ruta absoluta al proyecto
PROJECT_ROOT="/home/iamonardes/dreamerv3-exploration"

# Virtualenv
VENV="${PROJECT_ROOT}/.venvPPO"

# Script a ejecutar (ACTUALIZADO AL NUEVO NOMBRE)
SCRIPT="${PROJECT_ROOT}/PPO_RND/PPO_wrapper.py"

########################################
# Activación del entorno
########################################

if [ -d "$VENV" ]; then
  echo "[INFO] Activando virtualenv: $VENV"
  # shellcheck disable=SC1090
  source "${VENV}/bin/activate"
else
  echo "[ERROR] Virtualenv no encontrado en $VENV. Abortando."
  exit 1
fi

########################################
# Info de ejecución
########################################

echo "====================================="
echo "Running PPO + RND Training"
echo "====================================="
echo "User       : $(whoami)"
echo "Python     : $(which python)"
python --version
echo "Script     : ${SCRIPT}"
echo

########################################
# GPU info (Ajustado para detectar Mac/Nvidia)
########################################

if command -v nvidia-smi &> /dev/null; then
  echo "[GPU] Detectada NVIDIA:"
  nvidia-smi --query-gpu=gpu_name,memory.total --format=csv,noheader
elif [[ "$OSTYPE" == "darwin"* ]]; then
  echo "[GPU] Detectado Apple Silicon (MPS)"
else
  echo "[INFO] No se detectó aceleración de hardware externa"
fi
echo

########################################
# Variables de entorno
########################################

# Evita que Python genere archivos .pyc que ensucian el repo
export PYTHONDONTWRITEBYTECODE=1
# Optimización de hilos
export OMP_NUM_THREADS=4

########################################
# Ejecución
########################################

echo "Iniciando proceso..."
echo "-------------------------------------"

# Ejecutamos el script
if [ $# -gt 0 ]; then
    echo "Ejecutando comando externo: $@"
    "$@"
else
    echo "Ejecutando script por defecto: python ${SCRIPT}"
    python "${SCRIPT}"
fi

echo
echo "====================================="
echo "Entrenamiento finalizado correctamente"
echo "====================================="
