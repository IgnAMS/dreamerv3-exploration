import os
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3.common.callbacks import BaseCallback
from collections import defaultdict
import pickle
from pathlib import Path

class HeatmapCallback(BaseCallback):
    """
    A custom callback to log agent position and generate a heatmap plot periodically.
    """
    def __init__(self, size: int, save_freq: int, verbose: int = 0, log_dir: str = "", corner=False):
        super().__init__(verbose)
        self.size = size
        self.save_freq = save_freq
        self.position_counts = defaultdict(int)
        # self.log_dir = Path(log_dir)
        if log_dir != "":
            self.log_dir = Path(log_dir)
        else:
            self.log_dir = None
        # Inicializamos n_envs para que exista, aunque se establecerá en _on_training_start
        self.n_envs = 0 
        self.corner = corner

    def _on_training_start(self) -> None:
        """Called once before the first call to `_on_step()`."""
        if not self.log_dir:
            self.log_dir = Path(self.logger.dir)
            
        if self.model is not None and hasattr(self.model, 'n_envs'):
            self.n_envs = self.model.n_envs
        else:
            # En caso de un setup inusual, lanzamos un error más informativo
            raise AttributeError("El modelo no está disponible o no tiene el atributo 'n_envs'.")

    def _on_step(self) -> bool:
        """Called by the RL model at each step of the training."""
        # 1. Recolectar datos de todos los entornos vectorizados
        for i in range(self.n_envs):
            info = self.locals['infos'][i]
            if info and 'agent_pos' in info:
                pos = tuple(info['agent_pos'])
                self.position_counts[pos] += 1
        # 2. Llamar a generate_heatmap solo una vez cuando se alcanza la frecuencia de guardado.
        if self.n_calls % self.save_freq == 0:
            self.save_heatmap()
            
        return True

    def save_heatmap(self):
        """Generates and saves the heatmap plot."""
        if not self.position_counts:
            return
            
        # Inicializar la matriz del mapa de calor
        heatmap_matrix = np.zeros((self.size, self.size))
        
        # Llenar la matriz con los recuentos
        for (x, y), count in self.position_counts.items():
            # MiniGrid usa (x, y) donde x es la columna (ancho) e y es la fila (altura)
            if 0 <= x < self.size and 0 <= y < self.size:
                heatmap_matrix[y, x] = count
        
        step = self.num_timesteps
        pkl_filename = self.log_dir / f"heatmap_{self.size}_{int(step)}.pkl"
        with open(pkl_filename, "wb") as f:
            pickle.dump(dict(self.position_counts), f)
        