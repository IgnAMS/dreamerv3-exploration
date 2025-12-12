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
        # self.log_dir = self.logger.dir
        # CORRECCIÓN CLAVE: Aseguramos que self.n_envs esté definido 
        # obteniéndolo del modelo, que ya está disponible aquí.
        if not self.log_dir:
            self.log_dir = self.logger.dir
            
        if self.model is not None and hasattr(self.model, 'n_envs'):
            self.n_envs = self.model.n_envs
        else:
            # En caso de un setup inusual, lanzamos un error más informativo
            raise AttributeError("El modelo no está disponible o no tiene el atributo 'n_envs'.")


    def _on_step(self) -> bool:
        """Called by the RL model at each step of the training."""
        
        # 1. Recolectar datos de todos los entornos vectorizados
        # self.locals['infos'] es una lista de diccionarios, uno por entorno.
        # Gracias a _on_training_start, self.n_envs ahora está disponible.
        for i in range(self.n_envs):
            # Verificar si la info es válida y contiene la posición del agente
            info = self.locals['infos'][i]
            if info and 'agent_pos' in info:
                # agent_pos es un array numpy, lo convertimos a tupla para usarlo como clave
                pos = tuple(info['agent_pos'])
                self.position_counts[pos] += 1
                
         # 2. Llamar a generate_heatmap solo una vez cuando se alcanza la frecuencia de guardado.
        if self.n_calls % self.save_freq == 0:
            self.generate_heatmap()
            

        return True

    def generate_heatmap(self):
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
            
        # Crear el gráfico
        plt.figure(figsize=(8, 8))
        
        # Plotear la matriz 
        plt.imshow(heatmap_matrix, cmap='viridis', origin='lower')
        
        plt.title(f'Mapa de Calor de la Posición del Agente (Step: {self.num_timesteps})')
        plt.colorbar(label='Frecuencia de Visita')
        plt.xticks(np.arange(self.size))
        plt.yticks(np.arange(self.size))
        plt.grid(True, color='red', alpha=0.3)
        
        # Marcar las metas 
        corner_goal = (self.size - 2, self.size - 2)
        middle_goal = (int((self.size - 1) / 2), int((self.size - 1) / 2))
        
        # El plot usa (x, y), donde x es horizontal (columna) y y es vertical (fila)
        plt.plot(corner_goal[0], corner_goal[1], 'rs', markersize=10, label='Meta Esquina (1.0)')
        plt.plot(middle_goal[0], middle_goal[1], 'go', markersize=10, label='Meta Media (0.2)')
        plt.legend()
        
        # Guardar el archivo en el directorio de logs
        filename = os.path.join(self.log_dir, f'heatmap_step_{self.num_timesteps}.png')
        plt.savefig(filename)
        plt.close()

        if self.verbose > 0:
            print(f"Heatmap guardado en {filename}")
        
        return