import numpy as np
import gymnasium as gym
from gymnasium import spaces
from minigrid.envs.empty import EmptyEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal
from minigrid.wrappers import RGBImgObsWrapper 
from gymnasium.wrappers import FilterObservation, FlattenObservation

# Definición del entorno base
class RawMiddleGoal(EmptyEnv):
    """
    MiniGrid environment with two goals, basado en EmptyEnv.
    """
    def __init__(self, size=16, agent_start_pos=(1, 1), agent_start_dir=0, **kwargs):
        super().__init__(
            size=size,
            agent_start_pos=agent_start_pos,
            agent_start_dir=agent_start_dir,
            max_steps=4 * size * size,
            # Aseguramos render_mode="rgb_array" para la observación de imagen
            **kwargs, 
        )
        self.corner_goal_pos = None
        self.middle_goal_pos = None
        self.mission = 0

    def _gen_grid(self, width, height):
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        self.corner_goal_pos = (width - 2, height - 2)
        mid_x = int((width - 1) / 2)
        mid_y = int((height - 1) / 2)
        self.middle_goal_pos = (mid_x, mid_y)

        self.put_obj(Goal(), *self.corner_goal_pos)
        self.put_obj(Goal(), *self.middle_goal_pos)

        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        # self.mission = "reach the corner goal for a big reward or the middle for a small one"

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        if terminated:
            agent_pos_tuple = tuple(self.agent_pos)
            if agent_pos_tuple == self.corner_goal_pos:
                reward = 10.0
            elif agent_pos_tuple == self.middle_goal_pos:
                reward = 0.2
            else:
                 reward = 0.0

        info['agent_pos'] = self.agent_pos
        
        return obs, reward, terminated, truncated, info

class ExtractSingleKeyObs(gym.Wrapper):
    """
    Convierte un espacio de observación Dict de una sola clave (e.g., {'image': Box})
    en un simple Box (la imagen 3D). (Solución al error ValueError/AssertionError)
    """
    def __init__(self, env, key='image'):
        super().__init__(env)
        # Verificamos que el entorno envuelto sea un Dict con la clave especificada
        if key in env.observation_space:
            raise ValueError(f"El entorno envuelto debe tener un espacio de observación Dict con la clave '{key}'.")
            
        # Redefinimos el observation_space del wrapper como el Box de la imagen
        self.observation_space = env.observation_space[key]

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        # Devolvemos solo el valor de la clave 'image'
        return obs['image'], reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Devolvemos solo el valor de la clave 'image'
        return obs['image'], info
    
# Función de fábrica para crear el entorno con wrappers
def MiddleGridEnv(size=16):
    """
    Función de fábrica para el entorno personalizado RawMiddleGoal,
    con wrappers para asegurar un espacio de observación Box simple (imagen RGB)
    para Stable Baselines3 CnnPolicy.
    """
    
    # 1. Crear el entorno base con el modo de renderizado correcto
    env = RawMiddleGoal(size=size, render_mode="rgb_array")
    env = RGBImgObsWrapper(env) 
    env = FilterObservation(env, filter_keys=['image'])
    env = ExtractSingleKeyObs(env)
    return env


def CornerEnv(size=16):
    env = EmptyEnv(size=size, render_mode="rgb_array")
    env = RGBImgObsWrapper(env) 
    env = FilterObservation(env, filter_keys=['image'])
    env = ExtractSingleKeyObs(env)
    return env
