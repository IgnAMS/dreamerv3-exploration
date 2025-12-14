# embodied/envs/minigrid_empty.py
import functools
from typing import Optional, Tuple

import elements
import embodied
import numpy as np
import gymnasium as gym

from minigrid.envs.empty import EmptyEnv

from minigrid.wrappers import FullyObsWrapper, RGBImgPartialObsWrapper, RGBImgObsWrapper
from . import from_gym
from PIL import Image

from minigrid.core.grid import Grid
from minigrid.core.world_object import Goal


class SimpleEnv(embodied.Env):
    """
    Wrapper similar a SimpleGrid pero que crea dinámicamente un EmptyFixedEnv
    con tamaño arbitrario y lo envuelve en FromGym.

    Ejemplo:
      env = SimpleEmpty(task=8, size=(128,128), rgb_img_obs='full')
    """

    def __init__(
        self,
        task,                    # tamaño como int o string (p.ej. '32' o 32)
        render_mode: str = 'rgb_array',
        agent_start_pos: Tuple[int, int] = (1, 1),
        agent_start_dir: int = 0,
        max_steps: Optional[int] = None,
        make_env = None,
        **kwargs
    ):
        try:
            grid_size = int(task)
        except Exception:
            # if task is like "5x5" extract leading number
            if isinstance(task, str) and 'x' in task:
                grid_size = task.split("x")
            else:
                raise ValueError("task debe ser un int o 'NxN' (ej. '16x16')")

        # build the raw env (our EmptyFixedEnv)
        if make_env:
            self.raw = make_env(
                height=int(grid_size[0]),
                width=int(grid_size[1]),
                agent_start_pos=agent_start_pos,
                agent_start_dir=agent_start_dir,
                max_steps=max_steps,
                render_mode=render_mode,
                **kwargs  
            )
        else:
            print("Make agent no existe")
            self.raw = EmptyEnv(
                size=grid_size,
                agent_start_pos=agent_start_pos,
                agent_start_dir=agent_start_dir,
                max_steps=max_steps,
                render_mode=render_mode,
            )
        
        # wrap with your FromGym so rest of pipeline sees same interface
        self._from_gym = from_gym.FromGym(self.raw, obs_key='image', act_key='action')
        # target image size for resizing in _ensure_image (height, width)
        self._renderer = self._get_underlying_renderer(self._from_gym)

    @property
    def env(self):
        return self._from_gym

    def _get_core_env(self):
        """Descender env wrappers hasta llegar al core environment (EmptyFixedEnv)."""
        u = getattr(self._from_gym, "env", self._from_gym)
        for _ in range(10):
            if hasattr(u, "agent_pos") and hasattr(u, "agent_dir") and hasattr(u, "grid"):
                return u
            # try common nested names
            next_u = getattr(u, "env", None) or getattr(u, "_env", None) or getattr(u, "inner_env", None)
            if next_u is None or next_u is u:
                break
            u = next_u
        return None

    @property
    def obs_space(self):
        # Copiamos el obs_space declarado por FromGym y forzamos image a uint8 (H,W,3)
        spaces = self._from_gym.obs_space.copy()
        return spaces

    @property
    def info(self):
        """Info tipada (numpy arrays) para que el driver pueda apilar y luego usar."""
        out = {}
        core = self._get_core_env()
        out["grid_size"] = np.array([int(core.height), int(core.width)], dtype=np.int32)
        pos = core.agent_pos
        row, col = int(pos[0]), int(pos[1])
        out["agent_pos"] = np.array([row, col], dtype=np.int32)
        out["agent_dir"] = core.agent_dir
        out["agent_dir"] = core.agent_dir

        # has_renderer and tile_size
        out["has_renderer"] = np.asarray([1 if self._renderer is not None else 0], dtype=np.int8)

        return out

    @property
    def act_space(self):
        return self._from_gym.act_space

    def reset(self, **kwargs):
        obs = self._from_gym.reset(**kwargs)
        # obs = self._ensure_image(result)
        if "mission" in obs:
            del obs["mission"]
        return obs

    def step(self, action):
        obs = self._from_gym.step(action)
        print(obs["image"])
        if "mission" in obs:
            del obs["mission"]
        return obs

    def close(self):
        try:
            self._from_gym.close()
        except Exception:
            pass

    # ---- helpers ----
    def _get_underlying_renderer(self, env):
        """Desempaqueta env hasta encontrar uno que tenga .render method aceptando 'rgb_array'."""
        underlying = getattr(env, "env", env)  
        for _ in range(10):
            if hasattr(underlying, "render"):
                return underlying
            # bajar un nivel
            next_u = getattr(underlying, "env", None)
            if next_u is None or next_u is underlying:
                break
            underlying = next_u
        return underlying

    


class SimpleImageEnv(SimpleEnv):
    """
    Wrapper similar a SimpleGrid pero que crea dinámicamente un EmptyFixedEnv
    con tamaño arbitrario y lo envuelve en FromGym.

    Ejemplo:
      env = SimpleEmpty(task=8, size=(128,128), full_obs=True, rgb_img_obs='full')
    """

    def __init__(
        self,
        task,
        size: Tuple[int, int] = (80, 80),
        resize: str = 'pillow',
        full_obs: bool = True,
        rgb_img_obs: str = 'full',
        tile_size: Optional[int] = None,
        render_mode: str = 'rgb_array',
        agent_start_pos: Tuple[int, int] = (1, 1),
        agent_start_dir: int = 0,
        max_steps: Optional[int] = None,
        make_env = None,
        **kwargs
    ):
        assert resize in ('opencv', 'pillow'), resize
        
        super().__init__(
            task=task,                    # tamaño como int o string (p.ej. '32' o 32)
            render_mode=render_mode,
            agent_start_pos=agent_start_pos,
            agent_start_dir=agent_start_dir,
            max_steps=max_steps,
            make_env=make_env,
            **kwargs
        )

        raw = self.raw
        # apply wrappers for observations if requested
        if full_obs:
            raw = FullyObsWrapper(raw)

        # if rgb_img_obs and not kwargs.get("onehot", False):
        if rgb_img_obs:
            # normalize option
            if isinstance(rgb_img_obs, str) and rgb_img_obs.lower() == 'full':
                raw = RGBImgObsWrapper(raw)
            else:
                raw = RGBImgPartialObsWrapper(raw)
            """
            obs, info = raw.reset()
            image_data = obs["image"]
            img = Image.fromarray(image_data)
            filename = "test_observation_raw.png"
            img.save(filename)
            """
            
        # wrap with your FromGym so rest of pipeline sees same interface
        self._from_gym = from_gym.FromGym(raw, obs_key='image', act_key='action')
        # target image size for resizing in _ensure_image (height, width)
        self.size = tuple(size)
        self.resize = resize
        self.tile_size = tile_size
        self._renderer = self._get_underlying_renderer(self._from_gym)

    def reset(self, **kwargs):
        result = self._from_gym.reset(**kwargs)
        obs = self._ensure_image(result)
        if "mission" in obs:
            del obs["mission"]
        return obs

    def step(self, action):
        result = self._from_gym.step(action)
        obs = self._ensure_image(result)
        if "mission" in obs:
            del obs["mission"]
        return obs

    def _ensure_image(self, obs):
        # obs puede ser dict o un objeto más; asumimos dict-like
        if not isinstance(obs, dict):
            return obs

        img = obs.get('image', None)
        # es una imagen — la resizeamos a self.size si hace falta
        if (img.shape[0], img.shape[1]) != self.size:
            obs['image'] = self._resize(img, self.size, self.resize)
        else:
            # asegurar dtype uint8
            if img.dtype != np.uint8:
                obs['image'] = img.astype(np.uint8)
            
        return obs
        
        
    def _resize(self, image, size, method):
        if method == 'opencv':
            import cv2
            # cv2.resize espera size=(width,height)
            return cv2.resize(image, (size[1], size[0]), interpolation=cv2.INTER_AREA)
        elif method == 'pillow':
            img = Image.fromarray(image)
            img = img.resize((size[1], size[0]), Image.BILINEAR)
            return np.array(img)
        else:
            raise NotImplementedError(method)


class RawMiddlePoint(EmptyEnv):
    def __init__(
        self, 
        size,
        agent_start_pos,
        agent_start_dir,
        max_steps,
        render_mode,
        **kwargs
    ):
        super().__init__(
            size=size,
            agent_start_pos=agent_start_pos,
            agent_start_dir=agent_start_dir,
            max_steps=max_steps,
            render_mode=render_mode,
        )
        self.corner_goal_pos = None
        self.middle_goal_pos = None

    def _gen_grid(self, width, height):
        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)
        
        # Posición del objetivo de la esquina (Reward 1.0)
        self.corner_goal_pos = (width - 2, height - 2)
        
        # Posición del objetivo del medio (Reward 0.2)
        # Se usa int() para manejar cualquier tamaño de cuadrícula
        mid_x = int((width - 1) / 2)
        mid_y = int((height - 1) / 2)
        self.middle_goal_pos = (mid_x, mid_y)

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), *self.corner_goal_pos)
        self.put_obj(Goal(), *self.middle_goal_pos)

        # Place the agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()

        self.mission = "get to the green goal square"

    def step(self, action):
        # Llamar al step del padre (maneja movimiento, colisiones, y detección de Goal)
        obs, reward, terminated, truncated, info = super().step(action)

        # Convertir la posición del agente a tupla para una comparación fácil
        agent_pos_tuple = tuple(self.agent_pos)

        # Si el episodio terminó por caer en un Goal (terminated=True)
        if terminated:
            if agent_pos_tuple == self.corner_goal_pos:
                # Meta de la esquina: recompensa 1.0
                reward = 10.0
            elif agent_pos_tuple == self.middle_goal_pos:
                # Meta del medio: recompensa 0.2
                reward = 0.2
            # Si terminó por otra razón (ej. max_steps), se mantiene la recompensa por defecto.
        
        return obs, reward, terminated, truncated, info

class MiddleGoal(SimpleImageEnv):
    """
    Wrapper similar a SimpleGrid pero que crea dinámicamente un EmptyFixedEnv
    con tamaño arbitrario y lo envuelve en FromGym.

    Ejemplo:
      env = MiddleGoal(task=8, size=(128,128), full_obs=True, rgb_img_obs='full')
    """

    def __init__(
        self,
        task,                    # tamaño como int o string (p.ej. '32' o 32)
        size: Tuple[int, int] = (80, 80),
        resize: str = 'pillow',
        full_obs: bool = True,
        rgb_img_obs: str = 'full',
        tile_size: Optional[int] = None,
        render_mode: str = 'rgb_array',
        agent_start_pos: Tuple[int, int] = (1, 1),
        agent_start_dir: int = 0,
        max_steps: Optional[int] = None,
        **kwargs
    ):
        super().__init__(
            make_env=RawMiddlePoint,
            task=task,
            size=size,
            resize=resize,
            full_obs=full_obs,
            tile_size=tile_size,
            render_mode=render_mode,
            rgb_img_obs=rgb_img_obs,
            agent_start_pos=agent_start_pos,
            agent_start_dir=agent_start_dir,
            max_steps=max_steps,
            kwargs=kwargs           
        )
        

class CookiePedro(SimpleImageEnv):
    def __init__(
        self,
        task=None,
        **kwargs
    ):
        from cookie_env.env import CookieEnv
        super().__init__(
            task=task,
            make_env=CookieEnv,
            full_obs=False,
            rgb_img_obs="partial",
            agent_start_pos=(14, 14),
            onehot=False,
        )


# TODO: Fix Cookie pedro as a new entirely different model
# because it is not a picture!
class CookiePedroOneHot(SimpleEnv):
    def __init__(
        self,
        task=None,
        **kwargs
    ):
        from cookie_env.env import CookieEnv
        super().__init__(
            task=task,
            make_env=CookieEnv,
            agent_start_pos=(14, 14),
            onehot=True
        )
        
class DeterministicCookie(SimpleImageEnv):
    def __init__(
        self,
        task=None,
        **kwargs
    ):
        from cookie_env.env import CookieEnv
        from cookie_env.utils.spawner import deterministic_corner
        super().__init__(
            task=task,
            make_env=CookieEnv,
            cookie_spawner=deterministic_corner,
            agent_start_pos=(14, 14),
            onehot=False,
            **kwargs,
        )