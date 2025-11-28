import elements
import embodied
import numpy as np
import minigrid
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace
from minigrid.wrappers import FullyObsWrapper, RGBImgPartialObsWrapper, RGBImgObsWrapper
from . import from_gym
import gymnasium as gym

from PIL import Image

# no forzar import global de minigrid aquí (puede no estar en CI)
# we'll import the wrappers lazily below when needed


class SimpleGrid(embodied.Env):
    def __init__(self,
               task,
               size=(80, 80),
               resize='pillow',
               full_obs=True,
               rgb_img_obs='full',
               tile_size=None,
               render_mode='rgb_array',
               **kwargs):
        assert resize in ('opencv', 'pillow'), resize
        env_id = f"MiniGrid-Empty-{task}-v0"        
        raw = gym.make(env_id, render_mode=render_mode, **kwargs)

        if full_obs:
            raw = FullyObsWrapper(raw)

        if rgb_img_obs:
            # normalizar valor
            if rgb_img_obs is True:
                rgb_img_obs = 'partial'
            if isinstance(rgb_img_obs, str) and rgb_img_obs.lower() == 'full':
                raw = RGBImgObsWrapper(raw)
            else:
                raw = RGBImgPartialObsWrapper(raw)

        self._from_gym = from_gym.FromGym(raw, obs_key='image', act_key='action')
        self.size = tuple(size)
        self.resize = resize
        self.tile_size = tile_size

        self._renderer = self._get_underlying_renderer(self._from_gym)

    @property
    def env(self):
        return self._from_gym

    def _get_core_env(self):
        """Descender wrappers hasta encontrar el core que expone agent_pos/agent_dir/grid."""
        u = self._from_gym
        for _ in range(10):
            # common direct attributes (some wrappers expose them)
            if hasattr(u, "agent_pos") and hasattr(u, "agent_dir"):
                return u
            # prefer unwrapped
            if hasattr(u, "unwrapped"):
                try:
                    u = u.unwrapped
                    continue
                except Exception:
                    pass
            # descend typical wrapper attribute names
            next_u = getattr(u, "env", None) or getattr(u, "_env", None) or getattr(u, "inner", None) or getattr(u, "inner_env", None)
            if next_u is None or next_u is u:
                break
            u = next_u
        return None

    @property
    def info(self):
        """
        Diccionario con metadatos útiles del entorno:
          - grid_size / width / height (cuando se puede)
          - agent_pos (row, col) y agent_dir si están disponibles
          - has_renderer, tile_size
        """
        out = {}
        core = self._get_core_env()
        # grid size
        gsize = core.grid
        out["grid_size"] = np.array([int(core.height), int(core.width)], dtype=np.int32)

        # agent position & dir (normalize to (row, col))
        pos = core.agent_pos
        row, col = int(pos[1]), int(pos[0])
        out["agent_pos"] = np.array([row, col], dtype=np.int32)
        out["agent_dir"] = core.agent_dir
        # out["mission"] = core["mission"] 
        
        # renderer availability & tile_size
        out["has_renderer"] = np.array([1 if self._renderer is not None else 0], dtype=np.int8)
        # out["tile_size"] = np.array([self.tile_size], dtype=np.int32)

        return out
    
    @property
    def obs_space(self):
        # Copiamos el obs_space declarado por FromGym y forzamos image a uint8 (H,W,3)
        spaces = self._from_gym.obs_space.copy()
        spaces['image'] = elements.Space(np.uint8, (*self.size, 3))
        return spaces
    
    @property
    def act_space(self):
        return self._from_gym.act_space
    
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

    def close(self):
        try:
            self._from_gym.close()
        except Exception:
            pass

    # ---- helpers ----
    def _get_underlying_renderer(self, env):
        """Desempaqueta env hasta encontrar uno que tenga .render method aceptando 'rgb_array'."""
        underlying = getattr(env, "env", env)  # FromGym has .env pointing to the wrapped env
        # descender por .env/.inner_env/unwrapped
        for _ in range(10):
            # prefer unwrapped if present
            if hasattr(underlying, "unwrapped"):
                try:
                    underlying = underlying.unwrapped
                except Exception:
                    pass
            if hasattr(underlying, "render"):
                return underlying
            # bajar un nivel
            next_u = getattr(underlying, "env", None) or getattr(underlying, "inner_env", None)
            if next_u is None or next_u is underlying:
                break
            underlying = next_u
        return underlying

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
