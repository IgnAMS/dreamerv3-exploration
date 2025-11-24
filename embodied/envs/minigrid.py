import elements
import embodied
import numpy as np
import minigrid
from minigrid.wrappers import FullyObsWrapper, RGBImgPartialObsWrapper, RGBImgObsWrapper
from . import from_gym
import gymnasium as gym

from PIL import Image

# no forzar import global de minigrid aquí (puede no estar en CI)
# we'll import the wrappers lazily below when needed


class MiniGrid(embodied.Env):
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
        # construir env id si el usuario pasó solo "5x5"
        print(task)
        if isinstance(task, str) and ("MiniGrid" in task or task.startswith("MiniGrid-")):
            env_id = task
        else:
            env_id = f"MiniGrid-Empty-{task}-v0"

        # import local from_gym factory

        # Preparar gym env y aplicar wrappers *antes* de pasar a FromGym si hace falta
        
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
        # FromGym.reset() devuelve obs dict
        res = self._from_gym.reset(**kwargs)
        # some FromGym implementations may return (obs, info) — normalize:
        if isinstance(res, tuple) and len(res) == 2:
            obs, info = res
        else:
            obs = res
            obs = self._ensure_image(obs)
        
        return obs

    def step(self, action):
        # Forward action through FromGym
        res = self._from_gym.step(action)
        # FromGym likely returns an obs dict (or maybe tuple). Normalize:
        if isinstance(res, tuple) and len(res) == 2:
            obs, info = res
        else:
            obs = res
            obs = self._ensure_image(obs)
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
        """
        Garantiza que obs['image'] sea una imagen RGB uint8 del tamaño self.size.
        - Si obs ya contiene 'image' con shape (H,W,3) y tipo uint8 lo resizea.
        - Si no, pide renderer.render('rgb_array', tile_size=...) y lo resizea.
        """
        # obs puede ser dict o un objeto más; asumimos dict-like
        if not isinstance(obs, dict):
            return obs

        img = obs.get('image', None)
        if isinstance(img, np.ndarray) and img.ndim == 3 and img.shape[2] == 3:
            # es una imagen — la resizeamos a self.size si hace falta
            if (img.shape[0], img.shape[1]) != self.size:
                obs['image'] = self._resize(img, self.size, self.resize)
            else:
                # asegurar dtype uint8
                if img.dtype != np.uint8:
                    obs['image'] = img.astype(np.uint8)
            return obs

        # Si llegamos aquí, obs['image'] no es una imagen RGB: pedimos frame al renderer
        frame = None
        # si tenemos renderer y soporta tile_size en render:
        if self._renderer is not None:
            try:
                if self.tile_size is not None:
                    frame = self._renderer.render("rgb_array", tile_size=self.tile_size)
                else:
                    frame = self._renderer.render("rgb_array")
            except TypeError:
                # algunos renders no aceptan tile_size
                try:
                    frame = self._renderer.render("rgb_array")
                except Exception:
                    frame = None
            except Exception:
                frame = None

        if frame is None:
            # Fallback: si obs contiene 'image' simbólica (grid ints), intentamos convertirla a rgb
            # pero eso requiere lógica de palette; por simplicidad dejamos la misma observación.
            return obs

        # ahora tenemos frame; resizearlo si procede
        if (frame.shape[0], frame.shape[1]) != self.size:
            frame = self._resize(frame, self.size, self.resize)
        
        # asegurar uint8
        if frame.dtype != np.uint8:
            frame = (255 * np.clip(frame, 0, 1)).astype(np.uint8)

        obs['image'] = frame
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
