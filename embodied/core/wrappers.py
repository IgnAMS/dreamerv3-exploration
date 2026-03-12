import functools
import time

import elements
import numpy as np


class Wrapper:

  def __init__(self, env):
    self.env = env

  def __len__(self):
    return len(self.env)

  def __bool__(self):
    return bool(self.env)

  def __getattr__(self, name):
    if name.startswith('__'):
      raise AttributeError(name)
    try:
      return getattr(self.env, name)
    except AttributeError:
      raise ValueError(name)


class TimeLimit(Wrapper):

  def __init__(self, env, duration, reset=True):
    super().__init__(env)
    self._duration = duration
    self._reset = reset
    self._step = 0
    self._done = False

  def step(self, action):
    if action['reset'] or self._done:
      self._step = 0
      self._done = False
      if self._reset:
        action.update(reset=True)
        return self.env.step(action)
      else:
        action.update(reset=False)
        obs = self.env.step(action)
        obs['is_first'] = True
        return obs
    self._step += 1
    obs = self.env.step(action)
    if self._duration and self._step >= self._duration:
      obs['is_last'] = True
    self._done = obs['is_last']
    return obs


class ActionRepeat(Wrapper):

  def __init__(self, env, repeat):
    super().__init__(env)
    self._repeat = repeat

  def step(self, action):
    if action['reset']:
      return self.env.step(action)
    reward = 0.0
    for _ in range(self._repeat):
      obs = self.env.step(action)
      reward += obs['reward']
      if obs['is_last'] or obs['is_terminal']:
        break
    obs['reward'] = np.float32(reward)
    return obs


class ClipAction(Wrapper):

  def __init__(self, env, key='action', low=-1, high=1):
    super().__init__(env)
    self._key = key
    self._low = low
    self._high = high

  def step(self, action):
    clipped = np.clip(action[self._key], self._low, self._high)
    return self.env.step({**action, self._key: clipped})


class NormalizeAction(Wrapper):

  def __init__(self, env, key='action'):
    super().__init__(env)
    self._key = key
    self._space = env.act_space[key]
    self._mask = np.isfinite(self._space.low) & np.isfinite(self._space.high)
    self._low = np.where(self._mask, self._space.low, -1)
    self._high = np.where(self._mask, self._space.high, 1)

  @functools.cached_property
  def act_space(self):
    low = np.where(self._mask, -np.ones_like(self._low), self._low)
    high = np.where(self._mask, np.ones_like(self._low), self._high)
    space = elements.Space(np.float32, self._space.shape, low, high)
    return {**self.env.act_space, self._key: space}

  def step(self, action):
    orig = (action[self._key] + 1) / 2 * (self._high - self._low) + self._low
    orig = np.where(self._mask, orig, action[self._key])
    return self.env.step({**action, self._key: orig})


# class ExpandScalars(Wrapper):
#
#   def __init__(self, env):
#     super().__init__(env)
#     self._obs_expanded = []
#     self._obs_space = {}
#     for key, space in self.env.obs_space.items():
#       if space.shape == () and key != 'reward' and not space.discrete:
#         space = elements.Space(space.dtype, (1,), space.low, space.high)
#         self._obs_expanded.append(key)
#       self._obs_space[key] = space
#     self._act_expanded = []
#     self._act_space = {}
#     for key, space in self.env.act_space.items():
#       if space.shape == () and not space.discrete:
#         space = elements.Space(space.dtype, (1,), space.low, space.high)
#         self._act_expanded.append(key)
#       self._act_space[key] = space
#
#   @functools.cached_property
#   def obs_space(self):
#     return self._obs_space
#
#   @functools.cached_property
#   def act_space(self):
#     return self._act_space
#
#   def step(self, action):
#     action = {
#         key: np.squeeze(value, 0) if key in self._act_expanded else value
#         for key, value in action.items()}
#     obs = self.env.step(action)
#     obs = {
#         key: np.expand_dims(value, 0) if key in self._obs_expanded else value
#         for key, value in obs.items()}
#     return obs
#
#
# class FlattenTwoDimObs(Wrapper):
#
#   def __init__(self, env):
#     super().__init__(env)
#     self._keys = []
#     self._obs_space = {}
#     for key, space in self.env.obs_space.items():
#       if len(space.shape) == 2:
#         space = elements.Space(
#             space.dtype,
#             (int(np.prod(space.shape)),),
#             space.low.flatten(),
#             space.high.flatten())
#         self._keys.append(key)
#       self._obs_space[key] = space
#
#   @functools.cached_property
#   def obs_space(self):
#     return self._obs_space
#
#   def step(self, action):
#     obs = self.env.step(action).copy()
#     for key in self._keys:
#       obs[key] = obs[key].flatten()
#     return obs
#
#
# class FlattenTwoDimActions(Wrapper):
#
#   def __init__(self, env):
#     super().__init__(env)
#     self._origs = {}
#     self._act_space = {}
#     for key, space in self.env.act_space.items():
#       if len(space.shape) == 2:
#         space = elements.Space(
#             space.dtype,
#             (int(np.prod(space.shape)),),
#             space.low.flatten(),
#             space.high.flatten())
#         self._origs[key] = space.shape
#       self._act_space[key] = space
#
#   @functools.cached_property
#   def act_space(self):
#     return self._act_space
#
#   def step(self, action):
#     action = action.copy()
#     for key, shape in self._origs.items():
#       action[key] = action[key].reshape(shape)
#     return self.env.step(action)


class UnifyDtypes(Wrapper):

  def __init__(self, env):
    super().__init__(env)
    self._obs_space, _, self._obs_outer = self._convert(env.obs_space)
    self._act_space, self._act_inner, _ = self._convert(env.act_space)

  @property
  def obs_space(self):
    return self._obs_space

  @property
  def act_space(self):
    return self._act_space

  def step(self, action):
    action = action.copy()
    for key, dtype in self._act_inner.items():
      action[key] = np.asarray(action[key], dtype)
    obs = self.env.step(action)
    for key, dtype in self._obs_outer.items():
      obs[key] = np.asarray(obs[key], dtype)
    return obs

  def _convert(self, spaces):
    results, befores, afters = {}, {}, {}
    for key, space in spaces.items():
      before = after = space.dtype
      if np.issubdtype(before, np.floating):
        after = np.float32
      elif np.issubdtype(before, np.uint8):
        after = np.uint8
      elif np.issubdtype(before, np.integer):
        after = np.int32
      befores[key] = before
      afters[key] = after
      results[key] = elements.Space(after, space.shape, space.low, space.high)
    return results, befores, afters


class CheckSpaces(Wrapper):

  def __init__(self, env):
    assert not (env.obs_space.keys() & env.act_space.keys()), (
        env.obs_space.keys(), env.act_space.keys())
    super().__init__(env)

  def step(self, action):
    for key, value in action.items():
      self._check(value, self.env.act_space[key], key)
    obs = self.env.step(action)
    for key, value in obs.items():
      self._check(value, self.env.obs_space[key], key)
    return obs

  def _check(self, value, space, key):
    if not isinstance(value, (
        np.ndarray, np.generic, list, tuple, int, float, bool)):
      raise TypeError(f'Invalid type {type(value)} for key {key}.')
    if value in space:
      return
    dtype = np.array(value).dtype
    shape = np.array(value).shape
    lowest, highest = np.min(value), np.max(value)
    raise ValueError(
        f"Value for '{key}' with dtype {dtype}, shape {shape}, "
        f"lowest {lowest}, highest {highest} is not in {space}.")


class DiscretizeAction(Wrapper):

  def __init__(self, env, key='action', bins=5):
    super().__init__(env)
    self._dims = np.squeeze(env.act_space[key].shape, 0).item()
    self._values = np.linspace(-1, 1, bins)
    self._key = key

  @functools.cached_property
  def act_space(self):
    space = elements.Space(np.int32, self._dims, 0, len(self._values))
    return {**self.env.act_space, self._key: space}

  def step(self, action):
    continuous = np.take(self._values, action[self._key])
    return self.env.step({**action, self._key: continuous})


class ResizeImage(Wrapper):

  def __init__(self, env, size=(64, 64)):
    super().__init__(env)
    self._size = size
    self._keys = [
        k for k, v in env.obs_space.items()
        if len(v.shape) > 1 and v.shape[:2] != size]
    print(f'Resizing keys {",".join(self._keys)} to {self._size}.')
    if self._keys:
      from PIL import Image
      self._Image = Image

  @functools.cached_property
  def obs_space(self):
    spaces = self.env.obs_space
    for key in self._keys:
      shape = self._size + spaces[key].shape[2:]
      spaces[key] = elements.Space(np.uint8, shape)
    return spaces

  def step(self, action):
    obs = self.env.step(action)
    for key in self._keys:
      obs[key] = self._resize(obs[key])
    return obs

  def _resize(self, image):
    image = self._Image.fromarray(image)
    image = image.resize(self._size, self._Image.NEAREST)
    image = np.array(image)
    return image


# class RenderImage(Wrapper):
#
#   def __init__(self, env, key='image'):
#     super().__init__(env)
#     self._key = key
#     self._shape = self.env.render().shape
#
#   @functools.cached_property
#   def obs_space(self):
#     spaces = self.env.obs_space
#     spaces[self._key] = elements.Space(np.uint8, self._shape)
#     return spaces
#
#   def step(self, action):
#     obs = self.env.step(action)
#     obs[self._key] = self.env.render()
#     return obs


class BackwardReturn(Wrapper):

  def __init__(self, env, horizon):
    super().__init__(env)
    self._discount = 1 - 1 / horizon
    self._bwreturn = 0.0

  @functools.cached_property
  def obs_space(self):
    return {
        **self.env.obs_space,
        'bwreturn': elements.Space(np.float32),
    }

  def step(self, action):
    obs = self.env.step(action)
    self._bwreturn *= (1 - obs['is_first']) * self._discount
    self._bwreturn += obs['reward']
    obs['bwreturn'] = np.float32(self._bwreturn)
    return obs


class AddObs(Wrapper):

  def __init__(self, env, key, value, space):
    super().__init__(env)
    self._key = key
    self._value = value
    self._space = space

  @functools.cached_property
  def obs_space(self):
    return {
        **self.env.obs_space,
        self._key: self._space,
    }

  def step(self, action):
    obs = self.env.step(action)
    obs[self._key] = self._value
    return obs


class RestartOnException(Wrapper):

  def __init__(
      self, ctor, exceptions=(Exception,), window=300, maxfails=2, wait=20):
    if not isinstance(exceptions, (tuple, list)):
        exceptions = [exceptions]
    self._ctor = ctor
    self._exceptions = tuple(exceptions)
    self._window = window
    self._maxfails = maxfails
    self._wait = wait
    self._last = time.time()
    self._fails = 0
    super().__init__(self._ctor())

  def step(self, action):
    try:
      return self.env.step(action)
    except self._exceptions as e:
      if time.time() > self._last + self._window:
        self._last = time.time()
        self._fails = 1
      else:
        self._fails += 1
      if self._fails > self._maxfails:
        raise RuntimeError('The env crashed too many times.')
      message = f'Restarting env after crash with {type(e).__name__}: {e}'
      print(message, flush=True)
      time.sleep(self._wait)
      self.env = self._ctor()
      action['reset'] = np.ones_like(action['reset'])
      return self.env.step(action)


class AddGoalWrapper(Wrapper):
    """
    Modifica la observación de imagen añadiendo `goal_rows` filas al bottom.

    Parámetros
    ----------
    env       : entorno subyacente
    obs_key   : key de la imagen en obs_space  (default 'image')
    goal_rows : número de filas del grid a usar como goal  (default 1)
    """

    def __init__(self, env, obs_key: str = 'image', goal_rows: int = 1):
        super().__init__(env)
        self._env = env
        self._obs_key = obs_key
        self._goal_rows = goal_rows

        img_space = env.obs_space[obs_key]      # Space(uint8, (H, W, C))
        H, W, C = img_space.shape
        assert goal_rows <= H, f"goal_rows={goal_rows} > H={H}"
        self._H, self._W, self._C = H, W, C

        # imagen ampliada: H + goal_rows filas
        self._new_img_space = elements.Space(
            np.uint8, (H + goal_rows, W, C))
        # achieved_goal: las goal_rows extraídas, en su forma espacial
        self._achieved_space = elements.Space(
            np.uint8, (goal_rows, W, C))

        # placeholder de zeros para el goal al momento de colección
        self._goal_zeros = np.zeros((goal_rows, W, C), dtype=np.uint8)

    def __getattr__(self, name):
        return getattr(self._env, name)

    @property
    def obs_space(self):
        spaces = dict(self._env.obs_space)
        spaces[self._obs_key] = self._new_img_space   # imagen ampliada
        spaces['achieved_goal'] = self._achieved_space
        return spaces

    @property
    def act_space(self):
        return self._env.act_space

    def step(self, action):
        obs = self._env.step(action)
        img = obs.get(self._obs_key)                  # (H, W, C) uint8

        if img is not None:
            # achieved_goal = primeras goal_rows filas del grid
            achieved = img[:self._goal_rows].copy()   # (goal_rows, W, C)
            # imagen ampliada = imagen original + zeros al bottom (placeholder)
            new_img = np.concatenate(
                [img, self._goal_zeros], axis=0)      # (H+goal_rows, W, C)
        else:
            achieved = self._goal_zeros.copy()
            new_img = np.zeros(
                (self._H + self._goal_rows, self._W, self._C), dtype=np.uint8)

        obs[self._obs_key] = new_img
        obs['achieved_goal'] = achieved
        return obs

    def close(self):
        self._env.close()


# ---------------------------------------------------------------------------
# HER stream wrapper
# ---------------------------------------------------------------------------

class HERStream:
    """
    Wrapper de stream que relabela el goal en la imagen (últimas goal_rows filas).

    Parámetros
    ----------
    stream    : iterable de batches  dict{str: ndarray(B, T, ...)}
    obs_key   : key de la imagen en el batch  (default 'image')
    goal_rows : filas reservadas al bottom de la imagen para el goal
    strategy  : 'random_z' | 'future' | 'final' | 'random_ep'
    her_ratio : fracción de pasos que se relabelan
    seed      : semilla opcional
    """

    def __init__(
        self,
        stream,
        obs_key: str = 'image',
        goal_rows: int = 1,
        strategy: str = 'random_z',
        her_ratio: float = 0.8,
        seed: int | None = None,
    ):
        assert strategy in ('random_z', 'future', 'final', 'random_ep'), strategy
        self._stream = stream
        self._obs_key = obs_key
        self._goal_rows = goal_rows
        self._strategy = strategy
        self._her_ratio = her_ratio
        self._rng = np.random.default_rng(seed)

    def __iter__(self):
        for batch in self._stream:
            yield self._relabel(batch)

    # ------------------------------------------------------------------
    def _relabel(self, batch: dict) -> dict:
        batch = dict(batch)
        imgs = batch[self._obs_key].copy()  # (B, T, H+goal_rows, W, C) uint8
        B, T = imgs.shape[:2]
        H_obs = imgs.shape[2] - self._goal_rows   # filas reales de la obs

        # ── Fase de prueba: ruido aleatorio como goal ─────────────────
        if self._strategy == 'random_z':
            noise = self._rng.integers(
                0, 256,
                (B, T, self._goal_rows, imgs.shape[3], imgs.shape[4]),
                dtype=np.uint8)
            imgs[:, :, H_obs:, :, :] = noise
            batch[self._obs_key] = imgs
            return batch

        # ── Estrategias HER: fuente = achieved_goal almacenado ────────
        if 'achieved_goal' not in batch:
            return batch

        achieved = batch['achieved_goal']   # (B, T, goal_rows, W, C) uint8
        mask = self._rng.random((B, T)) < self._her_ratio

        if self._strategy == 'future':
            for b in range(B):
                for t in range(T):
                    if mask[b, t]:
                        t_future = int(self._rng.integers(t, T))
                        imgs[b, t, H_obs:] = achieved[b, t_future]

        elif self._strategy == 'final':
            for b in range(B):
                last_t = T - 1
                for t2 in range(T - 1, -1, -1):
                    if batch['is_last'][b, t2]:
                        last_t = t2
                        break
                for t in range(T):
                    if mask[b, t]:
                        imgs[b, t, H_obs:] = achieved[b, last_t]

        elif self._strategy == 'random_ep':
            for b in range(B):
                for t in range(T):
                    if mask[b, t]:
                        rb = int(self._rng.integers(B))
                        rt = int(self._rng.integers(T))
                        imgs[b, t, H_obs:] = achieved[rb, rt]

        batch[self._obs_key] = imgs
        return batch