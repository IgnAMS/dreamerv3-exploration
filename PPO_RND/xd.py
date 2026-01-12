import gymnasium as gym
import cookie_env  # importante: esto dispara el register
from embodied.envs.from_gym import FromGym
import numpy as np
from gymnasium import spaces
from minigrid.wrappers import FullyObsWrapper, RGBImgPartialObsWrapper, RGBImgObsWrapper

class ImageDirectionWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        img_space = env.observation_space["image"]
        dir_space = env.observation_space["direction"]

        H, W, C = img_space.shape
        self.n_dir = dir_space.n

        # Canales: RGB + direction one-hot
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(C + self.n_dir, H, W),
            dtype=np.float32
        )

    def observation(self, obs):
        # Imagen: HWC → CHW
        img = obs["image"].astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))  # (3,H,W)

        # Direction → mapas constantes
        direction = np.zeros((self.n_dir, img.shape[1], img.shape[2]), dtype=np.float32)
        direction[obs["direction"], :, :] = 1.0

        return np.concatenate([img, direction], axis=0)
    
print("\n\n")

env = gym.make("CookieEnv-v0")
env = RGBImgObsWrapper(env)
print(env.observation_space, "\n\n")
env2 = FromGym(env, obs_key="image")
print(env2.obs_space, "\n\n")

env3 = ImageDirectionWrapper(env)
print(env3.observation_space, "\n\n")
obs, _ = env3.reset()
print(obs.shape)