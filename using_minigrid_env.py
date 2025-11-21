import gymnasium as gym
from embodied.envs.from_gym import FromGym
import minigrid
import pygame
from minigrid.wrappers import FullyObsWrapper, RGBImgObsWrapper, ImgObsWrapper, RGBImgPartialObsWrapper
import time
import numpy as np
import matplotlib.pyplot as plt

KEY_TO_ACTION = {
    "left": 0,
    "right": 1,
    "up": 2,
    "space": 3,
    "b": 4,
}
raw_env = gym.make("MiniGrid-Empty-16x16-v0", render_mode="rgb_array")
obs, _ = raw_env.reset()
print(obs['image'].shape)
# full_obs_env = FullyObsWrapper(raw_env)
full_obs_env = RGBImgObsWrapper(raw_env)

obs, _ = full_obs_env.reset()
# plt.imshow(obs['image'])
# plt.axis("off")
# plt.savefig("primer_frame_obs_completa.png", bbox_inches="tight", pad_inches=0)

env = FromGym(full_obs_env, obs_key="image", act_key="action")

obs = env.reset()

print("Controles: ↑ left right space b q ESC")

pygame.init()
# pedir un frame inicial para conocer tamaño
frame = env.render()  # debería devolver rgb_array
if frame is None:
    raise RuntimeError("env.render() devolvió None. Crea el env con render_mode='rgb_array'.")

h, w, _ = frame.shape
window = pygame.display.set_mode((w, h))
pygame.display.set_caption("MiniGrid manual control (pygame)")

clock = pygame.time.Clock()
FPS = 30

# --- mapa de teclas a acciónes MiniGrid ---
# acciones clásicas: 0=left, 1=right, 2=forward, 3=toggle, 4=drop
KEY_TO_ACTION = {
    pygame.K_LEFT: 0,
    pygame.K_RIGHT: 1,
    pygame.K_UP: 2,
    pygame.K_SPACE: 3,
    pygame.K_b: 4,
}

print("Controles: ← → ↑ space b | q=resetea | ESC=salir")

running = True
paused = False

while running:
    # procesar eventos
    action = None
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_q:
                res = env.reset()
                if isinstance(res, tuple) and len(res) == 2:
                    obs, info = res
                else:
                    obs = res
                    info = {}        
                
            elif event.key in KEY_TO_ACTION:
                action = KEY_TO_ACTION[event.key]
            elif event.key == pygame.K_p:
                paused = not paused
                print("Paused:", paused)

    if paused:
        clock.tick(FPS)
        continue

    if action is None:
        # si no presionaste acción, simplemente dibujamos el frame actual
        frame = env.render()
    else:
        # ejecutar step
        result = env.step({'action': action, 'reset': False})    
        
        reward = float(result.get('reward', 0.0))
        terminated = bool(result.get('is_last', False))
        truncated = False
        # pedir frame tras step
        frame = env.render()

        print(f"Acción={action} reward={reward} terminated={terminated} truncated={truncated}")

        if terminated or truncated:
            print("Episodio terminó → reseteando")
            res = env.reset()
            obs = res
            info = {}
            frame = env.render()

    # --- dibujar frame en pygame ---
    if frame is not None:
        frame_to_show = frame
        surf = pygame.surfarray.make_surface(frame_to_show.swapaxes(0, 1))
        window.blit(surf, (0, 0))
        pygame.display.flip()

    clock.tick(FPS)

# cleanup
env.close()
pygame.quit()