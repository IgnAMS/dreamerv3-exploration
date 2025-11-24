# play_minigrid.py
import pygame
from embodied.envs.minigrid import SimpleGrid
import numpy as np

env = SimpleGrid(task="5x5", size=(80,80), tile_size=16,
                    rgb_img_obs='full', resize='pillow')

obs = env.reset()
frame = env._from_gym.render()  # rgb array

h, w, _ = frame.shape
pygame.init()
window = pygame.display.set_mode((w, h))
pygame.display.set_caption("MiniGrid play")
clock = pygame.time.Clock()
FPS = 30

KEY_TO_ACTION = {
    pygame.K_LEFT: 0,
    pygame.K_RIGHT: 1,
    pygame.K_UP: 2,
    pygame.K_SPACE: 3,
    pygame.K_b: 4,
}

running = True
while running:
    action = None
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_q:
                env.reset()
            elif event.key in KEY_TO_ACTION:
                action = KEY_TO_ACTION[event.key]

    if action is None:
        frame = env._from_gym.render()
    else:
        obs = env.step({'action': action, 'reset': False})
        frame = obs['image']

    if frame is not None:
        surf = pygame.surfarray.make_surface(frame.swapaxes(0,1))
        window.blit(surf, (0,0))
        pygame.display.flip()

    clock.tick(FPS)

env.close()
pygame.quit()
