#!/usr/bin/env python3
"""
Interactive player for SimpleEmpty (arbitrary size).

Requirements:
  pip install gymnasium minigrid pygame matplotlib embodied elements
(or use your project's venv that already contains them).

Usage:
  python play_minigrid.py
"""

import time
import numpy as np
import pygame
from pathlib import Path

# Import your wrapper (ajusta ruta si lo guardaste en otro módulo)
# from embodied.envs.minigrid_empty import SimpleEmpty
# Si tu archivo se llama diferente, ajusta la import.
from embodied.envs.new_minigrid import (
    SimpleImageEnv, 
    MiddleGoal, 
    RawMiddlePoint, 
    CookiePedro, 
    CookiePedroOneHot,
    DeterministicCookie,
    TwoCookies,
    CookiePedroFull,
    CookiePedroFullFixed,
    Corridor,
    TwoRooms
)

# key->action mapping (MiniGrid classic)
KEY_TO_ACTION = {
    # Movimiento clásico
    pygame.K_LEFT: 0,      # turn left
    pygame.K_RIGHT: 1,     # turn right
    pygame.K_UP: 2,        # forward
    pygame.K_TAB: 3,       # pick up
    pygame.K_PAGEUP: 3,    # pick up (alternative)
    pygame.K_LSHIFT: 4,    # drop
    pygame.K_PAGEDOWN: 4,  # drop (alternative)
    pygame.K_SPACE: 5,     # toggle
    pygame.K_RETURN: 6,    # done
}

"""
# key->action mapping (MiniGrid classic)
KEY_TO_ACTION = {
    pygame.K_LEFT: 0,   # turn left
    pygame.K_RIGHT: 1,  # turn right
    pygame.K_UP: 2,     # forward
    pygame.K_SPACE: 3,  # toggle / interact
    pygame.K_b: 4,      # drop
}
"""


def request_frame(env_wrapper, tile_size=None):
    """
    Intenta pedir imagen de distintas formas, para máxima compatibilidad:
    - env_wrapper.render() si lo soporta
    - si falla, intenta usar underlying renderer con tile_size
    """
    # try direct render (FromGym wrapper exposes render)
    try:
        frame = env_wrapper.render()
        if frame is not None:
            return frame
    except TypeError:
        # some render signatures want kwargs; ignore and try underlying renderer
        pass
    except Exception:
        pass

    # fallback: underlying renderer
    core = env_wrapper._get_underlying_renderer(env_wrapper)
    if core is None:
        # try access internal attribute
        core = getattr(env_wrapper, "_from_gym", None) or getattr(env_wrapper, "env", None)
    if core is None:
        return None

    # try with tile_size if provided
    try:
        if tile_size is not None:
            return core.render("rgb_array", tile_size=tile_size)
    except Exception:
        pass
    try:
        return core.render("rgb_array")
    except Exception:
        return None


def main():
    # Configurables
    GRID_SIZE = "18x29"             # prueba 8,16,32,64...
    PIXEL_SIZE = (160, 160)    # tamaño resultante de la imagen que verás (height,width)
    TILE_SIZE = 8              # tamaño del tile en px para el renderer; mayor => ventana más grande
    FULL_OBS = True
    RGB_IMG = 'full'           # 'full' para imagen completa, 'partial' para partial rgb
    RENDER_MODE = 'rgb_array'

    print("Creando env...")
    """
    env = Corridor(
        task="30",
        size=PIXEL_SIZE,
        resize='pillow',
        tile_size=TILE_SIZE,
        render_mode=RENDER_MODE,
    )
    """
    env = TwoRooms(
        task="18x29",
        size=PIXEL_SIZE,
        resize='pillow',
        tile_size=TILE_SIZE,
        render_mode=RENDER_MODE,
    ) 
    
    
    # reset and get starting obs
    obs = env.reset()

    # prepare pygame window from first frame
    frame = request_frame(env, tile_size=TILE_SIZE)
    if frame is None:
        raise RuntimeError("No pude obtener frame del env (asegura render_mode='rgb_array' y renderer presente).")

    h, w, _ = frame.shape
    pygame.init()
    window = pygame.display.set_mode((w, h))
    pygame.display.set_caption(f"MiniGrid {GRID_SIZE}x{GRID_SIZE} manual control")
    clock = pygame.time.Clock()
    FPS = 30

    print("Controles: ← → ↑  SPACE  b  |  q = reset  |  ESC = salir")
    running = True
    paused = False

    while running:
        action = None
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_q:
                    obs = env.reset()
                    frame = request_frame(env, tile_size=TILE_SIZE)
                elif event.key in KEY_TO_ACTION:
                    action = KEY_TO_ACTION[event.key]
                elif event.key == pygame.K_p:
                    paused = not paused
                    print("Paused:", paused)

        if paused:
            clock.tick(FPS)
            continue

        if action is None:
            # just re-render current frame
            frame = request_frame(env, tile_size=TILE_SIZE)
        else:
            # step via FromGym: expects dict with 'action' and 'reset'
            result = env.step({'action': int(action), 'reset': False})
            # print small info
            rew = float(result.get('reward', 0.0))
            term = bool(result.get('is_last', False))
            print(f"Acción={action} reward={rew:.3f} done={term}")
            frame = request_frame(env, tile_size=TILE_SIZE)
            if term:
                print("Episodio terminado — reseteando")
                obs = env.reset()
                frame = request_frame(env, tile_size=TILE_SIZE)

        if frame is not None:
            surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
            window.blit(surf, (0, 0))
            pygame.display.flip()
        clock.tick(FPS)

    # cleanup
    env.close()
    pygame.quit()


if __name__ == "__main__":
    main()
