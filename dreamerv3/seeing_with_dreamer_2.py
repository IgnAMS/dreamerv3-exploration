# seeing_with_driver.py
from dreamerv3.agent import sample
from dreamerv3.main import make_agent, make_env
import elements
from elements import Config
from PIL import Image
import jax
import jax.numpy as jnp
import numpy as np
from functools import partial as bind
import time
import uuid
import os
import ninjax as nj

# Driver import (embodied)
from embodied.core.driver import Driver   # ajusta la import si tu repo lo organiza distinto

# Opcional: si usas filtered_replay desde tu código
# from dreamerv3.replay_utils import filtered_replay  # <- ajusta import real si existe

print("UNO\n\n")

LOGDIR = "/home/iamonardes/logdir/dreamer/cookiepedrodeterministic18x29/size12m/05"
CKPT = f"{LOGDIR}/ckpt"
CONFIG = f"{LOGDIR}/config.yaml"

print("DOS\n\n")

config = Config.load(CONFIG)
config = elements.Flags(config).parse()
config.update({"jax": {"platform": "cpu"}})

print("TRES\n\n")

task = config.task.split("_")[-1]
agent = make_agent(config)

print(type(agent))
print(dir(agent))

print("CUATRO\n\n")

cp = elements.Checkpoint(CKPT)
cp.agent = agent
cp.load()
print("CINCO")

def describe(x):
    return {
        'type': type(x).__name__,
        'shape': getattr(x, 'shape', None),
        'dtype': getattr(x, 'dtype', None),
        'device': getattr(x, 'device', None),
    }

@nj.pure
def sample_prior(agent, deter):
    logit = agent.model.dyn._prior(deter)
    z_hat = agent.model.dyn._dist(logit).sample(seed=nj.seed())
    return z_hat, logit

def reconstruct_from_prior(agent, driver, image_np, reset):
    """
    Toma una imagen numpy H,W,3 (uint8) y devuelve la reconstrucción
    a partir del prior (hat{z}) como numpy uint8 H,W,3.
    No usa @jax.jit para evitar problemas host->device.
    """
    # Validaciones / normalizaciones
    img = np.asarray(image_np)
    assert img.ndim == 3 and img.shape[-1] == 3, f"Esperaba (H,W,3), recibí {img.shape}"
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)
    
    # 1) muestrear PRIOR via imagine (policy dummy)
    policyfn = lambda feat: sample(agent.model.pol(agent.model.feat2tensor(feat), 1))
    print("carry:", jax.tree.map(describe, driver.carry))
    dyn_carry = driver.carry[1]
    h_t = dyn_carry['deter'][0]
    carry_prior, (feat_prior, action) = sample_prior(agent, h_t)

    # 4) decodificar desde feat_prior
    dec_carry = agent.model.dec.initial(1)
    dec_carry, _, recons = agent.model.dec(dec_carry, feat_prior, reset, training=False, single=True)

    # Extra: en tu setup recons['image'] puede ser un objeto de salida
    if 'image' not in recons:
        print("BBB")
        
        # si el key no es 'image', intenta tomar la primera key disponible
        keys = list(recons.keys())
        if not keys:
            raise RuntimeError("Decoder no devolvió recons")
        k = keys[0]
        reconstructed = recons[k].mode()[0]
    else:
        print("AAA")
        reconstructed = recons['image'].mode()[0]

    # reconstructed está en [0,1] float -> escala a uint8
    recon_img = np.array((reconstructed * 255.0).clip(0, 255)).astype(np.uint8)
    return recon_img


def make_save_callback(agent, driver, out_dir="dreamer_prior_images"):
    os.makedirs(out_dir, exist_ok=True)

    def on_step(tran, _):
        """
        callback para driver.on_step
        tran: transición (dict) - puede contener arrays con batch dim o single
        _
        """
        # extraer la imagen del transition
        img = tran.get('image', None)
        reset = tran.get("reset", None)
        if img is None:
            return  # nada que hacer

        # si viene con batch dim (B,H,W,C) tomar primer elemento
        if img.ndim == 4:
            img_np = img[0]
        elif img.ndim == 3:
            img_np = img
        else:
            # intenta convertir object -> np
            img_np = np.asarray(img)
        
        # guarda original con timestamp/uuid para no sobreescribir
        uid = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
        orig_path = os.path.join(out_dir, f"orig_{uid}.png")
        Image.fromarray(img_np).save(orig_path)

        # reconstrucción desde prior
        try: 
            recon_img = reconstruct_from_prior(agent, driver, img_np, reset)
            recon_path = os.path.join(out_dir, f"prior_{uid}.png")
            Image.fromarray(recon_img).save(recon_path)
            print(f"Saved original -> {orig_path}; prior -> {recon_path}")
        except Exception as e:
            print("error en on step:", e)
            raise e
        # opcional: imprime ruta

    return on_step


if __name__ == "__main__":
    args = elements.Config(
        **config.run,
        replica=config.replica,
        replicas=config.replicas,
        logdir=config.logdir,
        batch_size=config.batch_size,
        batch_length=config.batch_length,
        report_length=config.report_length,
        consec_train=config.consec_train,
        consec_report=config.consec_report,
        replay_context=config.replay_context,
    )
    policy = lambda *args: agent.policy(*args, mode='train')
    TOTAL_STEPS = 300
    STEP_CHUNK = 10
    step = 0
    
    # crea el env y el driver
    print("Creando ambiente")
    fns = [bind(bind(make_env, config), 0)]
    driver = Driver(fns, parallel=not args.debug)

    # registra callbacks:
    save_cb = make_save_callback(agent, driver, out_dir="dreamer_prior_images")
    driver.on_step(save_cb)

    driver.reset(agent.init_policy)
    while step < TOTAL_STEPS:
        driver(policy, steps=STEP_CHUNK)
        step += STEP_CHUNK

    
    
# python3 -m dreamerv3.seeing_with_dreamer_2
