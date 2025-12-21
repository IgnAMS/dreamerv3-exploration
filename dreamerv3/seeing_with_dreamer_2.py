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

print("CUATRO\n\n")

cp = elements.Checkpoint(CKPT)
cp.agent = agent
cp.load()
print("CINCO")


def reconstruct_from_prior(agent, image_np):
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

    # arma obs para el encoder (batch dim 1)
    obs = {'image': jnp.asarray(img, dtype=jnp.uint8)[None]}
    reset = jnp.array([False], dtype=jnp.bool_)   # use jnp.bool_ explícito

    # 1) encode (siguiendo la API del Encoder en tu arquitectura)
    enc_carry = agent.enc.initial(1)
    enc_carry, _, tokens = agent.enc(enc_carry, obs, reset, training=False, single=True)

    # 2) inicializar RSSM y observar (obtener carry/h_t)
    state = agent.dyn.initial(1)
    action_zeros = jnp.zeros((1, agent.act_space['action'].shape[0]))
    carry, entry, feat_post = agent.dyn.observe(
        state,
        tokens,
        action=action_zeros,
        reset=reset,
        training=False,
        single=True,
    )

    # 3) muestrear PRIOR via imagine (policy dummy)
    policyfn = lambda c: sample(agent.dyn.pol(agent.dyn.feat2tensor(c), 1))
    carry_prior, (feat_prior, action) = agent.dyn.imagine(
        carry,
        policy=policyfn,
        length=1,
        training=False,
        single=True,
    )

    # 4) decodificar desde feat_prior
    dec_carry = agent.dec.initial(1)
    dec_carry, _, recons = agent.dec(dec_carry, feat_prior, reset, training=False, single=True)

    # Extra: en tu setup recons['image'] puede ser un objeto de salida
    if 'image' not in recons:
        # si el key no es 'image', intenta tomar la primera key disponible
        keys = list(recons.keys())
        if not keys:
            raise RuntimeError("Decoder no devolvió recons")
        k = keys[0]
        reconstructed = recons[k].mode()[0]
    else:
        reconstructed = recons['image'].mode()[0]

    # reconstructed está en [0,1] float -> escala a uint8
    recon_img = np.array((reconstructed * 255.0).clip(0, 255)).astype(np.uint8)
    return recon_img


def make_save_callback(agent, out_dir="dreamer_prior_images"):
    os.makedirs(out_dir, exist_ok=True)

    def on_step(tran, _):
        """
        callback para driver.on_step
        tran: transición (dict) - puede contener arrays con batch dim o single
        _
        """
        try:
            # extraer la imagen del transition
            img = tran.get('image', None)
            if img is None:
                return  # nada que hacer

            # si viene con batch dim (B,H,W,C) tomar primer elemento
            if isinstance(img, np.ndarray):
                if img.ndim == 4:
                    img_np = img[0]
                elif img.ndim == 3:
                    img_np = img
                else:
                    # intenta convertir object -> np
                    img_np = np.asarray(img)
            else:
                # si viene como jax array, pasar a numpy
                try:
                    img_np = np.asarray(img)
                    if img_np.ndim == 4:
                        img_np = img_np[0]
                except Exception:
                    return

            # guarda original con timestamp/uuid para no sobreescribir
            uid = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
            orig_path = os.path.join(out_dir, f"orig_{uid}.png")
            Image.fromarray(img_np).save(orig_path)

            # reconstrucción desde prior
            recon_img = reconstruct_from_prior(agent, img_np)
            recon_path = os.path.join(out_dir, f"prior_{uid}.png")
            Image.fromarray(recon_img).save(recon_path)

            # opcional: imprime ruta
            print(f"Saved original -> {orig_path}; prior -> {recon_path}")

        except Exception as e:
            # no queremos que un error en la callback rompa el driver loop
            print("Error en on_step callback:", e)
            return

    return on_step


if __name__ == "__main__":
    # crea el env y el driver
    fns = [bind(bind(make_env, config), 0)]
    driver = Driver(fns, parallel=False)

    # registra callbacks:
    save_cb = make_save_callback(agent, out_dir="dreamer_prior_images")
    driver.on_step(save_cb)

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

    while step < TOTAL_STEPS:
        driver(policy, steps=STEP_CHUNK)
        step += STEP_CHUNK

    
    
# python3 -m dreamerv3.seeing_with_dreamer_2
