from dreamerv3.agent import sample
from dreamerv3.main import make_agent, make_env
import elements
from elements import Config
from PIL import Image
import jax
import jax.numpy as jnp
import numpy as np

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


def reconstruct_and_save(agent, obs, out_original="original.png", out_recon="reconstruction.png"):
    """
    image_uint8: numpy array H,W,3 dtype uint8
    agent: agente ya cargado con pesos (como en tu script)
    Guarda original y reconstrucción por posterior (z_hat).
    """
     # 1) preparar inputs con batch dim
    img = np.asarray(obs["image"])
    reset = jnp.zeros((1,), dtype=jnp.bool_)

     # 2) encode -> tokens
    # encoder usually tiene signature: carry, entries, tokens = enc(carry, obs, reset, training, single=True)
    enc_carry = agent.enc.initial(1)   # según Encoder.initial en tu código esto devuelve {}
    enc_carry, _, tokens = agent.enc(enc_carry, obs, reset, training=False, single=True)
    # tokens : shape (1, token_dim)
    
    # 3) inicializar RSSM y observar
    state = agent.dyn.initial(1)
    action_zeros = jnp.zeros((1, agent.act_space['action'].shape[0]))
    carry, entry, feat = agent.dyn.observe(
        state,
        tokens,
        action=action_zeros,
        reset=reset,
        training=False,
        single=True,
    )
    
    # 4) Obtener el prior y el hat{z}
    # política dummy (acción cero)
    policyfn = lambda feat: sample(agent.dyn.pol(agent.dyn.feat2tensor(feat), 1))

    carry_prior, (feat_prior, action) = agent.dyn.imagine(
        carry,
        policy=policyfn,
        length=1,
        training=False,
        single=True,
    )
    
    # 4) decodificar desde el carry (prior)
    dec_carry = agent.dec.initial(1)
    dec_carry, _, recons = agent.dec(
        dec_carry,
        feat_prior,
        reset,
        training=False,
        single=True,
    )

    recon = recons['image'].mode()[0]      # [0,1]
    recon = (recon * 255).astype(np.uint8)
    
    # guardar
    Image.fromarray(img).save(out_original)
    Image.fromarray(recon).save(out_recon)


try:
    print("CUATRO\n\n")
    env = make_env(config, 0)
    obs = env.reset()
    print(obs["image"].shape)
    print("CINCO\n\n")
    reconstruct_and_save(agent, obs)
    

except Exception as e:
    env.close()
    raise e
else:
    env.close()
# python3 -m dreamerv3.seeing_with_dreamer