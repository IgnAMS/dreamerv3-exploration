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

try:
    print("CUATRO\n\n")
    env = make_env(config, 0)
    obs = env.reset()
    image = obs["image"][0]           # (H,W,3)
    image = image.astype(np.uint8)

    print("CINCO\n\n")


    @jax.jit
    def reconstruct(img):
        img = jnp.array(img)[None]
        # 1) encode
        embed = agent.enc(img)
        # 2) initial RSSM state
        state = agent.dyn.initial(batch_size=1)
        # 3) observe
        z, z_hat = agent.dyn.observe(
            state,
            embed,
            action=jnp.zeros((1, agent.act_space['action'].shape[0])),
            reset=jnp.zeros((1,), bool),
        )
        carry = state
        policyfn = lambda feat: sample(agent.dyn.pol(agent.dyn.feat2tensor(feat), 1))
        H = config.imag_length
        carry, (feat, action) = agent.dyn.imagine(carry, policy=policyfn, length=H, training=None, single=True)
        z_hat = carry["stoch"]
        
        # 4) features → decode
        # dyn.get_feat no existe! 
        # feat = agent.dyn.get_feat(z_hat)
        recon = agent.dec(feat).mode()
        return recon[0]

    recon = np.array(reconstruct(image))

    Image.fromarray(image).save("original.png")
    Image.fromarray(recon.astype(np.uint8)).save("reconstruction.png")
except Exception as e:
    env.close()
    raise e
else:
    env.close()
# python3 -m dreamerv3.seeing_with_dreamer