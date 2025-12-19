import yaml
import embodied
from dreamerv3.agent import Agent
from dreamerv3.main import make_agent, make_env
import elements
from elements import Config, Flags
from PIL import Image
import jax
import jax.numpy as jnp
import numpy as np

print("UNO\n\n")

LOGDIR = "/home/iamonardes/logdir/dreamer/cookiepedrodeterministic18x29/size12m/05"
CKPT = f"{LOGDIR}/ckpt/agent/"
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

    wm = agent.world_model
    print("CINCO\n\n")


    @jax.jit
    def reconstruct(img):
        embed = wm.encoder(jnp.array(img)[None])
        state = wm.rssm.initial(1)
        state, _ = wm.rssm.observe(
            state, embed, jnp.zeros((1,)), jnp.zeros((1,), bool)
        )
        feat = wm.rssm.get_feat(state)
        recon = wm.heads["image"](feat).mode()
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