import yaml
from embodied.envs.new_minigrid import CookiePedro, DeterministicCookie
import embodied
from dreamerv3.agent import Agent
import elements
from PIL import Image
import jax
import jax.numpy as jnp
import numpy as np

print("UNO\n\n")

LOGDIR = "/home/iamonardes/logdir/dreamer/cookiepedrodeterministic18x29/size12m/05"
CKPT = f"{LOGDIR}/ckpt/agent"
CONFIG = f"{LOGDIR}/config.yaml"

print("DOS\n\n")

with open(CONFIG, "r") as f:
    cfg_dict = yaml.safe_load(f)
parsed, other = elements.Flags(configs=['defaults']).parse_known()
config = elements.Config(cfg_dict)
for name in parsed.configs:
    config = config.update(cfg_dict[name])
config = elements.Flags(config).parse(other)
print("TRES\n\n")

task = config.task.split("_")[-1]
env = DeterministicCookie(task=task)

agent = Agent(
    env.obs_space,
    env.act_space,
    elements.Config(
      **config.agent,
      logdir=config.logdir,
      seed=config.seed,
      jax=config.jax,
      batch_size=config.batch_size,
      batch_length=config.batch_length,
      replay_context=config.replay_context,
      report_length=config.report_length,
      replica=config.replica,
      replicas=config.replicas,
  )
)

print("CUATRO\n\n")

cp = elements.Checkpoint(CKPT)
cp.agent = agent
cp.load()


print("CUATRO\n\n")

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

print("SEIS\n\n")

Image.fromarray(image).save("original.png")
Image.fromarray(recon.astype(np.uint8)).save("reconstruction.png")

print("Saved original.png and reconstruction.png")

# python3 -m dreamerv3.seeing_with_dreamer