import yaml
import embodied
from dreamerv3.agent import Agent
from dreamerv3.main import make_env
import elements
from elements import Config, Flags
from PIL import Image
import jax
import jax.numpy as jnp
import numpy as np

print("UNO\n\n")

LOGDIR = "/home/iamonardes/logdir/dreamer/cookiepedrodeterministic18x29/size12m/05"
CKPT = f"{LOGDIR}/ckpt/agent/AAAA"
CONFIG = f"{LOGDIR}/config.yaml"

print("DOS\n\n")
"""
with open(CONFIG, "r") as f:
    saved = yaml.safe_load(f)
argv = []
def dfs(node, prefix=[]):
    if not isinstance(node, dict):
        if isinstance(node, list):
            value = str(node).strip("[").strip("]")
            value = "".join(value.split(" "))
            argv.append("--" + ".".join(prefix) + "=" + value)
        else:
            argv.append("--" + ".".join(prefix) + "=" + str(node))
        return 
    
    for k, v in node.items():
        prefix.append(k)
        dfs(v, prefix)
        prefix.pop()

dfs(saved, [])

configs = yaml.safe_load(open("dreamerv3/configs.yaml"))
parsed, other = elements.Flags(configs=['defaults']).parse_known(argv)
config = elements.Config(configs['defaults'])
for name in parsed.configs:
    config = config.update(configs[name])
config = elements.Flags(config).parse(other)
"""

config = Config.load(CONFIG)
config = elements.Flags(config).parse()

print("TRES\n\n")

task = config.task.split("_")[-1]
env = make_env(config)


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