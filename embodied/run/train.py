import collections
from functools import partial as bind

import elements
import embodied
import numpy as np
import pickle 
from collections import defaultdict


class Heatmap:
  def __init__(self):
        self.heatmap = defaultdict(int)
  def increase(self, tran, env_index, **kwargs):
    if "agent_pos" in tran:
        x, y = map(int, tran["agent_pos"])
        # convert (x,y) -> (row,col)=(y,x) for indexing
        self.heatmap[(int(y), int(x))] += 1

def filtered_replay(replay, space, tran):
  filtered = {k: v for k, v in tran.items() if k in space}
  replay.add(filtered)


def _make_her_goal(achieved, stoch_rows, stoch_classes, rng):
  """
  Construye un goal double one-hot a partir de achieved_goal.
  achieved : (stoch_rows,) int32
  """
  row_idx   = int(rng.integers(stoch_rows))
  class_val = int(achieved[row_idx])
  row_oh  = np.zeros(stoch_rows,    dtype=np.float32)
  cls_oh  = np.zeros(stoch_classes, dtype=np.float32)
  row_oh[row_idx]   = 1.0
  cls_oh[class_val] = 1.0
  return np.concatenate([row_oh, cls_oh]), row_idx, class_val
 
 
def _her_reached(achieved, row_idx, class_val):
  return int(achieved[row_idx]) == class_val




def train(make_agent, make_replay, make_env, make_stream, make_logger, args):

  agent = make_agent()
  replay = make_replay()
  logger = make_logger()

  logdir = elements.Path(args.logdir)
  step = logger.step
  usage = elements.Usage(**args.usage)
  train_agg = elements.Agg()
  epstats = elements.Agg()
  episodes = collections.defaultdict(elements.Agg)
  policy_fps = elements.FPS()
  train_fps = elements.FPS()
  
  heatmap = Heatmap()
  
  batch_steps = args.batch_size * args.batch_length
  should_train = elements.when.Ratio(args.train_ratio / batch_steps)
  should_log = embodied.LocalClock(args.log_every)
  should_report = embodied.LocalClock(args.report_every)
  should_save = embodied.LocalClock(args.save_every)

  her_enabled = getattr(args.her, 'enabled', False)
  if her_enabled:
    her_k           = args.her.k
    her_strategy    = args.her.strategy
    her_stoch_rows  = args.dyn_stoch
    her_stoch_classes = args.dyn_classes
    her_rng         = np.random.default_rng()
    episode_buffers = defaultdict(list)
    
    
  @elements.timer.section('logfn')
  def logfn(tran, worker):
    episode = episodes[worker]
    tran['is_first'] and episode.reset()
    episode.add('score', tran['reward'], agg='sum')
    episode.add('length', 1, agg='sum')
    episode.add('rewards', tran['reward'], agg='stack')
    for key, value in tran.items():
      if not isinstance(value, np.ndarray):
        continue
      if value.dtype == np.uint8 and value.ndim == 3:
        if worker == 0:
          episode.add(f'policy_{key}', value, agg='stack')
      elif key.startswith('log/'):
        assert value.ndim == 0, (key, value.shape, value.dtype)
        episode.add(key + '/avg', value, agg='avg')
        episode.add(key + '/max', value, agg='max')
        episode.add(key + '/sum', value, agg='sum')
    if tran['is_last']:
      result = episode.result()
      logger.add({
          'score': result.pop('score'),
          'length': result.pop('length'),
      }, prefix='episode')
      rew = result.pop('rewards')
      if len(rew) > 1:
        result['reward_rate'] = (np.abs(rew[1:] - rew[:-1]) >= 0.01).mean()
      epstats.add(result)

  def her_replay_fn(tran, worker):
    """
    Acumula transiciones por episodio. Al final del episodio genera k
    transiciones relabeladas por cada timestep y las añade al replay,
    exactamente como en Algorithm 1 del paper HER.
    """
    # Solo acumular si tenemos achieved_goal (requiere HER activo en agente)
    if 'achieved_goal' not in tran:
      print("No hubo transition!")
      return

    buf = episode_buffers[worker]

    if tran['is_first']:
      buf.clear()

    buf.append({k: v.copy() if isinstance(v, np.ndarray) else v
                for k, v in tran.items()})

    if not tran['is_last']:
      return

    # ── Fin del episodio: relabelar según paper ───────────────────────────
    T = len(buf)
    replay_space = set(agent.spaces.keys())

    for t in range(T):
      for _ in range(her_k):

        # Samplear t' según estrategia
        if her_strategy == 'future':
          t_prime = int(her_rng.integers(t, T))
        elif her_strategy == 'final':
          t_prime = T - 1
        elif her_strategy == 'random_ep':
          t_prime = int(her_rng.integers(T))
        else:
          raise ValueError(her_strategy)

        achieved = buf[t_prime]['achieved_goal']  # (stoch_rows,)

        # Construir goal double one-hot
        g_prime, row_idx, class_val = _make_her_goal(
            achieved, her_stoch_rows, her_stoch_classes, her_rng)

        # Relabelar reward: 1 si el estado t alcanzó el goal
        reached = _her_reached(buf[t]['achieved_goal'], row_idx, class_val)

        her_tran = dict(buf[t])
        her_tran['goal']   = g_prime
        her_tran['reward'] = np.float32(1.0 if reached else 0.0)

        filtered = {k: v for k, v in her_tran.items() if k in replay_space}
        replay.add(filtered)

    episode_buffers[worker].clear()

  fns = [bind(make_env, i) for i in range(args.envs)]
  driver = embodied.Driver(fns, parallel=not args.debug)
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())
  driver.on_step(lambda tran, _: filtered_replay(replay, agent.spaces.keys(), tran))
  driver.on_step(logfn)
  driver.on_step(lambda tran, worker: heatmap.increase(tran, worker))
  driver.on_step(her_replay_fn)

  stream_train = iter(agent.stream(make_stream(replay, 'train')))
  stream_report = iter(agent.stream(make_stream(replay, 'report')))

  carry_train = [agent.init_train(args.batch_size)]
  carry_report = agent.init_report(args.batch_size)

  def trainfn(tran, worker):
    if len(replay) < args.batch_size * args.batch_length:
      return
    for _ in range(should_train(step)):
      with elements.timer.section('stream_next'):
        batch = next(stream_train)
      carry_train[0], outs, mets = agent.train(carry_train[0], batch)
      train_fps.step(batch_steps)
      if 'replay' in outs:
        replay.update(outs['replay'])
      train_agg.add(mets, prefix='train')
  driver.on_step(trainfn)

  cp = elements.Checkpoint(logdir / 'ckpt')
  cp.step = step
  cp.agent = agent
  cp.replay = replay
  if args.from_checkpoint:
    elements.checkpoint.load(args.from_checkpoint, dict(
        agent=bind(agent.load, regex=args.from_checkpoint_regex)))
  cp.load_or_save()

  print('Start training loop')
  policy = lambda *args: agent.policy(*args, mode='train')
  driver.reset(agent.init_policy)
  while step < args.steps:

    driver(policy, steps=10)

    if should_report(step) and len(replay):
      agg = elements.Agg()
      for _ in range(args.consec_report * args.report_batches):
        carry_report, mets = agent.report(carry_report, next(stream_report))
        agg.add(mets)
      logger.add(agg.result(), prefix='report')

    if should_log(step):
      logger.add(train_agg.result())
      logger.add(epstats.result(), prefix='epstats')
      logger.add(replay.stats(), prefix='replay')
      logger.add(usage.stats(), prefix='usage')
      logger.add({'fps/policy': policy_fps.result()})
      logger.add({'fps/train': train_fps.result()})
      logger.add({'timer': elements.timer.stats()['summary']})
      logger.write()

    if should_save(step):
      cp.save()
      with open(logdir / f"heatmap_{int(step)}.pkl", "wb") as f:
        pickle.dump(dict(heatmap.heatmap), f)
        
  with open(logdir / f"heatmap_{int(step)}.pkl", "wb") as f:
        pickle.dump(dict(heatmap.heatmap), f)
  logger.close()
