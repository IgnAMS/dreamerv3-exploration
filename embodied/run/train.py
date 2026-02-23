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

class LatentHERCallback:
    def __init__(self, replay, space, reward_fn, k=4, strategy='future'):
        """
        :param replay: Instancia de replay buffer de Dreamer.
        :param space: Claves (keys) permitidas del observation space.
        :param reward_fn: Función r(estado, meta) -> float
        :param k: Número de episodios relabelados a generar por cada original.
        :param strategy: Estrategia de muestreo de metas ('future', 'final', 'episode').
        """
        self.replay = replay
        self.space = space
        self.reward_fn = reward_fn
        self.k = k
        self.strategy = strategy
        
        # Diccionario para guardar el episodio en curso por cada worker paralelo
        self.episodes = collections.defaultdict(list)

    def __call__(self, tran, worker):
        self.episodes[worker].append(tran)
        filtered = {k: v for k, v in tran.items() if k in self.space}
        self.replay.add(filtered)

        # Si el episodio terminó, generamos los episodios HER
        if tran['is_last']:
            episode = self.episodes[worker]
            self.episodes[worker] = []  # Reseteamos el buffer para ese worker
            self._generate_her_episodes(episode)

    def _generate_her_episodes(self, episode):
        ep_len = len(episode)

        for _ in range(self.k):
            for t, tran in enumerate(episode):
                new_tran = tran.copy()
                if self.strategy == 'future':
                    goal_idx = np.random.randint(t, ep_len)
                elif self.strategy == 'final':
                    goal_idx = ep_len - 1
                else:
                    goal_idx = np.random.randint(0, ep_len)
                    
                goal_stoch = episode[goal_idx]['stoch']
                new_tran['her_goal'] = goal_stoch

                # Recalcular la recompensa
                new_tran['reward'] = self.reward_fn(new_tran['stoch'], goal_stoch)

                filtered = {k: v for k, v in new_tran.items() if k in self.space}
                self.replay.add(filtered)



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

  fns = [bind(make_env, i) for i in range(args.envs)]
  driver = embodied.Driver(fns, parallel=not args.debug)
  driver.on_step(lambda tran, _: step.increment())
  driver.on_step(lambda tran, _: policy_fps.step())
  if args.use_HER:
    reward_fn = lambda goal, stoch: 1.0 if np.linalg.norm(goal - stoch) < 0.2 else 0.0
    her_callback = LatentHERCallback(replay, space=agent.spaces.keys(), reward_fn=reward_fn)
    driver.on_step(her_callback)
  else:
    driver.on_step(lambda tran, _: filtered_replay(replay, agent.spaces.keys(), tran))
    
    
  driver.on_step(logfn)
  driver.on_step(lambda tran, worker: heatmap.increase(tran, worker))

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
