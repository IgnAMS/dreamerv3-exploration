from ICM_model import ForwardModel, InverseModel, ICM, RND, random_observations, CuriosityModule
from ICM_envs import make_vec_envs
from a2c_ppo_acktr.algo.ppo import PPO
from a2c_ppo_acktr.model import Policy
import gym
import numpy as np
import torch

from PPO_CONFIG import *

num_env_steps = 500000
num_updates = int(num_env_steps) // num_steps // num_processes
lr_decay_horizon = int(10e6) // num_steps // num_processes
log_interval = 10
eval_interval = 10

env_name = "corner_55"
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
envs = make_vec_envs(
    env_name, 1, num_processes, gamma,
    device, False, normalize_obs=False
)
n_launches = 5
name = "ICM"
cur_models = {
    # 'ForwardDynLoss': [ForwardModel, 10.],
    # 'InverseDynLoss': [InverseModel, 0.5],
    # 'RND': [RND, 10.],
    'ICM': [ICM, 10.]
}
for i in range(n_launches):
    if name == 'RND':
        cur_model = cur_models[name][0](
            envs.observation_space.shape[0], num_processes)
        cur_model.to(device)
        cur_model.init_obs_norm(random_observations(
            env_name, size=2000, device=device))
    else:
        cur_model = cur_models[name][0](
            envs.observation_space.shape[0], envs.action_space)
        cur_model.to(device)
        curiosity_module = CuriosityModule(
            cur_model, rew_coef=cur_models[name][1])
    print('Environment: {}, method: {}, {}'.format(env_name, name, i))
    actor_critic = Policy(envs.observation_space.shape, envs.action_space,
                            base_kwargs={'recurrent': False}).to(device)

    agent = PPO(actor_critic,
                clip_param,
                ppo_epochs,
                num_mini_batch,
                value_loss_coef,
                entropy_coef,
                lr,
                eps,
                max_grad_norm)
    stats = train_loop(agent, envs, env_name, num_updates, num_steps, curiosity_module=curiosity_module,
                        save_interval=save_interval, eval_interval=eval_interval, log_interval=log_interval,
                        time_limit=time_limit, curiosity_rew_after=0, curiosity_rew_before=None,
                        use_linear_lr_decay=True, lr_decay_horizon=lr_decay_horizon,
                        callbacks=None)
    with open(os.path.join(path, env_name, name, str(i)), 'wb') as f:
        pickle.dump(stats, f)



