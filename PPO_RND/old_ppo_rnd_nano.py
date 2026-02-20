import gymnasium as gym
from gymnasium.envs.registration import register
import cookie_env
from minigrid.wrappers import FullyObsWrapper, RGBImgPartialObsWrapper, RGBImgObsWrapper
from gymnasium import spaces
import os
from datetime import datetime
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.distributions.kl import kl_divergence
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from collections import deque

import matplotlib.pyplot as plt
import numpy as np
import sys
import numpy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  
dataType = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class RunningMeanStd:
    def __init__(self, shape):
        self.mean = torch.zeros(shape)
        self.var = torch.ones(shape)
        self.count = 1e-4

    def update(self, x):
        batch_mean = x.mean(0)
        batch_var = x.var(0, unbiased=False)
        batch_count = x.shape[0]

        delta = batch_mean - self.mean
        tot = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta**2 * self.count * batch_count / tot

        self.mean = new_mean
        self.var = M2 / tot
        self.count = tot
    
class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()
        C, H, W = obs_shape
        self.net = nn.Sequential(
            nn.Conv2d(C, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU()
        )

        with torch.no_grad():
            dummy = torch.zeros(1, C, H, W)
            self.out_dim = self.net(dummy).view(1, -1).size(1)

    def forward(self, x):
        x = x / 255.0
        return self.net(x).view(x.size(0), -1)

class ActorCritic(nn.Module):
    def __init__(self, obs_shape, n_actions, hidden=512, gru_hidden=256):
        super().__init__()

        C, H, W = obs_shape

        self.encoder = Encoder(C)
        self.fc = nn.Linear(self.encoder.out_dim, hidden)
        self.gru = nn.GRU(hidden, gru_hidden, batch_first=True)
        self.actor = nn.Linear(gru_hidden, n_actions)
        self.critic = nn.Linear(gru_hidden, 1)
        self.gru_hidden = gru_hidden

    def initial_state(self, batch):
        return torch.zeros(1, batch, self.gru_hidden)

    def forward(self, obs, h):
        # obs: [B,T,C,H,W]
        B,T,C,H,W = obs.shape
        obs = obs.reshape(B*T, C, H, W)

        z = self.encoder(obs)
        z = F.relu(self.fc(z))
        z = z.view(B,T,-1)
        z, h = self.gru(z, h)
        logits = self.actor(z)
        values = self.critic(z).squeeze(-1)

        return logits, values, h
    
class RND(nn.Module):
    def __init__(self, obs_shape, latent=512):
        super().__init__()

        C,H,W = obs_shape

        def make_net():
            return nn.Sequential(
                nn.Conv2d(C,32,8,4),
                nn.ReLU(),
                nn.Conv2d(32,64,4,2),
                nn.ReLU(),
                nn.Conv2d(64,64,3,1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(3136, latent),
                nn.ReLU(),
            )

        self.target = make_net()
        self.predictor = make_net()

        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, obs):
        with torch.no_grad():
            t = self.target(obs)
        p = self.predictor(obs)
        err = (p - t).pow(2).mean(dim=1)
        return err, p, t

class RNDWrapper:
    def __init__(self, rnd, beta=0.01):
        self.rnd = rnd
        self.beta = beta

    def intrinsic_reward(self, obs):
        r,_,_ = self.rnd(obs)
        return r
    
class PPOAgent:
    def __init__(self, model, rnd, lr=3e-4):
        self.model = model
        self.rnd = rnd

        self.opt = torch.optim.Adam(model.parameters(), lr)
        self.rnd_opt = torch.optim.Adam(rnd.predictor.parameters(), lr)

    def update(self, batch):
        obs = batch["obs"]
        actions = batch["actions"]
        returns = batch["returns"]
        adv = batch["advantages"]
        h0 = batch["h0"]

        logits, values, _ = self.model(obs, h0)
        dist = torch.distributions.Categorical(logits=logits)
        logp = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        ratio = torch.exp(logp - batch["old_logp"])

        clip = torch.clamp(ratio,0.8,1.2)*adv
        policy_loss = -torch.min(ratio*adv, clip).mean()
        value_loss = (returns - values).pow(2).mean()

        loss = policy_loss + 0.5*value_loss - 0.01*entropy

        self.opt.zero_grad()
        loss.backward()
        self.opt.step()

        # RND update
        r, p, t = self.rnd(obs[:,1:].reshape(-1,*obs.shape[2:]))

        rnd_loss = (p-t.detach()).pow(2).mean()

        self.rnd_opt.zero_grad()
        rnd_loss.backward()
        self.rnd_opt.step()



