"""
This file implement a base trainer class for both A2C and PPO trainers.

You should finish `evaluate_actions` and `compute_action`

-----
2019-2020 2nd term, IERG 6130: Reinforcement Learning and Beyond. Department
of Information Engineering, The Chinese University of Hong Kong. Course
Instructor: Professor ZHOU Bolei. Assignment author: PENG Zhenghao.
"""
import os

import gym
import numpy as np
import torch
from torch.distributions import Categorical
import torch.nn.functional as F

from .network import ActorCritic, MLP


class BaseTrainer:
    def __init__(self, env, config, frame_stack=4, _test=False):
        self.device = config.device
        self.config = config
        self.lr = config.LR
        self.num_envs = config.num_envs
        self.value_loss_weight = config.value_loss_weight
        self.entropy_loss_weight = config.entropy_loss_weight
        self.num_steps = config.num_steps
        self.grad_norm_max = config.grad_norm_max

        # if isinstance(env.observation_space, gym.spaces.Tuple):
        #     num_feats = env.observation_space[0].shape
        #     self.num_actions = env.action_space[0].n
        # else:
        #     num_feats = env.observation_space.shape
        #     self.num_actions = env.action_space.n

        num_feats = env.observation_space.shape
        self.num_actions = env.action_space

        self.num_feats = (1* frame_stack, *num_feats[0:])

        if _test:
            self.model = MLP(num_feats[0], self.num_actions)
        else:
            self.model = ActorCritic(self.num_feats, self.num_actions)
        self.model = self.model.to(self.device)
        self.model.train()

        self.setup_optimizer()
        self.setup_rollouts()

    def setup_optimizer(self):
        raise NotImplementedError()

    def setup_rollouts(self):
        raise NotImplementedError()

    def compute_loss(self, rollouts):
        raise NotImplementedError()

    def update(self, rollout):
        raise NotImplementedError()

    def compute_action(self, obs, deterministic=False):
        if isinstance(obs, np.ndarray):
            obs = torch.from_numpy(obs).to(self.device)

        if obs.dim() == 2:
            logits_cash, logits_beta, values = self.model(obs.view(1,-1))
        # else:
        #     logits, values = self.model(obs)

        # [TODO] Get the action and action's log probabilities based on the
        #  output logits
        # Hint:
        #   1. Use torch.distributions to help you build a distribution
        #   2. Remember to check the shape of action and log prob.
        #   3. When deterministic is True, return the action with maximum
        #    probability
        m = Categorical(logits=logits_cash)
        if deterministic:
            actions_cash = m.probs.argmax(dim=1, keepdim=True)           
        else:
            actions_cash = m.sample().view(-1, 1)

        log_probs_cash = F.log_softmax(logits_cash, dim=1)
        action_log_probs_cash = log_probs_cash.gather(1, actions_cash)

        m = Categorical(logits=logits_beta)
        if deterministic:
            actions_beta = m.probs.argmax(dim=1, keepdim=True)           
        else:
            actions_beta = m.sample().view(-1, 1)

        log_probs_beta = F.log_softmax(logits_beta, dim=1)
        action_log_probs_beta = log_probs_beta.gather(1, actions_beta)

        return values.view(-1, 1), actions_cash.view(-1, 1), action_log_probs_cash.view(
            -1, 1), actions_beta.view(-1, 1), action_log_probs_beta.view(-1, 1)

    def evaluate_actions(self, obs, act_cash, act_beta):
        """Run models to get the values, log probability and action
        distribution entropy of the action in current state"""
        if obs.dim() == 3:
            logits_cash, logits_beta, values = self.model(obs.view(obs.size(0), -1))
        # [TODO] Get the log probability of specified action, and the entropy of
        #  current distribution w.r.t. the output logits.
        # Hint: Use proper distribution to help you
        m = Categorical(logits=logits_cash)
        log_probs_cash = F.log_softmax(logits_cash, dim=1)
        action_log_probs_cash = log_probs_cash.gather(1, act_cash)
        dist_entropy_cash = m.entropy().mean()

        m = Categorical(logits=logits_beta)
        log_probs_beta = F.log_softmax(logits_beta, dim=1)
        action_log_probs_beta = log_probs_beta.gather(1, act_beta)
        dist_entropy_beta = m.entropy().mean()        

        assert dist_entropy_cash.shape == ()
        assert dist_entropy_beta.shape == ()
        return values.view(-1, 1), action_log_probs_cash.view(-1, 1), action_log_probs_beta.view(-1, 1), dist_entropy_cash+dist_entropy_beta

    def compute_values(self, obs):
        """Compute the values corresponding to current policy at current
        state"""
        if obs.dim() == 3:
            _, _, values = self.model(obs.view(obs.size(0), -1))
        return values

    def save_w(self, log_dir="", suffix=""):
        os.makedirs(log_dir, exist_ok=True)
        save_path = os.path.join(log_dir, "checkpoint-{}.pkl".format(suffix))
        torch.save(dict(
            model=self.model.state_dict(),
            optimizer=self.optimizer.state_dict()
        ), save_path)
        return save_path

    def load_w(self, log_dir="", suffix=""):
        save_path = os.path.join(log_dir, "checkpoint-{}.pkl".format(suffix))
        if os.path.isfile(save_path):
            state_dict = torch.load(
                save_path,
                torch.device('cpu') if not torch.cuda.is_available() else None
            )
            self.model.load_state_dict(state_dict["model"])
            self.optimizer.load_state_dict(state_dict["optimizer"])
