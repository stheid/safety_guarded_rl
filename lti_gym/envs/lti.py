import os
from datetime import datetime

import numpy as np
from gym import Env
from gym.spaces import Box
from torch.utils.tensorboard import SummaryWriter

"""
Simple LTI system
"""


class LinearTimeInvariantEnv(Env):
    def __init__(self, A, B, state_range=None, ref=None, tensorboard_log=None, steps_max=1000):
        self.A = np.array(A)
        self.B = np.array(B)
        self.dim = self.A.shape[0]
        self.state = None
        if ref is None:
            self.ref = np.zeros(self.dim)
        else:
            self.ref = ref

        if state_range is None:
            state_range = [-np.ones((self.dim, 1)), np.ones((self.dim, 1))]

        self.steps = None
        self.steps_max = steps_max
        self.cum_return = None
        self.writer = None
        if tensorboard_log:
            self.tensorboard_log = os.path.join(tensorboard_log, datetime.now().strftime('run %Y.%m.%d %X'))
        else:
            self.tensorboard_log = None
        self.action_space = Box(-np.ones(1), np.ones(1))
        self.observation_space = Box(*state_range)

    def reset(self):
        self.steps = 0
        self.cum_return = 0
        self.state = np.random.uniform(self.observation_space.low, self.observation_space.high)
        if self.tensorboard_log:
            self.writer = SummaryWriter(
                log_dir=os.path.join(self.tensorboard_log, datetime.now().strftime('epoch %Y.%m.%d %X')))
        return self.state

    def step(self, action: np.ndarray):
        self.steps += 1
        self.state += self.A @ self.state + self.B * action.T

        # validate state is within box constraints
        if not np.array_equal(self.state, np.clip(self.state, self.observation_space.low, self.observation_space.high)):
            return self.state, -1e3, True, dict(reason='left observation space constraint')

        # calculate reward
        mse = np.clip(np.power(self.state - self.ref, 2).sum(), 0, 1e5)
        reward = 1 - mse
        self.cum_return += reward

        return self.state, reward, self.steps > self.steps_max, {}

    def render(self, mode='human'):
        if self.writer:
            for i in range(self.dim):
                self.writer.add_scalar(f'state/x{i}', self.state[i], global_step=self.steps)

    def close(self):
        """
        closes the file
        """
        if self.writer:
            self.writer.close()
            self.writer = None
