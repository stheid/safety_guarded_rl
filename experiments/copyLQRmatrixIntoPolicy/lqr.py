from datetime import datetime
from os import makedirs

import gym
import numpy as np
import pandas as pd
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.save_util import load_from_zip_file, save_to_zip_file
from torch import nn

from lti_gym.agents import LQR_Agent

np.random.seed(0)

timestamp = datetime.now().strftime(f'%Y.%b.%d %X ')
makedirs(timestamp)
env = gym.make('lti_gym:simple_grid-v0')

agent = LQR_Agent(env.A, env.B)

returns = []
for i in range(100):
    obs = env.reset()
    for _ in range(env.steps_max):
        env.render()
        obs, rew, done, _ = env.step(agent.act(obs))  # pick three continous control actions randomly
        if done:
            break
    returns += [env.cum_return]
    env.close()
print(pd.Series(returns).describe())

with open(f'{timestamp}/env.txt', 'w') as f:
    print(str(env), file=f)
env = Monitor(env)

model = PPO('MlpPolicy', env, policy_kwargs=dict(activation_fn=nn.Identity, net_arch=[dict(pi=[1], vf=[64, 64])]))
model.save(f'{timestamp}/model')
data, params, tensors = load_from_zip_file(f'{timestamp}/model')
params['policy']['mlp_extractor.policy_net.0.weight'] = th.tensor(agent.policy, dtype=th.float32)
params['policy']['mlp_extractor.policy_net.0.bias'] = th.tensor([0], dtype=th.float32)
params['policy']['action_net.weight'] = th.tensor([[1]], dtype=th.float32)
params['policy']['action_net.bias'] = th.tensor([0], dtype=th.float32)
save_to_zip_file(f'{timestamp}/model', data, params, tensors)
model = model.load(f'{timestamp}/model')

np.random.seed(0)
returns = []
for i in range(100):
    obs = env.reset()
    for _ in range(env.steps_max):
        env.render()
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        if done:
            break
    returns += [env.cum_return]
    env.close()
print(pd.Series(returns).describe())
