from datetime import datetime
from os import makedirs

import gym
import numpy as np
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

np.random.seed(0)

timestamp = datetime.now().strftime(f'%Y.%b.%d %X ')
makedirs(timestamp)
env = gym.make('lti_gym:simple_grid-v0')

with open(f'{timestamp}/env.txt', 'w') as f:
    print(str(env), file=f)
env = Monitor(env)

model = PPO('MlpPolicy', env, verbose=1, tensorboard_log=f'{timestamp}/')
model.learn(total_timesteps=5000000)
model.save(f'{timestamp}/model')

env.tensorboard_log = f'{timestamp}/'
returns = []
for i in range(100):
    obs = env.reset()
    for _ in range(env.steps_max):
        env.render()
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
    returns += [env.cum_return]
    env.close()
print(pd.Series(returns).describe())
