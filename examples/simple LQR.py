import gym
import numpy as np
import pandas as pd

from lti_gym.agents import LQR_Agent

if __name__ == '__main__':
    env = gym.make('lti_gym:simple_grid-v0', tensorboard_log='test')

    np.random.seed(0)
    agent = LQR_Agent(env.A, env.B)

    returns = []
    for i in range(1):
        obs = env.reset()
        for _ in range(int(1e3)):
            env.render()
            obs, rew, done, _ = env.step(agent.act(obs))  # pick three continous control actions randomly
            if done:
                break
        returns += [env.cum_return]
        env.close()
    print(pd.Series(returns).describe())
