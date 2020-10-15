import gym
import numpy as np

from lti_gym.agents import LQR_Agent

if __name__ == '__main__':
    env = gym.make('lti_gym:simple_grid-v0')

    np.random.seed(0)
    agent = LQR_Agent(env.A, env.B)

    obs = env.reset()
    for _ in range(1000):
        env.render()
        obs, *__ = env.step(agent.act(obs))  # pick three continous control actions randomly
    print(env.cum_return)
    env.close()
