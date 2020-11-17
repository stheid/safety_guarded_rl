from contextlib import redirect_stdout, redirect_stderr
from datetime import datetime
from os import makedirs, devnull

import gym
import numpy as np
import pandas as pd
import stable_baselines3 as sb3
from imitation.algorithms.bc import BC
from imitation.data import rollout
from imitation.data.types import Trajectory
from stable_baselines3.common.policies import ActorCriticPolicy

from lti_gym.agents import LQR_Agent


def evaluate(agent, squeezed=False):
    np.random.seed(0)
    returns = []
    trajs = []
    for i in range(100):
        obss, acts, infos = [], [], []
        obss.append(env.reset().squeeze() if squeezed else env.reset())
        for _ in range(env.steps_max):
            env.render()
            acts.append(np.array(agent(obss[-1])))
            obs, rew, done, info = env.step(acts[-1])
            obss.append(obs.squeeze() if squeezed else obs)
            infos.append(info)
            if done:
                break
        returns += [env.cum_return]
        env.close()
        trajs.append(Trajectory(np.array(obss), np.array(acts), np.array(infos)))
    print(pd.Series(returns).describe())
    return trajs


# setup
timestamp = datetime.now().strftime(f'%Y.%b.%d %X ')
makedirs(timestamp, exist_ok=True)
env = gym.make('lti_gym:simple_grid-v0')

# record
agent = LQR_Agent(env.A, env.B)
trajs = evaluate(agent.act)
transitions = rollout.flatten_trajectories(trajs[0:1])

# clone
with redirect_stdout(open(devnull, 'w')) and redirect_stderr(open(devnull, 'w')):
    bc_trainer = BC(env.observation_space, env.action_space, expert_data=transitions, policy_class=ActorCriticPolicy)
    bc_trainer.train(n_epochs=1)

evaluate(lambda obs: bc_trainer.policy.predict(obs, deterministic=True)[0])


# copy policy
class CopyPolicy(ActorCriticPolicy):
    def __new__(cls, *args, **kwargs):
        return bc_trainer.policy


# continue training
with redirect_stdout(open(devnull, 'w')) and redirect_stderr(open(devnull, 'w')):
    model = sb3.PPO(CopyPolicy, env)
    model.learn(1000)
    model.save(f"{timestamp} /BC/checkpoint")

evaluate(lambda obs: model.predict(obs, deterministic=True)[0])
