from contextlib import redirect_stdout, redirect_stderr
from os import devnull
from tempfile import NamedTemporaryFile
from typing import Tuple

import gym
import numpy as np
import pandas as pd
import torch as th
from imitation.algorithms.bc import BC
from imitation.data import rollout
from imitation.data.types import Trajectory
from joblib import Parallel, delayed
from sacred import Experiment
from sacred.observers import FileStorageObserver
from stable_baselines3 import DDPG
from stable_baselines3.td3.policies import TD3Policy

from lti_gym.agents import LQR_Agent

ex = Experiment('BC_imitation')
ex.observers.append(FileStorageObserver('runs'))


def policy_factory(base, model=None):
    if model:
        class CopyPolicy(base):
            def __new__(*args, **kwargs):
                return model
    else:
        class CopyPolicy(base):
            pass
    return CopyPolicy


class TD3ACPolicy(TD3Policy):
    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        latent_pi = latent_sde = self.actor(obs)
        # the below is only a copy from the ActorCriticPolicy class.
        # The function is only returning dummy values because its not clear how the proper implementation would look like
        # distribution = self._get_action_dist_from_latent(latent_pi, latent_sde)
        # log_prob = distribution.log_prob(actions)
        return None, th.tensor([0], dtype=th.float32), th.tensor([0], dtype=th.float32)


@ex.config
def cfg():
    eval_eps = 100
    eps_steps = 1000
    bc_expert_eps = 100
    bc_train_eps = 100
    train_steps = 10 ** 5
    rl_classes = dict(learner=DDPG, policy=TD3ACPolicy)


@ex.automain
def main(eval_eps, eps_steps, bc_expert_eps, bc_train_eps, train_steps, rl_classes):
    def evaluate(env, agent, squeezed=False, bc_expert_eps=0):
        def exec_eps(i):
            np.random.seed(i)
            obss, acts, infos = [], [], []
            obss.append(env.reset().squeeze() if squeezed else env.reset())
            for _ in range(env.steps_max):
                env.render()
                acts.append(agent(obss[-1]))
                obs, rew, done, info = env.step(acts[-1])
                obss.append(obs.squeeze() if squeezed else obs)
                infos.append(info)
                if done:
                    break
            env.close()
            return env.cum_return, Trajectory(np.array(obss), np.array(acts), np.array(infos))

        results = Parallel(n_jobs=12)(delayed(exec_eps)(i) for i in range(max(eval_eps, bc_expert_eps)))
        returns = (list(zip(*results[:eval_eps])) + [[], []])[0]
        trajs = (list(zip(*results[:bc_expert_eps])) + [[], []])[1]

        return trajs, pd.Series(returns).describe()

    # setup
    env = gym.make('lti_gym:simple_grid-v0', steps_max=eps_steps)
    results = []

    # record
    agent = LQR_Agent(env.A, env.B)
    trajs, r = evaluate(env, agent.act, True, bc_expert_eps)
    results.append(r)

    if trajs:
        # clone behaviour
        transitions = rollout.flatten_trajectories(trajs)
        with redirect_stdout(open(devnull, 'w')) and redirect_stderr(open(devnull, 'w')):
            bc_trainer = BC(env.observation_space, env.action_space, expert_data=transitions,
                            policy_class=rl_classes['policy'])
            bc_trainer.train(n_epochs=bc_train_eps)

        _, r = evaluate(env, lambda obs: bc_trainer.policy.predict(obs, deterministic=True)[0])
        results.append(r)

        # copy policy
        PolicyClass = policy_factory(rl_classes['policy'], bc_trainer.policy)
    else:
        PolicyClass = policy_factory(rl_classes['policy'])

    # continue training
    with redirect_stdout(open(devnull, 'w')) and redirect_stderr(open(devnull, 'w')):
        model = rl_classes['learner'](PolicyClass, env)
        with NamedTemporaryFile() as t:
            model.save(t.name)
            ex.add_artifact(t.name, 'bc_model.zip')
        model.learn(train_steps)
        with NamedTemporaryFile() as t:
            model.save(t.name)
            ex.add_artifact(t.name, 'final_model.zip')

    _, r = evaluate(env, lambda obs: model.predict(obs, deterministic=True)[0])
    results.append(r)
    results = pd.DataFrame(results, index=['lqr', 'bc', 'final'] if len(results) == 3 else [['lqr', 'final']])

    with NamedTemporaryFile() as t:
        results.to_csv(t.name)
        ex.add_artifact(t.name, 'results.csv')
    if 'bc' in results.columns:
        ex.log_scalar('mean_return_bc', results.at['bc', 'mean'])
    ex.log_scalar('mean_return_final', results.at['final', 'mean'])
    return '\n' + results.to_string()
