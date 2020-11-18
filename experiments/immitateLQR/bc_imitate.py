from contextlib import redirect_stdout, redirect_stderr
from os import devnull
from tempfile import NamedTemporaryFile

import gym
import numpy as np
import pandas as pd
import stable_baselines3 as sb3
from imitation.algorithms.bc import BC
from imitation.data import rollout
from imitation.data.types import Trajectory
from joblib import Parallel, delayed
from sacred import Experiment
from sacred.observers import FileStorageObserver
from stable_baselines3.common.policies import ActorCriticPolicy

from lti_gym.agents import LQR_Agent

ex = Experiment('BC_imitation')
ex.observers.append(FileStorageObserver('runs'))


@ex.config
def cfg():
    eval_eps = 100
    eps_steps = 1000
    bc_expert_eps = 0
    bc_train_eps = 1
    train_steps = 1000000


@ex.automain
def main(eval_eps, eps_steps, bc_expert_eps, bc_train_eps, train_steps):
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
                            policy_class=ActorCriticPolicy)
            bc_trainer.train(n_epochs=bc_train_eps)

        _, r = evaluate(env, lambda obs: bc_trainer.policy.predict(obs, deterministic=True)[0])
        results.append(r)

        # copy policy
        class CopyPolicy(ActorCriticPolicy):
            def __new__(cls, *args, **kwargs):
                return bc_trainer.policy
    else:
        class CopyPolicy(ActorCriticPolicy):
            pass

    # continue training
    with redirect_stdout(open(devnull, 'w')) and redirect_stderr(open(devnull, 'w')):
        model = sb3.PPO(CopyPolicy, env)
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
