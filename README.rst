=================
Safety Guarded RL
=================

Preliminary experiments to evaluate the convergence and safety of RL algorithms when the policy update is guarded by an safety oracle.


Structure
---------

The implementation is basically split into two parts:

* lti_gym: Learning Environment
* guarded_learner: Safe learner based on stable baselines RL agents

The remainder is structured as follows:

* examples: simple examples how to use it
* experiments: all the experiments and evaluation


Usage
-----

.. code-block:: python

    import gym

    if __name__ == '__main__':
        env = gym.make('lti_gym:simple_grid-v0')

        env.reset()
        for _ in range(1000):
            env.render()
            env.step(env.action_space.sample())  # pick three continous control actions randomly
        env.close()