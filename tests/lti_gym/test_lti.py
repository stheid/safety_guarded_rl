import gym


def test_linear_time_invariant_env():
    env = gym.make('lti_gym:simple_grid-v0')

    env.reset()
    env.step(env.action_space.sample())  # pick three continous control actions randomly
    assert True
