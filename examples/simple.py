import gym

if __name__ == '__main__':
    env = gym.make('lti_gym:simple_grid-v0', tensorboard_log='test')

    env.reset()
    for _ in range(1000):
        env.render()
        env.step(env.action_space.sample())  # pick three continous control actions randomly
    env.close()
