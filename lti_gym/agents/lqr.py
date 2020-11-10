import numpy as np
from scipy.linalg import solve_discrete_are

from lti_gym.agents.base import Agent


class LQR_Agent(Agent):
    def __init__(self, A, B, Q=None, R=None):
        # https://en.wikipedia.org/wiki/Linear%E2%80%93quadratic_regulator#Infinite-horizon,_discrete-time_LQR

        A = np.array(A)
        B = np.array(B)

        # quadratic error matrices
        if Q is None:
            Q = np.eye(A.shape[0])
        if R is None:
            R = np.zeros((B.shape[1], B.shape[1]))

        P = solve_discrete_are(A, B, Q, R)
        self.policy = -np.linalg.inv(R + B.T @ P @ B) @ (B.T @ P @ A)

    def observe(self, reward, terminated):
        pass

    def act(self, obs):
        return self.policy @ obs
