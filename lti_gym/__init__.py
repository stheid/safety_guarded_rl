"""Top-level package for Safety Guarded RL."""

__author__ = """Stefan Heid"""
__email__ = 'stefan.heid@upb.de'

import numpy as np
from gym import register

capacitance = 1e-5  # micro Farad
inductance = 1e-3  # milli Henry
resistance = 5e-3  # milli Ohm

dt = 1e-5
steps_max = 1e3

state_range = [-np.full((2, 1), 1), np.full((2, 1), 1)]

register(
    id='simple_grid-v0',
    entry_point='lti_gym.envs:LinearTimeInvariantEnv',
    kwargs=dict(A=[[0, -1 / capacitance], [1 / inductance, -resistance / inductance]],
                B=np.array([[1 / capacitance, 0]]).T, dt=dt,
                state_range=state_range, steps_max=steps_max)
)
