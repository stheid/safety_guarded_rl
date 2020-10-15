"""Top-level package for Safety Guarded RL."""

__author__ = """Stefan Heid"""
__email__ = 'stefan.heid@upb.de'

import numpy as np
from gym import register

resistance = 2
capacitance = 4
inductance = 3

register(
    id='simple_grid-v0',
    entry_point='lti_gym.envs:LinearTimeInvariantEnv',
    kwargs=dict(A=[[0, -1 / capacitance], [1 / inductance, -resistance / inductance]],
                B=np.array([[1 / capacitance, 0]]).T)
)
