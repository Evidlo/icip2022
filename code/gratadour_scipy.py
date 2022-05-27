#!/bin/env python
## Evan Widloski - 2017-02-17
## ECE321 Mini Project

from platypus.algorithms import NSGAII
from platypus.core import Problem
from platypus.types import Real, Subset
import numpy as np
from scipy.optimize import minimize
from multiml.observation import test_sequence
from gratadour import f

# %% scene

frames_clean, frames = test_sequence()

# %% optimize

def f2(shift, frames):
    shifts = [(shift[0] * k, shift[1] * k) for k in range(1, len(frames) + 1)]
    return f(shifts, frames)

def register(frames):
    bounds = ((0, frames.shape[1]), (0, frames.shape[2]))
    x0 = np.random.randint(frames.shape[1], size=2)
    x0 = [120, 120]
    result = minimize(
        # f,
        f2,
        x0=x0,
        args=(frames,),
        bounds=bounds,
        method='SLSQP',
        options={'eps':1, 'ftol':1e-10},
        # disp=True,
        # workers=-1,
        # init='sobol',
        # popsize=500,
        # mutation=1.5,
    )

    return result
