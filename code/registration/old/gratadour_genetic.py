#!/bin/env python
## Evan Widloski - 2017-02-17
## ECE321 Mini Project

from platypus.algorithms import NSGAII
from platypus.core import Problem
from platypus.types import Real, Subset
import numpy as np
from scipy.optimize import differential_evolution as de
from multiml.observation import get_frames, add_noise
from gratadour import f

# %% scene

from skimage.data import hubble_deep_field
scene = hubble_deep_field()[:, :, 0]

num_frames = 30
frame_size=np.array((250, 250))
drift = np.array((10, 10))

frames_clean = get_frames(
    scene=scene,
    drift=drift,
    resolution_ratio=1,
    frame_size=frame_size,
    num_frames=num_frames,
    start=(0, 0),
)

frames = add_noise(
    frames_clean,
    noise_model='gaussian',
    dbsnr=-20,
)

# %% optimize

def f2(shift, frames):
    shifts = [(shift[0] * k, shift[1] * k) for k in range(1, len(frames) + 1)]
    return f(shifts, frames)

def register(frames):
    result = de(
        # f,
        f2,
        # [(0, 250)] * 2 * len(frames),
        [(0, frames.shape[1]), (0, frames.shape[2])],
        args=(frames,),
        disp=True,
        workers=-1,
        init='sobol',
        popsize=500,
        mutation=1.5,
    )

    return result
