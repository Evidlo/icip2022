#!/bin/env python
## Evan Widloski - 2017-02-17
## ECE321 Mini Project

import numpy as np
from multiml.observation import get_frames, add_noise
from scipy.optimize import fmin_cg

from gratadour import f, fprime

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

def register(frames):
    """Register by minimizing Gratadour cost

    Args:
        frames (ndarray): noisy frames of size (frames, width, height)

    Returns:
        ndarray: shift estimate of each frame
    """

    from scipy.optimize import fmin_cg

    # initialize with random shift
    x0 = np.random.random(len(frames) * 2)

    shifts = fmin_cg(f, x0=x0, args=(frames,)).reshape(len(frames), 2)

    return shifts
