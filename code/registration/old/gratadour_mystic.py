#!/bin/env python
## Evan Widloski - 2017-02-17
## ECE321 Mini Project

from mystic.penalty import quadratic_inequality
import numpy as np
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

@quadratic_inequality(constraint, k=1e4)
def penalty(x):
  return 0.0

from mystic.solvers import diffev2
from mystic.monitors import VerboseMonitor
mon = VerboseMonitor(10)

result = diffev2(objective, x0=bounds, penalty=penalty, npop=10, gtol=200, \
                 disp=False, full_output=True, itermon=mon, maxiter=M*N*100)

print result[0]
print result[1]
