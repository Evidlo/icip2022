#!/usr/bin/env python

from mas.data import strand_highres as scene
from observation import get_frames, add_noise
import matplotlib.pyplot as plt
import numpy as np
import rigidregistration
from multiml import register

# ----- scene generation -----
# %% scene

from skimage.data import hubble_deep_field
scene = hubble_deep_field()[:, :, 0]

drift = np.array((4, 4))
resolution_ratio=2

# ----- experiment -----
# %% experiment

def exp(*, dbsnr, noise_model, num_frames):

    frames_clean = get_frames(
        scene=scene,
        drift=drift,
        resolution_ratio=resolution_ratio,
        frame_size=(250, 250),
        num_frames=num_frames,
        start=(0, 0),
    )

    frames = add_noise(
        frames_clean,
        noise_model=noise_model,
        dbsnr=dbsnr
    )

    # reorder axes for rigidregistration
    frames_rigid = frames.copy()
    frames_rigid /= frames_rigid.max()
    frames_rigid = np.moveaxis(frames_rigid, (0, 1, 2), (2, 0, 1))

    # Instantiate imstack object.
    s=rigidregistration.stackregistration.imstack(frames_rigid)
    s.getFFTs()
    # Set Fourier mask
    s.makeFourierMask(mask='gaussian',n=2)
    # Calculate image shifts
    s.findImageShifts(findMaxima='pixel',verbose=False);
    # Create registered image stack and average
    s.get_averaged_image()

    # rigidRegistration drift estimate
    est_rigid = np.array((
        np.diff(s.shifts_x).mean(),
        np.diff(s.shifts_y).mean(),
    ))

    est_multiml = register(frames, 'down')

    return {
        'rigid_ae': np.linalg.norm(est_rigid - drift / resolution_ratio),
        'multiml_ae': np.linalg.norm(est_multiml - drift / resolution_ratio)
    }

from mas.misc import combination_experiment

result = combination_experiment(
    exp,
    dbsnr=[-40, -35, -30, -25, -20, -15, -10],
    noise_model=['gaussian'],
    iterations=20,
    num_frames=[20, 30, 40]
)

# ----- plotting -----
# %% plot

import seaborn as sns
sns.relplot(
    data=result,
    kind="line",
    x="dbsnr",
    # y="rigid_ae",
    y="multiml_ae",
    hue="num_frames",
    # style="num_frames",
    marker='o'
)
plt.show()
