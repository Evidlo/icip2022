#!/usr/bin/env python

from multiml.observation import get_frames, add_noise
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rigidregistration
from multiml.multiml import register

# ----- scene generation -----
# %% scene

from skimage.data import hubble_deep_field
scene = hubble_deep_field()[:, :, 0]

# ----- experiment -----
# %% experiment

def exp(*, dbsnr=None, max_count=None, noise_model, num_frames):

    # randomize drift amount - restrict ck<N
    frame_size=np.array((250, 250))
    # drift = np.random.randint((0, 0), frame_size / num_frames)
    drift = np.array((8, 8))

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
        noise_model=noise_model,
        dbsnr=dbsnr,
        max_count=max_count,
    )

    """
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
    """

    est_multiml = register(frames, 'down')
    # remap registered coordinates from [0, size] -> [-size/2, size/2]
    est_multiml = est_multiml - frame_size * (est_multiml > frame_size // 2)

    # compute absolute error
    return {
        # 'rigid': np.linalg.norm(est_rigid - drift),
        # 'est_rigid': est_rigid - drift,
        'multiml': np.linalg.norm(est_multiml - drift),
        'est_multiml': est_multiml - drift,
    }

from mas.misc import combination_experiment

result_gaussian = combination_experiment(
    exp,
    dbsnr=np.linspace(-40, -15, 10),
    noise_model=['gaussian'],
    iterations=50,
    num_frames=[20, 30, 40]
)
# result_poisson = combination_experiment(
#     exp,
#     max_count=np.linspace(10, 25, 50),
#     noise_model=['poisson'],
#     iterations=40,
#     num_frames=[20]
#     # num_frames=[20]
# )

# ----- plotting -----
# %% plot

x = pd.melt(
    # result_poisson,
    result_gaussian,
    id_vars=['dbsnr', 'noise_model', 'num_frames'],
    # id_vars=['max_count', 'noise_model', 'num_frames'],
    # value_vars=['rigid', 'multiml'],
    value_vars=['multiml'],
    var_name='method',
    value_name='abserr'
)

params = {
    'text.usetex' : True,
    'font.size' : 16,
    'font.family' : 'lmodern',
}
plt.rcParams.update(params)

import seaborn as sns
plot = sns.lineplot(
    data=x,
    x="dbsnr",
    # x="max_count",
    y="abserr",
    hue="num_frames",
    style="num_frames",
    marker='o',
    palette='bright'
)
sns.move_legend(plot, 'upper right')
plt.legend(loc='upper right', title='Number of Frames (K)')
plt.xlabel('SNR (dB)')
plt.ylabel('Registration Error (px)')
plt.grid(True, which='major', color='k')
plt.grid(True, which='minor')
plt.minorticks_on()
plt.tight_layout()
plt.margins(x=0)
# plot.set(yscale='log')
plt.show()

def saveit():
    plt.savefig('../paper/images/db_sweep.pdf', bbox_inches='tight')
