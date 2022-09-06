#!/usr/bin/env python
# Evan Widloski - 2022-02-01
# Monte Carlo simulation of algorithm error for various SNR and number of frames

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from multiml.observation import test_sequence

# ----- experiment -----
# %% experiment

def exp(*, dbsnr=None, max_count=None, noise_model, num_frames):

    size = 250
    # using nested pqdm does weird stuff to RNG
    np.random.seed(int.from_bytes(open('/dev/urandom', 'rb').read(4), 'big'))
    drift = np.random.randint(0, size / num_frames, 2)
    frames_clean, frames = test_sequence(
        drift=drift,
        dbsnr=dbsnr,
        num_frames=num_frames,
        frame_size = (size, size),
        jitter_std=1,
        rotation_std=1
    )

    from multiml.multiml import register

    est_multiml = register(frames, 'down')

    # compute absolute error
    return {
        'abs_error': np.linalg.norm(est_multiml - drift),
    }

from expsweep import experiment

result_gaussian = experiment(
    exp,
    dbsnr=np.linspace(-40, -15, 10),
    noise_model=['gaussian'],
    repeat=50,
    num_frames=[20, 30, 40]
)

# ----- plotting -----
# %% plot

params = {
    'text.usetex' : True,
    'font.size' : 16,
    'font.family' : 'lmodern',
}
plt.rcParams.update(params)

import seaborn as sns
plot = sns.lineplot(
    data=result,
    x="dbsnr",
    y="abs_error",
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
plt.show()

def saveit():
    plt.savefig('../paper/images/db_sweep.pdf', bbox_inches='tight')
