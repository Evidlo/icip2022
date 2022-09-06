#!/usr/bin/env python
# Evan Widloski - 2022-02-01
# Monte Carlo simulation comparing absolute error of algorithms

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from multiml.observation import test_sequence

# ----- experiment -----
# %% experiment

def exp(*, dbsnr, num_frames):

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

    # ----- Rigid -----
    # from registration.rigid import register as rigid_register
    # est_rigid = rigid_register(frames)
    # this one don't work too good
    # https://github.com/bsavitzky/rigidRegistration/issues/2

    # ----- Ginsburg -----
    from registration.ginsburg import register as ginsburg_register
    est_ginsburg = ginsburg_register(frames)

    # ----- Guizar Sicairos -----
    from registration.guizar import register as guizar_register
    est_guizar = guizar_register(frames)

    # ----- MultiML (ours) -----
    from multiml.multiml import register as multiml_register
    est_multiml = multiml_register(frames)


    # compute absolute error
    return {
        # 'Rigid': np.linalg.norm(est_rigid - drift),
        'Ginsburg': np.linalg.norm(est_ginsburg - drift),
        'Ours': np.linalg.norm(est_multiml - drift),
        'Guizar': np.linalg.norm(est_guizar - drift)
    }

from expsweep import experiment

result = experiment(
    exp,
    dbsnr=np.linspace(-40, -15, 10),
    repeat=50,
    num_frames=[20],
    merge=True
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
    y="result",
    hue="experiment",
    style="experiment",
    marker='o',
    palette='bright'
)
sns.move_legend(plot, 'upper right')
plt.legend(loc='upper right', title='Method')
plt.xlabel('SNR (dB)')
plt.ylabel('Registration Error (px)')
plt.grid(True, which='major', color='k')
plt.grid(True, which='minor')
plt.minorticks_on()
plt.tight_layout()
plt.margins(x=0)
plt.show()

def saveit():
    plt.savefig('../paper/images/method_compare.pdf', bbox_inches='tight')
