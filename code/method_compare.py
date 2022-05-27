#!/usr/bin/env python

from multiml.observation import get_frames, add_noise
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rigidregistration
from multiml.observation import test_sequence

# ----- experiment -----
# %% experiment

def exp(*, dbsnr, num_frames):

    size = 250
    drift = np.random.randint(0, size / num_frames, 2)
    frames_clean, frames = test_sequence(
        drift=drift,
        dbsnr=dbsnr,
        num_frames=num_frames,
        frame_size = (size, size),
    )

    # ----- Rigid -----
    # from registration.rigid import register as rigid_register
    # est_rigid = rigid_register(frames)

    # ----- Ginsburg -----
    from registration.ginsburg import register as ginsburg_register
    est_ginsburg = ginsburg_register(frames)

    # ----- Guizar Sicairos -----
    from registration.guizar import register as guizar_register
    est_guizar = guizar_register(frames)

    # ----- MultiML (ours) -----
    from multiml.multiml import register as multiml_register
    est_multiml = multiml_register(frames)
    # remap registered coordinates from [0, size] -> [-size/2, size/2]
    # est_multiml = est_multiml - frame_size * (est_multiml > frame_size // 2)


    # compute absolute error
    return {
        'Ginsburg': np.linalg.norm(est_ginsburg - drift),
        # 'Rigid': np.linalg.norm(est_rigid - drift),
        'Ours': np.linalg.norm(est_multiml - drift),
        'Guizar': np.linalg.norm(est_guizar - drift)
    }

from mas.misc import combination_experiment

result = combination_experiment(
    exp,
    dbsnr=np.linspace(-40, -15, 10),
    iterations=50,
    num_frames=[20]
)

# ----- plotting -----
# %% plot

x = pd.melt(
    result,
    id_vars=['dbsnr', 'num_frames'],
    # id_vars=['max_count', 'noise_model', 'num_frames'],
    value_vars=['Ginsburg', 'Ours', 'Guizar'],
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
    hue="method",
    style="method",
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
# plot.set(yscale='log')
plt.show()

def saveit():
    plt.savefig('../paper/images/method_compare.pdf', bbox_inches='tight')
