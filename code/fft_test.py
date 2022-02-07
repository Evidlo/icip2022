#!/usr/bin/env python

import numpy as np
from multiml import correlate_and_sum, scale_and_sum
from observation import add_noise
import matplotlib.pyplot as plt

scene = np.arange(10)

num_frames = 4
shift = 2
frames = np.vstack([np.roll(scene, -shift * n) for n in range(num_frames)])

# %% multiml

cs = correlate_and_sum(frames)

# %% fft

image_axes = range(1, len(frames.shape))
frames_freq = np.fft.fftn(frames, axes=image_axes)
frames_freq_freq = np.fft.fft(frames_freq, axis=0, n=2 * frames.shape[0] - 1)

cs2 = np.fft.ifftn(
    np.fft.ifft(frames_freq_freq * frames_freq_freq, axis=0),
    axes=image_axes
).real
