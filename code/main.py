#!/usr/bin/env python

from mas.data import strand_highres as scene
from observation import get_frames, add_noise
import matplotlib.pyplot as plt
import numpy as np

# ----- scene generation -----
# %% scene

from skimage.data import hubble_deep_field
scene = hubble_deep_field()[:, :, 0]

frames_clean = get_frames(
    scene=scene,
    resolution_ratio=2,
    drift=(4, 4),
    frame_size=(250, 250),
    num_frames=30,
    start=(0, 0),
)
frames = add_noise(
    frames_clean,
    noise_model='gaussian',
    dbsnr=-30
)

# ----- rigidRegistration -----
# %% rigid

import rigidregistration

frames_rigid = frames.copy()
frames_rigid /= frames_rigid.max()
frames_rigid = np.moveaxis(frames_rigid, (0, 1, 2), (2, 0, 1))

s=rigidregistration.stackregistration.imstack(frames_rigid)    # Instantiage imstack object.
s.getFFTs()
# Set Fourier mask
s.makeFourierMask(mask='gaussian',n=2)
# Calculate image shifts
s.findImageShifts(findMaxima='pixel',verbose=False);
# Create registered image stack and average
s.get_averaged_image()

# rigidRegistration drift estimate
est_rigid = (
    np.diff(s.shifts_x).mean(),
    np.diff(s.shifts_y).mean(),
)
print(est_rigid)

# ----- Multi ML -----
# %% multiml

from multiml import register
from multiml import correlate_and_sum, scale_and_sum, shift_and_sum

# csums = correlate_and_sum(frames)
# result, scaled = scale_and_sum(csums, scale_dir='down')
est_multiml = register(frames, 'down')
print(est_multiml)
recon = shift_and_sum(frames, est_multiml, mode='crop')

# %% plot - plot results

plt.subplot(1, 2, 1)
plt.imsave('../images/recon.png', recon, cmap='gist_heat')
plt.subplot(1, 2, 2)
plt.imsave('../images/recon_clean.png', frames_clean[len(frames_clean) // 2], cmap='gist_heat')

# plt.imshow(o.frames[0])
# plt.figure()
# plt.imshow(o.frames[-1])
# plt.show()
