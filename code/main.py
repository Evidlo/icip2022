#!/usr/bin/env python

from multiml.observation import get_frames, add_noise
from multiml.multiml import register, shift_and_sum, correlate_and_sum, scale_and_sum
import matplotlib.pyplot as plt
import numpy as np

# ----- scene generation -----
# %% scene

from skimage.data import hubble_deep_field
scene = hubble_deep_field()[:, :, 0]

frame_size = np.array((250, 250))
frames_clean = get_frames(
    scene=scene,
    resolution_ratio=1,
    drift=(1, 1),
    frame_size=frame_size,
    num_frames=30,
    start=(0, 0),
)
frames = add_noise(
    frames_clean,
    noise_model='gaussian',
    # noise_model=None,
    # dbsnr=-25
    dbsnr=-10
)
# plt.imsave('../paper/images/frame_clean.png', frames_clean[0], cmap='gist_heat')
# plt.imsave('../paper/images/frame.png', frames[0], cmap='gist_heat')


# from mas.strand_generator import StrandVideo
# sv = StrandVideo(drift_velocity=0.4e-3)
# frames = sv.frames

# ----- rigidRegistration -----
# %% rigid

import rigidregistration

frames_rigid = frames.copy()

stack = np.moveaxis(frames_rigid, (0, 1, 2), (2, 0, 1))
# normalize
stack /= stack.max()
# Load data and instantiate imstack object
# stack=np.rollaxis(imread(f),0,3)/float(2**16)           # Rearrange axes and normalize data
s=rigidregistration.stackregistration.imstack(stack)    # Instantiage imstack object.
s.getFFTs()
# Set Fourier mask
s.makeFourierMask(mask='gaussian', n=10)
# Calculate image shifts
s.findImageShifts(findMaxima='pixel',verbose=False);
s.get_imshifts()


# rigidRegistration drift estimate
est_rigid = (
    np.diff(s.shifts_x % frame_size[0]).mean(),
    np.diff(s.shifts_y % frame_size[1]).mean(),
)
print('rigid:', est_rigid)

# ----- ginsburg -----
# %% ginsburg

from image_registration import chi2_shift

ginsburg_ests_x = []
ginsburg_ests_y = []
for frame1, frame2 in zip(frames[:-1], frames[1:]):
    xoff, yoff, exoff, eyoff = chi2_shift(frame2, frame1)
    ginsburg_ests_x.append(xoff)
    ginsburg_ests_y.append(yoff)

est_ginsburg = (
    np.array(ginsburg_ests_x).mean(),
    np.array(ginsburg_ests_y).mean()
)

print('ginsburg:', est_ginsburg)

# ----- Multi ML -----
# %% multiml

# csums = correlate_and_sum(frames)
# result, scaled = scale_and_sum(csums, scale_dir='down')
est_multiml = register(frames, scale_dir='down', disable_print=True)
# remap registered coordinates from [0, size] -> [-size/2, size/2]
est_multiml = est_multiml - frame_size * (est_multiml > frame_size // 2)
print(est_multiml)
recon = shift_and_sum(frames, est_multiml, mode='crop')
recon_clean = shift_and_sum(frames_clean, est_multiml, mode='crop')

# %% plot - plot results

# plt.imsave('../paper/images/recon.png', recon, cmap='gist_heat')
# plt.imsave('../paper/images/recon_clean.png', recon_clean, cmap='gist_heat')

# plt.imshow(o.frames[0])
# plt.figure()
# plt.imshow(o.frames[-1])
# plt.show()
