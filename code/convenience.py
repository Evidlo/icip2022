#!/usr/bin/env python

import numpy as np
from skimage.data import hubble_deep_field
from multiml.observation import get_frames, add_noise
scene = hubble_deep_field()[:, :, 0]

def observe(dbsnr=-10, num_frames=20, drift=np.array((4, 4)), noise_model='gaussian'):
    max_count=None
    frame_size=np.array((250, 250))

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
    return frames, frames_clean
