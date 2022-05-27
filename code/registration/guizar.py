#!/usr/bin/env python

from skimage.registration import phase_cross_correlation
import numpy as np

def register(frames):
    results = []
    for frame_a, frame_b in zip(frames[:-1], frames[1:]):
        results.append(phase_cross_correlation(frame_a, frame_b)[0])

    results = np.array(results)
    return results.mean(axis=0)
