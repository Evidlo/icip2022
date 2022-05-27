#!/usr/bin/env python

import numpy as np
from scipy.ndimage import fourier_shift

def compute_ref(shifts, frames):

    if type(shifts) is not np.ndarray:
        shifts = np.array(shifts)

    shifts = shifts.reshape((len(frames), 2))
    shifts = np.round(shifts).astype(int)

    # compute reference image
    ref = np.zeros(frames[0].shape)
    for frame, shift in zip(frames, shifts):
        ref += np.roll(frame, shift, axis=range(len(shift)))
    ref /= len(frames)

    return ref


def f(shifts, frames, ref=None):
    """Cost function of Gratadour paper

    Args:
        shifts (ndarray): array containing shifts of frames in sequence
        frames (ndarray): noisy observed frames
        ref (ndarray, optional): precomputed reference image

    Returns:
        float: likelihood function evaluated at argument
    """

    frames /= frames.max()

    if type(shifts) is not np.ndarray:
        shifts = np.array(shifts)

    shifts = shifts.reshape((len(frames), 2))

    if ref is None:
        ref = compute_ref(shifts, frames)

    # compute cost
    cost = 0
    for frame, shift in zip(frames, shifts):
        cost += np.sum(
            (
                frame -
                np.fft.ifft2(fourier_shift(np.fft.fft2(ref), -shift)).real
            )**2
        )

    return cost / (frames.shape[1])


def register(frames):
    """Register by minimizing Gratadour cost

    Args:
        frames (ndarray): noisy frames of size (frames, width, height)

    Returns:
        ndarray: shift estimate of each frame
    """

    from scipy.optimize import fmin_cg

    # initialize with random shift
    x0 = np.random.random(len(frames) * 2)

    shifts = fmin_cg(f, x0=x0, args=(frames,)).reshape(len(frames), 2)

    return shifts
