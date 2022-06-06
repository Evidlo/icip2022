#!/usr/bin/env python

import numpy as np
from scipy.ndimage import fourier_shift
from pqdm.threads import pqdm

def correct_frames(shifts, frames):
    """Shift frames back to their correct position

    Args:
        shifts (ndarray): (K, 2) array of shifts
        frames (ndarray): (K, N, N) array of shifted images

    Returns:
        (K, N, N) array of corrected images
    """

    if type(shifts) is not np.ndarray:
        shifts = np.array(shifts)

    shifts = shifts.reshape((len(frames), 2))
    shifts = np.round(shifts).astype(int)

    corrected_frames = np.empty(frames.shape)

    for k, (frame, shift) in enumerate(zip(frames, shifts)):
        corrected_frames[k] = np.fft.ifft2(
            fourier_shift(np.fft.fft2(frame), shift)
        ).real

    return corrected_frames

def compute_ref(shifts, frames):
    """Shift frames back to their correct position

    Args:
        shifts (ndarray): (K, 2) array of shifts
        frames (ndarray): (K, N, N) array of shifted images

    Returns:
        (K, N, N) array of corrected images
    """

    corrected_frames = correct_frames(shifts, frames)
    ref = np.sum(corrected_frames, axis=0)
    # ref /= len(frames)

    return ref


def f(shifts, frames):
    """Cost function of Gratadour paper

    Args:
        shifts (ndarray): array containing shifts of frames in sequence
        frames (ndarray): noisy observed frames

    Returns:
        float: (scaled) likelihood function evaluated at argument
    """

    frames /= frames.max()

    if type(shifts) is not np.ndarray:
        shifts = np.array(shifts)

    shifts = shifts.reshape((len(frames), 2))

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

    return -cost / np.prod(frames.shape[1:])

# def f(shifts, frames):
#     """Cost function of Gratadour paper

#     Args:
#         shifts (ndarray): array containing shifts of frames in sequence
#         frames (ndarray): noisy observed frames

#     Returns:
#         float: (scaled) likelihood function evaluated at argument
#     """

#     frames /= frames.max()

#     if type(shifts) is not np.ndarray:
#         shifts = np.array(shifts)

#     shifts = shifts.reshape((len(frames), 2))

#     corrected_frames = correct_frames(shifts, frames)
#     ref = compute_ref(shifts, frames)

#     cost = 0
#     for a, frame_a in enumerate(corrected_frames):
#         cost += np.sum((frame_a - ref)**2)

#     return -cost / np.prod(frames.shape[1:])

def fprime(shifts, frames, ref=None):
    """Cost function gradient of Gratadour paper

    Args:
        shifts (ndarray): array containing shifts of frames in sequence
        frames (ndarray): noisy observed frames
        ref (ndarray, optional): precomputed reference image

    Returns:
        ndarray: derivative vector of size `shifts.shape`
    """

    if type(shifts) is not np.ndarray:
        shifts = np.array(shifts)

    ref = compute_ref(shifts, frames)
    original_cost = f(shifts, frames)

    def wiggle(delta):
        return f(shifts + delta, frames) - original_cost

    deltas = np.eye(np.prod(shifts.shape)).reshape((-1, *shifts.shape))

    derivative = np.array(pqdm(deltas, wiggle, n_jobs=32, disable=True)).reshape(shifts.shape)

    return derivative
