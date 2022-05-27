#!/usr/bin/env python
# coding: utf-8

import numpy as np
import rigidregistration

# %% forward

def register_debug(frames, maxima='pixel', mask='gaussian', n=2):

    stack = np.moveaxis(frames, (0, 1, 2), (2, 0, 1))

    # normalize
    stack /= stack.max()

    # %% register

    # Load data and instantiate imstack object
    s=rigidregistration.stackregistration.imstack(stack)    # Instantiage imstack object.
    s.getFFTs()

    # Set Fourier mask
    s.makeFourierMask(mask=mask,n=n)

    # Calculate image shifts
    s.findImageShifts(findMaxima=maxima,verbose=False);
    s.get_imshifts()

    return s

    return np.vstack((s.shifts_x, s.shifts_y)).T

def register(frames):
    return np.diff(register_debug(frames), axis=0).mean(axis=0)
