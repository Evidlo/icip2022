#!/usr/bin/env python

import numpy as np
from astroalign import find_transform


def register(frames):

    astroalign_ests = []
    for frame1, frame2 in zip(frames[:-1], frames[1:]):
        trans, _ = find_transform(frame1, frame2)
        astroalign_ests.append(trans.translation)
        # import ipdb
        # ipdb.set_trace()

    est_astroalign = np.mean(astroalign_ests, axis=0)

    return est_astroalign

if __name__ == '__main__':
    from multiml.observation import test_sequence
    fc, f = test_sequence(dbsnr=0)

    result = register(f)
    print(result)
