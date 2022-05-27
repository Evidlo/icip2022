#!/usr/bin/env python

import numpy as np
from image_registration import chi2_shift as ginsburg_register


def register(frames):

    ginsburg_ests_x = []
    ginsburg_ests_y = []
    for frame1, frame2 in zip(frames[:-1], frames[1:]):
        xoff, yoff, exoff, eyoff = ginsburg_register(frame2, frame1)
        ginsburg_ests_x.append(xoff)
        ginsburg_ests_y.append(yoff)

    est_ginsburg = (
        np.array(ginsburg_ests_x).mean(),
        np.array(ginsburg_ests_y).mean()
    )

    return est_ginsburg
