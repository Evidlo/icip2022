#!/usr/bin/env python

# estimate registration error within each correlation sum

#!/usr/bin/env python

from mas.data import strand_highres as scene
from mas.misc import experiment
from observation import get_frames, add_noise
from multiml import correlate_and_sum
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# %% scene - scene generation

# scene = np.zeros((2000, 2000))
# scene[450, 450] = 1
drift = (10, 10)
resolution_ratio = 2
frames_clean = get_frames(
    scene=scene,
    resolution_ratio=resolution_ratio,
    drift=drift,
    frame_size=(500, 500),
    num_frames=30,
    start=(0, 0),
)

def exp():

    frames = add_noise(
        frames_clean,
        noise_model='gaussian',
        dbsnr=0
    )

    csums = correlate_and_sum(frames)
    errors = []

    for n, csum in enumerate(csums):
        maximum = np.unravel_index(np.argmax(csum), csum.shape)
        errors.append(np.linalg.norm(np.array(drift) / resolution_ratio * n - np.array(maximum)))

    return {
        "errors": np.array(errors)
    }

result = experiment(exp, iterations=5)
sns.set()
sns.lineplot(
    x='max_x',
    y='max_y',
    data=result,
)
