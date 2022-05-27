import numpy as np
from skimage.transform import downscale_local_mean
from tqdm import tqdm

class Observation(object):

    def __init__(self, **kwargs):

        self.frames_clean = get_frames(**kwargs)
        self.frames = add_noise(self.frames, **kwargs)

    def get_frames(
        *,
        scene,
        resolution_ratio,
        num_frames,
        drift,
        frame_size,
        start=(0, 0),
    ):

        """
        Get linearly translated sequence of frames

        Args:
            scene (ndarray): high resolution input scene
            resolution_ratio (int): downsample factor from high resolution
                scene to low resolution frames
            frame_size (:obj:`list` of :obj:`int`): pixels of size of square frame
            num_frames (int): number of experiment frames
            drift (tuple): inter-frame drift
            start (tuple): top-left coordinate of first frame
            noise_model (function): function to apply noise to each frame
        """

        # FIXME check box bounds correctly, need to account for rotation
        assert (
            start[0] >= 0 and start[1] >= 0 and
            start[0] + num_frames * drift[0] + frame_size[0] < scene.shape[0] and
            start[1] + num_frames * drift[1] + frame_size[1] < scene.shape[1]
        ), "Frames drift outside of scene bounds"

        # drift should be integer amount on HR grid
        assert(type(drift[0]) is int and type(drift[1]) is int)

        # initialize output array
        frames = np.zeros((num_frames, frame_size[0], frame_size[1]))

        for frame_num in tqdm(range(num_frames), desc='Frames', leave=None, position=1):
            top = start[0] + frame_num * drift[0]
            left = start[1] + frame_num * drift[1]
            frames[frame_num] = downscale_local_mean(
                scene[
                    top:top + frame_size[0] * resolution_ratio,
                    left:left + frame_size[1] * resolution_ratio
                ],
                (resolution_ratio, resolution_ratio)
            )

        return frames

    def add_noise(frames, noise_model=None, max_count=None, dbsnr=0):
        """
        Add noise to frames

        Args:
            noise_model (str): 'poisson' or 'gaussian'
            dbsnr (float): target SNR in db
        """

        if noise_model == 'gaussian':
            var_sig = np.var(frames)
            var_noise = var_sig / 10**(dbsnr / 10)
            out = np.random.normal(loc=frames, scale=np.sqrt(var_noise))
        elif noise_model == 'poisson':
            if max_count is not None:
                sig_scaled = signal * (max_count / signal.max())
                # print('SNR:{}'.format(np.sqrt(sig_scaled.mean())))
                out = poisson.rvs(sig_scaled) * (signal.max() / max_count)
            else:
                avg_brightness = 10**(dbsnr / 10)**2
                sig_scaled = signal * (avg_brightness / signal.mean())
                out = poisson.rvs(sig_scaled) * (signal.mean() / avg_brightness)
        return out
