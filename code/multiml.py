#!/usr/bin/env python

import numpy as np
from tqdm import tqdm
from skimage.transform import rescale, downscale_local_mean
from mas.forward_model import size_equalizer

def correlate_and_sum(frames, mode='CC', np=np):
    """Correlate all frame combinations and sum each group of correlations

    Args:
        frames (ndarray): input images
        mode (str, default='PC'): type of correlation to use. ('PC', 'NCC', 'CC')

    Returns:
        ndarray: axes (group, corr_x_coord, corr_y_coord)
    """

    image_axes = range(1, len(frames.shape))
    frames_freq = np.fft.fftn(frames, axes=image_axes)

    product_sums = np.zeros(
        (len(frames) - 1, *frames.shape[1:]),
        dtype='complex128'
    )
    for time_diff in tqdm(range(1, len(frames_freq)), desc='Correlation', leave=None):
        products = frames_freq[:-time_diff] * frames_freq[time_diff:].conj()
        if mode.upper() == 'PC':
            product_sums[time_diff - 1] = np.sum(products / np.abs(products), axis=0)
        elif mode.upper() == 'CC':
            product_sums[time_diff - 1] = np.sum(products, axis=0)
        else:
            raise Exception('Invalid mode {}'.format(mode.upper()))

    return np.fft.ifftn(np.array(product_sums), axes=image_axes).real


def scale_and_sum(csums, scale_dir='down'):
    """Scale each correlation group so peaks are incident, then crop and sum

    Args:
        csums (ndarray): correlation sums
        scale_dir (str): 'up' or 'down'

    Returns:
        result (ndarray): summed result
        scaled_csums (ndarray): scaled and cropped version of correlation groups

    """

    # shape_x, shape_y = csums.shape[1], csums.shape[2]
    # result = np.zeros((shape_x, shape_y))
    output_shape = csums.shape[1:]
    result = np.zeros((output_shape))
    scaled_csums = []

    for diff, csum in enumerate(tqdm(csums, desc='Scale', leave=None), 1):
        # pretrim csum to relevant area to make scaling faster
        trim_slice = [slice(None, int(shape / (len(csums) / diff) + 1)) for shape in output_shape]
        scaled = csum[trim_slice]
        # print('precrop', scaled.shape)
        # scale correlation group so peaks are incident
        if scale_dir == 'up':
            scaled = rescale(csum, len(csums) / diff, anti_aliasing=False)
        else:
            # scaled = downscale_local_mean(csum, (diff, diff))
            scale_slice = [slice(None, None, diff)] * len(csums.shape[1:])
            scaled = csum[scale_slice]
            pad_slice = [slice(None, shape) for shape in scaled.shape]
            padded = np.zeros((output_shape))
            padded[pad_slice] = scaled
            # scaled = size_equalizer(scaled, output_shape, mode='topleft')
            scaled = padded

        # print('scale', len(csums) / diff)
        # trim off excess
        trim_slice = [slice(None, shape) for shape in output_shape]
        scaled = scaled[trim_slice]
        # print('crop')
        scaled_csums.append(scaled)
        result += scaled

    return result, np.array(scaled_csums)


def register(frames, scale_dir='up'):
    """Register frames with Multi ML method

    Args:
        frames (ndarray): stack of frames to register
        scale_dir (str): scale correlation groups 'up' (subpixel registration) or
            'down'
    """
    csums = correlate_and_sum(frames)
    result, scaled = scale_and_sum(csums, scale_dir=scale_dir)

    return np.unravel_index(np.argmax(result), result.shape)
