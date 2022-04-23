# This code is modified from https://github.com/facebookresearch/CovidPrognosis/blob/main/covidprognosis/data/transforms.py

import numpy as np
import torch

class HistogramNormalize(object):
    """
    Apply histogram normalization.
    Args:
        number_bins: Number of bins to use in histogram.
    """

    def __init__(self, number_bins=256):
        self.number_bins = number_bins

    def __call__(self, image):
        image = image.numpy()

        # get image histogram
        image_histogram, bins = np.histogram(
            image.flatten(), self.number_bins, density=True
        )
        cdf = image_histogram.cumsum()  # cumulative distribution function
        cdf = 255 * cdf / cdf[-1]  # normalize

        # use linear interpolation of cdf to find new pixel values
        image_equalized = np.interp(image.flatten(), bins[:-1], cdf)
        image_equalized.reshape(image.shape)

        image = torch.tensor(image_equalized.reshape(image.shape), dtype=torch.float32)
        return image


if __name__ == "__main__":
    pass