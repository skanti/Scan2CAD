import numpy as np

def gaussian3d(l, sigma=1.0):
    """
    creates gaussian kernel with side length l and a sigma of sigma
    """

    ax = np.arange(-l//2 + 1.0, l//2 + 1.0)
    xx, yy, zz = np.meshgrid(ax, ax, ax)

    kernel = np.exp(-(xx**2 + yy**2 + zz**2)/(2.0*sigma**2))

    return np.asarray(kernel, dtype=np.float32)

def gaussian1d(l, sigma=1.0):

    ax = np.arange(-l//2 + 1.0, l//2 + 1.0)
    xx = np.meshgrid(ax)

    kernel = np.exp(-np.power(xx, 2)/(2.0*sigma**2))

    return np.asarray(kernel, dtype=np.float32)
