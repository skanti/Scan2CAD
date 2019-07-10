import numpy as np
import torch


def nms(thresh, kernel_size, x):
    x_max = torch.nn.functional.max_pool3d(x, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
    x_binarized = torch.gt(x_max, thresh)
    x_nms = x_binarized & torch.eq(x, x_max)
    return x_nms
