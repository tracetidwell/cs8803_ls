import numbers
import random
import math
import numpy as np

import torch
from torch_geometric.transforms import LinearTransformation


class BandRotate(object):
    r"""Rotates node positions around a specific axis by a randomly sampled
    factor within a set of intervals
    Args:
        degree_centers list(float): List of rotation band degree_centers
        band_with (float, optional): Width to sample from centered around each band
        axis (int, optional): The rotation axis. (default: :obj:`0`)
    """

    def __init__(self, degree_centers, rot_noise=0.0, axis=0):
        assert len(degree_centers) != 0
        self.degree_centers = degree_centers
        self.rot_noise = rot_noise
        self.axis = axis

    def __call__(self, data):

        degree_index = np.random.choice(len(self.degree_centers), 1)[0]
        chosen_degree = self.degree_centers[degree_index] + (np.random.random()-.5)*self.rot_noise

        if self.axis == 0:
            data.band_x = int(degree_index)
            data.band_all = int(degree_index)
        else:
            data.band_y = int(degree_index)
            data.band_all += len(self.degree_centers)*int(degree_index)

        degree = math.pi * chosen_degree / 180.0
        sin, cos = math.sin(degree), math.cos(degree)

        if data.pos.size(1) == 2:
            matrix = [[cos, sin], [-sin, cos]]
        else:
            if self.axis == 0:
                matrix = [[1, 0, 0], [0, cos, sin], [0, -sin, cos]]
            elif self.axis == 1:
                matrix = [[cos, 0, -sin], [0, 1, 0], [sin, 0, cos]]
            else:
                matrix = [[cos, sin, 0], [-sin, cos, 0], [0, 0, 1]]
        return LinearTransformation(torch.tensor(matrix))(data)

    def __repr__(self):
        return '{}(Bands: {}, Variance: {}, axis={})'.format(self.__class__.__name__,
                self.degree_centers, self.rot_noise, self.axis)
