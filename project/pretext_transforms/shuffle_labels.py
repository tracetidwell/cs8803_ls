import numbers
import random
import math
import numpy as np

import torch
from torch_geometric.transforms import LinearTransformation


class ShuffleLabels(object):
    def __init__(self, shuffle_dict):
        self.shuffle_dict = shuffle_dict

    def __call__(self, data):
        data.y = self.shuffle_dict[int(data.y)]
        return data

    def __repr__(self):
        return '{}(Bands: {}, Variance: {}, axis={})'.format(self.__class__.__name__,
                self.degree_centers, self.band_width, self.axis)
