import os
import os.path as osp
import shutil
import glob
import itertools
import numpy as np
from sklearn.preprocessing import StandardScaler

import torch
from torch_geometric.data import Data, download_url, extract_zip
from torch_geometric.read import read_off

from utils.data import ProjectInMemoryDataset


def get_splits(data, split_type='medians'):

    pos = data.pos.numpy()

    if split_type == 'midpoints':
        return (np.max(pos, axis=0) - np.min(pos, axis=0)) / 2 + np.min(pos, axis=0)
    elif split_type == 'means':
        return np.mean(pos, axis=0)
    elif split_type == 'medians':
        return np.median(pos, axis=0)
    else:
        return [0, 0, 0]


def create_blocks(data, splits):

    blocks = []
    scaler = StandardScaler()
    full_pos = data.pos.numpy()
    #full_batch = data.batch.numpy()

    for x, y, z in list(itertools.product([0, 1], repeat=3)):

        if x == 0:
            mask_x = full_pos[:, 0] <= splits[0]
        else:
            mask_x = full_pos[:, 0] > splits[0]

        if y == 0:
            mask_y = full_pos[:, 1] <= splits[1]
        else:
            mask_y = full_pos[:, 1] > splits[1]

        if z == 0:
            mask_z = full_pos[:, 2] <= splits[2]
        else:
            mask_z = full_pos[:, 2] > splits[2]

        new_pos = full_pos[np.array([mask_x, mask_y, mask_z]).all(axis=0)]

        if new_pos.size > 0:
            # new_pos = (new_pos - np.min(new_pos, axis=0)) / (np.max(new_pos, axis=0) - np.min(new_pos, axis=0))
            # new_batch = full_batch[np.array([mask_x, mask_y, mask_z]).all(axis=0)]
            # blocks.append(Batch(torch.from_numpy(new_batch), pos=torch.from_numpy(new_pos)))
            new_pos = scaler.fit_transform(new_pos)
            #new_batch = full_batch[np.array([mask_x, mask_y, mask_z]).all(axis=0)]
            #blocks.append(Batch(torch.from_numpy(new_batch), pos=torch.from_numpy(new_pos)))
            blocks.append(torch.from_numpy(new_pos))

        else:
            # blocks.append(Batch(torch.tensor([]), pos=torch.tensor([])))
            blocks.append(torch.tensor([]))
            #blocks.append(Batch(torch.tensor([]), pos=torch.tensor([])))

    return blocks


class BlockModelNet(ProjectInMemoryDataset):
    r"""The ModelNet10/40 datasets from the `"3D ShapeNets: A Deep
    Representation for Volumetric Shapes"
    <https://people.csail.mit.edu/khosla/papers/cvpr2015_wu.pdf>`_ paper,
    containing CAD models of 10 and 40 categories, respectively.

    .. note::

        Data objects hold mesh faces instead of edge indices.
        To convert the mesh to a graph, use the
        :obj:`torch_geometric.transforms.FaceToEdge` as :obj:`pre_transform`.
        To convert the mesh to a point cloud, use the
        :obj:`torch_geometric.transforms.SamplePoints` as :obj:`transform` to
        sample a fixed number of points on the mesh faces according to their
        face area.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string, optional): The name of the dataset (:obj:`"10"` for
            ModelNet10, :obj:`"40"` for ModelNet40). (default: :obj:`"10"`)
        train (bool, optional): If :obj:`True`, loads the training dataset,
            otherwise the test dataset. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    urls = {
        '10':
        'http://vision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip',
        '40': 'http://modelnet.cs.princeton.edu/ModelNet40.zip'
    }

    def __init__(self, root, name='10', train=True, transform=None,
                 pre_transform=None, pre_filter=None):
        assert name in ['10', '40']
        self.name = name
        super(BlockModelNet, self).__init__(root, transform, pre_transform,
                                       pre_filter, raw_dir='raw', processed_dir='processed')
        path = self.processed_paths[0] if train else self.processed_paths[1]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return [
            'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor',
            'night_stand', 'sofa', 'table', 'toilet'
        ]

    @property
    def processed_file_names(self):
        return ['training.pt', 'test.pt']

    def download(self):
        path = download_url(self.urls[self.name], self.root)
        extract_zip(path, self.root)
        os.unlink(path)
        folder = osp.join(self.root, 'ModelNet{}'.format(self.name))
        shutil.rmtree(self.raw_dir)
        os.rename(folder, self.raw_dir)

    def process(self):
        torch.save(self.process_set('train'), self.processed_paths[0])
        torch.save(self.process_set('test'), self.processed_paths[1])

    def process_set(self, dataset):
        categories = glob.glob(osp.join(self.raw_dir, '*', ''))
        categories = sorted([x.split(os.sep)[-2] for x in categories])

        data_list = []
        for target, category in enumerate(categories):
            folder = osp.join(self.raw_dir, category, dataset)
            paths = glob.glob('{}/{}_*.off'.format(folder, category))

            for path in paths:
                data = read_off(path)
                data.y = torch.tensor([target])

                if self.pre_filter is not None:
                    data = self.pre_filter(data)
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                medians = get_splits(data)
                blocks = create_blocks(data, medians)

                for b1, b2 in itertools.permutations(range(8), 2):

                    if len(blocks[b1]) == 0 or len(blocks[b2]) == 0:
                        continue

                    # split = blocks[b1].shape[0]
                    # pos = torch.cat([blocks[b1], blocks[b2]])
                    # y = torch.tensor([b2])
                    data_list.append(Data(input1=blocks[b1], input2=blocks[b2],
                                          y=torch.tensor([b2])))

        return self.collate(data_list)

    def __repr__(self):
        return '{}{}({})'.format(self.__class__.__name__, self.name, len(self))
