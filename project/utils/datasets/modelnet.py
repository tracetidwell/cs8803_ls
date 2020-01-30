import os
import os.path as osp
import shutil
import glob

import torch
from torch_geometric.data import download_url, extract_zip
from torch_geometric.read import read_off

from utils.data import ProjectInMemoryDataset


class ModelNet(ProjectInMemoryDataset):
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
                 pre_transform=None, pre_filter=None, raw_name='raw', processed_name='processed', train_dict=None):
        assert name in ['10', '40']
        self.name = name
        self.train_dict = train_dict
        super(ModelNet, self).__init__(root, transform, pre_transform, pre_filter, raw_name, processed_name)
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
        print(self.processed_paths[0])
        torch.save(self.process_set('train'), self.processed_paths[0])
        print(self.processed_paths[1])
        torch.save(self.process_set('test'), self.processed_paths[1])

    def process_set(self, dataset):

        if dataset == 'train' and self.train_dict is not None:
            categories = sorted(self.train_dict.keys())

            data_list = []
            for target, category in enumerate(categories):
                folder = osp.join(self.raw_dir, category, dataset)
                files = self.train_dict[category]
                for file in files:
                    if file != '.DS_Store':
                        #print(osp.join(folder, file))
                        data = read_off(osp.join(folder, file))
                        data.y = torch.tensor([target])
                        data_list.append(data)

        else:
            categories = glob.glob(osp.join(self.raw_dir, '*', ''))
            categories = sorted([x.split(os.sep)[-2] for x in categories])

            data_list = []
            for target, category in enumerate(categories):
                folder = osp.join(self.raw_dir, category, dataset)
                paths = glob.glob('{}/{}_*.off'.format(folder, category))
                for path in paths:
                    data = read_off(path)
                    data.y = torch.tensor([target])
                    data_list.append(data)

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]

        return self.collate(data_list)

    def __repr__(self):
        return '{}{}({})'.format(self.__class__.__name__, self.name, len(self))