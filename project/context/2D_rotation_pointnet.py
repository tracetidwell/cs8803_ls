import sys
import argparse
import os.path as osp
sys.path.insert(1, '..')
import numpy as np

#Torch Imports
import torch
import torch.nn.functional as F
#from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#Internal Imports
from pretext_transforms import BandRotate
from models import PointNet
from context import Trainer
from utils.datasets import ModelNet
from dataset_splits import process_pickle_bysplit

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../data/ModelNet', help='Path to data')
    parser.add_argument('--sample_points', type=int, default=1032)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_workers', type=int, default=0)
    parser.add_argument('--n_epochs', type=int, default=50)
    parser.add_argument('--modelnet_version', type=str, default='10')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--load_model', action='store_true', help='Load a previous model')
    parser.add_argument('--checkpoints_path', default='saved_models', help='Path to load model from')
    parser.add_argument('--logs_path', default='runs', help='Path to load model from')
    parser.add_argument('--model_name', default='pointnet_pretext_160_2019-11-05.pt', help='Path to load model from')
    parser.add_argument('--processed_name', default='processed', help='Path to load model from')
    # parser.add_argument('--band_centers', nargs='*', type=int, action='store', default=[0, 10, 20, 30, 40, 50, 60, 70, 80],
    #                     help='List of rotation band centers')
    parser.add_argument('--rot_noise', type=int, default=5)
    parser.add_argument('--max_rot', type=int, default=80)
    parser.add_argument('--min_rot', type=int, default=0)
    parser.add_argument('--n_bands', type=int, default=9)
    parser.add_argument('--trans_noise', type=float, default=0.005)
    parser.add_argument('--save_every_n_epochs', type=int, default=5)
    parser.add_argument('--load_subset', action='store_true', help='Load a previous model')
    parser.add_argument('--subset_pct', type=int, default=100)
    args = parser.parse_args()

    path = osp.join(
        osp.dirname(osp.realpath(__file__)), f'{args.data_path}{args.modelnet_version}')

    step = int((args.max_rot - args.min_rot) / (args.n_bands - 1))
    degree_centers = [center for center in range(args.min_rot, args.max_rot+1, step)]

    print("=====================================")
    print("DOUBLE PREDICTION BAND_ROTATE"       )
    print("Number of Bands:", args.n_bands)
    print('Band Centers:', degree_centers)
    print("Band Noise:", args.rot_noise)
    print("Translate Noise:", args.trans_noise)
    print("=====================================")

    if args.load_subset:
        train_dict = process_pickle_bysplit(f'dataset_splits/ModelNet{str(args.modelnet_version)}_TrainingSplits.pickle', args.subset_pct)
    else:
        train_dict = None

    pre_transform = T.NormalizeScale()
    transform =  T.Compose([T.SamplePoints(args.sample_points),
                            T.RandomTranslate(args.trans_noise),
                            BandRotate(degree_centers, args.rot_noise),
                            T.RandomTranslate(args.trans_noise),
                            BandRotate(degree_centers, args.rot_noise, axis=1),
                            T.RandomTranslate(args.trans_noise)])
    train_dataset = ModelNet(path, str(args.modelnet_version), True, transform,
                             pre_transform, processed_name=args.processed_name,
                             train_dict=train_dict)
    test_dataset = ModelNet(path, str(args.modelnet_version), False, transform,
                            pre_transform, processed_name=args.processed_name)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.n_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=args.n_workers)

    model = PointNet(len(degree_centers)**2)
    simple_trainer = Trainer(model, log_dir=args.logs_path, lr=args.lr, predictor="band_all")
    if args.load_model:
        simple_trainer.load_path(osp.join(args.checkpoints_path, args.model_name))

    for epoch in range(1, args.n_epochs + 1):
        print(f'Running epoch {epoch} of {args.n_epochs}')
        train_acc = simple_trainer.train(train_loader, verbose=True)
        test_acc = simple_trainer.test(test_loader, verbose=True)
        if epoch % args.save_every_n_epochs == 0:
            simple_trainer.save(f"{args.checkpoints_path}/2d_rot_modelnet{args.modelnet_version}_{args.n_bands}bands_ckpt_{epoch}.pt")
        print('-----------------------------')

    simple_trainer.cleanup()
