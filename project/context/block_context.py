import itertools
import argparse
import datetime
import os.path as osp
import numpy as np
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader, Batch
from torch_geometric.nn import PointConv, fps, radius, global_max_pool

#Internal Imports
from models import BlockPointNet
from context import Trainer
from utils.datasets import BlockModelNet


def train(train_loader):

    model.train()

    for i, batch in enumerate(train_loader):

        #print('batch ', i)

        batch = batch.to(device)
        optimizer.zero_grad()

        try:
            y_hat = model(batch)
        except ValueError:
            continue

        loss = F.nll_loss(y_hat, batch.y)
        loss.backward()
        optimizer.step()


def test(test_loader):

    model.eval()
    correct = 0

    for i, batch in enumerate(test_loader):

        #print('batch ', i)

        batch = batch.to(device)
        optimizer.zero_grad()

        try:
            with torch.no_grad():
                pred = model(batch).max(1)[1]
            correct += pred.eq(batch.y).sum().item()
        except ValueError:
            continue

    return correct / len(test_loader.dataset)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../data/ModelNet', help='Path to data')
    parser.add_argument('--sample_points', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_workers', type=int, default=6)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--modelnet_version', type=str, default='10')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--load_model', action='store_true', help='Load a previous model')
    parser.add_argument('--checkpoints_path', default='../saved_models', help='Path to load model from')
    parser.add_argument('--logs_path', default='../runs', help='Path to load model from')
    parser.add_argument('--model_name', default='pointnet_pretext_160_2019-11-05.pt', help='Path to load model from')
    parser.add_argument('--processed_name', default='blocks', help='Path to load model from')
    args = parser.parse_args()

    path = osp.join(
        osp.dirname(osp.realpath(__file__)), f'{args.data_path}{args.modelnet_version}')

    pre_transform = T.Compose([
        T.NormalizeScale(),
        T.SamplePoints(args.sample_points)
    ])
    transform = None
    train_dataset = BlockModelNet(path, str(args.modelnet_version), True, transform, pre_transform, processed_dir=args.processed_name)
    test_dataset = BlockModelNet(path, str(args.modelnet_version), False, transform, pre_transform, processed_dir=args.processed_name)
    train_loader = BlockDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.n_workers)
    test_loader = BlockDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.n_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model = BlockPointNet().to(device)
    simple_trainer = Trainer(model, log_dir=args.logs_path, lr=args.lr, predictor="y")
    if args.load_model:
        simple_trainer.load(osp.join(args.checkpoints_path, args.model_name))
    #     model = torch.load(args.model_path).to(device)
    # else:


    for epoch in range(1, args.n_epochs + 1):
        train_acc = simple_trainer.train(train_loader, verbose=True)
        test_acc = simple_trainer.test(test_loader, verbose=True)
        if epoch % 10 == 0:
            simple_trainer.save(f"{args.checkpoints_path}/block_modelnet{args.modelnet_version}_ckpt_{epoch}.pt")

    simple_trainer.cleanup()

    # for epoch in range(1, args.n_epochs + 1):
    #
    #     print('Training epoch {}'.format(epoch))
    #     train(train_loader)
    #     test_acc = test(test_loader)
    #     print('Epoch: {:03d}, Test: {:.4f}'.format(epoch, test_acc))
    #     if epoch % 5 == 0:
    #         torch.save(model, f'checkpoints/blockpointnet_pretext_{epoch}_{datetime.date.today()}.pt')
    #         print('Saving model...')
    #     print('-'*15)
