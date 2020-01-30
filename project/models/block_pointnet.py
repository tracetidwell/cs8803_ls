import itertools
import argparse
import datetime
import os.path as osp
import numpy as np
from sklearn.preprocessing import StandardScaler

import torch
import torch.nn.functional as F
from torch.nn import Sequential as Seq, Linear as Lin, ReLU, BatchNorm1d as BN
import torch_geometric.transforms as T
from torch_geometric.nn import PointConv, fps, radius, global_max_pool

from utils.datasets import BlockModelNet
from utils.data import BlockBatch, BlockDataLoader


class SAModule(torch.nn.Module):
    def __init__(self, ratio, r, nn):
        super(SAModule, self).__init__()
        self.ratio = ratio
        self.r = r
        self.conv = PointConv(nn)

    def forward(self, x, pos, batch):
        idx = fps(pos, batch, ratio=self.ratio)
        row, col = radius(pos, pos[idx], self.r, batch, batch[idx],
                          max_num_neighbors=64)
        edge_index = torch.stack([col, row], dim=0)
        x = self.conv(x, (pos, pos[idx]), edge_index)
        pos, batch = pos[idx], batch[idx]
        return x, pos, batch


class GlobalSAModule(torch.nn.Module):
    def __init__(self, nn):
        super(GlobalSAModule, self).__init__()
        self.nn = nn

    def forward(self, x, pos, batch):
        x = self.nn(torch.cat([x, pos], dim=1))
        x = global_max_pool(x, batch)
        pos = pos.new_zeros((x.size(0), 3))
        batch = torch.arange(x.size(0), device=batch.device)
        return x, pos, batch


def MLP(channels, batch_norm=True):
    return Seq(*[
        Seq(Lin(channels[i - 1], channels[i]), ReLU(), BN(channels[i]))
        for i in range(1, len(channels))
    ])


class BlockPointNet(torch.nn.Module):
    def __init__(self):
        super(BlockPointNet, self).__init__()

        self.sa1_module = SAModule(0.5, 0.2, MLP([3, 64, 64, 128]))
        self.sa2_module = SAModule(0.25, 0.4, MLP([128 + 3, 128, 128, 256]))
        self.sa3_module = GlobalSAModule(MLP([256 + 3, 256, 512, 1024]))

        self.lin1 = Lin(2048, 1024)
        self.lin2 = Lin(1024, 512)
        self.lin3 = Lin(512, 256)
        self.lin4 = Lin(256, 8)

    # def forward(self, input1, input2):
    #     sa0_out1 = (input1.x, input1.pos, input1.batch)
    #     sa1_out1 = self.sa1_module(*sa0_out1)
    #     sa2_out1 = self.sa2_module(*sa1_out1)
    #     sa3_out1 = self.sa3_module(*sa2_out1)
    #     x1, pos1, batch1 = sa3_out1
    #
    #     sa0_out2 = (input2.x, input2.pos, input2.batch)
    #     sa1_out2 = self.sa1_module(*sa0_out2)
    #     sa2_out2 = self.sa2_module(*sa1_out2)
    #     sa3_out2 = self.sa3_module(*sa2_out2)
    #     x2, pos2, batch2 = sa3_out2

    def forward(self, data):

        sa0_out1 = (data.x, data.input1, data.batch1)
        sa1_out1 = self.sa1_module(*sa0_out1)
        sa2_out1 = self.sa2_module(*sa1_out1)
        sa3_out1 = self.sa3_module(*sa2_out1)
        x1, pos1, batch1 = sa3_out1

        sa0_out2 = (data.x, data.input2, data.batch2)
        sa1_out2 = self.sa1_module(*sa0_out2)
        sa2_out2 = self.sa2_module(*sa1_out2)
        sa3_out2 = self.sa3_module(*sa2_out2)
        x2, pos2, batch2 = sa3_out2

        x = x1 + x2
        x = torch.cat([x1, x2], dim=1)

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin4(x)

        return F.log_softmax(x, dim=-1)


def train(train_loader):

    model.train()

    for batch in train_loader:

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
    for data in loader:

        midpoints, means, medians, centers = get_splits(data)
        blocks = create_blocks(data, medians)

        c1 = np.random.choice(8)
        while blocks[c1].pos.size()[0] == 0:
            c1 = np.random.choice(8)

        c2 = np.random.choice(8)
        while blocks[c2].pos.size()[0] == 0 or c1 == c2:
            c2 = np.random.choice(8)

        blocks[c2].y = torch.tensor([c2])

        data1 = blocks[c1].to(device)
        data2 = blocks[c2].to(device)

        with torch.no_grad():
            pred = model(data1, data2).max(1)[1]
        correct += pred.eq(data2.y).sum().item()

    return correct / len(loader.dataset)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../data/ModelNet10', help='Path to data')
    parser.add_argument('--sample_points', type=int, default=10000)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--n_workers', type=int, default=6)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--modelnet_version', type=str, default='10')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--load_model', action='store_true', help='Load a previous model')
    parser.add_argument('--model_path', default='../checkpoints/pointnet_pretext_160_2019-11-05.pt', help='Path to load model from')
    args = parser.parse_args()

    # path = osp.join(
    #     osp.dirname(osp.realpath(__file__)), args.data_path)
    # pre_transform, transform = T.NormalizeScale(), T.SamplePoints(args.sample_points)
    # train_dataset = ModelNet(path, args.modelnet_version, True, transform, pre_transform)
    # test_dataset = ModelNet(path, args.modelnet_version, False, transform, pre_transform)
    # train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
    #                           num_workers=args.n_workers)
    # test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
    #                          num_workers=args.n_workers)
    #
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    path = osp.join(
        osp.dirname(osp.realpath(__file__)), args.data_path)
    pre_transform = T.Compose([
        T.NormalizeScale(),
        T.SamplePoints(args.sample_points)
    ])
    transform = None
    train_dataset = BlockModelNet(path, args.modelnet_version, True, transform, pre_transform)
    test_dataset = BlockModelNet(path, args.modelnet_version, False, transform, pre_transform)
    train_loader = BlockDataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.n_workers)
    test_loader = BlockDataLoader(test_dataset, batch_size=args.batch_size, shuffle=False,
                             num_workers=args.n_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.load_model:
        model = torch.load(args.model_path).to(device)
    else:
        model = Net().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(1, args.n_epochs + 1):
        print('Training epoch {}'.format(epoch))
        train(train_loader)
        test_acc = test(test_loader)
        print('Epoch: {:03d}, Test: {:.4f}'.format(epoch, test_acc))

        if epoch % 5 == 0:
            torch.save(model, f'../checkpoints/pointnet_pretext_{epoch}_{datetime.date.today()}.pt')
            print('Saving model...')

        print('-'*15)
