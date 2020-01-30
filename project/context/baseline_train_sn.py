import sys
import argparse
import os.path as osp
sys.path.insert(1, '..')

#Torch Imports
import torch
import torch.nn.functional as F
#from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#Internal Imports
from models import PointNetSeg
from context import SNTrainer
from utils.datasets import ShapeNet
from dataset_splits import process_pickle_bysplit

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../data/ShapeNet', help='Path to data')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--n_workers', type=int, default=0)
    parser.add_argument('--n_epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--categories', nargs='*', type=int, action='store', default=None,
                        help='List of categories to use for training')
    parser.add_argument('--load_model', action='store_true', help='Load a previous model')
    parser.add_argument('--model_name', default='pointnet_pretext_160_2019-11-05.pt', help='Path to load model from')
    parser.add_argument('--checkpoints_path', default='saved_models', help='Path to load model from')
    parser.add_argument('--logs_path', default='runs/starter', help='Path to load model from')
    parser.add_argument('--processed_name', type=str, default='processed', help='Path to load model from')
    parser.add_argument('--load_subset', action='store_true', help='Load a previous model')
    parser.add_argument('--subset_pct', type=int, default=100)
    args = parser.parse_args()

    print("=====================================")
    print("SHAPENET BASELINE  ")
    print("=====================================")

    path = osp.join(osp.dirname(osp.realpath(__file__)), f'{args.data_path}')

    if args.load_subset:
        train_dict = process_pickle_bysplit(f'dataset_splits/ShapeNet16_TrainingSplits.pickle', args.subset_pct)
    else:
        train_dict = None

    pre_transform = T.NormalizeScale()

    transform = T.Compose([
        T.RandomTranslate(0.01),
        T.RandomRotate(15, axis=0),
        T.RandomRotate(15, axis=1),
        T.RandomRotate(15, axis=2)
    ])

    train_dataset = ShapeNet(path, None, True, transform, pre_transform,
                             processed_name=args.processed_name,
                             train_dict=train_dict)
    test_dataset = ShapeNet(path, None, False, transform, pre_transform,
                            processed_name=args.processed_name)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.n_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=args.n_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.load_model:
        model = torch.load(osp.join(args.checkpoints_path, args.model_name)).to(device)
    elif args.categories is None:
        model = PointNetSeg(train_dataset.num_classes).to(device)
    else:
        model = PointNetSeg(len(args.categories)).to(device)

    # if args.categories is None:
    #     model = PointNetSeg(train_dataset.num_classes).to(device)
    # else:
    #     model = PointNetSeg(len(args.categories)).to(device)

    simple_trainer = SNTrainer(model, log_dir=args.logs_path, lr=args.lr, predictor="y")
    # if args.load_model:
    #     simple_trainer.load_path(osp.join(args.checkpoints_path, args.model_name))
    #test_acc = simple_trainer.test(test_loader, verbose=True)
    for epoch in range(1, args.n_epochs + 1):
        print(f'Training epoch {epoch}')
        train_acc = simple_trainer.train(train_loader, verbose=True)
        test_acc = simple_trainer.test(test_loader, verbose=True)
        if epoch % 10 == 0:
            #simple_trainer.save(f"{args.checkpoints_path}/baseline_modelnet{args.modelnet_version}_ckpt_{epoch}.pt")
            torch.save(model, f"{args.checkpoints_path}/baseline_shapenet_{args.subset_pct}pct_ckpt_{epoch}.pt")

    simple_trainer.cleanup()
