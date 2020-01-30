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
from pretext_transforms import BandRotate
from models import PointNet
from context import Trainer
from utils.datasets import ModelNet
from dataset_splits import process_pickle_bysplit

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='../data/ModelNet', help='Path to data')
    parser.add_argument('--sample_points', type=int, default=1024)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--n_workers', type=int, default=0)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--modelnet_version', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--load_model', action='store_true', help='Load a previous model')
    parser.add_argument('--checkpoints_path', default='saved_models', help='Path to load model from')
    parser.add_argument('--logs_path', default='runs/starter', help='Path to load model from')
    parser.add_argument('--model_name', default='pointnet_pretext_160_2019-11-05.pt', help='Path to load model from')
    parser.add_argument('--processed_name', type=str, default='processed', help='Path to load model from')
    parser.add_argument('--load_subset', action='store_true', help='Load a previous model')
    parser.add_argument('--subset_pct', type=int, default=100)
    args = parser.parse_args()

    path = osp.join(
        osp.dirname(osp.realpath(__file__)), f'{args.data_path}{args.modelnet_version}')

    print("=====================================")
    print("=====================================")
    print("MODELNET10 FROM PRETRAINED WEIGHTS   ")
    print("=====================================")
    print("=====================================")

    if args.load_subset:
        train_dict = process_pickle_bysplit(f'dataset_splits/ModelNet{str(args.modelnet_version)}_TrainingSplits_New.pickle', args.subset_pct)
    else:
        train_dict = None

    pre_transform = T.NormalizeScale()
    transform =  T.SamplePoints(args.sample_points)

    train_dataset = ModelNet(path, str(args.modelnet_version), True, transform,
                             pre_transform, processed_name=args.processed_name,
                             train_dict=train_dict)
    test_dataset = ModelNet(path, str(args.modelnet_version), False, transform,
                            pre_transform, processed_name=args.processed_name)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=args.n_workers)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size,
                             shuffle=True, num_workers=args.n_workers)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = PointNet(args.modelnet_version).to(device)

    # if args.load_model:
    #     state_dict = torch.load(osp.join(args.checkpoints_path, args.model_name))
    #     model.load_state_dict(state_dict)
    #     model = torch.load(osp.join(args.checkpoints_path, args.model_name)).to(device)
    # else:
    #     model = PointNet(args.modelnet_version).to(device)

    #DATA PREPARATION
    # pre_transform = T.NormalizeScale()
    # transform =  T.SamplePoints(1024)
    # train_dataset = ModelNet(path, '10', True, transform, pre_transform)
    # test_dataset = ModelNet(path, '10', False, transform, pre_transform)
    #
    # train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
    #                         num_workers=0)
    # test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True,
    #                        num_workers=0)


    # train_each = 10
    save_every = 5
    # unfreeze_after = 0
    #loadfrom = "runs/band_all/band_all_ckpt_45"
    #
    #
    # for x in loading.keys():
    #   print(x)

    '''
    RESET LINEAR - FULL TRAIN
    '''
    print("RESET LINEAR - FULL TRAIN")

    simple_trainer = Trainer(model, log_dir=args.logs_path, predictor="y")
    simple_trainer.load_path_for_transfer(osp.join(args.checkpoints_path, args.model_name))

    # load_state = torch.load(osp.join(args.checkpoints_path, args.model_name))
    # remove = ["lin1.bias", "lin1.weight","lin2.bias", "lin2.weight", "lin3.bias", "lin3.weight"]
    # for x in list(load_state.keys()):
    #     if x in remove:
    #         del load_state[x]
    #
    # own_state = simple_trainer.model.state_dict()
    # for name, param in load_state.items():
    #     if name not in own_state:
    #         continue
    #     own_state[name].copy_(param)
    # simple_trainer.model.fine_tuning_train_reset()

    test_acc = simple_trainer.test(test_loader, verbose=True)

    for epoch in range(0, args.n_epochs):
        train_acc = simple_trainer.train(train_loader, verbose=True)
        test_acc = simple_trainer.test(test_loader, verbose=True)
        if epoch % save_every == 0:
            #simple_trainer.save("pre_trained_ckpt_{}".format(epoch))
            torch.save(model, f"{args.checkpoints_path}/finetuning_modelnet{args.modelnet_version}_{args.subset_pct}pct_ckpt_{epoch}.pt")

    simple_trainer.cleanup()

    '''
    RESET LINEAR - FREEZE POINTNET
    '''
    # print("RESET LINEAR - FREEZE POINTNET")
    # model = PointNet(49)
    # simple_trainer = Trainer(model, log_dir="pre_trained_band_all/freeze_reset", predictor="y")
    # simple_trainer.load_path(loadfrom)
    # simple_trainer.model.fine_tuning_freeze_reset()
    # test_acc = simple_trainer.test(test_loader, verbose=True)
    #
    # for epoch in range(0, train_each):
    #     train_acc = simple_trainer.train(train_loader, verbose=True)
    #     test_acc = simple_trainer.test(test_loader, verbose=True)
    #     if epoch == unfreeze_after:
    #         simple_trainer.model.unfreeze()
    #     if epoch % save_every == 0:
    #         simple_trainer.save("pre_trained_ckpt_{}".format(epoch))
    #
    # simple_trainer.cleanup()

    '''
    TRAIN LINEAR - FREEZE POINTNET
    print("TRAIN LINEAR - FREEZE POINTNET")
    model = PointNet(10)
    simple_trainer = Trainer(model, log_dir="pre_trained_band_all/freeze_train", predictor="y")
    simple_trainer.load_path(loadfrom)
    simple_trainer.model.fine_tuning_freeze_train()
    test_acc = simple_trainer.test(test_loader, verbose=True)

    for epoch in range(0, train_each):
      train_acc = simple_trainer.train(train_loader, verbose=True)
      test_acc = simple_trainer.test(test_loader, verbose=True)

      if epoch == unfreeze_after:
          simple_trainer.model.unfreeze()
      if epoch % save_every == 0:
          simple_trainer.save("pre_trained_ckpt_{}".format(epoch))

    simple_trainer.cleanup()
    '''

    '''
    FULL TRAIN
    print("Full Train")
    model = PointNet(10)
    simple_trainer = Trainer(model, log_dir="pre_trained_band_all/whole", predictor="y")
    simple_trainer.load_path(loadfrom)
    test_acc = simple_trainer.test(test_loader, verbose=True)

    for epoch in range(0, train_each):
      train_acc = simple_trainer.train(train_loader, verbose=True)
      test_acc = simple_trainer.test(test_loader, verbose=True)
      if epoch % save_every == 0:
          simple_trainer.save("pre_trained_ckpt_{}".format(epoch))

    simple_trainer.cleanup()
    '''
