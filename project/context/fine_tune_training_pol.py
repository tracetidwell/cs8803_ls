import sys
import os.path as osp
sys.path.insert(1, '..')

#Torch Imports
import torch
import torch.nn.functional as F
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#Internal Imports
from pretext_transforms import BandRotate
from models import PointNet
from context_utils import Trainer


#Internal Imports
from pretext_transforms import BandRotate
from models import PointNet
from context import Trainer
from utils.datasets import ModelNet
from dataset_splits import process_pickle_bysplit


if __name__ == '__main__':

  path = osp.join(
      osp.dirname(osp.realpath(__file__)), '..', 'data/ModelNet10')

  print("=====================================")
  print("=====================================")
  print("MODELNET10 FROM PRETRAINED WEIGHTS   ")
  print("=====================================")
  print("=====================================")

  train_dict = process_pickle_bysplit(f'../dataset_splits/ModelNet10_TrainingSplits.pickle', 50)

  pre_transform = T.NormalizeScale()
  transform =  T.SamplePoints(1024)

  train_dataset = ModelNet(path, '10', True, transform, pre_transform, processed_name="subset_10", train_dict=train_dict)
  test_dataset = ModelNet(path, '10', False, transform, pre_transform, processed_name="subset_10")
  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,num_workers=0)
  test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True, num_workers=0)


  train_each = 10
  save_every = 5
  unfreeze_after = 0
  loadfrom = "runs/band_all/band_all_ckpt_45"

  loading = torch.load(loadfrom)
  remove = ["lin1.bias", "lin1.weight","lin2.bias", "lin2.weight", "lin3.bias", "lin3.weight"]
  for x in list(loading.keys()):
      if x in remove:
          del loading[x]

  for x in loading.keys():
      print(x)

  '''
  RESET LINEAR - FULL TRAIN
  '''
  print("RESET LINEAR - FULL TRAIN")
  model = PointNet(10)
  simple_trainer = Trainer(model, log_dir="pre_trained_band_all_50pct_2/baseline", predictor="y")
  '''
  own_state = simple_trainer.model.state_dict()
  for name, param in loading.items():
      if name not in own_state:
           continue
      own_state[name].copy_(param)
  simple_trainer.model.fine_tuning_train_reset()
  '''
  test_acc = simple_trainer.test(test_loader, verbose=True)

  for epoch in range(0, train_each):
      train_acc = simple_trainer.train(train_loader, verbose=True)
      test_acc = simple_trainer.test(test_loader, verbose=True)
      if epoch % save_every == 0:
          simple_trainer.save("pre_trained_ckpt_{}".format(epoch))

  simple_trainer.cleanup()
  '''
  RESET LINEAR - FREEZE POINTNET
  '''
  print("RESET LINEAR - FREEZE POINTNET")
  model = PointNet(49)
  simple_trainer = Trainer(model, log_dir="pre_trained_band_all/freeze_reset", predictor="y")
  simple_trainer.load_path(loadfrom)
  simple_trainer.model.fine_tuning_freeze_reset()
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
