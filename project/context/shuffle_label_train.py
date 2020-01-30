import sys
import os.path as osp
import numpy as np
sys.path.insert(1, '..')

#Torch Imports
import torch
import torch.nn.functional as F
from torch_geometric.datasets import ModelNet
import torch_geometric.transforms as T
from torch_geometric.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

#Internal Imports
from pretext_transforms import BandRotate, ShuffleLabels
from models import PointNet
from context_utils import Trainer

if __name__ == '__main__':

  path = osp.join(
      osp.dirname(osp.realpath(__file__)), '..', 'data/ModelNet10')


  print("=====================================")
  print("MODELNET10 SHUFFLED LABELS  ")
  print("=====================================")


  mapper = {}
  x = list(range(0,10))
  np.random.shuffle(x)
  for i in range(0,10):
      mapper[i] = x[i]

  pre_transform = T.NormalizeScale()
  transform =  T.Compose((T.SamplePoints(1024), ShuffleLabels(mapper)))

  train_dataset = ModelNet(path, '10', True, transform, pre_transform)
  test_dataset = ModelNet(path, '10', False, transform, pre_transform)
  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                            num_workers=0)
  test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True,
                           num_workers=0)

  model = PointNet(10)
  simple_trainer = Trainer(model, log_dir="runs/shuffled_labels", predictor="y")
  test_acc = simple_trainer.test(test_loader, verbose=True)
  for epoch in range(0, 20):
      train_acc = simple_trainer.train(train_loader, verbose=True)
      test_acc = simple_trainer.test(test_loader, verbose=True)
      if epoch % 3 == 0:
          simple_trainer.save("shuffled_labels_ckpt_{}".format(epoch))

  simple_trainer.cleanup()
