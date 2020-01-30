
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

if __name__ == '__main__':
  path = osp.join(
      osp.dirname(osp.realpath(__file__)), '..', 'data/ModelNet10')

  pre_transform = T.NormalizeScale()
  degree_centers = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

  degree_noise = 5
  translate_noise = .005
  print("=====================================")
  print("NAIVE SINGLE PREDICTION BAND_ROTATE"       )
  print("Degree Noise:", degree_noise)
  print("Translate Noise:", translate_noise)
  print("=====================================")

  transform =  T.Compose([T.SamplePoints(1024),
    T.RandomTranslate(translate_noise), BandRotate(degree_centers, degree_noise),
    T.RandomTranslate(translate_noise), BandRotate(degree_centers, degree_noise,axis=1),
    T.RandomTranslate(translate_noise)])


  train_dataset = ModelNet(path, '10', True, transform, pre_transform)
  test_dataset = ModelNet(path, '10', False, transform, pre_transform)
  train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                            num_workers=0)
  test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True,
                           num_workers=0)

  model = PointNet(len(degree_centers))
  simple_trainer = Trainer(model, log_dir="runs/testing", predictor="band_x")

  for epoch in range(1, 50):
      train_acc = simple_trainer.train(train_loader, verbose=True)
      test_acc = simple_trainer.test(test_loader, verbose=True)
      if epoch % 10 == 0:
          simple_trainer.save("naive_rotation_ckpt_{}".format(epoch))

  simple_trainer.cleanup()
