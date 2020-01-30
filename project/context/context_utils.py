import sys
import os.path as osp
sys.path.insert(1, '..')

#Torch Imports
import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

import torch_geometric.transforms as T
from torch_geometric.datasets import ModelNet
from torch_geometric.data import DataLoader
from torch_geometric.utils import intersection_and_union as i_and_u

#Sklearn Imports
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import numpy as np

#Internal Imports
from pretext_transforms import BandRotate
from models import PointNet


class Trainer():
    def __init__(self, model, log_dir=None, predictor='y', lr=0.001):
        self.device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.target_name = predictor

        self.step = 0
        self.log_dir = log_dir
        self.writer = SummaryWriter(self.log_dir)


    def load_path(self, path):

        self.model.load_state_dict(torch.load(path))


    def load_path_for_transfer(self, path):

        load_state = torch.load(path)
        remove = ["lin1.bias", "lin1.weight","lin2.bias", "lin2.weight", "lin3.bias", "lin3.weight"]
        for key in list(load_state.keys()):
            if key in remove:
                del load_state[key]

        own_state = self.model.state_dict()
        for name, param in load_state.items():
            if name not in own_state:
                continue
            own_state[name].copy_(param)
        self.model.fine_tuning_train_reset()


    def load_same(self, name):
        path = osp.join(self.log_dir, name)
        self.model.load_state_dict(torch.load(path))


    def save(self, path):
        #path = osp.join(self.log_dir, name)
        torch.save(self.model.state_dict(), path)


    def train(self, loader, verbose=False):

        self.model.train()
        tot_correct = 0

        for data in loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            forward = self.model(data)
            loss = F.nll_loss(forward, getattr(data, self.target_name))
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                pred = forward.max(1)[1]
                correct = pred.eq(getattr(data, self.target_name)).sum().item()
                tot_correct += correct
                self.step += 1

                if self.step % 10 == 0:
                    self.writer.add_scalar('Loss/Train_batches', loss, self.step)
                    self.writer.add_scalar('Accuracy/Train_batches', correct / len(getattr(data, self.target_name)), self.step)
                    self.writer.flush()

                    if verbose:
                        print('Step: {:03d}, Train: {:.4f}'.format(self.step, correct / len(getattr(data, self.target_name))))

        train_acc = tot_correct / len(loader.dataset)
        self.writer.add_scalar('Accuracy/Train', train_acc, self.step)
        self.writer.flush()

        return train_acc


    def test(self, loader, verbose):
      self.model.eval()

      correct = 0
      for data in loader:
          data = data.to(self.device)
          with torch.no_grad():
              pred = self.model(data).max(1)[1]
          correct += pred.eq(getattr(data, self.target_name)).sum().item()

      test_acc = correct / len(loader.dataset)
      self.writer.add_scalar('Accuracy/Test', test_acc, self.step)
      self.writer.flush()

      if verbose:
        print('STEP: {:03d}, TEST: {:.4f}'.format(self.step, test_acc))

      return test_acc


    def cleanup(self):
      self.writer.close()


class SNTrainer():
    def __init__(self, model, log_dir=None, predictor='y', lr=0.001):
        self.device = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.target_name = predictor

        self.step = 0
        self.log_dir = log_dir
        self.writer = SummaryWriter(self.log_dir)


    def load_path(self, path):

        self.model.load_state_dict(torch.load(path))


    def load_path_for_transfer(self, path):

        load_state = torch.load(path)
        remove = ["lin1.bias", "lin1.weight","lin2.bias", "lin2.weight", "lin3.bias", "lin3.weight"]
        for key in list(load_state.keys()):
            if key in remove:
                del load_state[key]

        own_state = self.model.state_dict()
        for name, param in load_state.items():
            if name not in own_state:
                continue
            own_state[name].copy_(param)
        self.model.fine_tuning_train_reset()


    def load_same(self, name):
        path = osp.join(self.log_dir, name)
        self.model.load_state_dict(torch.load(path))


    def save(self, path):
        #path = osp.join(self.log_dir, name)
        torch.save(self.model.state_dict(), path)


    def train(self, loader, verbose=False):

        self.model.train()
        tot_correct = total = 0

        for data in loader:
            data = data.to(self.device)
            self.optimizer.zero_grad()
            forward = self.model(data)
            loss = F.nll_loss(forward, getattr(data, self.target_name))
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                pred = forward.max(1)[1]
                correct = pred.eq(getattr(data, self.target_name)).sum().item()
                tot_correct += correct
                total += data.num_nodes
                self.step += 1

                if self.step % 10 == 0:
                    self.writer.add_scalar('Loss/Train_batches', loss, self.step)
                    self.writer.add_scalar('Accuracy/Train_batches', correct / data.num_nodes, self.step)
                    self.writer.flush()

                    if verbose:
                        print('Step: {:03d}, Train Acc: {:.4f}'.format(self.step, correct / len(getattr(data, self.target_name))))

        train_acc = tot_correct / total#len(loader.dataset)
        self.writer.add_scalar('Accuracy/Train', train_acc, self.step)
        self.writer.flush()

        return train_acc


    def test(self, loader, verbose):

        self.model.eval()
        correct = total = 0
        intersections, unions, categories = [], [], []

        for data in loader:
            data = data.to(self.device)
            with torch.no_grad():
                pred = self.model(data).max(1)[1]
            correct += pred.eq(getattr(data, self.target_name)).sum().item()
            total += data.num_nodes
            i, u = i_and_u(pred, getattr(data, self.target_name), loader.dataset.num_classes, data.batch)
            intersections.append(i.to(torch.device('cpu')))
            unions.append(u.to(torch.device('cpu')))
            categories.append(data.category.to(torch.device('cpu')))

        category = torch.cat(categories, dim=0)
        intersection = torch.cat(intersections, dim=0)
        union = torch.cat(unions, dim=0)

        ious = [[] for _ in range(len(loader.dataset.categories))]
        for j in range(len(loader.dataset)):
            i = intersection[j, loader.dataset.y_mask[category[j]]]
            u = union[j, loader.dataset.y_mask[category[j]]]
            iou = i.to(torch.float) / u.to(torch.float)
            iou[torch.isnan(iou)] = 1
            ious[category[j]].append(iou.mean().item())

        for cat in range(len(loader.dataset.categories)):
            ious[cat] = torch.tensor(ious[cat]).mean().item()

        test_acc = correct / total
        iou = torch.tensor(ious).mean().item()
        self.writer.add_scalar('Accuracy/Test', test_acc, self.step)
        self.writer.add_scalar('IOU/Test', iou, self.step)
        self.writer.flush()

        if verbose:
            print('STEP: {:03d}, TEST ACC: {:.4f}'.format(self.step, test_acc))
            print('STEP: {:03d}, TEST IOU: {:.4f}'.format(self.step, iou))

        return test_acc, iou


    def cleanup(self):
        self.writer.close()


if __name__ == '__main__':

  path = osp.join(
      osp.dirname(osp.realpath(__file__)), '..', 'data/ModelNet10')


  print("=====================================")
  print("MODELNET10 BASELINE  ")
  print("=====================================")

  pre_transform = T.NormalizeScale()
  transform =  T.SamplePoints(1024)

  test_dataset = ModelNet(path, '10', False, transform, pre_transform)
  test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True,
                           num_workers=0)

  model = PointNet(10)
  loadfrom = "runs/shuffled_labels/shuffled_labels_ckpt_15"
  #loadfrom = "runs/testing/naive_rotation_ckpt_10"
  #loadfrom = "runs/band_all/band_all_ckpt_40"
  #loadfrom = "runs/band_all_9/band_all_ckpt_45"

  model.load_state_dict(torch.load(loadfrom))
  X = []
  Y = []
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  model.eval()
  for data in test_loader:

      x = data.pos.numpy().reshape(-1, 3072)
      y = data.y.numpy()

      data = data.to(device)

      z = model.embed(data).cpu().detach().numpy()
      #print(z.shape)

      X.append(x)
      Y.append(y)

  newX = np.concatenate(X, axis=0)
  newY = np.concatenate(Y, axis=0)

  #pca = PCA(n_components=500)
  #newX = pca.fit_transform(newX)

  tsne = TSNE(n_components=2, random_state=0)
  X_2d = tsne.fit_transform(newX)

  target_ids = range(10)
  plt.figure(figsize=(6, 5))
  colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
  target_labs = 'bathtub', 'bed', 'chair', 'desk', 'dresser', 'monitor', 'night_stand', 'sofa', 'table', 'toilet'
  for i, c, label in zip(target_ids, colors, target_labs):
      plt.scatter(X_2d[newY == i, 0], X_2d[newY== i, 1], c=c, label=label)
  #plt.legend()
  plt.legend(bbox_to_anchor=(.9, 1.))
  plt.show()
