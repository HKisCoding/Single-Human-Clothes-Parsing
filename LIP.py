import math
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch import optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import transforms
from torchsummary import summary
from collections import OrderedDict
from torch.utils import model_zoo

import os
import cv2
import argparse
import numpy as np
from PIL import Image
import shutil
import matplotlib
import matplotlib.pyplot as plt

class LIP(data.Dataset):

    def __init__(self, root, train=True, transform=None, gt_transform=None ):
        self.root = root
        self.transform = transform
        self.gt_transform = gt_transform
        self.train = train  # trainval set or val set

        if self.train:
            self.train_image_path, self.train_gt_path = self.read_labeled_image_list(os.path.join(root, 'train'))
        else:
            self.val_image_path, self.val_gt_path = self.read_labeled_image_list(os.path.join(root, 'val'))
            # self.test_image_path = self.read_labeled_image_list(osp.join(root, 'test'))

    def __getitem__(self, index):
        if self.train:
            img, gt = self.get_a_sample(self.train_image_path, self.train_gt_path, index)
        else:
            img, gt = self.get_a_sample(self.val_image_path, self.val_gt_path, index)
        return img, gt

    def __len__(self):
        if self.train:
            return len(self.train_image_path)
        else:
            return len(self.val_image_path)

    def get_a_sample(self, image_path, gt_path, index):
        # get PIL Image
        img = Image.open(image_path[index])  # .resize((512,512),resample=Image.BICUBIC)
        if len(img.getbands()) != 3:
            img = img.convert('RGB')
        gt = Image.open(gt_path[index])  # .resize((30,30),resample=Image.NEAREST)
        if len(gt.getbands()) != 1:
            gt = gt.convert('L')

        if self.transform is not None:
            img = self.transform(img)
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)
        return img, gt

    def read_labeled_image_list(self, data_dir):
        # return img path list and groundtruth path list
        f = open(os.path.join(data_dir, 'id.txt' ), 'r')
        image_path = []
        gt_path = []
        for line in f:
            image = line.strip("\n")
            if self.train:
                image_path.append(os.path.join(data_dir, 'image', image + ".jpg"))
                gt_path.append(os.path.join(data_dir, 'gt', image + ".png"))
            else:
                image_path.append(os.path.join(data_dir, 'image', image + ".jpg"))
                gt_path.append(os.path.join(data_dir, 'gt', image + ".png"))
        return image_path, gt_path


class LIPWithClass(LIP):

    def __init__(self, root, num_cls=20, train=True, transform=None, gt_transform=None):
        LIP.__init__(self, root, train, transform, gt_transform)
        self.num_cls = num_cls

    def __getitem__(self, index):
        if self.train:
            img, gt, gt_cls = self.get_a_sample(self.train_image_path, self.train_gt_path, index)
        else:
            img, gt, gt_cls = self.get_a_sample(self.val_image_path, self.val_gt_path, index)
        return img, gt, gt_cls

    def get_a_sample(self, image_path, gt_path, index):
        # get PIL Image
        # gt_cls - batch of 1D tensors of dimensionality N: N total number of classes,
        # gt_cls[i, T] = 1 if class T is present in image i, 0 otherwise
        img = Image.open(image_path[index])
        if len(img.getbands()) != 3:
            img = img.convert('RGB')
        gt = Image.open(gt_path[index])
        if len(gt.getbands()) != 1:
            gt = gt.convert('L')
        # compute gt_cls
        gt_np = np.asarray(gt, dtype=np.uint8)
        gt_cls, _ = np.histogram(gt_np, bins=self.num_cls, range=(-0.5, self.num_cls-0.5), )
        gt_cls = np.asarray(np.asarray(gt_cls, dtype=np.bool), dtype=np.uint8)
        if self.transform is not None:
            img = self.transform(img)
        if self.gt_transform is not None:
            gt = self.gt_transform(gt)

        return img, gt, gt_cls