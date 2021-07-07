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
import argparse
import numpy as np
from PIL import Image
import shutil
import matplotlib
import matplotlib.pyplot as plt

from lipdata import LIPWithClass, LIP
from pspnet import PSPNet

# Initialize parameters
epochs = 40
start_lr = 0.0001
models_path = 'C:\Users\hp\Desktop\pspnet\model_path'
snapshot = None # to use the pretrained mode, give the link to this variable
data_path = 'C:\Users\hp\Desktop\pspnet\data_path'
alpha = 0.4
batch_size = 8
power = 0.9


def build_network(snapshot, models):
    epoch = 0
    net = models
    net = nn.DataParallel(net)
    if snapshot is not None:
        _, epoch = os.path.basename(snapshot).split('_')
        epoch = int(epoch)
        net.load_state_dict(torch.load(snapshot))
        print("Snapshot for epoch {} loaded from {}".format(epoch, snapshot))
    net = net.cuda()
    return net, epoch


def get_transform():
    transform_image_list = [
        transforms.Resize((256, 256), 3),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]

    transform_gt_list = [
        transforms.Resize((256, 256), 0),
        transforms.Lambda(lambda img: np.asarray(img, dtype=np.uint8)),
    ]

    data_transforms = {
        'img': transforms.Compose(transform_image_list),
        'gt': transforms.Compose(transform_gt_list),
    }
    return data_transforms


def get_dataloader():
    '''
        To follow this training routine you need a DataLoader that yields the tuples of the following format:
        (Bx3xHxW FloatTensor x, BxHxW LongTensor y, BxN LongTensor y_cls) where
        x - batch of input images,
        y - batch of groung truth seg maps,
        y_cls - batch of 1D tensors of dimensionality N: N total number of classes,
        y_cls[i, T] = 1 if class T is present in image i, 0 otherwise
    '''
    data_transform = get_transform()
    train_loader = DataLoader(LIPWithClass(root=data_path, transform=data_transform['img'],
                                           gt_transform=data_transform['gt']),
                              batch_size= batch_size,
                              shuffle=True,
                              )
    val_loader = DataLoader(LIP(root = data_path, train=False, transform=data_transform['img'],
                                gt_transform=data_transform['gt']),
                            batch_size= batch_size,
                            shuffle=False,
                            )
    return train_loader, val_loader

if __name__ == '__main__':
    train_loader, val_loader = get_dataloader()
    model = PSPNet(sizes=(1, 2, 3, 6), psp_size=2048, deep_features_size=1024)
    net, starting_epoch = build_network(snapshot, model)
    
    #scheduler = MultiStepLR(optimizer, milestones=milestones)

    for epoch in range(1+starting_epoch, 1+epochs):
        seg_criterion = nn.NLLLoss(weight=None)
        cls_criterion = nn.BCEWithLogitsLoss(weight=None)
        epoch_losses = []
        val_loss = []
        lr = start_lr * ((1 - epoch/epochs)**power)
        optimizer = optim.Adam(net.parameters(), lr=lr)
        net.train()

        for count, (x, y, y_cls) in enumerate(train_loader):
            # input data
            x, y, y_cls = x.cuda(), y.cuda().long(), y_cls.cuda().float()
            # forward
            out, out_cls = net(x)
            seg_loss, cls_loss = seg_criterion(out, y), cls_criterion(out_cls, y_cls)
            loss = seg_loss + alpha * cls_loss
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
            # print
            if (count % 20 ==0):      
                status = '[{0}] step = {1}/{2}, loss = {3:0.4f} avg = {4:0.4f}, LR = {5:0.7f}'.format(
                    epoch, count, len(train_loader),
                    loss.item(), np.mean(epoch_losses), lr)
                print(status)
        #scheduler.step()
        torch.save(net.state_dict(), os.path.join(models_path, '_'.join(["PSPNet", str(epoch)])))

    
        net.eval()
        with torch.no_grad():
            for count, (x, y, y_cls) in enumerate(val_loader):
            # input data
                x, y, y_cls = x.cuda(), y.cuda().long(), y_cls.cuda().float()
                # forward
                out, out_cls = net(x)
                seg_loss, cls_loss = seg_criterion(out, y), cls_criterion(out_cls, y_cls)
                loss = seg_loss + alpha * cls_loss
                val_loss.append(loss.item())
        print ("val_loss:", np.mean(val_loss))

    torch.save(net.state_dict(), os.path.join(models_path, '_'.join(["PSPNet", 'last'])))
