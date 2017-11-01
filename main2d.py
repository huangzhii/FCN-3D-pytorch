# 10/31/2017 Halloween Day Init
# Author: Zhi Huang
# Purdue University
# IU Center for Neuroimaging

import matplotlib.pyplot as plt
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import dataset.kitti
#from dataset import KittiRoadTrain  # NOQA
#from dataset import KittiRoadValidate
import numpy as np


parser = argparse.ArgumentParser(description='FCN 2D')
parser.add_argument('--batch_size', type=int, default=1, metavar='N',
                    help='input batch size for training (default=1)')
parser.add_argument('--test_batch_size', type=int, default=1, metavar='N',
                    help='input batch size for testing (default=1)')
parser.add_argument('--epochs', type=int, default=1000, metavar='N',
                    help='input number of epochs to train')
parser.add_argument('--lr', type=float, default=1.0e-10, metavar='LR',
                    help='learning rate')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)

root = '/media/zhi/Drive3/KITTI/rwth_kitti_semantics_dataset'

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
train_loader = torch.utils.data.DataLoader(dataset.KittiRoadTrain(root, split='train', 
              transform=True), batch_size = args.batch_size, shuffle=True, **kwargs)
validation_loader = torch.utils.data.DataLoader(dataset.KittiRoadValidate(root, split='validation',
                transform=True), batch_size = args.test_batch_size, shuffle=False, **kwargs)

def train(epoch):
    for batch_idx, (data, target) in enumerate(train_loader):
        if args.cuda:
            data, target = data.cuda(), target.cuda()
#        plt.imshow(data[0,0,:,:].cpu().numpy())
#        plt.title('data')
#        plt.pause(1)
#        plt.imshow(target[0,:,:].cpu().numpy())
#        plt.title('target')
#        plt.pause(1)



def test():
    
    
    
    for data, target in validation_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        print("data dim", data.size())
        print("target dim", target.size())
#        plt.imshow(data[0,0,:,:].cpu().numpy())
#        plt.title('data')
#        plt.pause(1)
#        plt.imshow(target[0,:,:].cpu().numpy())
#        plt.title('target')
#        plt.pause(1)


for epoch in range(1, args.epochs+1):
#    train(epoch)
    test()