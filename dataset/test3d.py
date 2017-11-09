#!/usr/bin/env python

import collections
import os.path as osp
import time

import numpy as np
import PIL.Image
import scipy.io
import torch
from torch.utils import data
import matplotlib.pyplot as plt
import pylab

def happyprint(string, obj):
    # print(string, obj)
    return


class Data3Dbase(data.Dataset):

    class_names = np.array([
        'bg',
        'obj'
    ])

    def __init__(self, root, split='train', transform=False):
        self.root = root
        self.split = split
        self._transform = transform

        dataset_dir = self.root
        self.files = collections.defaultdict(list)
        imgsets_file = osp.join(dataset_dir, 'train2.txt')
        for did in open(imgsets_file):
            did = did.strip('.png\n')
            img_file = osp.join(dataset_dir, 'images/%s.png' % did)
            lbl_file = osp.join(dataset_dir, 'labels/%s.png' % did)
            self.files[split].append({'img': img_file, 'lbl': lbl_file})

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        img_file = data_file['img']
        lbl_file = data_file['lbl']

        img = np.zeros((60,60,60))
        lbl = np.zeros((60,60,60))
        for i in range(20,40):
            for j in range(20,40):
                for k in range(20,40):
                    img[i,j,k] = 255
                    lbl[i,j,k] = 1

        for i in range(0,60):
            for j in range(0,60):
                for k in range(0,60):
                    img[i,j,k] = np.random.normal(img[i,j,k], 80, 1)
                    if img[i,j,k] < 0:
                        img[i,j,k] = 0
                    if img[i,j,k] > 255:
                        img[i,j,k] = 255
        img = np.expand_dims(img, axis=0)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        # plt.imshow(img[:,:,50])
        # plt.title('image')
        # plt.pause(1)
        return img, lbl

    def transform(self, img, lbl):
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        lbl = lbl.numpy()
        return img, lbl


class Data3DVal(Data3Dbase):

    def __init__(self, root, split='validation', transform=False):
        super().__init__(
            root, split=split, transform=transform)
        dataset_dir = self.root
        imgsets_file = osp.join(dataset_dir, 'test2.txt')
        for did in open(imgsets_file):
            did = did.strip('.png\n')
            img_file = osp.join(dataset_dir, 'images/%s.png' % did)
            lbl_file = osp.join(dataset_dir, 'labels/%s.png' % did)
            self.files['validation'].append({'img': img_file, 'lbl': lbl_file})

class Data3D(Data3Dbase):
    def __init__(self, root, split='train', transform=False):
        super().__init__(root, split=split, transform=transform)
