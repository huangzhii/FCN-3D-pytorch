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


class DentalClassSegBase(data.Dataset):

    class_names = np.array([
        'bg',
        '1',
        '2',
        '3'
    ])
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])

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
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        # load label
        lbl_file = data_file['lbl']
        lbl = PIL.Image.open(lbl_file)
        lbl = np.array(lbl, dtype=np.int32)
        # print("get unique values of lbl", np.unique(lbl))
        if self._transform:
            return self.transform(img, lbl)
        else:
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


class DentalClassSegValidate(DentalClassSegBase):

    def __init__(self, root, split='validation', transform=False):
        super(DentalClassSegValidate, self).__init__(
            root, split=split, transform=transform)
        dataset_dir = self.root
        imgsets_file = osp.join(dataset_dir, 'test2.txt')
        for did in open(imgsets_file):
            did = did.strip('.png\n')
            img_file = osp.join(dataset_dir, 'images/%s.png' % did)
            lbl_file = osp.join(dataset_dir, 'labels/%s.png' % did)
            self.files['validation'].append({'img': img_file, 'lbl': lbl_file})

class DentalClassSeg(DentalClassSegBase):
    def __init__(self, root, split='train', transform=False):
        super(DentalClassSeg, self).__init__(root, split=split, transform=transform)
