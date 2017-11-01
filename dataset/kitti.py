# 10/31/2017 Halloween Day Init
# Author: Zhi Huang
# Purdue University
# IU Center for Neuroimaging

import matplotlib.pyplot as plt
import torch
from torch.utils import data
import collections
import os.path as osp
import PIL.Image
import numpy as np


class KittiClass(data.Dataset):
    class_names = np.array([
            'Car',
            'Road',
            'Mark',
            'Building',
            'Sidewalk',
            'Tree/Bush',
            'Pole',
            'Sign',
            'Person',
            'Wall',
            'Sky',
            'Curb',
            'Grass/Dirt',
            'Void'
    ])

    def __init__(self, root, split='train', transform=False):
        self.root = root
        self.split = split
        self._transform = transform
        dataset_dir = self.root
        self.files = collections.defaultdict(list) # a dictionary-like object.
        imgsets_file = osp.join(dataset_dir, 'splits', 'train.txt')
        for filename in open(imgsets_file):
            filename = filename.strip('.pcd\n')
            img_file = osp.join(dataset_dir, 'images/%s.png' % filename)
            lbl_file = osp.join(dataset_dir, 'labels/%s.png' % filename)
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
        lbl = self.lbl_color_transform(lbl)
        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl
    
    def lbl_color_transform(self, lbl):
        row = lbl.shape[0]
        col = lbl.shape[1]
        lbl_mono = np.zeros((row, col))
        for x in range(0, row):
            for y in range(0, col):
                # print(lbl[x,y,:])
                if np.array_equal(lbl[x,y,:], [0,0,0]):
                    lbl_mono[x,y] = 0
                if np.array_equal(lbl[x,y,:], [0,0,255]):
                    lbl_mono[x,y] = 1
                if np.array_equal(lbl[x,y,:], [255,0,0]):
                    lbl_mono[x,y] = 2
                if np.array_equal(lbl[x,y,:], [255,255,0]):
                    lbl_mono[x,y] = 3
                if np.array_equal(lbl[x,y,:], [0,255,0]):
                    lbl_mono[x,y] = 4
                if np.array_equal(lbl[x,y,:], [255,0,255]):
                    lbl_mono[x,y] = 5
                if np.array_equal(lbl[x,y,:], [0,255,255]):
                    lbl_mono[x,y] = 6
                if np.array_equal(lbl[x,y,:], [255,0,153]):
                    lbl_mono[x,y] = 7
                if np.array_equal(lbl[x,y,:], [153,0,255]):
                    lbl_mono[x,y] = 8
                if np.array_equal(lbl[x,y,:], [0,153,255]):
                    lbl_mono[x,y] = 9
                if np.array_equal(lbl[x,y,:], [153,255,0]):
                    lbl_mono[x,y] = 10
                if np.array_equal(lbl[x,y,:], [255,153,0]):
                    lbl_mono[x,y] = 11
                if np.array_equal(lbl[x,y,:], [0,255,153]):
                    lbl_mono[x,y] = 12
                if np.array_equal(lbl[x,y,:], [0,153,153]):
                    lbl_mono[x,y] = 13
        return lbl_mono
    
    def transform(self, img, lbl):
        img = img[:,:, ::-1] # RGB -> BGR
        img = img.astype(np.float64)
        img = img.transpose(2,0,1)
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl
    
    def untransform(self, img, lbl):
        img = img.numpy()
        img = img.transpose(1,2,0)
        img = img.astype(np.uin8)
        img = img[:,:, ::-1]
        lbl = lbl.numpy()
        return img, lbl
    
class KittiRoadValidate(KittiClass):
    def __init__(self, root, split = 'validation', transform = False):
        super(KittiRoadValidate, self).__init__(root, split = split, transform = transform)
        dataset_dir = self.root
        imgsets_file = osp.join(dataset_dir, 'splits', 'test.txt')
        for did in open(imgsets_file):
            did = did.strip('.pcd\n')
            img_file = osp.join(dataset_dir, 'images/%s.png' % did)
            lbl_file = osp.join(dataset_dir, 'labels/%s.png' % did)
            self.files['validation'].append({'img': img_file, 'lbl': lbl_file})

class KittiRoadTrain(KittiClass):
    def __init__(self, root, split = 'train', transform = False):
        super(KittiRoadTrain, self).__init__(root, split = split, transform = transform)