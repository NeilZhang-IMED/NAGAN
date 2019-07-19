# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       sinaDataLoader
   Project Name:    segOCT
   Author :         Administrator
   Date:            2018/11/3
-------------------------------------------------
   Change Activity:
                   2018/11/3:
-------------------------------------------------
"""
import sys
sys.path.append('./')
sys.path.append('../')

import os
import time
import pdb
from PIL import Image

import torch
import torch.nn as nn
import torch.utils.data as data

import numpy as np

from utils.visualizer import Visualizer


class sinaLoader(nn.Container):
    def __init__(self, data_root, batch, num_val=165, num_worker=8):
        super(sinaLoader, self).__init__()
        # data_root: '/root/workspace/2018_OCT_transfer/dataset/sina/crop_mask/'
        self.img_root=os.path.join(data_root, 'data_aug_images')
        self.mask_root=os.path.join(data_root, 'data_aug_oct_mask')
        self.batch = batch
        self.num_val=num_val
        self.workers = num_worker

    def data_load(self):
        train_set = sinaTrainSet(img_root=self.img_root,
                                 mask_root=self.mask_root,
                                 num_val=self.num_val)
        val_set = sinaValSet(img_root=self.img_root,
                             mask_root=self.mask_root,
                             num_val=self.num_val)

        train_loader = data.DataLoader(
            dataset=train_set,
            batch_size=self.batch,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True
        )
        val_loader = data.DataLoader(
            dataset=val_set,
            # TODO: optimize
            batch_size=4,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True
        )

        return train_loader, val_loader


class sinaTrainSet(data.Dataset):
    def __init__(self, img_root, mask_root, num_val, label_root=None, transform=None):
        super(sinaTrainSet, self).__init__()
        assert len(os.listdir(img_root)) == len(os.listdir(label_root)), 'wrong...'
        self.img_root = img_root
        self.label_root = label_root
        self.mask_root = mask_root
        # all
        self.img_name_list = os.listdir(img_root)
        self.label_name_list = os.listdir(label_root)
        self.mask_name_list = os.listdir(mask_root)
        # train
        num_val = int(num_val)
        self.img_train_list = self.img_name_list[:-num_val]
        self.label_train_list = self.label_name_list[:-num_val]
        self.mask_train_list = self.mask_name_list[:-num_val]

        # TODO: add transform
        self.transform = transform

    def __len__(self):
        return len(self.img_train_list)

    def __getitem__(self, item):
        assert self.img_train_list[item] == self.mask_train_list[item], 'wrong...'
        img = Image.open(os.path.join(self.img_root, self.img_train_list[item]))
        mask = Image.open(os.path.join(self.mask_root, self.mask_train_list[item]))
        # border = Image.open(os.path.join(self.label_root, self.label_name_list[item]))
        # name = self.img_name_list[item]
        if self.transform is not None:
            # TODO: add
            pass
        return img, mask


class sinaValSet(sinaTrainSet):
    def __init__(self, img_root, mask_root, num_val, label_root=None, transform=None):
        # img_root, mask_root, num_val, etc should be same in sinaTrainSet
        super(sinaValSet, self).__init__(img_root, mask_root, num_val, label_root, transform)
        # val
        num_val=int(num_val)
        self.img_test_list = self.img_name_list[-num_val:]
        self.label_test_list = self.label_name_list[-num_val:]
        self.mask_test_list = self.mask_name_list[-num_val:]

    def __len__(self):
        return len(self.img_test_list)

    def __getitem__(self, item):
        assert self.img_test_list[item] == self.mask_test_list[item], 'wrong...'
        img = Image.open(os.path.join(self.img_root, self.img_test_list[item]))
        mask = Image.open(os.path.join(self.mask_root, self.mask_test_list[item]))
        # border = Image.open(os.path.join(self.label_root, self.label_name_list[item]))
        # name = self.img_name_list[item]
        if self.transform is not None:
            # TODO: add
            pass
        return img, mask


class sinaAnalyser(object):
    def __init__(self):
        self.vis = Visualizer(env='v99_sina', port=31432)
        img_root = '/root/workspace/2018_OCT_transfer/dataset/sina/crop_mask/data_aug_images'
        label_root = '/root/workspace/2018_OCT_transfer/dataset/sina/crop_mask/data_aug_label'
        mask_root = '/root/workspace/2018_OCT_transfer/dataset/sina/crop_mask/data_aug_oct_mask'
        self.sina_set = sinaTrainSet(img_root=img_root, label_root=label_root, mask_root=mask_root)
        self.sina_val = sinaValSet()

    def run(self):
        for item in range(self.sina_set.__len__()):
            img, mask = self.sina_set.__getitem__(item)
            if item % 10 == 0:
                print('hello sina TRAIN: {}'.format(item))
        for item in range(self.sina_val.__len__()):
            img, mask = self.sina_val.__getitem__(item)
            if item % 10 == 0:
                print('hello sina VAL: {}'.format(item))
            # mask = np.array(mask)
            # mask = mask / 3. * 255
            # mask.astype(int)
            # self.vis.img_cpu(name='sina-img', img_=img)
            # self.vis.img_cpu(name='sina-border', img_=border)
            # self.vis.img_cpu(name='sina-mask', img_=mask)
            # self.vis.text(name)
            # time.sleep(1)


def main():
    import tb
    tb.colour()
    sinaAnalyser().run()


if __name__ == '__main__':
    main()
