# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       chengDataLoader
   Project Name:    segOCT
   Author :         Administrator
   Date:            2018/11/4
-------------------------------------------------
   Change Activity:
                   2018/11/4:
-------------------------------------------------
"""
import sys
sys.path.append('./')
sys.path.append('../')

import os
import pdb
import time
from PIL import Image

import torch.nn as nn
import torch.utils.data as data

import numpy as np

from utils.visualizer import Visualizer

class volumeLoader(nn.Container):
    def __init__(self, volume_root, batch, num_worker):
        super(volumeLoader, self).__init__()
        self.volume_root = volume_root
        self.batch = batch
        self.workers = num_worker

    def data_load(self):
        test_set = volumeTestSet(self.volume_root)
        test_loader = data.DataLoader(
            dataset=test_set,
            batch_size=self.batch,
            shuffle=True,
            num_workers=self.workers,
            pin_memory=True
        )
        return test_loader


class volumeTestSet(data.Dataset):
    def __init__(self, volume_root, edge=False):
        super(volumeTestSet, self).__init__()
        # volume_root: '/workspace/2018_OCT_transfer/dataset/cheng'
        self.img_dir = os.path.join(volume_root, '566.fds')
        self.mask_dir = os.path.join(volume_root, '566.fds/mask_3')
        if edge:
            self.edge_dir = os.path.join(volume_root, '566.fds/save_gt')
        else:
            self.edge_dir = None
        self.img_name_list = os.listdir(self.mask_dir)

    def __len__(self):
        return len(self.img_name_list)

    def __getitem__(self, item):
        img = Image.open(os.path.join(self.img_dir, self.img_name_list[item]))
        mask = Image.open(os.path.join(self.mask_dir, self.img_name_list[item]))
        # TODO: add transform
        if self.edge_dir:
            edge = Image.open(os.path.join(self.edge_dir, self.img_name_list[item]))
            name = self.img_name_list[item]
            return img, mask, edge, name
        else:
            return img, mask


class allLoader(nn.Container):
    def __init__(self):
        super(allLoader, self).__init__()
        pass


class allTestSet(data.Dataset):
    def __init__(self):
        super(allTestSet, self).__init__()
        pass


class chengAnalyser(object):
    def __init__(self):
        self.vis = Visualizer(env='v99_cheng', port=31432)
        volume_root = '/root/workspace/2018_OCT_transfer/dataset/cheng'
        self.cheng_oct = volumeTestSet(volume_root, True)

    def run(self):
        for item in range(self.cheng_oct.__len__()):
            img, mask, edge, name = self.cheng_oct.__getitem__(item)
            img = np.swapaxes(np.array(img), 2, 1)
            img = np.swapaxes(img, 1, 0)

            mask = np.array(mask) / 3. * 255

            print('hello cheng OCT: {}'.format(item))
            # self.vis.img_cpu(name='img', img_=img)
            self.vis.img_cpu(name='mask', img_=mask)
            self.vis.img_cpu(name='edge', img_=edge)
            self.vis.text(text=name)

            # self.vis.text(text=name, name='1')
            # self.vis.text(text=name, name='2')
            pdb.set_trace()


def main():
    import tb
    tb.colour()
    chengAnalyser().run()

    # loader = transforms.Compose([
    #     transforms.Resize((512, 512)),  # scale imported image
    #     transforms.ToTensor()])  # transform it into a torch tensor
    #
    # Resize_224 = transforms.Resize((224, 224))
    #
    # def image_loader(image_name):
    #     image = Image.open(image_name)
    #
    #     # image = image.convert('L')
    #     image = Resize_224(image)
    #     # fake batch dimension required to fit network's input dimensions
    #     image = loader(image).unsqueeze(0)
    #
    #     print(image.shape)
    #
    #     return image.to(device, torch.float)


if __name__ == '__main__':
    main()
