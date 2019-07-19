# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       edge2mask
   Project Name:    segOCT
   Author :         Kang ZHOU
   Date:            2018/11/5
-------------------------------------------------
   Change Activity:
                   2018/11/5:
-------------------------------------------------
"""
import sys
sys.path.append('../')

import os
import pdb

from PIL import Image
import numpy as np

from utils.visualizer import Visualizer

class Edge2Mask(object):
    def __init__(self):
        self.img_dir = '/root/workspace/2018_OCT_transfer/dataset/cheng/566.fds'
        self.edge_dir = os.path.join(self.img_dir, 'save_gt')
        self.mask10_dir = os.path.join(self.img_dir, 'gt_10')
        self.mask_dir = os.path.join(self.img_dir, 'mask_3')
        self.edge_name_list = os.listdir(self.edge_dir)
        self.vis = Visualizer(env='v99_cheng', port=31432)

    def run(self):
        for edge_name in self.edge_name_list:
            save_mask_flag = True
            edge = np.array(Image.open(os.path.join(self.edge_dir, edge_name)))

            # cnt = 0
            # for column in range(edge.shape[1]):
            #     layer_index = np.where(edge[:, column] == 255)[0]
            #     if not layer_index.shape[0] == layer_index.shape[8] == layer_index.shape[9] == 255:
            #         cnt += 1
            # print('Edge_name: {}, Num of discontinuous: {}'.format(edge_name, cnt))

            for column in range(edge.shape[1]):
                layer_index = np.where(edge[:, column]==255)[0]
                if layer_index.shape[0] == 11:
                    edge[:, column][:layer_index[0]] = 0
                    edge[:, column][layer_index[0]:layer_index[8]] = 1
                    edge[:, column][layer_index[8]:layer_index[9]] = 2
                    edge[:, column][layer_index[9]:] = 3
                else:
                    save_mask_flag=False
            if save_mask_flag:
                mask = Image.fromarray(edge)
                mask.save(os.path.join(self.mask_dir, edge_name))
                print('Successful save mask: {}'.format(edge_name))
            else:
                print('Not discontinuous: {}'.format(edge_name))
            # pdb.set_trace()
        print('Successful save all mask!')



def main():
    import tb
    tb.colour()
    Edge2Mask().run()


if __name__ == '__main__':
    main()
