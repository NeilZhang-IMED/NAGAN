# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name:       visualizer
   Project Name:    octNet
   Author :         康
   Date:            2018/9/22
-------------------------------------------------
   Change Activity:
                   2018/9/22:
-------------------------------------------------
"""
import visdom
import numpy as np
import torch

import pdb


class Visualizer(object):
    def __init__(self, env='main', port=31430, **kwargs):
        self.viz = visdom.Visdom(env=env, port=port, **kwargs)

        # 画的第几个数，相当于横坐标
        # 比如（’loss',23） 即loss的第23个点
        self.index = {}
        self.log_text = ''

    def plot_many(self, d, loop_flag=None):
        '''
        一次plot多个或者一个
        @params d: dict (name, value) i.e. ('loss', 0.11)
        '''
        long_update = True
        if loop_flag == 0:
            long_update = False
        for k, v in d.items():
            self.plot(k, v, long_update)

    def plot(self, name, y, long_update, **kwargs):
        '''
        self.plot('loss', 1.00)
        One mame, one win: only one lie in a win.
        '''
        x = self.index.get(
            name, 0)  # dict.get(key, default=None). 返回指定键的值，如果值不在字典中返回default值
        self.viz.line(Y=np.array([y]), X=np.array([x]),
                      win=name,
                      opts=dict(title=name),
                      update='append' if (x > 0 and long_update) else None,
                      **kwargs)
        self.index[name] = x + 1    # Maintain the X

    def multi_plot(self, d, win, loop_i=1):
        """
        :param d: dict (name, value) i.e. ('loss', 0.11)
        :param win: only one win
        :param loop_i: i.e. plot testing loss and label
        :return:
        """
        for k, v in d.items():
            x = self.index.get(k, 0)
            self.viz.line(Y=np.array([v]), X=np.array([x]),
                          name=k,
                          win=win,
                          opts=dict(title=win, showlegend=True),
                          update='append' if (x > 0 and loop_i > 0) else None)
                          # update=None if (x == 0 or loop_i == 0) else 'append')
            self.index[k] = x + 1

    # To be delete
    # def multi_line(self, name, y, win=None, title='line', legend=None, update=True):
    #     # if not isinstance(x, list):
    #     #     x = [x]
    #     # if not (isinstance(y, list) or isinstance(y, np.ndarray)):
    #     #     y = np.array([y])
    #     # There can be multi line in one win.
    #     x = self.index.get(name, 0)
    #     self.viz.line(
    #         X=x,
    #         Y=np.array([y]),
    #         win=win,
    #         name=name,
    #         opts={'title': title, 'showlegend': True},
    #         update='append' if (update and x > 0) else None
    #     )
    def img(self, name, img_, **kwargs):
        '''
        self.img('input_img', t.Tensor(64, 64))
        self.img('input_imgs', t.Tensor(3, 64, 64))
        self.img('input_imgs', t.Tensor(100, 1, 64, 64))
        self.img('input_imgs', t.Tensor(100, 3, 64, 64), nrows=10)
        '''
        self.viz.images(img_.cpu().numpy(),
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )

    def img_cpu(self, name, img_, **kwargs):
        if not isinstance(img_, np.ndarray):
            img_ = np.array(img_)
        self.viz.images(img_,
                        win=name,
                        opts=dict(title=name),
                        **kwargs
                        )


    def images(self, img_list, name='images', nrow=2):
        """
        There are two or more images in on win.
        :return:
        """
        assert len(img_list) >= 2, 'If you want draw one image, please use img()'
        self.viz.images(
            img_list,
            nrow=nrow,
            win=name,
            opts=dict(title=name, caption='How random.')
        )

    # # TODO: remove
    # def image(self, name, image, title='image'):
    #     self.viz.image(
    #         image,
    #         win=name,
    #         opts={'title': name}
    #     )

    def text(self, text, name='text'):
        self.viz.text(text, win=name)

    def draw_roc(self, fpr, tpr):
        self.viz.line(Y=np.array(tpr), X=np.array(fpr),
                      name='roc_curve',
                      win='roc_curve',
                      opts=dict(title='roc_curve', showlegend=True))

    def __getattr__(self, name):
        '''
        self.function 等价于self.vis.function
        自定义的plot,image,log,plot_many等除外
        '''
        return getattr(self.vis, name)


def main():
    # TODO: visualize the distribution of train set and test set
    vis = Visualizer()
    # vis.line(2, 2, '332')

    # vis = visdom.Visdom(port=31430)
    # vis.line(np.array([2]), np.array([2]))


if __name__ == '__main__':
    main()
