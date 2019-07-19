"""
Neural Transfer Using PyTorch
=============================


**Author**: `Alexis Jacq <https://alexis-jacq.github.io>`_

**Edited by**: `Winston Herring <https://github.com/winston6>`_

Introduction
------------

This tutorial explains how to implement the `Neural-Style algorithm <https://arxiv.org/abs/1508.06576>`__
developed by Leon A. Gatys, Alexander S. Ecker and Matthias Bethge.
Neural-Style, or Neural-Transfer, allows you to take an image and
reproduce it with a new artistic style. The algorithm takes three images,
an input image, a content-image, and a style-image, and changes the input
to resemble the content of the content-image and the artistic style of the style-image.


.. figure:: /_static/img/neural-style/neuralstyle.png
   :alt: content1
"""

######################################################################
# Underlying Principle
# --------------------
#
# The principle is simple: we define two distances, one for the content
# (:math:`D_C`) and one for the style (:math:`D_S`). :math:`D_C` measures how different the content
# is between two images while :math:`D_S` measures how different the style is
# between two images. Then, we take a third image, the input, and
# transform it to minimize both its content-distance with the
# content-image and its style-distance with the style-image. Now we can
# import the necessary packages and begin the neural transfer.
#
# Importing Packages and Selecting a Device
# -----------------------------------------
# Below is a  list of the packages needed to implement the neural transfer.
#
# -  ``torch``, ``torch.nn``, ``numpy`` (indispensables packages for
#    neural networks with PyTorch)
# -  ``torch.optim`` (efficient gradient descents)
# -  ``PIL``, ``PIL.Image``, ``matplotlib.pyplot`` (load and display
#    images)
# -  ``torchvision.transforms`` (transform PIL images into tensors)
# -  ``torchvision.models`` (train or load pre-trained models)
# -  ``copy`` (to deep copy the models; system package)

from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import sys
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.utils import save_image
import os
import tqdm
import copy
from Model_VGG19 import *

import Model_VGG19
import Visualizer

VIS = Visualizer.Visualizer("VGG19")



######################################################################
# Next, we need to choose which device to run the network on and import the
# content and style images. Running the neural transfer algorithm on large
# images takes longer and will go much faster when running on a GPU. We can
# use ``torch.cuda.is_available()`` to detect if there is a GPU available.
# Next, we set the ``torch.device`` for use throughout the tutorial. Also the ``.to(device)``
# method is used to move tensors or modules to a desired device.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

######################################################################
# Loading the Images
# ------------------
#
# Now we will import the style and content images. The original PIL images have values between 0 and 255, but when
# transformed into torch tensors, their values are converted to be betweenimport sys
# 0 and 1. The images also need to be resized to have the same dimensions.
# An important detail to note is that neural networks from the
# torch library are trained with tensor values ranging from 0 to 1. If you
# try to feed the networks with 0 to 255 tensor images, then the activated
# feature maps will be unable sense the intended content and style.
# However, pre-trained networks from the Caffe library are trained with 0
# to 255 tensor images.
#
#
# .. Note::
#     Here are links to download the images required to run the tutorial:
#     `picasso.jpg <http://pytorch.org/tutorials/_static/img/neural-style/picasso.jpg>`__ and
#     `dancing.jpg <http://pytorch.org/tutorials/_static/img/neural-style/dancing.jpg>`__.
#     Download these two images and add them to a directory
#     with name ``images`` in your current working directory.

# desired size of the output image
imsize = 512 if torch.cuda.is_available() else 128  # use small size if no gpu

loader = transforms.Compose([
    transforms.Resize((512,512)),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor

Resize_224 = transforms.Resize((224, 224))


def image_loader(image_name):
    image = Image.open(image_name)

    #image = image.convert('L')
    image = Resize_224(image)
    # fake batch dimension required to fit network's input dimensions
    image = loader(image).unsqueeze(0)

    print(image.shape)

    return image.to(device, torch.float)


###############################################################################
#                   Walk the root for obtaining all images
###############################################################################
Main_path_cheng = "/home/imed/PycharmProjects/2018_OCT_transfer/datasets/cheng"
open_path = []
content_name = []
for root, dirs, files in os.walk(Main_path_cheng, topdown=False):
    num = len(files)

Main_path_sina = "/home/imed/PycharmProjects/2018_OCT_transfer/datasets/sina"
for root, dirs, files_sina in os.walk(Main_path_sina, topdown=False):
    num = len(files)


for name in tqdm.tqdm_gui(files):
    open_path.append(os.path.join(Main_path_cheng, name))
    print("Load image {}".format(name))
    namelist = name.split('.')
    content_name.append(namelist[0])

style_path = []
style_name = []
for name in tqdm.tqdm_gui(files_sina):
    style_path.append(os.path.join(Main_path_sina, name))
    print("Style image {}".format(name))
    namelist = name.split('.')
    style_name.append(namelist[0])

        # tmp_name = os.path.join(, name)
        # tmp_img.save(tmp_name, "jpeg")


k = 0

# Create output dirs if they don't exist
if not os.path.exists('output/Output)'):
    os.makedirs('output/Output')

if not os.path.exists('output/sina'):
    os.makedirs('output/sina')
if not os.path.exists('output/cheng'):
    os.makedirs('output/cheng')

for Input_image_path in tqdm.tqdm_gui(open_path):



    content_img = image_loader(Input_image_path)

    # print(content_img.size)

    style_img = image_loader(style_path[k])



    # style_img = Resize_224(style_img)
    # content_img = Resize_224(content_img)
    print(style_img.size())
    print(content_img.size())

    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"

    ######################################################################
    # Now, let's create a function that displays an image by reconverting a
    # copy of it to PIL format and displaying the copy using
    # ``plt.imshow``. We will try displaying the content and style images
    # to ensure they were imported correctly.

    unloader = transforms.ToPILImage()  # reconvert into PIL image

    plt.ion()
    #
    #
    # def imshow(tensor, title=None):
    #     image = tensor.cpu().clone()  # we clone the tensor to not do changes on it
    #     image = image.squeeze(0)  # remove the fake batch dimension
    #     image = unloader(image)
    #     plt.imshow(image)
    #     if title is not None:
    #         plt.title(title)
    #     plt.pause(0.001)  # pause a bit so that plots are updated
    #
    #
    # plt.figure()
    # imshow(style_img, title='Style Image')
    #
    # plt.figure()
    # imshow(content_img, title='Content Image')



    # desired depth layers to compute style/content losses :

    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    content_layers_default = ['conv_4']
    style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

    ######################################################################
    # Next, we select the input image. You can use a copy of the content image
    # or white noise.
    #

    cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
    cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)


    input_img = content_img.clone()
    # if you want to use white noise instead uncomment the below line:
    # input_img = torch.randn(content_img.data.size(), device=device)

    # add the original input image to the figure:







    ######################################################################
    # Gradient Descent
    # ----------------
    #
    # As Leon Gatys, the author of the algorithm, suggested `here <https://discuss.pytorch.org/t/pytorch-tutorial-for-neural-transfert-of-artistic-style/336/20?u=alexis-jacq>`__, we will use
    # L-BFGS algorithm to run our gradient descent. Unlike training a network,
    # we want to train the input image in order to minimise the content/style
    # losses. We will create a PyTorch L-BFGS optimizer ``optim.LBFGS`` and pass
    # our image to it as the tensor to optimize.
    #



    ######################################################################
    # Finally, we can run the algorithm.
    #

    output = run_style_transfer(cnn, cnn_normalization_mean, cnn_normalization_std,
                                content_img, style_img, input_img)

    VIS.img(img_=style_img, name='style Image')
    VIS.img(img_=content_img, name='Input Image')
    VIS.img(img_=output, name='putput Image')

    VIS.img(img_= (output-content_img)*200, name = 'diff')

    save_image(content_img, 'output/cheng/%s.png' % (content_name[k]))
    save_image(style_img, 'output/sina/%s.png' % (style_name[k]))

    save_image(output, 'output/Output/%s.png' % (content_name[k]))

    k += 1

