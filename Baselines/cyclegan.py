import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time
from PIL import Image

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
from datasets_utils import ImageDataset
from Baselines.utils import *
from Baselines.models_cycle import *
# from datasets import *
# from utils import *
from Visualizer import Visualizer
import torch.nn as nn
import torch.nn.functional as F
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "6"

VIS = Visualizer("CycleGAN")
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=200, help='number of epochs of training')
parser.add_argument('--dataset_name', type=str, default="monet2photo", help='name of the dataset')
# parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--batchSize', type=int, default=2, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='/home/jimmyliu/Tianyang/NAGAN/datasets', help='root directory of the dataset')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=512, help='size of image height')
parser.add_argument('--img_width', type=int, default=512, help='size of image width')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--sample_interval', type=int, default=100, help='interval between sampling images from generators')
parser.add_argument('--checkpoint_interval', type=int, default=-1, help='interval between saving model checkpoints')
parser.add_argument('--n_residual_blocks', type=int, default=9, help='number of residual blocks in generator')
opt = parser.parse_args()
print(opt)

# Create sample and checkpoint directories
os.makedirs('images/%s' % opt.dataset_name, exist_ok=True)
os.makedirs('saved_models/%s' % opt.dataset_name, exist_ok=True)

# Losses
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

cuda = True if torch.cuda.is_available() else False

# Calculate output of image discriminator (PatchGAN)
patch = (1, opt.img_height // 2**4, opt.img_width // 2**4)

# Initialize generator and discriminator
G_AB = GeneratorResNet(res_blocks=opt.n_residual_blocks)
G_BA = GeneratorResNet(res_blocks=opt.n_residual_blocks)
D_A = Discriminator()
D_B = Discriminator()

if cuda:
    G_AB = G_AB.cuda()
    G_BA = G_BA.cuda()
    D_A = D_A.cuda()
    D_B = D_B.cuda()
    criterion_GAN.cuda()
    criterion_cycle.cuda()
    criterion_identity.cuda()

if opt.epoch != 0:
    # Load pretrained models
    G_AB.load_state_dict(torch.load('saved_models/%s/G_AB_%d.pth' % (opt.dataset_name, opt.epoch)))
    G_BA.load_state_dict(torch.load('saved_models/%s/G_BA_%d.pth' % (opt.dataset_name, opt.epoch)))
    D_A.load_state_dict(torch.load('saved_models/%s/D_A_%d.pth' % (opt.dataset_name, opt.epoch)))
    D_B.load_state_dict(torch.load('saved_models/%s/D_B_%d.pth' % (opt.dataset_name, opt.epoch)))
else:
    # Initialize weights
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)

# Loss weights
lambda_cyc = 10
lambda_id = 0.5 * lambda_cyc

# Optimizers
optimizer_G = torch.optim.Adam(itertools.chain(G_AB.parameters(), G_BA.parameters()),
                                lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)

Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

# # Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Image transformations
transforms_ = [ transforms.Resize((opt.img_height,opt.img_width), Image.BICUBIC),
                transforms.ToTensor(),
                ]

# Training data loader
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True),
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu, drop_last=True)



# def sample_images(batches_done):
#     """Saves a generated sample from the test set"""
#     imgs = next(iter(val_dataloader))
#     real_A = Variable(imgs['A'].type(Tensor))
#     fake_B = G_AB(real_A)
#     real_B = Variable(imgs['B'].type(Tensor))
#     fake_A = G_BA(real_B)
#     img_sample = torch.cat((real_A.data, fake_B.data,
#                             real_B.data, fake_A.data), 0)
#     save_image(img_sample, 'images/%s/%s.png' % (opt.dataset_name, batches_done), nrow=5, normalize=True)

# ----------
#  Training
# ----------

prev_time = time.time()
for epoch in range(opt.epoch, opt.n_epochs):
    for i, batch in enumerate(dataloader):

        # Set model input
        real_A = Variable(batch['A'].type(Tensor))
        real_B = Variable(batch['B'].type(Tensor))

        # Adversarial ground truths (source discriminator labels_batch)
        valid = Variable(Tensor(np.ones((real_A.size(0), *patch))), requires_grad=False)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *patch))), requires_grad=False)

        # ------------------
        #  Train Generators
        # ------------------
        print(real_B.shape)

        optimizer_G.zero_grad()

        # Identity loss
        loss_id_A = criterion_identity(G_BA(real_A), real_A)
        loss_id_B = criterion_identity(G_AB(real_B), real_B)

        loss_identity = (loss_id_A + loss_id_B) / 2

        # GAN loss
        fake_B = G_AB(real_A)


        loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
        fake_A = G_BA(real_B)
        print(D_B(fake_A).shape)

        loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)



        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        # Cycle loss
        recov_A = G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        recov_B = G_AB(fake_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)

        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Total loss
        loss_G =    loss_GAN + \
                    lambda_cyc * loss_cycle + \
                    lambda_id * loss_identity

        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------

        optimizer_D_A.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_A(real_A), valid)
        # Fake loss (on batch of previously generated samples)
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2

        loss_D_A.backward()
        optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------

        optimizer_D_B.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_B(real_B), valid)
        # Fake loss (on batch of previously generated samples)
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2

        loss_D_B.backward()
        optimizer_D_B.step()

        loss_D = (loss_D_A + loss_D_B) / 2

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(dataloader) + i
        batches_left = opt.n_epochs * len(dataloader) - batches_done
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f, adv: %f, cycle: %f, identity: %f] ETA: %s" %
                                                        (epoch, opt.n_epochs,
                                                        i, len(dataloader),
                                                        loss_D.item(), loss_G.item(),
                                                        loss_GAN.item(), loss_cycle.item(),
                                                        loss_identity.item(), time_left))

        # # If at sample interval save image
        # if batches_done % opt.sample_interval == 0:
        #     sample_images(batches_done)


    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    VIS.img(name="A",img_=real_A)
    VIS.img(name="B",img_=real_B)
    VIS.img(name="A2B",img_=fake_B)

    if opt.checkpoint_interval != -1 and epoch % opt.checkpoint_interval == 0:
        # Save model checkpoints
        torch.save(G_AB.state_dict(), 'saved_models/%s/G_AB_%d.pth' % (opt.dataset_name, epoch))
        torch.save(G_BA.state_dict(), 'saved_models/%s/G_BA_%d.pth' % (opt.dataset_name, epoch))
        torch.save(D_A.state_dict(), 'saved_models/%s/D_A_%d.pth' % (opt.dataset_name, epoch))
        torch.save(D_B.state_dict(), 'saved_models/%s/D_B_%d.pth' % (opt.dataset_name, epoch))