#!/usr/bin/python3

import argparse
import sys
import os

import torchvision.transforms as transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
from collections import OrderedDict
import torch.nn.modules
import numpy as np
from models import Generator
from datasets_utils import ImageDataset
import Visualizer
viz = Visualizer.Visualizer("test")

parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='datasets/', help='root directory of the dataset')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--size', type=int, default=512, help='size of the data (squared assumed)')
parser.add_argument('--cuda', action='store_true', help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--generator_A2B', type=str, default='output/netG_A2B.pth', help='A2B generator checkpoint file')
opt = parser.parse_args()
print(opt)

os.makedirs('output/source', exist_ok=True)
os.makedirs('output/target', exist_ok=True)
os.makedirs('output/S2T', exist_ok=True)
# def tensor2image(tensor):
#     image = 127.5 * (tensor[0].cpu().float().numpy() + 1.0)
#     if image.shape[0] == 1:
#         image = np.tile(image, (3, 1, 1))
#     return image.astype(np.uint8)

def tensor2image(tensor):
    image = 0.5 * (tensor + 1.0)
    #if image.shape[0] == 1:
        #image = np.tile(image, (3, 1, 1))
    return image

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)
netG_B2A = Generator(opt.output_nc, opt.input_nc)

if opt.cuda:
    netG_A2B.cuda()
    netG_B2A.cuda()

# Load state dicts
temp_A2B = torch.load('output/netG_A2B_Time1.pth')
# temp_B2A = torch.load('output/netG_B2A.pth')


new_temp_A2B = OrderedDict()
for k, v in temp_A2B.items():
    name = k[7:]  # remove `module.`
    new_temp_A2B[name] = v
# load params

# new_temp_B2A = OrderedDict()
# for k, v in temp_B2A.items():
#     name = k[7:]  # remove `module.`
#     new_temp_B2A[name] = v
# # load params


netG_A2B.load_state_dict(new_temp_A2B)
# netG_B2A.load_state_dict(new_temp_B2A)

# Set model's test mode
netG_A2B.eval()
# netG_B2A.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)

# Dataset loader
# Dataset loader
transforms_ = [transforms.Resize((int(opt.size), int(opt.size)), Image.BICUBIC),
               # transforms.RandomCrop(opt.size),
               #transforms.RandomHorizontalFlip(),
               transforms.ToTensor(),
               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True),
                        batch_size=opt.batchSize, shuffle=False, num_workers=opt.n_cpu)

###################################

###### Testing######

# Create output dirs if they don't exist
# if not os.path.exists('output/fakeA'):
#     os.makedirs('output/fake_A')
# if not os.path.exists('output/fakeB'):
#     os.makedirs('output/fake_B')
#
# if not os.path.exists('output/A'):
#     os.makedirs('output/A')
# if not os.path.exists('output/B'):
#     os.makedirs('output/B')


for i, batch in enumerate(dataloader):
    # Set model input

    real_A = Variable(input_A.copy_(batch['A']))
    real_B = Variable(input_B.copy_(batch['B']))
    A_name = batch['A_name']
    # Generate output
    # fake_B = 0.5*(netG_A2B(real_A).data + 1.0)
    # fake_A = 0.5*(netG_B2A(real_B).data + 1.0)

    fake_B = netG_A2B(real_A).data



    # #
    viz.img("A",tensor2image(real_A))
    viz.img("B",tensor2image(real_B))
    viz.img("A2B",tensor2image(fake_B))
    # print("batch {}".format(i))

    # Save image files
    # print(A_name[0])

    save_image(tensor2image(real_A), 'output/source/%s' % (A_name[0]))
    save_image(tensor2image(real_B), 'output/target/%04d.png' % (i + 1))



    save_image(tensor2image(fake_B), 'output/S2T/{}'.format(A_name[0]) )

    sys.stdout.write('\rGenerated images %04d of %04d' % (i+1, len(dataloader)))

sys.stdout.write('\n')
###################################
