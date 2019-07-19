#!/usr/bin/python3

import argparse
import itertools
import pdb
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable
from PIL import Image
import torch

from models import Generator
from models import Discriminator
from util import ReplayBuffer
from util import LambdaLR
from util import Logger
from util import weights_init_normal
from datasets_utils import ImageDataset
from Model_VGG19 import gram_matrix


parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')

parser.add_argument('--n_epochs', type=int, default=12, help='number of epochs of training')
parser.add_argument('--batchSize', type=int, default=1, help='size of the batches')

parser.add_argument('--dataroot', type=str, default='datasets/', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=11, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=512, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--cuda', action='store_true',default=True, help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=16, help='number of cpu threads to use during batch generation')
parser.add_argument('--transfer', type=bool, default=False, help='Restore parameters of this network')
parser.add_argument('--style_weight', type=float, default=500,help='style weight')
parser.add_argument('--GAN_weight', type=float, default=0.01, help='GAN weight')




opt = parser.parse_args()
print(opt)

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

###### Definition of variables ######
# Networks
netG_A2B = Generator(opt.input_nc, opt.output_nc)

netD_A = Discriminator(opt.input_nc)

netD_A_self = Discriminator(opt.input_nc)


# netD_A_content = Discriminator(opt.input_nc)
# netD_B_content = Discriminator(opt.output_nc)

####### Restore Parameters #########
if opt.transfer == True:
    netG_A2B.load_state_dict('output/netG_A2B.pth')

    netD_A.load_state_dict('output/netD_A.pth')



if opt.cuda:
    netG_A2B.cuda()
    netG_A2B = torch.nn.DataParallel(netG_A2B, device_ids=range(torch.cuda.device_count()))

    netD_A.cuda()
    netD_A = torch.nn.DataParallel(netD_A, device_ids=range(torch.cuda.device_count()))



    netD_A_self.cuda()
    netD_A_self = torch.nn.DataParallel(netD_A, device_ids=range(torch.cuda.device_count()))



    # netD_A_content.cuda()
    # netD_A_content = torch.nn.DataParallel(netD_A_content, device_ids=range(torch.cuda.device_count()))
    #
    # netD_B_content.cuda()
    # netD_B_content = torch.nn.DataParallel(netD_B_content, device_ids=range(torch.cuda.device_count()))


netG_A2B.apply(weights_init_normal)

netD_A.apply(weights_init_normal)


netD_A_self.apply(weights_init_normal)



# Lossess
criterion_GAN = torch.nn.MSELoss()
criterion_cycle = torch.nn.L1Loss()
criterion_identity = torch.nn.L1Loss()

# Optimizers & LR schedulers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters()),
                                lr=opt.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=opt.lr, betas=(0.5, 0.999))

optimizer_D_A_self = torch.optim.Adam(netD_A_self.parameters(), lr=opt.lr, betas=(0.5, 0.999))


# optimizer_G = torch.optim.SGD(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),lr=opt.lr)
# optimizer_D_A = torch.optim.SGD(netD_A.parameters(),lr=opt.lr)
# optimizer_D_B = torch.optim.SGD(netD_B.parameters(),lr=opt.lr)


lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=LambdaLR(opt.n_epochs, opt.epoch, opt.decay_epoch).step)


# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if opt.cuda else torch.Tensor
input_A = Tensor(opt.batchSize, opt.input_nc, opt.size, opt.size)
input_B = Tensor(opt.batchSize, opt.output_nc, opt.size, opt.size)
target_real = Variable(Tensor(opt.batchSize).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(opt.batchSize).fill_(0.0), requires_grad=False)

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

# Dataset loader
transforms_ = [ transforms.Resize((int(opt.size), int(opt.size)), Image.BICUBIC),
                # transforms.RandomCrop(opt.size),
                #transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
dataloader = DataLoader(ImageDataset(opt.dataroot, transforms_=transforms_, unaligned=True), 
                        batch_size=opt.batchSize, shuffle=True, num_workers=opt.n_cpu,drop_last=True)

# Loss plot
logger = Logger(opt.n_epochs, len(dataloader))

# heat map
loader = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((32, 32)),
    transforms.Resize((512, 512)),  # scale imported image
    transforms.ToTensor()])  # transform it into a torch tensor




def image_loader(image):

    # fake batch dimension required to fit network's input dimensions
    image = loader(image)

    print(image.shape)

    return image







###################################
A_dis = 0.998
early_stop = 0

loss_best = 10000
###################################
###### Training ######
for epoch in range(opt.epoch, opt.n_epochs):

    if epoch>3:
        input("Press Enter to continue...")


    for i, batch in enumerate(dataloader):
        print("break")
        print(i)
        # Set model input
        real_A = Variable(input_A.copy_(batch['A']))
        real_B = Variable(input_B.copy_(batch['B']))

        ###### Generators A2B  ######
        optimizer_G.zero_grad()


        # GAN loss##############################################
        fake_B = netG_A2B(real_A)


        pred_fake, _ = netD_A(fake_B.detach())

        if pred_fake.size() != target_real.size():

            target_real = torch.unsqueeze(target_real, 1)

        loss_GAN_A2B = criterion_GAN(pred_fake, target_real)



        #####################################################
        # loss self

        pred_fake, _ = netD_A_self(fake_B.detach())

        if pred_fake.size() != target_fake.size():
            target_fake = torch.unsqueeze(target_fake, 1)

        loss_D_fake_self_A = criterion_GAN(pred_fake, target_real)


        loss_D_A_self = loss_D_fake_self_A

        #######################################################################
        # content loss
        _, content_feature_RA = netD_A_self(real_A)



        _, content_feature_RB = netD_A_self(real_B)
        _, content_feature_FB = netD_A_self(fake_B)

        content_loss_A = criterion_cycle(content_feature_FB[3], content_feature_RA[3].detach())
        # style loss



        Gram_RA_0 = gram_matrix(content_feature_RA[0])
        Gram_RA_1 = gram_matrix(content_feature_RA[1])
        Gram_RA_2 = gram_matrix(content_feature_RA[2])

        Gram_FB_0 = gram_matrix(content_feature_FB[0])
        Gram_FB_1 = gram_matrix(content_feature_FB[1])
        Gram_FB_2 = gram_matrix(content_feature_FB[2])

        Gram_RB_0 = gram_matrix(content_feature_RB[0])
        Gram_RB_1 = gram_matrix(content_feature_RB[1])
        Gram_RB_2 = gram_matrix(content_feature_RB[2])


        loss_style_B_0 = criterion_identity(Gram_FB_0, Gram_RB_0.detach())

        loss_style_B_1 = criterion_identity(Gram_FB_1, Gram_RB_1.detach())

        loss_style_B_2 = criterion_identity(Gram_FB_2, Gram_RB_2.detach())





        loss_style_B = loss_style_B_0 + loss_style_B_1 + loss_style_B_2

        loss_style = ( loss_style_B)*opt.style_weight

        # A_dis = torch.exp(torch.Tensor([-(epoch / 20)]))
        # A_dis = A_dis.cpu().detach().numpy()
        # A_dis = float(A_dis)
        # if epoch<16:
        #     loss_G = loss_style + (loss_GAN_A2B + loss_D_A_self)*opt.GAN_weight + content_loss_A
        #
        # else:
        #     loss_G = loss_style + (loss_GAN_A2B + loss_D_A_self) *(opt.GAN_weight * A_dis+0.0005) + content_loss_A


        A_dis = A_dis*0.999
        if epoch < 20:
            loss_G = loss_style + (loss_GAN_A2B + loss_D_A_self) * opt.GAN_weight + content_loss_A

        else:
            loss_G = loss_style + (loss_GAN_A2B + loss_D_A_self) * (opt.GAN_weight * A_dis) + content_loss_A



        loss_G.backward(retain_graph=True)
        
        optimizer_G.step()
        ###################################

        optimizer_G.zero_grad()


        ##################################################################################################
        #                       Discriminator
        ##################################################################################################


        ###### Discriminator A ######
        optimizer_D_A.zero_grad()

        # Real loss

        pred_real,_ = netD_A(real_B)
        loss_D_real = criterion_GAN(pred_real, target_real)




        #Fake loss

        pred_fake,_ = netD_A(fake_B.detach())

        # # print(pred_fake.size())
        if pred_fake.size() != target_fake.size():
            target_fake = torch.unsqueeze(target_fake, 1)
        loss_D_fake = criterion_GAN(pred_fake, target_fake)


        loss_D_A = loss_D_fake+loss_D_real
        loss_D_A.backward(retain_graph=True)

        optimizer_D_A.step()





        ###################################

        ###### Discriminator A_self ######
        optimizer_D_A_self.zero_grad()

        # # Real loss
        pred_real, _ = netD_A_self(real_A)
        loss_D_real_self_A = criterion_GAN(pred_real, target_real)


        # Fake loss
        # fake_B = fake_B_buffer.push_and_pop(fake_B)
        pred_fake, _ = netD_A_self(fake_B.detach())

        loss_D_fake_self_A = criterion_GAN(pred_fake, target_fake)


        loss_D_A_self = loss_D_fake_self_A + loss_D_real_self_A

        loss_D_A_self.backward()

        optimizer_D_A_self.step()




        ###################################

        diff = (fake_B - real_A)
        diff = diff.reshape(-1,512,512)
        diff = image_loader(diff.cpu().detach())

        diff = diff.reshape(512, 512)

        print("diff.shape\n")
        print(diff.shape)

        # Progress report (http://localhost:8097)

        logger.log({'loss_G': loss_G, 'loss_G_GAN': (loss_GAN_A2B ),\
                    'loss_style':loss_style,'loss_content':(content_loss_A), 'loss_D_self':(loss_D_A_self),
                     'loss_D': (loss_D_A )},
                    images={'real_A': real_A, 'real_B': real_B,  'fake_B': fake_B},
                   heatmaps={'heatmap': diff})
    # Save models checkpoints
    torch.save(netG_A2B.state_dict(), 'output/netG_A2B.pth')

    torch.save(netD_A.state_dict(), 'output/netD_A.pth')



    if loss_GAN_A2B<loss_best:
        loss_best = loss_GAN_A2B
    elif loss_GAN_A2B> loss_best:
        early_stop +=1

    if early_stop >5 or (loss_GAN_A2B > loss_best+1.8):
        break



    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()



###################################
