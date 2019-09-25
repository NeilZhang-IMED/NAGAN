import glob
import random
import os

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
# import cv2
import scipy.misc as misc

class ImageDataset(Dataset):
    def __init__(self, root, transforms_=None, unaligned=True, mode='train'):
        self.transform = transforms.Compose(transforms_)
        self.unaligned = unaligned

        self.files_A = sorted(glob.glob(os.path.join(root, '%s/sina_NAGAN' % mode) + '/*.*'))
        self.files_B = sorted(glob.glob(os.path.join(root, '%s/cheng_all' % mode) + '/*.*'))

    def __getitem__(self, index):


        item_A = self.transform(Image.open(self.files_A[index % len(self.files_A)]).convert("L"))

        temp_name = self.files_A[index % len(self.files_A)]
        Temp_NAME = temp_name.split('/')
        temp_name = Temp_NAME[-1]

        A_name = temp_name

        if self.unaligned:

            item_B = self.transform(Image.open(self.files_B[random.randint(0, len(self.files_B) - 1)]).convert("L"))
        else:

            item_B = self.transform(Image.open(self.files_B[index % len(self.files_B)]).convert("L"))

        return {'A': item_A, 'B': item_B, 'A_name': A_name}

    def __len__(self):
        return max(len(self.files_A), len(self.files_B))