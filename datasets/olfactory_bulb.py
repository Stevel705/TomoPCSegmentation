import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

import os
import cv2
import random
import logging
import numpy as np
from PIL import Image


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

class OlfactoryBulbDataset(Dataset):
    """
    
    """
    def __init__(self, data_dir: str, mode: str, scale = 0.5, is_transform=False, n_classes=1):
        """
        Args:
            
        """
        self.is_transform = is_transform
        self.scale = scale
        self.n_classes = n_classes
        
        IMAGE_PATH = os.path.join(mode, "imgs")
        MASK_PATH = os.path.join(mode, "masks")
        self.data_dir = data_dir
        self.img_path = os.path.join(self.data_dir, IMAGE_PATH)
        self.mask_path = os.path.join(self.data_dir, MASK_PATH)
        self.img_list = self.get_filenames(self.img_path)
        self.mask_list = self.get_filenames(self.mask_path)
        
    def __len__(self):
        return len(self.img_list)

    def transform(self, image, mask, scale):
        # Resize
        w, h = image.size
        newW, newH = int(scale * w), int(scale * h)  
        # newW, newH = 1024, 1024 
        resize = transforms.Resize(size=(newW, newH))
        image = resize(image)
        mask = resize(mask)

        # # Random crop
        # i, j, h, w = transforms.RandomCrop.get_params(
        #     image, output_size=(512, 512))
        # image = TF.crop(image, i, j, h, w)
        # mask = TF.crop(mask, i, j, h, w)

        if self.is_transform:
            # Random horizontal flipping
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)

            # # Random vertical flipping
            if random.random() > 0.5:
                degree = random.randint(-90, 90)
                image = TF.rotate(image, degree)
                mask = TF.rotate(mask, degree)

        # ====================================
        image = np.array(image)
        mask = np.array(mask)

        if len(image.shape) == 2:
            image = np.expand_dims(image, axis=2)

        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=2)

        # HWC to CHW
        image = image.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))
        
        if image.max() > 1:
            image = image / 255

        if self.n_classes == 1:
            # If multiclass then comment this block  
            if mask.max() > 1:
                mask = mask / 255
        
        # Transform to tensor
        image = torch.from_numpy(image).type(torch.FloatTensor)
        mask = torch.from_numpy(mask)

        return image, mask

    def __getitem__(self, idx):

        image = Image.open(self.img_list[idx])
        mask = Image.open(self.mask_list[idx])

        image, mask = self.transform(image, mask, self.scale)

        # trms = transforms.Compose([AddGaussianNoise(0., 0.0001)])
        return {
            'image': image,
            'mask': mask
        }

    def get_filenames(self, path):
        """Returns a list of absolute paths to images inside given `path`"""
        files_list = list()
        for filename in os.listdir(path):
            files_list.append(os.path.join(path, filename))
        return files_list