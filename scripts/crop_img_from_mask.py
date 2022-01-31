# python predict.py -i data/test/test_10.png -o output.jpg --model checkpoints/CP_epoch5.pth -v

import argparse
import logging
import os
from matplotlib import colors
import tifffile as tiff
from matplotlib.pyplot import sca

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import skimage.io
import cv2
from torchvision import transforms

from unet import UNet
from utils.data_vis import plot_img_and_mask, save_img_and_mask
from utils.dataset import BasicDataset
import glob
import matplotlib.pyplot as plt
from skimage import measure, exposure
from torch_snippets import *



def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()

    img = torch.from_numpy(BasicDataset.preprocess(full_img, scale_factor))

    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if net.n_classes > 1:
            probs = F.softmax(output, dim=1)
        else:
            probs = torch.sigmoid(output)

        probs = probs.squeeze(0)

        tf = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(full_img.size[1]),
                transforms.ToTensor()
            ]
        )

        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()

    return full_mask > out_threshold

def get_segment_crop(img, tol=0, mask=None):
    if mask is None:
        mask = img > tol
    return img[np.ix_(mask.any(1), mask.any(0))]

if __name__ == "__main__":
    net = UNet(n_channels=1, n_classes=1)
    # model = "checkpoints/crop_segmentation/CP_epoch5.pth"
    model = "checkpoints1/CP_epoch20.pth"
    scale = 0.3
    mask_threshold = 0.5
    logging.info("Loading model {}".format(model))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')
    net.to(device=device)
    net.load_state_dict(torch.load(model, map_location=device))

    logging.info("Model loaded !")

    

    # path_mask = "data/crop_segmentaion/imgs/raw_250.tif"
    path_mask = "data/crop_segmentaion/imgs/raw_1.tif"
    path_true_mask = "data/crop_segmentaion/masks/raw_1.png"
    logging.info("\nPredicting image {} ...".format(path_mask))
        
    # img = tiff.imread(fn)
    # print(img.shape)
    img = Image.open(path_mask)
        
    mask = predict_img(net=net,
                    full_img=img,
                    scale_factor=scale,
                    out_threshold=mask_threshold,
                    device=device)
    
    # print(np.unique(mask))
    # img = cv2.imread(fn)
    # img = skimage.io.imread(path_mask, plugin='tifffile')
    org_img = read("/media/stevel/files/other/SA/OB_test_seg/images/1_(1).png", -1)
    true_mask = skimage.io.imread(path_true_mask)
    
    mask = mask * 255
    true_mask = true_mask * 255
   
    plt.imshow(org_img, cmap='gray')
  
    # plt.axis('image')
    plt.axis('off')
    plt.legend()
    # plt.set_xticks([])
    # plt.set_yticks([])
    plt.show()
    # plt.savefig('binary_contours_2.png', dpi=300, bbox_inches='tight')