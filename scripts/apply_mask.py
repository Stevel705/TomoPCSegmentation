
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

path_img = "/home/stevel/Documents/ob/8/1/1.png"
path_true_mask = "data/crop_segmentaion/masks/raw_1.png"
# true_mask = skimage.io.imread(path_true_mask)

img = read(path_img, -1)
true_mask = read(path_true_mask, -1)
true_mask = true_mask * 255

# print(type(true_mask))
# print(type(img))
# dst = cv2.addWeighted(img, 0.5, true_mask, 0.5, 0)
# im_thresh_color = cv2.bitwise_and(img, true_mask)

plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), interpolation='none')
plt.imshow(true_mask, 'inferno', interpolation='none', alpha=0.5)

plt.axis('off')
# plt.gca().set_axis_off()
# plt.subplots_adjust(top = 1, bottom = 0, right = 1, left = 0, 
#             hspace = 0, wspace = 0)
# plt.margins(0,0)
plt.savefig('segm_mask.png', bbox_inches=0,  dpi=300 )
# plt.show()
# cv2.imwrite('apply_mask.jpg', im_thresh_color)
