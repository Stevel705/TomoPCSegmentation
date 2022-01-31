import skimage.io
import numpy as np
import matplotlib.pyplot as plt
import glob
import logging
import tifffile
import cv2

from skimage.filters import threshold_yen
from skimage.exposure import rescale_intensity
from skimage.io import imread, imsave


# for idx in range(1, 1002, 1):
#     logging.info("\Loading image {} ...".format(idx))
    
#     # img = tiff.imread(fn)
#     # print(img.shape)
#     # img = Image.open(path_img)
    
#     # img = cv2.imread(fn)
#     # img = skimage.io.imread(path_img, plugin='tifffile')
#     print(f"Image: {idx}")
#     img = imread(f"data/preprocessing/max_projection/max_1.tif", plugin='tifffile')
#     img = np.array(img, dtype=np.float32)
#     imsave(f"data/preprocessing/max_projection/max_1.tif", img)

img = imread(f"data/preprocessing/max_projection/max_1.tif", plugin='tifffile')
img = np.array(img, dtype=np.float32)
imsave(f"data/preprocessing/max_projection/max_1.tif", img)