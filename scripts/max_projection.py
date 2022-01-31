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





# for i, path_img in enumerate(glob.glob("data/10/*.tif")): #data/full/*/*.tif
for idx in range(1, 1001, 10):
    logging.info("\Loading image {} ...".format(idx))
    
    # img = tiff.imread(fn)
    # print(img.shape)
    # img = Image.open(path_img)
    
    # img = cv2.imread(fn)
    # img = skimage.io.imread(path_img, plugin='tifffile')
    
    with tifffile.TiffWriter(f'data/stacks_10/stack_{idx}.tif') as stack:
            for i in range(idx, idx+10):
                print(f"data/test_masked/reco_{i}.tif")
                stack.save(
                    tifffile.imread(f"data/test_masked/reco_{i}.tif"), 
                    photometric='minisblack', 
                    contiguous=True
                )

    img = skimage.io.imread(f'data/10/stack_{idx}.tif', plugin='tifffile')
    IM_MAX= np.max(img, axis=0)
    
    imsave(f'data/max_projection/max_{idx}.tif', IM_MAX)

    # logging.info("Visualizing results for image {}, close to continue ...".format(path_img))
    # filename = path_img.split('/')[-1:][0]
    # print("Image id:", i, filename)
    # plot_img_and_mask(img, mask) 
    # save_img_and_mask(img, mask, fn) 
    # skimage.io.imsave(f"data/test_masked/{filename}", masked_image, plugin='tifffile')