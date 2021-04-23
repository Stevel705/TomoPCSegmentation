import json
from pathlib import Path
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure, morphology
from skimage import morphology

import data_manager as dm
import dots_scanner
import viewer

NUM_OF_TIF_SLICES = 2120


def load_data(sample_name, num_of_files=NUM_OF_TIF_SLICES):
    for n in range(num_of_files):
        yield dm.get_img2d_from_server(sample_name, n)


def generate_sequential_file_names(num_of_files=NUM_OF_TIF_SLICES):
    name = lambda num: "0" * (4-len(str(num))) + str(num)
    for n in range(num_of_files):
        yield name(n)+'.tif'


if __name__ == "__main__":
    num_of_slice = 1001
    images = load_data("gecko_123438")
    dm.save_tif(next(images), "gecko", "1")
    img2d = dm.get_img2d_from_database('1.tif', 'gecko')

    # file_names = generate_sequential_file_names()
    # for i in file_names:
    #     print(i)

    # fig, ax = plt.subplots(figsize=(7, 7))
    # ax.imshow(next(images))
    # dm.save_plot(fig, "plots", "1")

    fig, axes = plt.subplots(ncols=1, nrows=1, figsize=(14, 14))#, constrained_layout=True)
    #axes = axes.flatten()
    
    thresh = lambda x: x>np.percentile(x.ravel(), 50)
    eq_img = exposure.equalize_hist(img2d)
    
    #axes[1].imshow(eq_img, cmap='gray')

    img2d_mask = thresh(eq_img)
    #axes[2].imshow(img2d_mask, cmap='gray')


    img2d_mask = dots_scanner.get_small_pores_mask(img2d,
                                                   img2d_mask,
                                                   percentile_glob=90,
                                                   min_large_contour_length=1000,
                                                   window_size=100)
    img2d_mask = morphology.binary_opening(img2d_mask, selem=morphology.disk(2))
    print(img2d_mask.sum()/img2d_mask.size)

    viewer.view_applied_mask(img2d, img2d_mask, axes, alpha=1)
    #axes[3].imshow(img2d_mask, cmap='gray')
    #axes[3].imshow(morphology.binary_opening(img2d_mask, selem=morphology.disk(2)), cmap='gray')


    # squares = [([500, 800], [500,800]),
    #            ([400, 900], [400,900])]
    # axes[0].imshow(img2d, cmap='gray')
    # viewer.view_applied_rectangle(img2d, *squares[0], axes[0], color='red')
    # viewer.view_applied_rectangle(img2d, *squares[1], axes[0], color='green')
    # axes[0].set_title("original image")

    # #axes[1].imshow(exposure.equalize_adapthist(img2d, clip_limit=0.05), cmap='gray')
    # viewer.view_applied_mask(exposure.equalize_adapthist(img2d, clip_limit=clip_limit), img2d_mask, axes[1], alpha=1)
    # viewer.view_applied_rectangle(img2d, *squares[0], axes[1], color='red')
    # viewer.view_applied_rectangle(img2d, *squares[1], axes[1], color='green')
    # axes[1].set_title("adaptive contrast")

    # viewer.view_region(exposure.equalize_adapthist(img2d, clip_limit=clip_limit), img2d_mask, axes[2], *squares[0])
    # axes[2].set_title("RED box zoomed", fontdict={'color': 'red'})

    # viewer.view_region(img2d, img2d_mask, axes[3], *squares[1], alpha=0.3)
    # axes[3].set_title("GREEN box zoomed", fontdict={'color': 'green'})
    
    dm.save_plot(fig, "plots", "section2")