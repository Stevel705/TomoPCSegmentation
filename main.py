import json
from pathlib import Path
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
from skimage import exposure, morphology
from skimage import morphology
from tqdm import tqdm

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
        yield name(n)


if __name__ == "__main__":
    num_of_slice = 1001
    sample = "gecko_123438"
    
    images = load_data(sample)
    shot_names = generate_sequential_file_names()

    # img2d = dm.get_img2d_from_database('1.tif', 'gecko')
    # for i in file_names:
    #     print(i)

    # fig, ax = plt.subplots(figsize=(7, 7))
    # ax.imshow(next(images))
    # dm.save_plot(fig, "plots", "1")

    for img2d, shot_name in tqdm(zip(images, shot_names), total=NUM_OF_TIF_SLICES):
        thresh = lambda x: x>np.percentile(x.ravel(), 50)
        eq_img = exposure.equalize_hist(img2d)
        

        img2d_mask = thresh(eq_img)


        img2d_mask = dots_scanner.get_small_pores_mask(img2d,
                                                    img2d_mask,
                                                    percentile_glob=90,
                                                    min_large_contour_length=1000,
                                                    window_size=100)
        img2d_mask = morphology.binary_opening(img2d_mask, selem=morphology.disk(2))
        
        
        dm.save_tif(img2d_mask, sample, shot_name)