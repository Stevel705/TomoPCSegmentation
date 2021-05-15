import numpy as np
from tqdm import tqdm
from skimage.segmentation import random_walker
from porespy.filters import chunked_func
from skimage.filters import median
from skimage.morphology import ball
from helper import crop
import time

import data_manager as dm


NUM_OF_TIF_SLICES = 500
count = 0


def binarize_img(im, thrs1, thrs2, max_iter):
    start_time = time.time()
    markers = np.zeros_like(im)
    markers[im > thrs1] = 1
    markers[im < thrs2] = 2
    t = random_walker(im, markers, beta=100)
    global count
    print("count", count, " of ", max_iter, f" %: {count/max_iter:.2f}" ,"| time (min): ", (time.time() - start_time) / 60)
    count += 1
    
    return t<2


if __name__ == "__main__":
    num_of_slice = 1001
    sample = "gecko_123438"

    step = 300
    z_range = [300, 600]
    coords = tuple([slice(300, 850), slice(300, 850)])

    start_time = time.time()
    shot_names = [dm.generate_tif_file_name(n) for n in range(*z_range)]

    image_3d = dm.assemble_3d_server(sample, z_range)[:, 350:950, 300:850]

    thrs1 = 0.000266 #np.percentile(image_3d.flat[::5], 95)
    thrs2 = -1.54e-05 # np.percentile(image_3d.flat[::5], 35)

    image_3d = median(image_3d, selem=ball(1))
    print(image_3d.shape)

    divs = np.array(image_3d.shape) // np.array([20, 40, 40])
    image_3d = chunked_func(func=binarize_img,
                            im=image_3d,
                            thrs1=thrs1,
                            thrs2=thrs2,
                            max_iter=divs[0]*divs[1]*divs[2],
                            divs=divs,
                            overlap=(3, 5, 5),
                            cores=2)

    for img2d, shot_name in tqdm(zip(image_3d, shot_names), total=len(image_3d)):
        dm.save_tif(img2d, sample, shot_name)
    end_time = time.time()
    print("cumulative time (min): ", (end_time-start_time)/60)
