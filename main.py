import numpy as np
from tqdm import tqdm
from skimage.segmentation import random_walker
from porespy.filters import chunked_func
from skimage.filters import median
from skimage.morphology import ball

import data_manager as dm


NUM_OF_TIF_SLICES = 2120


def binarize_img(im, thrs1, thrs2):
    markers = np.zeros_like(im)
    markers[im > thrs1] = 1
    markers[im < thrs2] = 2
    t = random_walker(im, markers, beta=100)
    
    return t


if __name__ == "__main__":
    num_of_slice = 1001
    sample = "gecko_123438"

    shot_names = [dm.generate_tif_file_name(n) for n in range(NUM_OF_TIF_SLICES)]

    image_3d = dm.assemble_3d_server(sample, [400, 500])
    thrs1 = np.percentile(image_3d.flat[::5], 99)
    thrs2 = np.percentile(image_3d.flat[::5], 50)

    image_3d = median(image_3d, selem=ball(1))
    print(image_3d.shape)

    image_3d = chunked_func(func=binarize_img,
                            im=image_3d,
                            thrs1=thrs1,
                            thrs2=thrs2,
                            divs=[2, 20, 20],
                            overlap=10)

    # img2d = dm.get_img2d_from_database('1.tif', 'gecko')
    # for i in file_names:
    #     print(i)

    # fig, ax = plt.subplots(figsize=(7, 7))
    # ax.imshow(next(images))
    # dm.save_plot(fig, "plots", "1")

    for img2d, shot_name in tqdm(zip(image_3d, shot_names), total=NUM_OF_TIF_SLICES):
        dm.save_tif(img2d, sample, shot_name)
