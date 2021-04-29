import matplotlib.pyplot as plt
from skimage import exposure
from tqdm import tqdm

import data_manager as dm
from helper import crop


def preview(sample_name, n, contrast=True, server=True):
    if server:
        img = dm.get_img2d_from_server(sample_name, n)
    else:
        file_name = dm.generate_tif_file_name(n, True)
        img = dm.get_img2d_from_database(file_name,
                                         folder_name="crop_"+sample_name)

    if contrast:
        clip_limit = 0.1
    
        img = exposure.equalize_adapthist(img, clip_limit=clip_limit)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img, cmap="gray")
    dm.save_plot(fig, "sections", sample_name+"_"+str(n))


def crop_cube(sample_name, shape_2d, center_2d, slice_numbers):
    n_slices = slice_numbers[1]-slice_numbers[0]
    data = dm.load_data(sample_name, slice_numbers)

    for img, shot_name in tqdm(data, total=n_slices):
        img = crop(img, shape_2d, center=center_2d)

        dm.save_tif(img, "crop_"+sample_name, shot_name)


if __name__ == "__main__":
    sample_name = "gecko_123438"
    # for num_of_slice in [100, 250, 299]:
    #     server = False
    #     preview(sample_name, num_of_slice, server=server)

    shape_2d = [300, 300]
    center_2d = [500, 650]
    slice_numbers = [1000, 1300]

    crop_cube(sample_name, shape_2d, center_2d, slice_numbers)