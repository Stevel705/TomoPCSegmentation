import matplotlib.pyplot as plt
from skimage import exposure
from tqdm import tqdm

import data_manager as dm
from helper import crop

def _load_data(sample_name, num_of_files):
    for n in range(*num_of_files):
        yield dm.get_img2d_from_server(sample_name, n)


def _generate_file_name(num, add_extention_tif=False):
    extention = ".tif" if add_extention_tif else ""
    return "0" * (4-len(str(num))) + str(num) + extention


def _generate_sequential_file_names(num_of_files):
    for n in range(num_of_files):
        yield _generate_file_name(n)


def preview(sample_name, n, contrast=True, server=True):
    if server:
        img = dm.get_img2d_from_server(sample_name, n)
    else:
        file_name = _generate_file_name(n, True)
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
    images = _load_data(sample_name, slice_numbers)
    file_names = _generate_sequential_file_names(n_slices)

    for img, shot_name in tqdm(zip(images, file_names), total=n_slices):
        img = crop(img, shape_2d, center=center_2d)

        dm.save_tif(img, "crop_"+sample_name, shot_name)


if __name__ == "__main__":
    sample_name = "gecko_123438"
    num_of_slice = 100
    server = False
    preview(sample_name, num_of_slice, server=server)

    # shape_2d = [500, 500]
    # center_2d = None
    # slice_numbers = [500, 1000]

    # crop_cube(sample_name, shape_2d, center_2d, slice_numbers)