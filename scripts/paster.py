import numpy as np
from tqdm import tqdm

import configures
import data_manager as dm


if __name__ == "__main__":
    params = configures.InputParameters()
    sample = params.sample
    coords = params.coords_2d
    z_range = params.z_range

    masks3d = dm.assemble_3d_database(sample, z_range)
    image3d = dm.assemble_3d_server(sample, z_range)
    print(len(masks3d))
    print(len(image3d))

    new_folder = "gecko_123438_plus"
    for n, mask, image in zip(np.arange(*z_range), masks3d, image3d):
        mask_wide = np.zeros(image.shape, dtype=bool)
        mask_wide[coords] = mask
        dm.save_tif(mask_wide, new_folder, dm.generate_tif_file_name(n))
