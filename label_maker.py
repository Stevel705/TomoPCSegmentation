from scipy.ndimage import label
import data_manager as dm
import matplotlib.pyplot as plt
import numpy as np
import dots_scanner as ds


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield [i, i + n]


def remove_small_clusters(sample, 
                          min_cluster_size = 7000,
                          save_folder_tag="labeled"):
    for ranges in chunks(range(0, 2119), 100):
        ranges = ranges if ranges[1]<2120 else [2000, 2120]
        print("processing: ", ranges)
        masks3d = dm.assemble_3d_database(sample, ranges)
        masks3d_filtered = ds.find_big_ones_clusters(masks3d,
                                                     min_cluster_length=min_cluster_size,
                                                     min_cluster_order=None)
        masks3d_filtered = masks3d_filtered.astype(np.uint8)

        for n, mask2d in zip(range(*ranges), masks3d_filtered):
            dm.save_tif(mask2d, sample+"_"+save_folder_tag, dm.generate_tif_file_name(n))


def remove_big_clusters(sample, 
                        min_cluster_order = 2,
                        save_folder_tag="filtered"):
    for ranges in chunks(range(0, 2119), 100):
        ranges = ranges if ranges[1]<2120 else [ranges[0], 2120]
        print("processing: ", ranges)
        masks3d = dm.assemble_3d_database(sample, ranges)
        print(masks3d.sum())
        masks3d_filtered = ds.find_big_ones_clusters(masks3d,
                                                     min_cluster_length=None,
                                                     min_cluster_order=min_cluster_order)
        masks3d_filtered = masks3d_filtered.astype(np.uint8)

        print(masks3d_filtered.sum())
        masks3d_filtered = masks3d - masks3d_filtered
        print(masks3d_filtered.sum())

        for n, mask2d in zip(range(*ranges), masks3d_filtered):
            dm.save_tif(mask2d, sample+"_"+save_folder_tag, dm.generate_tif_file_name(n))


if __name__ == "__main__":
    sample = "gecko_123438_plus"
    remove_small_clusters(sample)



# labeled_array, _ = label(masks3d)
# cluster_labels, cluster_sizes = np.unique(labeled_array, return_counts=True)
# print("labeled")
# print(cluster_sizes)

# fig, ax = plt.subplots(figsize=(6,6))
# ax.hist(cluster_sizes[2:], bins = 100, log=True)

# dm.save_plot(fig, "labeling", "size hist")