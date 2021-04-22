import data_manager as dm
import matplotlib.pyplot as plt

if __name__ == "__main__":
    img2d = dm.get_img2d_from_database("reco_000600.tif")

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.imshow(img2d)
    dm.save_plot(fig, "plots", "section")