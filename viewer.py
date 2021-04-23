import matplotlib.pyplot as plt
import numpy as np
from skimage.draw import rectangle_perimeter


def view_applied_mask(img, mask, ax, alpha=0):
    ax.imshow(img, cmap='gray', interpolation='none')

    mask = np.ma.masked_where(~mask, mask)
    ax.imshow(mask, cmap='hsv', interpolation='none', alpha=alpha)


def view_applied_rectangle(img, xlims, ylims, ax, color):
    img = np.zeros(img.shape, dtype=np.uint8)
    start = (xlims[0], ylims[0])
    end = (xlims[1], ylims[1])
    rr, cc = rectangle_perimeter(start, end=end, shape=img.shape)

    ax.plot(rr, cc, color=color)


def view_region(img, mask, ax, xlims, ylims, alpha=1):
    img = img[slice(*ylims), slice(*xlims)]
    mask = mask[slice(*ylims), slice(*xlims)]

    view_applied_mask(img, mask, ax, alpha=alpha)