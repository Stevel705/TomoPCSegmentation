# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.4.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %matplotlib inline

# %%
import numpy as np
import pylab as plt
import h5py
from tqdm.notebook import tqdm

from scipy import ndimage as ndi
from scipy.ndimage.morphology import (distance_transform_edt, binary_fill_holes,
                binary_closing, binary_opening, binary_dilation, binary_erosion)
from skimage.feature import peak_local_max
from skimage.measure import regionprops
from skimage.segmentation import watershed, random_walker, clear_border
from skimage.morphology import disk, ball, convex_hull_image
from pathlib import Path
import cv2

# %%
plt.rcParams['figure.facecolor'] = 'white'

# %%
data_folderes = ['/diskmnt/b/makov/robotom/74f73e80-3048-4b86-b67b-80b7d631bb85',]

# %%
df_number = 0
df = Path(data_folderes[df_number]).glob('tomo_rec.polymer_PDLG-7502 (for resorption) dry.h5')
df=list(df)[0]
data = h5py.File(df, 'r')['Reconstruction'][()]
data = data[10: -10, 500:-380 , 400:-500]
# if df_number == 0 :
#     data = data[210:,300:1000, 200:900]
# elif df_number == 1 :
#     data = data[1850:2400, 400:1600, 400:1700]
# elif df_number ==2:
#     data = data[1790:2320, 580:1300, 500:1260]

# %%
def create_mask3(img):
    thrs1 = np.percentile(img.flat[::5], 70)
    thrs2 = np.percentile(img.flat[::5], 25)  
    
#     need_squeeze = False
#     if img.ndim == 2:
#         need_squeeze = True
#         img = img[np.newaxis,:,:]
    res = np.zeros_like(img)
    N = 3
    overlap = 5
    border = np.rint(np.linspace(0, img.shape, N+1, True)).T.astype(int)
    b0_orig = border[:,:-1].copy()
    b0_orig[:,1:] = overlap
    x0_orig, y0_orig, z0_orig = b0_orig
    
    b0 = border[:,:-1].copy()
    b0[:,1:]-=overlap
    x0, y0, z0 = b0
    
    b1_orig = border[:,1:].copy()
    b1_orig[:, :-1] = overlap
    b1_orig[:,-1] = 0
    x1_orig, y1_orig, z1_orig = b1_orig
    
    b1 = border[:,1:].copy()
    b1[:, :-1]+=overlap
    x1, y1, z1 = b1
            
    for ix in range(len(x0)):
        for iy in range(len(y0)):
            for iz in range(len(z0)):
                t_img = img[x0[ix]:x1[ix],
                            y0[iy]:y1[iy],
                            z0[iz]:z1[iz]]
                markers = np.zeros_like(t_img)
                markers[t_img > thrs1] = 1
                markers[t_img < thrs2] = 2
                rt = random_walker(t_img, markers, beta=1)
#                 x11 = x1_orig[ix] if not x1_orig[ix] == 0 else 1
#                 y11 = y1_orig[iy] if not y1_orig[iy] == 0 else 1
#                 z11 = z1_orig[iz] if not z1_orig[iz] == 0 else 1
                
                tt = rt[x0_orig[ix]: x1[ix] - x1_orig[ix] - x0[ix], 
                        y0_orig[iy]: y1[iy] - y1_orig[iy] - y0[iy],
                        z0_orig[iz]: z1[iz] - z1_orig[iz] - z0[iz]]
        
#                 if np.prod(tt.shape) == 0:
#                     print(x0_orig[ix], x1[ix] - x1_orig[ix] - x0[ix], 
#                           y0_orig[iy], y1[iy] - y1_orig[iy] - y0[iy],
#                           z0_orig[iz], z1[iz] - z1_orig[iz] - z0[iz], rt.shape)
                    
                res[x0[ix] + x0_orig[ix] : x1[ix] - x1_orig[ix],
                    y0[iy] + y0_orig[iy] : y1[iy] - y1_orig[iy],
                    z0[iz] + z0_orig[iz] : z1[iz] - z1_orig[iz]] = tt
    return res


# %%
# sample3 = data[100:200, 100:200, 100:200] #[data.shape[0]//2-10:data.shape[0]//2+10]
sample3 = data
mask_vanila3 = create_mask3(sample3)-1

# %%
for N in range(3):
    t_slice = mask_vanila3.take(mask_vanila3.shape[N]//2, N)
    plt.figure(figsize=(15,15))
    plt.imshow(sample3.take(sample3.shape[N]//2, N), cmap=plt.cm.gray)
    plt.contour(t_slice, colors=['r'])
    plt.show()

# %%
mask_no_stones = binary_closing(mask_vanila3, ball(2),border_value=1)


# %%
for N in range(3):
    t_slice = mask_no_stones.take(mask_vanila3.shape[N]//2, N)
    plt.figure(figsize=(15,15))
    plt.imshow(sample3.take(sample3.shape[N]//2, N), cmap=plt.cm.gray)
    plt.contour(t_slice, colors=['r'])
    plt.show()

# %%
pores_closed = clear_border(mask_no_stones)

# %%
for N in range(3):
    t_slice = mask_no_stones.take(mask_vanila3.shape[N]//2, N)
    plt.figure(figsize=(15,15))
    plt.imshow(t_slice, cmap=plt.cm.gray)
#     plt.imshow(sample3.take(sample3.shape[N]//2, N), cmap=plt.cm.gray)
    plt.contour(pores_closed.take(mask_vanila3.shape[N]//2, N),colors='r')
    plt.show()

# %%
mask_sample = np.asarray([convex_hull_image(1-m) for m in mask_no_stones])

# %%
for N in range(3):
    t_slice = mask_no_stones.take(mask_vanila3.shape[N]//2, N)
    plt.figure(figsize=(15,15))
    plt.imshow(t_slice, cmap=plt.cm.gray)
#     plt.imshow(sample3.take(sample3.shape[N]//2, N), cmap=plt.cm.gray)
    plt.contour(mask_sample.take(mask_vanila3.shape[N]//2, N),colors='r')
    plt.show()

# %%
mask_pores = mask_sample*mask_no_stones

# %%
for N in range(3):
    t_slice = mask_pores.take(mask_vanila3.shape[N]//2, N)
    plt.figure(figsize=(15,15))
    plt.imshow(sample3.take(sample3.shape[N]//2, N), cmap=plt.cm.gray)
    plt.contour(t_slice, colors=['r'])
    plt.show()

# %%
pores = mask_pores

# %%
pores_t = pores #[200:300, 200:500, 200:500]
# mask_t = mask[200:300, 200:500, 200:500]
pores_dtf = distance_transform_edt(pores_t)
pores_dtf_r = distance_transform_edt(1-pores_t)

# %%
plt.figure(figsize=(15,15))
plt.imshow(pores[:,pores_dtf_r.shape[1]//2,:])
plt.colorbar(orientation='horizontal')
plt.show()


# %%
# # #https://scikit-image.org/docs/stable/auto_examples/segmentation/plot_watershed.html#sphx-glr-auto-examples-segmentation-plot-watershed-py
local_maxi = peak_local_max(pores_dtf, indices=False, 
                            threshold_abs=2, min_distance=10,# footprint=np.ones((3, 3, 3)),
                           labels=pores_t)# 
markers, num_features = ndi.label(local_maxi)#, np.ones((3, 3, 3)))
labels = watershed(-pores_dtf, markers, mask=pores_t)

# %%
markers, num_features = ndi.label(labels)
num_features

# %%
regions=regionprops(markers)

# %%
plt.figure(figsize=(15,15))
plt.imshow(pores_t[pores_t.shape[0]//2])
plt.colorbar(orientation='horizontal')
plt.contour(labels[markers.shape[0]//2],colors='r')

plt.show()


# %%
def rand_cmap(nlabels, type='bright', first_color_black=True, last_color_black=False, verbose=True):
    """
    Creates a random colormap to be used together with matplotlib. Useful for segmentation tasks
    :param nlabels: Number of labels (size of colormap)
    :param type: 'bright' for strong colors, 'soft' for pastel colors
    :param first_color_black: Option to use first color as black, True or False
    :param last_color_black: Option to use last color as black, True or False
    :param verbose: Prints the number of labels and shows the colormap. True or False
    :return: colormap for matplotlib
    """
    from matplotlib.colors import LinearSegmentedColormap
    import colorsys
    import numpy as np


    if type not in ('bright', 'soft'):
        print ('Please choose "bright" or "soft" for type')
        return

    if verbose:
        print('Number of labels: ' + str(nlabels))

    # Generate color map for bright colors, based on hsv
    if type == 'bright':
        randHSVcolors = [(np.random.uniform(low=0.0, high=1),
                          np.random.uniform(low=0.2, high=1),
                          np.random.uniform(low=0.9, high=1)) for i in range(nlabels)]

        # Convert HSV list to RGB
        randRGBcolors = []
        for HSVcolor in randHSVcolors:
            randRGBcolors.append(colorsys.hsv_to_rgb(HSVcolor[0], HSVcolor[1], HSVcolor[2]))

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]

        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Generate soft pastel colors, by limiting the RGB spectrum
    if type == 'soft':
        low = 0.6
        high = 0.95
        randRGBcolors = [(np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high),
                          np.random.uniform(low=low, high=high)) for i in range(nlabels)]

        if first_color_black:
            randRGBcolors[0] = [0, 0, 0]

        if last_color_black:
            randRGBcolors[-1] = [0, 0, 0]
        random_colormap = LinearSegmentedColormap.from_list('new_map', randRGBcolors, N=nlabels)

    # Display colorbar
    if verbose:
        from matplotlib import colors, colorbar
        from matplotlib import pyplot as plt
        fig, ax = plt.subplots(1, 1, figsize=(15, 0.5))

        bounds = np.linspace(0, nlabels, nlabels + 1)
        norm = colors.BoundaryNorm(bounds, nlabels)

        cb = colorbar.ColorbarBase(ax, cmap=random_colormap, norm=norm, spacing='proportional', ticks=None,
                                   boundaries=bounds, format='%1i', orientation=u'horizontal')

    return random_colormap

# %%
new_cmap = rand_cmap(np.max(markers), type='bright', first_color_black=True, last_color_black=False, verbose=True)

# %%
for i in range(3):
    plt.figure(figsize=(10,10))
    
    plt.imshow(labels.take(markers.shape[i]//2, i), cmap=new_cmap, interpolation='nearest')
#     plt.colorbar(orientation='horizontal')
    plt.show()


# %%
vol = [r.area for r in regions if r.area>1]
vol = sorted(vol)[:-2]
vol = np.asarray(vol, dtype=np.float32)*(10**3) # pixel size to um
d = np.power(vol/(4.*np.pi/3), 1./3)
# #volume of each pore
# vol = np.zeros((num_features+1), dtype=int)
# for x in tqdm_notebook(labels.flat):
#     vol[x] += 1
print(len(vol))

# %%
total_vol = np.sum(vol)

# %%
xv, yv = np.histogram(vol[1:], bins=30)
yv = (yv[1:]+yv[:-1])/2.
non_zero = np.argwhere(xv>0)
yv =yv[non_zero]
xv =xv[non_zero]
# xv = xv/np.sum(xv)*100  # to percent

plt.figure(figsize=(10,10))
# plt.plot(yv,xv,'o')
# plt.plot(yv,xv*yv/total_vol*100,'o')
plt.plot(np.power(yv/(4.*np.pi/3), 1./3),xv*yv/total_vol*100,'o')
plt.grid()
plt.show()

# %%
xd, yd = np.histogram(d[1:], bins=50)
xd = xd/len(xd)*100  # to percent
plt.figure(figsize=(10,10))
plt.semilogy(yd[1:],xd,'o')
plt.grid()
plt.show()

# %%
print(f"fraction of closed pore is {np.sum(pores_closed)/np.sum(pores)*100} %")

# %%
np.savez('pores_paper/tomo.npz', xv, yv)

# %%
from tomotools import save_amira, reshape_volume

# %%
import os


# %%
def reshape_volume(volume, reshape):
    if reshape == 1:
        return volume.astype('float32')
    
    res = np.zeros([s // reshape for s in volume.shape], dtype='float32')
    xs, ys, zs = [s * reshape for s in res.shape]
    for x, y, z in np.ndindex(reshape, reshape, reshape):
        res += volume[x:xs:reshape, y:ys:reshape, z:zs:reshape]
    return res / reshape ** 3

def save_amira(in_array, out_path, name, reshape=3, pixel_size=9.0e-3):
    data_path = str(out_path)
    os.makedirs(data_path, exist_ok=True)
    name = name.replace(' ', '_')
    with open(os.path.join(data_path, name + '.raw'), 'wb') as amira_file:
        reshaped_vol = reshape_volume(in_array, reshape)
        reshaped_vol.tofile(amira_file)
        file_shape = reshaped_vol.shape
        with open(os.path.join(data_path, 'tomo.' + name + '.hx'), 'w') as af:
            af.write('# Amira Script\n')
            # af.write('remove -all\n')
            template_str = '[ load -unit mm -raw ${{SCRIPTDIR}}/{}.raw ' + \
                           'little xfastest float 1 {} {} {}  0 {} 0 {} 0 {} ] setLabel {}\n'
            af.write(template_str.format(
                name,
                file_shape[2], file_shape[1], file_shape[0],
                pixel_size * reshape * (file_shape[2] - 1),
                pixel_size * reshape * (file_shape[1] - 1),
                pixel_size * reshape * (file_shape[0] - 1),
                name)
            )



# %%
save_amira(1-mask_no_stones, '.', 'sample', 1)

# %%
# #Raduis of each pore
# tt = local_maxi*pores_dtf  #todo.fixit
# xr, yr = np.histogram(tt.flat, bins=100)
# xr0, yr0 = np.histogram(np.power(vol,1./3), bins=1000)

# %%
# plt.figure(figsize=(15,15))
# plt.semilogy(yr[1:],xr[:],'o')
# plt.semilogy(yr0[2:],xr0[1:],'o')
# plt.xlim([0,20])
# plt.grid()
# plt.show()

# %% [markdown]
# # Plot figures

# %%
xz=np.load('pores_paper/tomo.npz')
xv = xz['arr_0']
yv = xz['arr_1']

# %%
pores_mur_data = np.loadtxt('pores_paper/3/PDLG_7502_init.vr', skiprows=3)
pores_mur_data = pores_mur_data[200:440] 
pores_mur_data[:,0]*=0.2e-3 #muliply to magic value from vvo and go to um
pores_mur_data[:,1]*=1e4 #muliply to magic value from vvo and go to um

# %%
hg_pores = np.loadtxt('pores_paper/4a2.csv')

# %%

plt.figure(figsize=(10, 6))
plt.loglog(pores_mur_data[::15,0],pores_mur_data[::15,1]*4./3*np.pi*pores_mur_data[::15,0]*5e5/100, '-o', label = 'SAS')
plt.xlabel('Pore diameter, um')
plt.loglog(hg_pores[:,0],hg_pores[:,1]/100, '-o', label= 'HG porosimetry')
plt.grid()
plt.loglog(np.power(yv/(4.*np.pi/3), 1./3),xv*yv/1e8, 'o', label='Laboratory microCT')  # 1e9 should be total_vol
plt.loglog(np.logspace(0,1.8,5), np.random.rand(5)/10+0.07, '--o', label='SR XPCT')
plt.ylabel("Cumulative pore volume, r.u.")
# plt.ylabel('Releative volume (separate for each method), rel.u.')
plt.legend()
plt.twinx()
plt.plot([3e-2, 1, 100],[0, 20, 100], 'k--x', label='Fraction of open pores, %')
plt.ylabel('Fraction of open pores, %')
plt.legend()

# %%
xz=np.load('pores_paper/tomo.npz')
xv = xz['arr_0']
yv = xz['arr_1']

# %%
xv

# %%
