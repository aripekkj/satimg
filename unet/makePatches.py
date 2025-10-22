# -*- coding: utf-8 -*-
"""
Created on Fri Aug 22 10:52:47 2025

@author: E1008409
"""

import os
import glob
import numpy as np
from patchify import patchify
from PIL import Image, ImageEnhance
import rasterio as rio
from rasterio.transform import Affine

#fp
#fp = '/mnt/d/users/e1008409/MK/sun_glint/test_img/DJI_20240604155416_0593_D_edit.JPG'
fp = '/mnt/d/users/e1008409/MK/biodiversea/A6/hanko/mml/hanko_mml_masked_clip.tif'
fp_mask = '/mnt/d/users/e1008409/MK/biodiversea/A6/hanko/mml/rasterized_mask/hanko_mml_mask_v0_3.tif'

#fp = '/mnt/d/users/e1008409/MK/biodiversea/A6/hanko/wv/2011/'


# outdir
imgname = os.path.basename(fp).split('_')[-3]
outdir = os.path.join(os.path.dirname(fp), imgname + '_patches_256')
if os.path.isdir(outdir) == False:
    os.mkdir(outdir)
# patch size
psize=256
overlap=20

# open
#im = Image.open(fp)
#img = np.array(im)
#im.close()
# read image
with rio.open(fp) as src:
    img = src.read()
    meta = src.meta
img = img.transpose(1,2,0)
# select divisible area
def ImgModulo(image, patch_size):
    wm = image.shape[1] % patch_size
    hm = image.shape[0] % patch_size
    w = img.shape[1] - wm
    h = img.shape[0] - hm    
    return h,w
h,w = ImgModulo(img, psize)
img = img[0:h, 0:w, :]

# modulo Image width for patch size
#m = img.shape[1] % psize

# patchify and save patches
patches = patchify(img, (psize, psize, 3), step=psize-overlap) # use patch size - modulo as step to make division even

# =============================================================================
# for i in range(patches.shape[0]):
#     for j in range(patches.shape[1]):
#         # single patch
#         single_patch = patches[i,j,0,:,:,:]
# #        single_patch = np.transpose(single_patch, (2,0,1))
#         # save
#         patch_out = os.path.join(outdir, imgname + '_' + str(i) + '_' + str(j) + '.jpg')
#         im_p = Image.fromarray(single_patch)
#         #im_p = ImageEnhance.Contrast(im_p).enhance(2)
#         im_p.save(patch_out)
# 
# =============================================================================
# -------------------------------------------------- #

def computePatchTransform(patch_row, patch_col, patch_size, meta_transform):
    # compute transform for patch
    topy = meta_transform[5] - (patch_row*patch_size*meta_transform[0])
    topx = meta_transform[2] + (patch_col*patch_size*meta_transform[0])
    patch_tf = Affine(meta_transform[0], meta_transform[1], topx,
                      meta_transform[3], meta_transform[4], topy)
    return patch_tf
  
# create georeferenced patches for image 
for i in range(patches.shape[0]):
    for j in range(patches.shape[1]):
        # single patch
        single_patch = patches[i,j,0,:,:,:]
        single_patch = np.transpose(single_patch, (2,0,1))
        # patch transform
        patch_tf = computePatchTransform(i, j, psize-overlap, meta['transform'])        
        # don't write patch if all nodata
        if np.isnan(single_patch).all() == True:
            continue
        # update meta
        patchmeta = meta.copy()
        patchmeta.update(transform=patch_tf,
                         width=psize,
                         height=psize)
        # save
        patch_out = os.path.join(outdir, imgname + '_' + str(i) + '_' + str(j) + '.tif')
        with rio.open(patch_out, 'w', **patchmeta, compress='DEFLATE') as dst:
            dst.write(single_patch.astype(patchmeta['dtype']))


# ----------------------------------------------- #
# output folder for mask tiles
maskdir = outdir + '_masks'
# read image mask
with rio.open(fp_mask) as src:
    mask = src.read()
    maskmeta = src.meta
# transpose
mask = mask.transpose(1,2,0)
# patchify and save patches
mask_patches = patchify(mask, (psize, psize, 1), step=psize-overlap) # use patch size - modulo as step to make division even

# create georeferenced patches for mask
for i in range(mask_patches.shape[0]):
    for j in range(mask_patches.shape[1]):
        # single patch
        single_patch = mask_patches[i,j,0,:,:,:]
        single_patch = np.transpose(single_patch, (2,0,1))
        # patch transform
        patch_tf = computePatchTransform(i, j, psize-overlap, maskmeta['transform'])        
        # don't write patch if all nodata
        if np.isnan(single_patch).all() == True:
            continue
        # update meta
        maskpatchmeta = maskmeta.copy()
        maskpatchmeta.update(transform=patch_tf,
                         width=psize,
                         height=psize)
        # save
        patch_out = os.path.join(maskdir, imgname + '_' + str(i) + '_' + str(j) + '_mask.tif')
        with rio.open(patch_out, 'w', **maskpatchmeta, compress='DEFLATE') as dst:
            dst.write(single_patch.astype(maskpatchmeta['dtype']))
# list masks
maskfiles = [f for f in glob.glob(os.path.join(maskdir, '*.tif'))]
# keep masks where all values are not nodata
for file in maskfiles:
    with rio.open(file) as src:
        gtpatch = src.read()
    # check if nonzeros
    if np.all(gtpatch==0) == True:
        os.remove(file)


# --------------------------------------------- #
# converting .tif masks saved from gimp
# read 
for f in maskfiles:
    with rio.open(f) as src:
        mask = src.read()
        mprof = src.profile
    # select mask layer
    mask = mask[1]
    mask = np.expand_dims(mask, axis=0)
    mask = np.logical_not(mask).astype(int)
    mprofout = mprof.copy()
    mprofout.update(count=1)
    maskout = os.path.join(maskdirout, os.path.basename(f))
    with rio.open(maskout, 'w', **mprofout) as dst:
        dst.write(mask)

# ----------------------------------------------- #

# testing 
from PIL import ImageEnhance
import matplotlib.pyplot as plt
# list patches
files = [f for f in glob.glob(os.path.join(outdir, '*.jpg'))]
file = files[1017]
print(file)
im = Image.open(file)
ime = ImageEnhance.Contrast(im).enhance(2)

fig,ax = plt.subplots(1,2)
ax[0].imshow(im)
ax[1].imshow(ime)
plt.tight_layout()
plt.show()

im.close()


