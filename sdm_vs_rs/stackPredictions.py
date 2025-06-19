# -*- coding: utf-8 -*-
"""
Created on Tue Jun 10 10:11:59 2025

Combine several overlapping prediction rasters 


@author: E1008409
"""


import os 
import glob
import pandas as pd
import numpy as np
import rasterio as rio
import rioxarray as rxr
import matplotlib.pyplot as plt


# fp 
fp = '/mnt/d/users/e1008409/MK/DNASense/FIN/classification/XGB'
files = [f for f in glob.glob(os.path.join(fp, '*proba.tif'))]
m = os.path.basename(fp)
# read 
arrays = []
for f in files:
    with rio.open(f) as src:
        img = src.read()
        profile = src.profile
        names = src.descriptions
        print(img.shape)
    arrays.append(img)

# stack to array
array = np.stack(arrays)

# compute mean per pixel
arr_mean = np.nanmean(array, axis=0)
plt.imshow(arr_mean[1])
# save multiband preds
fp_out = os.path.join(fp, m + '_mean.tif')
with rio.open(fp_out, 'w', **profile) as dst:
    dst.write(arr_mean)

# hardcoded prediction # 
# set thresholds
percentiles = [75, 80, 85, 90]
thrs = [np.nanpercentile(arr_mean, p) for p in percentiles ]

# dataframe to store area change
area_df = pd.DataFrame(index=percentiles)

# select name
n = names[1] 
n_ind = names.index(n) 
# selected index from array
n_array = arr_mean[1]
for t in thrs:
    # filter by probability and convert to 0,1
    b_array = np.where(np.isnan(n_array), 0, n_array)
    b_array = np.where(b_array < t, 0, 1)
    b_array = np.expand_dims(b_array, 0)
    
    # compute pixel counts and convert to area
    area = np.count_nonzero(b_array) * 100 / 10000 # ha
    area_df.loc[percentiles[thrs.index(t)], m + '_threshold'] = round(t, 3)
    area_df.loc[percentiles[thrs.index(t)], m + '_area_ha'] = round(area, 2)
    
    # output profile
    outprof = profile.copy()
    outprof.update(count=1,
                   nodata=0,
                   dtype='uint8')
    name = n.split(' prob')[0]
    band_descriptions = ['Nodata', name]
    fp_out = os.path.join(fp, name + '_' + m + '_percntile_' + str(percentiles[thrs.index(t)]) + '.tif')
    
    with rio.open(fp_out, 'w', **outprof) as dst:
        dst.write(b_array)
        dst.set_band_description(1, name) #tuple(band_descriptions)

# save dataframe
area_df.to_csv(os.path.join(fp, m + '_predicted_fucus_area.csv'), sep=';')


# plot maps
import matplotlib.pyplot as plt
import matplotlib
# fp 
fp = '/mnt/d/users/e1008409/MK/DNASense/FIN/classification/'
models = ['RF', 'SVM', 'XGB']

# map files
maps = []
for m in models:
    m_maps = [f for f in glob.glob(os.path.join(fp, m, 'Fucus*.tif'))]
    maps.extend(m_maps)

# plot
fig, ax = plt.subplots(4,3, figsize=(12,20), sharex=True, sharey=True)
for m in models:
    maps_to_plot = [f for f in maps if m in f]
    for ma in maps_to_plot:
        # get plotting indices
        r = maps_to_plot.index(ma)
        c = models.index(m)
        with rio.open(ma) as src:
            predmap = src.read(1)
        ax[r,c].imshow(predmap, cmap = matplotlib.colors.ListedColormap(['white', 'purple']))
        ax[r,c].set_yticklabels([])
        ax[r,c].set_xticklabels([])
        if r == 0:
            ax[r,c].set_title(m)
plt.tight_layout()        
plt.show()
























