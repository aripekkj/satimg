# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 13:13:09 2023




@author: E1008409
"""


import sys
import os
os.getcwd()
os.chdir('/mnt/c/users/e1008409/.spyder-py3')
import time
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
import matplotlib.pyplot as plt

from collections import Counter
from indices import normalizedDifference, EG, GCC, TGI

from skimage.segmentation import felzenszwalb
from rasterstats import zonal_stats
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, StratifiedKFold, RepeatedStratifiedKFold, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    
def computePCA(img):
    # flatten each band
    flat_data = np.empty((img.shape[1]*img.shape[2], img.shape[0]))

    for i in range(img.shape[0]):
        band = img[i,:,:]
        flat_data[:,i-1] = band.flatten()
    # replace nan
    flat_data = np.where(np.isnan(flat_data), 0, flat_data)
    # delete nan
    test = np.delete(flat_data, np.where(flat_data == 0), axis=0)

    # sklearn PCA
    from sklearn.decomposition import PCA
    pca = PCA()
    pca.fit(test)  
    # cumulative variance
    var_cumu = np.cumsum(pca.explained_variance_ratio_)*100
    # apply pca
    pca_out = pca.transform(flat_data)
    # reshape 
    pca_out = np.reshape(pca_out, (img.shape[1], img.shape[2], img.shape[0]))
    pca_out = np.transpose(pca_out, (2,0,1))
    # select pc's that explain 99% of variation
    #pca_out = pca_out[0:3,:,:]
    return pca_out, var_cumu    


fp_img = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Finland/S2_LS1_20180715_rrs_v1_clip_ext_median3x3_filt.tif'
#fp_toa_img = '/mnt/d/users/e1008409/MK/S2/T34VENVEP_20180715T100031__B02B03B04B08_merge_msk.tif'
#fp_img = '/mnt/d/users/e1008409/MK/S2/c2rcc/subset_L1C_T34VEP_A006411_20160913T100023_s2resampled_10_C2RCC_Rrs_clip_median3x3_filt_masked_WV2extent.tif'
fp_depth = '/mnt/d/users/e1008409/MK/syvyysmalli/depth_20210126_clipped_transformed_32634_selkameri_south_ext.tif'#poriWV2.tif #T34VEP
#fp_img = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Finland/Pori/S2_LS1_20180715_rrs_v1_clip_ext_pori.tif'
#fp_depth = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Finland/Pori/depth_10m_pori.tif'

# read
with rio.open(fp_img) as src:
    img = src.read((2,3,4,5,6,7))
    meta = src.meta

with rio.open(fp_depth) as src:
    depth = src.read()
    dmeta = src.meta
# change nodata
img = np.where(img == meta['nodata'], np.nan, img)
meta.update(nodata=np.nan)
depth = np.where(depth == dmeta['nodata'], np.nan, depth)
dmeta.update(nodata=np.nan) #update metadata
# mask any negative pixel in all bands
#img[:, np.any(img <= 0, axis=0)] == meta['nodata']
# mask img negative pixels from depth
#depth = np.where(img < 0, np.nan, depth)

# mask deep areas or where depth nan 
img = np.where((depth < -5) | (np.isnan(depth)), np.nan, img)
# convert depth to abs values
depth = np.abs(depth)

# compute normalized difference
ndvi = normalizedDifference(img, (5,2))
#ndti = normalizedDifference(img, (4,2)) # red-edge/green
# mask image 
img = np.where(ndvi > 0.1, meta['nodata'], img)
#img = np.where((ndti < 0) & (ndti > -0.3), meta['nodata'], img) 

# nodata mask
if meta['nodata'] == None:
    print('Nodata not defined')
elif np.isnan(meta['nodata']):
    nodatamask = np.where(np.isnan(img[0]), True, False)
else:
    nodatamask = np.where(img[0] == meta['nodata'], True, False)
upmeta = meta.copy()
upmeta.update(count=6)
# save masked image
img_out = fp_img.split('.tif')[0] + '_depth_masked.tif'
with rio.open(img_out, 'w', **upmeta, compress='LZW') as dst:
    dst.write(img.astype(upmeta['dtype']))

#####################
# 2. segment image
# set some parameters
n = 5
sig = 0.0005
s = 0.050

# segmentation algorithms
start = time.time()
segments = felzenszwalb(img, scale=s, sigma=sig, min_size=n, channel_axis=0)
#segments = quickshift(img, ratio=1.0, kernel_size=5, max_dist=10, convert2lab=False)
#segments = slic(img, n_segments=n, compactness=0.01, sigma=sig, convert2lab=True, channel_axis=0, mask=datamask[0])
#segments = watershed(img_scaled)
end = time.time()
#segments = felzenszwalb(img_scaled, scale=1, sigma=0.9, min_size=20, multichannel=True)
elapsed = end - start
print('Time elapsed: %.2f' % elapsed, 'seconds')
# mask nodata area
segments = np.where(nodatamask == True, 0, segments)
# expand dims
segments = np.expand_dims(segments, axis=0)
# counts of unique values, ie. size of segments
seg_n = np.bincount(segments.reshape(-1))
seg_no = np.nonzero(seg_n)[0]
seg_nos = np.vstack((seg_no, seg_n[seg_no]))
seg_no_mean = np.mean(seg_nos[1][1:])
print('Average number of pixels in segment:', seg_no_mean)


# save the segments
#segdir = os.path.join(os.path.dirname(os.path.dirname(fp)))
segdir = os.path.join(os.path.dirname(fp_img), 'segment')
if os.path.isdir(segdir) == False:
    os.mkdir(segdir)    
segfile = os.path.join(segdir, os.path.basename(fp_img).split('.')[0] + 'ndvimasked_segments_rgbre_scale_' + str(n) + '_' +
                       str(s) + '_' + str(sig) + 'sigma_felzenszwalb.tif') #sigma_felzenszwalb
# update metadata
upmeta = meta.copy()
upmeta.update(
    dtype = rio.uint32,
    nodata = 0,
    count = 1)
# save
with rio.open(segfile, 'w', **upmeta, compress='LZW') as dst:
    dst.write(segments.astype(upmeta['dtype']))
# read segments image
with rio.open(segfile) as src:
    segments = src.read()

# compute pca
pca, varcumu = computePCA(img)
pca = np.where(nodatamask == True, np.nan, pca)
print(varcumu) # print cumulative variance explained

# save pca
pcadir = os.path.join(os.path.dirname(fp_img), 'pca')
if os.path.isdir(pcadir) == False:
    os.mkdir(pcadir)
pcaout = os.path.join(pcadir, 'pca.tif')
pcameta = meta.copy()
pcameta.update(count=pca.shape[0])
with rio.open(pcaout, 'w', **pcameta, compress='LZW') as dst:
    dst.write(pca.astype(pcameta['dtype']))
# compute indices
#gbi = normalizedDifference(img, (1,0))
eg = EG(img, (0,1,2))
#gcc = GCC(img, (0,1,2))
#tgi = TGI(img, [0,1,2], 'S2A')
#ndvi = normalizedDifference(img, (7,3))
#ndti = normalizedDifference(img, (2,1))
# band ratios
bg = img[0]/img[1]
gr = img[1]/img[2]
bg = np.expand_dims(bg, axis=0)
gr = np.expand_dims(gr, axis=0)
eg = np.expand_dims(eg, axis=0)
#gbi = np.expand_dims(gbi, axis=0)
ndvi = np.expand_dims(ndvi, axis=0)

# stack to image
img = img.transpose(1,2,0)

#ndti = ndti.transpose(1,2,0)
#tgi = tgi.transpose(1,2,0)
#gcc = gcc.transpose(1,2,0)
eg = eg.transpose(1,2,0)
bg = bg.transpose(1,2,0)
gr = gr.transpose(1,2,0)
#gbi = gbi.transpose(1,2,0)
ndvi = ndvi.transpose(1,2,0)
depth = depth.transpose(1,2,0)
pca = pca.transpose(1,2,0)
stack = np.dstack((img, eg, bg, gr, ndvi, pca[:,:,0:3])) #eg, bg, gr, ndvi,

img = np.transpose(img, (2,0,1))
depth = np.transpose(depth, (2,0,1))
# save stack
stack = np.transpose(stack, (2,0,1))
# update nodata
stack = np.where(nodatamask == True, np.nan, stack)
stack_out = os.path.join(os.path.dirname(fp_img), os.path.basename(fp_img).split('.tif')[0] + '_vis_ind_pca_stack.tif')
stack_meta = meta.copy()
stack_meta.update(dtype='float32',
                  nodata=np.nan,
                  count=stack.shape[0])
with rio.open(stack_out, 'w', **stack_meta, compress='LZW') as dst:
    dst.write(stack.astype(stack_meta['dtype']))

# drop unnecessary layers and release memory
import gc
del img 
del pca
del depth
gc.collect()

# read stack
with rio.open(stack_out) as src:
    stack = src.read()

# read ground truth and skf indices
fp_gt = '/mnt/d/users/e1008409/MK/Velmu-aineisto/sdm_vs_rs/velmudata_07112022_selkameri_south_bounds_edit.gpkg'
#fp_ind = '/mnt/d/users/e1008409/MK/Velmu-aineisto/sdm_vs_rs/test_stratifiedKFold_indices.json'
fp_gt_pts = '/mnt/d/users/e1008409/MK/Velmu-aineisto/sdm_vs_rs/velmudata_07112022_selkameri_south_bounds_edit_S2cell.gpkg'

# read
gdf = gpd.read_file(fp_gt_pts, engine='pyogrio')
# edit
gdf = gdf.drop(columns=[*gdf.columns[-23:-1]], axis=1) # if reading already sampled points
#gdf.new_class = np.where((gdf.new_class == 2) & (gdf.field_depth > 3), 6, gdf.new_class)
#gdf = gdf[gdf.new_class != 0]
#gdfout = fp_gt.split('.')[0] + '_edit.gpkg'
#gdf.to_file(gdfout, driver='GPKG', engine='pyogrio')
#print(len(gdf[(gdf.fucaceae_cov >= 30) & (gdf.field_depth < 2)]))
# plot histgram of months
gdf.pvm = pd.to_datetime(gdf.pvm)
gdf.pvm.groupby([gdf.pvm.dt.year, gdf.pvm.dt.month]).count().plot(kind='bar') #plot

# select pts by date
gdf = gdf[(gdf.year >= 2010) & (gdf.year <= 2022)]

# check crs
src = rio.open(fp_img)
if gdf.crs.to_epsg() != src.crs.to_epsg():
    print('Reprojecting points')
    gdf = gdf.to_crs(epsg=src.crs.to_epsg())
src.close()

# plot hist coverage
fig, ax = plt.subplots()
ax.hist(gdf.fucaceae_cov)
plt.show()

# check geometry, explode if MultiPoint
if gdf.geometry.geom_type.str.contains('MultiPoint').any() == True:
    sp = gdf.geometry.explode()
    # get point coords
    coords = [(x,y) for x,y in zip(sp.x, sp.y)]
else:
    # get point coords
    coords = [(x,y) for x,y in zip(gdf.geometry.x, gdf.geometry.y)]

# open image and sample segments
src = rio.open(segfile)
gdf['segment_ids'] = [x for x in src.sample(coords)]
# close dataset
src.close()
# extract list
gdf['segment_ids'] = gpd.GeoDataFrame(gdf.segment_ids.tolist(), index=gdf.index)

# sample stack for pixel based classification
src = rio.open(stack_out) 
gdf['img_stack'] = [x for x in src.sample(coords)]
# close dataset
src.close()
# extract list
stack_list = ['Band2', 'Band3', 'Band4', 'Band5', 'Band6', 'Band7', 'eg', 'bg', 'gr', 'ndvi',
              'pca1', 'pca2', 'pca3'] #'eg', 'bg', 'gr', 'ndvi',

gdf[stack_list] = gpd.GeoDataFrame(gdf.img_stack.tolist(), index=gdf.index)

# drop
gdf = gdf.drop('img_stack', axis=1)
gdf = gdf.dropna(subset=stack_list) # exclude raster nodata pts

###################################
# handle possible duplicates, ie. points that are within same raster cell
# pts that have exact same sampled values<
subset = ['Band2', 'Band3', 'Band4', 'Band5', 'Band6', 'Band7']
gdf['duplicate'] = gdf.duplicated(subset=subset, keep=False)
print('Duplicated rows:', len(gdf[gdf.duplicate==True]))
# create id for finding duplicate rows
gdf['duplicate_id'] = gdf.groupby(subset).ngroup()
# select ones occurring more than once
v = gdf.duplicate_id.value_counts()
v = list(v.index[v.gt(1)])
# check if duplicates have same or different class
for i in v:
    sel = gdf[gdf.duplicate_id == i]

    if len(sel) > 1:
        print(sel[['sykeid', 'new_class']])
        # find most common value
        c = Counter(sel.new_class)
        val, count = c.most_common()[0]
        # if equal count of different values, get class from row with highest vegetation cover
        if count == 1:
            sel_id = sel['sykeid'][sel.savcov == sel.savcov.max()].iloc[0] # select id where sav coverage is highest
            droplist = sel['sykeid'][sel['sykeid'] != sel_id] # list of id's to drop
            # drop other ids from gdf
            gdf = gdf[~gdf.sykeid.isin(droplist)]
        else:
            print('Majority value in', i, val, 'with count', count)
            sel_id = sel['sykeid'][sel.new_class == val].iloc[0] # keep one row with majority value
            droplist = sel['sykeid'][sel['sykeid'] != sel_id] # list of id's to drop
            # drop other ids from gdf
            gdf = gdf[~gdf.sykeid.isin(droplist)]

# select columns
stack_list.append('new_class')
stack_list.append('field_depth')
stack_list.append('segment_ids')
stack_list.append('sykeid')
stack_list.append('geometry')
#stack_list.append('fucaceae_cov')
gdf_train = gdf[stack_list]
# drop nan rows
gdf_train = gdf_train.dropna()
# new index row starting from 0
#gdf['index2'] = range(0, len(gdf), 1)
#gdf_train.new_class = np.where((gdf_train.field_depth > 2) & ((gdf_train.new_class == 1) | (gdf_train.new_class == 2)), 6, gdf_train.new_class)
# drop duplicates
#gdf_train_d = gdf_train.drop_duplicates(subset=['Band2', 'Band3', 'Band4', 'Band5', 'Band6', 'Band7', 'Band8', 'Band8a'])
###################################
print(len(gdf[(gdf.fucaceae_cov > 50) & ( gdf.field_depth < 1)]))

# save
gdf_out = fp_gt.split('.')[0] + '_S2_cell_6classes.gpkg'
gdf.to_file(gdf_out, driver='GPKG', engine='pyogrio')
gdf = gpd.read_file(gdf_out, engine='pyogrio')

##############
# use below for segment based classification

data = gdf[['segment_ids', 'savcov', 'new_class']]
data = data.loc[(gdf_train.index.to_list())]
data = data.rename(columns={'new_class': 'classes'})

# remove any nan rows
data = data[data.classes.notna()]
data = data.astype(int)
# segment ground truth
gt_d = dict()

# get unique segment ids and class value
for i in np.unique(data.segment_ids):
    test = data[data.segment_ids == i]

    # test if multiple classes within segment
    if len(np.unique(test.classes)) > 1:
        # find most common value
        c = Counter(test.classes)
        val, count = c.most_common()[0]
        # if equal count of different values, get class from row with highest vegetation cover
        if count == 1:
            test['classes'][test.savcov == test.savcov.max()]
        else:
            print('Majority value in', i, val, 'with count', count)
            # save to dict
            gt_d[i] = val
    else:
        gt_d[i] = test.classes.iloc[0] # get first class value if all values are same

# to dataframe
gt_d = pd.DataFrame.from_dict(gt_d, orient='index', columns=['classes'])
# reset index and rename column for later
gt_d = gt_d.reset_index()
gt_d = gt_d.rename(columns={'index': 'segment_id'})

# join geometry to unified gt
gt_d = gt_d.merge(gdf[['segment_ids', 'geometry']], how='left', left_on='segment_id', right_on='segment_ids')

# save
gt_d = gpd.GeoDataFrame(gt_d, geometry=gt_d.geometry, crs='32634')
gt_d.to_file(os.path.join(segdir, 'gt_d.gpkg'), driver='GPKG', engine='pyogrio')
############################
# Assign class labels to segments
############################
# new layer for segment classes
new_layer = np.zeros(shape=segments.shape)

# assign labels, this takes a little time
for idx, row in gt_d.iterrows():
    new_layer = np.where(segments == row.segment_ids, row.classes, new_layer) 

gt = new_layer
# save gt
#gtout = gt.reshape((1,meta['height'], meta['width']))
# mask
#gtout = np.where(nodatamask == True, 0, gtout)

#gtout = np.where(gtout == 3, 5, gtout)
gt_out = os.path.join(segdir, 's' + str(s) + 'sig' + str(sig) + 'n' + str(n) + 's2_gt.tif')
with rio.open(gt_out, 'w', **upmeta, compress='LZW') as dst:
    dst.write(gt.astype(upmeta['dtype']))
# read gt
with rio.open(gt_out) as src:
    gt = src.read()
gt = gt.reshape(-1)
# sample gt segments
# get point coords

if gdf_train.geometry.geom_type.str.contains('MultiPoint').any() == True:
    sp = gdf_train.geometry.explode()
    # get point coords
    coords = [(x,y) for x,y in zip(sp.x, sp.y)]
else:
    # get point coords
    coords = [(x,y) for x,y in zip(gdf_train.geometry.x, gdf_train.geometry.y)]

#coords = [(x,y) for x,y in zip(gdf_train.geometry.x, gdf_train.geometry.y)]
src = rio.open(gt_out)
gdf_train['gt_sampled'] = [x for x in src.sample(coords)]
# close dataset
src.close()
# extract list
gdf_train['gt_sampled'] = gpd.GeoDataFrame(gdf_train.gt_sampled.tolist(), index=gdf_train.index)
# drop 0's
gdf_train = gdf_train[gdf_train.gt_sampled != 0]
#gdf_train = gdf_train[gdf_train.gt_sampled != 5]


###################################
# Plot bands and classes
#gdf_train.new_class = np.where(gdf_train.new_class == 3, 5, gdf_train.new_class)
#gdf_train.new_class = np.where(gdf_train.new_class == 7, 1, gdf_train.new_class)

# group by class and compute average spectra
# select cols
df_s = gdf_train[['new_class', 'Band1', 'Band2', 'Band3', 'Band4', 'Band5', 'Band6', 'Band7', 'Band8', 'Band8a']]
# groupby class
dg = df_s.groupby('new_class').mean()
# select band columns
#dg = dg[cols]


#%%
# ------------------------------------------------ #
# 1. create spectral plot. Average of all points
fig, ax = plt.subplots()

ax.plot(dg.loc[1], color='green', label='Mixed SAV')
ax.plot(dg.loc[2], color='#A27C1F', label='Brown algae')
ax.plot(dg.loc[3], color='yellow', label='Sand')
ax.plot(dg.loc[4], color='red', label='Bare')
ax.plot(dg.loc[5], color='purple', label='Low SAV')
ax.plot(dg.loc[6], color='blue', label='Deep water')
ax.plot(dg.loc[7], color='#FFD7A0', label='Turbid')
#ax.plot(dg.iloc[4], color='#92E335', label='Vascular')
ax.set_xticklabels([442.7, 492.4, 559.8, 664.6, 704.1, 740.5, 782.8, 832.8, 864.7]) #442.7, 492.4, 559.8, 664.6, 704.1, 740.5, 782.8, 832.8, 864.7, 945.1, 1373.5, 1613.7, 2202.4
ax.set_xlabel('Wavelength (nm)')
ax.set_ylabel('Remote sensing reflectance (Rrs) $sr^{-1}$')
ax.legend()
ax.grid(True)
ax.set_title('Average spectra per class, S2')

plt.tight_layout()
# save 
plotout = os.path.join(os.path.dirname(fp_img), 'plots', 'Spectra_per_class_S2_VENVEP_acolite.png')
plt.savefig(plotout, dpi=150, format='PNG')

# spectral plot by depth
dzones = [] 
n = 0
while n < 4:
    dzone = (n, n+1)
    dzones.append(dzone)
    n += 1

bands = ['Band2', 'Band3', 'Band4', 'Band5', 'Band6', 'Band7']
wavels = [492.4, 559.8, 664.6, 704.1, 740.5, 782.8]#, 832.8, 864.7]
# deep water class
dw = df_s[df_s['new_class'] == 7].groupby('new_class').mean()
dw = dw[cols[:-1]]

# create spectral plot of each depth zone
fig, ax = plt.subplots(2,2, figsize=(12,8))

for i in dzones:
    # make selection
    dfsel = gdf[(gdf.field_depth > i[0]) & (gdf.field_depth <= i[1])]
    # drop class 4
    #dfsel = dfsel[dfsel.classes < 4]
    # group
    dfsel_group = dfsel.groupby('new_class').mean(numeric_only=True)    
    # drop 0
    if 0 in dfsel_group.index:
        dfsel_group = dfsel_group.drop(0)
    # select band columns
    dfsel_group = dfsel_group[bands]
    # reset index
    #dfsel_group = dfsel_group.reset_index()
    
    # row, col params for multiplot
    if i[0] < 2:
        r = 0
        c = i[0]
    elif i[0] >= 2:
        r = 1
        c = i[0] - 2
    
    cl_in_dzone = list(np.unique(dfsel_group.index))
    print(cl_in_dzone)
    colors = ['green', '#888f29', 'yellow', 'purple', 'blue', 'pink']
    labs = ['Mixed SAV', 'Brown algae', 'Bare', 'Sparse SAV', 'Deep water', 'Turbid']
    
    for j in cl_in_dzone:
        # get index
        base = dfsel_group.index.get_loc(j)
        # plot
        ax[r,c].plot(dfsel_group.loc[j], linewidth=0.7, color=colors[base], label=labs[base])
    #    ax[r,c].plot(dfsel_group.loc[2], linewidth=0.7, color=, label=)
    #    ax[r,c].plot(dfsel_group.loc[3], linewidth=0.7, color=, label=)
  #  ax[r,c].plot(dw.loc[4], linewidth=0.7, linestyle='--', color='#0034CA', label='Deep water') 
    ax[r,c].set_title(str(i[0]) + '-' + str(i[1]) + ' meters', fontsize=10) 
    ax[r,c].grid(True, color='#B0B0B0')
    ax[r,c].set_xticks(list(np.arange(len(bands))))     
    ax[r,c].set_xticklabels(wavels, rotation=0)
    ax[r,c].tick_params(axis='x', which='major', pad=-1, labelsize=8)
    ax[r,c].tick_params(axis='y', which='major', pad=-1, labelsize=8)
    ax[r,c].set_facecolor(color='#D9D9D9')
"""    
# select deep water
dfsel_deep = df[df.field_depth > 5].groupby('new_class').mean(numeric_only=True)
dfsel_deep = dfsel_deep[cols]

# add to plot
ax[2,1].plot(dfsel_deep.loc[1], linewidth=0.7, color='#443726', label='Bare bottom or \nsparse SAV')
ax[2,1].plot(dfsel_deep.loc[2], linewidth=0.7, color='green', label='Mixed SAV')
ax[2,1].plot(dfsel_deep.loc[3], linewidth=0.7, color='#A27C1F', label='Dense Zostera')
ax[2,1].plot(dw.loc[4], linewidth=0.7, linestyle='--', color='#0034CA', label='Deep water')    
ax[2,1].set_title('> 5 meters', fontsize=10)
ax[2,1].grid(True, color='#B0B0B0')
ax[2,1].set_xticks(list(np.arange(len(bands))))     
ax[2,1].set_xticklabels(bands)
ax[2,1].tick_params(axis='x', which='major', pad=-1, labelsize=8)
ax[2,1].tick_params(axis='y', which='major', pad=-1, labelsize=8)

ax[2,1].set_facecolor(color='#D9D9D9')
"""
# set labels
#plt.setp(ax[-1, :], xlabel='Wavelenth (nm)')
#plt.setp(ax[1, 0], ylabel='Remote sensing reflectance $(sr^{-1})$')
fig.supxlabel('Wavelenth (nm)')
fig.supylabel('Remote sensing reflectance $(sr^{-1})$')
plt.suptitle('Average spectra at depths (m)')

# legend
handles, labels = ax[0,1].get_legend_handles_labels()
#lgd = fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.12, 0.93))
legend = ax[0,1].legend(handles, labels, ncol=1, frameon=True,
                    loc='upper right', bbox_to_anchor=(1.35, 1.03))
plt.tight_layout()
# save
plt.savefig(os.path.join(os.path.dirname(fp_img), 'plots', 'Avg_spectra_at_depths.png'), dpi=300, format='PNG', bbox_inches='tight')


#################################
# plot spectral angles of sav / brown algae
sa = gdf[(gdf.new_class == 3) & (gdf.field_depth < 3)]
br = gdf[(gdf.new_class == 2) & (gdf.field_depth < 3)]
gr = gdf[(gdf.new_class == 1) & (gdf.field_depth < 3)]
dw = gdf[(gdf.new_class == 6)]

fig, ax = plt.subplots()
ax.scatter(sa.Band3, sa.Band4, s=5, color='yellow', marker='s')
ax.scatter(br.Band3, br.Band4, s=5, color='#888f29', marker='v')
ax.scatter(gr.Band3, gr.Band4, s=0.5, color='green', alpha=0.6)
ax.scatter(dw.Band3, dw.Band4, s=0.5, color='blue', alpha=0.6, marker='*')
ax.grid()
plt.show()

#%%

###################################
# pts that have exact same sampled values
gdf_train['duplicate'] = gdf_train.duplicated(subset=['Band2', 'Band3', 'Band4', 'Band5', 'Band6', 'Band7', 'Band8', 'Band8a'])
print('Duplicated rows:', len(gdf_train[gdf_train.duplicate==True]))
# drop duplicates
gdf_train_d = gdf_train.drop_duplicates(subset=['Band2', 'Band3', 'Band4', 'Band5', 'Band6', 'Band7', 'Band8', 'Band8a'])
###################################

# Machine learning
# normalize data
from sklearn.preprocessing import normalize
from sklearn.model_selection import cross_val_predict
stack_re = stack.reshape((stack.shape[0],-1)).transpose((1,0))
stack_re = np.where(np.isnan(stack_re), 0, stack_re) # change nan
stack_re_n = normalize(stack_re, axis=1)

# train test data
X = stack_re_n[gt > 0]
y = gt[gt > 0]

# train test from point data
#gdf_train = gdf_train[gdf_train.new_class != 0]
#gdf_train = gdf_train[gdf_train.field_depth < 5]
#gdf_train[gdf_train.new_class == 2] = 1
X_gdf = gdf_train.drop(['new_class', 'sykeid', 'geometry', 'segment_ids', 'field_depth'], axis=1)
y_gdf = gdf_train['new_class']

# normalize 
X_gdf = normalize(X_gdf)

# random forest classifier
rf = RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=5 , max_features='sqrt',
                            bootstrap=True, oob_score=True,
                            random_state=42)


#%%
# get point stratifiedKFold indices
train_ind = dict()
test_ind = dict()
test_ids = [] # list for all ids
test_df = pd.DataFrame(columns=['SAV','Fucus', 'Bare', 'Sparse', 'Deep', 'Turbid', 'sykeid'])  #'Fucus',
# point data
X_gdf = gdf_train.drop(['Band2', 'Band3', 'Band4', 'Band5', 'Band6', 'Band7', 'eg', 'bg', 'gr', 'ndvi'], axis=1)
y_gdf = gdf_train['new_class']
# reshape segment layer
segments_re = segments.reshape(-1)
# Stratified KFold
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for i, (train_index, test_index) in enumerate(skf.split(X_gdf,y_gdf)):
    fold = 'fold_' + str(i+1)

    #train_ind[fold] = train_index.tolist()
    #test_ind[fold] = test_index.tolist()
    
    # sanity check
#    fold_pts_test = X_gdf.iloc[test_index]
#    pts_out = os.path.join(os.path.dirname(fp_gt), 'fold_pts.gpkg')
#    fold_pts_test.to_file(pts_out, driver='GPKG', layer=fold)
    # select train and test rows
    train_ind[fold] = X_gdf['sykeid'].iloc[train_index]  #[X_gdf.sykeid.isin(train_index)].tolist()
    test_ind[fold] = X_gdf['sykeid'].iloc[test_index]
    # test that train and test are separate 
    set(train_ind[fold]).intersection(test_ind[fold])    
    # select train,test data according to fold split
    X_train_fold = X_gdf[X_gdf.sykeid.isin(train_ind[fold])]
    y_train_fold = y_gdf[X_gdf.sykeid.isin(train_ind[fold])]
    X_test_fold = X_gdf[X_gdf.sykeid.isin(test_ind[fold])]
    y_test_fold = y_gdf[X_gdf.sykeid.isin(test_ind[fold])]
    
# =============================================================================
#     # OBIA approach
#     # test that train and test segment ids are separate
#     print(len(set(X_train_fold['segment_ids']).intersection(X_test_fold['segment_ids'])))    
# 
#     # function for creating mask
#     def makeIndexMask(values, mask_array):
#         idx_array = np.zeros(shape=mask_array.shape)
#         for i in values:        
#             idx_array = np.where(mask_array == i, 1, idx_array)
#         return idx_array
#     
#     # get data from image stack by segment ids
#     X_train_idx = makeIndexMask(X_train_fold.segment_ids.unique(), segments_re)
#     X_train = stack_re[X_train_idx.astype(bool)]    
# 
#     X_test_idx = makeIndexMask(X_test_fold.segment_ids.unique(), segments_re)
#     X_test = stack_re[X_test_idx.astype(bool)]    
# 
#     # Assign class labels to segments
#     y_train = np.zeros(shape=segments_re.shape) # empty array
#     # assign labels, this takes a little time
#     for idx, row in X_train_fold.iterrows():
#         y_train = np.where(segments_re == row.segment_ids, row.new_class, y_train) 
# #    y_train = np.where(nodatamask==True, 0, y_train) # mask nodata (optional)
#   
#     y_test = np.zeros(shape=segments_re.shape) # empty array
#     # assign labels, this takes a little time
#     for idx, row in X_test_fold.iterrows():
#         y_test = np.where(segments_re == row.segment_ids, row.new_class, y_test) 
# #    y_test = np.where(nodatamask==True, 0, y_test) # mask nodata (optional)
#     # sanity check: save y 
#     y_train_out = y_train.reshape(segments.shape)
#     y_test_out = y_test.reshape(segments.shape)
#     y_train_outpath = os.path.join(segdir, 'y_train_fold_' + str(i+1) + '.tif')
#     y_test_outpath = os.path.join(segdir, 'y_test_fold_' + str(i+1) + '.tif')
#     with rio.open(y_train_outpath, 'w', **upmeta, compress='LZW') as dst:
#         dst.write(y_train_out.astype(upmeta['dtype']))
#     with rio.open(y_test_outpath, 'w', **upmeta, compress='LZW') as dst:
#         dst.write(y_test_out.astype(upmeta['dtype']))
#     y_train = y_train[y_train > 0]    
#     y_test = y_test[y_test > 0]
# #    print(np.unique(y_train, return_counts=True))    
# #    print(np.unique(y_test, return_counts=True))    
#     
# =============================================================================
    
        
    # train test with point data
    X_train = np.array(X_train_fold.drop(columns=['new_class', 'field_depth', 'sykeid', 'geometry'])) #'segment_ids',
    y_train = np.array(y_train_fold)
    X_test = np.array(X_test_fold.drop(columns=['new_class', 'field_depth', 'sykeid', 'geometry']))
    y_test = np.array(y_test_fold)
    # normalize
    X_train = normalize(X_train)
    X_test = normalize(X_test)
    # random forest classifier
    rf = RandomForestClassifier(n_estimators=700, max_depth=None, n_jobs=5, max_features='sqrt',
                                #bootstrap=True, oob_score=True,
                                random_state=42)
    rf.fit(X_train, y_train)
    print(f"Fold {i}:")
    print(rf.score(X_test, y_test))
    # predict test
    predf = pd.DataFrame()
    #predf['truth'] = y_test
    predf['predict'] = rf.predict(X_test)
    # classification report
    print(metrics.classification_report(y_test, predf.predict))

    # predict test points
    X_test_fold = X_test_fold.drop(['new_class', 'field_depth', 'sykeid', 'geometry'], axis=1)
    X_test_fold_arr = np.array(X_test_fold)
    pred = rf.predict_proba(X_test_fold_arr) # predict probabilities
    # to df
    pred_df = pd.DataFrame(pred, columns=['SAV', 'Fucus', 'Bare', 'Sparse', 'Deep', 'Turbid'])    #
    # add sykeid column
    pred_df['sykeid'] = test_ind[fold].values.astype(int)
    # concat
    test_df = pd.concat([test_df, pred_df])

    # sykeid's to list
#    test_ids.append(X_gdf['sykeid'][X_gdf.index2.isin(test_index)].tolist())
 #   train_ind.append(train_index.tolist())
 #   test_ind.append(test_index.tolist())
    
#    print(f"  Train: index={train_index}")
#    print(f"  Test:  index={test_index}")
# test uniqueness of test folds
#set(test_ind['fold_1']).intersection(test_ind['fold_3'])
# extract list of lists
#test_ids = [i for j in test_ids for i in j]

# fit all data (obia)
X = stack_re_n[gt > 0]
y = gt[gt > 0]
# points
X = np.array(gdf_train.drop(columns=['new_class', 'field_depth', 'segment_ids', 'sykeid', 'geometry']))
y = np.array(gdf_train.new_class)
# norm
X = normalize(X)

# random forest classifier
rf = RandomForestClassifier(n_estimators=700, max_depth=None, n_jobs=5, max_features='sqrt',
                            #bootstrap=True, oob_score=True,
                            random_state=42)
rf.fit(X, y)


# change column type
#test_df['sykeid'] = test_df['sykeid'].astype(int)
gdf['sykeid'] = gdf['sykeid'].astype(int)

# join observed coverage
test_df = test_df.merge(gdf[['sykeid', 'bralgae_cov', 'field_depth', 'geometry']], left_on='sykeid', right_on='sykeid')
# presence
test_pres = test_df[test_df.fucaceae_cov >= 30]
# correlation
from scipy.stats import pearsonr
corr, _ = pearsonr(test_df.bralgae_cov, test_df.Fucus)
print('Pearsons correlation: %.3f' % (corr))

# plot predicted probability and observed coverage
fig, ax = plt.subplots()
ax.scatter(test_df.bralgae_cov, test_df.Fucus, s=0.7)
ax.text(0, test_df.Fucus.max()-0.05, 'Pearsons correlation: %.3f' % (corr))
p1, p0 = np.polyfit(test_df.bralgae_cov, test_df.Fucus, deg=1)
ax.axline(xy1=(0, p0), slope=p1, lw=0.5, color='black', alpha=0.7)
ax.grid(alpha=0.5)
ax.set_xlim(0,100)
ax.set_xlabel('Brown algae coverage')
ax.set_ylim(0,1)
ax.set_ylabel('Probability')
plt.title('Stratified K-fold RFClassifier \nFucus cover to predicted probability, S2')
plt.tight_layout()
# save 
plotdir = os.path.join(os.path.dirname(fp_img), 'plots')
if os.path.isdir(plotdir) == False:
    os.mkdir(plotdir)
plotout = os.path.join(plotdir, 'Fucus_to_pred_proba_S2_acolite_VENVEP.png')
plt.savefig(plotout, dpi=150, format='PNG')
plt.show()

# plot predicted probability against depth
fig, ax = plt.subplots()
ax.scatter(test_pres.field_depth, test_pres.Fucus, s=0.7)
#p1, p0 = np.polyfit(test_df.fucaceae_cov, test_df.Fucus, deg=1)
#ax.axline(xy1=(0, p0), slope=p1, lw=0.5, color='black', alpha=0.7)
ax.grid(alpha=0.5)
ax.set_xlabel('Depth')
ax.set_ylabel('Probability')
plt.title('Stratified K-fold RFClassifier \nFucus cover to depth, S2')
plt.tight_layout()

# save
import json
test_ind_outdir = '/mnt/d/users/e1008409/MK/Velmu-aineisto/sdm_vs_rs'
test_ind_out = os.path.join(test_ind_outdir, 'test_VENVEP_stratifiedKFold.json')
if os.path.isdir(test_ind_outdir) == False:
    os.mkdir(test_ind_out)
with open(test_ind_out, 'w') as fp_out:
    json.dump(test_ind, fp_out, indent=4)
# save df
test_df_out = test_df[['Fucus', 'sykeid', 'fucaceae_cov']]
test_df_out = test_df_out.rename({'Fucus': 'fucus_predproba', 'fucaceae_cov': 'fucus_cover'}, axis=1)
df_out = os.path.join(test_ind_outdir, 'S2_acolite_VENVEP_test_predictions.csv')
test_df_out.to_csv(df_out, sep=';')

df_out = os.path.join(test_ind_outdir, 'S2_acolite_WV2ext_test_predictions.gpkg')
test_gdf = gpd.GeoDataFrame(test_df, geometry=test_df.geometry)
test_gdf.to_file(df_out, driver='GPKG')

# drop columns
X_gdf_train = X_gdf.drop(['geometry', 'sykeid', 'segment_ids', 'index2'], axis=1)
# CV
n_scores = cross_val_score(rf, X_gdf_train, y_gdf, cv=skf, scoring='accuracy')
print('n=',str(n), '%.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))
# CV predictions
cv_pred = cross_val_predict(rf, X_gdf_train, y_gdf, cv=skf, n_jobs=10, method='predict_proba')
# to dataframe
df_cv_pred = pd.DataFrame(cv_pred, columns=['SAV', 'Fucus', 'Sand', 'Sparse', 'Deep'])
df_cv_pred['sykeid'] = test_ids
# convert column to int
df_cv_pred['sykeid'] = df_cv_pred['sykeid'].astype(int)
gdf['sykeid'] = gdf['sykeid'].astype(int)
# join field observations
df_cv_pred = df_cv_pred.merge(gdf[['sykeid', 'fucaceae_cov']], left_on='sykeid', right_on='sykeid')

#%%
# Stratified CV by segments
# get class foreach unique segment id
segments_re = segments.reshape(-1)
X_segments = segments_re[gt > 0]
segdf = pd.DataFrame(np.unique(X_segments), columns=['segment_ids'])
# new empty column
segdf['new_class'] = 0
# set values
for i in segdf.segment_ids:
    # create boolean indexer
    seg_class = np.unique(gt[segments_re == i])
    assert len(seg_class > 1)
    segdf.loc[segdf['segment_ids'] == i, 'new_class'] = seg_class[0].astype(int)
Xseg = np.array(segdf.segment_ids)
yseg = np.array(segdf.new_class)
# train test split    
Xseg_train, Xseg_test, yseg_train, yseg_test = train_test_split(Xseg, yseg,
                                                    test_size=0.3,
                                                    random_state=42, shuffle=True,
                                                    stratify=yseg)
# get data by segment ids
def makeIndexMask(values, mask_array):
    idx_array = np.zeros(shape=mask_array.shape)
    for i in values:        
        idx_array = np.where(mask_array == i, 1, idx_array)
    return idx_array

X_train_idx = makeIndexMask(Xseg_train, segments_re)
X_train = stack_re[X_train_idx.astype(bool)]

X_test_idx = makeIndexMask(Xseg_test, segments_re)
X_test = stack_re[X_test_idx.astype(bool)]

y_train = gt[X_train_idx.astype(bool)]
y_test = gt[X_test_idx.astype(bool)]

# random forest classifier
rf = RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=5 , max_features='sqrt',
                            bootstrap=True, oob_score=True,
                            random_state=42)
# Stratified CV by segments
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
n_scores = cross_val_score(rf, X_train, y_train, cv=skf, scoring='accuracy')
print('n=',str(n), '%.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

# CV
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)

# fit rf
rf.fit(X_train, y_train)
print(rf.score(X_test, y_test))

# predict test
predf = pd.DataFrame()
#predf['truth'] = y_test
predf['predict'] = rf.predict(X_test)
# classification report
print(metrics.classification_report(y_test, predf.predict))

#%%
# test predicting in depth zones
dzones = []
n = 0
while n < 4:
    dzone = (n, n+1)
    dzones.append(dzone)
    n += 1
for i in dzones:
    test_gdf = gdf_train[(gdf_train.field_depth >= i[0]) & (gdf_train.field_depth < i[1])]
    X_gdf = test_gdf.drop(['new_class', 'sykeid', 'geometry', 'depth'], axis=1)
    y_gdf = test_gdf['new_class']
    # train test pts to arrays
    X_gdf_train_arr = np.array(X_gdf)
    y_gdf_arr = np.array(y_gdf)
    # normalize 
    X_gdf_train_arr = normalize(X_gdf_train_arr, axis=1)

    # train test split    
    X_train, X_test, y_train, y_test = train_test_split(X_gdf_train_arr, y_gdf_arr,
                                                        test_size=0.3,
                                                        random_state=42, shuffle=True,
                                                        stratify=y_gdf_arr)
# test predicting in depth zones
depth_re = depth.reshape(-1,1)
dzones = []
n = 0
while n < 4:
    dzone = (n, n+1)
    dzones.append(dzone)
    n += 1
for i in dzones:
    # depth mask
    dmask = np.where((depth_re >= i[0]) & (depth_re < i[1]), True, False)
    # mask stack
    stack_m = np.where(dmask == True, stack_re_n, 0)
    gt_m = np.where(dmask[:,0] == True, gt, 0)
    
    # train test data
    X = stack_m[gt_m > 0]
    y = gt_m[gt_m > 0]

    # train test split    
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.3,
                                                        random_state=42, shuffle=True,
                                                        stratify=y)
    # fit rf
    rf.fit(X_train, y_train)
    print(rf.score(X_test, y_test))
    # predict test
    predf = pd.DataFrame()
    #predf['truth'] = y_test
    predf['predict'] = rf.predict(X_test)
    # classification report
    print('Depth zone', str(i[0])+'-'+str(i[1]))
    print(metrics.classification_report(y_test, predf.predict))
#%%
X_gdf_train_arr = np.array(X_gdf)
y_gdf_arr = np.array(y_gdf)
# normalize 
X_gdf_train_arr = normalize(X_gdf_train_arr, axis=1)
X = X_gdf_train_arr
y = y_gdf_arr

# train test split    
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42, shuffle=True,
                                                    stratify=y)
######### Optimizing parameters ############### These can be very slow
param_grid = { 
    'n_estimators': np.arange(100, 1000, 100), 
    'max_features': ['sqrt'], 
    'max_depth': [None, 6, 9], 
    'max_leaf_nodes': [None, 6, 9], 
} 
# grid search CV
grid_search = GridSearchCV(RandomForestClassifier(), 
                           param_grid=param_grid) 
grid_search.fit(X_train, y_train) 
print(grid_search.best_params_)
# fit rf
rf = grid_search.best_estimator_
rf.fit(X_train, y_train)
print(rf.score(X_test, y_test))
#NOTE: did not get good results

# randomized search CV
random_search = RandomizedSearchCV(RandomForestClassifier(), 
								param_grid) 
random_search.fit(X_train, y_train) 
print(random_search.best_params_) 
# use best paramas for model
rf = random_search.best_estimator_
rf.fit(X_train, y_train)
print(rf.score(X_test, y_test))
################################################
# random forest classifier
rf = RandomForestClassifier(n_estimators=700, max_depth=None, n_jobs=5 , max_features='sqrt',
                            bootstrap=True, oob_score=True,
                            random_state=42)
skfold = StratifiedKFold(n_splits=5, random_state=7, shuffle=True)
results = cross_val_score(rf, X_train, y_train, cv=skfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
rf.fit(X_train, y_train)
print(rf.oob_score_)
# predict test
predf = pd.DataFrame()
#predf['truth'] = y_test
predf['predict_rf'] = rf.predict(X_test)
# classification report
print(metrics.classification_report(y_test, predf.predict_rf))



# xgboost
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder() 
le.fit(np.unique(y_train)) # fit classes
y_train_le = le.transform(y_train) # transform
y_le = le.transform(y)
print(np.unique(y_train_le)) # check result
y_test_le = le.transform(y_test)
params = {'n_estimators': 300,
          'max_depth': 10,
          'learning_rate': 0.1,
          'objective': 'multi:softprob',
          'booster': 'gbtree',
          'verbosity':1,
          'eval_metric': ['merror', 'mlogloss'],
#          'early_stopping_rounds': 10,
          'num_class': 6}
bst = XGBClassifier(**params)
#bst = XGBClassifier(n_estimators=300, max_depth=10, learning_rate=0.1, **params)
evalset = [(X_test, y_test_le)]
bst.fit(X_train, y_train_le)
y_pred = bst.predict(X_test)
acc = accuracy_score(y_test_le, y_pred)
print(acc)
evals_result = bst.evals_result()
#print(evals_result['validation_0']['mlogloss'])
skfold = StratifiedKFold(n_splits=5, random_state=7, shuffle=True)
results = cross_val_score(bst, X_train, y_train_le, cv=skfold)
print("Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

fig, ax = plt.subplots()
ax.plot(np.arange(0,len(evals_result['validation_0']['mlogloss'])), evals_result['validation_0']['mlogloss'])
#ax.plot(np.arange(0,150), evals_result['validation_1']['logloss']) 
ax.grid()
ax.set_xlabel('Iteration')
ax.set_ylabel('M-logloss')
plt.show()

predf['predict_xgb'] = bst.predict(X_test)
#predf['predict'] = predf['predict'] +1
# classification report
print(metrics.classification_report(y_test_le, predf.predict_xgb))



#lightgbm
import lightgbm as lgb
# set params
params = {'num_leaves': 50, 'objective': 'multiclass', 'num_class': 6}
# train
clf = lgb.LGBMClassifier()
clf.fit(X_train, y_train, eval_set=[(X_test, y_test), (X_train, y_train)])
print('Training accuracy {:.4f}'.format(clf.score(X_train,y_train)))
print('Testing accuracy {:.4f}'.format(clf.score(X_test,y_test)))

predf['predict_lgbm'] = clf.predict(X_test)
# classification report
print(metrics.classification_report(y_test, predf.predict))
# plot model
lgb.plot_metric(clf)
# confusion matrix
cm = metrics.confusion_matrix(y_test, predf.predict)
disp = metrics.ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
disp.plot()


# set gdf_train index
gdf_test = gdf_train.set_index('index2')
# merge sykeid and classes
df_cv_pred = df_cv_pred.join(gdf_test[['new_class', 'geometry']], on='fold_index', how='left')


# save
df_out = os.path.join(os.path.dirname(fp_gt), 'cv_10fold_pred.csv')
df_cv_pred.to_csv(df_out, sep=';', columns=df_cv_pred.columns)

#######################################
# predict for the full image
stack_re = stack.reshape((stack.shape[0],-1)).transpose((1,0))
stack_re = np.where(np.isnan(stack_re), 0, stack_re) # replace nans

preds = [] # list for split array predictions
# find largest number within range for modulo 0
modulos = []
for i in np.arange(32,1024,1):
    if len(stack_re) % i == 0:
        modulos.append(i)
patch_size = np.max(modulos)        

# split for prediction
split_array = np.split(stack_re_n, patch_size, axis=0)
j = 1
for i in split_array: # NOTE: parallelize
    prediction = rf.predict(i)
    #prediction = bst.predict(i)
    #prediction = clf.predict(i)
    preds.append(prediction)
    print(str(j),'/',str(len(split_array)))
    j += 1

# patch predictions back to single array
predicted = np.stack(preds)
predicted = predicted.reshape(stack_re.shape[0]) 
#predicted = le.inverse_transform(predicted) # transform
# prediction back to 2D array
predicted = predicted.reshape(1, meta['height'], meta['width'])

# mask nodata
predicted = np.where(nodatamask == True, 0, predicted)

# outfile
outdir = os.path.join(os.path.dirname(fp_img), 'classification')
if os.path.isdir(outdir) == False:
    os.mkdir(outdir)
outfile = os.path.join(outdir, os.path.basename(fp_img).split('.')[0] + '_s' + str(s) + '_n' + str(n) + '_RF_pts.tif')
# update metadata
upmeta = meta.copy()
upmeta.update(dtype='uint8',
              nodata=0,
              count=1)

with rio.open(outfile, 'w', **upmeta, compress='LZW') as dst:
    dst.write(predicted.astype(rio.uint8))

#%%
# RFRegressor test
X_gdf_r = gdf_train.drop(['new_class', 'segment_ids', 'geometry', 'fucaceae_cov'], axis=1)
y_gdf_r = gdf_train['fucaceae_cov']

X_train, X_test, y_train, y_test = train_test_split(X_gdf_r, y_gdf_r,
                                                    test_size=0.3,
                                                    random_state=42, shuffle=True)

rfr = RandomForestRegressor(n_estimators=150, max_depth=None, n_jobs=5 ,max_features='sqrt',
                            bootstrap=True, oob_score=True,
                            random_state=42)

rfr.fit(X_train, y_train)
rfr.score(X_test, y_test)

rfr_pred = pd.DataFrame()
rfr_pred['true'] = y_test
rfr_pred['pred'] = rfr.predict(X_test)
from scipy.stats import pearsonr
corr, _ = pearsonr(rfr_pred.pred, rfr_pred.true)
print('Pearsons correlation: %.3f' % (corr))

# plot predicted probability and observed coverage
fig, ax = plt.subplots()
ax.scatter(rfr_pred.true, rfr_pred.pred, s=0.7)
ax.text(0, 0.9, 'Pearsons correlation: %.3f' % (corr))
p1, p0 = np.polyfit(rfr_pred.true, rfr_pred.pred, deg=1)
ax.axline(xy1=(0, p0), slope=p1, lw=0.5, color='black', alpha=0.7)
ax.grid(alpha=0.5)
ax.set_xlabel('Fucus coverage')
ax.set_ylabel('Predicted')
plt.title('Fucus cover to predicted probability WV-2')
plt.tight_layout()
# save 
plotout = os.path.join(os.path.dirname(fp_img), 'plots', 'Fucus_to_pred_proba_WV-2.png')
plt.savefig(plotout, dpi=150, format='PNG')
plt.show()


# get point KFold indices
train_ind = dict()
test_ind = dict()
test_ids = [] # list for all ids
test_df = pd.DataFrame(columns=['pred', 'sykeid'])

kf = KFold(n_splits=10, shuffle=True, random_state=42)
for i, (train_index, test_index) in enumerate(kf.split(X_gdf_r,y_gdf_r)):
    fold = 'fold_' + str(i+1)
    #train_ind[fold] = train_index.tolist()
    #test_ind[fold] = test_index.tolist()
    
    # sanity check
#    fold_pts_test = X_gdf.iloc[test_index]
#    pts_out = os.path.join(os.path.dirname(fp_gt), 'fold_pts.gpkg')
#    fold_pts_test.to_file(pts_out, driver='GPKG', layer=fold)
    # select train and test rows
    train_ind[fold] = X_gdf_r['sykeid'][X_gdf_r.index2.isin(train_index)].tolist()
    test_ind[fold] = X_gdf_r['sykeid'][~X_gdf_r.index2.isin(train_index)].tolist()
    
    X_train_fold = X_gdf_r[X_gdf_r.index2.isin(train_index)]
    y_train_fold = y_gdf_r[X_gdf_r.index2.isin(train_index)]
    X_test_fold = X_gdf_r[~X_gdf_r.index2.isin(train_index)]
    y_test_fold = y_gdf_r[~X_gdf_r.index2.isin(train_index)]
    
    # drop cols
    X_train_fold = X_train_fold.drop(['sykeid', 'index2'], axis=1)
    X_test_fold = X_test_fold.drop(['sykeid', 'index2'], axis=1)
    # normalize
    X_train_fold = normalize(X_train_fold)
    X_test_fold = normalize(X_test_fold)
    
    # random forest classifier
    rf = RandomForestRegressor(n_estimators=150, max_depth=None, n_jobs=5) # ,max_features='sqrt',
                              #  bootstrap=True, oob_score=True,
                              #  random_state=42)
    rf.fit(X_train_fold, y_train_fold)
    rf.score(X_test_fold, y_test_fold)
    pred = rf.predict(X_test_fold)
    # to df
    pred_df = pd.DataFrame(pred, columns=['pred'])    
    # add sykeid column
    pred_df['sykeid'] = test_ind[fold]
    # concat
    test_df = pd.concat([test_df, pred_df])   

    # sykeid's to list
    test_ids.append(X_gdf_r['sykeid'][~X_gdf_r.index2.isin(train_index)].tolist())
 #   train_ind.append(train_index.tolist())
 #   test_ind.append(test_index.tolist())
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"  Test:  index={test_index}")
# test uniqueness of test folds
set(test_ind['fold_1']).intersection(test_ind['fold_3'])
# extract list of lists
test_ids = [i for j in test_ids for i in j]

# change column type
test_df['sykeid'] = test_df['sykeid'].astype(int)
gdf['sykeid'] = gdf['sykeid'].astype(int)

# join observed coverage
test_df = test_df.merge(gdf[['sykeid', 'fucaceae_cov']], left_on='sykeid', right_on='sykeid')
# correlation
from scipy.stats import pearsonr
corr, _ = pearsonr(test_df.fucaceae_cov, test_df.pred)
print('Pearsons correlation: %.3f' % (corr))

# plot predicted probability and observed coverage
fig, ax = plt.subplots()
ax.scatter(test_df.fucaceae_cov, test_df.pred, s=0.7)
ax.text(0, max(test_df.pred)-1, 'Pearsons correlation: %.3f' % (corr))
p1, p0 = np.polyfit(test_df.fucaceae_cov, test_df.pred, deg=1)
ax.axline(xy1=(0, p0), slope=p1, lw=0.5, color='black', alpha=0.7)
ax.grid(alpha=0.5)
ax.set_xlabel('Observed')
ax.set_ylabel('Predicted')
plt.title('K-Fold RFRegressor predicted Fucus cover (%), WV-2')
plt.tight_layout()
# save 
plotout = os.path.join(os.path.dirname(fp_img), 'plots', 'KFold_prediction_result_WV-2.png')
plt.savefig(plotout, dpi=150, format='PNG')
plt.show()


#%%
from sklearn.neural_network import MLPClassifier
# point stratifiedKFold and check that no other points within the same segment 
train_seg_ind = dict()
test_seg_ind = dict()
test_seg_indices = []
fold_preds = []
fold_segs = []

# train test data
X_gdf_ind = gdf_train.drop(['new_class', 'sykeid'], axis=1)
y_gdf_ind = gdf_train[['gt_sampled', 'index2']]


skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
for i, (train_index, test_index) in enumerate(skf.split(X_gdf_ind, y_gdf_ind.gt_sampled)):
    # select train and test rows
    train_fold = X_gdf_ind[X_gdf_ind.index2.isin(train_index)]
    test_fold = X_gdf_ind[X_gdf_ind.index2.isin(test_index)] 
    
    # list of segments which in test and train
    test_ids = sorted(set(test_fold.segment_ids).intersection(train_fold.segment_ids))
    # exclude test fold segment ids from train 
    train_fold = train_fold[~train_fold.segment_ids.isin(test_ids)]
    
    # keep only rows with majority class in test
    
    # drop duplicates
    #test_fold = test_fold[test_fold.duplicated('segment_ids') == False]
    print(np.unique(test_fold.gt_sampled, return_counts=True))
   # break
    fold = 'fold_' + str(i+1)
    #train_seg_ind[fold] = train_fold.sykeid.tolist()
    #test_seg_ind[fold] = test_fold.sykeid.tolist()
#    test_indices.append(test_index.tolist())
#    train_ind.append(train_index.tolist())
 #   test_ind.append(test_index.tolist())
#    print(f"Fold {i}:")
#    print(f"  Train: index={train_index}")
#    print(f"  Test:  index={test_index}")
    ########
    fold_y = np.zeros(shape=segments.shape).astype('uint32')
    fold_X = np.zeros(shape=segments.shape).astype('uint32')
    X_indices = []
    y_indices = []
    # get test, train segments
    start = time.time()
    #for seg_id in test_fold.segment_ids:
    #    fold_y = np.where(segments == seg_id, segments, fold_y) 
    for seg_id in train_fold.segment_ids:
        fold_X = np.where(segments == seg_id, segments, fold_X) 
    end = time.time()
    print('Selecting', fold, 'fold segments took %.2f' % (end-start), 'seconds' )
    #print(np.unique(fold_test)) 
    
    # sanity check, save fold
 #   foldmeta = upmeta.copy()
 #   fold_out = os.path.join(segdir, 'fold_1_test.tif')    
 #   with rio.open(fold_out, 'w', **foldmeta, compress='LZW') as dst:
 #       dst.write(fold_y.astype(foldmeta['dtype']))
    
    # get points for testing
    fold_gdf_test = X_gdf_ind[~X_gdf_ind.segment_ids.isin(np.unique(fold_X.tolist()))]
    set(fold_gdf_test.segment_ids).intersection(np.unique(fold_X.tolist()))
    # sanity check, save test pts
    pts_out = os.path.join(os.path.dirname(fp_gt), 'fold_pts.gpkg')
    fold_gdf_test.to_file(pts_out, driver='GPKG', layer=fold)    

    # select stack columns
    cols = list(gdf_train.columns[:-7])
    #fold_gdf_test_X = fold_gdf_test[['Band1', 'Band2', 'Band3', 'Band4', 'Band5', 'Band8', 'tgi', 'gcc', 'eg', 'bg', 'gr', 'gbi', 'depth', 'pca1', 'pca2', 'pca3']]    
    fold_gdf_test_X = fold_gdf_test[cols]
    fold_gdf_test_y = fold_gdf_test[['gt_sampled']]    
    # convert to array
    fold_array_test_X = np.array(fold_gdf_test_X)
    fold_array_test_y = np.array(fold_gdf_test_y).reshape(-1)
    # normalize
    fold_array_test_X = normalize(fold_array_test_X, axis=1)
    
    # reshape and get data where nonzero
#    fold_y_re = fold_y.reshape(-1)
#    fold_gt = gt[fold_y_re >0]
#    fold_y_indices = np.argwhere(fold_y_re)
 #   y_indices.append(fold_y_indices)
    # segment ids of test fold
#    fold_segments = segments.reshape(-1)[fold_y_indices]
    
    fold_X_re = fold_X.reshape(-1)
    fold_X_indices = np.argwhere(fold_X_re)
    X_indices.append(fold_X_indices)    
    # select pixels from stack/ground truth
    fold_X_train = stack_re_n[fold_X_re > 0]
    fold_y_train = gt[fold_X_re > 0]
#    fold_X_test = stack_re_n[fold_y_re > 0]
#    fold_y_test = gt[fold_y_re > 0]
    
    # drop 0's
    fold_X_train = fold_X_train[fold_y_train > 0]
    fold_y_train = fold_y_train[fold_y_train > 0]
#    fold_X_test = fold_X_test[fold_y_test > 0]
#    fold_y_test = fold_y_test[fold_y_test > 0]
    
    # random forest classifier
    rf = RandomForestClassifier(n_estimators=150, max_depth=None, n_jobs=5 ,max_features='sqrt',
                                bootstrap=True, oob_score=True,
                                random_state=42)
  #  mlp = MLPClassifier(activation='relu', solver='adam', max_iter=5000, random_state=42)

    # fit 
    rf.fit(fold_X_train, fold_y_train)
  #  mlp.fit(fold_X_train, fold_y_train)
    rf.score(fold_array_test_X, fold_array_test_y)
  #  mlp.score(fold_X_test, fold_y_test)
    
    print(metrics.classification_report(fold_array_test_y, rf.predict(fold_array_test_X)))

    # predict
    fold_pred = rf.predict_proba(fold_array_test_X)

    fold_segs.append(fold_gdf_test.index2)
    fold_preds.append(fold_pred[:,1])


# concat array
fold_test_result = np.concatenate(fold_preds, axis=0)
fold_segs = np.concatenate(fold_segs, axis=0)


test_df = pd.DataFrame(np.concatenate(fold_preds, axis=0), columns=['proba'])
test_df['index2'] = fold_segs#np.concatenate(fold_segs, axis=0)    


########
fold_test = np.zeros(shape=segments.shape).astype('uint32')
fold_train = np.zeros(shape=segments.shape).astype('uint32')
X_indices = []
y_indices = []
# create iterable arrays of indices based on folds
for f in test_ind.keys():
    test_indices = test_ind.get(f)
    train_indices = train_ind.get(f)
#    break
    # get segment ids by fold indices
    test_segments = gdf.segment_ids.iloc[test_indices]
#    train_segments = gdf.loc[gdf.index.difference(test_indices), 'segment_ids']
    # or 
    train_segments = gdf.segment_ids.iloc[train_indices]
    # get test, train segments
    for seg_id in test_segments:
        fold_test = np.where(segments == seg_id, segments, fold_test) 
    for seg_id in train_segments:
        fold_train = np.where(segments == seg_id, segments, fold_train) 
    #print(np.unique(fold_test)) 
    
    # reshape and get indices where nonzero
    fold_test_re = fold_test.reshape(-1)
    fold_test_indices = np.argwhere(fold_test_re)
    y_indices.append(fold_test_indices)
    
    fold_train_re = fold_train.reshape(-1)
    fold_train_indices = np.argwhere(fold_train_re)
    X_indices.append(fold_train_indices)    
    
#######################
# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42, shuffle=True,
                                                    stratify=y)
# fit rf
rf.fit(X_train, y_train)
rf.score(X_test, y_test)

# predict test
predf = pd.DataFrame()
#predf['truth'] = y_test
predf['predict'] = rf.predict(X_test)
# classification report
print(metrics.classification_report(y_test, predf.predict))

#######################################
# predict for the full image
#stack_re = np.where(np.isnan(stack_re), 0, stack_re) # replace nans
preds = [] # list for split array predictions
# find largest number within range for modulo 0
modulos = []
for i in np.arange(32,1024,2):
    if len(stack_re) % i == 0:
        modulos.append(i)
patch_size = np.max(modulos)        

# split for prediction
split_array = np.split(stack_re_n, patch_size, axis=0)
j = 0
for i in split_array: # NOTE: parallelize
    prediction = rf.predict(i)
#    prediction = rf.predict_proba(i)
    preds.append(prediction)
    print(str(j),'/',str(len(split_array)))
    j += 1

# predictions to single array
predicted = np.stack(preds)
predicted = predicted.reshape(stack_re.shape[0]) 
# prediction back to 2D array
predicted = predicted.reshape(1, meta['height'], meta['width'])


# mask nodata
predicted = np.where(nodatamask == True, 0, predicted)

# outfile
outdir = os.path.join(os.path.dirname(fp_img), 'classification')
if os.path.isdir(outdir) == False:
    os.mkdir(outdir)
outfile = os.path.join(outdir, os.path.basename(fp_img).split('.')[0] + '_segmpix_RFclassification.tif')
# update metadata
upmeta = meta.copy()
upmeta.update(dtype='uint8',
              nodata=0,
              count=1)

with rio.open(outfile, 'w', **upmeta, compress='LZW') as dst:
    dst.write(predicted.astype(rio.uint8))



#################################
# predict probabilities
j = 0
pred_probs = [] 
for i in split_array: # NOTE: parallelize
    prediction_prob = rf.predict_proba(i)
#    prediction = rf.predict_proba(i)
    pred_probs.append(prediction_prob)
    print(str(j),'/',str(len(split_array)))
    j += 1
# predictions to single array
pred_prob = np.stack(pred_probs)
n_classes = len(np.unique(y))
pred_prob = pred_prob.reshape(stack_re.shape[0],n_classes) 
# prediction back to 2D array
predicted_proba = pred_prob.reshape(meta['height'], meta['width'], n_classes)
# transpose
predicted_proba = predicted_proba.transpose((2,0,1))
# mask nodata
predicted_proba = np.where(nodatamask == True, np.nan, predicted_proba)
# select 1 layer (brown algae)
proba_out = predicted_proba
#proba_out = np.expand_dims(proba_out, axis=0)
# outfile
outfile = os.path.join(outdir, os.path.basename(fp_img).split('.')[0] + '_segmpix_multiclass_vis_pca_RFclassification_manualedited_ptstrain_brownalgae_proba.tif')
# update metadata
probameta = meta.copy()
probameta.update(dtype=proba_out.dtype.name,
              nodata=np.nan,
              count=n_classes)

with rio.open(outfile, 'w', **probameta, compress='LZW') as dst:
    dst.write(proba_out.astype(probameta['dtype']))


    

#######################################
# Multi layer perceptron

from keras.models import Sequential
from keras.layers import Flatten, Dense, BatchNormalization

model = Sequential()
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1))

























