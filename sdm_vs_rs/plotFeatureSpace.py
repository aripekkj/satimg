# -*- coding: utf-8 -*-
"""
Created on Fri Nov 15 12:25:13 2024

Make feature space plots

@author: E1008409
"""

import os
import rasterio as rio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pandas as pd 
import geopandas as gpd
from sklearn.cluster import KMeans


# Denmark
fp_pts = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Denmark/Eelgrass_2018-2023/Eelgrass_Kattegat__zos_algae_2022_3035_sampled_preds.gpkg'

# Estonia
fp = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Estonia/S2_LS1Est_2015_merge.tif'
fp_pts = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Estonia/Estonia_habitat_data_new_3035_edit.gpkg'

# Greece
fp = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Kreikka/S2_LSxGreece_10m_20230828_v101_3035_clip_bands.tif'
fp_pts = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Kreikka/Greece_habitat_data_3035.gpkg'
fp_pts = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Kreikka/Greece_train_pts_digitize.gpkg'
# Netherlands
fp = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Netherlands/S2/20171015/S2_LSxNL_20171015_v101_rrs_clip2.tif'
fp_pts = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Netherlands/Wadden_Sea_habitat_data_3035.gpkg'
fp_pts = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Netherlands/WaddenSea_train_pts_digitize.gpkg'

# define output folder for plots
outdir = os.path.join(os.path.dirname(fp_pts), 'plots')
if os.path.isdir(outdir) == False:
    os.mkdir(outdir)

def sampleRasterToGDF(raster_fp, geodataframe):
    
    # sample coords
    with rio.open(raster_fp) as src:
        meta = src.meta
        crs = src.crs.to_epsg()
        # check crs
        if geodataframe.crs != src.crs:
            geodataframe = geodataframe.to_crs(src.crs)
        # check geometry, explode if MultiPoint
        if geodataframe.geometry.geom_type.str.contains('MultiPoint').any() == True:
            sp = geodataframe.geometry.explode()
            # get point coords
            coords = [(x,y) for x,y in zip(sp.x, sp.y)]
        else:
            # get point coords
            coords = [(x,y) for x,y in zip(geodataframe.geometry.x, geodataframe.geometry.y)]
        # sample
        geodataframe['sampled'] = [x for x in src.sample(coords)]
    # create column names
    cols = ['Band_' + str(f+1) for f in np.arange(0,meta['count'])]
    # extract sampled to columns
    geodataframe[cols] = gpd.GeoDataFrame(geodataframe.sampled.tolist(), index=geodataframe.index)
    # drop sampled
    geodataframe = geodataframe.drop('sampled', axis=1)
    
    return geodataframe

# read
gdf = gpd.read_file(fp_pts, engine='pyogrio')
# sample raster to gdf
gdf = sampleRasterToGDF(fp, gdf)
#gdf = gdf.drop(columns='hab_class', axis=1)
#gdf = gdf.rename(columns={'new_class': 'hab_class',
#                          'Band1': 'Band_1',
#                          'Band2': 'Band_2',
#                          'Band3': 'Band_3',
#                         'Band4': 'Band_4',
#                          'Band5': 'Band_5',
#                          'Band8': 'Band_8',
#                          })

# ndti
gdf['ndti'] = (gdf.Band_4 - gdf.Band_3) / (gdf.Band_4 + gdf.Band_3)

# set turbid and deep water classes
gdf['hab_class'] = np.where(gdf.depth >= 3.5, 'deep', gdf.hab_class)
gdf['hab_class'] = np.where((gdf.ndti >= -0.35) & (gdf.depth >= 3), 'turbid', gdf.hab_class)


cols = ['Band_1', 'Band_2', 'Band_3', 'Band_4', 'Band_5', 'Band_8']
# check if na points exist
nans = gdf[gdf[cols].isna().any(axis=1)]
if len(nans) > 0:
    # drop nans
    #gdf = gdf.dropna(subset=cols, how='all')
    gdf = gdf[~gdf.index.isin(nans.index)]

# select columns for KMeans
X = gdf[cols]
# KMeans
kmeans = KMeans(n_clusters=3, random_state=42, n_init='auto').fit(X)
gdf['kmeans'] = kmeans.predict(X)

# save gdf
gdf_out = fp_pts.split('.gpkg')[0] + '_edit.gpkg'
gdf.to_file(gdf_out, engine='pyogrio')

with rio.open(fp) as src:
    img = src.read((1,2,3,4,5,8))
#img = img[:,5000:10000,5000:10000] # select subset from very large images
# select sampled points by habitat type
print(gdf.hab_class.unique())

hab1 = gdf[gdf.hab_class == 'Fucus']
hab2 = gdf[gdf.hab_class == 'Other']
hab3 = gdf[gdf.hab_class == 'Low']
hab4 = gdf[gdf.hab_class == 'deep']
hab5 = gdf[gdf.hab_class == 'turbid']

hab1 = gdf[gdf.hab_class == 1]
hab2 = gdf[gdf.hab_class == 2]
hab3 = gdf[gdf.hab_class == 3]
hab4 = gdf[gdf.hab_class == 4]
km0 = gdf[gdf.kmeans == 0]
km1 = gdf[gdf.kmeans == 1]
# select which bands to plot
Band1, Band2 = 'Band_2', 'Band_3'
idx1, idx2 = 1,2 # corresponding indices
# 2D plot
figout = os.path.join(outdir, 'S2_FSpace_habitat_data_' + Band1 + '_' + Band2 + '.png')
fig, ax = plt.subplots(1,2)
ax[0].scatter(img[idx1], img[idx2], s=0.4, color='grey')
ax[0].scatter(hab2[Band1], hab2[Band2], s=0.4, color='#65ef4d', label=hab2.hab_class.unique()[0])
ax[0].scatter(hab3[Band1], hab3[Band2], s=0.4, color='#ffef88', label=hab3.hab_class.unique()[0])
ax[0].scatter(hab4[Band1], hab4[Band2], s=0.4, color='blue', label=hab4.hab_class.unique()[0])
ax[0].scatter(hab5[Band1], hab5[Band2], s=0.4, color='#cb07b1', label=hab5.hab_class.unique()[0])
ax[0].scatter(hab1[Band1], hab1[Band2], s=0.4, color='#916a06', label=hab1.hab_class.unique()[0])
#ax[0].scatter(kmeans.cluster_centers_[0][1], kmeans.cluster_centers_[0][2], s=40, facecolors='none', edgecolors='green', label='KMeans 0')
#ax[0].scatter(kmeans.cluster_centers_[1][1], kmeans.cluster_centers_[1][2], s=40, facecolors='none', edgecolors='blue', label='KMeans 1')
#ax[0].scatter(kmeans.cluster_centers_[2][1], kmeans.cluster_centers_[2][2], s=40, facecolors='none', edgecolors='orange', label='KMeans 2')
#ax[0].scatter(kmeans.cluster_centers_[3][4], kmeans.cluster_centers_[3][3], s=40, facecolors='none', edgecolors='purple', label='KMeans 3')
ax[0].set_xlabel(Band1)
ax[0].set_ylabel(Band2)
ax[0].set_xlim(0, 0.1)
ax[0].set_ylim(0, 0.1)
# set rectangle where to zoom in second plot
xlim0, xlim1 = gdf[Band1].min(), gdf[Band1].max()
ylim0, ylim1 = gdf[Band2].min(), gdf[Band2].max()
width = xlim1 - xlim0
height = ylim1 - ylim0
rect = patches.Rectangle((xlim0, ylim0), width, height, lw=1, ls='--', edgecolor='white', facecolor='none')
ax[0].add_patch(rect)

ax[1].scatter(img[idx1], img[idx2], s=0.4, color='grey')
ax[1].scatter(hab2[Band1], hab2[Band2], s=0.4, color='#65ef4d') #65ef4d
ax[1].scatter(hab3[Band1], hab3[Band2], s=0.4, color='#ffef88')
ax[1].scatter(hab4[Band1], hab4[Band2], s=0.4, color='blue')
ax[1].scatter(hab5[Band1], hab5[Band2], s=0.4, color='#cb07b1')
ax[1].scatter(hab1[Band1], hab1[Band2], s=0.4, color='#916a06') #0b750b
#ax[1].scatter(kmeans.cluster_centers_[0][1], kmeans.cluster_centers_[0][2], s=40, facecolors='none', edgecolors='green', label='KMeans 0')
#ax[1].scatter(kmeans.cluster_centers_[1][1], kmeans.cluster_centers_[1][2], s=40, facecolors='none', edgecolors='blue', label='KMeans 1')
#ax[1].scatter(kmeans.cluster_centers_[2][1], kmeans.cluster_centers_[2][2], s=40, facecolors='none', edgecolors='orange', label='KMeans 2')
#ax[1].scatter(kmeans.cluster_centers_[3][4], kmeans.cluster_centers_[3][3], s=40, facecolors='none', edgecolors='purple', label='KMeans 3')
ax[1].set_xlabel(Band1)
ax[1].set_ylabel(Band2)
ax[1].set_xlim(xlim0, xlim1)
ax[1].set_ylim(ylim0, ylim1)

ax[0].set_facecolor('black')
ax[1].set_facecolor('black')
ax[0].legend(loc='upper left')
plt.suptitle('Sentinel-2 feature space for EST habitat data')
plt.tight_layout()
plt.savefig(figout, dpi=300)
plt.show()


# try 3D plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(img[0], img[1], img[2], s=0.1, color='black', alpha=0.1, zorder=1)
ax.scatter(hab1.Band_2, hab1.Band_3, hab1.Band_4, s=0.5, color='green')
ax.scatter(hab2.Band_2, hab2.Band_3, hab2.Band_4,s=0.5, color='#ffef88')
ax.set_xlabel('Blue', color='blue')
ax.set_ylabel('Green', color='green')
ax.set_zlabel('Red', color='red')
plt.tight_layout()
plt.show()










































