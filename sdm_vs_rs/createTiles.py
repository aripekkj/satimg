# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 08:00:19 2024



@author: E1008409
"""

import os
import gc
import threading

import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import rioxarray as rxr
import rasterio as rio
from rasterio.vrt import WarpedVRT
from rasterio.transform import Affine

fp = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Denmark/S2_LS2d_20220812_v1.01.tif'
fp_shape = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Denmark/Data_layers/Bathymetry_Composite_cleaned_0-10m_depth_4326.gpkg'

fp_out = os.path.join(os.path.dirname(fp), os.path.basename(fp).split('.')[0] + '_3035.tif') 

# read
gdf = gpd.read_file(fp_shape)
xds = rxr.open_rasterio(fp)

# =============================================================================
# # multithread
xds = rxr.open_rasterio(fp, 
    chunks=True,
#    lock=False,
    lock=threading.Lock(), # when too many file handles open
    )
# check gdf geometry
if gdf.crs != xds.rio.crs:
    gdf = gdf.to_crs(xds.rio.crs)
# # clip with geometry
xds = xds.rio.clip(gdf.geometry.values)
# # reproject
xds = xds.rio.reproject("EPSG:3035", resolution=10, nodata=np.nan)
# set nodata
xds.rio.write_nodata(np.nan, inplace=True)
# # save
xds.rio.to_raster(
    fp_out, tiled=True, lock=threading.Lock(),
)
# 
# =============================================================================

# bounds from gdf
bounds = gdf.bounds
bounds_tl = (10.5, 55.5, 11, 56) # manually
# bounds from raster
with rio.open(fp) as src:
    meta = src.meta
# metainfo has top left corner and cell sizes so we can compute other corners by cell size and width
bbox_minx = meta['transform'][2] 
bbox_maxy = meta['transform'][5]
bbox_maxx = meta['transform'][2] + meta['transform'][0] * meta['width'] 
bbox_miny = meta['transform'][5] - meta['transform'][4] * meta['height'] 

def computeTileBounds(raster_fp, tilesize, tilewidth_no, tileheight_no):
    # bounds from raster
    with rio.open(fp) as src:
        meta = src.meta
    fwm = meta['width'] % tilesize # leftover if not full tile
    fhm = meta['height'] % tilesize # leftover if not full tile
    bbox_maxx = meta['transform'][2] + meta['transform'][0] * meta['width'] 
    bbox_miny = meta['transform'][5] - meta['transform'][4] * meta['height'] 

    # metainfo has top left corner and cell sizes so we can compute other corners by cell size and width
    minx = meta['transform'][2] + meta['transform'][0] * tilesize * tilewidth_no
    maxy = meta['transform'][5] + meta['transform'][4] * tilesize * tileheight_no 
    maxx = meta['transform'][2] + meta['transform'][0] * tilesize * (tilewidth_no + 1)
    miny = meta['transform'][5] + meta['transform'][4] * tilesize * (tileheight_no + 1) 

    # if new bounds exceed original, use specific tilesize
    if maxx > bbox_maxx:
        maxx = minx + fwm * meta['transform'][0]
    if miny > bbox_miny:
        miny = miny + fhm * meta['transform'][4]
    else:
        bounds = (minx, miny, maxx, maxy)
    return bounds

# set tile size
tilesize = 5000
# compute bounds for tiles
fw = np.arange(0, int(meta['width'] / tilesize)+1, 1) # how many full tiles fits in width
fh = np.arange(0, int(meta['height'] / tilesize)+1, 1) # how many full tiles fits in height

for i in fw:
    for j in fh:
        
        clip_bounds = computeTileBounds(fp, tilesize, i, j)    
        print(clip_bounds)
        try:
            # clip by bounds
            xds_c = xds.rio.clip_box(
                minx=clip_bounds[0],
                miny=clip_bounds[1],
                maxx=clip_bounds[2],
                maxy=clip_bounds[3],
                crs="EPSG:4326")
            # clip with geometry
            xds_c = xds_c.rio.clip(gdf.geometry.values)
            # reproject
            xds_c_r = xds_c.rio.reproject("EPSG:3035", resolution=10)
        except:
            continue
# =============================================================================
#         # write
#         with rio.open(fp) as src:
#             outmeta = src.meta
#             tf = outmeta['transform']
#             tf_out = Affine(tf[0], tf[1], clip_bounds[0], # update top-left coords to Affine
#                             tf[3], tf[4], clip_bounds[3])
#             # update meta
#             outmeta.update(width = xds_c.shape[1],
#                            height = xds_c.shape[2],
#                            transform = tf_out)        
        clip_out = os.path.join(os.path.dirname(fp), 'tiles', os.path.basename(fp).split('.')[0] + '_' + str(i) + str(j) + '.tif') 
#             with rio.open(clip_out, 'w', **outmeta, compress='LZW') as dst:
#                 dst.write(xds_c)
# 
# =============================================================================
        xds_c_r.rio.to_raster(clip_out.split('.')[0] + '_3035.tif', compress='LZW')
        













