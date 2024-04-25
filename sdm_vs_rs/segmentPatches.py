# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:36:15 2024

@author: E1008409
"""

import sys
import os
import time
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
import xarray as xr
import rioxarray as rxr
import matplotlib.pyplot as plt
from skimage.segmentation import felzenszwalb
from dask import delayed
from dask import compute
from collections import Counter

fp = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Denmark/S2_LS2c_20220812_v1_3035.tif'
fp_pts = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Denmark/Eelgrass_2018-2023/Eelgrass_Kattegat__multiclass_2018_32632.gpkg'

xds = rxr.open_rasterio(fp)
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

# function to get stats
def segStats(segment_id, segments, img_array):
    # select pixels by id
    segpx = img_array[segments == segment_id]
#    egpx = eg_index[segments == segment_id]
    
    # stats
    segmean = np.mean(segpx, axis=0).tolist() # mean value for each band
#   egmean = np.mean(egpx, axis=0)
#    segmean.append(egmean) # add index mean to list
    segmean.append(segment_id)
    return segmean

def sampleRaster(raster_fp, geodataframe_fp):
    # read points
    gdf = gpd.read_file(geodataframe_fp, engine='pyogrio')
    # sample coords
    with rio.open(raster_fp) as src:
        meta = src.meta
        crs = src.crs.to_epsg()
        # check crs
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
        # get point coords
        coords = [(x,y) for x,y in zip(gdf.geometry.x, gdf.geometry.y)]
        # sample
        gdf['sampled'] = [x for x in src.sample(coords)]
    return gdf

# set tile size
tilesize = 5000
# compute bounds for tiles
fw = np.arange(0, int(meta['width'] / tilesize)+1, 1) # how many full tiles fits in width
fh = np.arange(0, int(meta['height'] / tilesize)+1, 1) # how many full tiles fits in height
#####################
# image segmentation parameters 
n = 10
sig = 0.0005
s = 0.30

# create segmentation patches
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
                crs=xds.rio.crs)            
        except:
            continue
        # felzenswalb segmentation 
        start = time.time()
        segments = felzenszwalb(xds_c[0:xds_c.band.shape[0]].values, scale=s, sigma=sig, min_size=n, channel_axis=0)
        end = time.time()
        elapsed = end - start
        print('Time elapsed: %.2f' % elapsed, 'seconds')
        # mask nodata areas
        nodatamask = np.where((xds_c[2].values == xds.rio.nodata) | (np.isnan(xds_c[2].values)), True, False)
        segments = np.where(nodatamask == True, 0, segments)
        # check that all segments are not nodata
        if not np.any(segments) == True:
            continue
        # expand dims
        segments = np.expand_dims(segments, axis=0)
        # define profile
        segmeta = meta.copy()
        segmeta.update(dtype='uint32',
                       width=segments.shape[2],
                       height=segments.shape[1],
                       nodata=0,
                       count=1,
                       transform=xds_c.rio.transform())
        segdir = os.path.join(os.path.dirname(fp), 'segmentation')
        if os.path.isdir(segdir) == False:
            os.mkdir(segdir)
        clip_out = os.path.join(segdir, os.path.basename(fp).split('_')[1] + '_n' + str(n) + '_s' + str(s).split('.')[1] + str(i) + str(j) + '.tif') 

        # save
        with rio.open(clip_out, 'w', **segmeta, compress='LZW') as dst:
            dst.write(segments.astype(segmeta['dtype']))
        
        # =============================================================================        
        # finer scale segmentation where is field data     
        # sample raster
        gdf = sampleRaster(clip_out, fp_pts)
        
        # extract sampled list
        gdf['segments'] = gpd.GeoDataFrame(gdf.sampled.tolist(), index=gdf.index)
        gdf = gdf.drop('sampled', axis=1)
        
        # get segment ids where is field data
        segments_ids = gdf.segments[gdf.segments > 0].unique()
        # create mask for segmented area with field data 
        segments_fd = np.zeros(shape=segments.shape)
        for segid in segments_ids:
            segments_fd = np.where(segments == segid, 1, segments_fd)
        # select area from image
        img = np.where(segments_fd == 1, xds_c[0:xds_c.band.shape[0]].values, 0)
        # image segmentation parameters 
        n2 = 5
        sig2 = 0.0005
        s2 = 0.10
        start = time.time()
        segments2 = felzenszwalb(img, scale=s2, sigma=sig2, min_size=n2, channel_axis=0)
        end = time.time()
        elapsed = end - start
        print('Time elapsed: %.2f' % elapsed, 'seconds')
        # combine new segmentation to previous
        segments_iter = np.where(segments_fd == 1, segments2, segments)
        # save
        clip_out_iter = clip_out.split('.')[0] + '_iter.tif'
        with rio.open(clip_out_iter, 'w', **segmeta, compress='LZW') as dst:
            dst.write(segments_iter.astype(segmeta['dtype']))
        # sample new segment ids to gdf       
        gdf = sampleRaster(clip_out_iter, fp_pts)
        # extract sampled list
        gdf['segments'] = gpd.GeoDataFrame(gdf.sampled.tolist(), index=gdf.index)
        gdf = gdf.drop('sampled', axis=1)
        # handle possible duplicates, ie. points that are within same segment
        # pts that have exact same sampled values
        subset = ['segments']
        gdf['duplicate'] = gdf.duplicated(subset=subset, keep=False)
        print('Duplicated rows:', len(gdf[gdf.duplicate==True]))
        # create id for finding duplicate rows
        gdf['duplicate_id'] = gdf.groupby(subset).ngroup()
        # select ones occurring more than once
        v = gdf.duplicate_id.value_counts()
        v = list(v.index[v.gt(1)])
        # check if duplicates have same or different class
        for vi in v:
            sel = gdf[gdf.duplicate_id == vi]
        
            if len(sel) > 1:
        #        print(sel[['ObservationsstedId', 'new_class']])
                # find most common value
                c = Counter(sel.new_class)
                val, count = c.most_common()[0]
                # if equal count of different values, get class from row with highest vegetation cover
                if count == 1:
                    sel_id = sel['ObservationsstedId'][sel.Coverage_pct == sel.Coverage_pct.max()].index[0] # select id where sav coverage is highest
                    droplist = sel.index[sel.index != sel_id] # indices to drop
                    # drop from gdf
                    gdf = gdf.drop(droplist)
                else:
                    print('Majority value in', vi, val, 'with count', count)
                    sel_id = sel['ObservationsstedId'][sel.new_class == val].index[0] # keep one row with majority value
                    droplist = sel.index[sel.index != sel_id] # indices to drop
                    # drop from gdf
                    gdf = gdf.drop(droplist)
        # add geometry x and y columns
        gdf['x'] = gdf.geometry.x
        gdf['y'] = gdf.geometry.y
        # =============================================================================    
        # compute statistics for each segment
        # reshape
        seg_re = segments_iter.reshape(-1)
        img_re = xds_c[0:xds_c.band.shape[0]].values
        img_re = img_re.reshape((img_re.shape[0],-1)).transpose((1,0))
        delayed_funcs = []
        # create delayed functions
        for sg_id in np.unique(seg_re)[1:]: # start from index 1 to skip 0 ie. nodata
            delayed_funcs.append(delayed(segStats)(sg_id, seg_re, img_re))
        # execute delayed functions
        start = time.time()
        result = compute(delayed_funcs)
        end_time = time.time()
        print('Processed in %.3f minutes' % ((end_time - start)/60))
            
        # extract result to dataframe
        segstats = pd.DataFrame(result[0], columns=['Band1', 'Band2', 'Band3', 'Band4', 'Band5', 'Band6', 'Band7', 'Band8', 'Band9', 'Band10', 'segment_id'])
        # join class 
        segstats = segstats.set_index('segment_id').join(gdf.set_index('segments'))
        
        # add raster patch name df
        clip_basename = os.path.basename(clip_out_iter)
        segstats['tilename'] = clip_basename
        # GeoDataFrame of sampled segments
        segstats_gdf = segstats.dropna(subset='geometry')
        segstats_gdf = gpd.GeoDataFrame(segstats_gdf, geometry=gpd.points_from_xy(segstats_gdf.x, segstats_gdf.y))
        # delete first segmentation file as it is temporary layer
        os.remove(clip_out)
        # save 
        segstats_dir = os.path.join(os.path.dirname(clip_out), 'segstats')
        if os.path.isdir(segstats_dir) == False:
            os.mkdir(segstats_dir)
        segstats_out = os.path.join(segstats_dir, os.path.basename(clip_out_iter).split('.')[0] + '_segstats.csv')
        segstats_gdf_out = os.path.join(segstats_dir, os.path.basename(clip_out_iter).split('.')[0] + '_segstats_gdf.gpkg')
        segstats.to_csv(segstats_out, sep=';')
        segstats_gdf.to_file(segstats_gdf_out, driver='GPKG', engine='pyogrio')
        






