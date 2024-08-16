# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:36:15 2024

Create OBIA based ML training dataset from raster and field inventory point observations. 

TODO: Consider adding another segmentation iteration where several points within segment
    Parallelize and clean
    Don't save gdf if no points
    
@author: Ari-Pekka Jokinen
"""

import sys
import os
import time
import numpy as np
import scipy as sc
import pandas as pd
import geopandas as gpd
import rasterio as rio
import xarray as xr
import rioxarray as rxr
import itertools
import json
from sklearn.decomposition import PCA
from skimage.segmentation import felzenszwalb
from dask import delayed
from dask import compute
from collections import Counter

fp = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Denmark/S2_LS2b_20220812_v1_3035.tif'
fp = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Denmark/S2_LS2a_20220812_v1.01.tif'
fp_pts = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Denmark/Eelgrass_2018-2023/Eelgrass_Kattegat__multiclass_2018_32632.gpkg'
fp_poly = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Denmark/Data_layers/Bathymetry_Composite_cleaned_0-10m_depth_4326.gpkg'


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
    pca = PCA()
    pca.fit(test)    
    # apply pca
    pca_out = pca.transform(flat_data)
    # reshape 
    pca_out = np.reshape(pca_out, (img.shape[1], img.shape[2], img.shape[0]))
    pca_out = np.transpose(pca_out, (2,0,1))
    # select pc's that explain 99% of variation
    #pca_out = pca_out[0:3,:,:]
    return pca_out    

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

def clipToCRS(clip_bounds, raster_fp):
    xds = rxr.open_rasterio(fp)
    poly = gpd.read_file(fp_poly)

    # check poly crs
    if poly.crs != xds.rio.crs:
        poly = poly.to_crs(xds.rio.crs)    
# clip by bounds
    xds_c = xds.rio.clip_box(
        minx=clip_bounds[0],
        miny=clip_bounds[1],
        maxx=clip_bounds[2],
        maxy=clip_bounds[3],
        crs=xds.rio.crs)
    # clip with geometry
    xds_c = xds_c.rio.clip(poly.geometry.values)
    # reproject
    xds_c = xds_c.rio.reproject("EPSG:3035", resolution=10)             
    return xds_c

def maskAndSave(data_array):
    # change nodata
#    xds_c = np.where(data_array.values == data_array._FillValue, np.nan, data_array.values)
    xds_c = xr.where(data_array == data_array._FillValue, np.nan, data_array)
    
    # mask inf
    xds_c = xr.where(np.isfinite(xds_c), xds_c, np.nan)
    
    # compute ndwi and mask land
    #TODO use function
    ndwi = (xds_c.values[2]-xds_c.values[7]) / (xds_c.values[2]+xds_c.values[7])
    # mask land
    xds_c = xr.where(ndwi < -0.3, np.nan, xds_c)
    
    # set nodata
    xds_c.rio.write_nodata(np.nan, inplace=True)
    # set spatial ref
    xds_c['spatial_ref'] = data_array.spatial_ref
    # save masked patch
    # save
    tiledir = os.path.join(os.path.dirname(fp), 'tiles')
    if os.path.isdir(tiledir) == False:
        os.mkdir(tiledir)
    fp_out = os.path.join(tiledir, os.path.basename(fp).split('_')[1] + '_' + str(i) + '_' + str(j) + '.tif')
    xds_c.rio.to_raster(fp_out, compress='LZW')
    return xds_c

tiledir = os.path.join(os.path.dirname(fp), 'tiles')
if os.path.isdir(tiledir) == False:
    os.mkdir(tiledir)

xds = rxr.open_rasterio(fp)

# bounds from raster
with rio.open(fp) as src:
    meta = src.meta

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

# dict to save segment values
#result = dict()
result = pd.DataFrame()
cols = ['Band1', 'Band2', 'Band3', 'Band4', 'Band5', 'Band6', 'Band7', 'Band8', 'Band9', 'Band10']
pcacols = ['pca1', 'pca2', 'pca3', 'pca4', 'pca5', 'pca6', 'pca7', 'pca8', 'pca9', 'pca10']

# create segmentation patches
for i in fw:   
    for j in fh:
        clip_bounds = computeTileBounds(fp, tilesize, i, j)    
        try:        
            data = clipToCRS(clip_bounds, fp)
        except:
            continue
        xds_c = maskAndSave(data)
        
        # compute pca
        pca = computePCA(xds_c.values)
        pca = np.where(np.isnan(xds_c.values), np.nan, pca) # mask nan
        pcameta = meta.copy()
        pcameta.update(nodata=np.nan,
                       height=pca.shape[1],
                       width=pca.shape[2],
                       transform=xds_c.rio.transform())        
        with rio.open(os.path.join(tiledir, 'LS2a_pca_tile_' + str(i) + '_' + str(j) + '.tif'), 'w', **pcameta, compress='LZW') as dst:
            dst.write(pca.astype(pcameta['dtype'])) 
        # felzenswalb segmentation 
        start = time.time()
        segments = felzenszwalb(xds_c[0:xds_c.band.shape[0]].values, scale=s, sigma=sig, min_size=n, channel_axis=0)
#        segments = felzenszwalb(pca, scale=s, sigma=sig, min_size=n, channel_axis=0) # test if segmenting PCA makes different result - not significantly
        end = time.time()
        elapsed = end - start
        print('Time elapsed: %.2f' % elapsed, 'seconds')
        # mask nodata areas
        nodatamask = np.where((xds_c[2].values == xds.rio.nodata) | (np.isnan(xds_c[2].values)), True, False)
        segments = np.where(nodatamask == True, 0, segments)
        # check that all segments are not nodata
        if not np.any(segments) == True:
            print('All nodata')
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
        clip_out = os.path.join(segdir, os.path.basename(fp).split('_')[1] + '_n' + str(n) + '_s' + str(s).split('.')[1] + str(i) + '_' + str(j) + '.tif') 

        # save
        with rio.open(clip_out, 'w', **segmeta, compress='LZW') as dst:
            dst.write(segments.astype(segmeta['dtype']))
        
        # =============================================================================        
        # finer scale segmentation where is field data     
        #TODO consider adding new segmentation iteration if different bottom classes within segment
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
        img = np.where(segments_fd == 1, xds_c[0:xds_c.band.shape[0]].values, np.nan)
        # mask
        #img = np.where(nodatamask == True, np.nan, img)
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
        
        #TODO 
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
                    print('Majority value in', vi, 'is', val, 'with count', count)
                    sel_id = sel['ObservationsstedId'][sel.new_class == val].index[0] # keep one row with majority value
                    droplist = sel.index[sel.index != sel_id] # indices to drop
                    # drop from gdf
                    gdf = gdf.drop(droplist)
        # add geometry x and y columns
        gdf['x'] = gdf.geometry.x
        gdf['y'] = gdf.geometry.y
        # =============================================================================    
        # select pixels from each segment
        # reshape
        seg_re = segments_iter.reshape(-1)
        img_re = xds_c[0:xds_c.band.shape[0]].values
        img_re = img_re.reshape((img_re.shape[0],-1)).transpose((1,0))
        pca_re = pca.reshape((pca.shape[0],-1)).transpose((1,0))
        
        # pixel values to dict
        for sg_id in np.unique(gdf.segments)[1:]: # start from index 1 to skip 0 ie. nodata
            
            idx = gdf[gdf.segments == sg_id].index # get index    
            pxvals = img_re[seg_re == sg_id] # get img pixels
            #pxlist = [vals.tolist() for vals in pxvals]
            pcavals = pca_re[seg_re == sg_id] # get pca pixels
            #pcalist = [vals.tolist() for vals in pcavals]
            new_class = int(gdf.new_class[gdf.segments == sg_id].values[0]) # get habitat class
            point_id = gdf.point_id[gdf.segments == sg_id].values[0] # get point id
            
            # to dict
            #result[int(point_id)] = dict(img=pxlist,
            #                   pca=pcalist,
            #                   new_class=new_class)
            segresult = pd.DataFrame(pxvals, columns=cols)
            segresult[pcacols] = pcavals
            segresult['new_class'] = new_class
            segresult['segment_id'] = sg_id
            segresult['point_id'] = point_id
            
            result = pd.concat([result, segresult])

# delete first segmentation file as it is temporary layer
os.remove(clip_out)
# save 
segvals_dir = os.path.join(os.path.dirname(clip_out), 'segvals')
if os.path.isdir(segvals_dir) == False:
    os.mkdir(segvals_dir)
segvals_out = os.path.join(segvals_dir, os.path.basename(clip_out_iter).split('_')[0] + '_segvals.csv')
result.to_csv(segvals_out, sep=';') 
#with open(segvals_out, 'w') as of:
#    json.dump(result, of, indent=4)






