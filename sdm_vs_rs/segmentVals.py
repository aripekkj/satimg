# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:36:15 2024

Create OBIA based ML training dataset from raster and field inventory point observations. 

All datasets need to have same crs

TODO: 
    Parallelize(?) and clean
    Don't save gdf if no points
    
@author: Ari-Pekka Jokinen
"""

import sys
import os
import time
import math
import numpy as np
import scipy as sc
import pandas as pd
import geopandas as gpd
import rasterio as rio
import xarray as xr
import rioxarray as rxr
import matplotlib.pyplot as plt
from shapely.geometry import box
from sklearn.decomposition import PCA
from skimage.segmentation import felzenszwalb
from sklearn.preprocessing import LabelEncoder
from dask import delayed
from dask import compute
from collections import Counter

# set whether to check for duplicates within segments
check_duplicates = True
use_bathymetry = True
save_intermediates = False # set whether to save intermediate segmentation results

fp = sys.argv[1]
fp_pts = sys.argv[2]
if use_bathymetry == True:
    fp_bathy = sys.argv[3]

# DNASense
fp = '/mnt/d/users/e1008409/MK/DNASense/FIN/s2_ncdf/20180715_T34_merge_nan.tif'
fp_bathy = '/mnt/d/users/e1008409/MK/DNASense/FIN/bathymetry_nan.tif'
fp_pts = '/mnt/d/users/e1008409/MK/DNASense/FIN/Finland_habitat_data_ml_new_32634.gpkg'

# Finland
fp = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Finland/S2_LS1_20180715_v101_3035_clip.tif'
fp_pts = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Finland/Finland_habitat_data_ml_5m_env_sampled_encoded_LS1_20180715.gpkg'
# Finland VHR
fp = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Finland_VHR/WorldView2_2014_09_08_10_36_23_L2W_Rrs_3035.tif'
fp_pts = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Finland_VHR/FinlandVHR_habitat_data_ml_5m_env_sampled_encoded_LS1_20180715.gpkg'
fp_bathy = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Finland_VHR/bathymetry/bathymetry_2m_3035.tif'

# Denmark
fp = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Denmark/S2_LS2c_20220812_v101_3035.tif'
fp_pts = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Denmark/Kattegat_habitat_data_encoded_img_sampled_edit.gpkg'
fp_bathy = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Denmark/bathymetry/bathy_resample_c_10m_3035.tif'
# Estonia
fp = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Estonia/S2_LS1Est_2015_merge.tif'
fp_pts = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Estonia/Estonia_habitat_data_hab_class_3035_edit.gpkg'
fp_poly = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Estonia/aoi_3035.gpkg'
fp_bathy = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Estonia/bathymetry/depth_mean_bilinear_resample_10m_3035.tif'

# Black Sea
fp = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/BlackSea/S2_LSxBLK_20200313_v1_3035_masked.tif'
fp_pts = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/BlackSea/Black_Sea_habitat_data_ml.gpkg'
fp_bathy = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/BlackSea/bathymetry/bathymetry_res_10m_3035.tif'

# Greece
fp = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Kreikka/S2_LSxGreece_10m_20230828_v101_3035_clip_bands_masked.tif'
fp_pts = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Kreikka/Greece_habitat_data_ml.gpkg'
fp_bathy = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Kreikka/bathymetry/bathymetry_10m_3035.tif'

# Norway
fp = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Norway/S2_LS3Norway_C_10m_20170721_v1_clip_ndwimasked.tif'
fp_pts = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Norway/Norway_habitat_data_ml_encoded.gpkg'
fp_bathy = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Norway/bathymetry/Emodnet_depth_10m_3035.tif'

# Netherlands
fp = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Netherlands/S2/20171015/S2_LSxNL_20171015_v101_rrs_clip.tif'
fp_pts = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Netherlands/Wadden_Sea_habitat_data_ml.gpkg'
fp_poly = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Netherlands/WaddenSea_roi_3035.gpkg'

def computeTileBounds(raster_fp, tilesize, tilewidth_no, tileheight_no):
    # bounds from raster
    with rio.open(raster_fp) as src:
        meta = src.meta
    fwm = meta['width'] % tilesize # leftover if not full tile
    fhm = meta['height'] % tilesize # leftover if not full tile
    bbox_maxx = meta['transform'][2] + meta['transform'][0] * meta['width'] 
    bbox_miny = meta['transform'][5] + meta['transform'][4] * meta['height'] 

    # metainfo has top left corner and cell sizes so we can compute other corners by cell size and width
    minx = meta['transform'][2] + meta['transform'][0] * tilesize * tilewidth_no
    maxy = meta['transform'][5] + meta['transform'][4] * tilesize * tileheight_no 
    maxx = meta['transform'][2] + meta['transform'][0] * tilesize * (tilewidth_no + 1)
    miny = meta['transform'][5] + meta['transform'][4] * tilesize * (tileheight_no + 1) 

    # if new bounds exceed original, use specific tilesize
    if maxx > bbox_maxx:
        maxx = minx + fwm * meta['transform'][0]
    if miny < bbox_miny:
        miny = miny + fhm * meta['transform'][4]
    
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

def sampleRaster(raster_fp, geodataframe_fp):
    # read points
    geodf = gpd.read_file(geodataframe_fp, engine='pyogrio')
    # sample coords
    with rio.open(raster_fp) as src:
        meta = src.meta
        crs = src.crs.to_epsg()
        # check crs
        if geodf.crs != src.crs:
            geodf = geodf.to_crs(src.crs)
        # get point coords
        coords = [(x,y) for x,y in zip(geodf.geometry.x, geodf.geometry.y)]
        # sample
        geodf['sampled'] = [x for x in src.sample(coords)]
    return geodf

def sampleRasterToGDF(raster_fp, geodataframe):
    # read points
    #gdf = gpd.read_file(geodataframe_fp, engine='pyogrio')
    # sample coords
    with rio.open(raster_fp) as src:
        meta = src.meta
        crs = src.crs.to_epsg()
        # check crs
        if geodataframe.crs != src.crs:
            geodataframe = geodataframe.to_crs(src.crs)
        # get point coords
        coords = [(x,y) for x,y in zip(geodataframe.geometry.x, geodataframe.geometry.y)]
        # sample
        geodataframe['sampled'] = [x for x in src.sample(coords)]
    return geodataframe

def clipToCRS(clip_bounds, raster_fp):
    xds = rxr.open_rasterio(raster_fp)
    #poly = gpd.read_file(fp_poly, engine='pyogrio')

    # check poly crs
    #if poly.crs != xds.rio.crs:
    #    poly = poly.to_crs(xds.rio.crs)    
    # clip by bounds
    xds_c = xds.rio.clip_box(
        minx=clip_bounds[0],
        miny=clip_bounds[1],
        maxx=clip_bounds[2],
        maxy=clip_bounds[3],
        crs=xds.rio.crs)
    # clip with geometry
    #xds_c = xds_c.rio.clip(poly.geometry.values)
    # reproject
    #xds_c = xds_c.rio.reproject("EPSG:3035", resolution=10)             
    return xds_c

def clipGDF(fp_points, bounds):
    geodf = gpd.read_file(fp_points,engine='pyogrio')
    polygon = box(bounds[0], bounds[1], bounds[2], bounds[3])
    # clip
    geodf = geodf.clip(polygon)
    return geodf

def maskAndSave(data_array, tiledir, sfx_w, sfx_h):
    # change nodata
    xds_c = xr.where(data_array == data_array._FillValue, np.nan, data_array)
    
    # mask inf
    xds_c = xr.where(np.isfinite(xds_c), xds_c, np.nan)
    
    # set nodata
    xds_c.rio.write_nodata(np.nan, inplace=True)
    # set spatial ref
    xds_c['spatial_ref'] = data_array.spatial_ref
    # save masked patch
    # save
    if os.path.isdir(tiledir) == False:
        os.mkdir(tiledir)
    splitbase = os.path.basename(fp).split('_')
    fp_out = os.path.join(tiledir, splitbase[1] + '_' + splitbase[2] + '_' + str(sfx_w) + '_' + str(sfx_h) + '.tif')
    xds_c.rio.to_raster(fp_out, compress='LZW')
    return xds_c, fp_out


basedir = os.path.dirname(fp)
tiledir = os.path.join(basedir, 'tiles')
if os.path.isdir(tiledir) == False:
    os.mkdir(tiledir)

xds = rxr.open_rasterio(fp)

# bounds from raster
with rio.open(fp) as src:
    profile = src.profile

max_dim = max(xds.shape)
# set tile size
tilesize = math.ceil(max_dim/2)
# compute bounds for tiles
fw = np.arange(0, int(profile['width'] / tilesize)+1, 1) # how many full tiles fits in width
fh = np.arange(0, int(profile['height'] / tilesize)+1, 1) # how many full tiles fits in height
#####################

# dict to save segment values
#result = dict()
result = pd.DataFrame()
gdf_out = gpd.GeoDataFrame()
cols = ['Band1', 'Band2', 'Band3', 'Band4', 'Band5', 'Band6', 'Band7', 'Band8', 'Band9', 'Band10']
pcacols = ['pca1', 'pca2', 'pca3', 'pca4', 'pca5', 'pca6', 'pca7', 'pca8', 'pca9', 'pca10']
# VHR
cols = ['Band1', 'Band2', 'Band3', 'Band4', 'Band5', 'Band6', 'Band7']
pcacols = ['pca1', 'pca2', 'pca3', 'pca4', 'pca5', 'pca6', 'pca7']


# read
gdf = gpd.read_file(fp_pts, engine='pyogrio')
print(gdf.hab_class_ml.unique())

prefix_split = os.path.basename(fp).split('_')
prefix = prefix_split[1] + '_' + prefix_split[2]

#TODO move to another script 
# label encode strings class
le = LabelEncoder()
le.fit(gdf.hab_class_ml.unique())
gdf['int_class'] = le.transform(gdf.hab_class_ml)
gdf['int_class'] = gdf['int_class'] + 1 # start unique classes with 1
# create point id column
gdf['point_id'] = gdf.index + 1
# save
fp_pts_encoded = fp_pts.split('.gpkg')[0] + '_encoded.gpkg'
#if gdf.crs != 3035: # check CRS
#    gdf = gdf.to_crs(3035)
gdf.to_file(fp_pts_encoded, driver='GPKG', engine='pyogrio')


# create segmentation patches
for i in fw:   
    for j in fh:
        print(i, j)
        clip_bounds = computeTileBounds(fp, tilesize, i, j)    
        try:        
            data = clipToCRS(clip_bounds, fp)
        except:
            print('No data in bounds')
            continue
        xds_c, xds_c_path = maskAndSave(data, tiledir, i, j)
        # test that array is not empty
        if np.isnan(xds_c.values).all() == True:
            print('All nodata')
            continue
        
        # compute pca
        pca, pca_var = computePCA(xds_c.values)
        pca = np.where(np.isnan(xds_c.values), np.nan, pca) # mask nan
        # pca variance to df and save
        df_pcavar = pd.DataFrame(pca_var, columns=['PCA_var'])
        pcameta = profile.copy()
        pcameta.update(nodata=np.nan,
                       height=pca.shape[1],
                       width=pca.shape[2],
                       transform=xds_c.rio.transform())        
        
        pcadir = os.path.join(basedir, 'pca')
        if os.path.isdir(pcadir) == False:
            os.mkdir(pcadir)
        pcaout = os.path.join(pcadir, prefix + '_pca_tile_' + str(tilesize) + str(i) + '_' + str(j) + '.tif')
        with rio.open(pcaout, 'w', **pcameta) as dst:
            dst.write(pca.astype(pcameta['dtype'])) 
        # save pcavar_df
        df_pcavar.to_csv(os.path.join(pcadir, prefix + '_pca_var.csv'))
        # compute difference between rows
        df_pcavar['diff'] = df_pcavar['PCA_var'].diff(axis=0)
        # get threshold where explained variance increases < 1
        threshold = df_pcavar[df_pcavar['diff'] < 1].index[0]-1
        
        pcafig = os.path.join(pcadir, 'Var_explained_PCA.png')
        # plot
        fig,ax = plt.subplots()
        ax.plot(df_pcavar['PCA_var'])
        ax.set_xticks(np.arange(0,len(df_pcavar)))
        ax.set_xticklabels(np.arange(1,len(df_pcavar)+1))
        ax.set_xlabel('Principal component')
        ax.set_ylabel('Variance explained')
        ax.axhline(df_pcavar['PCA_var'].iloc[threshold], ls='--', color='gray', alpha=0.5) # threshold line
        ax.grid()
        plt.suptitle('Cumulative sum of variance explained by principal components')
        plt.tight_layout()
        plt.savefig(pcafig, dpi=300)
        plt.show()
        
        # image segmentation parameters 
        n = 5
        sig = np.nanstd(xds.values[1:4,:,:]) #0.0005
        s = 0.10
        # felzenswalb segmentation 
        start = time.time()
        segments = felzenszwalb(xds_c[1:4].values, scale=s, sigma=sig, min_size=n, channel_axis=0) # use 10m visible bands, 0:xds_c.band.shape[0] <- this would use all bands
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
        segmeta = profile.copy()
        segmeta.update(dtype='uint32',
                       width=segments.shape[2],
                       height=segments.shape[1],
                       nodata=0,
                       count=1,
                       crs=xds_c.rio.crs,
                       transform=xds_c.rio.transform())
        segdir = os.path.join(basedir, 'segmentation')
        if os.path.isdir(segdir) == False:
            os.mkdir(segdir)
        clip_out = os.path.join(segdir, prefix + str(tilesize) + '_n' + str(n) + '_s' + str(s).split('.')[1] + str(i) + '_' + str(j) + '.tif') 

        # save
        with rio.open(clip_out, 'w', **segmeta) as dst:
            dst.write(segments.astype(segmeta['dtype']))
        
        if use_bathymetry == True:
            # select clip area from bathymetry raster
            bathy = clipToCRS(clip_bounds, fp_bathy)
            if bathy.shape[1] > data.shape[1]:
                bathy = bathy[:,0:data.shape[1],:]
            if bathy.shape[2] > data.shape[2]:
                bathy = bathy[:,:,0:data.shape[2]]    
            bathy_c, bathy_c_path = maskAndSave(bathy, os.path.dirname(fp_bathy), i, j) # save tile

        # =============================================================================        
        # finer scale segmentation where is field data     
        #TODO consider adding new segmentation iteration if different bottom classes within segment
        # clip pts to extent
        gdf = clipGDF(fp_pts_encoded, clip_bounds)
        # check that points exists within clip bounds
        if len(gdf) == 0:
            continue
        # sample raster
        gdf = sampleRasterToGDF(clip_out, gdf)
        # extract sampled list
        gdf['segments'] = gpd.GeoDataFrame(gdf.sampled.tolist(), index=gdf.index)
        gdf = gdf.drop('sampled', axis=1)
        # sample sat.img pixel values
        gdf = sampleRasterToGDF(xds_c_path, gdf)
        gdf[cols] = gpd.GeoDataFrame(gdf.sampled.tolist(), index=gdf.index)
        gdf = gdf.drop('sampled', axis=1)
        # sample PCA pixel values
        gdf = sampleRasterToGDF(pcaout, gdf)
        gdf[pcacols] = gpd.GeoDataFrame(gdf.sampled.tolist(), index=gdf.index)
        gdf = gdf.drop('sampled', axis=1)
        if use_bathymetry == True:
            # sample bathy pixel values
            gdf = sampleRasterToGDF(bathy_c_path, gdf)
            gdf['bathymetry'] = gpd.GeoDataFrame(gdf.sampled.tolist(), index=gdf.index)
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
        n2 = 3
        s2 = s/2
        start = time.time()
        segments2 = felzenszwalb(img[1:4], scale=s2, sigma=sig, min_size=n2, channel_axis=0)
        end = time.time()
        elapsed = end - start
        print('Time elapsed: %.2f' % elapsed, 'seconds')
        # mask segments where field points don't overlap
        segments2 = np.where(segments_fd == 1, segments2, 0)
        # get ids
        old_ids = np.unique(segments2).tolist()
        old_ids.remove(0) # drop 0 as it is nodata
        # update segment IDs so they are unique to first segmentation
        new_id_start = np.max(segments) + 1 # get maxmimum value from 1st segmentation and add 1
        new_id_end = new_id_start + len(np.unique(segments2)) # length of new ids as end
        new_ids = np.arange(new_id_start, new_id_end).tolist() # new id values
        for o in old_ids:
            segments2 = np.where(segments2 == o, new_ids[old_ids.index(o)], segments2) # update new segment ids
        
        if save_intermediates == True:
            # save new segments only
            segments2_out = clip_out.split('.')[0] + '_new_segments.tif'
            with rio.open(segments2_out, 'w', **segmeta) as dst:
                dst.write(segments2.astype(segmeta['dtype']))
        
        # combine new segmentation to previous
        segments_iter = np.where(segments_fd == 1, segments2, segments)
        # save
        clip_out_iter = clip_out.split('.')[0] + '_iter.tif'
        with rio.open(clip_out_iter, 'w', **segmeta) as dst:
            dst.write(segments_iter.astype(segmeta['dtype']))

        
        # TESTING check that old and new segment do not have same id
#        test = np.unique(segments).tolist()
#        to_test = np.unique(segments2).tolist()
#        same_ids = [k for k in to_test if k in test]
        # select same ids from raster 
#        same_segments = np.zeros(shape=segments.shape)
#        for sameid in same_ids:
#            same_segments = np.where(segments_iter == sameid, segments_iter, same_segments)
#        same_out = clip_out_iter.split('.tif')[0] + '_same_ids.tif'
#        with rio.open(same_out, 'w', **segmeta) as dst:
#            dst.write(same_segments.astype(segmeta['dtype']))
        
        
        # sample new segment ids to gdf       
        gdf = sampleRasterToGDF(clip_out_iter, gdf)
        # extract sampled list
        gdf['segments'] = gpd.GeoDataFrame(gdf.sampled.tolist(), index=gdf.index)
        gdf = gdf.drop('sampled', axis=1)
        
        if check_duplicates == True:
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
            #        print(sel[['ObservationsstedId', 'int_class']])
                    # find most common value
                    c = Counter(sel.int_class)
                    val, count = c.most_common()[0]
                    # select point index to keep if multiple pts within segment
                    if count == 1:
                        # if equal count of different values, get class from row with highest vegetation cover
                        #sel_id = sel['point_id'][sel.savcov == sel.savcov.max()].index[0] # select id where sav coverage is highest, DK Coverage_pct
                        # random select
                        sel_id = sel['point_id'][sel.int_class == np.random.choice(list(c.keys()))].index[0]
                        droplist = sel.index[sel.index != sel_id] # indices to drop
                        # drop from gdf
                        gdf = gdf.drop(droplist)
                    else:
                        print('Majority value in', vi, 'is', val, 'with count', count)
                        sel_id = sel['point_id'][sel.int_class == val].index[0] # keep one row with majority value, DK ObservationsstedId
                        droplist = sel.index[sel.index != sel_id] # indices to drop
                        # drop from gdf
                        gdf = gdf.drop(droplist)
        
        # =============================================================================    
        # select pixels from each segment
        # reshape
        seg_re = segments_iter.reshape(-1)
        img_re = xds_c[0:xds_c.band.shape[0]].values
        img_re = img_re.reshape((img_re.shape[0],-1)).transpose((1,0))
        pca_re = pca.reshape((pca.shape[0],-1)).transpose((1,0))
        if use_bathymetry == True:
            bathy_re = bathy_c[0:bathy_c.band.shape[0]].values
            bathy_re = bathy_re.reshape((bathy_re.shape[0],-1)).transpose((1,0))
        
        # pixel values to dict
        for sg_id in np.unique(gdf.segments)[1:]: # start from index 1 to skip 0 ie. nodata
            
            idx = gdf[gdf.segments == sg_id].index # get index    
            pxvals = img_re[seg_re == sg_id] # get img pixels
            #pxlist = [vals.tolist() for vals in pxvals]
            pcavals = pca_re[seg_re == sg_id] # get pca pixels
            #pcalist = [vals.tolist() for vals in pcavals]
            if use_bathymetry == True:
                bathyvals = bathy_re[seg_re == sg_id] # get bathy layers pixels
            int_class = int(gdf.int_class[gdf.segments == sg_id].values[0]) # get habitat class
            point_id = gdf.point_id[gdf.segments == sg_id].values[0] # get point id, DK point_id
            
            segresult = pd.DataFrame(pxvals, columns=cols)
            segresult[pcacols] = pcavals
            if use_bathymetry == True:
                segresult['bathymetry'] = bathyvals
            segresult['int_class'] = int_class
            segresult['segment_id'] = sg_id
            segresult['point_id'] = point_id
            
            result = pd.concat([result, segresult])

        # concat gdf
        gdf_out = pd.concat([gdf_out, gdf])
# drop duplicate column as it is not needed
gdf_out = gdf_out.drop(['duplicate', 'duplicate_id'], axis=1)        
# delete first segmentation file as it is temporary layer
os.remove(clip_out)
# reset index
result = result.reset_index(drop=True)

# check if na points exist
traincols = cols+pcacols
if use_bathymetry == True:
    traincols.append('bathymetry')
result[traincols] = result[traincols].replace('', pd.NA) # replace empty with NaN
nan_point_ids = result['point_id'][result[traincols].isna().any(axis=1)]
# drop rows that contain NA values
result = result[~result.index.isin(nan_point_ids.index)]

# check that point ids match in result and gdf
if sorted(result.point_id.unique()) == sorted(gdf.point_id) == False:
    missing_id = [p for p in result.point_id.unique() if p not in gdf.point_id]
    # 
    print('GeoDataFrame missing point_id', str(missing_id))


#if len(nans) > 0:
    # drop nans
#    gdf_out = gdf_out[~gdf_out.index.isin(nans.index)]
    # check na's also in segmented data
#    result = result.loc[~result['point_id'].isin(nans.point_id.tolist())] # exclude nan segments by point id

# save datatable
segvals_dir = os.path.join(segdir, 'segvals')
if os.path.isdir(segvals_dir) == False:
    os.mkdir(segvals_dir)
segvals_out = os.path.join(segvals_dir, prefix + '_segvals.csv')
result.to_csv(segvals_out, sep=';') 
# save geodataframe
gdf_outfile = os.path.join(os.path.dirname(fp_pts), fp_pts.split('.')[0] + '_' + prefix + '.gpkg')
gdf_out.to_file(gdf_outfile, driver='GPKG', engine='pyogrio')






