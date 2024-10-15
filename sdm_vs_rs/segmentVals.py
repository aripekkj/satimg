# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 12:36:15 2024

Create OBIA based ML training dataset from raster and field inventory point observations. 

All datasets need to have same crs

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
from shapely.geometry import box
from sklearn.decomposition import PCA
from skimage.segmentation import felzenszwalb
from sklearn.preprocessing import LabelEncoder
from dask import delayed
from dask import compute
from collections import Counter

fp = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Finland/S2_LS1_20180715_v101_3035_clip.tif'
fp_pts = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Finland/LG_AHV_aineistot_2024-02-23_selkameri_south_3035_classes.gpkg'
fp_pts = '/mnt/d/users/e1008409/MK/Velmu-aineisto/sdm_vs_rs/velmudata_07112022_selkameri_south_bounds_edit.gpkg'
fp_poly = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Finland/ranta10_selkameri_south_3035.gpkg'

fp = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Denmark/S2_LS2b_20220812_v1_3035.tif'
#fp_pts = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Denmark/Macroalgae_2018-2023/KattegatM_habitat_data.gpkg'
fp_pts = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Denmark/Eelgrass_2018-2023/Eelgrass_Kattegat__multiclass_2018_32632.gpkg'
fp_poly = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Denmark/Data_layers/Bathymetry_Composite_cleaned_0-10m_depth_4326.gpkg'

fp = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Kreikka/S2_LSxGreece_10m_20230828_v101_3035_clip_bands.tif'
fp_pts = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Kreikka/Greece_habitat_data.gpkg'
fp_poly = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Kreikka/ROI_3035.gpkg'
fp_bathy = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Kreikka/SDB/S2_LSxGreece_10m_B2B3_logbr_LinRegressor_SDB.tif'


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
    poly = gpd.read_file(fp_poly, engine='pyogrio')

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

def clipGDF(fp_pts, bounds):
    gdf = gpd.read_file(fp_pts,engine='pyogrio')
    polygon = box(bounds[0], bounds[1], bounds[2], bounds[3])
    # clip
    gdf = gdf.clip(polygon)
    return gdf

def maskAndSave(data_array, tiledir):
    # change nodata
#    xds_c = np.where(data_array.values == data_array._FillValue, np.nan, data_array.values)
    xds_c = xr.where(data_array == data_array._FillValue, np.nan, data_array)
    
    # mask inf
    xds_c = xr.where(np.isfinite(xds_c), xds_c, np.nan)
    
    # compute ndwi and mask land
    #TODO use function
    #ndwi = (xds_c.values[2]-xds_c.values[7]) / (xds_c.values[2]+xds_c.values[7])
    # mask land
    #xds_c = xr.where(ndwi < 0, np.nan, xds_c)
    
    # set nodata
    xds_c.rio.write_nodata(np.nan, inplace=True)
    # set spatial ref
    xds_c['spatial_ref'] = data_array.spatial_ref
    # save masked patch
    # save
    if os.path.isdir(tiledir) == False:
        os.mkdir(tiledir)
    fp_out = os.path.join(tiledir, os.path.basename(fp).split('_')[1] + '_' + str(i) + '_' + str(j) + '.tif')
    xds_c.rio.to_raster(fp_out, compress='LZW')
    return xds_c, fp_out

basedir = os.path.dirname(fp)
tiledir = os.path.join(basedir, 'tiles')
if os.path.isdir(tiledir) == False:
    os.mkdir(tiledir)

xds = rxr.open_rasterio(fp)

# bounds from raster
with rio.open(fp) as src:
    meta = src.meta

# set tile size
tilesize = 3000
# compute bounds for tiles
fw = np.arange(0, int(meta['width'] / tilesize)+1, 1) # how many full tiles fits in width
fh = np.arange(0, int(meta['height'] / tilesize)+1, 1) # how many full tiles fits in height
#####################

# dict to save segment values
#result = dict()
result = pd.DataFrame()
gdf_out = gpd.GeoDataFrame()
cols = ['Band1', 'Band2', 'Band3', 'Band4', 'Band5', 'Band6', 'Band7', 'Band8', 'Band9', 'Band10']
pcacols = ['pca1', 'pca2', 'pca3', 'pca4', 'pca5', 'pca6', 'pca7', 'pca8', 'pca9', 'pca10']


# 
gdf = gpd.read_file(fp_pts, engine='pyogrio')
print(gdf.hab_class.unique())

#gdf['new_class'][gdf.new_class == 5] = 1
#gdf['new_class'][gdf.new_class == 4] = 3
gdf = gdf.rename(columns={'new_class': 'habitat'})
#gdf['hab_class'] = np.where(gdf.depth <= -10, 'deep_water', gdf.hab_class) 
# label encode strings class
le = LabelEncoder()
le.fit(gdf.hab_class.unique())
gdf['new_class'] = le.transform(gdf.hab_class)
gdf['new_class'] = gdf['new_class'] + 1 # start unique classes with 1
# create point id column
gdf['point_id'] = gdf.index + 1
print(gdf.new_class.unique())
print(gdf[['hab_class', 'new_class']])
# save
fp_pts_encoded = fp_pts.split('.gpkg')[0] + '3035_encoded.gpkg'
gdf = gdf.to_crs(3035)
gdf.to_file(fp_pts_encoded, driver='GPKG', engine='pyogrio')
# update filepath
fp_pts = fp_pts_encoded

# set whether to check for duplicates
check_duplicates = True

# create segmentation patches
for i in fw:   
    for j in fh:

        clip_bounds = computeTileBounds(fp, tilesize, i, j)    
        try:        
            data = clipToCRS(clip_bounds, fp)
        except:
            continue
        xds_c, xds_c_path = maskAndSave(data, tiledir)
        # test that array is not empty
        if np.isnan(xds_c.values).all() == True:
            continue
        
        # compute pca
        pca, pca_var = computePCA(xds_c.values)
        pca = np.where(np.isnan(xds_c.values), np.nan, pca) # mask nan
        # pca variance to df and save
        df_pcavar = pd.DataFrame(pca_var, columns=['PCA_var'])
        pcameta = meta.copy()
        pcameta.update(nodata=np.nan,
                       height=pca.shape[1],
                       width=pca.shape[2],
                       transform=xds_c.rio.transform())        
        prefix_split = os.path.basename(fp).split('_')
        prefix = prefix_split[1] + '_' + prefix_split[2]
        pcadir = os.path.join(basedir, 'pca')
        if os.path.isdir(pcadir) == False:
            os.mkdir(pcadir)
        pcaout = os.path.join(pcadir, prefix + '_pca_tile_' + str(tilesize) + str(i) + '_' + str(j) + '.tif')
        with rio.open(pcaout, 'w', **pcameta, compress='LZW') as dst:
            dst.write(pca.astype(pcameta['dtype'])) 
        # save pcavar_df
        df_pcavar.to_csv(os.path.join(pcadir, 'pca_var.csv'))
        
        # image segmentation parameters 
        n = 5
        sig = np.nanstd(xds.values[1:4,:,:]) #0.0005
        s = 0.30
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
                       crs=xds_c.rio.crs,
                       transform=xds_c.rio.transform())
        segdir = os.path.join(basedir, 'segmentation')
        if os.path.isdir(segdir) == False:
            os.mkdir(segdir)
        clip_out = os.path.join(segdir, prefix + str(tilesize) + '_n' + str(n) + '_sig' + str(sig).split('.')[1] + '_s' + str(s).split('.')[1] + str(i) + '_' + str(j) + '.tif') 

        # save
        with rio.open(clip_out, 'w', **segmeta, compress='LZW') as dst:
            dst.write(segments.astype(segmeta['dtype']))
        

        # ---------------------
        # feature extractor for RF
        from keras.models import Sequential
        from keras.layers import Conv2D
        activation = 'relu'

        #test = xds_c[0:4,1024:1280,768:1024].values # select subpatch for testing
#        plt.imshow(test[2])
        #TODO use only visual light wavelengths in creating filters
        shape = xds_c.shape
        N=3
        nfilt = 32
        filtcols = ['filt_' + str(f+1) for f in np.arange(0,nfilt)]
        feat_extractor = Sequential()
        feat_extractor.add(Conv2D(nfilt, N, activation=activation, padding='same', input_shape=(shape[1], shape[2], 3))) #shape[0]
        feat_extractor.add(Conv2D(nfilt, N, activation=activation, padding='same', kernel_initializer='he_normal'))
 #       feat_extractor.add(MaxPooling2D())
 #       feat_extractor.add(Conv2D(32, N, activation=activation, padding='same', kernel_initializer='he_normal'))
        
        t = np.transpose(xds_c.values[1:4], (1,2,0)) # select visual bands
        t = np.expand_dims(t, axis=0)
        filt = feat_extractor.predict(t)
        filt = np.squeeze(filt)
#        plt.imshow(tX[:,:,9])
        
        # save filters
        filt_dir = os.path.join(basedir, 'filters')
        if os.path.isdir(filt_dir) == False:
            os.mkdir(filt_dir)
        filt_out = os.path.join(filt_dir, 'filters_' + str(nfilt) + '_N' + str(N) + '_' + str(i) + '_' + str(j) + '.tif')
        fmeta = segmeta.copy()
        fmeta.update(dtype='float32',
                     nodata=np.nan,
                     count=nfilt
                     )        
        filt = np.transpose(filt, (2,0,1))
        with rio.open(filt_out, 'w', **fmeta, compress='LZW') as dst:
            dst.write(filt.astype(fmeta['dtype']))

        # select clip area from bathymetry raster
        bathy = clipToCRS(clip_bounds, fp_bathy)        
        bathy_c, bathy_c_path = maskAndSave(bathy, os.path.dirname(fp_bathy)) # save tile

        # =============================================================================        
        # finer scale segmentation where is field data     
        #TODO consider adding new segmentation iteration if different bottom classes within segment
        # clip pts to extent
        gdf = clipGDF(fp_pts, clip_bounds)
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
        # sample filters pixel values
        gdf = sampleRasterToGDF(filt_out, gdf)
        gdf[filtcols] = gpd.GeoDataFrame(gdf.sampled.tolist(), index=gdf.index)
        gdf = gdf.drop('sampled', axis=1)
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
        n2 = 5
        s2 = s/2
        start = time.time()
        segments2 = felzenszwalb(img, scale=s2, sigma=sig, min_size=n2, channel_axis=0)
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
            #        print(sel[['ObservationsstedId', 'new_class']])
                    # find most common value
                    c = Counter(sel.new_class)
                    val, count = c.most_common()[0]
                    # if equal count of different values, get class from row with highest vegetation cover
                    if count == 1:
                        sel_id = sel['point_id'][sel.savcov == sel.savcov.max()].index[0] # select id where sav coverage is highest, DK Coverage_pct
                        droplist = sel.index[sel.index != sel_id] # indices to drop
                        # drop from gdf
                        gdf = gdf.drop(droplist)
                    else:
                        print('Majority value in', vi, 'is', val, 'with count', count)
                        sel_id = sel['point_id'][sel.new_class == val].index[0] # keep one row with majority value, DK ObservationsstedId
                        droplist = sel.index[sel.index != sel_id] # indices to drop
                        # drop from gdf
                        gdf = gdf.drop(droplist)
        # add geometry x and y columns
        epsg = str(gdf.crs.to_epsg())
        gdf['x_' + epsg] = gdf.geometry.x
        gdf['y_' + epsg] = gdf.geometry.y
        
        # =============================================================================    
        # select pixels from each segment
        # reshape
        seg_re = segments_iter.reshape(-1)
        img_re = xds_c[0:xds_c.band.shape[0]].values
        img_re = img_re.reshape((img_re.shape[0],-1)).transpose((1,0))
        pca_re = pca.reshape((pca.shape[0],-1)).transpose((1,0))
        filt_re = filt.reshape((filt.shape[0],-1)).transpose((1,0))
        bathy_re = bathy_c[0:bathy_c.band.shape[0]].values
        bathy_re = bathy_re.reshape((bathy_re.shape[0],-1)).transpose((1,0))
        
        # pixel values to dict
        for sg_id in np.unique(gdf.segments)[1:]: # start from index 1 to skip 0 ie. nodata
            
            idx = gdf[gdf.segments == sg_id].index # get index    
            pxvals = img_re[seg_re == sg_id] # get img pixels
            #pxlist = [vals.tolist() for vals in pxvals]
            pcavals = pca_re[seg_re == sg_id] # get pca pixels
            #pcalist = [vals.tolist() for vals in pcavals]
            filtvals = filt_re[seg_re == sg_id] # get filter layers pixels
            bathyvals = bathy_re[seg_re == sg_id] # get bathy layers pixels
            new_class = int(gdf.new_class[gdf.segments == sg_id].values[0]) # get habitat class
            point_id = gdf.point_id[gdf.segments == sg_id].values[0] # get point id, DK point_id
            
            # to dict
            #result[int(point_id)] = dict(img=pxlist,
            #                   pca=pcalist,
            #                   new_class=new_class)
            segresult = pd.DataFrame(pxvals, columns=cols)
            segresult[pcacols] = pcavals
            segresult[filtcols] = filtvals
            segresult['bathymetry'] = bathyvals
            segresult['new_class'] = new_class
            segresult['segment_id'] = sg_id
            segresult['point_id'] = point_id
            
            result = pd.concat([result, segresult])

        # concat gdf
        gdf_out = pd.concat([gdf_out, gdf])
        
# delete first segmentation file as it is temporary layer
os.remove(clip_out)
# save 
segvals_dir = os.path.join(segdir, 'segvals')
if os.path.isdir(segvals_dir) == False:
    os.mkdir(segvals_dir)
segvals_out = os.path.join(segvals_dir, prefix + '_segvals.csv')
result.to_csv(segvals_out, sep=';') 
# save geodataframe
gdf_outfile = os.path.join(os.path.dirname(fp_pts), fp_pts.split('.')[0] + prefix + '_sampled.gpkg')
gdf_out.to_file(gdf_outfile, driver='GPKG', engine='pyogrio')
#with open(segvals_out, 'w') as of:
#    json.dump(result, of, indent=4)






