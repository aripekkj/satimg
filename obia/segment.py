# -*- coding: utf-8 -*-
"""
Created on Wed Jan 19 15:02:37 2022

Image segmentation

@author: E1008409
"""


import os
import glob
import numpy as np
import scipy
import pandas as pd
import time
import gc
from skimage import exposure
from skimage.segmentation import quickshift, slic, felzenszwalb, watershed
import geopandas as gpd
from shapely.geometry import box
import matplotlib.pyplot as plt
import rasterio as rio
from rasterio import mask
from rasterio.crs import CRS
from fiona.crs import from_epsg
from rasterstats import zonal_stats

from sklearn import metrics
from sklearn.model_selection import cross_val_score, RepeatedKFold, RepeatedStratifiedKFold
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


def getPolyFeatures(gdf):
    import json
    return [json.loads(gdf.to_json())['features'][0]['geometry']]

def cropRaster(raster_fp, poly_fp):
    # read poly
    gdf = gpd.read_file(poly_fp)
    # polygon coords
    polycoords = getPolyFeatures(gdf)
    # mask raster
    d = rio.open(raster_fp)
    d_out, tf_out = mask.mask(d, shapes=polycoords, crop=True)
    d.close()
    
    return d_out, tf_out

def getRioBbox(filepath):
    """
    Get bounding box of a raster layer 

    Parameters
    ----------
    filepath : str
        String filepath.

    Returns
    -------
    Bounding box as GeoDataFrame and JSON format list of bounding box coordinates

    """
    import rasterio as rio
    import geopandas as gpd
    from shapely.geometry import box
    from fiona.crs import from_epsg
    import json
    
    with rio.open(filepath) as src:
        bbox = box(src.bounds[0],src.bounds[1],src.bounds[2],src.bounds[3]) # left bottom, right top
     # polygonize bbox
    bbox_poly = gpd.GeoDataFrame({'geometry': bbox}, index=[0], crs=src.crs)#from_epsg(3067))
    # get features 
    coords = [json.loads(bbox_poly.to_json())['features'][0]['geometry']]
    return bbox_poly, coords
    

def normalizedDifference(image_array, band_indices):
    """
    Compute normalized difference from given bands

    Parameters
    ----------
    image_array : array
        DESCRIPTION.
    band_indices : integer tuple
        DESCRIPTION.

    Returns
    -------
    Normalized difference

    """
    # computation
    ndiff = (image_array[band_indices[0]].astype(float) - image_array[band_indices[1]].astype(float)) / (image_array[band_indices[0]] + image_array[band_indices[1]])
    return ndiff


def VARI(image_array, band_indices_rgb):
    """
    Compute Visible atmospherically resistant index
    
    VARI = (Green-Red) / (Green+Red-Blue)
    
    Reference: Gitelson, A., et al. "Vegetation and Soil Lines in Visible Spectral Space: A Concept and Technique for Remote Estimation of Vegetation Fraction. International Journal of Remote Sensing 23 (2002): 2537−2562.

    Parameters
    ----------
    image_array : TYPE
        DESCRIPTION.
    band_indices : tuple
        Band indices for red, green and blue band

    Returns
    -------
    VARI

    """
    vari = (image_array[band_indices_rgb[1]]-image_array[band_indices_rgb[0]]) / (image_array[band_indices_rgb[1]]+image_array[band_indices_rgb[0]]-image_array[band_indices_rgb[2]])
    return vari
    
def GCC(image_array, band_indices):
    """
    Compute Green Colour Coordinate (GCC)
    GCC = green / (Red+Green+Blue)
    
    Parameters
    ----------
    image_array : TYPE
        DESCRIPTION.
    band_indices : list or tuple
        indices in order blue, green, red

    Returns
    -------
    GCC

    """
    gcc = image_array[band_indices[1]] / (image_array[band_indices[2]]+image_array[band_indices[1]]+image_array[band_indices[0]])
    return gcc    

def EG(image_array, band_indices):
    """
    Compute Excess Greenness
    EG = 2*G-(R+B)

    Parameters
    ----------
    image_array : TYPE
        DESCRIPTION.
    band_indices : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    eg = (2*image_array[band_indices[1]]) - (image_array[band_indices[2]] + image_array[band_indices[0]])
    return eg
 
def deepWater(image_array):
    """
    Optically deep waters based on turbidity proxy from Red-Edge band (wavelength 704nm) by Caballero et al. (2019)

    Parameters
    ----------
    img : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    log_sdbodw = -0.251*np.log(image_array)+0.8
    return log_sdbodw
    
# path to image
fp = '/mnt/d/users/e1008409/MK/worldview/pori/14SEP08103623-M2AS-015473758110_01_P001_toa_reflectance_clip.tif'
fp = '/mnt/d/users/e1008409/MK/worldview/pori/ac/WorldView2_2014_09_08_10_36_23_L2W_Rrs_clip_median3x3_filt.tif'

#fp_test = 'D:\\Users\\E1008409\\MK\S2\\ac\\S2B_MSIL1C_20190903T100029_N0208_R122_T34WFT_20190903T121103\\S2B_MSI_2019_09_03_10_02_29_T34WFT_L2W_RrsB2B3B4B8_median3x3_filt.tif'
fp_polymask = 'D:\\Users\\E1008409\\MK\\S2\\ac\\S2B_MSIL1C_20190903T100029_N0208_R122_T34WFT_20190903T121103\\cloudmask.gpkg'
# depth mask
fp_pca = '/mnt/d/users/e1008409/MK/worldview/pori/ac/pca/L2W_VIS_PCA.tif'
#fp_depth = 'D:\\Users\\E1008409\\MK\\syvyysmalli\\depth_20210126_10m_32634_T34WFT.tif'
#fp_depth = '/mnt/d/users/e1008409/MK/worldview/pori/sdb/14SEP08103623-M2AS-015473758110_01_P001_toa_reflectance_clip_deglint_ndvi_watermask_median3x3_filt_RFRegressor_SDB_cbybybrgygr.tif'
fp_depth = '/mnt/d/users/e1008409/MK/worldview/pori/ac/sdb/WorldView2_2014_09_08_10_36_23_L2W_Rrs_clip_median3x3_filt_RFRegressor_SDB_allpts_gryrtgi_masked.tif'
fp_depth2 = '/mnt/d/users/e1008409/MK/syvyysmalli/depth_20210126_clipped_transformed_pori_bilinear2m_32634.tif'

fp_s2_704 = 'D:\\Users\\E1008409\\MK\\S2\\ac\\S2B_MSIL1C_20190903T100029_N0208_R122_T34WFT_20190903T121103\\S2B_MSI_2019_09_03_10_02_29_T34WFT_L2W_Rrs_704_bilinear10m.tif'

# 1. Inputs

# depth layer
with rio.open(fp_depth) as src:
    depth = src.read()
    dmeta = src.meta
# change depth nodata to nan
depth = np.where(depth == dmeta['nodata'], np.nan, depth)

with rio.open(fp_depth2) as src:
    depthmodel = src.read()
    dm_meta = src.meta
# change depth nodata to nan
depthmodel = np.where(depthmodel == dm_meta['nodata'], np.nan, depthmodel)

# 704nm band
with rio.open(fp_s2_704) as src:
    imgre = src.read()

# read pca
with rio.open(fp_pca) as src:
    pca = src.read()
########
# get polymask features
polymask = gpd.read_file(fp_polymask)
# open image and mask with polygon
with rio.open(fp_test) as src:
    # check crs
    if polymask.crs.to_epsg() != src.crs.to_epsg():
        polymask.to_crs(epsg=src.crs.to_epsg())
    # get polyfeatures and mask
#    polys = getPolyFeatures(polymask)    
    polys = polymask.geometry.values
    img, img_tf = mask.mask(src, shapes=polys, all_touched=True, invert=True, crop=False, nodata=src.meta['nodata'])
    meta = src.meta
#plt.imshow(img[0])
# If no polymask, open image with rasterio
with rio.open(fp) as src:
    img = src.read()
    nodata = src.nodata
    meta = src.meta
    bounds = src.bounds
    
# mask deep water areas 
img = np.where(np.isnan(depth), np.nan, img)
# mask optically deep water based on blue, green and nir band reflectance
thresh_b, thresh_g, thresh_nir = 0.003, 0.0025, 0.007 # perämeri
img = np.where( ((img[0] < thresh_b) & (img[3] < thresh_nir)) & ((img[1] < thresh_g) & (img[3] < thresh_nir)), np.nan, img)   
# mask negative values
img = np.where(img < 0, np.nan, img)

# nodata mask
if np.isnan(meta['nodata']):
    nodatamask = np.where(np.isnan(img[0]), True, False)
else:
    nodatamask = np.where(img[0] == meta['nodata'], True, False)
datamask = np.isfinite(img)

# save masked image
img_masked = fp_test.split('.')[0] + '_masked.tif'
with rio.open(img_masked, 'w', **meta) as dst:
    dst.write(img.astype(rio.float32))

#####################
# 2. segment image
# set some parameters
n = 10
sig = 0.001 #np.nanstd(img[1:5]) # visible bands
s = 0.1

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
# reshape
segments = segments.reshape(1,segments.shape[0], segments.shape[1])
# save the segments
#segdir = os.path.join(os.path.dirname(os.path.dirname(fp)))
segdir = os.path.join(os.path.dirname(fp), 'segment')
if os.path.isdir(segdir) == False:
    os.mkdir(segdir)    
segfile = os.path.join(segdir, os.path.basename(fp).split('.')[0] + 'masked_segments_rgb_scale_' + str(n) + '_' +
                       str(s) + '_' + str(sig) + 'sigma_felzenszwalb.tif') #sigma_felzenszwalb
#segfile = os.path.join(segdir, 'tgi2' + str(n) + '_' +
#                       str(s) + '_' + str(sig) + 'sigma_felzenszwalb.tif') #sigma_felzenszwalb

# update metadata
upmeta = meta.copy()
upmeta.update(
    dtype = rio.uint32,
    nodata = 0,
    count = 1)

# mask deep water areas from segments
segments = np.where((depthmodel < -6) | (depthmodel == 0), 0, segments)

# save
with rio.open(segfile, 'w', **upmeta, compress='LZW') as dst:
    dst.write(segments.astype(upmeta['dtype']))

with rio.open(segfile) as src:
    segments = src.read()



#### Compute NDVI and other indices useful ####

# indices and band ratios
ndvi = normalizedDifference(img, (6,4))
yrvi = normalizedDifference(img, (3,4))
rervi = normalizedDifference(img, (5,4))
eg = EG(img, (1,2,4))
grvi = normalizedDifference(img, (2,4))
vari = VARI(img, (4,2,1))
gcc = GCC(img, (1,2,4))

print('NDVI min %.2f, max %.2f' % (np.nanmin(ndvi), np.nanmax(ndvi)))

def TGI(image_array, band_indices, central_wavelengths):
    """
    Computes Triangular Greenness Index
    
    TGI = ((lred-lblue)*(rreg-rgreen)-(lred-lgreen)*(rred-rblue)) / 2

    Reference: Hunt, E., C. Daughtry, J. Eitel, and D. Long. "Remote Sensing Leaf Chlorophyll Content Using a Visible Band Index." Agronomy Journal 103, No. 4 (2011): 1090-1099.

    Parameters
    ----------
    image_array : TYPE
        DESCRIPTION.
    band_indices : List
        Band indices for Blue,Green,Red (in this order)
    central_wavelengths : List
        Central wavelengths for Blue,Green,Red bands (in this order). Selection according to band indices

    Returns
    -------
    TGI

    """
    center_wls = list(np.array(central_wavelengths)[band_indices])
    tgi = ((center_wls[2]-center_wls[0])*(image_array[band_indices[2]]-image_array[band_indices[1]]) - (center_wls[2]-center_wls[1])*(image_array[band_indices[2]]-image_array[band_indices[0]])) / 2
    return tgi        
    
############### TGI #################
s2a_central_wl = [442.7, 492.4, 559.8, 664.6, 704.1, 740.5, 782.8, 832.8, 864.7, 945.1, 1373.5, 1613.7, 2202.4]
s2b_central_wl = [442.2, 492.1, 559.0, 664.9, 703.8, 739.1, 779.7, 832.9, 864.0, 943.2, 1376.9, 1610.4, 2185.7] 
wv2_ms_central50_wl = [427.3, 477.9, 546.2, 607.8, 658.8, 723.7, 831.3, 908.0]
wv2_ms_central05_wl = [427.0, 478.3, 545.8, 607.7, 658.8, 724.1, 832.9, 949.3]
tgi = TGI(img, [1,2,4], wv2_ms_central50_wl)

# stack to image
img = img.transpose(1,2,0)
ndvi = ndvi.reshape(1, ndvi.shape[0], ndvi.shape[1])
grvi = grvi.reshape(1, grvi.shape[0], grvi.shape[1])
tgi = tgi.reshape(1, tgi.shape[0], tgi.shape[1])
gcc = gcc.reshape(1, gcc.shape[0], gcc.shape[1])
vari = vari.reshape(1, vari.shape[0], vari.shape[1])
eg = eg.reshape(1, eg.shape[0], eg.shape[1])
ndvi = ndvi.transpose(1,2,0)
grvi = grvi.transpose(1,2,0)
tgi = tgi.transpose(1,2,0)
gcc = gcc.transpose(1,2,0)
vari = vari.transpose(1,2,0)
eg = eg.transpose(1,2,0)
depth = depth.transpose(1,2,0)
pca = pca.transpose(1,2,0)
stack = np.dstack((img, ndvi, grvi, tgi, gcc, vari, eg, depth, pca))

# stack segments to image
segments = segments.transpose(1,2,0)
stack = np.dstack((stack, segments))
stack = np.where(stack == 0, np.nan, stack)
# tranpose
stack = stack.transpose(2,0,1)
segments = segments.transpose((2,0,1))
ndvi = ndvi.transpose((2,0,1))
# save ndvi
ndviout = os.path.join(segdir, 'ndvi.tif')
ndvimeta = meta.copy()
ndvimeta.update(count=1)
with rio.open(ndviout, 'w', **ndvimeta) as dst:
    dst.write(ndvi.astype(rio.float32))
# save tgi
outdir = os.path.join(os.path.dirname(fp_test), 'aux_layers')
if os.path.isdir(outdir) == False:
    os.mkdir(outdir)
tgi_out = os.path.join(outdir, 'tgi2.tif')
tgimeta = meta.copy()
tgimeta.update(count=1)
with rio.open(tgi_out, 'w', **tgimeta) as dst:
    dst.write(tgi.astype(rio.float32),1)
  
img = img.transpose(2,0,1)
# threshold mask of optical depth
absdepth = np.abs(depth) # absolute values
absdepth = absdepth.transpose(2,0,1)
# optical water depth from 704nm band (Caballero)
odw = deepWater(img[5])
stack = np.where(absdepth > odw, np.nan, stack)

# test
test = np.zeros(shape=(1, img.shape[1], img.shape[2]))
test = np.where(absdepth[0] > odw, 1, 0)
testout = os.path.join(os.path.dirname(fp_test), 'odwmask.tif')
testmeta = meta.copy()
testmeta.update(count=1,
                dtype='uint8',
                nodata=0)
with rio.open(testout, 'w', **testmeta) as dst:
    dst.write(test.astype(rio.uint8),1)
    

from skimage.segmentation import chan_vese

chvese = chan_vese(tgi[:,:,0])    
chvese = chvese.reshape(1, chvese.shape[0], chvese.shape[1])
# mask deep water areas from segments
chvese = np.where((depthmodel < -6) | (depthmodel == 0), 0, chvese)
# save
chanvese_out = os.path.join(segdir, 'tgi_chanvese.tif')
with rio.open(chanvese_out, 'w', **upmeta, compress='LZW') as dst:
    dst.write(chvese.astype(rio.uint32))

    
##############
# 3. Extract spectral features for segments
##############
"""
# import module for parallel processing
from joblib import Parallel

# helper funcion to collect result
def getResult(result):
    global results
    results.append(result)    

# function to get spectral features
def segment_features(segment_pixels):
    features = []
    npixels, nbands = segment_pixels.shape
    for b in range(nbands):
        stats = scipy.stats.describe(segment_pixels[:, b], nan_policy='omit')
        band_stats = list(stats.minmax) + list(stats)[2:]
        if npixels == 1:
            # in this case the variance = nan, change it 0.0
            band_stats[3] = 0.0
        features += band_stats
        return features
"""
# reshape 
#segments_ = np.reshape(segments_, (segments_.shape[0], segments_.shape[1]))

# read segments
#with rio.open(segfile) as src:
#    segments = src.read()
#    segmeta = src.meta
###############
# Spetral properties of segments
# Compute statistics and indices for segments
###############

# unique segment ids. This below is from example https://towardsdatascience.com/object-based-land-cover-classification-with-python-cbe54e9c9e24 
# May get unpractically slow on large image, consider parallelization
segment_ids = np.unique(segments)[1:] # exclude 0

objects = []
object_ids = []

# len
l = len(segment_ids)
def segStatsSimple(segment_id, segments, img_array):

    import numpy as np
    from scipy import stats as st
        
    # function to get spectral features
    def segment_features(segment_pixels):
        features = []
        npixels, nbands = segment_pixels.shape
        for b in range(nbands):
            stats = st.describe(segment_pixels[:, b], nan_policy='omit')
            band_stats = list(stats.minmax) + list(stats)[2:]
            band_stats = list(np.ma.getdata(band_stats))
            if npixels == 1:
                # in this case the variance = nan, change it 0.0
                band_stats[3] = 0.0
            features += band_stats
        
        return features

    segment_pixels = np.where(segments == segment_id, img_array, np.nan) # img[segments == ob_id]
    segpx = segment_pixels.reshape(segment_pixels[0].reshape(-1).shape[0], img_array.shape[0])
    
    object_features = segment_features(segpx)
    return object_features, segment_id
    
def segStats(segment_id, segments=segments, img_array=img):#, depth_array=depth):

    import numpy as np
    from scipy import stats as st
        
    # function to get spectral features
    def segment_features(segment_pixels):
        features = []
        npixels, nbands = segment_pixels.shape
        for b in range(nbands):
            stats = st.describe(segment_pixels[:, b], nan_policy='omit')
            band_stats = list(stats.minmax) + list(stats)[2:]
            band_stats = list(np.ma.getdata(band_stats))
            if npixels == 1:
                # in this case the variance = nan, change it 0.0
                band_stats[3] = 0.0
            features += band_stats
        
        return features

    #segment_pixels = np.where(segments == segment_id, img_array, np.nan) # img[segments == ob_id]
    #segpx = segment_pixels.reshape(segment_pixels[0].reshape(-1).shape[0], img_array.shape[0])
    # select pixels by id
    seg_img = [i[segments[0]==segment_id] for i in img]
    seg_img = np.stack(seg_img)
    # compute other statistics
    seg_ndvi = normalizedDifference(seg_img, (3,2))
    ndvi_mean = np.nanmean(seg_ndvi)
    seg_grvi = normalizedDifference(seg_img, (1,2))
    grvi_mean = np.nanmean(seg_grvi)
    seg_gcc = GCC(seg_img, (0,1,2))
    gcc_mean = np.nanmean(seg_gcc)
    # depth min max mean
#    seg_depth = np.where(segments == segment_id, depth_array, np.nan)
#    seg_depth = depth[segments==segment_id]
#    depth_min = np.nanmin(seg_depth)
#    depth_max = np.nanmax(seg_depth)
#    depth_mean = np.nanmean(seg_depth)
    
    object_features = segment_features(seg_img.T)
    # add indices to features
    object_features.append(ndvi_mean)
    object_features.append(grvi_mean)
    object_features.append(gcc_mean)
    # add depth stats
#    object_features.append(depth_min)    
#    object_features.append(depth_max)    
#    object_features.append(depth_mean)    
    
    return object_features, segment_id

# parallel processing
#pool = mp.Pool(processes=(mp.cpu_count()-1))
#test = [pool.apply(segStats, args=(x, segments, img, ndvi)) for x in segment_ids[0:1000]]

# dask
from dask import delayed
from dask import compute

delayed_funcs = []

# create delayed functions
for sg_id in segment_ids:
    delayed_funcs.append(delayed(segStats)(sg_id, segments, img))#, depth))

# execute delayed functions
start = time.time()
result = compute(delayed_funcs)
end_time = time.time()
print('Processed in %.3f minutes' % ((end_time - start)/60))

# separate segment features and segment ids to objects and object ids lists
for i in result[0]:    
    objects.append(i[0])
    object_ids.append(i[1])


    
### for loop ### super slow with large image
# time
#start = time.time()
# get spectral properties, this takes some time
#for ob_id in segment_ids:
#    segment_pixels = np.where(segments == ob_id, img, 0) # img[segments == ob_id]
#    segpx = segment_pixels.reshape(segment_pixels[0].reshape(-1).shape[0], 6)
    # ndvi average
#    seg_ndvi = np.where(segments == ob_id, ndvi, np.nan)
#    ndvi_mean = np.nanmean(seg_ndvi)
    
#    object_features = segment_features(segpx)
    # add ndvi to features
#    object_features.append(ndvi_mean)

#    objects.append(object_features)
#    object_ids.append(ob_id)
    # print progress
#    if ob_id % 10 == 0:
#        print(str(ob_id) + '/' + str(l))
# end time
#end_time = time.time()
#print('Processed in %.3f minutes' % ((end_time - start)/60))

# reshape for unsupervised classification
stats = np.zeros(shape=segments.shape)
ob_len = np.arange(0,len(objects[0]))

for o in object_ids:
    stats = np.where(segments == o, objects[object_ids.index(o)][2], stats)
stats = stats.reshape(-1,1)

# nan or inf to 0
stats = np.where(np.isfinite(stats), stats, 0)

# ------------------------ # 
# unsupervised classification
# ------------------------ #
from sklearn.cluster import k_means

kmeans = k_means(stats, n_clusters=5)
# back to 2-d array
clustered = kmeans[1].reshape(segments[0].shape)
plt.imshow(clustered)

# save
unsup_dir = os.path.join(os.path.dirname(fp_test), 'result')
if os.path.isdir(unsup_dir) == False:
    os.mkdir(unsup_dir)
unsup_out = os.path.join(unsup_dir, 'n5_clustered.tif')
unsup_meta = meta.copy()
unsup_meta.update(dtype='int32',
                  nodata=0)

with rio.open(unsup_out, 'w', **unsup_meta) as dst:
    dst.write(clustered.astype(rio.int32),1)

################
# supervised classification
# 4. read ground truth
fp_gt = "/mnt/d/users/e1008409/MK/Velmu-aineisto/velmudata_07112022_ketokari_pori_savcov_brgrrealgae_classes.gpkg"
#fp_gt = 'D:\\Users\\E1008409\\MK\\Velmu-aineisto\\velmudata_07112022_edit_bcover_subhighvasc_algae_2019+-1_T34WFT.gpkg'
#fp_gt = '/mnt/d/users/e1008409/MK/Velmu-aineisto/velmudata_07112022_pori_new.gpkg'
gdf = gpd.read_file(fp_gt)

# select 
gdf = gdf[((gdf.new_class != 7) & (gdf.syvyys_mitattu >= -3)) | (gdf.new_class == 7)]

# check crs
src = rio.open(fp)
if gdf.crs.to_epsg() != src.crs.to_epsg():
    print('Reprojecting points')
    gdf = gdf.to_crs(epsg=src.crs.to_epsg())
src.close()

# drop None rows in bcover
#gdf = gdf[gdf['bcover'].notna()]

################################
# Sample ndvi
# check geometry, explode if MultiPoint
if gdf.geometry.geom_type.str.contains('MultiPoint').any() == True:
    sp = gdf.geometry.explode()
    # get point coords
    coords = [(x,y) for x,y in zip(sp.x, sp.y)]
else:
    # get point coords
    coords = [(x,y) for x,y in zip(gdf.geometry.x, gdf.geometry.y)]
# sample ndvi
src = rio.open(ndviout)
gdf['ndvi'] = [x for x in src.sample(coords)]
# close dataset
src.close()
# extract list
gdf['ndvi'] = gpd.GeoDataFrame(gdf.ndvi.tolist(), index=gdf.index)



# unique class int for bottom cover
gdf['bcover_int'] = None
for idx, row in gdf.iterrows():
    if row['syvyys_mitattu'] < -2:
        gdf.loc[idx, 'bcover_int'] = 6
    elif row['ndvi'] > 0.3:
        gdf.loc[idx, 'bcover_int'] = 4
    elif row['bcover'].startswith('sand'):
        gdf.loc[idx, 'bcover_int'] = 1
    elif row['bcover'].startswith('soft'):
        gdf.loc[idx, 'bcover_int'] = 2
    elif row['bcover'].startswith('hard'):
        gdf.loc[idx, 'bcover_int'] = 3
    elif row['savcov'] > 30:
        gdf.loc[idx, 'bcover_int'] = 5
#    elif row['bcover'].startswith('algae'):
#        gdf['bcover_int'].loc[idx] = 4
#    elif row['bcover'].startswith('subvasc'):
#        gdf['bcover_int'].loc[idx] = 4
    elif row['bcover'].startswith('Mixed'):
        gdf.loc[idx, 'bcover_int'] = 0
    elif row['bcover'].startswith('other'):
        gdf.loc[idx, 'bcover_int'] = 0


# very simple classes
gdf['vegclass'] = 0
for idx, row in gdf.iterrows():
    if row['syvyys_mitattu'] < -4.5:
        gdf.loc[idx, 'vegclass'] = 3
    elif row['savcov'] < 30:
        gdf.loc[idx, 'vegclass'] = 2
    elif row['savcov'] >= 30:
        gdf.loc[idx, 'vegclass'] = 1
    elif row['ndvi'] > 0.3:
        gdf.loc[idx, 'vegclass'] = 4

        
###### sample segments to prevent duplicates


# open image and sample segments
src = rio.open(segfile)
gdf['segment_ids'] = [x for x in src.sample(coords)]
# close dataset
src.close()
# extract list
gdf['segment_ids'] = gpd.GeoDataFrame(gdf.segment_ids.tolist(), index=gdf.index)
# drop nan
gdf = gdf[gdf['ndvi'].notna()]

###### check that points (buffered) are within one segment 
# buffer
#gdf.geometry = gdf.geometry.buffer(5)
# use zonal stats to get raster values
# rasterstats
stats = zonal_stats(gdf.geometry, segments[0], stats=['unique'], affine=meta['transform'], all_touched=True, nodata=0)#, nodata=stack_meta['nodata'])
statlist = [d['unique'] for d in stats]
gdf['segments_in_buffer'] = statlist
# drop where > 1 segment
gdf = gdf[gdf.segments_in_buffer == 1]


# drop
gdf = gdf[gdf.segment_ids != 0] # exclude 0 (nodata)
gdf = gdf.drop_duplicates(subset=('geometry'), keep='first') # drop duplicate geometries

gdf = gdf[gdf.new_class != 0]
"""
#################### Dataframe approach ######################
#####  pixel values from image bands for each segment
# dataframe from first band
img_segments = pd.DataFrame({
  'segments': segments.reshape(-1), 
  'band1_pixels': img[0].reshape(-1)
})

# loop through other bands and add columns
for i in np.arange(1, len(img)):
    colname = 'band' + str(i+1) + '_pixels'
    temp = pd.DataFrame({
      'segments': segments.reshape(-1), 
      colname: img[i].reshape(-1)
    })
    # add column
    img_segments[colname] = temp[colname]

# del temp and release memory
del(temp)
gc.collect()

# merge sampled ground truth by segment id
img_segments = img_segments.merge(d[['classes', 'classnames', 'field_depth', 'segment_ids']], how='left', left_on='segments', right_on='segment_ids')

# select rows with ground truth
temp = img_segments[img_segments.segment_ids.notna()]

# columns where to compute stats
columns = ['segments', 'band1_pixels', 'band2_pixels', 'band3_pixels', 'band4_pixels', 'band5_pixels', 'band6_pixels']

# selection by columns
train = temp[columns] # train data
full_data = img_segments[columns]
"""


data = gdf[['segment_ids', 'savcov', 'new_class']]

data = data.rename(columns={'new_class': 'classes'})

# remove any nan rows
data = data[data.classes.notna()]
data = data.astype(int)
# segment ground truth
gt_d = dict()

from collections import Counter

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


############################
# Assign class labels to segments
############################
# new layer for segment classes
new_layer = np.zeros(shape=segments.shape)

# assign labels, this takes a little time
for idx, row in gt_d.iterrows():
    new_layer = np.where(segments == row.segment_ids, row.classes, new_layer) 
# reshape
gt = new_layer.reshape((-1))
stack_re = stack.reshape((stack.shape[0],-1)).transpose((1,0))
stack_re = np.where(np.isnan(stack_re), 0, stack_re)

# save gt
gtout = gt.reshape((1,meta['height'], meta['width']))
gt_out = os.path.join(segdir, 'n' + str(n) + '_gt.tif')
with rio.open(gt_out, 'w', **upmeta) as dst:
    dst.write(gtout.astype(rio.uint32))

# normalize data
from sklearn.preprocessing import normalize
stack_re_n = normalize(stack_re, axis=1)

# train test data
X = stack_re_n[gt > 0]
y = gt[gt > 0]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42, shuffle=True,
                                                    stratify=y)

#X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,
#                                                  test_size = 0.25,
#                                                  random_state=42, #suffle=True,
#                                                 stratify=y_train)
    
############################
# Train test dataframe OBIA statistics
df_obi = pd.DataFrame(objects, columns=['bmin', 'bmax', 'bmean', 'bvar', 'bskew', 'bkurt',
                                        'gmin', 'gmax', 'gmean', 'gvar', 'gskew', 'gkurt',
                                        'rmin', 'rmax', 'rmean', 'rvar', 'rskew', 'rkurt',
                                        'irmin', 'irmax', 'irmean', 'irvar', 'irskew', 'irkurt',
                                        'ndvimean', 'grvimean', 'gccmean', 'mind', 'maxd', 'meand'])
df_obi['seg_id'] = object_ids

# join bcover class by segment id
df_obi = df_obi.join(gt_d[['segment_id', 'classes']].set_index('segment_id'), on='seg_id')

df_obi = df_obi.dropna()
# save df
outdir = os.path.join(os.path.dirname(fp_test), 'classification')
if os.path.isdir(outdir) == False:
    os.mkdir(outdir)
df_obi.to_csv(os.path.join(outdir, 'df_obi2.csv'), sep=';')

# split
X = df_obi.drop(['classes'], axis=1)
y = df_obi[['classes', 'seg_id']]
X_train, X_test, y_train, y_test = train_test_split(X, y['classes'],
                                                    test_size=0.3,
                                                    random_state=42, shuffle=True,
                                                    stratify=y['classes'])

#####################
# Support Vector Machine
from sklearn import svm

# fit model
svm_clf = svm.SVC(decision_function_shape='ovr')
svm_clf.fit(X_train, y_train)

# predict test
predf = pd.DataFrame()
predf['truth'] = y_test
predf['predict'] = svm_clf.predict(X_test)

#############################
# Random Forest classifier

# set range of trees to test in RF
n_est = np.arange(10, 150, 5)
# list of oob errors
oob_scores = []
#n_est = [100]
for n in n_est:
    # random forest classifier
    rf = RandomForestClassifier(n_estimators=n, max_depth=None, n_jobs=5 ,max_features='sqrt', bootstrap=True, oob_score=True,
                                random_state=42)
    rf.fit(X_train, y_train)
    # out of bag error
    oob_error = np.round(rf.oob_score_, 2)    
    # add oob score to list
    oob_scores.append(oob_error)
    print('OOB error with ', str(n), ' trees is', str(oob_error))
    
# plot OOB scores
fig, ax = plt.subplots()
ax.plot(n_est, oob_scores, lw=0.5, color='black')
plt.show()

# CV
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
n_scores = cross_val_score(rf, X_test, y_test, cv=cv, scoring='accuracy')
print('n=',str(n), '%.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

predf = pd.DataFrame()
predf['truth'] = y_test
predf['predict'] = rf.predict(X_test)
# join segment_id
#predf = predf.join(y['seg_id'])
#predf = predf.drop_duplicates()

##################################
# Multi-layer perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

# test
gt_n = np.where(gt == 5, 6, gt)
# train test data
X = stack_re[gt > 0]
y = gt_n[gt_n > 0]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42, shuffle=True,
                                                    stratify=y)
# standard scale data
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
# same transformation to test data
X_test_scaled = scaler.transform(X_test)

clf = MLPClassifier(solver='lbfgs', max_iter=5000, random_state=42)
clf.fit(X_train_scaled, y_train)

# predict test
predf = pd.DataFrame()
predf['truth'] = y_test
predf['predict'] = clf.predict(X_test_scaled)


#######################
# classification report
print(metrics.classification_report(y_test, predf.predict))
# create confusion matrix
cm = metrics.confusion_matrix(predf['truth'], predf['predict'])
#val_cm = metrics.confusion_matrix(y_val, rf.predict(X_val))
# compute row and col sums
total = cm.sum(axis=0)
rowtotal = cm.sum(axis=1)


# create cm DataFrame
cmdf = np.vstack([cm,total])
b = np.array([[rowtotal[0]], [rowtotal[1]], [rowtotal[2]], 
              [rowtotal[3]], [rowtotal[4]], #[rowtotal[5]],
              [rowtotal.sum()]])
cmdf = np.hstack((cmdf, b))
cols = ['Mixed\nSAV', 'Brown\nalgae', 'Green\nalgae', 'Bare or\n LowSAV', 'Deep\n water', 'Total']
cmdf = pd.DataFrame(cmdf, index=cols,
                    columns = cols)

# print
print(pd.crosstab(predf.truth, predf.predict, margins=True))

o_accuracy = np.sum(cm.diagonal()) / np.sum(cm.sum(axis=0))
p_accuracy = cm.diagonal() / cm.sum(axis=0) # producer's accuracy
u_accuracy = cm.diagonal() / cm.sum(axis=1) # user's accuracy

# plot 
import seaborn as sns
sns.set_theme(style='white')
fig, ax = plt.subplots()
ax = sns.heatmap(cmdf, annot=True, cmap='Blues', fmt='.0f', cbar=False)
ax.xaxis.set_ticks_position('top')
ax.tick_params(axis='both', which='both', length=0)

#fig.suptitle('Bottomtype classification accuracy')
fig.suptitle('MLP Benthic classification')
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(fp), 'plots', 'mlp_2_confusion_matrix.png'), dpi=150, format='PNG')
print('Overall accuracy %.2f' % (o_accuracy))
print('Users accuracy', u_accuracy)
print('Producers accuracy', p_accuracy)


#######################################
# prediction for image stack

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
split_array = np.split(stack_re, patch_size, axis=0)
j = 0
for i in split_array: # NOTE: parallelize
    # standard scale patch
    i_scaled = scaler.transform(i)
    prediction = clf.predict(i_scaled)
    preds.append(prediction)
    print(str(j),'/',str(len(split_array)))
    j += 1

# predictions to single array
predicted = np.stack(preds)
predicted = predicted.reshape(stack_re.shape[0]) 
# prediction back to 2D array
predicted = predicted.reshape(1, meta['height'], meta['width'])

# predict as large single array
#prediction = rf.predict(stack_re)
# prediction back to 2D array
#predicted = prediction.reshape(1,img.shape[1], img.shape[2])

# mask nodata
predicted = np.where(nodatamask == True, 0, predicted)

# ndvi threshold classification (above ground areas)
predicted = np.where(ndvi >= 0.2, 8, predicted) # above water/ground vegetation
predicted = np.where((ndvi >= 0) & (ndvi < 0.2), 9, predicted)

# outfile
outdir = os.path.join(os.path.dirname(fp), 'classification')
if os.path.isdir(outdir) == False:
    os.mkdir(outdir)
outfile = os.path.join(outdir, os.path.basename(fp).split('.')[0] + '_segmpix_multiclass_pca_MLPclassification_simpler.tif')
# update metadata
upmeta = meta.copy()
upmeta.update(dtype='uint8',
              nodata=0,
              count=1)

with rio.open(outfile, 'w', **upmeta, compress='LZW') as dst:
    dst.write(predicted.astype(rio.uint8))




######################################
# Prediction for segments

# predict for all segments
obj_df = pd.DataFrame(objects, columns = df_obi.columns[:-2])
obj_df['seg_id'] = object_ids
obj_df = obj_df.dropna()
obj_df['predicted_class'] = rf.predict(obj_df[obj_df.columns[:-1]])
# create prediction raster where predicted values are associated to segment_ids
prediction = np.zeros(shape=segments.shape)
# map predicted values to segment_ids (this is slooooow)
for idx, row in obj_df.iterrows():
    prediction = np.where(segments==row['seg_id'], row['predicted_class'], prediction)
    print(str(idx), '/', str(len(obj_df)))
    
# outfile, metadata and save
outdir = os.path.dirname(fp_test)
outfile = os.path.join(outdir, 'obia_classification2.tif')

outmeta = meta.copy()
outmeta.update(dtype='uint8',
               nodata=0,
               count=1)
with rio.open(outfile, 'w', **outmeta) as dst:
    dst.write(prediction.astype(rio.uint8))












############### old ###############
obi = []

for o in objects:
    if np.isnan(o).any() == False:
        obi.append(o)

# create numpy array from rf classifiation and save to raster
clf = np.copy(segments)
for segment_id, klass in zip(segment_ids, predicted):
    clf[clf == segment_id] = klass

# mask nodata areas
clf = np.where(nodatamask == True, 0, clf)
clf = np.where(clf > 6, 0, clf)

# save
outfile = os.path.join(os.path.dirname(fp_test), os.path.basename(fp_test)[:-4] + '_seg_predicted.tif')
outmeta = meta.copy()
outmeta.update(dtype='uint16',
               nodata=0,
               count=1)

with rio.open(outfile, 'w', **outmeta) as dst:
    dst.write(clf.astype(rio.uint16))



# prediction to array and save
predout = predicted.reshape((1, img.shape[1], img.shape[2]))




# function to get basic stats
def describeStats(dataframe):
    import numpy as np
    
    # segment id
    sid = dataframe['segments'].iloc[0]
    # descriptive stats
    stats = dataframe.describe(percentiles=[], include=[np.number])
    # add segment id to column
    stats['segment'] = sid
    # drop cols, rows
    stats = stats.drop('segments', axis=1) # statistics from segment id 
    stats = stats.drop('count', axis=0) # pixel count
    # reset index
    stats = stats.reset_index()
    # rename 'index' column
    stats = stats.rename(columns={'index': 'stats'})
    # result 
    return stats 

# get unique segment ids
unique_segments = np.unique(temp.segments).tolist()
unique_segments_full = np.unique(full_data.segments).tolist()[1:] # exclude first element in the list, which is 0 - nodata

# multiprocess
pool = mp.Pool(processes=mp.cpu_count())
stats = pool.map(describeStats, [train[train.segments == seg] for seg in unique_segments]) # training data

# computing stats for all segments, Takes some time
start = time.time()
stats_full = pool.map(describeStats, [full_data[full_data.segments == seg] for seg in unique_segments_full]) # full data
end = time.time()
#segments = felzenszwalb(img_scaled, scale=1, sigma=0.9, min_size=20, multichannel=True)
elapsed = end - start
print('Time elapsed: %.2f' % elapsed, 'seconds')


# stack dataframes from list
mdf = pd.concat(stats, axis=0)
mdf_full = pd.concat(stats_full, axis=0)
# join bottom type class and field depth 
mdf = mdf.merge(gt, how='left', left_on='segment', right_on='segment')

# multi index
mdf = mdf.set_index(['segment', 'stats'])


############################
# Segment stats to arrays
############################

xds = rioxr.open_rasterio(segfile)

new = np.copy(segments)
new = np.where(segments == segment_id, mdf.iloc[0][0], new)

for segid in mdf.index.levels[0]:
    # select by segment id
    sel = mdf.loc[[segid]]
    # concatenate stats to segments

    print(segid)
    break
segdf = pd.DataFrame(segments[0])






##################################
# Train and test data
##################################

# 


train = mdf.unstack().sample(frac=0.7, random_state=42).stack()
# 
train_data = train.drop('classes', axis=1)
train_classes = train['classes']

test = mdf.drop(train.index)

test_data = test.drop('classes', axis=1)
test_classes = test['classes']


# random forest classifier
rf = RandomForestClassifier(n_estimators=500, max_depth=None, n_jobs=5 ,max_features='sqrt', bootstrap=True, oob_score=True)
rf.fit(train_data.values, train_classes.values)

cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
n_scores = cross_val_score(rf, train_data.values, train_classes.values, cv=cv, scoring='accuracy')
print('%.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))


# predict
predf = pd.DataFrame()
predf['predict'] = rf.predict(test_data.values)

# add new column to test
test['prediction'] = predf.predict.values

# 
# create confusion matrix
cm = metrics.confusion_matrix(test['classes'], test['prediction'])
#val_cm = metrics.confusion_matrix(y_val, rf.predict(X_val))
# compute row and col sums
total = cm.sum(axis=0)
rowtotal = cm.sum(axis=1)


"""

##### get segments representing different land cover types
# unique classes
classes = np.unique(d.classes)

# drop na
test = d

# test that each unique segment represent single class
for i in np.unique(test.segment_ids):
    sel = test[test.segment_ids == i]
    # round to nearest integer if multiple classes found
    if len(np.unique(sel.classes)) == 1:
        print('Found multiple classes in segment ' + str(i))
        index_min = min(test[test.segment_ids == i].index)
        index_max = max(test[test.segment_ids == i].index)
        test['classes'].loc[index_min:index_max] = round(np.mean(sel.classes))


# test that each unique segment represent single class
for i in np.unique(test.segment_ids):
    sel = test[test.segment_ids == i]
    assert len(np.unique(sel.classes)) == 1, 'Found multiple classes in segment ' + str(i)

test[['sykeid', 'classes', 'segment_ids']][test.segment_ids == 26345]

test = test.groupby('segment_ids').mean().round()

segment_list = test.index.tolist()

# train test data
selected = img_segments[['segments', 'band1_pixels', 'band2_pixels', 'band3_pixels',
       'band4_pixels', 'band5_pixels', 'band6_pixels', 'classes']]
X = selected[~selected.segments.isin(segment_list)]
y = selected[selected.segments.isin(segment_list)]


segments_per_class = {}

# time
start = time.time()
for sid in classes:
    segment_pixels = segments[gt == sid]
    segments_per_class[sid] = set(segment_pixels)
end_time = time.time()
print('Processed in %.3f' % (end_time - start))

intersection = set()
accum = set()

for class_segments in segments_per_class.values():
    intersection |= accum.intersection(class_segments)
    accum |= class_segments
assert len(intersection) == 0, 'Segment represent multiple classes'
"""

#############
# Create stack of img and segments
img = np.transpose(img, (1,2,0))
img = np.where(np.isnan(img), 0, img)
segments = np.transpose(segments, (1,2,0))

img_stack = np.dstack((img, segments))

############
# Classification
############

fp_gt = "D:\\Users\\E1008409\\MK\\freshabit\\pohjanlaatu\\training_bottomtype_points.gpkg"

# read ground truth polygons 
gdf = gpd.read_file(fp_gt) # layer='training2_sand_soft_hard_class'

# rasterize ground truth
# create dictionary of geometry and class
gtlist = list(zip(d.geometry, d.classes)) # gdf.MC_ID

# rasterize ground truth
from rasterio.features import rasterize
gt = rasterize(gtlist, out_shape=(img.shape[1:]), fill=0, 
                            transform=meta['transform'], dtype='uint8')

# train test split
X = img_stack[gt > 0]
y = gt[gt > 0]

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=123)


# initialize randomForest classifier
rf = RandomForestClassifier(n_estimators=500, max_depth=None, n_jobs=5 ,max_features='sqrt', bootstrap=True, oob_score=True)

# CV
cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=1)
n_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='accuracy')
print('%.3f (%.3f)' % (np.mean(n_scores), np.std(n_scores)))

# fit model to training data
rf.fit(X_train, y_train)

# predict
predf = pd.DataFrame()
predf['truth'] = y_test
predf['predict'] = rf.predict(X_test)

# classification report
print(metrics.classification_report(y_test, rf.predict(X_test)))
# create confusion matrix
cm = metrics.confusion_matrix(predf['truth'], predf['predict'])
#val_cm = metrics.confusion_matrix(y_val, rf.predict(X_val))
# compute row and col sums
total = cm.sum(axis=0)
rowtotal = cm.sum(axis=1)










# get unique values (0 is the background, or no data, value so it is not included) for each land cover type
classes = np.unique(gt)[1:]

# for each class (land cover type) record the associated segment IDs
segments_per_class = {}
for klass in classes:
    segments_of_class = segments[gt == klass]
    segments_per_class[klass] = set(segments_of_class)
    print("Training segments for class", klass, ":", len(segments_of_class))
 
# make sure no segment ID represents more than one class
intersection = set()
accum = set()
for class_segments in segments_per_class.values():
    intersection |= accum.intersection(class_segments)
    accum |= class_segments
    break
assert len(intersection) == 0, "Segment(s) represent multiple classes"

#    if len(intersection) == 0:
#        print('Segment represents multiple classes, removing segment')
#       for v in class_segments:
#            segments_per_class.pop(v)


train_img = np.copy(segments)
threshold = train_img.max() + 1  # make the threshold value greater than any land cover class value

# all pixels in training segments assigned value greater than threshold
for klass in classes:
    class_label = threshold + klass
    for segment_id in segments_per_class[klass]:
        train_img[train_img == segment_id] = class_label
 
# training segments receive land cover class value, all other segments 0
train_img[train_img <= threshold] = 0
train_img[train_img > threshold] -= threshold

# create objects and labels for training data
training_objects = []
training_labels = []
for klass in classes:
    class_train_object = [v for i, v in enumerate(objects) if segment_ids[i] in segments_per_class[klass]]
    training_labels += [klass] * len(class_train_object)
    training_objects += class_train_object
 
classifier = RandomForestClassifier(n_jobs=-1)  # setup random forest classifier
classifier.fit(training_objects, training_labels)  # fit rf classifier
predicted = classifier.predict(objects)  # predict with rf classifier

# create numpy array from rf classifiation and save to raster
clf = np.copy(segments)
for segment_id, klass in zip(segment_ids, predicted):
    clf[clf == segment_id] = klass

clf_img = np.where(img == nodata, nodata, clf)
    
outdir = os.path.dirname(fp) 
outfile = os.path.join(outdir, 'puruvesi_bottomtype_classified_segments.tif')

outmeta = meta.copy()
outmeta.update(dtype='uint8')

with rasterio.open(outfile, 'w', **outmeta) as dst:
    dst.write(clf_img.astype(rasterio.uint8))







        
# pool object
pool = mp.Pool(mp.cpu_count()-2)

for sid in segment_ids[0:10]:
    pool.apply_async(segment_features, args=(mimg[segments == sid]), callback=getResult)

pool.close()
pool.join()


for id in segment_ids:
    segment_pixels = img[segments == id]
    object_features = segment_features(segment_pixels)
    objects.append(object_features)
    object_ids.append(id)


from joblib import Parallel, delayed
import math

def sqrt_func(i, j):
    time.sleep(1)
    return math.sqrt(i**j)

Parallel(n_jobs=2)(delayed(sqrt_func)(i, j) for i in range(5) for j in range(2))





















