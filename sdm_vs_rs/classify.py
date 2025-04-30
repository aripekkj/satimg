# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 08:38:51 2024

@author: E1008409
"""

import sys
import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import rasterio as rio
import rioxarray as rxr
import time
import json
import pickle
from sklearnex import patch_sklearn 
patch_sklearn()
from sklearn import metrics
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from dask import delayed
from dask import compute
import dask.bag as db
import gc


basedir = sys.argv[1]
fp_pts = sys.argv[2]
use_bathymetry = True

# read pts to get class names
gdf = gpd.read_file(fp_pts, engine='pyogrio')
print(gdf[['hab_class_ml', 'int_class']].groupby('hab_class_ml').mean())
habitats = sorted(gdf.hab_class_ml.unique().tolist())
# create band descriptions
descriptions = [i + ' prob' for i in habitats]
# get pca variance
fp_pca = os.path.join(basedir, 'pca', '*pca_var.csv')
pcafile = [f for f in glob.glob(fp_pca)][0]
pcavar = pd.read_csv(pcafile, sep=',')
# compute difference between rows
pcavar['diff'] = pcavar['PCA_var'].diff()
# get threshold where explained variance increases < 1
threshold = pcavar[pcavar['diff'] < 1].index[0]
# list of pca cols to select
pcs = list(np.arange(1, threshold+1))


tiledir = os.path.join(basedir, 'tiles')
# model filepath
modeldir = os.path.join(basedir, 'model')
# list models
models = [m for m in glob.glob(os.path.join(modeldir, '*.sav'))]
#scaler_fp = os.path.join(modeldir, 'scaler.sav')
#models.remove(scaler_fp) # remove scaler from model list
#scaler = pickle.load(open(scaler_fp, 'rb')) # load scaler
# map tilenames
tiles = [os.path.basename(file).split('.')[0][-3:] for file in glob.glob(os.path.join(tiledir, '*.tif'))]


def predict(tileid, basedir, model_fp, band_descriptions, pcs, use_bathymetry):
    # find files with tile suffix
    fp_tile = glob.glob(os.path.join(basedir, 'tiles', '*' + tileid + '.tif'))[0]
    fp_pca = glob.glob(os.path.join(basedir, 'pca', '*' + tileid + '.tif'))[0]
    if use_bathymetry == True:
        fp_bathy = glob.glob(os.path.join(basedir, 'bathymetry', '*' + tileid + '.tif'))[0]
    
    with rio.open(fp_tile) as src:
        img = src.read((1,2,3,4,5,8))
        meta = src.meta
    # pca
    with rio.open(fp_pca) as src:
        pca = src.read(pcs)
    if use_bathymetry == True:
        with rio.open(fp_bathy) as src:
            bathy = src.read()
    # create nodata mask
    nodatamask = np.where((img[2] == meta['nodata']) | (np.isnan(img[2])), 0, 1)
    
    # transpose and stack (if using other than image bands)
    img = img.transpose(1,2,0)
    pca = pca.transpose(1,2,0)
    if use_bathymetry == True:
        bathy = bathy.transpose(1,2,0)
    if use_bathymetry == True:    
        img_stack = np.dstack((img, pca, bathy)) 
    else:
        img_stack = np.dstack((img, pca)) 
    img_stack = img_stack.transpose(2,0,1)
    img = img.transpose(2,0,1)
    pca = pca.transpose(2,0,1)
    if use_bathymetry == True:
        bathy = bathy.transpose(2,0,1)
        
    #img_stack = img
    del(img)
    gc.collect()
    # reshape to 1d
    img_re = img_stack.reshape((img_stack.shape[0],-1)).transpose((1,0))
    img_re = np.where((np.isnan(img_re)) | (img_re == meta['nodata']), 0, img_re) # replace nans
    
    del(img_stack)
    gc.collect()
    # standardize data
#    img_re = scaler.transform(img_re)
#    img_re_shape = img_re.shape

    # read model 
    clf = pickle.load(open(model_fp, 'rb'))
# Below was used if the image was too big to predict as a single stack
#    predsproba = [] # list for split array predictions
    # find largest number within range for modulo 0. 
#    modulos = []
#    for i in np.arange(2,1024,1):
#       if len(img_re) % i == 0:
#            modulos.append(i)
#    patch_size = np.max(modulos)        
    
    # split for prediction
#    split_array = np.split(img_re, patch_size, axis=0)
#    del(img_re)
#    gc.collect()
    
#    j = 1
#    for i in split_array: # NOTE: parallelize
        #prediction = rf.predict(i)
        #prediction = clf.predict(i)
        #prediction = clf.predict(i)
        #preds.append(prediction)
#        predproba = clf.predict_proba(i)
#        predsproba.append(predproba)
#        print(str(j),'/',str(len(split_array)))
#        j += 1
    # patch predictions back to single array
    #predicted = np.stack(predsproba)
    # predict
    predicted = clf.predict_proba(img_re)
#    predicted = predicted.reshape(img_re_shape[0], len(clf.classes_))
    # prediction back to 2D array
    predicted = predicted.reshape(meta['height'], meta['width'],len(clf.classes_)).transpose(2,0,1)
#    plt.imshow(nodatamask)
    predicted = np.where(nodatamask == 0, np.nan, predicted)
    # outfile
    outdir = os.path.join(basedir, 'classification')
    if os.path.isdir(outdir) == False:
        os.mkdir(outdir)
    outfile = os.path.join(outdir, os.path.basename(fp_tile).split('.')[0] + '_' + os.path.basename(model_fp).split('.sav')[0] + '_proba.tif')
    # update metadata
    upmeta = meta.copy()
    upmeta.update(count=len(clf.classes_))
    # save
    with rio.open(outfile, 'w', **upmeta, compress='LZW') as dst:
        dst.write(predicted.astype(upmeta['dtype']))
        dst.descriptions = tuple(band_descriptions)
    # -------------------------------------------------------------- #


# create delayed functions
delayed_funcs = []
for t in tiles:
    for m in models:
        pred = delayed(predict(t, basedir, m, descriptions, pcs, use_bathymetry))
        delayed_funcs.append(pred)    
# compute
compute(delayed_funcs)















