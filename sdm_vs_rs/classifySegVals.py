# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 08:39:25 2024

@author: E1008409
"""


import os
import glob
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import rasterio as rio
import rioxarray as rxr
import time
import json
import pickle
from sklearn import metrics
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import normalize, StandardScaler, QuantileTransformer, PowerTransformer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from dask import delayed
from dask import compute
import dask.bag as db


# directory for pixel value csv files
fp = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Denmark/segmentation/segvals'
# filepath for ground truth points with sampled satimg values
fp_pts = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Denmark/Eelgrass_2018-2023/Eelgrass_Kattegat__multiclass_2018_32632_LS2_sampled.gpkg'
# read
gdf = gpd.read_file(fp_pts, engine='pyogrio')
# columns to select for train, test data
cols = ['Band1', 'Band2', 'Band3', 'Band4', 'Band5', 'Band8']

# read and merge dataframes
df = pd.DataFrame()
for file in glob.glob(os.path.join(fp, '*segvals.csv')):
    df1 = pd.read_csv(file, sep=';')
    df = pd.concat([df, df1])

# test --------------------------- #
# select cols
X = np.array(df[cols])#, 'pca1', 'pca2', 'pca3']])
y = np.array(df['new_class'])

X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    shuffle=True,
                                                    stratify=y)

# random forest classifier
rf = RandomForestClassifier(n_estimators=150, max_depth=None, n_jobs=5, max_features='sqrt',
                            bootstrap=True, oob_score=True,
                            random_state=42)
# fit rf
rf.fit(X_train, y_train)
rf.score(X_test, y_test)

# predict test
predf = pd.DataFrame()
#predf['truth'] = y_test
predf['predict'] = rf.predict(X_test)
# classification report
print(metrics.classification_report(y_test, predf.predict))
#----------------------------------#

# get unique point_id, bottom class values
dfm = df.groupby('point_id')
dfg = dfm.apply(lambda x: x['new_class'].unique())
dfg = dfg.apply(pd.Series)
dfg = dfg.reset_index()
dfg.rename(columns={0: 'new_class'}, inplace=True)

# make stratified KFolds
folds = dict()
skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
for i, (train, test) in enumerate(skf.split(dfg.point_id, dfg.new_class)):
    # save train, test indices to dictionary
    k = 'fold_' + str(i+1)
    folds[k] = (train.tolist(), test.tolist())

for t in train:
#    print(t)
    if t in set(test):
        print('Value found in test')
    
gdf['proba_RF'] = None
gdf['test_fold'] = None
# evaluate
for f in folds:
    # get fold point id's for train, test indices 
    train_pts = dfg['point_id'][dfg.index.isin(folds[f][0])]
    test_pts = dfg['point_id'][dfg.index.isin(folds[f][1])]
    
    # select fold train, test from gdf
    df_train = df[df.point_id.isin(train_pts)]
    df_test = df[~df.point_id.isin(train_pts)]
    # select columns
    X_train = df_train[cols]#[['Band1', 'Band2', 'Band3', 'Band4', 'Band5', 'Band8', 'pca1', 'pca2', 'pca3']]
    y_train = df_train['new_class']
    X_test = df_test[cols]#[['Band1', 'Band2', 'Band3', 'Band4', 'Band5', 'Band8', 'pca1', 'pca2', 'pca3']]
    y_test = df_test['new_class']

    #results = cross_val_score(rf, X, y, cv=skf)
    #print("RF Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    # convert to array
    X_train_arr = np.array(X_train)
    y_train_arr = np.array(y_train)
    X_test_arr = np.array(X_test)
    y_test_arr = np.array(y_test)
    # select fold train, test from array
#    X_train_f = X[folds[f][0]]
#    y_train_f = y[folds[f][0]]
#    X_test_f = X[folds[f][1]]
#    y_test_f = y[folds[f][1]]
    
    # fit rf
    rf.fit(X_train_arr, y_train_arr)
    print('RF', rf.score(X_test_arr, y_test_arr))
    # predictions for test array
    predf = pd.DataFrame()
    predf['predict'] = rf.predict(X_test_arr)
    # classification report
    print('RF', metrics.classification_report(y_test_arr, predf.predict))
    
    # predict proba to gdf
    gdf_test = gdf[gdf.point_id.isin(test_pts)] # get test points
    gdf_test = gdf_test.dropna(subset=cols)
    gdf_train = gdf[gdf.point_id.isin(train_pts)] # get train points
    
    gdf_test_arr = np.array(gdf_test[cols]) # convert to array
    pred_proba = rf.predict_proba(gdf_test_arr) #predict
    gdf['proba_RF'].iloc[gdf_test.index] = pred_proba[:,0] # predicted value to gdf
    
    gdf['test_fold'].iloc[gdf_test.index] = f  

# save
gdf['proba_RF'] = gdf['proba_RF'].astype(float)
gdf_out = fp_pts.split('.')[0] + '_preds.gpkg'
gdf.to_file(gdf_out, engine='pyogrio')
    
# TODO plots?
# ----------------------------------------------------------------------- #
# TODO consider moving layer prediction to separate script
# fit all data
rf.fit(X, y)

# create layer with Bands and PCA for prediction
fp_img = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Finland/S2_LS1_20180715_rrs_v1_clip_ext.tif'
fp_img = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Denmark/S2_LS2a_20220812_v1_3035.tif'
fp_pca = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Denmark/S2_LS2c_20220812_v1_3035_pca.tif'

with rio.open(fp_img) as src:
    img = src.read((1,2,3,4,5,8))
    meta = src.meta
# pca
with rio.open(fp_pca) as src:
    pca = src.read((1,2,3))
    

# transpose and stack
img = img.transpose(1,2,0)
pca = pca.transpose(1,2,0)
img_stack = np.dstack((img, pca))
img_stack = img_stack.transpose(2,0,1)
img = img.transpose(2,0,1)
pca = pca.transpose(2,0,1)

img_stack = img
# reshape to 1d
img_re = img_stack.reshape((img_stack.shape[0],-1)).transpose((1,0))
img_re = np.where((np.isnan(img_re)) | (img_re == meta['nodata']), 0, img_re) # replace nans
preds = [] # list for split array predictions
# find largest number within range for modulo 0
modulos = []
for i in np.arange(32,1024,1):
    if len(img_re) % i == 0:
        modulos.append(i)
patch_size = np.max(modulos)        

# split for prediction
split_array = np.split(img_re, patch_size, axis=0)
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
predicted = predicted.reshape(img_re.shape[0]) 
#predicted = le.inverse_transform(predicted) # transform
# prediction back to 2D array
predicted = predicted.reshape(1, meta['height'], meta['width'])
# mask nodata
nodatamask = np.where((img[2] == meta['nodata']) | (np.isnan(img[2])), 0, 1)
#plt.imshow(nodatamask)
predicted = np.where(nodatamask == 0, 0, predicted)
# outfile
outdir = os.path.join(os.path.dirname(fp_img), 'classification')
if os.path.isdir(outdir) == False:
    os.mkdir(outdir)
outfile = os.path.join(outdir, os.path.basename(fp_img).split('.')[0] + '_RF_segvals.tif')
# update metadata
upmeta = meta.copy()
upmeta.update(dtype='uint8',
              nodata=0,
              count=1)
# save
with rio.open(outfile, 'w', **upmeta, compress='LZW') as dst:
    dst.write(predicted.astype(rio.uint8))
# -------------------------------------------------------------- #    














