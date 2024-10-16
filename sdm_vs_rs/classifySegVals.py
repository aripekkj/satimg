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
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score
from dask import delayed
from dask import compute
import dask.bag as db
import gc

########################################################
# filepaths
# sansibar
fp = '/mnt/d/users/e1008409/MK/sansibar/Sansibar_WV2_25_9_2016/segmentation/segvals'
fp_pts = '/mnt/d/users/e1008409/MK/sansibar/Opetusdatat/habitat_field_data_wv2_encoded01_sampled.gpkg'

# directory for pixel value csv files
fp = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Kreikka/segmentation/segvals'

# filepath for ground truth points with sampled satimg values
fp_pts = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Kreikka/Greece_habitat_data3035_encodedLSxGreece_10m_sampled.gpkg'
fp_pts = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Finland/LG_AHV_aineistot_2024-02-23_selkameri_south_3035_classes_encodedLS1_20180715_sampled.gpkg'
fp_pts = '/mnt/d/users/e1008409/MK/Velmu-aineisto/sdm_vs_rs/velmudata_07112022_selkameri_south_bounds_edit3035_encodedLS1_20180715_sampled.gpkg'

########################################################
# read data
gdf = gpd.read_file(fp_pts, engine='pyogrio')
#gdf = gdf.rename(columns={'sykeid':'point_id'})
# columns to search for na
cols = ['Band1', 'Band2', 'Band3', 'Band4', 'Band5', 'Band6', 'Band7', 'Band8']

# check if na points exist
nans = gdf[gdf[cols].isna().any(axis=1)]
#nanout = fp_pts.split('.')[0] + '_nans.gpkg'
#nans.to_file(nanout, engine='pyogrio')
if len(nans) > 0:
    # drop nans
    #gdf = gdf.dropna(subset=cols, how='all')
    gdf = gdf[~gdf.index.isin(nans.index)]
gdf['duplicated'] = gdf.duplicated(subset=cols, keep=False) # check for duplicates
print('Duplicated rows:', len(gdf[gdf['duplicated']==True]))
gdf = gdf[gdf['duplicated'] == False] # drop duplicates that resulted from nodata (Finnish data, define nodata to raster)

gdf_out = fp_pts.split('.gpkg')[0] + '_dropna.gpkg'
gdf.to_file(gdf_out, engine='pyogrio')
gdf = gpd.read_file(gdf_out, engine='pyogrio')

# read and merge dataframes
df = pd.DataFrame()
for file in glob.glob(os.path.join(fp, '*segvals.csv')):
    df1 = pd.read_csv(file, sep=';')
    df = pd.concat([df, df1])
# set train cols
traincols = df.columns[1:-3]
traincols = traincols.drop(['Band6', 'Band7', 'Band9', 'Band10', 'pca4', 'pca5', 'pca6', 'pca7', 'pca8', 'pca9', 'pca10'])
df = df[~df[traincols].isna().any(axis=1)] # exclude nan

# select cols
X = np.array(df[traincols])#, 'pca1', 'pca2', 'pca3']])
#X = np.where(np.isnan(X), 0, X) # replace nan
y = np.array(df['new_class'])
# standardize data
scaler = StandardScaler().fit(X)
X = scaler.transform(X)
le = LabelEncoder() 
le.fit(np.unique(y)) # fit classes
y = le.transform(y) # transform
print(np.unique(y, return_counts=True)) # check result

########################################################
# define models #
models = {'RF': {'model': RandomForestClassifier(n_jobs=6),
                 'params': {"n_estimators": [50, 100, 150, 200, 500], "max_features": ['sqrt', 'log2']}
                 },
          'SVM': {'model': SVC(probability=True),
                  'params': {"kernel": ['poly', 'rbf', 'sigmoid'], "C": [100, 10, 1.0, 0.1, 0.01], "gamma": [100, 10, 1.0, 0.1, 0.01, 'scale']}},
          'XGB': {'model': XGBClassifier(),
                  'params': {'objective': ['multi:softmax'], 'eval_metric': ['mlogloss'],
                             'n_estimators': [50, 100, 150], 'max_depth': [3,6],
                             'subsample': [0.5], 'num_class': [len(np.unique(y))]}
                  }
#          'LightGBM':
#          'MLP':
          }

# Stratified KFold for hyperparameter tuning
skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
# hyperparameter optimization
for m in models:
    gcv = GridSearchCV(models[m]['model'], param_grid=models[m]['params'], scoring='accuracy', cv=skf)
    result = gcv.fit(X,y)
    # summarize result
    print('Scores: %s' % result.scoring)
    print('Best Score: %s' % result.best_score_)
    print('Best Hyperparameters: %s' % result.best_params_)
    # save best params
    models[m]['best_params'] = result.best_params_

#----------------------------------#
#params = {'objective': 'multi:softmax', 'eval_metric': 'mlogloss'}
#bst = XGBClassifier(n_estimators=100, max_depth=10, learning_rate=0.1, subsample=0.5, num_class=len(np.unique(y)), **params)
#eval_set = [(X_test, y_test)]
#bst.fit(X_train, y_train, eval_set=eval_set)
#y_pred = bst.predict(X_test)
#acc = accuracy_score(y_test, y_pred)


#----------------------------------#

# make stratified KFolds
folds = dict()
skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
# use field obs points
for i, (train, test) in enumerate(skf.split(gdf.point_id, gdf.new_class)):
    # save train, test point_id's to dictionary
    k = 'fold_' + str(i+1)
    tr_pts = gdf['point_id'].iloc[train].tolist() # get point_id's by index
    te_pts = gdf['point_id'].iloc[test].tolist()
    folds[k] = (tr_pts, te_pts)
    #double check that sets are separate (can be removed later)
    for t in train:
    #    print(t)
        if t in set(test):
            print('Value found in test')
    
gdf['test_fold'] = None
# create columns for all class probabilities
proba_cols = []

for m in models:
    for k in sorted(gdf.hab_class.unique()):
        proba_col_name = 'proba_' + m + '_' + str(k)
        proba_cols.append(proba_col_name)
# add columns to dataframe
for p in proba_cols:
    gdf[p] = None

# accuracy df
df_acc_cols = ['RF_OA', 'SVM_OA', 'XGB_OA']
df_acc = pd.DataFrame(index=folds.keys(),  columns=[df_acc_cols])

# evaluate
for f in folds:
    # select pixel values by point_id
    df_train = df[df.point_id.isin(folds[f][0])]
    df_test = df[df.point_id.isin(folds[f][1])]
    
    # select columns
    X_train = df_train[traincols]#[['Band1', 'Band2', 'Band3', 'Band4', 'Band5', 'Band8', 'pca1', 'pca2', 'pca3']]
    y_train = df_train['new_class']
    X_test = df_test[traincols]#[['Band1', 'Band2', 'Band3', 'Band4', 'Band5', 'Band8', 'pca1', 'pca2', 'pca3']]
    y_test = df_test['new_class']

    # convert to array
    X_train_arr = np.array(X_train)
    y_train_arr = le.transform(np.array(y_train))
    X_test_arr = np.array(X_test)
    y_test_arr = le.transform(np.array(y_test))
    print('Classes in train set', np.unique(y_train_arr ,return_counts=True))
    print('Classes in test set', np.unique(y_test_arr ,return_counts=True))
    
    # fit classifier
    for m in models:
        clf = models[m]['model'].set_params(**models[m]['best_params']) # set model parameters according to GridSearchCV
        clf.fit(X_train_arr, y_train_arr) # fit data
        print(m, clf.score(X_test_arr, y_test_arr))
        
        # predictions for test array
        predf = pd.DataFrame()
        predf['predict_clf'] = clf.predict(X_test_arr)
        # classification report
        print(m, metrics.classification_report(y_test_arr, predf.predict_clf))
        
        # add result to df
        df_acc.loc[f, m + '_OA'] = accuracy_score(y_test_arr, predf.predict_clf)
#    precision_score(y_test_arr, predf.predict_rf, labels=[1,2])
#    recall_score(y_test_arr, predf.predict_rf)
        
        # predict proba to gdf
        gdf_test = gdf[gdf.point_id.isin(folds[f][1])] # get test points
        gdf_test = gdf_test.dropna(subset=traincols)
        
        gdf_test_arr = scaler.transform(np.array(gdf_test[traincols])) # convert to array and standardize
        pred_proba = clf.predict_proba(gdf_test_arr) #predict
    
        # select columns
        probacols = [c for c in proba_cols if m in c]
        # add predictions for all classes
        for n in np.arange(len(probacols)): 
            print(n)
            gdf[probacols[n]].iloc[gdf_test.index] = pred_proba[:,n]
        
    # predicted value to gdf
    gdf['test_fold'].iloc[gdf_test.index] = f  

# save
#gdf['proba_RF'] = gdf['proba_RF'].astype(float)
gdf[proba_cols] = gdf[proba_cols].astype(float)

gdf_out = os.path.join(os.path.dirname(fp_pts), os.path.basename(fp_pts).split('_')[0] + '_preds.gpkg') # img-pca-filt-bathy-class
gdf.to_file(gdf_out, engine='pyogrio')

# model dir
modeldir = os.path.join(os.path.dirname(os.path.dirname(fp)), 'model')
if os.path.isdir(modeldir) == False:
    os.mkdir(modeldir)

# plot overall accuracies
figpath = os.path.join(modeldir, 'ML_OA_accuracies.png')
fig, ax = plt.subplots()
ax.boxplot(df_acc)
ax.set_xticklabels(['RF', 'SVM', 'XGB'])
plt.suptitle('10-fold CV Overall accuracies')
plt.savefig(figpath, dpi=150)
plt.show()

for m in models:
    clf = models[m]['model']
    # fit all data
    clf.fit(X, y)

    clf_out = os.path.join(modeldir, m + '_obia.sav') #TODO automatize filename creation
    pickle.dump(clf, open(clf_out, 'wb'))

# LS1Finland_velmu2022_img-pca-filt-class_RF_obia_px.sav
# ----------------------------------------------------------------------- #
# TODO consider moving layer prediction to separate script
# TODO read model

fp_img = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Kreikka/S2_LSxGreece_10m_20230828_v101_3035_clip_bands.tif'
fp_img = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Denmark/S2_LS2a_20220812_v1_3035.tif'
fp_img = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Norway/S2_LS3Norway_B_10m_20170721_v1_clip.tif'

# paths for rasters
fp_dir = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Kreikka/tiles'
fp_pcadir = os.path.join(os.path.dirname(fp_dir),'pca')
fp_filtdir = os.path.join(os.path.dirname(fp_dir), 'filters')
fp_bathydir = os.path.join(os.path.dirname(fp_dir), 'SDB')
# model filepath
fp_model = os.path.join(os.path.dirname(fp_dir), 'model', 'XGB_obia.sav')

#sansibar
fp_tile = '/mnt/d/users/e1008409/MK/sansibar/Sansibar_WV2_25_9_2016/tiles/01_.tif'
fp_pca = '/mnt/d/users/e1008409/MK/sansibar/Sansibar_WV2_25_9_2016/tiles/pca/16SEP25073556-M2AS-056096543010_pca_tile_.tif'


# map image tiles and respective tiles for aux layers
tiles = [os.path.basename(file).split('.')[0][-3:] for file in glob.glob(os.path.join(fp_dir, '*.tif'))]

# TODO wrap code below to functions and parallel computing
for tileid in tiles:
    # find files with suffix
    fp_tile = glob.glob(os.path.join(fp_dir, '*' + tileid + '.tif'))[0]
    fp_pca = glob.glob(os.path.join(fp_pcadir, '*' + tileid + '.tif'))[0]
    fp_filt = glob.glob(os.path.join(fp_filtdir, '*' + tileid + '.tif'))[0]
    fp_bathy = glob.glob(os.path.join(fp_bathydir, '*' + tileid + '.tif'))[0]
    
    
    
    with rio.open(fp_tile) as src:
        img = src.read((1,2,3,4,5,8))
        meta = src.meta
    # pca
    with rio.open(fp_pca) as src:
        pca = src.read((1,2,3))
    with rio.open(fp_filt) as src:
        filt = src.read()
    with rio.open(fp_bathy) as src:
        bathy = src.read()
#    bathy = np.pad(bathy[0], (0,1), 'edge')
#    bathy = np.expand_dims(bathy, axis=0)        
    # create nodata mask
    nodatamask = np.where((img[2] == meta['nodata']) | (np.isnan(img[2])), 0, 1)
    
    # read model 
    clf = pickle.load(open(fp_model, 'rb'))
    
    # transpose and stack (if using other than image bands)
    img = img.transpose(1,2,0)
    pca = pca.transpose(1,2,0)
    filt = filt.transpose(1,2,0)
    bathy = bathy.transpose(1,2,0)
    
    img_stack = np.dstack((img, pca, filt, bathy))
    img_stack = img_stack.transpose(2,0,1)
    img = img.transpose(2,0,1)
    pca = pca.transpose(2,0,1)
    filt = filt.transpose(2,0,1)
    bathy = bathy.transpose(2,0,1)
    
    #img_stack = img
    del(img)
    gc.collect()
    # reshape to 1d
    img_re = img_stack.reshape((img_stack.shape[0],-1)).transpose((1,0))
    img_re = np.where((np.isnan(img_re)) | (img_re == meta['nodata']), 0, img_re) # replace nans
    preds = [] # list for split array predictions
    predsproba = [] # list for split array predictions
    del(img_stack)
    gc.collect()
    # standardize data
    img_re = scaler.transform(img_re)
    img_re_shape = img_re.shape
    
    # find largest number within range for modulo 0
    modulos = []
    for i in np.arange(2,1024,1):
        if len(img_re) % i == 0:
            modulos.append(i)
    patch_size = np.max(modulos)        
    
    # split for prediction
    split_array = np.split(img_re, patch_size, axis=0)
    del(img_re)
    gc.collect()
    
    j = 1
    for i in split_array: # NOTE: parallelize
        #prediction = rf.predict(i)
        prediction = clf.predict(i)
        #prediction = clf.predict(i)
        preds.append(prediction)
        #predproba = rf.predict_proba(i)
        #predsproba.append(predproba)
        print(str(j),'/',str(len(split_array)))
        j += 1
    # patch predictions back to single array
    predicted = np.stack(preds)
    predicted = predicted.reshape(img_re_shape[0]) 
    predicted = le.inverse_transform(predicted) # transform
    print(np.unique(predicted))
    # prediction back to 2D array
    predicted = predicted.reshape(1, meta['height'], meta['width'])
    plt.imshow(nodatamask)
    predicted = np.where(nodatamask == 0, 0, predicted)
    # outfile
    outdir = os.path.join(os.path.dirname(fp_dir), 'classification')
    if os.path.isdir(outdir) == False:
        os.mkdir(outdir)
    outfile = os.path.join(outdir, os.path.basename(fp_tile).split('.')[0] + '_' + os.path.basename(fp_model).split('.sav')[0] + '.tif')
    # update metadata
    upmeta = meta.copy()
    upmeta.update(dtype='uint8',
                  nodata=0,
                  count=1)
    # save
    with rio.open(outfile, 'w', **upmeta, compress='LZW') as dst:
        dst.write(predicted.astype(rio.uint8))
    # -------------------------------------------------------------- #    

# print habitat and encoded class
print(gdf[['hab_class', 'new_class']].groupby('hab_class').mean())










