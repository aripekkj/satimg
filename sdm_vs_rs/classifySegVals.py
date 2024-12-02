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
from sklearn.model_selection import cross_val_score, StratifiedKFold, StratifiedGroupKFold
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
fp = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Norway/segmentation/segvals'
fp = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Estonia/segmentation/segvals'
fp = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Kreikka/segmentation/segvals'
fp = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Netherlands/S2/20171015/segmentation/segvals'

# filepath for ground truth points with sampled satimg values
fp_pts = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Norway/Norway_habitat_data_3035_edit_encoded_LS3Norway.gpkg'
fp_pts = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Estonia/Estonia_habitat_data_hab_class_3035_edit_encoded_LS1Est_2015.gpkg'
fp_pts = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Netherlands/Wadden_Sea_habitat_data_encoded_LSxNL_20171015.gpkg'
fp_pts = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/BlackSea/Black_Sea_habitat_data_3035_encoded_LSxBLK_20200313.gpkg'
fp_pts = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Kreikka/Greece_habitat_data_ml_encoded_LSxGreece_10m.gpkg'
fp_pts = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Finland/LG_AHV_aineistot_2024-02-23_selkameri_south_3035_classes_encodedLS1_20180715_sampled.gpkg'
fp_pts = '/mnt/d/users/e1008409/MK/Velmu-aineisto/sdm_vs_rs/velmudata_07112022_selkameri_south_bounds_edit3035_encodedLS1_20180715_sampled.gpkg'

########################################################
# model dir
modeldir = os.path.join(os.path.dirname(os.path.dirname(fp)), 'model')
if os.path.isdir(modeldir) == False:
    os.mkdir(modeldir)
prefix = os.path.basename(fp_pts).split('_')[0]
# read data
gdf = gpd.read_file(fp_pts, engine='pyogrio')

# read and merge dataframes
df = pd.DataFrame()
for file in glob.glob(os.path.join(fp, '*segvals.csv')):
    print(file)
    df1 = pd.read_csv(file, sep=';')
    df = pd.concat([df, df1])
del(df1)
# set train cols
traincols = df.columns[1:-3]
traincols = traincols.drop(['Band6', 'Band7', 'Band9', 'Band10', 'pca4', 'pca5', 'pca6', 'pca7', 'pca8', 'pca9', 'pca10'])
#df[traincols] = df[traincols].replace(0, np.nan)
#df_nan = df[df[traincols].isna().any(axis=1)] # exclude nan
#df_nan.to_csv(os.path.join(fp, 'df_nan.csv'), sep=';')
# replace 0 with nan

# select cols
#X = np.array(df[traincols])#, 'pca1', 'pca2', 'pca3']])
#y = np.array(df['int_class'])
# standardize data
scaler = StandardScaler().fit(df[traincols])
#X = scaler.transform(df[traincols])
le = LabelEncoder() 
le.fit(np.unique(df.int_class)) # fit classes
#y = le.transform(y) # transform
print(np.unique(df.int_class, return_counts=True)) # check result

########################################################
# define models #

models = {'RF': {'model': RandomForestClassifier(n_jobs=6),
                 'params': {"n_estimators": [50, 150, 200, 500], "max_depth": [3,6], "max_features": ['sqrt', 'log2']}
                 },
          'SVM': {'model': SVC(probability=True),
                  'params': {"kernel": ['rbf',], "C": [100, 10, 1.0, 0.1, 0.01], "gamma": [100, 10, 1.0, 0.1, 0.01]}},
          'XGB': {'model': XGBClassifier(),
                  'params': {'objective': ['multi:softmax'], 'eval_metric': ['mlogloss'], 'learning_rate':[0.1, 0.3],
                             'n_estimators': [50, 150, 200, 500], 'max_depth': [3,6],
                             'subsample': [0.5], 'num_class': [len(np.unique(df.int_class))]}
                  }
#          'LightGBM':
#          'MLP':
          }
# data split for optimization
X_train_ids, X_test_ids, y_train, y_test = train_test_split(gdf.point_id, gdf.int_class, 
                                                            train_size= 0.7, random_state=42,
                                                            stratify=gdf.int_class)
Xtr_opt = scaler.transform(df[traincols][df.point_id.isin(X_train_ids)])
ytr_opt = le.transform(df['int_class'][df.point_id.isin(X_train_ids)])
Xte_opt = scaler.transform(df[traincols][df.point_id.isin(X_test_ids)])
yte_opt = le.transform(df['int_class'][df.point_id.isin(X_test_ids)])
groups = np.array(df['point_id'][df.point_id.isin(X_train_ids)])
# Stratified KFold for hyperparameter tuning
sgkf = StratifiedGroupKFold(n_splits=10, shuffle=False)
for i, (train_index, test_index) in enumerate(sgkf.split(Xtr_opt, ytr_opt, groups)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"         group={groups[train_index]}")
    print(f"  Test:  index={test_index}")
    print(f"         group={groups[test_index]}")
    print('Classes in train set', np.unique(ytr_opt[train_index] ,return_counts=True))
    print('Classes in test set', np.unique(ytr_opt[test_index] ,return_counts=True))
# hyperparameter optimization
for m in models:
    gcv = GridSearchCV(models[m]['model'], param_grid=models[m]['params'], scoring='accuracy', cv=sgkf)
    result = gcv.fit(Xtr_opt, ytr_opt, groups=groups)
    # summarize result
    print('Scores: %s' % result.scoring)
    print(m, 'Best Score: %s' % result.best_score_)
    print(m, 'Best Hyperparameters: %s' % result.best_params_)
    # save best params
    models[m]['best_params'] = result.best_params_

# permutation importance
from sklearn.inspection import permutation_importance
for m in models:
    model = models[m]['model'].fit(Xtr_opt, ytr_opt)
    perm_result = permutation_importance(
        model, Xte_opt, yte_opt, n_repeats=10, scoring='accuracy')
    sorted_importances_idx = perm_result.importances_mean.argsort() # use if you want to plot by sorted values
    sorted_importances = pd.DataFrame(
        perm_result.importances[sorted_importances_idx].T,
        columns=traincols[sorted_importances_idx],
        )
    importances = pd.DataFrame(
        perm_result.importances.T,
        columns=traincols,
        )
    # plot
    fig, ax = plt.subplots()
    ax = importances.plot.box(vert=False, whis=10)
    ax.axvline(0, ls='--', color='black', alpha=0.6)
    ax.set_title(m + ' permutation importance')
    plt.tight_layout()
    plot_out = os.path.join(modeldir, prefix + '_' + m + '_perm_importance.png')
    plt.savefig(plot_out, dpi=300, format='PNG')

# try model performance on test
for m in models:
    predf = pd.DataFrame()
    predf['truth'] = yte_opt
    clf = models[m]['model'].set_params(**models[m]['best_params']) # set best model parameters from GridSearchCV
    clf.fit(Xtr_opt, ytr_opt)
    predf['predict'] = clf.predict(Xte_opt)
    # create confusion matrix
    cm = metrics.confusion_matrix(predf['truth'], predf['predict'])
    val_cm = metrics.confusion_matrix(yte_opt, clf.predict(Xte_opt))
    # compute row and col sums
    total = cm.sum(axis=0)
    rowtotal = val_cm.sum(axis=1)
    rowtotal = np.expand_dims(rowtotal, axis=0).T #expand dims and transpose
    rowtotal_sum = np.array(rowtotal.sum()) 
    rowtotal = np.vstack([rowtotal, rowtotal_sum]) # stack row sum
    # create cm DataFrame
    cmdf = np.vstack([cm,total]) # vertical stack
    cmdf = np.hstack((cmdf, rowtotal)) # horizontal stack
    cm_cols = gdf.hab_class.unique().tolist()
    cm_cols.append('Total')
    cmdf = pd.DataFrame(cmdf, index=cm_cols,
                        columns = cm_cols)
    # save confusion matrix dataframe as csv
    cmdf_name = m + '_cm.csv'
    cmdf_out = os.path.join(modeldir, cmdf_name)
    cmdf.to_csv(cmdf_out, sep=';')
    # print
    print(pd.crosstab(predf.truth, predf.predict, margins=True))
    # compute common accuracy metrics
    o_accuracy = np.sum(cm.diagonal()) / np.sum(cm.sum(axis=0))
    p_accuracy = cm.diagonal() / cm.sum(axis=0) # producer's accuracy
    u_accuracy = cm.diagonal() / cm.sum(axis=1) # user's accuracy
    print(m + ' Overall accuracy %.2f' % (o_accuracy))
    print(m + ' Users accuracy', u_accuracy)
    print(m + ' Producers accuracy', p_accuracy)    
    # plot 
#    import seaborn as sns
#    sns.set_theme(style='white')
#   fig, ax = plt.subplots()
#    ax = sns.heatmap(cmdf, annot=True, cmap='Blues', fmt='.0f', cbar=False)
#   ax.xaxis.set_ticks_position('top')
#    ax.tick_params(axis='both', which='both', length=0)
#    fig.suptitle('Classification accuracy')
#    plt.tight_layout()
    #plt.savefig(os.path.join(os.path.dirname(fp), 'plots', 'filename.png'), dpi=150, format='PNG')

#----------------------------------#

# make stratified KFolds
folds = dict()
skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
# use field obs points
for i, (train, test) in enumerate(skf.split(gdf.point_id, gdf.int_class)):
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

# evaluate
for f in folds:
    # select pixel values by point_id
    df_train = df[df.point_id.isin(folds[f][0])]
    df_test = df[df.point_id.isin(folds[f][1])]
    
    # select columns
    X_train = scaler.transform(df_train[traincols])
    y_train = le.transform(df_train['int_class'])
    X_test = scaler.transform(df_test[traincols])
    y_test = le.transform(df_test['int_class'])

    print('Classes in train set', np.unique(y_train ,return_counts=True))
    print('Classes in test set', np.unique(y_test ,return_counts=True))
    
    # fit classifier
    for m in models:
        clf = models[m]['model'].set_params(**models[m]['best_params']) # set model parameters according to GridSearchCV
        clf.fit(X_train, y_train) # fit data
        print(m, clf.score(X_test, y_test))
        
        # predictions for test array
        predf = pd.DataFrame()
        predf['predict_clf'] = clf.predict(X_test)
        # classification report
        print(m, metrics.classification_report(y_test, predf.predict_clf))
        
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
gdf[proba_cols] = gdf[proba_cols].astype(float)

gdf_out = os.path.join(os.path.dirname(fp_pts), prefix + '_preds.gpkg') # img-pca-filt-bathy-class
gdf.to_file(gdf_out, engine='pyogrio')

# save models
X = scaler.transform(df[traincols])
y = le.transform(df.int_class) # transform
for m in models:
    clf = models[m]['model']
    # fit all data
    clf.fit(X, y)
    clf_out = os.path.join(modeldir, m + '_' + prefix + '.sav') # filename
    pickle.dump(clf, open(clf_out, 'wb'))

# ----------------------------------------------------------------------- #
# TODO consider moving layer prediction to separate script
# TODO read model

fp_img = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Netherlands/S2/20171015/S2_LSxNL_20171015_v101_rrs_clip.tif'
fp_img = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/BlackSea/S2_LSxBLK_20200313_v1_3035.tif'
fp_img = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Kreikka/S2_LSxGreece_10m_20230828_v101_3035_clip_bands.tif'
fp_img = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Denmark/S2_LS2a_20220812_v1_3035.tif'
fp_img = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Norway/S2_LS3Norway_C_10m_20170721_v1_clip.tif'
fp_img = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Estonia/S2_LS1Est_2015_merge.tif'

# paths for rasters
fp_dir = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Norway/tiles'
fp_pcadir = os.path.join(os.path.dirname(fp_dir),'pca')
fp_bathydir = os.path.join(os.path.dirname(fp_dir), 'bathymetry')

# model filepath
fp_model = os.path.join(os.path.dirname(fp_dir), 'model', 'RF_NO.sav')

#sansibar
fp_tile = '/mnt/d/users/e1008409/MK/sansibar/Sansibar_WV2_25_9_2016/tiles/01_.tif'
fp_pca = '/mnt/d/users/e1008409/MK/sansibar/Sansibar_WV2_25_9_2016/tiles/pca/16SEP25073556-M2AS-056096543010_pca_tile_.tif'


# map image tiles and respective tiles for aux layers
tiles = [os.path.basename(file).split('.')[0][-3:] for file in glob.glob(os.path.join(fp_dir, '*C*.tif'))]

fp_tile = fp_img
fp_pca = os.path.join(fp_pcadir, 'pca.tif')
fp_bathy = os.path.join(fp_bathydir, 'depth_mean_bilinear_resample_10m_3035.tif')

# TODO wrap code below to functions and parallel computing
for tileid in tiles:
    # find files with suffix
    fp_tile = glob.glob(os.path.join(fp_dir, '*C*' + tileid + '.tif'))[0]
    fp_pca = glob.glob(os.path.join(fp_pcadir, '*C*' + tileid + '.tif'))[0]
    fp_bathy = glob.glob(os.path.join(fp_bathydir, '*' + tileid + '.tif'))[0]
    
    with rio.open(fp_tile) as src:
        img = src.read((1,2,3,4,5,8))
        meta = src.meta
    # pca
    with rio.open(fp_pca) as src:
        pca = src.read((1,2,3))
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
    bathy = bathy.transpose(1,2,0)
    
    img_stack = np.dstack((img, pca))#, bathy)) 
    img_stack = img_stack.transpose(2,0,1)
    img = img.transpose(2,0,1)
    pca = pca.transpose(2,0,1)
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



