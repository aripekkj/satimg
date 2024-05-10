# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 07:58:59 2024

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

# file
#fp = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Denmark/Eelgrass_2018-2023/Eelgrass_Kattegat_Zostera_2018_S2_sampled_3035_S2cell.gpkg'
fp = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Denmark/LS2abc_manualedit_segment_stats.gpkg'
# read
gdf = gpd.read_file(fp, engine='pyogrio')
#gdf = pd.read_csv(fp, sep=';')

# select columns for train, test
cols = ['Band1','Band2','Band3','Band4','Band5','Band6','Band7','Band8','Band9','Band10', 'gr', 'gb', 'eg']#, 'Depth_M']
#cols = ['Band2','Band3','Band4','Band5','Band8','gr', 'gb', 'eg']#, 'Depth_M']


# drop invalid values
gdf.replace([np.inf, -np.inf], np.nan, inplace=True)
gdf = gdf.dropna(subset=cols)

#gdf = gdf[gdf.Depth_M < 4]

#np.unique(gdf.new_class, return_counts=True)
#gdf = gdf[gdf.ObservationsstedId != 127936]
#gdf = gdf[gdf.ObservationsstedId != 127932]


#%%
# -------------------------------------------------------------------------- #
# group by class and compute average spectra
# select cols
df_s = gdf[['new_class', 'Band1', 'Band2', 'Band3', 'Band4', 'Band5', 'Band6', 'Band7', 'Band8', 'Band9', 'Band10']]
# groupby class
dg = df_s.groupby('new_class').mean()
# select band columns
#dg = dg[cols]

# spectral plot by depth
dzones = [] 
n = 0
while n < 4:
    dzone = (n, n+1)
    dzones.append(dzone)
    n += 1

bands = ['Band2', 'Band3', 'Band4', 'Band5', 'Band6', 'Band7', 'Band8']
wavels = [492.4, 559.8, 664.6, 704.1, 740.5, 782.8, 832.8]#, 864.7]
# deep water class
dw = df_s[df_s['new_class'] == 4].groupby('new_class').mean()
dw = dw[bands]
pr = 2
pc = 2
# create spectral plot of each depth zone
fig, ax = plt.subplots(pr,pc, figsize=(12,8))

for i in dzones:
    # make selection
    dfsel = gdf[(gdf.Depth_M > i[0]) & (gdf.Depth_M <= i[1])]
    # drop class 4
    #dfsel = dfsel[dfsel.classes < 4]
    # group
    dfsel_group = dfsel.groupby('new_class').mean(numeric_only=True)    
    # drop 0
    if 0 in dfsel_group.index:
        dfsel_group = dfsel_group.drop(0)
    # select band columns
    dfsel_group = dfsel_group[bands]
    # reset index
    #dfsel_group = dfsel_group.reset_index()
    
    # row, col params for multiplot
    if i[0] < 2:
        r = 0
        c = i[0]
    elif i[0] >= 2:
        r = 1
        c = i[0] - 2
    
    cl_in_dzone = list(np.unique(dfsel_group.index))
    print(cl_in_dzone)
    colors = ['green', '#888f29', 'yellow', 'blue']
    labs = ['Seagrass', 'Sparse', 'Bare', 'Deep water']
    
    for j in cl_in_dzone:
        # get index
        base = dfsel_group.index.get_loc(j)
        # plot
        ax[r,c].plot(dfsel_group.loc[j], linewidth=0.7, color=colors[base], label=labs[base])
    #    ax[r,c].plot(dfsel_group.loc[2], linewidth=0.7, color=, label=)
    #    ax[r,c].plot(dfsel_group.loc[3], linewidth=0.7, color=, label=)
    ax[r,c].plot(dw.loc[4], linewidth=0.7, linestyle='--', color='#0034CA', label='Deep water') 
    ax[r,c].set_title(str(i[0]) + '-' + str(i[1]) + ' meters', fontsize=10) 
    ax[r,c].grid(True, color='#B0B0B0')
    ax[r,c].set_xticks(list(np.arange(len(bands))))     
    ax[r,c].set_xticklabels(wavels, rotation=0)
    ax[r,c].tick_params(axis='x', which='major', pad=-1, labelsize=8)
    ax[r,c].tick_params(axis='y', which='major', pad=-1, labelsize=8)
    ax[r,c].set_facecolor(color='#D9D9D9')
"""    
# select deep water
dfsel_deep = df[df.field_depth > 5].groupby('new_class').mean(numeric_only=True)
dfsel_deep = dfsel_deep[cols]

# add to plot
ax[2,1].plot(dfsel_deep.loc[1], linewidth=0.7, color='#443726', label='Bare bottom or \nsparse SAV')
ax[2,1].plot(dfsel_deep.loc[2], linewidth=0.7, color='green', label='Mixed SAV')
ax[2,1].plot(dfsel_deep.loc[3], linewidth=0.7, color='#A27C1F', label='Dense Zostera')
ax[2,1].plot(dw.loc[4], linewidth=0.7, linestyle='--', color='#0034CA', label='Deep water')    
ax[2,1].set_title('> 5 meters', fontsize=10)
ax[2,1].grid(True, color='#B0B0B0')
ax[2,1].set_xticks(list(np.arange(len(bands))))     
ax[2,1].set_xticklabels(bands)
ax[2,1].tick_params(axis='x', which='major', pad=-1, labelsize=8)
ax[2,1].tick_params(axis='y', which='major', pad=-1, labelsize=8)

ax[2,1].set_facecolor(color='#D9D9D9')
"""
# set labels
#plt.setp(ax[-1, :], xlabel='Wavelenth (nm)')
#plt.setp(ax[1, 0], ylabel='Remote sensing reflectance $(sr^{-1})$')
fig.supxlabel('Wavelenth (nm)')
fig.supylabel('Remote sensing reflectance $(sr^{-1})$')
plt.suptitle('Average spectra at depths (m)')

# legend
handles, labels = ax[0,1].get_legend_handles_labels()
#lgd = fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.12, 0.93))
legend = ax[0,1].legend(handles, labels, ncol=1, frameon=True,
                    loc='upper right', bbox_to_anchor=(1.35, 1.03))
plt.tight_layout()
#%%
# ------------------------------------------------------------------------------

# X, y
X = np.array(gdf[cols])
y = np.array(gdf['new_class'])
# change class
y = np.where(y == 2, 1, y)
# encode classes to 0,1,2,...
le = LabelEncoder() 
le.fit(np.unique(y)) # fit classes
y_train_le = le.transform(y) # transform
y_le = le.transform(y)
print('Original:', np.unique(y), 'Encoded:', np.unique(y_le))
# replace negative values
X = np.where(X < 0, 0, X)
# =============================================================================
# # scale
# qt = QuantileTransformer()
# X_pt = qt.fit_transform(X)
# # visualize data
# fig, ax = plt.subplots(1,2)
# for c in np.arange(0,X.shape[1],1):
#     ax[0].hist(X[:,c])
#     ax[1].hist(X_pt[:,c])
# plt.show()
# =============================================================================


# make stratified KFolds
folds = dict()
skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
for i, (train, test) in enumerate(skf.split(X, y_le)):
    # save train, test indices to dictionary
    k = 'fold_' + str(i+1)
    folds[k] = (train.tolist(), test.tolist())

# split
X_train, X_test, y_train, y_test = train_test_split(X, y_le,
                                                    test_size=0.3,
                                                    random_state=42, shuffle=True,
                                                    stratify=y_le)
# define models
rf = RandomForestClassifier(n_estimators=150, max_depth=None, n_jobs=5, max_features='sqrt',
                            bootstrap=True, oob_score=True,
                            random_state=42)
params = {'n_estimators': 300,
          'max_depth': 10,
          'learning_rate': 0.1,
          'objective': 'multi:softprob',
          'booster': 'gbtree',
          'verbosity':1,
          'eval_metric': ['merror', 'mlogloss'],
          #'early_stopping_rounds': 20,
          'num_class': len(np.unique(y_le))}
bst = XGBClassifier(**params)

gdf['proba_RF'] = None
gdf['proba_XGB'] = None
gdf['test_fold'] = None
# evaluate
#for i, (train, test) in enumerate(skf.split(X, y_le)):
for f in folds:
    
    # select fold train, test from gdf
#    X_train_f = X[X.index.isin(folds[f].tolist())]
#    y_train_f = y[y.index.isin(folds[f].tolist())]
#    X_test_f = X[~X.index.isin(folds[f].tolist())]
#    y_test_f = y[~y.index.isin(folds[f].tolist())]
    #results = cross_val_score(rf, X, y, cv=skf)
    #print("RF Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))
    # convert to array
#    X_train = np.array(X_train_f)
#    y_train = np.array(y_train_f)
#    X_test = np.array(X_test_f)
#    y_test = np.array(y_test_f)
    # select fold train, test from array
    X_train_f = X[folds[f][0]]
    y_train_f = y_le[folds[f][0]]
    X_test_f = X[folds[f][1]]
    y_test_f = y_le[folds[f][1]]
    
    # fit rf
    rf.fit(X_train_f, y_train_f)
    print('RF', rf.score(X_test_f, y_test_f))
    predf = pd.DataFrame()
    #predf['truth'] = y_test
    predf['predict'] = rf.predict(X_test_f)
    # classification report
    print('RF', metrics.classification_report(y_test_f, predf.predict))
    # predict proba to gdf
    pred_proba = rf.predict_proba(X_test_f)
    gdf['proba_RF'].iloc[folds[f][1]] = pred_proba[:,0].tolist()

    evalset = [(X_train_f, y_train_f), (X_test_f, y_test_f)]
    bst.fit(X_train_f, y_train_f, eval_set=evalset, verbose=False)
    y_pred = bst.predict(X_test_f)
    acc = metrics.accuracy_score(y_test_f, y_pred)
    print('XGB', acc)
    predf = pd.DataFrame()
    #predf['truth'] = y_test
    predf['predict'] = bst.predict(X_test_f)
    # classification report
    print('XGB', metrics.classification_report(y_test_f, predf.predict))
    # predict proba to gdf
    pred_proba = bst.predict_proba(X_test_f)
    gdf['proba_XGB'].iloc[folds[f][1]] = pred_proba[:,0].tolist()
    
    # add test fold number to gdf
    gdf['test_fold'].iloc[folds[f][1]] = f

test = (gdf.proba_RF[gdf.new_class==1], gdf.proba_RF[gdf.new_class==3], gdf.proba_RF[gdf.new_class==4])
test2 = (gdf.proba_XGB[gdf.new_class==1], gdf.proba_XGB[gdf.new_class==3], gdf.proba_XGB[gdf.new_class==4])

plotdir = os.path.join(os.path.dirname(fp), 'plots')
if os.path.isdir(plotdir) == False:
    os.mkdir(plotdir)

# plot
fig, ax = plt.subplots()
pos1 = np.arange(len(test))-0.2
pos2 = np.arange(len(test2))+0.2
ax.boxplot(test, positions=pos1, manage_ticks=False)
ax.boxplot(test2, positions=pos2, manage_ticks=False)
ax.set_xticks([0,1,2])
ax.set_xticklabels(['Seagrass', 'Bare', 'Deep water'])
plt.suptitle('Seagrass probability RF and XGB between classes (m)')
plt.show()
plt.tight_layout()
plt.savefig(os.path.join(plotdir, 'Seagrass_proba_boxplot.png'), dpi=150)

# save gdf
outpath = os.path.dirname(fp)
outfile = os.path.join(outpath, os.path.basename(fp).split('.gpkg')[0] + '_preds.gpkg')
gdf.to_file(outfile, engine='pyogrio')
# save folds as json
json_out = os.path.join(outpath, 'LS2_train_test_folds.json')
with open(json_out, 'w') as of:
    json.dump(folds, of, indent=4)
#TODO validate model and save model
# train model with all data
rf.fit(X, y_le)
bst.fit(X, y_le)
#bst.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)])
# save model
modeldir = os.path.join(os.path.dirname(fp), 'model')
if os.path.isdir(modeldir) == False:
    os.mkdir(modeldir)
rf_out = os.path.join(modeldir, 'LS2_RF.sav')
bst_out = os.path.join(modeldir, 'LS2_XGB.sav')

pickle.dump(rf, open(rf_out, 'wb'))
pickle.dump(bst, open(bst_out, 'wb'))

# ----------------------------------------------------------------------- #
# predict for segments
#fp_segstat = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Denmark/segmentation/segstats/LS2c_n10_s300_iter_segstats.csv'
#segdir = os.path.dirname(os.path.dirname(fp_segstat))
segstats_dir = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Denmark/segmentation/segstats/'
model_fp = '/scratch/project_2000393/remote_sensing/S2/sdm_rs/model/LS2_XGB.sav'
# map files
segfiles = [file for file in glob.glob(os.path.join(segstats_dir, '*.csv'))]


def predict(array, model):
    pred = model.predict(array)[0]
    return pred
def predictProba(array, model):
    pred = model.predict_proba(array)[0]
    return pred


for sf in segfiles[23:24]:
    # read 
    df = pd.read_csv(sf, sep=';')
    # drop invalid values
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna(subset=cols)
    #TODO read model
    model = pickle.load(open(model_fp, 'rb'))
    # predict class for each segment
    #preds = [model.predict(np.array(row[1][cols]).reshape(1,-1))[0] for row in df.iterrows()] # convert df row as array and predict
    preds_delayed = []
    for row in df.iterrows():
        preds_delayed.append(delayed(predict)((np.array(row[1][cols]).reshape(1,-1)), rf))
    start = time.time()
    preds = compute(preds_delayed)
    end_time = time.time()
    print('Processed in %.3f minutes' % ((end_time - start)/60))
    # predicted values to df
    df['predicted'] = le.inverse_transform(preds[0]) # convert to original classes
    # map predicted values to array
    # find segmented image corresponding to csv
    segdir = os.path.dirname(os.path.dirname(sf))
    segimg = os.path.join(segdir, df.tilename[0])#os.path.basename(fp_segstat).split('_segstat')[0] + '.tif')
    #TODO turn to function and parallelize below
    # read
    xds = rxr.open_rasterio(segimg)
    prediction = np.zeros(shape=xds.rio.shape) # empty array to assign classes
    #prediction = np.expand_dims(prediction, axis=0)
    start = time.time()
    for segid in df.segment_id:
        prediction = np.where(xds.values == segid, df.predicted.loc[df.index[df.segment_id == segid]], prediction)
    end = time.time()
    print('Elapsed ', end-start)
    # get metadata
    with rio.open(segimg) as src:
        meta=src.meta
    meta.update(dtype='uint8')    
    # save
    raster_out = os.path.join(os.path.dirname(segdir), 'classification', df.tilename[0].split('.tif')[0] + '_Depth_RF_classification.tif')
    with rio.open(raster_out, 'w', **meta) as dst:
        dst.write(prediction.astype(meta['dtype']))

# ----------------------------------------------------------------------- #
# read full image and predict
# =============================================================================
# fp_img = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Denmark/S2_LS2c_20220812_v1_3035.tif'
# with rio.open(fp_img) as src:
#     img = src.read()
#     meta = src.meta
# # reshape to 1d
# img_re = img.reshape((img.shape[0],-1)).transpose((1,0))
# img_re = np.where(np.isnan(img_re), 0, img_re) # replace nans
# preds = [] # list for split array predictions
# # find largest number within range for modulo 0
# modulos = []
# for i in np.arange(32,1024,1):
#     if len(img_re) % i == 0:
#         modulos.append(i)
# patch_size = np.max(modulos)        
# 
# # split for prediction
# split_array = np.split(img_re, patch_size, axis=0)
# j = 1
# for i in split_array: # NOTE: parallelize
#     prediction = rf.predict(i)
#     #prediction = bst.predict(i)
#     #prediction = clf.predict(i)
#     preds.append(prediction)
#     print(str(j),'/',str(len(split_array)))
#     j += 1
# 
# # patch predictions back to single array
# predicted = np.stack(preds)
# predicted = predicted.reshape(img_re.shape[0]) 
# predicted = le.inverse_transform(predicted) # transform
# # prediction back to 2D array
# predicted = predicted.reshape(1, meta['height'], meta['width'])
# # mask nodata
# nodatamask = np.where(img[2] == meta['nodata'], 0, 1)
# predicted = np.where(img[2] == meta['nodata'], 0, predicted)
# # outfile
# outdir = os.path.join(os.path.dirname(fp_img), 'classification')
# if os.path.isdir(outdir) == False:
#     os.mkdir(outdir)
# outfile = os.path.join(outdir, os.path.basename(fp_img).split('.')[0] + '_RF_pts.tif')
# # update metadata
# upmeta = meta.copy()
# upmeta.update(dtype='uint8',
#               nodata=0,
#               count=1)
# # save
# with rio.open(outfile, 'w', **upmeta, compress='LZW') as dst:
#     dst.write(predicted.astype(rio.uint8))
# # -------------------------------------------------------------- #    
# 
# =============================================================================



