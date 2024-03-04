# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 10:51:40 2024

Compare classifiers

@author: E1008409
"""



import sys
import os
os.getcwd()
os.chdir('/mnt/c/users/e1008409/.spyder-py3')
import time
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, StratifiedKFold, KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import normalize
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import accuracy_score

from xgboost import XGBClassifier
import lightgbm as lgb

# filepaths
fp = '/mnt/d/users/e1008409/MK/S2/ac/S2A_L2W_20180715_T34VENVEP_merge/S2A_MSI_2018_07_15_10_03_43_T34VENVEP_L2W_RrsB1B2B3B4B5B6B7B8B8A_south_median3x3_filt_stack.tif'
fp_gt = '/mnt/d/users/e1008409/MK/S2/ac/S2A_L2W_20180715_T34VENVEP_merge/segment/n5s2_gt.tif'

# read data
with rio.open(fp) as src:
    stack = src.read()
    meta = src.meta
with rio.open(fp_gt) as src:
    gt = src.read()
    gtmeta = src.meta
    
# nodata mask - read masked array instead?
if meta['nodata'] == None:
    print('Nodata not defined')
elif np.isnan(meta['nodata']):
    nodatamask = np.where(np.isnan(stack[0]), True, False)
else:
    nodatamask = np.where(stack[0] == meta['nodata'], True, False)
    
# reshape
stack_re = stack.reshape(-1, meta['count'])
# normalize
stack_re_n = normalize(stack_re, axis=1)

# train test data
X = stack_re_n[gt > 0]
y = gt[gt > 0]
# train test split    
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=42, shuffle=True,
                                                    stratify=y)


# random forest classifier
rf = RandomForestClassifier(n_estimators=200, max_depth=None, n_jobs=5 , max_features='sqrt',
                            bootstrap=True, oob_score=True,
                            random_state=42)

# fit rf
rf.fit(X_train, y_train)
print(rf.score(X_test, y_test))

# predict test
predf = pd.DataFrame()
#predf['truth'] = y_test
predf['predict'] = rf.predict(X_test)
# classification report
print(metrics.classification_report(y_test, predf.predict))



# xgboost
new_y = y_train - 1 # TODO: one hot encoding
new_y_test = y_test -1
bst = XGBClassifier(n_estimators=150, max_depth=10, learning_rate=0.1)
eval_set = [(X_test, new_y_test)]
bst.fit(X_train, new_y, eval_set=eval_set)
y_pred = bst.predict(X_test)
acc = accuracy_score(new_y_test, y_pred)
print(acc)

#lightgbm
# set params
params = {'num_leaves': 50, 'objective': 'multiclass', 'num_class': 7}
# train
clf = lgb.LGBMClassifier()
clf.fit(X_train, y_train, eval_set=[(X_test, y_test), (X_train, y_train)])
print('Training accuracy {:.4f}'.format(clf.score(X_train,y_train)))
print('Testing accuracy {:.4f}'.format(clf.score(X_test,y_test)))

predf = pd.DataFrame()
#predf['truth'] = y_test
predf['predict'] = clf.predict(X_test)
# classification report
print(metrics.classification_report(y_test, predf.predict))
# plot model
lgb.plot_metric(clf)
# confusion matrix
cm = metrics.confusion_matrix(y_test, predf.predict)
disp = metrics.ConfusionMatrixDisplay(cm, display_labels=clf.classes_)
disp.plot()

#######################################
# compare classifiers



#######################################
# predict for the full image
stack_re = stack.reshape((stack.shape[0],-1)).transpose((1,0))
stack_re = np.where(np.isnan(stack_re), 0, stack_re) # replace nans

preds = [] # list for split array predictions
# find largest number within range for modulo 0
modulos = []
for i in np.arange(32,1024,2):
    if len(stack_re) % i == 0:
        modulos.append(i)
patch_size = np.max(modulos)        

# normalize
stack_re_n = normalize(stack_re, axis=1)
# split for prediction
split_array = np.split(stack_re_n, patch_size, axis=0)
j = 1
for i in split_array: # NOTE: parallelize
    #prediction = rf.predict(i)
    #prediction = bst.predict(i)
    prediction = clf.predict(i)
    preds.append(prediction)
    print(str(j),'/',str(len(split_array)))
    j += 1

# patch predictions back to single array
predicted = np.stack(preds)
predicted = predicted.reshape(stack_re.shape[0]) 
# prediction back to 2D array
predicted = predicted.reshape(1, meta['height'], meta['width'])

#predicted = predicted + 1

# mask nodata
predicted = np.where(nodatamask == True, 0, predicted)

# outfile
outdir = os.path.join(os.path.dirname(fp_img), 'classification')
if os.path.isdir(outdir) == False:
    os.mkdir(outdir)
outfile = os.path.join(outdir, os.path.basename(fp_img).split('.')[0] + '_LGBM_obia.tif')
# update metadata
upmeta = meta.copy()
upmeta.update(dtype='uint8',
              nodata=0,
              count=1)

with rio.open(outfile, 'w', **upmeta, compress='LZW') as dst:
    dst.write(predicted.astype(rio.uint8))













