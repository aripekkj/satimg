# -*- coding: utf-8 -*-
"""
Created on Mon Aug 12 08:39:25 2024

@author: E1008409
"""

import sys
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
from sklearnex import patch_sklearn 
patch_sklearn()
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
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
# filepath
fp = sys.argv[1]
fp_pts = sys.argv[2]

########################################################
# model dir
modeldir = os.path.join(os.path.dirname(os.path.dirname(fp)), 'model')
if os.path.isdir(modeldir) == False:
    os.mkdir(modeldir)
prefix = os.path.basename(fp_pts).split('_')[0]
# read data
gdf = gpd.read_file(fp_pts, engine='pyogrio')

# get pca variance
fp_pca = os.path.join(fp, 'pca', '*pca_var.csv')
pcafile = [f for f in glob.glob(fp_pca)][0]
pcavar = pd.read_csv(pcafile, sep=',')
# compute difference between rows
pcavar['diff'] = pcavar['PCA_var'].diff()
# get threshold where explained variance increases < 1
threshold = pcavar[pcavar['diff'] < 1].index[0]
# list of pca cols to select
pcacols = ['pca' + str(n) for n in np.arange(1, threshold+1)]
print('Selected PCs:', str(threshold))

# read and merge dataframes
df = pd.DataFrame()
for file in glob.glob(os.path.join(fp, '*segvals.csv')):
    print(file)
    df1 = pd.read_csv(file, sep=';')
    df = pd.concat([df, df1])
del(df1)
# set train cols
# set train cols
colset = ['Band1', 'Band2', 'Band3', 'Band4', 'Band5', 'Band8'] + pcacols #+ ['bathymetry']
traincols = df.columns[1:-3].intersection(colset) #get the same columns as on list

# standardize data
scaler = StandardScaler().fit(df[traincols])
le = LabelEncoder() 
le.fit(np.unique(df.int_class)) # fit classes
#y = le.transform(y) # transform
print(np.unique(df.int_class, return_counts=True)) # check result

########################################################
# define models #

models = {'RF': {'model': RandomForestClassifier(n_jobs=6, class_weight='balanced'),
                 'params': {"n_estimators": [50, 150, 200, 500], "max_depth": [3,6], "max_features": ['sqrt', 'log2'],
                            "min_samples_leaf":[1,2,4], "min_samples_split":[2,5,10],
                           "bootstrap":[True,False]}},
          'SVM': {'model': SVC(probability=True, class_weight='balanced'),
                  'params': {"kernel": ['rbf',], "C": [100, 10, 1.0, 0.1, 0.01], "gamma": [100, 10, 1.0, 0.1, 0.01]}},
          'XGB': {'model': XGBClassifier(eval_metric='mlogloss', verbosity=0, device='cuda'),
                  'params': {'objective': ['multi:softprob'], 'learning_rate':[0.0001, 0.001, 0.01, 0.1, 0.3],
                             'n_estimators': [50, 150, 200, 500], 'max_depth': [3,6],
                             'subsample': [0.8, 0.5], 'num_class': [len(np.unique(df.int_class))]}
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
    gcv = GridSearchCV(models[m]['model'], param_grid=models[m]['params'], scoring='accuracy', cv=sgkf, return_train_score=True, n_jobs=6)
    result = gcv.fit(Xtr_opt, ytr_opt, groups=groups)
    # summarize result
    print('Scores: %s' % result.scoring)
    print(m, 'Best Score: %s' % result.best_score_)
    print(m, 'Best Hyperparameters: %s' % result.best_params_)
    # save best params
    models[m]['best_params'] = result.best_params_
    
    # scores 
    test_score = result.cv_results_['mean_test_score']
    train_score = result.cv_results_['mean_train_score']
    
    fig, ax = plt.subplots()
    ax.plot(train_score, color='blue', label='train')
    ax.plot(test_score, color='orange', label='test')
    ax.legend()
    plt.tight_layout()
    plot_out = os.path.join(modeldir, prefix + '_' + m + '_learningcurve.png')
    plt.savefig(plot_out, dpi=300, format='PNG')

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
    cm_cols = gdf.hab_class_ml.unique().tolist()
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
    for k in sorted(gdf.hab_class_ml.unique()):
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
        
        gdf_test_arr = scaler.transform(gdf_test[traincols]) # standardize
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

gdf_out = os.path.join(os.path.dirname(fp_pts), prefix + '_preds.gpkg')
gdf.to_file(gdf_out, engine='pyogrio')

# save models
X = scaler.transform(df[traincols]) # transform
y = le.transform(df.int_class) # transform
for m in models:
    clf = models[m]['model']
    # fit all data
    clf.fit(X, y)
    clf_out = os.path.join(modeldir, m + '_' + prefix + '.sav') # filename
    pickle.dump(clf, open(clf_out, 'wb'))

# ----------------------------------------------------------------------- #
