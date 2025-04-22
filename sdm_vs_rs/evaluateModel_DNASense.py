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
from sklearn.preprocessing import normalize, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score, StratifiedKFold, StratifiedGroupKFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, cohen_kappa_score
from dask import delayed
from dask import compute
import dask.bag as db
import gc

########################################################
# filepath
fp = sys.argv[1]
fp_pts = sys.argv[2]

########################################################
fp = '/mnt/d/users/e1008409/MK/DNASense/FIN/2023'
fp_pts = os.path.join(fp, 'Finland_habitat_data_ml_2016-2023_v3_encoded.gpkg')
# model dir
modeldir = os.path.join(fp, 'model_')
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
for file in glob.glob(os.path.join(fp, 'segmentation', 'segvals', '*segvals.csv')):
    print(file)
    df1 = pd.read_csv(file, sep=';')
    df = pd.concat([df, df1])
del(df1)

# set train cols
colset = ['Band1', 'Band2', 'Band3', 'Band4', 'Band5', 'Band8'] + pcacols #+ ['bathymetry']
traincols = df.columns[1:-3].intersection(colset) #get the same columns as on list

# standardize data
#scaler = StandardScaler().fit(df[traincols])
# save scaler
#scaler_out = os.path.join(modeldir, 'scaler.sav')
#pickle.dump(scaler, open(scaler_out, 'wb'))
#X = scaler.transform(df[traincols])
le = LabelEncoder() 
le.fit(np.unique(df.int_class)) # fit classes
#y = le.transform(y) # transform
print(np.unique(df.int_class, return_counts=True)) # check result

########################################################
# define models #

models = {'RF': {'model': RandomForestClassifier(n_jobs=6),
                 'params': {"n_estimators": [50, 150, 200, 500], 'class_weight':['balanced'], "max_depth": [3,6],
                            "max_features": ['sqrt', 'log2']}},
          'SVM': {'model': SVC(probability=True, class_weight='balanced'),
                  'params': {"kernel": ['rbf',], "C": [100, 10, 1.0, 0.1, 0.01], "gamma": [100, 10, 1.0, 0.1, 0.01]}},
          'XGB': {'model': XGBClassifier(eval_metric='mlogloss', verbosity=0),
                  'params': {'objective': ['multi:softprob'], 'device':['cuda'], 
                             'learning_rate':[0.001,0.01,0.1,1],
                             'n_estimators': [50, 150, 200, 500], 'max_depth': [3,6], 
                             'subsample': [0.8, 0.5], 'max_delta_step':[0,1], 
                             'min_child_weight': [0, 0.5, 1],
                             'num_class': [len(np.unique(df.int_class))]}},
          }
# data split for optimization
X_train_pts, X_test_pts, y_train, y_test = train_test_split(gdf, gdf.int_class, 
                                                            test_size=0.1, random_state=42,
                                                            stratify=gdf.int_class)
X_train_pts, X_val_pts, y_train, y_val = train_test_split(X_train_pts, y_train, 
                                                            test_size=0.225, random_state=42,
                                                            stratify=y_train)
print('Train set proportion', len(X_train_pts)/len(gdf))
print('Test set proportion', len(X_test_pts)/len(gdf))
print('Validation set proportion', len(X_val_pts)/len(gdf))

X_train = df[traincols][df.point_id.isin(X_train_pts.point_id)]
y_train = le.transform(df['int_class'][df.point_id.isin(X_train_pts.point_id)])
X_val = df[traincols][df.point_id.isin(X_val_pts.point_id)]
y_val = le.transform(df['int_class'][df.point_id.isin(X_val_pts.point_id)])
groups = np.array(df['point_id'][df.point_id.isin(X_train_pts.point_id)])
groups_val = np.array(df['point_id'][df.point_id.isin(X_val_pts.point_id)])

# Stratified KFold for hyperparameter tuning
sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True)
for i, (train_index, val_index) in enumerate(sgkf.split(X_train, y_train, groups)):
    print(f"Fold {i}:")
    print(f"  Train: index={train_index}")
    print(f"         group={groups[train_index]}")
    print(f"  Test:  index={val_index}")
    print(f"         group={groups[val_index]}")
    print('Classes in train set', np.unique(y_train[train_index] ,return_counts=True))
    print('Classes in test set', np.unique(y_train[val_index] ,return_counts=True))
# make small test sample for testing
    
# hyperparameter optimization
for m in models:
    #scv = GridSearchCV(models[m]['model'], param_grid=models[m]['params'], scoring='accuracy', cv=sgkf, return_train_score=True, n_jobs=-1)
    rcv = RandomizedSearchCV(models[m]['model'], param_distributions=models[m]['params'], scoring='accuracy', cv=sgkf, return_train_score=True, n_jobs=-1)
    
    #TODO 
    # make pipeline 
    pipeline = Pipeline([('scaler', StandardScaler()),
                         (m, models[m]['model'])])
    pparams = pipeline.get_params()
    param_dict = dict()
    for p in models[m]['params']:
        for pp in pparams:
            if p in pp:
                param_dict[pp] = models[m]['params'].get(p)
    # # set estimator params
    search = RandomizedSearchCV(pipeline, param_distributions=param_dict, scoring='accuracy', cv=sgkf, return_train_score=True, n_jobs=-1)
    result = search.fit(X_train, y_train, groups=groups)
    
#    X_tr = np.vstack([X_train, X_val])
#    scaler = StandardScaler().fit(X_tr)
#    X_tr = scaler.transform(X_tr)
#    y_tr = np.concatenate([y_train, y_val])
#    groups_tr = np.concatenate([groups, groups_val])
#    result = rcv.fit(X_tr, y_tr, groups=groups_tr)
#    result = scv.fit(Xtr_opt, ytr_opt, groups=groups)
    # summarize result
    print('Scores: %s' % result.scoring)
    print(m, 'Best Score: %s' % result.best_score_)
    print(m, 'Best Hyperparameters: %s' % result.best_params_)
    # save best params
    best_params = dict()
    for k in result.best_params_:
        print(k)
        key = k.split('__')[1] # drop underscores from pipeline parameter names
        best_params[key] = result.best_params_.get(k) # set value
    models[m]['best_params'] = best_params
    # save model best params
    param_dict = models[m]['best_params']
    param_dict_out = os.path.join(modeldir, m + '_best_params.json')
    with open(param_dict_out, 'w') as f:
        json.dump(param_dict, f, indent=4)

    # scores 
    test_score = result.cv_results_['mean_test_score']
    train_score = result.cv_results_['mean_train_score']
    
    fig, ax = plt.subplots()
    ax.plot(train_score, color='blue', label='train')
    ax.plot(test_score, color='orange', label='test')
    ax.legend()
    plt.suptitle(m + ' Cross validation scores')
    plt.tight_layout()
    plot_out = os.path.join(modeldir, prefix + '_' + m + '_CV_scores.png')
    plt.savefig(plot_out, dpi=300, format='PNG')

    if m == 'XGB':
        # add early stopping to params
        param_dict['early_stopping_rounds'] = 20
        # re-train with optimized params and early stopping (see 'Early Stopping' in: https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html )
        models[m]['model'] = XGBClassifier(**param_dict)
        models[m]['model'].fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)])
    
        #retrieve performance metrics
        results = models[m]['model'].evals_result()
        epochs = len(results['validation_0']['mlogloss'])
        x_axis = range(0, epochs)
        # plot log loss
        fig, ax = plt.subplots()
        ax.plot(x_axis, results['validation_0']['mlogloss'], label='Train')
        ax.plot(x_axis, results['validation_1']['mlogloss'], label='Test')
        ax.legend()
        ax.set_ylabel('MLogLoss')
        plt.suptitle(m + ' multi logloss with optimized params and early stopping')
        plot_out = os.path.join(modeldir, m + '_performance.png')
        plt.savefig(plot_out, dpi=150, format='png')
        #plt.show()         
    
    # test set    
    scaler = StandardScaler().fit(df[traincols][df.point_id.isin(X_test_pts.point_id)])
    X_test = scaler.transform(df[traincols][df.point_id.isin(X_test_pts.point_id)])
    y_test = le.transform(df['int_class'][df.point_id.isin(X_test_pts.point_id)])
        
    # dataframe for results
    predf = pd.DataFrame()
    predf['truth'] = y_test
    
    # set model
    clf = models[m]['model'].set_params(**models[m]['best_params']) # set best model parameters from CV search
    X_tr = np.vstack([X_train, X_val])
    y_tr = np.concatenate([y_train, y_val])
    if m != 'XGB':
        clf.fit(X_tr, y_tr)
    X_train = scaler.transform(X_train)
    clf.fit(X_train, y_train)
    # predict on test set
    predf['predict'] = clf.predict(X_test)
    # create confusion matrix
    cm = metrics.confusion_matrix(predf['truth'], predf['predict'])
    # compute row and col sums
    total = cm.sum(axis=0)
    rowtotal = cm.sum(axis=1)
    rowtotal = np.expand_dims(rowtotal, axis=0).T #expand dims and transpose
    rowtotal_sum = np.array(rowtotal.sum()) 
    rowtotal = np.vstack([rowtotal, rowtotal_sum]) # stack row sum
    # create cm DataFrame
    cmdf = np.vstack([cm,total]) # vertical stack
    cmdf = np.hstack((cmdf, rowtotal)) # horizontal stack
    cm_cols = sorted(X_test_pts.hab_class_ml.unique().tolist())
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
    kappa = cohen_kappa_score(predf.truth, predf.predict)
    print('Cohens Kappa %.2f' % (kappa))
    # plot
    #import seaborn as sns
    #sns.set_theme(style='white')
    #fig, ax = plt.subplots()
    #ax = sns.heatmap(cmdf, annot=True, cmap='Blues', fmt='.0f', cbar=False)
    #ax.xaxis.set_ticks_position('top')
    #ax.tick_params(axis='both', which='both', length=0)
    #fig.suptitle('Confusion matrix of test set classifications')
    #plt.tight_layout()
    #plt.savefig(os.path.join(modeldir, m + '_cm.png'), dpi=150, format='PNG')
    
    # fit all data to model and save
    X = df[traincols]
    X = scaler.fit_transform(X)
    y = le.fit_transform(df['int_class'])
    if m == 'XGB':
        # define output model parameters without early stopping
        param_dict['early_stopping_rounds'] = None    
        clf = models[m]['model'].set_params(**param_dict)
    clf.fit(X, y) # fit all data before saving
    model_out = os.path.join(modeldir, m + '.sav')
    pickle.dump(clf, open(model_out, 'wb'))

    
# permutation importance
from sklearn.inspection import permutation_importance
# initiate plot
fig, ax = plt.subplots(1,3)
for m in models:
    model = models[m]['model'].fit(X_train, y_train)
    perm_result = permutation_importance(
        model, X_train, y_train, n_repeats=10, scoring='accuracy')
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
    ax_i = list(models.keys()).index(m)
    ax[ax_i].boxplot(importances, vert=False)
    ax[ax_i].set_yticklabels([])
    ax[0].set_yticklabels(importances.columns)
    ax[ax_i].axvline(0, ls='--', color='black', alpha=0.6)
    ax[ax_i].set_title(m)
fig.suptitle('Permutation importance')
fig.supxlabel('Decrease in accuracy score')
plt.tight_layout()
plot_out = os.path.join(modeldir, 'model' + '_perm_importance.png')
plt.savefig(plot_out, dpi=300, format='PNG')


