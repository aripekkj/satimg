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
from sklearn.model_selection import train_test_split, RandomizedSearchCV, GridSearchCV
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
fp = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Greece'
fp_pts = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Greece/Greece_habitat_data_ml_filtered.gpkg'
# model dir
modeldir = os.path.join(fp, 'model')
if os.path.isdir(modeldir) == False:
    os.mkdir(modeldir)
prefix = os.path.basename(fp_pts).split('_')[0]
# read data
gdf = gpd.read_file(fp_pts, engine='pyogrio')
# copy depth column with different name for prediction
gdf['bathymetry'] = gdf.depth

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
# set train cols
colset = ['Band1', 'Band2', 'Band3', 'Band4', 'Band5', 'Band8'] + pcacols + ['bathymetry']
traincols = df.columns[1:-3].intersection(colset) #get the same columns as on list

# standardize data
#scaler = StandardScaler().fit(df[traincols])
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
                  'params': {"kernel": ['rbf'], "C": [100, 10, 1.0, 0.1, 0.01], "gamma": [100, 10, 1.0, 0.1, 0.01]}},
          'XGB': {'model': XGBClassifier(eval_metric='mlogloss', verbosity=0, device='cuda'),
                  'params': {'objective': ['multi:softmax'], 'learning_rate':[0.0001, 0.001, 0.01, 0.1, 0.3],
                             'n_estimators': [50, 150, 200, 500], 'max_depth': [3,6],
                             'subsample': [0.8, 0.5], 'num_class': [len(np.unique(df.int_class))]}
                  }
          }



# output columns 
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

# make stratified KFolds for points
folds = dict()
skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
# use field obs points
for i, (train, test) in enumerate(skf.split(gdf.point_id, gdf.int_class)):
    # save train, test point_id's to dictionary
    k = 'fold_' + str(i+1)
    tr_pts = gdf['point_id'].iloc[train].tolist() # get point_id's by index
    te_pts = gdf['point_id'].iloc[test].tolist()
    folds[k] = (tr_pts, te_pts)
    #double check that sets are separate (this step can be removed later)
    for t in train:
    #    print(t)
        if t in set(test):
            print('Value found in test')

# evaluate
for f in folds:
    print('Evaluating:', f)
    # select pixel values by point_id
    df_train = df[df.point_id.isin(folds[f][0])]
    df_test = df[df.point_id.isin(folds[f][1])]
    
    # select columns
    X_train = df_train[traincols].to_numpy()
    y_train = le.transform(df_train['int_class'])
    X_test = df_test[traincols].to_numpy()
    y_test = le.transform(df_test['int_class'])
    # select groups (ie. segments)
    groups = df['point_id'][df.point_id.isin(folds[f][0])].to_numpy()
    print('Classes in train set', np.unique(y_train ,return_counts=True))
    print('Classes in test set', np.unique(y_test ,return_counts=True))
    
    # StratifiedGroupKFold for hyperparameter tuning
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=False)
    for i, (train_index, test_index) in enumerate(sgkf.split(X_train, y_train, groups)):
        print(f"Fold {i}:")
        print(f"  Train: index={train_index}")
        print(f"         group={groups[train_index]}")
        print(f"  Test:  index={test_index}")
        print(f"         group={groups[test_index]}")
        print('Classes in train set', np.unique(y_train[train_index] ,return_counts=True))
        print('Classes in test set', np.unique(y_train[test_index] ,return_counts=True))
    # hyperparameter optimization
    for m in models:
#        if m != 'RF':
#            continue
        # make pipeline 
        pipeline = Pipeline([('scaler', StandardScaler()),
                             ('classifier', models[m]['model'])])
        pparams = pipeline.get_params()
        param_dict = dict()
        for p in models[m]['params']:
            for pp in pparams:
                if p in pp:
                    param_dict[pp] = models[m]['params'].get(p)

        # set estimator params
        search = RandomizedSearchCV(pipeline, param_distributions=param_dict, scoring='balanced_accuracy', cv=sgkf, return_train_score=True, n_iter=10, n_jobs=-1)
        result = search.fit(X_train, y_train, groups=groups)
#        gcv = GridSearchCV(models[m]['model'], param_grid=models[m]['params'], scoring='accuracy', cv=sgkf, return_train_score=True, n_jobs=6)
        
        # summarize result
        print(f, 'Scores: %s' % result.scoring)
        print(m, f, 'Best Score: %s' % result.best_score_)
        print(m, f, 'Best Hyperparameters: %s' % result.best_params_)
        # save best params to dict
        models[m]['best_params_'+f] = result.best_params_
        models[m]['cv_result_'+f] = result.cv_results_
        
        # scores 
        test_score = result.cv_results_['mean_test_score']
        train_score = result.cv_results_['mean_train_score']

        #clf = models[m]['model'].set_params(**models[m]['best_params']) # set model parameters according to GridSearchCV
        clf = result.best_estimator_
        X_train_scaled = StandardScaler().fit_transform(X_train)
        clf.fit(X_train_scaled, y_train) # fit data
        X_test_scaled = StandardScaler().fit_transform(X_test)
        print(m, f, clf.score(X_test_scaled, y_test))
    
        # predictions for test array
        predf = pd.DataFrame()
        predf['truth'] = y_test
        predf['predict_clf'] = clf.predict(X_test_scaled)
        # classification report
        print(m, metrics.classification_report(y_test, predf.predict_clf))
        cm = metrics.confusion_matrix(predf['truth'], predf['predict_clf'])
        # save confusion matrix of each fold 
        models[m].setdefault('test_cm', []).append(cm)
        # predict proba to gdf
        gdf_test = gdf[gdf.point_id.isin(folds[f][1])] # get test points
        gdf_test = gdf_test.dropna(subset=traincols)
        
        gdf_test_arr = StandardScaler().fit_transform(gdf_test[traincols].to_numpy()) # to array
        pred_proba = clf.predict_proba(gdf_test_arr) #predict
    
        # select columns
        probacols = [c for c in proba_cols if m in c]
        # add predictions for all classes
        for n in np.arange(len(probacols)): 
            print(n)
#            gdf[probacols[n]].iloc[gdf_test.index] = pred_proba[:,n]
            gdf.loc[gdf_test.index, probacols[n]] = pred_proba[:,n]
        
        # predicted value to gdf
#        gdf['test_fold'].iloc[gdf_test.index] = f  
        gdf.loc[gdf_test.index, 'test_fold'] = f

# save model dict
models_dict_out = os.path.join(modeldir, 'models_cv_result.npy')
np.save(models_dict_out, models)
# save param dict
param_dict_out = os.path.join(modeldir, 'best_params.json')
with open(param_dict_out, 'w') as f_out:
    json.dump(param_dict, f_out, indent=4)

for m in models:
#    if m != 'RF':
#        continue
    # plot hyperparameter optimization results
    fig, ax = plt.subplots()
    cv_fold_keys = [key for key in models[m].keys() if 'cv_result' in key]
    mean_train = []
    mean_test = []
    for c in cv_fold_keys:
        mean_train.append(models[m][c]['mean_train_score'])
        mean_test.append(models[m][c]['mean_test_score'])
    ax.plot(np.mean(mean_train, axis=1), color='blue', label='train')
    ax.plot(np.mean(mean_test, axis=1), color='orange', label='test')
    ax.set_xlabel('CV fold')
    ax.legend()
    title_str = m + ': Mean balanced accuracy from \n CV and hyperparameter search' 
    plt.suptitle(title_str)
    plt.tight_layout()
    plot_out = os.path.join(modeldir, prefix + '_' + m + '_learningcurve.png')
    plt.savefig(plot_out, dpi=300, format='PNG')

# permutation importance
from sklearn.inspection import permutation_importance
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
    fig, ax = plt.subplots()
    ax = importances.plot.box(vert=False, whis=10)
    ax.axvline(0, ls='--', color='black', alpha=0.6)
    ax.set_title(m + ' permutation importance')
    plt.tight_layout()
    plot_out = os.path.join(modeldir, prefix + '_' + m + '_perm_importance.png')
    plt.savefig(plot_out, dpi=300, format='PNG')

# TODO create cm plots from fold results
# # try model performance on test
# for m in models:
#     predf = pd.DataFrame()
#     predf['truth'] = yte_opt
#     clf = models[m]['model'].set_params(**models[m]['best_params']) # set best model parameters from GridSearchCV
#     clf.fit(Xtr_opt, ytr_opt)
#     predf['predict'] = clf.predict(Xte_opt)
#     # create confusion matrix
#     cm = metrics.confusion_matrix(predf['truth'], predf['predict'])
#     val_cm = metrics.confusion_matrix(yte_opt, clf.predict(Xte_opt))
#     # compute row and col sums
#     total = cm.sum(axis=0)
#     rowtotal = val_cm.sum(axis=1)
#     rowtotal = np.expand_dims(rowtotal, axis=0).T #expand dims and transpose
#     rowtotal_sum = np.array(rowtotal.sum()) 
#     rowtotal = np.vstack([rowtotal, rowtotal_sum]) # stack row sum
#     # create cm DataFrame
#     cmdf = np.vstack([cm,total]) # vertical stack
#     cmdf = np.hstack((cmdf, rowtotal)) # horizontal stack
#     cm_cols = gdf.hab_class_ml.unique().tolist()
#     cm_cols.append('Total')
#     cmdf = pd.DataFrame(cmdf, index=cm_cols,
#                         columns = cm_cols)
#     # save confusion matrix dataframe as csv
#     cmdf_name = m + '_cm.csv'
#     cmdf_out = os.path.join(modeldir, cmdf_name)
#     cmdf.to_csv(cmdf_out, sep=';')
#     # print
#     print(pd.crosstab(predf.truth, predf.predict, margins=True))
#     # compute common accuracy metrics
#     o_accuracy = np.sum(cm.diagonal()) / np.sum(cm.sum(axis=0))
#     p_accuracy = cm.diagonal() / cm.sum(axis=0) # producer's accuracy
#     u_accuracy = cm.diagonal() / cm.sum(axis=1) # user's accuracy
#     print(m + ' Overall accuracy %.2f' % (o_accuracy))
#     print(m + ' Users accuracy', u_accuracy)
#     print(m + ' Producers accuracy', p_accuracy)    
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

# save
gdf[proba_cols] = gdf[proba_cols].astype(float)

gdf_out = os.path.join(os.path.dirname(fp_pts), prefix + '_preds.gpkg')
gdf.to_file(gdf_out, engine='pyogrio')

# save models
X = StandardScaler.fit_transform(df[traincols]) # transform
y = le.transform(df.int_class) # transform
for m in models:
    clf = models[m]['model']
    # fit all data
    clf.fit(X, y)
    clf_out = os.path.join(modeldir, m + '_' + prefix + '.sav') # filename
    pickle.dump(clf, open(clf_out, 'wb'))

# ----------------------------------------------------------------------- #
