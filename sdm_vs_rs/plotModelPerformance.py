# -*- coding: utf-8 -*-
"""
Created on Thu May 15 15:28:10 2025

Plot model training results

@author: E1008409
"""



import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
import seaborn as sns
import argparse

CLI = argparse.ArgumentParser()
CLI.add_argument(
    'directory',
    type=str,
    help='Drectory path')
CLI.add_argument(
    "fp_pts",
    type=str,
    help='Filepath for point data')
CLI.add_argument(
    "cv_result",
    type=str,
    help='Filepath for cv_result .npy')
args = CLI.parse_args()

fp = args.directory
fp_pts = args.fp_pts
fp_npy = args.cv_result

# fp
#fp = '/mnt/d/users/e1008409/MK/DNASense/FIN/2023/model_10fold'
#fp = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Greece'
#fp_pts = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Greece/Greece_habitat_data_ml_filtered.gpkg'
#fp_npy = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Greece/model/models_cv_result.npy'

modeldir = os.path.dirname(fp_npy)
# load
models = np.load(fp_npy, allow_pickle=True).item()

# plot hyperparameter optimization results
ls = ['--', '-.', ':']
fig, ax = plt.subplots()
handles, labels = ax.get_legend_handles_labels()
for m in models:
#    if m != 'RF':
#        continue    
    cv_fold_keys = [key for key in models[m].keys() if 'cv_result' in key]
    mean_train = []
    mean_test = []
    n = list(models.keys()).index(m)
    for c in cv_fold_keys:
        mean_train.append(models[m][c]['mean_train_score'])
        mean_test.append(models[m][c]['mean_test_score'])
    ax.plot(np.mean(mean_train, axis=1), color='blue', label='train', ls=ls[n])
    ax.plot(np.mean(mean_test, axis=1), color='orange', label='test', ls=ls[n])
    line = Line2D([0], [0], linestyle=ls[n], color='black', label=m)
    handles.append(line)
patch1 = mpatches.Patch(color='blue', label='Train')
patch2 = mpatches.Patch(color='orange', label='Test')
handles.extend([patch1, patch2])
ax.set_xticks(np.arange(0,10,1))
ax.set_xticklabels(np.arange(1,11,1))
ax.set_yticks(np.arange(0,1.1,0.1))
yrange = np.arange(0,1.1,0.1).round(1)
ax.set_yticklabels(yrange)
ax.set_xlabel('CV fold')
#handles, labels = ax.get_legend_handles_labels()

#legend = ax.legend(handles, labels, ncol=1, frameon=True,
#                    loc='upper right', bbox_to_anchor=(1.1,1.0))
plt.legend(handles=handles)
title_str = 'Mean balanced accuracy from \n CV hyperparameter search'
plt.suptitle(title_str)
plt.tight_layout()
plot_out = os.path.join(modeldir, 'models_CV_accuracy.png')
plt.savefig(plot_out, dpi=300, format='PNG')

# ---------------------------------------------------- #

# load
models = np.load(fp_npy, allow_pickle=True).item()

prefix = os.path.basename(fp_pts).split('_')[0]
# read data
gdf = gpd.read_file(fp_pts, engine='pyogrio')
# copy depth column with different name for prediction

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
colset = ['Band1', 'Band2', 'Band3', 'Band4', 'Band5', 'Band8'] + pcacols + ['bathymetry']
traincols = df.columns[1:-3].intersection(colset) #get the same columns as on list
# fit label encoder
le = LabelEncoder().fit(df['int_class'])
# set same folds as in model evaluation
folds = dict()
skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)
# use field obs points
for i, (train, test) in enumerate(skf.split(gdf.point_id, gdf.int_class)):
    # save train, test point_id's to dictionary
    k = 'fold_' + str(i+1)
    tr_pts = gdf['point_id'].iloc[train].tolist() # get point_id's by index
    te_pts = gdf['point_id'].iloc[test].tolist()
    folds[k] = (tr_pts, te_pts)

# class names
classes = np.unique(gdf.hab_class_ml)
pred_df = pd.DataFrame()
cms = []
acc_df = pd.DataFrame(index=list(folds.keys()))

# evaluation folds
for f in folds:
    # select pixel values by point_id
    df_train = df[df.point_id.isin(folds[f][0])]
    df_test = df[df.point_id.isin(folds[f][1])]    
    # select columns
    X_train = StandardScaler().fit_transform(df_train[traincols].to_numpy())
    y_train = le.transform(df_train['int_class'])
    X_test = StandardScaler().fit_transform(df_test[traincols].to_numpy())
    y_test = le.transform(df_test['int_class'])
    # try model performance on test
    predf = pd.DataFrame()
    predf['truth'] = y_test
    predf['fold'] = f

    for m in models:
#        if m != 'RF':
#            continue        
        # get tuned hyperparameters
        pparams = models[m]['params'] 
        param_dict = dict()
        for p in pparams:
            for pp in models[m]['best_params_'+f]:
                if p in pp:
                    param_dict[p] = models[m]['best_params_'+f].get(pp)
        clf = models[m]['model'].set_params(**param_dict) # set best model parameters
        clf.fit(X_train, y_train) # fit data
        predf[m + '_predict'] = clf.predict(X_test)
        # create confusion matrix
        cm = metrics.confusion_matrix(predf['truth'], predf[m + '_predict'])
        val_cm = metrics.confusion_matrix(y_test, clf.predict(X_test))
    #    # compute row and col sums
        total = cm.sum(axis=0)
        rowtotal = val_cm.sum(axis=1)
        rowtotal = np.expand_dims(rowtotal, axis=0).T #expand dims and transpose
        rowtotal_sum = np.array(rowtotal.sum()) 
        rowtotal = np.vstack([rowtotal, rowtotal_sum]) # stack row sum
    #    # create cm DataFrame
        cmdf = np.vstack([cm,total]) # vertical stack
        cmdf = np.hstack((cmdf, rowtotal)) # horizontal stack
        cms.append(cmdf)
        cm_cols = gdf.hab_class_ml.unique().tolist()
        cm_cols.append('Total')
        cmdf = pd.DataFrame(cmdf, index=cm_cols,
                            columns = cm_cols)
        
    #     # print
        print(pd.crosstab(predf.truth, predf[m+'_predict'], margins=True))
         # compute common accuracy metrics
        o_accuracy = np.sum(cm.diagonal()) / np.sum(cm.sum(axis=0))
        p_accuracy = cm.diagonal() / cm.sum(axis=0) # producer's accuracy
        u_accuracy = cm.diagonal() / cm.sum(axis=1) # user's accuracy
        
        print(m + ' Overall accuracy %.2f' % (o_accuracy))
        print(m + ' Users accuracy', u_accuracy)
        print(m + ' Producers accuracy', p_accuracy)    
        # store accuracies to df
        fold_oa = m + '_oa'
        pacc_cols = [m  + '_' + c + '_pa' for c in classes]
        uacc_cols = [m  + '_' + c + '_ua' for c in classes]
        
        acc_df.loc[f, fold_oa] = o_accuracy
        acc_df.loc[f, pacc_cols] = p_accuracy
        acc_df.loc[f, uacc_cols] = u_accuracy
        

    # concat result
    pred_df = pd.concat([pred_df, predf])
# redefine index
pred_df.index = np.arange(0,len(pred_df))
cms = np.array(cms)
# dropna
acc_df = acc_df.dropna()
# save dataframe
acc_df_out = os.path.join(modeldir, 'acc_df.csv')
acc_df.to_csv(acc_df_out, sep=';')
# save described dataframe
acc_df_desc = acc_df.describe()
acc_df_desc_out = os.path.join(modeldir, 'acc_df_describe.csv')
acc_df_desc.to_csv(acc_df_desc_out, sep=';')


# read 
acc_df = pd.read_csv(acc_df_out, sep=';')

# plot overall accuracies
fig, ax = plt.subplots()
print('Average overall accuracy %.2f' % (np.mean(acc_df[m+'_oa'])))
ax_i = list(models.keys()).index(m)
# plot
ax.boxplot(acc_df[['RF_oa', 'SVM_oa', 'XGB_oa']], vert=True)
ax.set_title('CV model overall accuracy')
ax.set_xticklabels(['RF', 'SVM', 'XGB'])
plt.tight_layout()
oa_out = os.path.join(modeldir, 'OA_accuracies.png')
plt.savefig(oa_out, dpi=300, format='PNG')

# multiplot for users and producers accuracy for each class
nclass = len(classes)
ncol = np.arange(0,2,1)
nrow = np.arange(0,nclass)
 
hablist = list(np.unique(gdf.hab_class_ml))
fig, ax = plt.subplots(len(nrow), 2, figsize=(12,8), sharex=True, sharey=True)

for hab in hablist:
    cols_to_plot = [h for h in acc_df.columns if hab in h]
    
    # select producer's and user's accuracy columns
    cols_pa = [c for c in cols_to_plot if 'pa' in c]
    cols_ua = [c for c in cols_to_plot if 'ua' in c]
    # Filter data using np.isnan
    plot_pa = acc_df[cols_pa].to_numpy()
    mask = ~np.isnan(plot_pa)
    filtered_pa = [d[ma] for d, ma in zip(plot_pa.T, mask.T)]
    
    plot_ua = acc_df[cols_ua].to_numpy()
    mask = ~np.isnan(plot_ua)
    filtered_ua = [d[ma] for d, ma in zip(plot_ua.T, mask.T)]
    
    nr = hablist.index(hab)
    ax[nr,0].set_title(hab, x=1)
    ax[nr,0].axhline(0.5, ls='--', lw=0.5, alpha=0.4, color='black')    
    # plot
    ax[nr,0].boxplot(filtered_pa, vert=True) # acc_df[cols_pa]    
    ax[nr,1].boxplot(filtered_ua, vert=True)
    ax[nr,1].axhline(0.5, ls='--', lw=0.5, alpha=0.4, color='black')
    
    if nr == len(nrow)-1:           
        ax[nr,0].set_xticks([1,2,3], labels=list(models.keys()))
        ax[nr,0].set_xlabel('Producers accuracy')
        ax[nr,1].set_xlabel('Users accuracy')

plt.tight_layout()
fig.suptitle('CV model accuracies on test set', y=1.05)
plot_ua_pa_out = os.path.join(modeldir, 'P_U_accuracies.png')
plt.savefig(plot_ua_pa_out, dpi=300, format='PNG')



    # row, col params for multiplot
#    if i[0] < pr:
#        r = 0
#        c = i[0] -1
#    elif i[0] >= pc:
#        r = 1
#        c = i[0] - pc -1


# # save confusion matrix dataframe as csv
# cmdf_name = m + '_cm.csv'
# cmdf_out = os.path.join(modeldir, cmdf_name)
# cmdf.to_csv(cmdf_out, sep=';')

# # plot 
# sns.set_theme(style='white')
# fig, ax = plt.subplots()
# ax = sns.heatmap(np.mean(cms, axis=0), annot=True, cmap='Blues', fmt='.0f', cbar=False)
# ax.xaxis.set_ticks_position('top')
# ax.tick_params(axis='both', which='both', length=0)
# fig.suptitle('Average classification accuracy')
# plt.tight_layout()
# plt.savefig(os.path.join(os.path.dirname(fp), 'plots', 'filename.png'), dpi=150, format='PNG')
# #----------------------------------#


# fig, ax = plt.subplots(1,3)
# for m in models:
#     # select columns to plot
#     cols_to_plot = [col for col in df_perm.columns if m in col]
#     # plot
#     ax_i = list(models.keys()).index(m)
# #    ax[ax_i] = df_perm[cols_to_plot].plot.box(vert=False, whis=10)
#     ax[ax_i].boxplot(df_perm[cols_to_plot], vert=False)
#     ax[ax_i].axvline(0, ls='--', color='black', alpha=0.6)
#     ax[ax_i].set_yticklabels([])
#     ax[0].set_yticklabels(traincols)
#     ax[ax_i].set_title(m)
# fig.suptitle('Permutation importance')
# plt.tight_layout()
# plot_out = os.path.join(modeldir, prefix + '_perm_importance.png')
# plt.savefig(plot_out, dpi=300, format='PNG')















































