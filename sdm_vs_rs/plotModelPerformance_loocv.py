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

def plot_confusion_with_metrics(cm, model, output_dir, labels=None, cmap="Blues"):
    """
    Plot confusion matrix with:
        - raw counts
        - row-wise percentages
        - column-wise percentages
        - metrics table (precision, recall, F1)
    """
    
    n = cm.shape[0]

    if labels is None:
        labels = [f"Class {i}" for i in range(n)]

    # --- Compute row + column percentages ---
    row_sums = cm.sum(axis=1, keepdims=True)
    col_sums = cm.sum(axis=0, keepdims=True)

    row_pct = cm / row_sums
    col_pct = cm / col_sums

    # --- Confusion matrix annotation ---
    annot = np.empty_like(cm, dtype="object")
    for i in range(n):
        for j in range(n):
            annot[i, j] = (
                f"{int(cm[i,j]):,}\n"
                f"R:{row_pct[i,j]*100:4.1f}% \n C:{col_pct[i,j]*100:4.1f}%"
            )

    # --- Metrics ---
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    metrics_df = pd.DataFrame({
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }, index=labels).round(4)

    # --- Plot confusion matrix ---
    fig, ax = plt.subplots(2, 1, figsize=(11, 14), gridspec_kw={"height_ratios": [3, 1]})

    hm = sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        linewidths=0.5,
        linecolor="gray",
        ax=ax[0],
        annot_kws={"size": 35 / np.sqrt(len(cm))}
    )
    
    hm.set_xticklabels(hm.get_xmajorticklabels(), fontsize = 20)
    hm.set_yticklabels(hm.get_xmajorticklabels(), fontsize = 20)
    
    ax[0].set_title(model + " 10-Fold Averaged Confusion Matrix ", fontsize=20)
    #ax[0].set_xlabel("Predicted", fontsize=13)
    #ax[0].set_ylabel("Actual", fontsize=13)

    # --- Table below the heatmap ---
    ax[1].axis("off")
    table = ax[1].table(
        cellText=metrics_df.values,
        rowLabels=metrics_df.index,
        colLabels=metrics_df.columns,
        cellLoc='center',
        loc='center'
    )
    table.scale(1, 2)
    table.auto_set_font_size(False)
    table.set_fontsize(20)

    plt.tight_layout()
    
    #output filename
    plot_out = os.path.join(output_dir, model + '_cm_metrics_plot.png')
    plt.savefig(plot_out, dpi=300)
    #plt.show()

    return row_pct, col_pct, metrics_df

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

# fp for testing
#fp_pts = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/spatial_block/BlackSea/Black_Sea_habitat_data_init_LSxBLK_20200313_buf100_folds.gpkg'
#fp_npy = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/spatial_block/BlackSea/model/models_cv_result.npy'
#fp = os.path.dirname(fp_pts)

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

# folds
n_folds = [c for c in gdf.columns if '_train' in c] # get number of folds from train fold columns
folds = ['fold_' + str(i) for i in np.arange(1,len(n_folds)+1,1)]

# class names
classes = np.unique(gdf.hab_class_ml)
pred_df = pd.DataFrame()
#cms = []
#cm_list = [] # list to store fold cm matrix
acc_df = pd.DataFrame(index=folds)

for m in models.keys():
    # evaluation folds
    for f in folds:
        
        # segment ids for train ,test in fold
        f_train = f + '_train'
        f_test = f + '_test'
        train_ids = gdf['segments'][gdf[f_train] == True].values
        test_ids = gdf['segments'][gdf[f_test] == True].values
        
        # test that no same values in train and test
        test_similarity = set(train_ids).intersection(set(test_ids))
        if len(test_similarity) != 0: 
            print('Found same values in train and test sets')
            break
    
        # select by segment id
        df_train = df[df.segment_id.isin(train_ids)]
        df_test = df[df.segment_id.isin(test_ids)]
        
        # select columns
        X_train = StandardScaler().fit_transform(df_train[traincols].to_numpy())
        y_train = le.transform(df_train['int_class'])
        X_test = StandardScaler().fit_transform(df_test[traincols].to_numpy())
        y_test = le.transform(df_test['int_class'])
        # try model performance on test
        predf = pd.DataFrame()
        predf['truth'] = y_test
        predf['fold'] = f
        
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
                
        # concat result
        pred_df = pd.concat([pred_df, predf])
    print('Model results', m)
    # create confusion matrix
    cm = metrics.confusion_matrix(pred_df['truth'], pred_df[m + '_predict'])
#    cm_list.append(cm)
#    val_cm = metrics.confusion_matrix(y_test, clf.predict(X_test))
#    # compute row and col sums
    total = cm.sum(axis=0)
    rowtotal = cm.sum(axis=1)
    rowtotal = np.expand_dims(rowtotal, axis=0).T #expand dims and transpose
    rowtotal_sum = np.array(rowtotal.sum()) 
    rowtotal = np.vstack([rowtotal, rowtotal_sum]) # stack row sum
#    # create cm DataFrame
    cmdf = np.vstack([cm,total]) # vertical stack
    cmdf = np.hstack((cmdf, rowtotal)) # horizontal stack
#    cms.append(cmdf)
    cm_cols = gdf.hab_class_ml.unique().tolist()
    cm_cols.append('Total')
    cmdf = pd.DataFrame(cmdf, index=cm_cols,
                        columns = cm_cols)
    
#     # print
    print(cmdf)
#    print(pd.crosstab(pred_df.truth, pred_df[m+'_predict'], margins=True))
     # compute common accuracy metrics
    o_accuracy = np.sum(cm.diagonal()) / np.sum(cm.sum(axis=0))
    p_accuracy = cm.diagonal() / cm.sum(axis=0) # producer's accuracy
    u_accuracy = cm.diagonal() / cm.sum(axis=1) # user's accuracy
    kappa = metrics.cohen_kappa_score(predf.truth, predf[m + '_predict']) #kappa statistic
    print('Cohens Kappa %.2f' % (kappa))
    print(m + ' Overall accuracy %.2f' % (o_accuracy))
    print(m + ' Users accuracy', u_accuracy)
    print(m + ' Producers accuracy', p_accuracy)    
    # store accuracies to df
    fold_oa = m + '_oa'
    pacc_cols = [m  + '_' + c + '_pa' for c in classes]
    uacc_cols = [m  + '_' + c + '_ua' for c in classes]
    kappa_col = m + '_kappa'
    
    acc_df.loc[f, fold_oa] = o_accuracy
    acc_df.loc[f, pacc_cols] = p_accuracy
    acc_df.loc[f, uacc_cols] = u_accuracy
    acc_df.loc[f, kappa_col] = kappa

#    cms_arr = np.array(cm_list)

    row_pct, col_pct, metrics_df = plot_confusion_with_metrics(cm, m, modeldir, labels=classes)
    
# redefine index
pred_df.index = np.arange(0,len(pred_df))
#cms = np.array(cms)

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












































