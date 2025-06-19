# -*- coding: utf-8 -*-
"""
Created on Tue Jun 17 15:38:35 2025

Summarize DNASense model evaluation results

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
from sklearn import metrics
import seaborn as sns
import argparse

# dir
fp = '/mnt/d/users/e1008409/MK/DNASense/FIN/model'

# map npy files
files = [f for f in glob.glob(os.path.join(fp, '*/*/acc_df.csv'))]

models = ['RF', 'SVM', 'XGB']

fig, ax = plt.subplots()
# plot results from files
for f in files:
    # read
    df = pd.read_csv(f, sep=';')
    # get year from folder name
    year = os.path.basename(os.path.dirname(os.path.dirname(f)))
    idx = files.index(f)
    ax.boxplot(df[['RF_oa', 'SVM_oa', 'XGB_oa']], vert=True, positions=np.array(np.arange(idx, idx+0.75, 0.25)))
    ax.set_title('CV model overall accuracy')

    # add year:
    sec = ax.secondary_xaxis(location=1)
    sec.set_xticks([idx+0.25], labels=[year])
    sec.tick_params('x', length=0)
ax.set_xticklabels(models*len(files), rotation=45)
plt.tight_layout()
oa_out = os.path.join(fp, 'OA_accuracies.png')
plt.savefig(oa_out, dpi=300, format='PNG')

plt.show()

# plot kappa statistic
fig, ax = plt.subplots()
# plot results from files
for f in files:
    # read
    df = pd.read_csv(f, sep=';')
    # get year from folder name
    year = os.path.basename(os.path.dirname(os.path.dirname(f)))
    idx = files.index(f)
    ax.boxplot(df[['RF_kappa', 'SVM_kappa', 'XGB_kappa']], vert=True, positions=np.array(np.arange(idx, idx+0.75, 0.25)))
    ax.set_title('CV model overall accuracy')

    # add year:
    sec = ax.secondary_xaxis(location=1)
    sec.set_xticks([idx+0.25], labels=[year])
    sec.tick_params('x', length=0)
ax.set_xticklabels(models*len(files), rotation=45)
plt.tight_layout()
oa_out = os.path.join(fp, 'Kappa_accuracies.png')
plt.savefig(oa_out, dpi=300, format='PNG')

plt.show()




# multiplot for users and producers accuracy for each class
nclass = 4
ncol = np.arange(0,2,1)
nrow = np.arange(0,nclass)
 
hablist = ['Deep_water', 'Fucus', 'Low', 'OtherSAV'] #list(np.unique(gdf.hab_class_ml))
fig, ax = plt.subplots(len(nrow), 2, figsize=(12,8), sharex=True, sharey=True)
idx_pos = []
for f in files:
    # read
    df = pd.read_csv(f, sep=';')
    # get year from folder name
    year = os.path.basename(os.path.dirname(os.path.dirname(f)))
    idx = files.index(f)
    idx_pos.extend(np.arange(idx, idx+0.75, 0.25).tolist())
    for hab in hablist:
        cols_to_plot = [h for h in df.columns if hab in h]
        
        # select producer's and user's accuracy columns
        cols_pa = [c for c in cols_to_plot if 'pa' in c]
        cols_ua = [c for c in cols_to_plot if 'ua' in c]
        # Filter data using np.isnan
        plot_pa = df[cols_pa].to_numpy()
        mask = ~np.isnan(plot_pa)
        filtered_pa = [d[ma] for d, ma in zip(plot_pa.T, mask.T)]
        
        plot_ua = df[cols_ua].to_numpy()
        mask = ~np.isnan(plot_ua)
        filtered_ua = [d[ma] for d, ma in zip(plot_ua.T, mask.T)]
        
        nr = hablist.index(hab)
        ax[nr,0].set_title(hab, x=1)
        ax[nr,0].axhline(0.5, ls='--', lw=0.5, alpha=0.4, color='black')    
        # plot
        ax[nr,0].boxplot(filtered_pa, vert=True, positions=np.array(np.arange(idx, idx+0.75, 0.25)), labels=models) # acc_df[cols_pa]    
        ax[nr,1].boxplot(filtered_ua, vert=True, positions=np.array(np.arange(idx, idx+0.75, 0.25)), labels=models)
        ax[nr,1].axhline(0.5, ls='--', lw=0.5, alpha=0.4, color='black')
        

        if nr == 0:        
            sec = ax[nr, 0].secondary_xaxis(location=1)
            sec.set_xticks([idx+0.25], labels=[year])
            sec.tick_params('x', length=0)
            sec = ax[nr, 1].secondary_xaxis(location=1)
            sec.set_xticks([idx+0.25], labels=[year])
            sec.tick_params('x', length=0)

        if nr == len(nrow)-1:
 #           ax[nr,0].set_xticks(idx_pos, labels=models*len(files))
            ax[nr,0].set_xlabel('Producers accuracy')
            ax[nr,1].set_xlabel('Users accuracy')
            ax[nr,0].tick_params(axis='x', labelrotation=45)
            ax[nr,1].tick_params(axis='x', labelrotation=45)
plt.tight_layout()
fig.suptitle('CV model accuracies on test set', y=1.05)
plot_ua_pa_out = os.path.join(fp, 'P_U_accuracies.png')
plt.savefig(plot_ua_pa_out, dpi=300, format='PNG')










