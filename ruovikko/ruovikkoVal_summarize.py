# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 13:19:04 2026

@author: E1008409
"""

import os
import glob
import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT_DIR = Path("/mnt/d/users/e1008409/MK/Ruovikko/validointi/")
METHODS = ["CORINE", "Bayes"] 

def read_confusion_matrix(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=';')

def binary_metrics_from_cm(cm: pd.DataFrame):
    cm = cm.drop('Unnamed: 0', axis=1)
    if cm.shape != (2, 2):
        return None
    
#    TN, FP = cm.iloc[0, 0], cm.iloc[0, 1] # TN,FP,FN,TP as in binary classification. see https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
#    FN, TP = cm.iloc[1, 0], cm.iloc[1, 1]

    TP, FN = cm.iloc[0, 0], cm.iloc[0, 1] 
    FP, TN = cm.iloc[1, 0], cm.iloc[1, 1]
    

    precision = TP / (TP + FP) #if (TP + FP) > 0 else np.nan
    recall = TP / (TP + FN) #if (TP + FN) > 0 else np.nan
    f1 = 2 * precision * recall / (precision + recall) #\
         #if (precision + recall) > 0 else np.nan

    specificity = TN / (TN + FP) #if (TN + FP) > 0 else np.nan
    balanced_accuracy = (recall + specificity) / 2

    print('p', precision)
    print('r', recall)

    return precision, recall, f1, specificity, balanced_accuracy


fig,ax = plt.subplots(1,2)

for m in METHODS:
    # Collect metrics per file
    precision, recall, f1, balanced_accuracy = [], [], [], []

    for file in glob.glob(os.path.join(ROOT_DIR, f'**/*{m}*threshold_0.csv'), recursive=True):
            print(file)
            cm = read_confusion_matrix(file)
            metrics = binary_metrics_from_cm(cm)
            if metrics is not None:
                p, r, f, s, ba = metrics
                precision.append(p)
                recall.append(r)
                f1.append(f)
                balanced_accuracy.append(ba)
    
    # Drop NaNs
    precision = [x for x in precision if not np.isnan(x)]
    recall = [x for x in recall if not np.isnan(x)]
    f1 = [x for x in f1 if not np.isnan(x)]
    balanced_accuracy = [x for x in balanced_accuracy if not np.isnan(x)]
    
    # Boxplots
    idx = METHODS.index(m)
#    plt.figure(figsize=(8, 5))
    ax[idx].boxplot(
        [precision, recall, f1, balanced_accuracy],
    )
    # Set xticklabels
    labels = ["Precision", "Recall", "F1-score", "Balanced \nAccuracy"]
    ax[idx].set_xticklabels(labels)   
    # Rotate labels 45 degrees
    ax[idx].tick_params(axis='x', labelrotation=45)
    ax[idx].set_ylabel("Score")
    ax[idx].set_title(f"{m} interpretation")
    ax[idx].set_ylim(0, 1)
    ax[idx].grid(alpha=0.3)
# title for whole plot, output filename and save
plt.suptitle("Accuracy Metrics for Reedbed Presence for years 2016-2024")
m_str = '_'.join(METHODS)
fig_out = os.path.join(ROOT_DIR, f'val_acc_{m_str}.png')
plt.tight_layout()
plt.savefig(fig_out, dpi=300)
plt.show()
    
        
# Different Bayes thresholds
# Collect metrics per file
years = ['2016','2017','2018','2019','2020','2021','2022','2023','2024']
metrics_dict = {}

for y in years:
    precision, recall, f1, specificity, balanced_accuracy = [], [], [], [], []
    print(y)
    for file in glob.glob(os.path.join(ROOT_DIR, f'**{y}/cm_plots/*.csv'), recursive=True):
        print(file)
        thr = file.split('_')[-3]
        cm = read_confusion_matrix(file)
        metrics = binary_metrics_from_cm(cm)
        if metrics is not None:
            p, r, f, s, ba = metrics
            precision.append((thr, p))
            recall.append((thr, r))
            f1.append((thr, f))
            specificity.append((thr, s))
            balanced_accuracy.append((thr, ba))
        print(metrics)

    # store metrics to dict
    metrics_dict[y] = {'precision': precision,
                       'recall': recall,
                       'f1_score': f1,
                       'specificity': specificity,
                       'balanced_accuracy': balanced_accuracy
                       }

# save as json
bayes_metrics_out = os.path.join(ROOT_DIR, 'bayes_metrics.json')
j = json.dumps(metrics_dict, indent=4)
with open(bayes_metrics_out, "w") as f:
    print(j, file=f)
    
# plot
plotout = os.path.join(ROOT_DIR, 'bayes_threshold_metrics_plot.png')

metrics = ['precision', 'recall', 'f1_score', 'balanced_accuracy']
xticklabels = [-5,-4,-3,-2,-1,0,1,2,3,4]
yticks = np.arange(0,1,0.1)

fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True)
axes = axes.flatten()

for ax, metric in zip(axes, metrics):

    for year, year_data in sorted(metrics_dict.items()):

        # sort by first element of tuple
        points = sorted(year_data[metric], key=lambda x: float(x[0]))

        x = [float(p[0]) for p in points]
        y = [float(p[1]) for p in points]

        ax.plot(x, y, marker='o', label=year)

    ax.set_title(metric.replace('_', ' ').title())
    ax.set_ylabel(metric.replace('_', ' ').title())
    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels)
    ax.set_yticks(yticks)
    ax.axvline(127, ls='--', color='gray', alpha=0.5)
    ax.grid(True, alpha=0.3)

axes[2].set_xlabel('Threshold')
axes[3].set_xlabel('Threshold')

axes[1].legend(title='Year', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout()
plt.savefig(plotout, dpi=300)
plt.show()


# ROC Curve
plotout = os.path.join(ROOT_DIR, 'bayes_roc_plot.png')

plt.figure(figsize=(8, 6))

for year, data in sorted(metrics_dict.items()):

    roc_points = sorted(
        zip(data['specificity'], data['recall']),
        key=lambda x: float(x[0][0])  # sort by threshold
    )

    fpr = [1 - float(spec[1]) for spec, _ in roc_points]
    tpr = [float(rec[1]) for _, rec in roc_points]

    plt.plot(fpr, tpr, marker='o', label=year)

plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate (Recall)')
plt.title('ROC Curve')
plt.legend()
plt.grid(True)
plt.savefig(plotout, dpi=300)
plt.show()
















