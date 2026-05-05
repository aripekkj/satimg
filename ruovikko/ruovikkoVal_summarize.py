# -*- coding: utf-8 -*-
"""
Created on Mon Apr 20 13:19:04 2026

@author: E1008409
"""

import os
import glob
import pandas as pd
import numpy as np
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

    TN, FP = cm.iloc[0, 0], cm.iloc[0, 1]
    FN, TP = cm.iloc[1, 0], cm.iloc[1, 1]

    precision = TP / (TP + FP) if (TP + FP) > 0 else np.nan
    recall = TP / (TP + FN) if (TP + FN) > 0 else np.nan
    f1 = 2 * precision * recall / (precision + recall) \
         if (precision + recall) > 0 else np.nan

    specificity = TN / (TN + FP) if (TN + FP) > 0 else np.nan
    balanced_accuracy = (recall + specificity) / 2

    return precision, recall, f1, balanced_accuracy


fig,ax = plt.subplots(1,2)

for m in METHODS:
    # Collect metrics per file
    precision, recall, f1, balanced_accuracy = [], [], [], []

    for file in glob.glob(os.path.join(ROOT_DIR, f'**/*{m}*.csv'), recursive=True):
            print(file)
            cm = read_confusion_matrix(file)
            metrics = binary_metrics_from_cm(cm)
            if metrics is not None:
                p, r, f, ba = metrics
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
plt.suptitle("Accuracy Metrics for years 2016-2024")
fig_out = os.path.join(ROOT_DIR, f'val_acc_{m}.png')
plt.tight_layout()
#plt.savefig(fig_out, dpi=300)
plt.show()

    







