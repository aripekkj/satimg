# -*- coding: utf-8 -*-
"""
Created on Thu Feb 20 10:37:22 2025

Train classifier

@author: E1008409
"""

import argparse
import os
import pandas as pd
import numpy as np
import scipy as sc
import matplotlib.pyplot as plt
import pickle
import json
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.utils import compute_class_weight, compute_sample_weight
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import cross_val_score, StratifiedKFold, StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score


CLI = argparse.ArgumentParser()
CLI.add_argument(
    "directory",
    type=str,
    help='Directory for train, test, validation folders')
#fp = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/CMS'
args = CLI.parse_args()
fp = args.directory

train_dir = os.path.join(fp, 'train')
validation_dir = os.path.join(fp, 'validation')
test_dir = os.path.join(fp, 'test')
model_dir = os.path.join(fp, 'model')

if os.path.isdir(model_dir) == False:
    os.mkdir(model_dir)

# read data
train = pd.read_csv(os.path.join(train_dir, 'train.csv'), sep=';')
validation = pd.read_csv(os.path.join(validation_dir, 'validation.csv'), sep=';')
test = pd.read_csv(os.path.join(test_dir, 'test.csv'), sep=';')

# check for nodata
train = train.dropna()
validation = validation.dropna()
test = test.dropna()
# concatenate data
df = train
for i in [validation, test]:
    # create unique segment ids
    new_range = np.arange(df.segment.max()+1, len(i)+df.segment.max()+1)
    i['segment'] = new_range
    df = pd.concat([df, i])    

# columns to select training data
traincols = ['R', 'G', 'B']
# label encoder
le = LabelEncoder().fit(df.int_class)
y = le.transform(df.int_class)
#temp Perform stratified random sampling
sample_size = 1000
df = df.groupby('int_class').apply(lambda x: x.sample(min(len(x), sample_size))).reset_index(drop=True)


# models
models = {'RF': {'model': RandomForestClassifier(n_jobs=6, class_weight='balanced'),
                 'params': {"n_estimators": [50, 150, 200, 500], "max_depth": [3,6], "max_features": ['sqrt', 'log2']}
                 },
          'XGB': {'model': XGBClassifier(eval_metric='mlogloss', objective='multi:softmax', num_class=len(np.unique(train.int_class))),
                  'params': {'learning_rate':[0.001,0.01,0.1],
                             'n_estimators': [50, 150, 200, 500], 'max_depth': [3,6],
                             'subsample': [0.8, 0.5], },
                  }
          }

folds = dict()
skf = StratifiedGroupKFold(n_splits=10, random_state=42, shuffle=True)
# use field obs points
for i, (train, test) in enumerate(skf.split(df, df.int_class, groups=df.segment)):
    # save train, test point_id's to dictionary
    k = 'fold_' + str(i+1)
    folds[k] = (train, test)
    #double check that sets are separate (this step can be removed later)
    for t in train:
    #    print(t)
        if t in set(test):
            print('Value found in test')

# evaluate
for i, (train, test) in enumerate(skf.split(df, df.int_class, groups=df.segment)):
    break
    f = 'fold_' + str(i+1)
    print('Evaluating:', f)
    # select fold train and test
    df_train = df.iloc[train]
    df_test = df.iloc[test]
    
    # select columns
    X_train = df_train[traincols].to_numpy()
    y_train = le.transform(df_train['int_class'])
    X_test = df_test[traincols].to_numpy()
    y_test = le.transform(df_test['int_class'])

    # split validation set from fold X_train
    X_train, X_val, y_train, y_val = train_test_split(df_train, df_train['int_class'], 
                                                      test_size=0.2, random_state=42, 
                                                      stratify=df_train['int_class'])
    y_train = le.transform(y_train)
    y_val = le.transform(y_val)

    # select groups (ie. segments)
    train_groups = X_train['segment'].to_numpy()
    eval_groups = X_val['segment'].to_numpy()
    X_train = X_train[traincols]
    X_val= X_val[traincols]
    print('Classes in train set', np.unique(y_train ,return_counts=True))
    print('Classes in test set', np.unique(y_test ,return_counts=True))
    # sample weights
    train_weights = compute_sample_weight('balanced', y_train)
    train_class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
    eval_weights = compute_sample_weight('balanced', y_val)
    eval_class_weights = compute_class_weight('balanced', classes=np.unique(y_val), y=y_val)
    
    # StratifiedGroupKFold for hyperparameter tuning
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=False)
    for m in models:
        print(m)
        break
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
        search = RandomizedSearchCV(pipeline, param_distributions=param_dict, scoring='balanced_accuracy', cv=sgkf, return_train_score=True, n_iter=10, n_jobs=-1, refit=True)
        # find best params
        if m == 'XGB':
            result = search.fit(X_train, y_train, classifier__sample_weight=train_weights, groups=train_groups)
        else:
            result = search.fit(X_train, y_train, groups=train_groups)
    
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
        ax.set_xticks(np.arange(0,10))
        ax.set_xticklabels(np.arange(1,11))
        ax.set_xlabel('Fold')
        plt.suptitle(m + ' CV learning curve')
        plt.tight_layout()
        plot_out = os.path.join(model_dir, m + '_learningcurve.png')
        plt.savefig(plot_out, dpi=150, format='png')
        #plt.show()
        # get best estimator
        clf = result.best_estimator_
        if m == 'XGB':
            # scale data as not in pipeline
            scaler = StandardScaler().fit(X_train)
            X_train = scaler.transform(X_train)
            X_val = scaler.transform(X_val)
            # re-train with optimized params and early stopping (see 'Early Stopping' in: https://xgboost.readthedocs.io/en/stable/python/sklearn_estimator.html)
            best_params = models[m]['model'].get_params()
            for k in result.best_params_.keys():
               new_key = k.split('__')[1]
               best_params[new_key] = result.best_params_[k]
            # fit without early stop
            clf = XGBClassifier(**best_params)
            clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)],
                    sample_weight=train_weights,
                    sample_weight_eval_set=[train_weights, eval_weights]
                    )
            result_no_stop = clf.evals_result()
            # fit with early stop
            best_params['early_stopping_rounds'] = 10
            clf = XGBClassifier(**best_params)
            clf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)],
                    sample_weight=train_weights,
                    sample_weight_eval_set=[train_weights, eval_weights]
                    )
        #    clf.fit(X_train, y_train, 
        #            classifier__eval_set=[(X_train, y_train), (X_val, y_val)], 
        #            classifier__sample_weight=train_weights,
        #            classifier__early_stopping_rounds=10,
        #            classifier__group=train_groups,
        #            classifier__eval_group=eval_groups,
        #            classifier__sample_weight_eval_set=eval_class_weights)
            #retrieve performance metrics
            results = clf.evals_result()
            epochs = len(results['validation_0']['mlogloss'])
            epochs1 = len(result_no_stop['validation_0']['mlogloss'])
            x_axis1 = range(0, epochs1)
            x_axis = range(0, epochs)
            # plot log loss
            fig, ax = plt.subplots(2,1)
            ax[0].plot(x_axis1, result_no_stop['validation_0']['mlogloss'], label='Train', color='gray')
            ax[0].plot(x_axis1, result_no_stop['validation_1']['mlogloss'], label='Test', color='black')
            ax[1].plot(x_axis, results['validation_0']['mlogloss'], label='Train early stop')
            ax[1].plot(x_axis, results['validation_1']['mlogloss'], label='Test early stop')
            #ax.legend()
            plt.ylabel('MLogLoss')
            plt.suptitle(m + ' multi logloss with optimized params and early stopping')
            plot_out = os.path.join(model_dir, m + '_performance.png')
            plt.savefig(plot_out, dpi=150, format='png')
            #plt.show()
            
        # save model best params
        param_dict = models[m]['best_params']
        param_dict_out = os.path.join(model_dir, m + '_best_params.json')
        with open(param_dict_out, 'w') as f:
            json.dump(param_dict, f, indent=4)
            
        # define test set
        X_test = test[traincols].to_numpy()
        y_test = le.transform(np.array(test['int_class']))
    
        # dataframe for results
        predf = pd.DataFrame()
        predf['truth'] = y_test
        
        
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
        cm_cols = sorted(train.int_class.unique().tolist())
        cm_cols.append('Total')
        cmdf = pd.DataFrame(cmdf, index=cm_cols,
                            columns = cm_cols)
        # save confusion matrix dataframe as csv
        cmdf_name = m + '_cm.csv'
        cmdf_out = os.path.join(test_dir, cmdf_name)
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
        #    fig.suptitle('Confusion matrix of test set classifications')
        #    plt.tight_layout()
        #plt.savefig(os.path.join(os.path.dirname(fp), 'plots', m + '_cm.png'), dpi=150, format='PNG')
        
        # fit all data to model and save
        X = df[traincols].to_numpy()
        y = le.transform(df['int_class'])
#        X = np.vstack([X_train, X_val, X_test])
#        y = np.concatenate([y_train, y_val, y_test])
        if m == 'XGB':
            # define output model parameters without early stopping
            param_dict['early_stopping_rounds'] = None    
            clf = models[m]['model'].set_params(**param_dict)
#        clf.fit(X, y) # fit all data before saving
        model_out = os.path.join(model_dir, m + '.sav')
        pickle.dump(clf, open(model_out, 'wb'))
        
    
    
    
    












