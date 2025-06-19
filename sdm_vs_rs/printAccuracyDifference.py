# -*- coding: utf-8 -*-
"""
Created on Thu Jun 12 15:48:55 2025

Compare accuracy statistic tables

@author: E1008409
"""

import os
import pandas as pd


# files
fp = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Denmark/model/acc_df_describe.csv'
fp2 = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Denmark_/model/acc_df_describe_.csv'

# read
df_e = pd.read_csv(fp, sep=';')
df = pd.read_csv(fp2, sep=';')
# set index
df_e = df_e.set_index('Unnamed: 0')
df = df.set_index('Unnamed: 0')

# substract dataframes
df_s = df_e.subtract(df)
# compute change in percentage
df_p = round(((df_e - df) / df)*100, 1)

# Overall
print('RF Overall accuracy change: %.2f percent' % (df_s.RF_oa.loc['mean']/df.RF_oa.loc['mean']) )
print('SVM Overall accuracy change: %.2f percent' % (df_s.SVM_oa.loc['mean']/df.SVM_oa.loc['mean']) )
print('XGB Overall accuracy change: %.2f percent' % (df_s.XGB_oa.loc['mean']/df.XGB_oa.loc['mean']) )

# Precision
print('RF Producers accuracy change: %.2f percent' % (df_s.RF_SAV_pa.loc['mean']/df.RF_SAV_pa.loc['mean']) )
print('SVM Producers accuracy change: %.2f percent' % (df_s.SVM_SAV_pa.loc['mean']/df.SVM_SAV_pa.loc['mean']) )
print('XGB Producers accuracy change: %.2f percent' % (df_s.XGB_SAV_pa.loc['mean']/df.XGB_SAV_pa.loc['mean']) )
# Recall
print('RF Users accuracy change: %.2f percent' % (df_s.RF_SAV_ua.loc['mean']/df.RF_SAV_ua.loc['mean']) )
print('SVM Users accuracy change: %.2f percent' % (df_s.SVM_SAV_ua.loc['mean']/df.SVM_SAV_ua.loc['mean']) )
print('XGB Users accuracy change: %.2f percent' % (df_s.XGB_SAV_ua.loc['mean']/df.XGB_SAV_ua.loc['mean']) )


# create table of before, after, change, change in %
models = ['RF', 'SVM', 'XGB']
df_t = pd.DataFrame(columns=['nonedited', 'edited', 'change', 'change %'])
df_t['nonedited'] = df.T['mean'].round(3)
df_t['edited'] = df_e.T['mean'].round(3)
df_t['change'] = df_s.T['mean'].round(3)
df_t['change %'] = df_p.T['mean']

# save
outfile = os.path.join(os.path.dirname(fp), 'difference_table.csv')
df_t.to_csv(outfile, sep=';')

























