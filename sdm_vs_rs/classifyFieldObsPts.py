# -*- coding: utf-8 -*-
"""
Created on Thu Feb 22 10:07:29 2024

Generate generalized classes from field observations for SDM/RS study

@author: E1008409
"""


import numpy as np
import pandas as pd
import geopandas as gpd
import time 
import os
import fiona
from datetime import datetime 
import matplotlib.pyplot as plt

def rowSumFromCols(dataframe, column_list):
    """
    Calculate row sum from columns

    Parameters
    ----------
    dataframe : Pandas DataFrame
        DESCRIPTION.
    column_list : List
        DESCRIPTION.

    Returns
    -------
    colsum : Pandas Series
        DESCRIPTION.

    """
    cols = []
    for substr in column_list:
        temp = [c for c in dataframe.columns if substr in c]
        cols.extend(temp)
    # column sum row wise
    colsum = dataframe[cols].sum(axis=1)
    return colsum

# fp
fp = '/mnt/d/users/e1008409/MK/Velmu-aineisto/combined.velmu.aland.data.07.11.2022.csv'
fp_out = '/mnt/d/users/e1008409/MK/Velmu-aineisto/velmudata_07112022_selkameri_aoi_bounds.csv'
fp_shape = '/mnt/d/users/e1008409/MK/syke/ranta10_selkameri_aoi_32634.gpkg'

# read files
df = pd.read_csv(fp, sep=';', decimal=',', encoding='latin-1')
aoi = gpd.read_file(fp_shape)

# drop na string rows in menetelma
df = df[df['menetelma'] != 'NA']
df = df[df['menetelma'] != 'None']
df = df.dropna(subset=['menetelma'])
# replace
df = df.mask(df.menetelma == 'Dive', 21)
# to int
df.menetelma = df.menetelma.astype(str).astype(float)

# drop observations of floating sav
df = df.drop(columns=['Ajelehtiva_Fucus_elossa', 'Ajelehtiva_Fucus_kuollut', 'Ajelehtiva_rihmalevä', 'Ajelehtiva_makrofyytti', 'Ajelehtiva_tunnistamaton_kasviaines'])

# compute sum of selected column range
df['savcov'] = df[df.columns[154:-173]].sum(axis=1) # NOTE:check end column

# make datetime column
df['date'] = pd.to_datetime(df['pvm'], format='mixed')
df['year'] = pd.DatetimeIndex(df['date']).year
# print
print(df.date.head())
print(df.year.head())

#_______________________________________________
# spatial limit
if len(aoi.geometry) > 1:
    aoi = aoi.dissolve(by='objectid')
print(len(aoi.geometry))
# crs
aoi3067 = aoi.to_crs(epsg=3067)
# get bounds
bounds = aoi3067.bounds
# select df points within bounds
df = df[(df.e_euref >= bounds.minx.values[0]) & (df.e_euref <= bounds.maxx.values[0]) & (df.n_euref >= bounds.miny.values[0]) & (df.n_euref <= bounds.maxy.values[0])]
#_______________________________________________

# select rows by bottom type coverage 
df = df[df['pohja_yht'] >= 1]
# select by method
df = df[df.menetelma.isin([21,23,25,42])]
# column to numeric
#df.syvyys_mitattu = df.syvyys_mitattu.str.replace(',','.').astype(float)
#df.syvyys_poikkeama = df.syvyys_poikkeama.str.replace(',','.').astype(float)
# compute fixed depth
#df['syvyys_korjattu'] = df.syvyys_mitattu + df.syvyys_poikkeama

# fix depths to floats
if df.syvyys_mitattu.dtype != 'float64':
    # convert syvyys_mitattu to numeric
    df['temp'] = df.syvyys_mitattu.str.split('-')
    df[['temp', 'temp2']] = pd.DataFrame(df.temp.tolist(), index=df.index)
    df['field_depth'] = df['temp2'].str.replace(',','.').astype(float)
    # drop temp columns
    df = df.drop(['temp', 'temp2'], axis=1)
else:
    # absolute values
    df['field_depth'] = abs(df.syvyys_mitattu)
   
# bottom type columns to float
for col in df.columns[18:37]:
    df[col] = (df[col].replace(',','.', regex=True).replace('NA',0, regex=True).astype(float))


# find which column has the highest value
s = df.eq(df[df.columns[18:37]].max(axis=1), axis=0).sum(axis=1) # this can be used to mask rows which have multiple same values
df['subs_highest'] = df[df.columns[18:37]].idxmax(axis=1, skipna=True)
df['subs_high_val'] = df[df.columns[18:37]].max(axis=1, skipna=True)
df['subs_highest'] = df['subs_highest'].mask(s > 1, 'Mixed') # defines mixed bottom type if same values

# compute row sums for selected group
fucaceae = ['Fucus_sp', 'Fucus_vesiculosus', 'Fucus_radicans']
brown_algae = ['Chorda_filum', 'Halosiphon_tomentosum', 'Fucus_sp', 'Fucus_vesiculosus', 'Fucus_radicans',
               'Elachista_fucicola', 'Ectocarpus_siliculosus', 'Pylaiella_littoralis']

# sp. coverage
df['fucaceae_cov'] = rowSumFromCols(df, fucaceae)
# compute percentage from total coverage
df['fucus_from_total'] = df['fucaceae_cov']/df['savcov']*100

# column for new classes
df['new_class'] = 0
# copy df
df_b = df #backup when testing

# define classes, 1-dense mixed 3-sparse sav 4-very sparse or no sav 5-deep water
for idx, row in df.iterrows():
    if ((row.field_depth > 3) & (row.subs_highest != 'hiekka')):
        df['new_class'].loc[idx] = 5 # deep water other than sand substrate
    elif (row.field_depth > 5):
        df['new_class'].loc[idx] = 5 # deep water 
    elif ((row.savcov >= 30) & (row.fucus_from_total < 30)):# & (row.field_depth <= 3)):
        df['new_class'].loc[idx] = 1 # mixed sav
    elif ((row.savcov >= 30) & (row.fucus_from_total >= 30) & (row.field_depth <= 3)):
        df['new_class'].loc[idx] = 2 # fucus dominating
    elif ((row.savcov < 30) & (row.subs_highest == 'hiekka') & (row.field_depth < 5)):
        df['new_class'].loc[idx] = 3 # sandy substrate
    elif (row.savcov < 30):# & (row.field_depth <= 4):
        df['new_class'].loc[idx] = 4 # other substrate sparse or bare

# save
df.to_csv(fp_out, sep=';')











