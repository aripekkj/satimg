# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 08:46:45 2024

@author: E1008409
"""


import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import rioxarray as rxr

fp = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Denmark/S2_LS2a_20220812_v1.01.tif'
xds = rxr.open_rasterio(fp, decode_coords='all')


# point data
# fp
fp = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Denmark/Eelgrass_2018-2023/Eelgrass_Kattegat_DK_2018_23_Fjord_removed.csv'
# read
df = pd.read_csv(fp, sep=',', encoding='latin-1')

df['date_dt'] = pd.to_datetime(df.Date, format='%Y%m%d')
# print
df.date_dt.dt.month.unique()
df.date_dt.min()

# select only zostera rows
#df = df[df.Species_name == 'Zostera marina']

# create classes column, Zostera, Other, Bare
df['new_class'] = 0
for idx, row in df.iterrows():
    if (row.Depth_M >= 6):
        df['new_class'].loc[idx] = 3
    elif (row.Species_name == 'Zostera marina') & (row.Coverage_pct >= 30):
        df['new_class'].loc[idx] = 1
    elif (row.Species_name != 'Zostera marina') & (row.Coverage_pct >= 30):
        df['new_class'].loc[idx] = 2
    elif row.Coverage_pct < 30:
        df['new_class'].loc[idx] = 0
# select by year
df_y = df[df.date_dt.dt.year == 2018]
# drop duplicates
df_y = df_y.drop_duplicates(subset=['ObservationEndX_UTM32', 'ObservationEndY_UTM32'], keep=False)

# plot
fig, ax = plt.subplots()
ax.hist(df_y['Coverage_pct'][(df_y['Depth_M'] < 3) & (df_y['Species_name'] == 'Zostera marina')])
plt.show()

# to gdf
gdf = gpd.GeoDataFrame(df_y, geometry=gpd.points_from_xy(df_y.ObservationEndX_UTM32, df_y.ObservationEndY_UTM32), crs=32632)
# save
gdf_out = fp.split('DK')[0] + '_classes_2018_32632.gpkg'
gdf.to_file(gdf_out, driver='GPKG', engine='pyogrio')


# ----------------------------------------------- 
# fp
fp = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Denmark/Macroalgae_2018-2023/Macroalgae_Kattegat_DK_2018_23_Fjord_removed.csv'
# read
df = pd.read_csv(fp, sep=',', encoding='latin-1')

df['date_dt'] = pd.to_datetime(df.Date, format='%d/%m/%Y')


# create classes column for Brown algae
df['new_class'] = 0
for idx, row in df.iterrows():
    if (row.Species_name == 'Zostera marina') & (row.Coverage_pct >= 30):
        df['new_class'].loc[idx] = 2
    elif (row.Species_name != 'Zostera marina') & (row.Coverage_pct >= 30):
        df['new_class'].loc[idx] = 1
    elif row.Coverage_pct < 30:
        df['new_class'].loc[idx] = 0
# select by year
df_y = df[df.date_dt.dt.year == 2018]

# to gdf
gdf = gpd.GeoDataFrame(df_y, geometry=gpd.points_from_xy(df_y.ObservationEndX_UTM32, df_y.ObservationEndY_UTM32), crs=32632)
# save
gdf_out = fp.split('DK')[0] + '2018_32632.gpkg'
gdf.to_file(gdf_out, driver='GPKG', engine='pyogrio')



# -----------------------------------------------
# Wadden Sea
fp = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Netherlands/Eelgrass_data/2017/Seagrass_WaddenSea_2017.csv'
df = pd.read_csv(fp, sep=';')
# to gdf
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df.LON84, df.LAT84), crs=4326)
# compute total coverage column
# compute sum of selected column range
gdf['totalcov'] = gdf[['bed_zosnol', 'bed_zosmar', 'bed_rupmar']].sum(axis=1) 
# create presence column
gdf['presence'] = np.where(gdf.totalcov > 0, 1, 0)

# counts
print(len(gdf[gdf.totalcov > 0.5]))

# reproject
gdf = gdf.to_crs(epsg=3035)
# save
gdf_out = fp.split('.')[0] + '_3035.gpkg'
gdf.to_file(gdf_out, driver='GPKG', engine='pyogrio')


