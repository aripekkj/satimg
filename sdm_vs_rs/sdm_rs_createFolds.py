# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 13:48:28 2026

Create buffers for spatial block and folds

@author: E1008409
"""

import os
import geopandas as gpd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder


# filepaths
fp_poly = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Greece_spatial_block/segmentation/LSxGreece_10m10000_n5_s10_0_segments_iter_w_field_obs.gpkg'
fp_all_pts = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Greece_spatial_block/Greece_habitat_data_ml.gpkg'
fp_pts = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Greece_spatial_block/Greece_habitat_data_ml_LSxGreece_10m.gpkg'

# read
poly = gpd.read_file(fp_poly)
poly = poly.dissolve('value').reset_index()

gdf_all = gpd.read_file(fp_all_pts)
gdf = gpd.read_file(fp_pts)
# sample hab_class to polys
poly = gpd.sjoin(poly, gdf[['geometry', 'int_class']], how='left')
# drop index_right column
poly = poly.drop('index_right', axis=1).reset_index(drop=True)
# buffers
buffers = [100, 1000]
# buffer 
poly_buf = gpd.GeoDataFrame(geometry=poly.buffer(100, resolution=20), crs = poly.crs)

# make stratified KFolds for points
skf = StratifiedKFold(n_splits=10, random_state=42, shuffle=True)

for i, (train, test) in enumerate(skf.split(poly.value, poly.int_class)):
    # column names for folds    
    k_tr = f'fold_{i+1}_train'
    k_te = f'fold_{i+1}_test'
    # select test polygons
    test_poly = poly.loc[poly.index[test]]
    gdf[k_te] = False
    # set test points by segment id
    for idx, row in gdf.iterrows():
        if row.segments_iter in (test_poly.value.values):
            gdf.loc[idx, k_te] = True
    # select buffered test polygons
    test_poly_buf = poly_buf.loc[poly_buf.index[test]]
    
    # spatial join
    joined = gpd.sjoin(
        gdf,
        test_poly_buf,
        predicate='within',
        how='left'
    )
    # drop duplicates
    joined = joined.drop_duplicates('ID')
    # detect the correct index_right column
    right_col = [c for c in joined.columns if c.startswith("index_right")][0]

    # True if point is inside at least one test polygon
    inside_mask = joined[right_col].notna()

    # store mask (True = keep for training)   
    # set train fold to column
    gdf[k_tr] = ~inside_mask

    # set train test for polygons
    #poly[k_tr] = True # default column to true value
    #poly.loc[test, k_tr] = False # set test polygon rows false
    # set train test for buffered polygons
    poly_buf[k_tr] = True # default column to true value
    poly_buf.loc[test, k_tr] = False # set test polygon rows false

# sample train test folds to segment polys
poly = gpd.sjoin(poly, gdf, how='left')
    
# save
gdf_out = fp_pts.split('.')[0] + '_folds.gpkg'
gdf.to_file(gdf_out)
# save buffered polygons
poly_out = fp_poly.split('.')[0] + '_folds.gpkg'
poly.to_file(poly_out)
# save buffered polygons
buffered_out = fp_poly.split('.')[0] + '_buffered.gpkg'
poly_buf.to_file(buffered_out)




