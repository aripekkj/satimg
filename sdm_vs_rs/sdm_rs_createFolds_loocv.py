# -*- coding: utf-8 -*-
"""
Created on Wed Feb 18 13:48:28 2026

Create buffers for spatial block and folds

@author: E1008409
"""

import sys
import os
import geopandas as gpd
import rasterio as rio
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import LabelEncoder

def setTrainFold(geodataframe, buffer, fold_name, column):
    # spatial join
    joined = gpd.sjoin(
        geodataframe,
        buffer,
        predicate='intersects',
        how='left'
    )
    # drop duplicates
    joined = joined.drop_duplicates(column)
    # detect the correct index_right column
    right_col = [c for c in joined.columns if c.startswith("index_right")][0]

    # True if point is inside at least one test polygon
    inside_mask = joined[right_col].notna()
    # set points outside buffer as train
    geodataframe[fold_name] = ~inside_mask
    geodataframe[fold_name] = geodataframe[fold_name].fillna(False).astype(bool)
    return geodataframe


fp_poly = sys.argv[1]
fp_all_pts = sys.argv[2]
fp_pts = sys.argv[3]

# filepaths
fp_poly = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/spatial_block/BlackSea/segmentation/LSxBLK_2020031312000_n5_s10_0_iter.gpkg'
fp_all_pts = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/spatial_block/BlackSea/Black_Sea_habitat_data_init_encoded.gpkg'
fp_pts = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/spatial_block/BlackSea/Black_Sea_habitat_data_init_LSxBLK_20200313.gpkg'

# set buffer
buffer = 100

# read
poly = gpd.read_file(fp_poly)
poly = poly.dissolve('value').reset_index()
gdf = gpd.read_file(fp_pts)
gdf_all = gpd.read_file(fp_all_pts)

# sample hab_class to polys
poly = gpd.sjoin(poly, gdf[['geometry', 'int_class']], how='left')
# drop index_right column
poly = poly.drop('index_right', axis=1).reset_index(drop=True)
# dropnan, ie. polys that don't have points
poly = poly[poly['int_class'].notna()].reset_index()
# select points with segment value above 0 (nodata in segments)
gdf = gdf[gdf.segments > 0 ]

# buffer 
poly_buf = gpd.GeoDataFrame(geometry=poly.buffer(buffer, resolution=20), crs = poly.crs)

# make stratified KFolds for points
loo = LeaveOneOut()

for i, (train, test) in enumerate(loo.split(poly.value)):
    # column names for folds    
    k_tr = f'fold_{i+1}_train'
    k_te = f'fold_{i+1}_test'
    
    # select test polygons
    test_poly = poly.loc[poly.index[test]]
    # select buffered test polygons
    test_poly_buf = poly_buf.loc[poly_buf.index[test]]
    
    # test fold column, default to false
    gdf[k_te] = False
    # set test points to True by segment id
    for idx, row in gdf.iterrows():
        if row.segments in (test_poly.value.values):
            gdf.loc[idx, k_te] = True
    
    # set train folds for filtered points
    gdf = setTrainFold(gdf, test_poly_buf, k_tr, 'point_id')

    # ------------------------------ #
    # join test fold by point id for unfiltered points
    gdf_all = gdf_all.merge(gdf[['point_id', k_te]], on='point_id', how='left')
    # keep NaN values in bool column
    gdf_all[k_te] = gdf_all[k_te].fillna(False).astype(bool)
    # set train folds for filtered points
    gdf_all = setTrainFold(gdf_all, test_poly_buf, k_tr, 'point_id')
    # ------------------------------ #

    # set train, test folds for polygons
    poly = setTrainFold(poly, test_poly_buf, k_tr, 'value')
    poly[k_te] = False # default to false
    poly.loc[test, k_te] = True # set test fold
    poly[k_te] = poly[k_te].fillna(False).astype(bool)
    # set train test for buffered polygons
    poly_buf[k_tr] = True # default column to true value
    poly_buf.loc[test, k_tr] = False # set test polygon rows false
    poly_buf.loc[test, k_te] = True # set test fold
    poly_buf[k_te] = poly_buf[k_te].fillna(False).astype(bool)

# get true in test_fold columns to single column
testcols = [c for c in gdf.columns if 'test' in c]
gdf['test_fold'] = gdf[testcols].idxmax(axis=1)

# merge by point_id to unfiltered points
gdf_all = gdf_all.merge(gdf[['point_id', 'test_fold']], on='point_id', how='left')

# save
gdf_out = fp_pts.split('.')[0] + '_buf' + str(buffer) + '_folds.gpkg'
gdf.to_file(gdf_out)
# save
gdf_all_out = fp_all_pts.split('.')[0] + '_buf' + str(buffer) + '_folds.gpkg'
gdf_all.to_file(gdf_all_out)
# save polygons
poly_out = fp_poly.split('.')[0] + '_buf' + str(buffer) + '_folds.gpkg'
poly.to_file(poly_out)
# save buffered polygons
buffered_out = fp_poly.split('.')[0] + '_buf' + str(buffer) + '_buffered.gpkg'
poly_buf.to_file(buffered_out)




