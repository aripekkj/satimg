# -*- coding: utf-8 -*-
"""
Created on Wed Dec 10 18:39:45 2025

@author: E1008409
"""

import argparse
import os
import glob
import pandas as pd
import numpy as np
import xarray as xr
import rioxarray as rxr
import rasterio as rio
from dask import delayed
from dask import compute


CLI = argparse.ArgumentParser()
CLI.add_argument(
    "filepath",
    type=str,
    help='Filepath to raster')
CLI.add_argument(
    "Band_name",
    type=str,
    help='Name of the raster band')

#fp = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/spatial_block/Denmark/classification/*RF*.tif'
#name ='SAV'

@delayed
def saveBand(fp, name):
# =============================================================================
#     # read 
#     ds = rxr.open_rasterio(fp)
#     # coords
#     band, x, y = ds.indexes.values()
#     
#     # get index
#     for n in ds.long_name:
#         if name in n:
#             i = ds.long_name.index(n)
# #    i = ds.long_name.index(name)
#     
#     # new dataset to store single band
#     xr_dataset = xr.Dataset()
#     
#     xr_dataset[name] = xr.DataArray(ds.data[i], 
#         dims=('y', 'x'), 
#         coords={'x': x,
#                 'y': y}
#             )
#     # add crs
#     xr_dataset = xr_dataset.rio.write_crs(ds.rio.crs)
#     # save
#     xds_out = fp.split('.tif')[0] + '_' + name + '.tif'
#     xr_dataset.rio.to_raster(xds_out, compress='DEFLATE')
# 
#     print('Saved', xds_out)
#     
# =============================================================================
    # use rasterio
    with rio.open(fp) as src:
        img = src.read()
        dsc = src.descriptions
        profile = src.profile
    # get index
    for n in dsc:
        if name in n:
            i = dsc.index(n)
    # select band
    band_out = img[i,:,:]
    # ensure data within range 0 1
    band_out = np.where((band_out < 0) | (band_out > 1), np.nan, band_out)
    # expand dims
    band_out = np.expand_dims(band_out, axis=0)
    # output profile
    outprof = profile.copy()
    outprof.update(count=1,
                   compress='DEFLATE')    
    # save
    xds_out = fp.split('.tif')[0] + '_' + name + '.tif'
    with rio.open(xds_out, 'w', **outprof) as dst:
        dst.write(band_out)

    print('Saved', xds_out)
    

#if __name__ == 'main':
args = CLI.parse_args()
fp = args.filepath
name = args.Band_name
# map files
files = [f for f in glob.glob(fp)]
print(files)
delayed_funcs = []
for f in files:
    delayed_funcs.append(saveBand(f, name))        
compute(delayed_funcs)


