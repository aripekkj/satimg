# -*- coding: utf-8 -*-
"""
Created on Wed Jun  4 09:02:34 2025

Merge multiple rasters

@author: E1008409
"""

import os 
import glob
import argparse
import numpy as np
import rasterio as rio
import rioxarray as rxr
from rioxarray.merge import merge_arrays


CLI = argparse.ArgumentParser()
CLI.add_argument(
    "directory",
    type=str,
    help='Directory for train, test, validation folders')

args = CLI.parse_args()
# dirpath
fp = args.directory
# list models
models = ['RF', 'SVM', 'XGB']

for m in models:
# =============================================================================
#     # read datasets
#     ds = [rxr.open_rasterio(f) for f in glob.glob(os.path.join(fp, '*.tif'))]
#     # merge
#     merged = merge_arrays(ds)
#     # output file
#     out_fp = os.path.join(fp, m + '_merged.tif')
#     # save to file 
#     merged.rio.to_raster(out_fp, driver='GTiff', compress='DEFLATE')
# 
#     
# =============================================================================
    # use rasterio
    files = [f for f in glob.glob(os.path.join(fp, '*.tif'))]
    
    dest, output_transform = rio.merge.merge(files)
    
    with rio.open(files[0]) as src:
        outprof = src.profile.copy()
        outprof.update(
            {
                "driver": "GTiff",
                "count": 1,
                "nodata": np.nan,
                "height": dest.shape[1],
                "width": dest.shape[2],
                "transform": output_transform,
                "compress": 'DEFLATE'
            }
        )
    # output file
    out_fp = os.path.join(fp, m + '_merged.tif')
    
    with rio.open(out_fp, "w", **outprof) as dst:
        dst.write(dest)
