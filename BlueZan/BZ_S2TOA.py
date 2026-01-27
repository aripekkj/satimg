# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 08:43:37 2026

Save same resolution bands from Sentinel-2 as GeoTiff

@author: Matkauser
"""

import argparse
import os
import glob
import numpy as np
import rasterio as rio


CLI = argparse.ArgumentParser()
CLI.add_argument('filepath',
                 type=str,
                 help='Filepath for Sentinel-2 .SAFE file')
CLI.add_argument('Res',
                 type=int,
                 help='Resolution of image bands to select. 10, 20 or 60')
args = CLI.parse_args()
# args
fp = args.filepath
res = args.Res

def S2TOA(fp, res):
    if res == 10:
        bands = ['B02', 'B03', 'B04', 'B08']
    
    stack = []
    for b in bands:
        # find band
        f = glob.glob(os.path.join(fp, '**', '*IMG_DATA', '*' + b + '*.jp2'), recursive=True)
        print(f)
        # read image
        with rio.open(f[0]) as src:
            img = src.read()
            profile = src.profile
        # append to stack
        stack.append(img)
    
    # list to array
    img_array = np.vstack(stack)
    print('Array shape', img_array.shape)
    
    # create output profile
    outprof = profile.copy()
    outprof.update(count=len(stack),
                   dtype='uint32',
                   compress='DEFLATE',
                   driver='GTiff')
    # create outfile
    namesplit = os.path.basename(fp).split('_')
    outname = namesplit[0] + '_' + namesplit[1] + '_' + namesplit[2] + '_' + namesplit[5] + '_' + str(res) + 'm_bands.tif'
    outfile = os.path.join(os.path.dirname(fp), outname)
    print(outfile)
    print(outprof)
    # save
    with rio.open(outfile, 'w', **outprof) as dst:
        dst.write(img_array)
    print('Saved file to', outfile)
    
def main():
    S2TOA(fp, res)

if __name__ == '__main__':
    main()















