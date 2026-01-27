# -*- coding: utf-8 -*-
"""
Spyder Editor

Satellite image Gap-analysis based on overlay of polygons (digitized or obia)
and field points.

This is a temporary script file.
"""

import os
import pandas as pd
import rasterio as rio
from rasterio.features import rasterize
import geopandas as gpd
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from sklearn.preprocessing import LabelEncoder

fp_poly = 'D:/BlueZan/S2/Chwaka/obia/obia_7_0007.gpkg' # filepath for polygons. Polygons can be digitized or created with image segmentation
fp_img = 'D:/BlueZan/S2/Chwaka/S2_2018_chwaka.tif' # filepath for satellite image
fp_pts = 'D:/BlueZan/Field_work/BlueZan_habitat_data_11_2024_labels.gpkg' # filepath for field data with habitat labels
#fp_pts = 'D:/BlueZan/S2/Chwaka/ZS_field_data_KMeans_sampled.gpkg' # filepath for field data with habitat labels

labelcol = 'subs_highest' # column name for habitat type labels
# set which bands to plot
bands = (1,2)


# ------------------------ #
 
def spatialJoin(pts, polys):
    gdf = gpd.read_file(pts)
    poly = gpd.read_file(polys)

    # ensure same crs
    if gdf.crs != poly.crs:
        gdf = gdf.to_crs(poly.crs)
    
    # spatial join of habitat label
    gdf_join = poly.sjoin(gdf, how='inner')

    return gdf_join

if __name__ == '__main__':
    # spatial join
    gdf = spatialJoin(fp_pts, fp_poly)
    print(gdf.columns)
    print(gdf.crs)
    print(gdf[labelcol].head())

    # encode string labels to int
    le = LabelEncoder()
    gdf['int_class'] = le.fit_transform(gdf[labelcol])+1
    print(np.unique(gdf[labelcol]))
    print(np.unique(gdf['int_class']))

    # read image
    with rio.open(fp_img) as src:
        img = src.read()
        profile = src.profile
    # masked array
    img_ma = ma.masked_where(img == 0, img)
    print('Image shape', img.shape)

    # rasterize polygon

    # create tuples of geometry, value pairs, where value is the attribute value you want to burn
    geom_value = ((geom,value) for geom, value in zip(gdf.geometry, gdf['int_class']))

    # Rasterize vector using the shape and transform of the raster
    rasterized = rasterize(geom_value,
                                    out_shape = img[0].shape,
                                    transform = profile['transform'],
                                    all_touched = True,
                                    fill = 0,   # background value
                                    dtype = 'uint16')

    # save rasterized
    outprof = profile.copy()
    outprof.update(count=1,
                   dtype='uint16')
    prefix = os.path.basename(fp_pts).split('_')[0]
    outfile = os.path.join(os.path.dirname(fp_poly), prefix + '_rasterized.tif')
    with rio.open(outfile, 'w', **outprof) as dst:
        dst.write(rasterized, 1)
    print('Saved habitat annotation raster to', outfile)

    # select clusters that have point data
    clusters = np.unique(rasterized)[1:].tolist()
    print('Classes', clusters)
    # Turn on interactive mode
    plt.ion()

    fig, ax = plt.subplots()

    ax.scatter(img_ma[bands[0]], img_ma[bands[1]], s=0.1, alpha=0.1, color='gray')    
    ax.set_xlabel('Band ' + str(bands[0] +1 ) )
    ax.set_ylabel('Band ' + str(bands[1] +1) )

    for c in clusters:
        print('Cluster', c)
        # get raster group
        group1 = np.where(rasterized == c, img, 0)
        # drop zeros
        group1 = ma.masked_where(group1 == 0, group1)
        # get mean
        mean1 = np.mean(group1[bands[0]])
        mean2 = np.mean(group1[bands[1]])
        # scatter    
        ax.scatter(group1[bands[0]], group1[bands[1]], s=0.1, alpha=0.5)
        # add centroid to the plot
        ax.scatter(mean1, mean2, marker='v', s=0.5)
        ax.text(mean1, mean2, ha='center', s=c,
                     bbox=dict(facecolor='white', alpha=0.2))
        # Draw the updated plot
        plt.draw()
        plt.pause(2)  # Pause so we can see the update

    # Turn off interactive mode and keep final plot open
    plt.ioff()
    #handles, labels = ax.get_legend_handles_labels()
    #plt.legend(handles=handles, labels=labels)
    plt.tight_layout()
    plt.show()


