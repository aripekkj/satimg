# -*- coding: utf-8 -*-
"""
Spyder Editor

Gap-analysis based on unsupervised classification and field points

This is a temporary script file.
"""

import os
import pandas as pd
import rasterio as rio
import geopandas as gpd
import numpy as np
import numpy.ma as ma
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations


fp_uns = 'D:/BlueZan/S2/Chwaka/KMeans/S2_2018_15kmeans.tif' # filepath for raster clusters, e.g. from KMeans
fp_img = 'D:/BlueZan/S2/Chwaka/S2_2018_chwaka.tif' # filepath for satellite image
fp_pts = 'D:/BlueZan/S2/Chwaka/ZS_field_data_KMeans_sampled.gpkg' # filepath for field points
plot_all = False # set False if plotting only clusters where is point data

# set which bands to plot
bands = (1,2)

# -------------------------- #

def sampleRaster(raster_fp, geodataframe_fp):
    # read points
    gdf = gpd.read_file(geodataframe_fp, engine='pyogrio')
    
    # sample coords
    with rio.open(raster_fp) as src:
        meta = src.meta
        crs = src.crs.to_epsg()
        # check crs
        if gdf.crs != src.crs:
            gdf = gdf.to_crs(src.crs)
        # check geometry, explode if MultiPoint
        if gdf.geometry.geom_type.str.contains('MultiPoint').any() == True:
            sp = gdf.geometry.explode()
            coords = [(x,y) for x,y in zip(sp.geometry.x, sp.geometry.y)]
        else:
            # get point coords
            coords = [(x,y) for x,y in zip(gdf.geometry.x, gdf.geometry.y)]
        # sample
        gdf['sampled'] = [x for x in src.sample(coords)]
        
        # extract list
        gdf['clusters'] = gpd.GeoDataFrame(gdf.sampled.tolist(), index=gdf.index)
        gdf = gdf.drop('sampled', axis=1)

    return gdf


def list_difference(list1, list2):
    """
    Returns elements present in list1 but not in list2.
    """
    return list(set(list1) - set(list2))


if __name__ == '__main__':

    # sample
    pts = sampleRaster(fp_uns, fp_pts)
    # read images
    with rio.open(fp_img) as src:
        img = src.read()
    # masked array
    img_ma = ma.masked_where(img == 0, img)

    with rio.open(fp_uns) as src:
        uns = src.read()
    print('Image shape', img.shape)
    print('Uns shape', uns.shape)

    # select clusters that have point data
    clusters = np.unique(pts.clusters)
    # get clusters that do not have data points
    clusters_out = list_difference(np.unique(uns).tolist(), np.unique(pts.clusters).tolist())
    print('Clusters not covered by data points', clusters_out)

    # Turn on interactive mode
    plt.ion()

    fig, ax = plt.subplots()

    ax.scatter(img_ma[bands[0]], img_ma[bands[1]], s=0.1, alpha=0.1, color='gray')    
    ax.set_xlabel('Band ' + str(bands[0] +1 ) )
    ax.set_ylabel('Band ' + str(bands[1] +1) )

    for c in clusters:
        print('Cluster', c)
        # get raster group
        group1 = np.where(uns == c, img, 0)
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

    if plot_all == True:
        for cu in clusters_out[1:]:
            print('Cluster', cu)
            # get raster group
            group1 = np.where(uns == cu, img, 0)
            # drop zeros
            group1 = ma.masked_where(group1 == 0, group1)
            # get mean
            mean1 = np.mean(group1[bands[0]])
            mean2 = np.mean(group1[bands[1]])
            # scatter    
            ax.scatter(group1[bands[0]], group1[bands[1]], s=0.1, alpha=0.5)
            # add centroid to the plot
            ax.scatter(mean1, mean2, marker='v', s=0.5)
            ax.text(mean1, mean2, ha='center', s=cu,
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























