# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 18:24:03 2025

@author: E1008409
"""

import argparse
import os
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
from rasterio.features import geometry_mask
from rasterio.windows import from_bounds
from rasterio.windows import Window
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns


def subsetGDF(geodataframe, depth_thr, veg_cover_thr, year):
    geodataframe = geodataframe[geodataframe['RUUDUN_SYVYYS'] >= depth_thr] # select shallow points

    # select by date
    start_date = pd.to_datetime(f'{year}-5-1', format="%Y-%m-%d")
    end_date = pd.to_datetime(f'{year}-8-31', format="%Y-%m-%d") 
    geodataframe = geodataframe[(geodataframe['PVM'] >= start_date) & (geodataframe['PVM'] <= end_date)]
    
    cols = ['Bolboschoenus maritimus',
             'Phragmites australis',
             'Schoenoplectus','Schoenoplectus lacustris','Schoenoplectus tabernaemontani',
             'Typha','Typha angustifolia','Typha latifolia']
    # selection on columns
    sel = geodataframe[cols + ['ID', 'PVM', 'geometry']]
    # compute coverage
    sel['vegcov'] = sel[cols].sum(axis=1, numeric_only=True)
    # set presence/absence
    sel['presence'] = np.where(sel.vegcov >= veg_cover_thr, 1, 0)
    
    print(f'Dataframe subset size {len(sel)}')
    print('Min date', geodataframe['PVM'].min())
    print('Max date', geodataframe['PVM'].max())
    
    return sel

def float_raster_to_binary_large(
    input_raster,
    output_raster,
    threshold,
    nodata_out=0):
    """
    Convert a large float raster to a binary raster using windowed processing.

    Parameters
    ----------
    input_raster : str
        Path to input float raster.
    output_raster : str
        Path to output binary raster.
    threshold : float
        Threshold value (>= threshold → 1, else 0).
    nodata_out : int, optional
        Output NoData value (default: 0).
    """

    with rio.open(input_raster) as src:
        outprofile = src.profile.copy()
        nodata_in = src.nodata
        
        # Update output profile
        outprofile.update(
            dtype=rio.uint8,
            count=1,
            nodata=nodata_out,
            compress="lzw"
        )
        
        with rio.open(output_raster, "w", **outprofile) as dst:

            # Iterate over blocks/windows
            for _, window in src.block_windows(1):
                data = src.read(1, window=window)
                # Create output
                binary = np.zeros(data.shape, dtype=np.uint8)
                binary[data >= threshold ] = 1
                
                # Handle NoData safely
                if nodata_in is not None:
                    binary[data == nodata_in] = nodata_out

                dst.write(binary, 1, window=window)

def sampleRaster(raster_fp, geodataframe):    
    # sample coords
    with rio.open(raster_fp) as src:
        profile = src.profile
        crs = src.crs.to_epsg()
        # check crs
        if geodataframe.crs != src.crs:
            geodataframe = geodataframe.to_crs(src.crs)
        # check geometry, explode if MultiPoint
        if geodataframe.geometry.geom_type.str.contains('MultiPoint').any() == True:
            sp = geodataframe.geometry.explode()
            coords = [(x,y) for x,y in zip(sp.geometry.x, sp.geometry.y)]
        else:
            # get point coords
            coords = [(x,y) for x,y in zip(geodataframe.geometry.x, geodataframe.geometry.y)]
        # column name from raster filename
        colname = os.path.basename(raster_fp).split('.')[0]
        # sample
        geodataframe[colname] = [x for x in src.sample(coords)]
        # extract sampled list to column
        geodataframe[colname] = gpd.GeoDataFrame(geodataframe[colname].tolist(), index=geodataframe.index)
    
    # select rows where sampled is not nodata
    geodataframe = geodataframe[geodataframe[colname] != profile['nodata']]
    
    return geodataframe

def sampleRasterZonal_fast(
    raster_fp,
    geodataframe,
    buffer_distance
    ):
    """
    Optimized zonal check for large rasters.
    Detects presence of 0 and/or 1 inside buffered geometries.
    """

    gdf = geodataframe.copy()
    
    with rio.open(raster_fp) as src:
        transform = src.transform
        raster_crs = src.crs
        nodata = src.nodata
    
        # Reproject vectors once
        if gdf.crs != raster_crs:
            gdf = gdf.to_crs(raster_crs)
    
        # Handle MultiPoints
        if gdf.geometry.geom_type.str.contains("MultiPoint").any():
            gdf = gdf.explode(index_parts=False)
    
        # Buffer geometries
        buffered = gdf.geometry.buffer(buffer_distance)
    
        base = os.path.splitext(os.path.basename(raster_fp))[0]
        has_0 = []
        has_1 = []
        
        for geom in buffered:
            # Bounding box → raster window
            minx, miny, maxx, maxy = geom.bounds
            window = from_bounds(
                    minx, miny, maxx, maxy,
                    transform=transform
                )
 
            # Read only needed data
            data = src.read(1,window=window)

            if data.size == 0:
                has_0.append(False)
                has_1.append(False)
                continue

            # Mask geometry within window
            mask = geometry_mask(
                [geom],
                out_shape=data.shape,
                transform=src.window_transform(window),
                all_touched=True,
                invert=True
            )

            values = data[mask]

            if nodata is not None:
                values = values[values != nodata]

            if values.size == 0:
                has_0.append(False)
                has_1.append(False)
                continue

            # Fast presence check
            found_0 = False
            found_1 = False

            if np.any(values == 0):
                found_0 = True
            if np.any(values == 1):
                found_1 = True

            has_0.append(found_0)
            has_1.append(found_1)

        gdf[f"{base}_has_0"] = has_0
        gdf[f"{base}_has_1"] = has_1

    return gdf

def plot_confusion_with_metrics(cm, output_dir, rastername, labels=None, cmap="Blues"):
    """
    Plot confusion matrix with:
        - raw counts
        - row-wise percentages
        - column-wise percentages
        - metrics table (precision, recall, F1)
    """
    
    n = cm.shape[0]

    if labels is None:
        labels = [f"Class {i}" for i in range(n)]

    # --- Compute row + column percentages ---
    row_sums = cm.sum(axis=1, keepdims=True)
    col_sums = cm.sum(axis=0, keepdims=True)

    row_pct = cm / row_sums
    col_pct = cm / col_sums

    # --- Confusion matrix annotation ---
    annot = np.empty_like(cm, dtype="object")
    for i in range(n):
        for j in range(n):
            annot[i, j] = (
                f"{int(cm[i,j]):,}\n"
                f"R:{row_pct[i,j]*100:4.1f}% \n C:{col_pct[i,j]*100:4.1f}%"
            )

    # --- Metrics ---
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1 = 2 * precision * recall / (precision + recall)

    metrics_df = pd.DataFrame({
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1
    }, index=labels).round(4)

    # --- Plot confusion matrix ---
    fig, ax = plt.subplots(2, 1, figsize=(11, 14), gridspec_kw={"height_ratios": [3, 1]})

    hm = sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        linewidths=0.5,
        linecolor="gray",
        ax=ax[0],
        annot_kws={"size": 35 / np.sqrt(len(cm))}
    )
    
    hm.set_xticklabels(hm.get_xmajorticklabels(), fontsize = 20)
    hm.set_yticklabels(hm.get_xmajorticklabels(), fontsize = 20)
    
    titlename = rastername.replace('_', ' ')
    ax[0].set_title(f"{titlename} \n Validation Confusion Matrix", fontsize=20)
    #ax[0].set_xlabel("Predicted", fontsize=13)
    #ax[0].set_ylabel("Actual", fontsize=13)

    # --- Table below the heatmap ---
    ax[1].axis("off")
    table = ax[1].table(
        cellText=metrics_df.values,
        rowLabels=metrics_df.index,
        colLabels=metrics_df.columns,
        cellLoc='center',
        loc='center'
    )
    table.scale(1, 2)
    table.auto_set_font_size(False)
    table.set_fontsize(20)

    plt.tight_layout()
    
    #output filename
    plot_out = os.path.join(output_dir, f'{rastername}_bayes_thr_{BAYES_THRESHOLD}_cm_metrics_plot.png')
    plt.savefig(plot_out, dpi=300)
    #plt.show()

    return row_pct, col_pct, metrics_df

def confusion_matrix(geodataframe, rastername, outdir):
    colname =  [c for c in geodataframe.columns if '_has_1' in c]
    # create confusion matrix
    cm = metrics.confusion_matrix(geodataframe['presence'], geodataframe[colname])
    # to dataframe
    cmdf = pd.DataFrame(cm)
    # save confusion matrix as csv
    cmdf_name = rastername + '_cm_vegcov_threshold_' + str(VEG_COVER_THR) + '.csv'
    cmdf_out = os.path.join(outdir, cmdf_name)
    cmdf.to_csv(cmdf_out, sep=';')
    
    return cm
    

    
CLI = argparse.ArgumentParser()

CLI.add_argument('Raster_dir', 
                 str, 
                 help='Main directory with subdirectories that have raster files')
CLI.add_argument('Point_fp', 
                 str, 
                 help='Input point file (.gpkg)')

args = CLI.parse_args()


rasterdir = '/mnt/d/users/e1008409/MK/Ruovikko/validointi'
fp_pts = '/mnt/d/users/e1008409/MK/Velmu-aineisto/2025/LG_AHV_aineistot_2025-05-26.gpkg'

rasterdir = args.Raster_dir
fp_pts = args.Point_fp


# parameters for functions
DEPTH_THR = -3
VEG_COVER_THR = 1
BAYES_THRESHOLD = 127

if __name__ == '__main__':
    # read points
    gdf = gpd.read_file(fp_pts, engine='pyogrio')
    
    # map raster files
    files = [f for f in glob.glob(os.path.join(rasterdir, '**/*Bayes*.tif'), recursive=True)]
    # 
    for f in files:
        break
        year = int(os.path.basename(os.path.dirname(f)))
        outdir = os.path.dirname(f)
        rastername = os.path.basename(f).split('.')[0]
        # subset gdf
        sel = subsetGDF(gdf, DEPTH_THR, VEG_COVER_THR, year)    
        if len(sel) == 0:
            print(f'No data for year {year}')
            continue
        # convert bayes to binary
        converted_out = os.path.join(outdir, f'{rastername}_thr_{BAYES_THRESHOLD}.tif')
        float_raster_to_binary_large(f, converted_out, BAYES_THRESHOLD, nodata_out=255)
        # sample raster at point locations and remove nodata rows
        sel = sampleRaster(converted_out, sel)
        # zonal
        #test = sel[sel['ID'] == 614764]
        zon = sampleRasterZonal_fast(converted_out, sel, buffer_distance=10)
        # save points
        gdf_out = os.path.join(outdir, 'velmu_vegcov_threshold_' + str(VEG_COVER_THR) + '_bolbo_phrag_schoen_typha.gpkg')
        zon.to_file(gdf_out, engine='pyogrio')
        # create confusion matrix and plot    
        cm = confusion_matrix(zon, rastername, outdir=outdir)
        plot_confusion_with_metrics(cm, outdir, rastername, labels=['Absence', 'Presence'])    
        

    
    