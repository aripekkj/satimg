# -*- coding: utf-8 -*-
"""
Created on Fri Nov 10 13:12:10 2023

Get Satellite derived bathymetry on optically shallow waters

Caballero et al. (2023) Confronting turbidity

@author: E1008409
"""

import sys
import os
os.getcwd()
os.chdir('/mnt/c/users/e1008409/.spyder-py3')
import glob
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
import matplotlib.pyplot as plt
import math
from indices import normalizedDifference

from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, RepeatedKFold, RepeatedStratifiedKFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# function to compute logarithmic band ratio from given bands
def StumpfLogBandRatio(image_array, bands):
    """
    Compute logarithmic band ratio as ln(1000*pi*band) / ln(1000*pi*band), see Stumpf et al. (2003)

    Parameters
    ----------
    image_array : np.array
        Image array
    bands : tuple or list
        Band indices 

    Returns
    -------
    logbr : np.array
        Log-ratio of image bands

    """
    logbr = (np.log(1000*np.pi*image_array[bands[0]])) / (np.log(1000*np.pi*image_array[bands[1]]))
    logbr = logbr.reshape((logbr.shape[0], logbr.shape[1]))
    
    return logbr

fp_img = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Kreikka/S2_LSxGreece_10m_20230828_v101_3035_clip_bands.tif'
fp_pts = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/Kreikka/Greece_habitat_data.gpkg'

fp_img = '/mnt/d/users/e1008409/MK/sansibar/Sansibar_WV2_25_9_2016/16SEP25073556-M2AS-056096543010_01_P001_toa_reflectance.tif'
fp_turb = '/mnt/d/users/e1008409/MK/S2/ac/S2A_MSIL1C_20160913T100022_N0204_R122_T34VEP_20160913T100023/S2A_MSI_2016_09_13_10_00_23_T34VEP_L2W_TUR_Nechad2009_665_clip.tif'
fp_pts = '/mnt/d/users/e1008409/MK/sansibar/Stumpf/Depth_transect_on_sand.shp'

# read images, points
with rio.open(fp_img) as src:
    img = src.read()
    meta = src.meta
    
with rio.open(fp_turb) as src:
    turb = src.read()
# read pts
gdf = gpd.read_file(fp_pts)

# ndvi
ndvi = normalizedDifference(img, (6,4))
# mask image
img = np.where(ndvi > 0, np.nan, img)
# log band ratio
logbr = StumpfLogBandRatio(img, (0,2))
loggr = StumpfLogBandRatio(img, (1,2))
# expand
logbr = np.expand_dims(logbr, axis=0)
# metadata
logbrmeta = meta.copy()
logbrmeta.update(count=1)
# save
outdir = os.path.join(os.path.dirname(fp_img))
if os.path.isdir(outdir) == False:
    os.mkdir(outdir)
logbrout = os.path.join(outdir, 'logbr.tif')
with rio.open(logbrout, 'w', **logbrmeta, compress='LZW') as dst:
    dst.write(logbr.astype(logbrmeta['dtype']))

# check crs
src = rio.open(fp_img)
if gdf.crs.to_epsg() != src.crs.to_epsg():
    print('Reprojecting points')
    gdf = gdf.to_crs(epsg=src.crs.to_epsg())
src.close()

# check geometry, explode if MultiPoint
if gdf.geometry.geom_type.str.contains('MultiPoint').any() == True:
    sp = gdf.geometry.explode()
    # get point coords
    coords = [(x,y) for x,y in zip(sp.x, sp.y)]
else:
    # get point coords
    coords = [(x,y) for x,y in zip(gdf.geometry.x, gdf.geometry.y)]

# open image and sample segments
src = rio.open(logbrout)
gdf['logbr'] = [x for x in src.sample(coords)]
# close dataset
src.close()
# extract list
gdf['logbr'] = gpd.GeoDataFrame(gdf.logbr.tolist(), index=gdf.index)
# drop
gdf = gdf[gdf.logbr != 0] # exclude 0 (nodata)

# select column
gdf_train = gdf[['depth', 'logbr']]
# dropnan
gdf_train = gdf_train.dropna()
gdf_train['depth'] = abs(gdf_train.depth)
# plot
fig, ax = plt.subplots()
ax.scatter(abs(gdf_train.depth), gdf_train.logbr, s=0.5, color='Blue')
ax.set_xlabel('Depth (m)')
ax.set_ylabel('Logbr (m)')
ax.plot(0,1, lw=0.7, color='Red')
ax.grid()
plt.tight_layout()
plt.show()
# remove outliers
q = np.quantile(gdf_train.logbr, [0.25, 0.75])
iqr = q[1] - q[0]
lower_lim = q[0] - 1.5*iqr
upper_lim = q[1] + 1.5*iqr
fig, ax = plt.subplots()
ax.boxplot(gdf_train.logbr)
ax.axhline(lower_lim)
ax.axhline(upper_lim)
plt.show()

gdf_train = gdf_train[(gdf_train.logbr >= lower_lim) & (gdf_train.logbr <= upper_lim)]

# plot
fig, ax = plt.subplots()
ax.scatter(gdf_train.depth, gdf_train.logbr, s=0.5, color='Blue')
ax.set_xlabel('Depth (m)')
ax.set_ylabel('Logbr (m)')
ax.plot(0,1, lw=0.7, color='Red')
ax.grid()
plt.tight_layout()
plt.show()

# Create train and test data
#################

X = np.array(gdf_train.logbr).reshape(-1,1)
y = np.array(gdf_train['depth'])

# train and test
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3,
                                                    random_state=123)
fig, ax = plt.subplots()
ax.scatter(X, y, s=0.5, color='Blue')
plt.show()

# fit
reg = LinearRegression().fit(X_train, y_train)
reg.score(X_train, y_train)

# predict
pred = reg.predict(X_test)
score = reg.score(X_test, y_test)

# rmse
mse = metrics.mean_squared_error(pred, y_test)
rmse = np.sqrt(mse)
# model
b = reg.intercept_
m = reg.coef_[0]
# plot
fig, ax = plt.subplots()
ax.scatter(pred, y_test, s=0.5, color='Blue')
ax.set_xlabel('Predicted depth (m)')
ax.set_ylabel('Field depth (m)')
ax.axline((0,0), slope=1, lw=0.5, color='black', alpha=0.7)
ax.grid()
plt.tight_layout()
plt.show()
print('R2: %.2f' % (score))
print('RMSE: %.2f' % (rmse))

# reshape logbr for predict
predict = np.reshape(logbr, (-1,1))
# replace nan
predict = np.where(np.isnan(predict), 0, predict)
# predict
prediction = reg.predict(predict)

# reshape to 2D 
predicted = prediction.reshape((1, img.shape[1], img.shape[2]))

# mask nodata
predicted = np.where(np.isnan(img[0]), np.nan, predicted)

# file out
outname = os.path.join(os.path.basename(fp_img)[:-4] + '_LinRegressor_SDB.tif')
outfile = os.path.join(outdir, outname)

# save
outmeta = meta.copy()
outmeta.update(count=1)

with rio.open(outfile, 'w', **outmeta, compress='LZW') as dst:
    dst.write(predicted.astype(rio.float32))







# initialize randomForest regressor
rf = RandomForestRegressor(n_estimators=100, max_depth=None, n_jobs=5 , bootstrap=True, oob_score=True)

# fit training data
rf.fit(X_train, y_train)

# predict
predf = pd.DataFrame()
predf['truth'] = y_test
predf['predict'] = rf.predict(X_test)
predf['res'] = predf.truth - predf.predict

# accuracy measures
mse = metrics.mean_squared_error(predf.truth, predf.predict)
rmse = np.sqrt(mse)
r2 = metrics.r2_score(predf.truth, predf.predict)

#rmse = np.sqrt(np.mean(predf['diff']**2))
#print(predf.truth.corr(predf.predict))
print(rmse)

textstr = '\n'.join((
    r'$r^2$=%.2f' % (r2 ,),
    r'rmse=%.2f' % (rmse ,)
    ))

# plot
fig, ax = plt.subplots(2,1)
ax[0].scatter(predf.truth, predf.predict, s=1)
ax[0].set_xlabel('Field depth')
ax[0].set_ylabel('Predicted depth')
ax[0].text(0.05, 0.73, textstr, transform=ax[0].transAxes)
ax[0].grid(alpha=0.4)
ax[0].axline((0,0), slope=1, color='black', alpha=0.5, lw=0.4)

ax[1].scatter(predf.truth, predf.res, s=1, color='red')
ax[1].axhline(0, ls='--', alpha=0.7, lw=0.6, color='black')
ax[1].set_ylabel('Residual')
ax[1].set_xlabel('Depth')
ax[1].grid(alpha=0.4)
#plt.suptitle(cols)
plt.tight_layout()
#plt.savefig(figpath, dpi=150)
plt.show()
 





























