# -*- coding: utf-8 -*-
"""
Created on Thu Nov 25 15:36:22 2021

Functions to convert WorldView satellite image DN to TOA reflectance

@author: Ari-Pekka Jokinen
"""

import pandas as pd
import rasterio as rio
import numpy as np
import glob
import os
import re
import datetime
import math

def getFile(image_directory, file_extension):
    for f in glob.glob(os.path.join(image_directory, '*.' + file_extension)):
        return f

def readIMDtoDict(imd_file):
    lines = open(imd_file).readlines()
    
    map_to_dict = False
    imd = dict()
    mapping_dict = dict()
    
    for line in lines:
        print(line)
        if line == 'END;':
            break
        # check if BEGIN_GROUP in string
        if 'BEGIN_GROUP' in line:
            key = 'name'
            value = re.search('=   (.*);', line) 
            map_to_dict = True
            mapping_dict[key] = value
            print('Map to Dictionary: ', map_to_dict)
            continue
        if 'END_GROUP' in line:
            map_to_dict = False
            key = re.findall('=\s(.*)\n', line)[0]
            value = mapping_dict
            # add dictionary as key, value
            imd[key] = value
            # clear dict
            mapping_dict = dict()
            print('Map to Dictionary: ', map_to_dict)
            continue
        if map_to_dict == True:
            # split string
            split = line.split(" = ")
            if len(split) == 1:
                map_to_list = True
            else:
                map_to_list = False
                
            if map_to_list == True:
                try:
                    value = float(re.findall('\t(.*),', line))
                except:
                    value = re.findall('\t(.*),', line)
                mapping_dict[key].append(value)
                continue
            else:
                # get key, value
                key = re.findall('\t(.*)\s=',line)[0] #split[0]
                if key == 'datumOffset' or key == 'mapProjParam':
                    mapping_dict[key] = []
                    continue
                try: 
                    value = float(re.findall("=(.*);", line)[0])
                except:    
        #            search_value = re.findall(r"[-+]?\d*\.\d+|\d+", line)
                    value = re.findall("=(.*);", line)[0]
            # add to dictionary
            mapping_dict[key] = value
            continue
        else:
            # split string
            split = line.split(" = ")
            # assign key
            key = re.findall('(.*)\s=',line)[0] #split[0]
            # get value
            search_value = re.search('"(.*)"', split[1])
            if search_value == None:
                search_value = re.search('=(.*);', line)
                value = search_value.group(1)
            else:    
                value = search_value.group(1)
            # add key, value to dictionary
            imd[key] = value
    
    return imd

# functions to convert WorldView DN to TOA radiance
#𝐿 = 𝐺𝐴𝐼𝑁 ∗ 𝐷𝑁 ∗ (𝑎𝑏𝑠𝑐𝑎𝑙𝑓𝑎𝑐𝑡𝑜𝑟 / 𝑒𝑓𝑓𝑒𝑐𝑡𝑖𝑣𝑒𝑏𝑎𝑛𝑑𝑤𝑖𝑡ℎ) + 𝑂𝐹𝐹𝑆𝐸𝑇

def getAbsCalEffBandWidth(imd_dict, bandkey):
    abscalfact = imd_dict[bandkey]['absCalFactor']
    effectiveBw = imd_dict[bandkey]['effectiveBandwidth']
    
    return abscalfact, effectiveBw

def computeTOA(bandarray, gain, offset, abscalfactor, effectivebandwith):
    """
    Computes TOA radiance

    Parameters
    ----------
    bandarray : TYPE
        DESCRIPTION.
    gain : TYPE
        DESCRIPTION.
    offset : TYPE
        DESCRIPTION.
    abscalfactor : TYPE
        DESCRIPTION.
    effectivebandwith : TYPE
        DESCRIPTION.

    Returns
    -------
    band_toa : TYPE
        DESCRIPTION.

    """
    band_toa = gain * bandarray * (abscalfactor / effectivebandwith) + offset
    return band_toa

def getEarthSunDist(imd_dict):
    """
    Calculate Earth-Sun distance

    Parameters
    ----------
    imd_dict : TYPE
        DESCRIPTION.

    Returns
    -------
    None.

    """
    # get acquisition time from IMD
    acq_time = imd_dict['MAP_PROJECTED_PRODUCT']['earliestAcqTime']
    # convert to datetime object
    dtime = datetime.datetime.strptime(acq_time, ' %Y-%m-%dT%H:%M:%S.%f%z')
    year, month, day = dtime.year, dtime.month, dtime.day # extract year,month,day
    # universal time for hh:mm:ss
    ut = dtime.hour + (dtime.minute/60) + (dtime.second/3600)
    # compute Julian day
    if month == 1 or month == 2:
        year = year-1
        month = month+12
    A = int(year/100)
    B = 2-A+int(A/4)
    JD = int(365.25*(year+4716))+int(30.6001*(month+1))+day+(ut/24)+B-1524.5
    # compute Earth-Sun distance
    D = JD - 2451545.0
    g = 357.529 + 0.98560028 * D
    g_rad = math.radians(g)
    d_ES = 1.00014-0.01671*math.cos(g_rad) - 0.00014*math.cos(2*g)
    
    # check that value is between 0.983 - 1.017 (WV3 Radiometric documentation)
    if (d_ES > 0.983) & (d_ES < 1.017) == False:
        raise ValueError('Value not within valid range')
        
    return d_ES

def getSolarZenithAngle(imd_dict):
    """
    return mean solar zenith angle from IMD file

    Parameters
    ----------
    imd_dict : TYPE
        DESCRIPTION.

    Returns
    -------
    Solar zenith angle in radians

    """
    mean_sun_el = imd_dict['IMAGE_1']['meanSunEl'] 
    sol_zenith_angle = math.radians(90 - mean_sun_el)
    
    return sol_zenith_angle
        
def convertToTOA(image_directory, imd_dict, band_gain_offset_df, ssi, ssi_method='Thuillier'):
    """
    Convert WV image to Top-of-Atmosphere reflectance product

    Parameters
    ----------
    image_directory : 'str'
        Input image directory with .IMD file
    imd_dict : Dictionary
        Dictionary from .IMD file
    band_gain_offset_df : DataFrame
        DataFrame of GAIN and OFFSET values for image bands
    ssi : Dictionary
        Solar spectral irradiance
    ssi_method : 'str', optional
        Select method used in solar spectral irradiance computation. Possible values are 'Thuillier', 'ChKur' and 'WRC'.
        The default is 'Thuillier'.

    Returns
    -------
    toa : nd array
        n dimensional array of top of atmosphere reflectance values.
    meta : Dictionary
        Image metadata

    """
    # image file
    with rio.open(getFile(image_directory, 'TIF')) as src:
        img = src.read()
        meta = src.meta
    print(img.shape)
    # create mask
    img_masked = np.ma.masked_values(img, 0)
    
    # empty array to store result
    toa = np.empty(shape=img.shape)
    
    # convert each band
    for i in np.arange(0, len(img)):
        # select df row
        gain, offset = band_gain_offset_df.iloc[i][1], band_gain_offset_df.iloc[i][2]
        # get abscalfactor and effective bandwith from IMD dictionary
        abscalfactor, effectiveBw = getAbsCalEffBandWidth(imd_dict, band_gain_offset_df.iloc[i][3])
        
        # select band
        band_sel = img_masked[i]
        
        # compute TOA radiance
        band_toa = computeTOA(band_sel, gain, offset, abscalfactor, effectiveBw)
    
        # get parameters for TOA reflectance conversion
        ES_dist = getEarthSunDist(imd_dict)
        if imd_dict['bandId'] == 'Multi': 
            band_ssi = ssi[ssi_method][i+1] # begin indexing at +1 to exclude PAN 
        if imd_dict['bandId'] == 'P':
            band_ssi = ssi[ssi_method][i]         
        sol_z = getSolarZenithAngle(imd_dict)
        # convert to TOA reflectance
        toa_ref = (band_toa*ES_dist**2*math.pi) / (band_ssi*math.cos(sol_z))
    
        # assign to empty array
        toa[i] = toa_ref
    
    return toa, meta


###### Set filepaths #########

img_dir = 'D:\\Users\\E1008409\\MK\\worldview\\ketokari\\013396930020_01_P001_MUL'
imgpan_dir = 'D:\\Users\\E1008409\\MK\\worldview\\ketokari\\013396930020_01_P001_PAN'

# get IMD file
imd_file = getFile(img_dir, 'IMD')
imd_pan = getFile(imgpan_dir, 'IMD')

# IMD to dict
imd = readIMDtoDict(imd_file)
imdpan = readIMDtoDict(imd_pan)

# gain and offset values
wv3_gain = [0.923, 0.863, 0.905, 0.907, 0.938, 0.945, 0.980, 0.982, 0.954, 1.160, 1.184, 1.173, 1.187, 1.286, 1.336, 1.340, 1.392]
wv3_offset = [-1.700, -7.154, -4.189, -3.287, -1.816, -1.350, -2.617, -3.752, -1.507, -4.479, -2.248, -1.806, -1.507, -0.622, -0.605, -0.423, -0.302]
wv3_bandsimd = ['BAND_P', 'BAND_C', 'BAND_B', 'BAND_G', 'BAND_Y', 'BAND_R', 'BAND_RE', 'BAND_N', 'BAND_N2']
wv3_bands = ['PAN', 'COASTAL', 'BLUE', 'GREEN', 'YELLOW', 'RED', 'REDEDGE', 'NIR1', 'NIR2', 'SWIR1', 'SWIR2',
             'SWIR3', 'SWIR4', 'SWIR5', 'SWIR6', 'SWIR7', 'SWIR8']
# WV3 band averaged solar spectral irradiance
wv3_ssi = {'Thuillier': [1574.41, 1757.89, 2004.61, 1830.18, 1712.07, 1535.33, 1348.08, 1055.94, 858.77, 479.019, 263.797, 225.283, 197.552, 90.4178, 85.0642, 76.9507, 68.0988],
           'ChKur': [1578.28, 1743.9, 1974.53, 1858.1, 1748.87, 1550.58, 1303.4, 1063.92, 858.632, 478.873, 257.55, 221.448, 191.583, 86.5651, 82.0035, 74.7411, 66.3906],
           'WRC': [1583.58, 1743.81, 1971.48, 1856.26, 1749.4, 1555.11, 1343.95, 1071.98, 863.296, 494.595, 261.494, 230.518, 196.766, 80.365, 74.7211, 69.043, 59.8224]
           }

# to dataframe
wv_godf = pd.DataFrame({'bands':wv3_bands,
                        'gain': wv3_gain,
                        'offset': wv3_offset}, index=wv3_bands)

# subset dataframe to selected bands
pan_ms_godf = wv_godf[0:9]
# add imd band names
pan_ms_godf['imd'] = wv3_bandsimd

# multispectral bands
ms_godf = pan_ms_godf[1:9]
# panchromatic 
pan_godf = pan_ms_godf[0:1]

# convert to TOA
img_toa, meta = convertToTOA(img_dir, imd, ms_godf, wv3_ssi)
panimg_toa, panmeta = convertToTOA(imgpan_dir, imdpan, pan_godf, wv3_ssi)

# MULTISPECTRAL output filename and save to file
inputname = os.path.basename(getFile(img_dir, 'TIF'))
outdir = os.path.join(os.path.dirname(img_dir))
outname = os.path.join(outdir, inputname[:-4] + '_toa_reflectance.tif')

# update nodata in metadata
outmeta = meta
outmeta.update(dtype='float32',
               nodata=0)
# save
with rio.open(outname, 'w', **outmeta) as dst:
    dst.write(img_toa.astype(rio.float32))


# PANCHROMATIC output filename and save to file
inputname = os.path.basename(getFile(imgpan_dir, 'TIF'))
outdir = os.path.join(os.path.dirname(imgpan_dir))
outname = os.path.join(outdir, inputname[:-4] + '_toa_reflectance.tif')

panmeta.update(dtype='float32',
               nodata=0)
# save
with rio.open(outname, 'w', **panmeta) as dst:
    dst.write(panimg_toa.astype(rio.float32))






