# -*- coding: utf-8 -*-
"""
Created on Wed May  8 10:10:04 2024

Spectral plot of classes by depth


@author: E1008409
"""

import os
import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import math

fp = '/mnt/d/users/e1008409/MK/OBAMA-NEXT/sdm_vs_rs/BlackSea/Black_Sea_habitat_data_3035_sampled.gpkg'
outdir = os.path.dirname(fp)
plot_out = os.path.join(outdir, 'SpectralPlot_BlackSea.png')

gdf = gpd.read_file(fp, engine='pyogrio')
# define depth column
gdf['depth'] = abs(gdf.depth)

# create dict of names for column renaming
bandcols = gdf.columns[-11:-1]
bandnums = ['Band1', 'Band2', 'Band3', 'Band4', 'Band5', 'Band6', 'Band7', 'Band8', 'Band9', 'Band10']
newcols = dict(zip(bandcols, bandnums))
# rename
gdf = gdf.rename(columns=(newcols))
# group by class and compute average spectra
# select cols
df_s = gdf[['hab_class', 'Band1', 'Band2', 'Band3', 'Band4', 'Band5', 'Band6', 'Band7', 'Band8', 'Band9', 'Band10']]
# groupby class
dg = df_s.groupby('hab_class').mean()
# select band columns
#dg = dg[cols]

# spectral plot by depth
dzones = [] 
n = 0
while n < math.ceil(gdf.depth.max()):
    dzone = (n, n+1)
    dzones.append(dzone)
    n += 1

bands = ['Band1', 'Band2', 'Band3', 'Band4', 'Band5', 'Band6', 'Band7', 'Band8']
wavels = [443, 492.4, 559.8, 664.6, 704.1, 740.5, 782.8, 832.8]#, 864.7]
# deep water class
dw = df_s[df_s['new_class'] == 4].groupby('new_class').mean()
dw = dw[bands]
# number of rows,cols to create a grid that fits all depth zones
pr = 4
pc = 3
# create indices for plotting
rows = np.arange(0,pr)    
cols = np.arange(0,pc)
mgrid = np.meshgrid(cols, rows)
axs = list(zip(mgrid[1].reshape(-1), mgrid[0].reshape(-1)))

# create spectral plot of each depth zone
fig, ax = plt.subplots(pr,pc, figsize=(12,8))

for i in dzones:
    # make selection
    dfsel = gdf[(gdf.depth > i[0]) & (gdf.depth <= i[1])]
    
    # drop class 4
    #dfsel = dfsel[dfsel.classes < 4]
    # group
    dfsel_group = dfsel.groupby('hab_class').mean(numeric_only=True)    
    # drop 0
    if 0 in dfsel_group.index:
        dfsel_group = dfsel_group.drop(0)
    # select band columns
    dfsel_group = dfsel_group[bands]
    
    # reset index
    #dfsel_group = dfsel_group.reset_index()
    
    # row, col params for multiplot
#    if i[0] < pr:
#        r = 0
#        c = i[0] -1
#    elif i[0] >= pc:
#        r = 1
#        c = i[0] - pc -1
    # get indices for plotting
    r = axs[dzones.index(i)][0]
    c = axs[dzones.index(i)][1]
    # skip if no points within depth zone
    if len(dfsel) == 0:
        continue
    # select classes within depth zone
    cl_in_dzone = list(np.unique(dfsel_group.index))
    print(cl_in_dzone)
    colors = ['#888f29', 'brown', 'yellow', 'green']#, '#888f29', 'blue']
#    labs = ['Seagrass', 'Sparse', 'Bare', 'Deep water']

    for j in cl_in_dzone:
        # get index
        base = dfsel_group.index.get_loc(j)
        # plot
        ax[r,c].plot(dfsel_group.loc[j], linewidth=0.7, color=colors[base], label=cl_in_dzone[base])
    #    ax[r,c].plot(dfsel_group.loc[2], linewidth=0.7, color=, label=)
    #    ax[r,c].plot(dfsel_group.loc[3], linewidth=0.7, color=, label=)
    #ax[r,c].plot(dw.loc[4], linewidth=0.7, linestyle='--', color='#0034CA', label='Deep water') 
    ax[r,c].set_title(str(i[0]) + '-' + str(i[1]) + ' meters', fontsize=10) 
    ax[r,c].grid(True, color='#B0B0B0')
    ax[r,c].set_xticks(list(np.arange(len(bands))))     
    ax[r,c].set_xticklabels(wavels, rotation=45)
    ax[r,c].tick_params(axis='x', which='major', pad=-1, labelsize=8)
    ax[r,c].tick_params(axis='y', which='major', pad=-1, labelsize=8)
    ax[r,c].set_facecolor(color='#D9D9D9')
"""    
# select deep water
dfsel_deep = df[df.field_depth > 5].groupby('new_class').mean(numeric_only=True)
dfsel_deep = dfsel_deep[cols]

# add to plot
ax[2,1].plot(dfsel_deep.loc[1], linewidth=0.7, color='#443726', label='Bare bottom or \nsparse SAV')
ax[2,1].plot(dfsel_deep.loc[2], linewidth=0.7, color='green', label='Mixed SAV')
ax[2,1].plot(dfsel_deep.loc[3], linewidth=0.7, color='#A27C1F', label='Dense Zostera')
ax[2,1].plot(dw.loc[4], linewidth=0.7, linestyle='--', color='#0034CA', label='Deep water')    
ax[2,1].set_title('> 5 meters', fontsize=10)
ax[2,1].grid(True, color='#B0B0B0')
ax[2,1].set_xticks(list(np.arange(len(bands))))     
ax[2,1].set_xticklabels(bands)
ax[2,1].tick_params(axis='x', which='major', pad=-1, labelsize=8)
ax[2,1].tick_params(axis='y', which='major', pad=-1, labelsize=8)

ax[2,1].set_facecolor(color='#D9D9D9')
"""
# set labels
#plt.setp(ax[-1, :], xlabel='Wavelenth (nm)')
#plt.setp(ax[1, 0], ylabel='Remote sensing reflectance $(sr^{-1})$')
fig.supxlabel('Wavelength (nm)')
fig.supylabel('Remote sensing reflectance $(sr^{-1})$')
plt.suptitle('Average spectra at depths (m)')

# legend
handles, labels = ax[0,1].get_legend_handles_labels()
#lgd = fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(1.12, 0.93))
legend = ax[0,2].legend(handles, labels, ncol=1, frameon=True,
                    loc='upper right', bbox_to_anchor=(1.6,1.1))
plt.tight_layout()
plt.savefig(plot_out, dpi=300)























