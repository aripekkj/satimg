# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 18:24:03 2025

@author: E1008409
"""
import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio as rio
from sklearn import metrics

def sampleRaster(raster_fp, geodataframe):    
    # sample coords
    with rio.open(raster_fp) as src:
        meta = src.meta
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
        # sample
        geodataframe['sampled'] = [x for x in src.sample(coords)]
    return geodataframe


fp = '/mnt/d/users/e1008409/MK/syke/rantakosteikko_Markus/rantakosteikko_vedessa_2024_kynnystys.tif'
fp2 = '/mnt/d/users/e1008409/MK/syke/rantakosteikko_Markus/rantakosteikko_vedessa_2024_kynnystys_focalmin3_20m.tif'
fp_pts = '/mnt/d/users/e1008409/MK/Velmu-aineisto/2024/LG_AHV_aineistot_2024-02-23.gpkg'

# read points
gdf = gpd.read_file(fp_pts, engine='pyogrio')
gdf = gdf[gdf['RUUDUN_SYVYYS'] >= -2.5]
gdf = gdf[gdf.YEAR > 2016]
gdf = sampleRaster(fp, gdf)
gdf['sampled'] = gpd.GeoDataFrame(gdf.sampled.tolist(), index=gdf.index)

# get column names where sampled == 1
sam = gdf[gdf.sampled == 1]
sam = sam[sam.columns[1:-32]]
cols = []
out1 = sam.gt(1).apply(lambda x: x.index[x].tolist(), axis=1)
all_vals = out1.explode().unique().tolist()
# remove any other than strings
for i in all_vals:
    if type(i) != str:
        all_vals.remove(i)
all_vals = sorted(all_vals)

# select
cols = ['Acer platanoides',
 'Acorus calamus', 'Agrostis', 'Agrostis stolonifera', 
 'Alisma plantago-aquatica',
 'Alisma wahlenbergii',
 'Alnus glutinosa',
 'Alnus incana',
 'Angelica archangelica',
 'Angelica sylvestris',
 'Argentina anserina',
 'Argentina anserina subsp. anserina',
 'Atriplex',
 'Atriplex prostrata',
 'Avenella flexuosa',
 'Bolboschoenus maritimus',
 'Butomus umbellatus',
 'Calamagrostis',
 'Calamagrostis neglecta',
 'Calla palustris',
 'Calliergon cordifolium',
 'Calliergon megalophyllum',
 'Callitriche',
 'Callitriche cophocarpa',
 'Callitriche hamulata',
 'Callitriche hermaphroditica',
 'Callitriche palustris',
 'Caltha palustris',
 'Carex',
 'Carex acuta',
 'Carex aquatilis',
 'Carex cespitosa',
 'Carex diandra',
 'Carex lasiocarpa',
 'Carex nigra',
 'Carex pseudocyperus',
 'Carex rostrata',
 'Carex vesicaria',
 'Chamaenerion angustifolium',
 'Chenopodium',
 'Cicuta virosa',
 'Cirsium vulgare',
 'Comarum palustre',
 'Convallaria majalis',
 'Convolvulus sepium',
 'Cyperaceae',
 'Deschampsia',
 'Deschampsia bottnica',
 'Deschampsia cespitosa',
 'Drepanocladus',
 'Drepanocladus aduncus',
 'Drepanocladus sordidus',
 'Elatine',
 'Elatine hydropiper',
 'Elatine orthosperma',
 'Elatine triandra',
 'Eleocharis',
 'Eleocharis acicularis',
 'Eleocharis mamillata',
 'Eleocharis palustris',
 'Eleocharis palustris subsp. palustris var. lindbergii',
 'Eleocharis parvula',
 'Eleocharis quinqueflora',
 'Eleocharis uniglumis',
 'Elodea canadensis',
 'Elytrigia repens',
 'Epilobium',
 'Equisetum fluviatile',
 'Euphorbia palustris',
 'Festuca rubra',
 'Filipendula ulmaria',
 'Galium',
 'Galium palustre',
 'Galium palustre subsp. palustre',
 'Glyceria fluitans',
 'Glyceria maxima',
 'Halerpestes cymbalaria',
 'Hieracium',
 'Hippophaë rhamnoides',
 'Hippuris',
 'Hippuris lanceolata',
 'Hippuris tetraphylla',
 'Hippuris vulgaris',
 'Honckenya peploides',
 'Hydrocharis morsus-ranae',
 'Impatiens glandulifera',
 'Iris pseudacorus',
 'Juncus',
 'Juncus alpinoarticulatus',
 'Juncus articulatus',
 'Juncus balticus',
 'Juncus bufonius',
 'Juncus bulbosus',
 'Juncus filiformis',
 'Juncus gerardi',
 'Lathyrus palustris',
 'Lemna minor',
 'Lemna trisulca',
 'Leontodon',
 'Leucanthemum vulgare',
 'Limosella aquatica',
 'Lycopus europaeus',
 'Lysimachia',
 'Lysimachia maritima',
 'Lysimachia thyrsiflora',
 'Lysimachia vulgaris',
 'Lythrum salicaria',
 'Mentha arvensis',
 'Menyanthes trifoliata',
 'Myosotis',
 'Myosotis scorpioides',
 'Myrica gale',
 'Nuphar lutea',
 'Nuphar pumila',
 'Nymphaea alba',
 'Nymphaea candida',
 'Nymphaea tetragona',
 'Nymphaeaceae',
 'Odontites',
 'Oenanthe aquatica',
 'Oxyrrhynchium speciosum',
 'Parnassia palustris',
 'Pedicularis palustris',
 'Persicaria amphibia',
 'Persicaria foliosa',
 'Persicaria hydropiper',
 'Persicaria lapathifolia subsp. lapathifolia',
 'Persicaria lapathifolia subsp. pallida',
 'Persicaria maculosa',
 'Peucedanum palustre',
 'Phalaroides arundinacea',
 'Phragmites australis',
 'Plantago major',
 'Plantago maritima',
 'Poa',
 'Poa trivialis',
 'Poaceae',
 'Potentilla erecta',
 'Ranunculus',
 'Ranunculus acris',
 'Ranunculus baudotii',
 'Ranunculus circinatus',
 'Ranunculus confervoides',
 'Ranunculus repens',
 'Ranunculus reptans',
 'Ranunculus schmalhausenii',
 'Ribes nigrum',
 'Rorippa palustris',
 'Rosaceae',
 'Rubus',
 'Rumex',
 'Rumex aquaticus',
 'Rumex longifolius',
 'Sagina procumbens',
 'Sagittaria',
 'Sagittaria natans',
 'Sagittaria sagittifolia',
 'Sagittaria ×lunata',
 'Salix',
 'Salix phylicifolia',
 'Sarmentypnum procerum',
 'Sarmentypnum trichophyllum',
 'Schoenoplectus',
 'Schoenoplectus lacustris',
 'Schoenoplectus tabernaemontani',
 'Scirpus sylvaticus',
 'Scorpidium',
 'Scorzoneroides autumnalis',
 'Scutellaria galericulata',
 'Sedum acre',
 'Senecio squalidus',
 'Solanum dulcamara',
 'Sparganium',
 'Sparganium angustifolium',
 'Sparganium emersum',
 'Sparganium erectum',
 'Sparganium gramineum',
 'Sparganium natans',
 'Spergularia marina',
 'Sphagnum platyphyllum',
 'Spirodela polyrhiza',
 'Stratiotes aloides',
 'Symphytum',
 'Tanacetum vulgare',
 'Trifolium',
 'Triglochin maritima',
 'Triglochin palustris',
 'Tripleurospermum maritimum',
 'Tussilago farfara',
  'Typha',
  'Typha angustifolia',
  'Typha latifolia',
  'Urtica',
  'Urtica dioica',
  'Valeriana excelsa',
  'Valeriana excelsa subsp. salina',
  'Vicia cracca',
  'Viola',
  'Warnstorfia fluitans',
  ]

cols2 = ['Bolboschoenus maritimus','Phragmites australis','Schoenoplectus','Schoenoplectus lacustris','Schoenoplectus tabernaemontani','Typha','Typha angustifolia','Typha latifolia']
veg_threshold = 10
sel = gdf[cols2 + ['geometry']]
# compute coverage
sel['vegcov'] = sel[cols2].sum(axis=1, numeric_only=True)
# set presence/absence
sel['presence'] = np.where(sel.vegcov >= veg_threshold, 1, 0)
# sample raster
sel = sampleRaster(fp, sel)
sel['sampled'] = gpd.GeoDataFrame(sel.sampled.tolist(), index=sel.index)
# difference
sel['same'] = sel.presence == sel.sampled
# save points
gdf_out = os.path.join(os.path.dirname(fp), 'velmu_vegcov_threshold_' + str(veg_threshold) + '_phrag_schoen_typha.gpkg')
sel.to_file(gdf_out, engine='pyogrio')

# create confusion matrix
cm = metrics.confusion_matrix(sel['presence'], sel['sampled'])
# compute row and col sums
total = cm.sum(axis=0)
rowtotal = cm.sum(axis=1)
rowtotal = np.expand_dims(rowtotal, axis=0).T #expand dims and transpose
rowtotal_sum = np.array(rowtotal.sum()) 
rowtotal = np.vstack([rowtotal, rowtotal_sum]) # stack row sum
# create cm DataFrame
cmdf = np.vstack([cm,total]) # vertical stack
cmdf = np.hstack((cmdf, rowtotal)) # horizontal stack
cm_cols = ['False', 'True']
cm_cols.append('Total')
cmdf = pd.DataFrame(cmdf, index=cm_cols,
                    columns = cm_cols)
# save confusion matrix dataframe as csv
cmdf_name = 'merirantakosteikko_10m_cm_vegcov_threshold_' + str(veg_threshold) + '.csv'
cmdf_out = os.path.join(os.path.dirname(fp), cmdf_name)
cmdf.to_csv(cmdf_out, sep=';')
# print
print(pd.crosstab(sel.presence, sel.sampled, margins=True))
# compute common accuracy metrics
o_accuracy = np.sum(cm.diagonal()) / np.sum(cm.sum(axis=0))
p_accuracy = cm.diagonal() / cm.sum(axis=0) # producer's accuracy
u_accuracy = cm.diagonal() / cm.sum(axis=1) # user's accuracy
print(' Overall accuracy %.2f' % (o_accuracy))
print(' Users accuracy', u_accuracy)
print(' Producers accuracy', p_accuracy)    





