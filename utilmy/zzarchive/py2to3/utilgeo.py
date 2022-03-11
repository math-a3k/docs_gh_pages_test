'''
utilitaire to process map data

Postgres GIS
geopandas
mapleaf

'''
# %autoreload
import os, sys
DIRCWD=  'D:/_devs/Python01/project27/' if sys.platform.find('win')> -1   else  '/home/ubuntu/notebook/' if os.environ['HOME'].find('ubuntu')>-1 else '/media/sf_project27/'
os.chdir(DIRCWD); sys.path.append(DIRCWD + '/aapackage'); # sys.path.append(DIRCWD + '/linux/aapackage')
execfile( DIRCWD + '/aapackage/allmodule.py')
import util,  numpy as np, gc
# util.a_module_generatedoc("blaze")   util.a_module_codesample('blaze')    util.a_help()    util.a_module_codesample('pandas')
#----------------------------------------------------------------------------------------
import  pandas as pd, sqlalchemy as sql, dask.dataframe as dd, dask, datanalysis as da
from attrdict import AttrDict as dict2
from collections import defaultdict
import arrow
#####################################################################################


def df_to_geojson(df, col_properties, lat='latitude', lon='longitude'):
    """function df_to_geojson
    Args:
        df:   
        col_properties:   
        lat:   
        lon:   
    Returns:
        
    """
    geojson = {'type':'FeatureCollection', 'features':[]}
    
    for _, row in df.iterrows():
        feature = {'type':'Feature',
                   'properties':{},
                   'geometry':{'type':'Point', 'coordinates':[]}}
        feature['geometry']['coordinates'] = [row[lon],row[lat]]
        
        for prop in properties:
            feature['properties'][prop] = row[prop]
        geojson['features'].append(feature)
    return geojson
    




'''
pip install django-pandas



'''




    
    
    
    



