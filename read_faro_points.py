# -*- coding: utf-8 -*-
"""
Created on Fri May 31 16:27:46 2019
get specific faro points
@author: tangk
"""
import glob
import pandas as pd
import numpy as np
from initialize import dataset
import re

def read_faro(path, delete_cols=[]):
    faro_data = pd.read_excel(path, header=None)
    faro_data = faro_data.reset_index(drop=True).dropna(axis=0, how='all').dropna(axis=1, how='all')
    faro_data = np.concatenate(faro_data.apply(lambda x: tuple(x.dropna()), axis=1).values)
    return faro_data

def get_points(faro_data, number):
    r = re.compile('[0-9.]+')
    i = np.where(faro_data=='#' + number)[0]
    if len(i)==0:
        return [np.nan, np.nan, np.nan]
    else:
        i = i[0]
    f = np.vectorize(lambda x: x.replace(',','.'))
    coords = f(faro_data[i+1:i+4])
    if all(map(r.match, coords)):
        return coords.astype(np.float32)
    else:
        return [np.nan, np.nan, np.nan]
    

paths = glob.glob('P:\\2019\\19-6000\\19-6020 (FACE À FACE)\\*\\*Mesures Faro.xls')

#points = ['17','22','28','38','39','87','95','96','100','101','102','103','104','105',
#          '107','108','109','400','401','402','403','404',
#          '900','901','902','1011','1012','1017','1023','1024','1025','1026']

dataset.get_data(['features', 'stats'])

faro_points = {}

r = re.compile('#[12]3-\d+')
for path in paths:
    faro_data = read_faro(path)
    faro_data = np.array(list(filter(lambda x: isinstance(x, str), faro_data)))
    points = np.unique([x[4:] for x in faro_data if r.match(x)])
    
    tc1 = faro_data[0]
    tc2 = faro_data[1]
    
    for p in points:
        if p not in faro_points:
            faro_points[p] = pd.DataFrame(columns=['_'.join((p, i)) for i in ['x','y','z']], index=dataset.table.index)
            
        if tc1 in faro_points[p].index:
            faro_points[p].loc[tc1] = get_points(faro_data, '13-' + p)
        if tc2 in faro_points[p].index:
            faro_points[p].loc[tc2] = get_points(faro_data, '23-' + p)

faro_points = pd.concat(faro_points.values(), axis=1).dropna(axis=1, how='all')
