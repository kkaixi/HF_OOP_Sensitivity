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

def read_faro(path, delete_cols=[]):
    faro_data = pd.read_excel(path, header=None)
    faro_data = faro_data.reset_index(drop=True).dropna(axis=0, how='all').dropna(axis=1, how='all')
    faro_data = np.concatenate(faro_data.apply(lambda x: tuple(x.dropna()), axis=1).values)
    return faro_data

def get_points(faro_data, number):
    i = np.where(faro_data=='#' + number)[0]
    if len(i)==0:
        return [np.nan, np.nan, np.nan]
    else:
        i = i[0]
    coords = faro_data[i+1:i+4]
    return coords.astype(np.int8)


paths = glob.glob('P:\\2019\\19-6000\\19-6020 (FACE À FACE)\\*\\*Mesures Faro.xls')

points = ['104','103','404','108','107','403', '96', '1023']
faro_points = pd.DataFrame(index=dataset.table.index, 
                           columns=['_'.join((p, i)) for p in points for i in ['x','y','z']],
                           dtype=np.int8)
for path in paths:
    faro_data = read_faro(path)
    tc1 = faro_data[0]
    tc2 = faro_data[1]
    for p in points:
        if tc1 in faro_points.index:
            faro_points.loc[tc1, ['_'.join((p, i)) for i in ['x','y','z']]] = get_points(faro_data, '13-' + p)
        if tc2 in faro_points.index:
            faro_points.loc[tc2, ['_'.join((p, i)) for i in ['x','y','z']]] = get_points(faro_data, '23-' + p)
