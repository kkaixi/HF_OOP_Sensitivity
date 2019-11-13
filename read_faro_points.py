# -*- coding: utf-8 -*-
"""
Created on Fri May 31 16:27:46 2019
get specific faro points
@author: tangk
"""

#faro_data = import_faro_data(path)

import glob
import pandas as pd
import numpy as np
from initialize import dataset
import re
from PMG.COM.writebook_xls2 import import_faro_data

#def get_points(faro_data, number):
#    r = re.compile('[0-9.]+')
#    i = np.where(faro_data=='#' + number)[0]
#    if len(i)==0:
#        return [np.nan, np.nan, np.nan]
#    else:
#        i = i[0]
#    f = np.vectorize(lambda x: x.replace(',','.'))
#    coords = f(faro_data[i+1:i+4])
#    if all(map(r.match, coords)):
#        return coords.astype(np.float32)
#    else:
#        return [np.nan, np.nan, np.nan]
#    
#
#paths = glob.glob('P:\\2019\\19-6000\\19-6020 (FACE À FACE)\\*\\*Mesures Faro.xls')
#
##points = ['17','22','28','38','39','87','95','96','100','101','102','103','104','105',
##          '107','108','109','400','401','402','403','404',
##          '900','901','902','1011','1012','1017','1023','1024','1025','1026']
#
#dataset.get_data(['features', 'stats'])
#
#faro_points = {}
#
#r = re.compile('#[12]3-\d+')
#for path in paths:
#    faro_data = read_faro(path)
#    faro_data = np.array(list(filter(lambda x: isinstance(x, str), faro_data)))
#    points = np.unique([x[4:] for x in faro_data if r.match(x)])
#    
#    tc1 = faro_data[0]
#    tc2 = faro_data[1]
#    
#    for p in points:
#        if p not in faro_points:
#            faro_points[p] = pd.DataFrame(columns=['_'.join((p, i)) for i in ['x','y','z']], index=dataset.table.index)
#            
#        if tc1 in faro_points[p].index:
#            faro_points[p].loc[tc1] = get_points(faro_data, '13-' + p)
#        if tc2 in faro_points[p].index:
#            faro_points[p].loc[tc2] = get_points(faro_data, '23-' + p)
#
#faro_points = pd.concat(faro_points.values(), axis=1).dropna(axis=1, how='all')


#%% get data from previous year
from string import ascii_lowercase

paths = glob.glob('Q:\\2018\\18-6000\\18-6020 (FACE A FACE)\\*\\*Mesures Faro.xls')
#paths = paths + glob.glob('P:\\2019\\19-6000\\19-6020 (FACE À FACE)\\*\\*Mesures Faro.xls')
itercode = iter([i + j + k for i in ascii_lowercase for j in ascii_lowercase for k in ascii_lowercase])
faro_dictionary = {}
all_faro = {}
for path in paths:
    print(path)
    faro_data = import_faro_data(path)
    tc1 = faro_data['crash_info']['TC1']
    tc2 = faro_data['crash_info']['TC2']
    test_faro = {tc1: [], tc2: []}
    
    for key, points in faro_data['faro_points'].items():
        points_ = points
        if points is None:
            continue
        # melt dataframe if table of xyz coords
        if all([i in points.columns for i in ['ID', 'X','Y','Z','Description']]):
            points['Description'] = points['ID'].apply(lambda x: x[2:])
            points = points.drop('ID', axis=1)
            points = pd.melt(points[['Description','X','Y','Z']], id_vars='Description', value_vars=['X','Y','Z'], var_name='coord')
            values = points.pop('value')
            points = points.apply(lambda x: '_'.join(x), axis=1).rename('Description')
            points = pd.concat([points, values], axis=1) 
        elif 'Texte' in key:
            points['Description'] = points['Description'].apply(lambda x: '-'.join((key.lstrip('Mannequin ').rstrip(' Texte')[1], x)))
        else:
            continue
        
        if 'Mannequin 2' in key:
            tc = tc2
        else:
            tc = tc1
            
        # add description: code to dictionary if not in it already
        points = points.set_index('Description').squeeze().rename(tc)
        names = {name: faro_dictionary[name] if name in faro_dictionary else next(itercode) for name in points.index}
        faro_dictionary.update(names)
#        points = points.rename(faro_dictionary).rename(tc)
        test_faro[tc].append(points)
    if len(test_faro[tc1])>0:
        test_faro[tc1] = pd.concat(test_faro[tc1])
    else:
        _ = test_faro.pop(tc1)
    if len(test_faro[tc2])>0:
        test_faro[tc2] = pd.concat(test_faro[tc2])
    else:
        _ = test_faro.pop(tc2)
    all_faro.update(test_faro)        
all_faro = pd.DataFrame.from_dict(all_faro, orient='index')
        # create a dataframe of the points                 
        
        
    
#    all_faro.append(faro_data)

#for faro_data in all_faro:
#    text = faro_data['faro_points']['Mannequin 13 Texte']
#    print(faro_data['crash_info']['TC1'])
#    if text is not None: 
#        text = text.set_index('Description')
#        if text.loc['Déplacement du siège'].values[0]!='N/A':
#            print(text.loc['Déplacement du siège'].astype(int)/text.loc['Déplacement du siège total'].astype(int))
#    print('\n')
#    
#    text = faro_data['faro_points']['Mannequin 23 Texte']
#    print(faro_data['crash_info']['TC2'])
#    if text is not None: 
#        text = text.set_index('Description')
#        if text.loc['Déplacement du siège'].values[0]!='N/A':
#            print(text.loc['Déplacement du siège'].astype(int)/text.loc['Déplacement du siège total'].astype(int))
#    print('\n')
    

