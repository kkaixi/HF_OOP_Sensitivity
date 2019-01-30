# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 12:51:24 2019

@author: tangk
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from PMG.read_data import initialize
from PMG.COM.get_props import *
from PMG.COM.plotfuns import *
from PMG.COM.arrange import arrange_by_group

#%%
directory = 'C:\\Users\\tangk\\Desktop\\HF OOP Sensitivity\\'
cutoff = range(100,1600)

channels = ['13HEAD0000HFACXA',
            '13NECKUP00HFFOXA',
            '13NECKUP00HFFOZA',
            '13CHST0000HFACXC',
            '13CHST0000HFDSXB',
            '13LUSP0000HFFOXA',
            '13PELV0000HFACXA',
            '11ILACLE00THFOXA',
            '11ILACRI00THFOXA',
            '13FEMRLE00HFFOZB',
            '13FEMRRI00HFFOZB']

query_list = [['HF_POS',['OOP','STD']]]

table, t, chdata = initialize(directory, 
                              channels,
                              cutoff,
                              query_list=query_list,
                              verbose=True)

#%% plot overlays (unpaired)
plot_channels = channels
for ch in plot_channels:
    x = arrange_by_group(table, chdata[ch], 'HF_POS')
    x = {'STD': x['STD'], 'OOP': x['OOP']}
    fig, ax = plt.subplots()
    ax = plot_overlay(ax, t, x)
    set_labels(ax, {'title': ch, 'legend': {}})

#%% plot overlays (paired)
plot_channels = channels

grouped = table.groupby('HF_POS')
pairs = np.intersect1d(grouped.get_group('OOP')['MODEL'],grouped.get_group('STD')['MODEL'])
for ch in plot_channels:
    for pair in pairs:
        subset = table.query('MODEL==\'' + pair + '\'')
        x = arrange_by_group(subset, chdata[ch], 'HF_POS')
        fig, ax = plt.subplots()
        ax = plot_overlay(ax, t, x)
        set_labels(ax, {'title': (pair,ch), 'legend': {}})
        plt.show()
        plt.close(fig)
    
