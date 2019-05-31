# -*- coding: utf-8 -*-
"""
Created on Tue May 14 12:23:32 2019

@author: tangk
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly_express as px
from PMG.COM.plotfuns import *
from PMG.COM.arrange import arrange_by_group
from initialize import dataset

dataset.get_data(['features', 'stats'])
dataset.get_data(['timeseries'])

#%% get statistically significant differences in features
sig_ch = condense_df(pd.concat(dataset.stats.values(), axis=1))
sig_ch.columns = list(dataset.stats.keys())

sig_ch_all = condense_df(sig_ch[['std_vs_oop_wilcox','std_vs_oop_t']])
condense_df(sig_ch.drop(sig_ch_all.index)[['std1_vs_oop_wilcox']])
#%% plot confidence bands
#plot_channels = ['13HEAD0000HFACYA',
#                 '13CHST0000HFACYC',
#                 '13PELV0000HFACYA',
#                 '13NECKUP00HFFOXA',
#                 '13NECKUP00HFFOYA',
#                 '13CHST0000HFDSXB']
plot_channels = dataset.channels
for ch in plot_channels:
    fig, ax = plt.subplots()
    x1 = arrange_by_group(dataset.table.query('HF_POS<3'), dataset.timeseries[ch], 'HF_POS')
    x2 = arrange_by_group(dataset.table.query('HF_POS==3'), dataset.timeseries[ch], 'MODEL')
    ax, lines = plot_bands(ax, dataset.t, x1, legend=False)
    ax = plot_overlay(ax, dataset.t, x2, line_specs={k: {'linewidth': 0.5} for k in x2.keys()})
#    ax = plot_overlay(ax, dataset.t, x2, line_specs={k: {'linewidth': 0.5, 'color': 'b'} for k in x2.keys()})
    lines.extend(ax.lines[-len(x2):])
    ax = set_labels(ax, {'title': ch})
    ax.legend(lines, ['STD POS 1','STD POS 2'] + list(x2.keys()), bbox_to_anchor=(1,1))
    fig.savefig(dataset.directory + 'TS_' + ch + '.png', bbox_inches='tight')
    plt.show()
    plt.close(fig)

#%% plot peak values
plot_channels = ['Max_13HEAD0000HFACYA',
                 'Max_13CHST0000HFACYC',
                 'Max_13PELV0000HFACYA',
                 'Max_13NECKUP00HFFOYA',
                 'Min_13HEAD0000HFACYA',
                 'Min_13CHST0000HFACYC',
                 'Min_13PELV0000HFACYA',
                 'Min_13NECKUP00HFFOYA',
                 'Min_13CHST0000HFDSXB',
                 'Min_13NECKUP00HFFOXA',
                 'Min_13FEMRLE00HFFOZB',
                 'Min_13FEMRRI00HFFOZB',
                 'Min_13ILACLE00HFFOXA',
                 'Max_13NIJCIPTEHF00YX',
                 'Max_13NIJCIPTFHF00YX',
                 'Min_13RIBS01LEHFDSXB',
                 'Min_13RIBS01RIHFDSXB',
                 'Min_13RIBS02LEHFDSXB',
                 'Min_13RIBS02RIHFDSXB',
                 'Min_13RIBS03LEHFDSXB',
                 'Min_13RIBS03RIHFDSXB',
                 'Min_13RIBS04LEHFDSXB',
                 'Min_13RIBS04RIHFDSXB',
                 'Min_13RIBS05LEHFDSXB', 
                 'Min_13RIBS05RIHFDSXB',
                 'Min_13RIBS06LEHFDSXB',
                 'Min_13RIBS06RIHFDSXB']
for ch in plot_channels:
    fig, ax = plt.subplots()
    ax = sns.barplot(x='HF_POS', y=ch, data=pd.concat((dataset.features[ch].abs(), dataset.table['HF_POS']), axis=1))
    ax = set_labels(ax, {'title': ch})
    plt.show()
    plt.close(fig)


#%% plot peak values as a function of installation stuff
chx_list = ['SEATBACK_ANGLE',
            'SLOUCH',
            'PELVIS_ANGLE',
            'HEAD_CG_Y-SEAT_Y']
chy_list = ['Min_13HEAD0000HFACXA',
            'Min_13NECKUP00HFFOXA',
            'Max_13HEAD0000HFACYA',
            'Max_13ILACLE00HFFOXA',
            'Min_13FEMRLE00HFFOZB']

subset = dataset.table.query('YEAR==2019')
subset['SLOUCH'] = subset['SLOUCH'].replace(np.nan, 0)
dataset.features['SLOUCH'] = dataset.features['SLOUCH'].replace(np.nan, 0)
for chx in chx_list:
    for chy in chy_list:
        x = arrange_by_group(subset, dataset.features[chx], 'SLOUCH')
        y = arrange_by_group(subset, dataset.features[chy], 'SLOUCH')
        
        fig, ax = plt.subplots()
        ax = plot_scatter(ax, x, y)
        ax = set_labels(ax, {'xlabel': chx, 'ylabel': chy})
        
        plt.show()
        plt.close(fig)
        
#%% plot overlays (paired)
plot_channels = dataset.channels

grouped = dataset.table.groupby('HF_POS')
pairs = np.intersect1d(grouped.get_group('OOP')['MODEL'],grouped.get_group('STD')['MODEL'])
for ch in plot_channels:
    for pair in pairs:
        subset = dataset.table.query('MODEL==\'' + pair + '\'')
        x = arrange_by_group(subset, dataset.timeseries[ch], 'HF_POS')
        fig, ax = plt.subplots()
        ax = plot_overlay(ax, dataset.t, x)
        set_labels(ax, {'title': (pair,ch), 'legend': {}})
        plt.show()
        plt.close(fig)
    