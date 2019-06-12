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

#%% get angles from faro points
from read_faro_points import faro_points
for i in ['x','y','z']:
    faro_points[['_'.join((p, i)) for p in ['104','103','404']]] = faro_points[['_'.join((p, i)) for p in ['104','103','404']]].sub(faro_points['103_'+i], axis=0)
    faro_points[['_'.join((p, i)) for p in ['108','107','403']]] = faro_points[['_'.join((p, i)) for p in ['108','107','403']]].sub(faro_points['107_'+i], axis=0)

def get_angle(u, v):
    num = np.sum(u.values*v.values, axis=1)
    denom = np.sqrt(np.sum(u.values**2, axis=1))*np.sqrt(np.sum(v.values**2, axis=1))
    ratio = num/denom
    return np.degrees(np.arccos(ratio))
    
angle_left = get_angle(faro_points[['104_x','104_z']], faro_points[['404_x','404_z']])
angle_right = get_angle(faro_points[['108_x','108_z']], faro_points[['403_x','403_z']])

dataset.features['left_leg_angle'] = angle_left
dataset.features['right_leg_angle'] = angle_right
dataset.features['left_leg_dx'] = faro_points['404_x'] - faro_points['104_x']
dataset.features['right_leg_dx'] = faro_points['403_x'] - faro_points['108_x']
dataset.features['dx_pelvis'] = faro_points['403_x'] - faro_points['404_x']
dataset.features['dz_chin_to_belt'] = (faro_points['96_z'] - faro_points['1023_x']).abs()


dataset.features['Min_13FEMRxx00HFFOZB'] = dataset.features[['Min_13FEMRLE00HFFOZB','Min_13FEMRRI00HFFOZB']].min(axis=1)
dataset.features['Max_13FEMRxx00HFFOZB'] = dataset.features[['Max_13FEMRLE00HFFOZB','Max_13FEMRRI00HFFOZB']].max(axis=1)
femur_peak = dataset.features[['Min_13FEMRxx00HFFOZB','Max_13FEMRxx00HFFOZB']].abs().idxmax(axis=1)


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
                 'Max_13NIJCIPTFHF00YX']
for ch in plot_channels:
    fig, ax = plt.subplots()
    ax = sns.barplot(x='HF_POS', y=ch, ci='sd', data=pd.concat((dataset.features[ch].abs(), dataset.table['HF_POS']), axis=1))
    ax = set_labels(ax, {'title': ch})
    plt.show()
    plt.close(fig)


#%% regression
chx_list = ['Min_13FEMRLE00HFFOZB',
            'Max_13FEMRLE00HFFOZB',
            'Min_13FEMRRI00HFFOZB',
            'Max_13FEMRRI00HFFOZB',
            'Min_13FEMRxx00HFFOZB',
            'Max_13FEMRxx00HFFOZB',
            'Peak_13FEMRxx00HFFOZB']
chy_list = ['Min_13CHST0000HFDSXB']

for chx in chx_list:
    for chy in chy_list:
        x = arrange_by_group(dataset.table.drop('TC08-109'), dataset.features[chx], 'HF_POS')
        y = arrange_by_group(dataset.table.drop('TC08-109'), dataset.features[chy], 'HF_POS')
        
        fig, ax = plt.subplots()
        ax = plot_scatter(ax, x, y)
        ax = set_labels(ax, {'xlabel': chx, 'ylabel': chy})
        
        plt.show()
        plt.close(fig)
