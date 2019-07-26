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
from PMG.COM.helper import r2, corr, rho

dataset.get_data(['features', 'stats'])


#dataset.get_data(['timeseries'])

#%% get angles from faro points
from read_faro_points import faro_points
faro_points.at['TC18-214',['101_y','102_y']] = np.nan

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
x_var = 'ABDOMEN_BELT_PENETRATION'
plot_channels = condense_df(dataset.stats['belt_pen_t']).index
#plot_channels = ['Max_13HEAD0000HFACYA',
#                 'Max_13CHST0000HFACYC',
#                 'Max_13PELV0000HFACYA',
#                 'Max_13NECKUP00HFFOYA',
#                 'Min_13HEAD0000HFACYA',
#                 'Min_13CHST0000HFACYC',
#                 'Min_13PELV0000HFACYA',
#                 'Min_13NECKUP00HFFOYA',
#                 'Min_13CHST0000HFDSXB',
#                 'Min_13NECKUP00HFFOXA',
#                 'Min_13FEMRLE00HFFOZB',
#                 'Min_13FEMRRI00HFFOZB',
#                 'Min_13ILACLE00HFFOXA',
#                 'Max_13NIJCIPTEHF00YX',
#                 'Max_13NIJCIPTFHF00YX']
for ch in plot_channels:
    fig, ax = plt.subplots()
    ax = sns.barplot(x=x_var, y=ch, ci='sd', data=pd.concat((dataset.features[ch].abs(), dataset.table), axis=1))
    ax = set_labels(ax, {'title': ch})
    plt.show()
    plt.close(fig)


#%% get subset
    
#subset = dataset.table.query('SEATBACK_ANGLE>=0 and SEATBACK_ANGLE <22').index.drop([]) 
subset = dataset.table.query('HF_POS==2').index.drop([])     

chy = 'Min_13CHST0000HFDSXB'
#chy = 'Min_13NECKUP00HFFOXA'

X = faro_points.loc[subset]
X['belt_dz'] = X['1023_z'] - X['1024_z']
X['belt_dx'] = X['1023_x'] - X['1024_x']
X['shoulder_belt_lap_belt_dz'] = X['1024_z'] - X['1025_z']
X['shoulder_belt_lap_belt_dx'] = X['1024_x'] - X['1025_x']
X['seat_dz'] = X['901_z'] - X['900_z']
X['seat_dx'] = X['901_x'] - X['900_x']
X['pelvis_dx'] = X['403_x'] - X['404_x']
X['pelvis_dz'] = X['403_z'] - X['404_z']
X['shoulder_dx'] = X['100_x'] - X['400_x']
X['shoulder_dz'] = X['100_z'] - X['400_z']
X['pelvis_angle'] = dataset.table.loc[X.index, 'PELVIS_ANGLE']
X['seatback_angle'] = dataset.table.loc[X.index, 'SEATBACK_ANGLE']
X['chin_belt_dist'] = dataset.table.loc[X.index, 'CHIN_BELT_DISTANCE']
X['knee_dy'] = X['107_y'] - X['103_y']
X['ankle_dy'] = X['108_y'] - X['104_y']
X['heel_dy'] = X['109_y'] - X['105_y']
X['left_knee_pelvis_dz'] = X['103_z'] - X['404_z']
X['left_knee_pelvis_dx'] = X['103_x'] - X['404_x']
X['left_knee_pelvis_dy'] = X['103_y'] - X['404_y']
X['right_knee_pelvis_dz'] = X['403_z'] - X['107_z']
X['right_knee_pelvis_dy'] = X['403_y'] - X['107_y']
X['right_knee_pelvis_dx'] = X['403_x'] - X['107_x']
X['left_thigh_angle_Y'] = X['left_knee_pelvis_dz']/X['left_knee_pelvis_dx']
X['right_thigh_angle_Y'] = X['right_knee_pelvis_dz']/X['right_knee_pelvis_dx']
X['left_thigh_angle_Z'] = X['left_knee_pelvis_dy']/X['left_knee_pelvis_dx']
X['right_thigh_angle_Z'] = X['right_knee_pelvis_dy']/X['right_knee_pelvis_dx']
X['veh_cg_peak'] = dataset.features.loc[X.index, 'Min_10CVEHCG0000ACXD']
X['belt_angle_side'] = dataset.table.loc[X.index, 'BELT_ANGLE_SIDE']
X['belt_angle_front'] = dataset.table.loc[X.index, 'BELT_ANGLE_FRONT']
X['seat_track_pos'] = X['1017_x'] - X['1012_x']
X['chin_top_belt_dz'] = X['96_z'] - X['1023_z']
X['chin_bottom_belt_dz'] = X['96_z'] - X['1024_z']
X['right_shoulder_elbow_dz'] = X['100_z'] - X['101_z']
X['right_shoulder_elbow_dy'] = X['100_y'] - X['101_y']
X['right_shoulder_elbow_dx'] = X['100_x'] - X['101_x']
X['chest_angle'] = X['belt_dx']/X['belt_dz']
X['right_upper_arm_angle_Y'] = X['right_shoulder_elbow_dx']/X['right_shoulder_elbow_dz']
X['right_upper_arm_angle_Z'] = X['right_shoulder_elbow_dy']/X['right_shoulder_elbow_dx']
X['right_upper_arm_angle_X'] = X['right_shoulder_elbow_dy']/X['right_shoulder_elbow_dz']
X['pretensioner_time'] = dataset.table.loc[X.index, 'PRETENSIONER_TIME']
X['pretension_load'] = dataset.table.loc[X.index, 'PRETENSION_LOAD']
X['corr_sb_dist'] = dataset.table.loc[X.index, 'CORR_SB_DIST']
X['corr_sb_dist_2'] = dataset.table.loc[X.index, 'CORR_SB_DIST_2']
X['latch_seat_dy'] = X['28_y'] - X['22_y']
X['left_femur_load'] = dataset.features.loc[X.index, 'Min_13FEMRLE00HFFOZB']

# center y coords to the center of the seat
X[X.filter(regex='._y').columns] = X.filter(regex='._y').subtract(X['22_y'], axis=0)
X[X.filter(regex='._z').columns] = X.filter(regex='._z').subtract(X['87_z'], axis=0)
X[X.filter(regex='._x').columns] = X.filter(regex='._x').subtract(X['1012_x'], axis=0)
X = X.drop(faro_points.columns, axis=1)
X = X.drop(['left_knee_pelvis_dz',
            'left_knee_pelvis_dy',
            'left_knee_pelvis_dx',
            'right_knee_pelvis_dz',
            'right_knee_pelvis_dy',
            'right_knee_pelvis_dx',
            'right_shoulder_elbow_dx','right_shoulder_elbow_dy','right_shoulder_elbow_dz'], axis=1)

Y = dataset.features.loc[subset, chy].to_frame()

X, Y = preprocess_data(X, Y, missing_x='mean', missing_y='drop')
#%% regression
chx_list = ['veh_cg_peak',
            'belt_angle_front',
            'knee_dy',
            'seatback_angle',
            'pelvis_angle',
            'latch_seat_dy',
            'shoulder_belt_lap_belt_dx',
            'right_upper_arm_angle_X',
            'right_upper_arm_angle_Y',
            'pretension_load',
            'belt_angle_side',
            'shoulder_dx']

chy_list = [chy]

for chx in chx_list:
    for chy in chy_list:
        if chx==chy: continue
    
        if chx in dataset.table.columns:
            x = arrange_by_group(dataset.table.loc[subset], dataset.table[chx], 'HF_POS')
        elif chx in dataset.features.columns:
            x = arrange_by_group(dataset.table.loc[subset], dataset.features[chx], 'HF_POS')
        elif chx in faro_points.columns:
            x = arrange_by_group(dataset.table.loc[subset], faro_points[chx], 'HF_POS')
        else:
            x = arrange_by_group(dataset.table.loc[subset], X[chx], 'HF_POS')
            
        if chy in dataset.table.columns:
            y = arrange_by_group(dataset.table.loc[subset], dataset.table[chy], 'HF_POS')
        elif chy in dataset.features.columns:
            y = arrange_by_group(dataset.table.loc[subset], dataset.features[chy], 'HF_POS')
        elif chy in faro_points.columns:
            y = arrange_by_group(dataset.table.loc[subset], faro_points[chy], 'HF_POS')
        else:
            y = arrange_by_group(dataset.table.loc[subset], X[chy], 'HF_POS')
        
#        if abs(rho(dataset.features[chx],dataset.features[chy]))<0.3:
#            continue
        fig, ax = plt.subplots()
        ax = plot_scatter(ax, x, y)
        ax = set_labels(ax, {'xlabel': chx, 'ylabel': chy, 'legend': {}})
        plt.show()
        plt.close(fig)

#%%
import plotly_express as px
from plotly.offline import plot
f = px.scatter_3d(x='right_upper_arm_angle_Y',
                  y='belt_angle_front',
                  z=chy,
                  color='HF_POS',
                  data_frame=pd.concat((X, dataset.features, dataset.table), axis=1))
plot(f)
#%% find variables to test
from PMG.COM.linear_model import *
from sklearn.linear_model import *
from functools import partial
import statsmodels.api as sm

model = iter([Lars(n_nonzero_coefs=i) for i in range(1, 30)])
eval_model = partial(SMRegressionWrapper, model=sm.OLS)

vs = VariableSelector(X, Y, model, eval_model=eval_model, incr_thresh=0.03, corr_thresh=0.5, corr_fun='kendall')
vs.find_variables()
vs.plot_variables()
print(vs.eval_results.rr.summary())

#%%
# 1: pretensioner time, belt angle front
# 2: belt angle front, chin bottom belt dz, right upper arm angle x
# 3: 
# all: belt dx
# 1 and 2 only: right upper arm angle x
x_in = ['pretension_load','belt_angle_side','pelvis_angle','latch_seat_dy','shoulder_dx']
Xtest = X[x_in]
Xtest = sm.add_constant(Xtest)

model = sm.OLS(Y, Xtest)
rr = model.fit()
print(rr.summary())

# how well does the model predict responses in pos 3?
y_pred = rr.predict(Xtest.loc[dataset.table.query('HF_POS==3').index])
y_act = Y.squeeze()[dataset.table.query('HF_POS==3').index]
plt.plot(y_pred-y_act,'.')
#%% other measures 
#X['right_elbow_wrist_dx'] = X['101_x'] - X['102_x']
#X['right_elbow_wrist_dy'] = X['101_y'] - X['102_y']
#X['right_elbow_wrist_dz'] = X['101_z'] - X['102_z']
#X['left_elbow_top_belt_dx'] = X['1023_x'] - X['400_x']
#X['left_elbow_top_belt_dy'] = X['1023_y'] - X['400_y']
#X['left_elbow_top_belt_dz'] = X['1023_z'] - X['400_z']
#X['left_elbow_bottom_belt_dx'] = X['1024_x'] - X['400_x']
#X['left_elbow_bottom_belt_dy'] = X['1024_y'] - X['400_y']
#X['left_elbow_bottom_belt_dz'] = X['1024_z'] - X['400_z']
#X['right_elbow_top_belt_dx'] = X['1023_x'] - X['100_x']
#X['right_elbow_top_belt_dy'] = X['1023_y'] - X['100_y']
#X['right_elbow_top_belt_dz'] = X['1023_z'] - X['100_z']
#X['right_elbow_bottom_belt_dx'] = X['1024_x'] - X['100_x']
#X['right_elbow_bottom_belt_dy'] = X['1024_y'] - X['100_y']
#X['right_elbow_bottom_belt_dz'] = X['1024_z'] - X['100_z']
#X['left_shoulder_wrist_dx'] = X['400_x'] - X['402_x']
#X['left_shoulder_wrist_dy'] = X['400_y'] - X['402_y']
#X['left_shoulder_wrist_dz'] = X['400_z'] - X['402_z']
#X['right_shoulder_wrist_dx'] = X['100_x'] - X['102_x']
#X['right_shoulder_wrist_dy'] = X['100_y'] - X['102_y']
#X['right_shoulder_wrist_dz'] = X['100_z'] - X['102_z']
#X['left_elbow_wrist_dx'] = X['401_x'] - X['402_x']
#X['left_elbow_wrist_dy'] = X['401_y'] - X['402_y']
#X['left_elbow_wrist_dz'] = X['401_z'] - X['402_z']