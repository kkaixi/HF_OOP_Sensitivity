# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 12:51:24 2019

@author: tangk
"""

from PMG.read_data import PMGDataset
from PMG.COM.get_props import get_peaks
import json
import pandas as pd


directory = 'P:\\Data Analysis\\Projects\\HF OOP Sensitivity\\'
cutoff = range(100,1600)
channels = ['13HEAD0000HFACXA','13HEAD0000HFACYA','13HEAD0000HFACZA','13HEAD0000HFACRA',
            '13HEAD0000HFAVXD','13HEAD0000HFAVYD','13HEAD0000HFAVZD',
            '13NECKUP00HFFOXA','13NECKUP00HFFOYA','13NECKUP00HFFOZA','13NECKUP00HFFORA',
            '13NECKUP00HFMOXB','13NECKUP00HFMOYB','13NECKUP00HFMOZB','13NECKUP00HFMORB',
            '13CLAVLE00HFFOXA','13CLAVLE00HFFOYA','13CLAVLE00HFFOZA','13CLAVLE00HFFORA',
            '13CLAVRI00HFFOXA','13CLAVRI00HFFOYA','13CLAVRI00HFFOZA','13CLAVRI00HFFORA',
            '13CHST0000HFACXC','13CHST0000HFACYC','13CHST0000HFACZC','13CHST0000HFACRC',
            '13CHST0000HFDSXB',
            '13RIBS01LEHFDSXB', '13RIBS01RIHFDSXB', '13RIBS02LEHFDSXB', '13RIBS02RIHFDSXB',
            '13RIBS03LEHFDSXB', '13RIBS03RIHFDSXB', '13RIBS04LEHFDSXB', '13RIBS04RIHFDSXB',
            '13RIBS05LEHFDSXB', '13RIBS05RIHFDSXB', '13RIBS06LEHFDSXB', '13RIBS06RIHFDSXB',
            '13LUSP0000HFFOXA','13LUSP0000HFFOYA','13LUSP0000HFFOZA','13LUSP0000HFFORA',
            '13LUSP0000HFMOXA','13LUSP0000HFMOYA',
            '13ILACLE00HFFOXA',
            '13ILACLE00HFMOYA',
            '13ILACRI00HFFOXA',
            '13ILACRI00HFMOYA',
            '13PELV0000HFACXA','13PELV0000HFACYA','13PELV0000HFACZA','13PELV0000HFACRA',
            '13FEMRLE00HFFOZB',
            '13FEMRRI00HFFOZB',
            '13SEBE0000B3FO0D',
            '13SEBE0000B6FO0D',
            '10SIMELE00INACXD',
            '10SIMERI00INACXD',
            '10CVEHCG0000ACXD','10CVEHCG0000ACYD','10CVEHCG0000ACZD','10CVEHCG0000ACRD',
            '13HICR0000HF00RA', '13HICR0036HF00RA', '13HICR0015HF00RA', '13BRIC0000HFAV0D',
            '13NIJCIPTEHF00YX', '13NIJCIPTFHF00YX', '13NIJCIPCFHF00YX', '13NIJCIPCEHF00YX',
            '13HEAD003SHFACRA', '13CHST003SHFACRC']

table_filters = {'query': 'SPEED==48 and HF_POS>=1 and HF_POS<=3',
                 'drop': ['TC18-209']}
preprocessing = None

dataset = PMGDataset(directory, channels=channels, cutoff=cutoff, verbose=False)
dataset.table_filters = table_filters
dataset.preprocessing = preprocessing

if __name__=='__main__':
    dataset.get_data(['timeseries'])
    table = dataset.table
    features = get_peaks(dataset.timeseries)
    
    faro_data = ['HF_POS','SEAT_Y','SEAT_TRACK','SEATBACK_ANGLE','PELVIS_ANGLE','HEAD_CG_Y']
    features[faro_data] = dataset.table[faro_data]
    features['HEAD_CG_Y-SEAT_Y'] = features['HEAD_CG_Y'] - features['SEAT_Y']
    
    features.to_csv(directory + 'features.csv')
    
    # json file specifying statistical tests to be done
    to_JSON = {'project_name': '',
               'directory'   : directory,
               'cat'     : {'STD_1': dataset.table.query('HF_POS==1').index.tolist(),
                            'STD_2': dataset.table.query('HF_POS==2').index.tolist(),
                            'OOP': dataset.table.query('HF_POS==3').index.tolist(),
                            'STD_ALL': dataset.table.query('HF_POS<3').index.tolist(),
                            'ALL': dataset.table.index.tolist()},
               'test'    : [{'name': 'std_vs_oop_wilcox',
                             'test1': 'STD_ALL',
                             'test2': 'OOP',
                             'testname': 'wilcox.test',
                             'data': 'features',
                             'args': 'paired=FALSE, exact=FALSE, correct=TRUE, conf.level=0.95'},
                            {'name': 'std_vs_oop_t',
                             'test1': 'STD_ALL',
                             'test2': 'OOP',
                             'testname': 't.test',
                             'data': 'features',
                             'args': 'paired=FALSE, exact=FALSE, correct=TRUE, conf.level=0.95'},
                            {'name': 'std1_vs_oop_wilcox',
                             'test1': 'STD_1',
                             'test2': 'OOP',
                             'testname': 'wilcox.test',
                             'data': 'features',
                             'args': 'paired=FALSE, exact=FALSE, correct=TRUE, conf.level=0.95'},
                            {'name': 'std1_vs_oop_t',
                             'test1': 'STD_1',
                             'test2': 'OOP',
                             'testname': 't.test',
                             'data': 'features',
                             'args': 'paired=FALSE, exact=FALSE, correct=TRUE, conf.level=0.95'},
                            {'name': 'std2_vs_oop_wilcox',
                             'test1': 'STD_2',
                             'test2': 'OOP',
                             'testname': 'wilcox.test',
                             'data': 'features',
                             'args': 'paired=FALSE, exact=FALSE, correct=TRUE, conf.level=0.95'},
                            {'name': 'std2_vs_oop_t',
                             'test1': 'STD_2',
                             'test2': 'OOP',
                             'testname': 't.test',
                             'data': 'features',
                             'args': 'paired=FALSE, exact=FALSE, correct=TRUE, conf.level=0.95'},
                            {'name': 'std1_vs_std2_wilcox',
                             'test1': 'STD_1',
                             'test2': 'STD_2',
                             'testname': 'wilcox.test',
                             'data': 'features',
                             'args': 'paired=FALSE, exact=FALSE, correct=TRUE, conf.level=0.95'},
                            {'name': 'std1_vs_std2_t',
                             'test1': 'STD_1',
                             'test2': 'STD_2',
                             'testname': 't.test',
                             'data': 'features',
                             'args': 'paired=FALSE, exact=FALSE, correct=TRUE, conf.level=0.95'},
                            {'name': 'hf_pos_aov',
                             'test1': 'ALL',
                             'variables': ['HF_POS'],
                             'formula': 'HF_POS',
                             'testname': 'aov',
                             'data': 'features',
                             'model_args': None,
                             'test_args': None}],
                'test2'  : None}    
    
    for test in to_JSON['test']:
        test['test1'] = to_JSON['cat'][test['test1']]
        if 'test2' in test:
            test['test2'] = to_JSON['cat'][test['test2']]
    with open(directory+'params.json','w') as json_file:
        json.dump(to_JSON,json_file)
    

