#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 15:32:25 2022

@author: timf
"""
#%% dependencies 
import pandas as pd
import numpy as np
import sys

sys.path.append('./code/')
from resilience_fxns import prep_census_data
#%% import 
df_mvmt = pd.read_csv('data/raleigh-durham_tracts_plus_colonial_hack_diff.csv')
df_mvmt.drop('geometry', axis = 1, inplace = True)

#%%
#df = pd.read_csv('data/dp02-2019.tsv', delimiter = '\t', 
#                     dtype = {'state': str, 'county': str, 'tract': str})

#%% merge all census files
file_list = ['data/dp02-2019.tsv', 'data/dp03-2019.tsv', 'data/dp04-2019.tsv', 'data/dp05-2019.tsv']
df_census = pd.DataFrame()

for file_name in file_list: 
    df = pd.read_csv(file_name, delimiter = '\t', 
                         dtype = {'state': str, 'county': str, 'tract': str})
    print(df.shape)
    if df_census.empty: 
        df_census = df
    else:
        df_census = df_census.merge(df, how = 'outer', on = ['state', 'county', 'tract'])

print(df_census.shape)
        
#%%

df_census_pe = df_census.loc[:, df_census.columns.str.endswith('PE') | [ x in ('state', 'county', 'tract') for x in df_census.columns]]
df_census_pe = df_census_pe.set_index(['state', 'county', 'tract'])
df_census_pe[(df_census_pe < 0)] = np.nan
df_census_pe = df_census_pe.loc[df_census_pe['DP02_0001PE'] != 0, :]
df_census_pe = df_census_pe.loc[:, df_census_pe.isna().sum() < df_census_pe.shape[0]]
df_census_pe[df_census_pe['DP02_0001PE']> 0]


#%%

