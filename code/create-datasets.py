#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep  1 15:32:25 2022

@author: timf
"""
#%% dependencies 
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
import sys

sys.path.append('./code/')
from resilience_fxns import prep_census_data
#%% import 
df_mob = pd.read_csv('data/raleigh-durham_tracts_w_baseline.csv', 
                     dtype = {'state': str, 'county': str, 'tract': str})
#df_mob.drop('geometry', axis = 1, inplace = True)
df_mob = df_mob.set_index(['state', 'county', 'tract'])

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
        
#%% basic data prep, dealing with missing values, etc.
df_census_pe = df_census.loc[:, df_census.columns.str.endswith('PE') | [ x in ('state', 'county', 'tract') for x in df_census.columns]] 
df_census_pe = df_census_pe.set_index(['state', 'county', 'tract'])
df_census_pe = df_census_pe.mask(df_census_pe < 0, np.nan)
df_census_pe = df_census_pe.loc[df_census_pe['DP02_0001PE'] != 0, :]
df_census_pe = df_census_pe.loc[:, df_census_pe.isna().sum() < df_census_pe.shape[0]]
df_census_pe = df_census_pe.loc[:, [not any(df_census_pe.loc[:, x] > 100) for x in df_census_pe.columns]]

#%% impute missing data with mean in each column
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer = imputer.fit(df_census_pe)

df_census_imputed = pd.DataFrame(imputer.transform(df_census_pe))
df_census_imputed.columns = df_census_pe.columns
df_census_imputed = pd.concat([df_census_imputed, df_census_pe.reset_index().loc[:, ['state', 'county', 'tract']]], axis = 1)
df_census_imputed = df_census_imputed.set_index(['state', 'county', 'tract'])

#%% merge with mobility data and write

df = df_mob.join(df_census_imputed, how = 'inner')
df.to_csv('data/raleigh-durham_tracts_469_analysis.csv')

#%%








