#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 14:01:04 2022

@author: timf
"""

#%% dependencies
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

#%% import data
df = pd.read_csv('data/raleigh-durham_tracts_469_analysis.csv').set_index(['state', 'county', 'tract'])

#%% remove outliers
df = df.loc[(df.dist_tmd_avg < 50) & (df.abs_dist_diff_avg < 80) ,:]

#%% get subsets for analysis
df_census = df.loc[:, 'DP02_0002PE':'DP05_0089PE']
df_mob = df.loc[:, 'visits_pre_avg':'dist_avg_base']

### model with elastic net
#%% Abs Distance ~ Census Only 
X = df_census
y = df_mob.loc[:,'abs_dist_diff_avg']
# with CV

en_mod_cv = ElasticNetCV(l1_ratio = .999, max_iter=1000)
en_mod_cv.fit(X, y)
en_mod_cv.score(X, y) # 0.126

en_mod_cv.l1_ratio_
en_mod_cv.alpha_ # 1819
abs(en_mod_cv.coef_)
np.where(abs(en_mod_cv.coef_) > 0.001)
np.sqrt(en_mod_cv.mse_path_).min(axis = 0).mean() # 5.9km 
y.mean() # 19.7

#%% Net Distance ~ Census Only 
X = df_census
y = df_mob.loc[:, 'net_dist_diff_avg']

# with CV

en_mod_cv = ElasticNetCV(l1_ratio = .999, max_iter=1000)
en_mod_cv.fit(X, y)
en_mod_cv.score(X, y) # 0! nothing!

np.sqrt(en_mod_cv.mse_path_).min(axis = 0).mean() # 5.37

np.where(abs(en_mod_cv.coef_) > 0.001)
#%% Abs Distance ~ Census + TMD: Oldham's Method

X = pd.concat([df_mob.loc[:, 'dist_tmd_avg'], df_census], axis = 1)
y = df_mob.loc[:,'abs_dist_diff_avg']

en_mod_cv = ElasticNetCV(l1_ratio = .999, max_iter=1000)
en_mod_cv.fit(X, y)

en_mod_cv.score(X, y) # 0.71
en_mod_cv.l1_ratio_
en_mod_cv.alpha_ # 1819
abs(en_mod_cv.coef_)
np.where(abs(en_mod_cv.coef_) > 0.001)
np.sqrt(en_mod_cv.mse_path_).min(axis = 0).mean() # 3.57km

#%% Net Distance ~ Census + TMD: Oldham's Method

X = pd.concat([df_mob.loc[:, 'dist_tmd_avg'], df_census], axis = 1)
y = df_mob.loc[:,'net_dist_diff_avg']

en_mod_cv = ElasticNetCV(l1_ratio = .999, max_iter=1000)
en_mod_cv.fit(X, y)

en_mod_cv.score(X, y) # 0.71
en_mod_cv.l1_ratio_
en_mod_cv.alpha_ # 1819
abs(en_mod_cv.coef_)
np.where(abs(en_mod_cv.coef_) > 0.001)
np.sqrt(en_mod_cv.mse_path_).min(axis = 0).mean() # 3.57km

#%%
#X = pd.concat([df_mob.loc[:, 'dist_tmd_avg'], df_census], axis = 1)
X = pd.concat([df_mob.loc[:, 'visits_tmd_avg'], df_census], axis = 1)

y = df_mob.loc[:,'abs_dist_diff_avg'].rank()
#y = df_mob.loc[:,'abs_visits_diff_avg']#.rank()

# without CV 
en_mod = ElasticNet(l1_ratio = 0)
en_mod.fit(X, y)
en_mod.score(X, y)

# with CV

en_mod_cv = ElasticNetCV(l1_ratio = .999, max_iter=1000)
en_mod_cv.fit(X, y)
en_mod_cv.score(X, y) # 0.136

en_mod_cv.l1_ratio_
en_mod_cv.alpha_ # 1819
abs(en_mod_cv.coef_)
np.where(abs(en_mod_cv.coef_) > 0.001)
np.sqrt(en_mod_cv.mse_path_).min(axis = 0).mean() # 85.49 rank error
en_mod_cv.mse_path_
