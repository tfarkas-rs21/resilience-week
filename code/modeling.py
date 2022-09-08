#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  2 14:01:04 2022

@author: timf
"""

#%% dependencies
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools
sns.set_theme(style="darkgrid")


#%% import data
df = pd.read_csv('data/raleigh-durham_tracts_469_analysis.csv', 
                 dtype = {'state': str, 'county': str, 'tract': str}).set_index(['state', 'county', 'tract'])

#%% remove outliers
df = df.loc[(df.dist_tmd_avg < 50) & (df.abs_dist_diff_avg < 80) ,:]
df = df[df.index.isin(['063', '183'], level=1)] # Wake and Durham counties only

#%% get subsets for analysis
df_census = df.loc[:, 'DP02_0002PE':'DP05_0089PE']
df_mob = df.loc[:, 'visits_pre_avg':'dist_avg_base']

### Modeling with elastic net
## Abs Distance Rank ~ Census + TMD Rank: Oldham's Method
#%% data prep
X_raw = pd.concat([df_mob.loc[:, 'dist_tmd_avg'].rank(pct = True), df_census], axis = 1)

# replace average with pre-disturbance mobility
X2 = X_raw.copy()
X2['dist_tmd_avg'] = df_mob.loc[:, 'dist_pre_avg'].rank(pct = True)

y = df_mob.loc[:,'abs_dist_diff_avg'].rank(pct = True)

# scale data for modeling
scaler = StandardScaler()
X = scaler.fit_transform(X_raw, y)
X2 = scaler.fit_transform(X2, y)

#%% run & evaluate models
# cross-validance for regularization (alpha) and L1 ratio 
en_mod_cv = ElasticNetCV(l1_ratio = [0.001, 0.1, 0.5, 0.9, 0.999], max_iter=1000)
en_mod_cv.fit(X, y)
en_mod_cv.alpha_ # 0.029
en_mod_cv.l1_ratio_ # 0.999

# refit with pure Lasso
en_mod = ElasticNet(alpha = en_mod_cv.alpha_, l1_ratio = 1)
en_mod.fit(X, y)

# get scores on Oldham's as typical
en_mod.score(X, y) # 0.685 - not x-valid
cross_val_score(en_mod, X, y, cv = 10, scoring = 'r2').mean() # 0.645
-cross_val_score(en_mod, X, y, cv = 10, scoring = 'neg_root_mean_squared_error').mean() # 0.166 --> 16% error

# get scores using pre as typical
en_mod.score(X2, y) # 0.506 - with pre value, not x-valid
y_pred = en_mod.predict(X2)
np.sqrt(mean_squared_error(y, y_pred)) # 0.203 --> 20% error

X_indices = np.where(abs(en_mod.coef_) > 0.001)[0]
X3 = X_raw.iloc[:, X_indices]
en_mod.coef_[X_indices]

## Abs Distance Rank ~ Census Only: Oldham's Method
#%% data prep
X_raw = df_census
y = df_mob.loc[:,'abs_dist_diff_avg'].rank(pct = True)

scaler = StandardScaler()
X = scaler.fit_transform(X_raw, y)

#%% run & evaluate models
# cross-validance for regularization (alpha) and L1 ratio 
en_mod_cv = ElasticNetCV(l1_ratio = [0.001, 0.1, 0.5, 0.9, 0.999], max_iter=1000)
en_mod_cv.fit(X, y)
en_mod_cv.alpha_ # 0.029
en_mod_cv.l1_ratio_ # 0.999

# refit with pure Lasso
en_mod = ElasticNet(alpha = en_mod_cv.alpha_, l1_ratio = 1)
en_mod.fit(X, y)

en_mod.score(X, y) 
cross_val_score(en_mod, X, y, cv = 10, scoring = 'r2').mean() 
-cross_val_score(en_mod, X, y, cv = 10, scoring = 'neg_root_mean_squared_error').mean() 

## Abs Distance Rank ~ TMD Rank Only: Oldham's Method
#%% data prep
X_raw = pd.DataFrame({'dist_tmd_avg': df_mob.loc[:, 'dist_tmd_avg'].rank(pct = True)})
y = df_mob.loc[:,'abs_dist_diff_avg'].rank(pct = True)

scaler = StandardScaler()
X = scaler.fit_transform(X_raw, y)

#%% run & evaluate models
# cross-validance for regularization (alpha) and L1 ratio 
en_mod_cv = ElasticNetCV(l1_ratio = [0.001, 0.9, 0.999], max_iter=10000)
en_mod_cv.fit(X, y)
en_mod_cv.alpha_ 
en_mod_cv.l1_ratio_

# refit Lasso
en_mod = ElasticNet(alpha = en_mod_cv.alpha_, l1_ratio = 1)
en_mod.fit(X, y)

# get scores on Oldham's as typical
en_mod.score(X, y) 
cross_val_score(en_mod, X, y, cv = 10, scoring = 'r2').mean() 
-cross_val_score(en_mod, X, y, cv = 10, scoring = 'neg_root_mean_squared_error').mean()

### plotting 
#%% data for trace figures
rmse_L1 = np.sqrt(en_mod_cv.mse_path_).mean(axis = 2)[1]
alphas_L1 = en_mod_cv.alphas_[1]
rmse_trace = pd.DataFrame({'alpha':alphas_L1, 'RMSE':rmse_L1})
rmse_trace.to_csv('data/rmse-trace-rank-dist-L1.csv')
sns.relplot(x = 'alpha', y = 'RMSE', kind = 'line', data = rmse_trace)

#%% Bland-Altman
ax = sns.relplot(x = df_mob.loc[:, 'dist_tmd_avg'].rank(pct = True), 
            y = df_mob.loc[:, 'abs_dist_diff_avg'].rank(pct = True))
ax.set_xlabels('% Rank Average Distance Traveled')
ax.set_ylabels('% Rank Absolute Change In Distance Traveled')

#%% t-test histograms
df_mob_2020 = pd.read_csv('data/raleigh-durham_2020_comparison.csv', 
                          dtype = {'state': str, 'county': str, 'tract': str})
df_mob_2020 = df_mob_2020.drop('geometry', axis = 1, inplace = False)
df_mob_2020 = df_mob_2020.set_index(['state', 'county', 'tract'])

fig, ax = plt.subplots(ncols = 2)
sns.histplot(df_mob['abs_dist_diff_avg'], kde = True, ax = ax[0])
sns.histplot(df_mob_2020['abs_diff_avg'], kde = True, ax = ax[0], color = 'orange')
sns.histplot(df_mob['net_dist_diff_avg'], kde = True, ax = ax[1])
sns.histplot(df_mob_2020['diff_avg'], kde = True, ax = ax[1], color = 'orange')
ax[0].set_xlim(0, 50)
ax[1].set_xlim(-25, 25)
fig.supylabel("Count Of Census Tracts")
fig.supxlabel("Average Difference In Distance Travelled (km)")
ax[0].set(xlabel = None, ylabel = None)
ax[1].set(xlabel = None, ylabel = None)
ax[0].text(3, 40, 'A', size = 'x-large', weight = 'bold')
ax[1].text(-22, 46, 'B', size = 'x-large', weight = 'bold')
fig.tight_layout()
plt.show()

### Simulation
#%% independent gammas
from numpy.random import default_rng
from scipy.stats import pearsonr, rankdata
rng = default_rng()

def get_oldhams_r2(n_tracts = 250, gamma_shape = 10, n_devs = 1000):
    abs_diffs = []
    oldhams = []
    
    for _ in range(n_tracts):
        pre = rng.gamma(gamma_shape, size = n_devs) 
        post = rng.gamma(gamma_shape, size = n_devs)
        abs_diff_avg = np.abs(post - pre).mean()
        abs_diffs.append(abs_diff_avg)
        oldham = ((post + pre)/2).mean()
        oldhams.append(oldham)
    
    return(pearsonr(rankdata(abs_diffs), rankdata(oldhams))[0] ** 2)

r2s = np.array([get_oldhams_r2() for _ in range(100)])
r2s.mean()

#%% mvnormal with empirical covariance matrix
from numpy.random import default_rng
from scipy.stats import pearsonr, rankdata

rng = default_rng()
empcovmat = np.cov(df_mob.dist_pre_avg, df_mob.dist_post_avg) # empritical covariance matrix
pre_mean = df_mob.dist_pre_avg.mean() 
post_mean = df_mob.dist_post_avg.mean() 

def get_oldhams_r2_mvnorm(n_tracts = 250, n_devs = 1000,
                          means = np.array([0, 0]), 
                          covarmat = np.array([[1, 0], [0, 1]])) :
    abs_diffs = []
    oldhams = []
    
    for _ in range(n_tracts):
        mvnorms = rng.multivariate_normal(means, covarmat, size = n_devs)
        abs_diff_avg = np.abs(mvnorms[:, 1] - mvnorms[:, 0]).mean()
        abs_diffs.append(abs_diff_avg)
        oldham = ((mvnorms[:, 1] + mvnorms[:, 0])/2).mean()
        oldhams.append(oldham)
    
    return(pearsonr(abs_diffs, oldhams)[0] ** 2)

r2s = np.array([get_oldhams_r2_mvnorm(means = np.array([pre_mean, post_mean]), 
                                      covarmat = empcovmat) for _ in range(1000)])
r2s.mean() # always very close to 0

#%% bootstrap data
from random import choices

df_boot = np.column_stack([X.dist_tmd_avg, y])
nrows = df_boot.shape[0]

r2_boot = []
for _ in range(1000):
    indices = choices(np.arange(nrows), k = nrows)
    df_temp = df_boot[indices, :]
    r2_boot.append(pearsonr(df_temp[:, 0], df_temp[:, 1])[0] ** 2)

np.array(r2_boot).mean()

#%% histograms for simulation vs. bootstrap
fig, ax = plt.subplots()
plot_type = 'probability'
sns.histplot(r2s, kde = True,stat = plot_type, ax = ax)
sns.histplot(r2_boot, kde = True, stat = plot_type, ax = ax, color = 'orange')
true_r2 = pearsonr(X_raw.dist_tmd_avg, y)[0] ** 2 # 0.66
ax.legend(labels=['simulation', 'bootstrap'], loc = 'upper center')
ax.set_xlabel('Squared Pearson Correlation')
ax.set_ylabel('Density')
plt.show()

