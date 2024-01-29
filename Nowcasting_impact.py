import statistics
import matplotlib.pyplot as plt
import xlsxwriter
import pandas as pd
from statsmodels.tsa.tsatools import lagmat
from numpy import log
from numpy import exp
from numpy import abs
import numpy as np
from datetime import timedelta
from datetime import datetime
import scipy.optimize
from sklearn import linear_model
from scipy import optimize
from numpy import dot
from scipy.optimize import minimize
from statsmodels.tsa.stattools import adfuller
from arch.unitroot import PhillipsPerron
from statsmodels.tools.numdiff import approx_hess2
from statsmodels.tools.numdiff import approx_hess3
from numpy.linalg import inv
from scipy.stats import norm
import seaborn as sns
from statsmodels.tsa.statespace.dynamic_factor_mq import DynamicFactorMQ
from load_fredmd_data import load_fredmd_data
import datetime as dt
from fredapi import Fred
from functools import reduce
from intersection import intersection



data = pd.read_excel ('E:\ForecastingApp\macro-model\code\\Nowcasting\Fed_data.xls')
data=data.set_index('DATES')
data[['IPFPNSS','IPFINAL','IPMANSICS','IPCONGD','PAYEMS']]=data[['IPFPNSS','IPFINAL','IPMANSICS','IPCONGD','PAYEMS']].pct_change()
data[['GDPC1']]=data[['GDPC1']].pct_change().where(data.notna())
data[['ICSA', 'UNRATE', 'IPFPNSS','BAAFF','IPFINAL','VIXCLS','AAAFF','IPCONGD','PAYEMS','DTB6']]=data[['ICSA', 'UNRATE', 'IPFPNSS','BAAFF','IPFINAL','VIXCLS','AAAFF','IPCONGD','PAYEMS','DTB6']].diff()
data=data[['IPFINAL', 'GDPC1']]
print(data)
factor_name='Factor'
lags=1
k_endog_monthly_=2#data.shape[1]-1
pred_step=1
# F1 is the name of a block of factors. The number of distinct factors in F1 block is determined in factor_multiplicities.
keys = list(data.columns[0:].values)
#Create a dict of variables where all the values equal F1. Each variable is assumed to depend on F1 block of factors.
factor_struct = dict.fromkeys(keys,[factor_name])
print(factor_struct)
dt_range = pd.date_range(dt.datetime(year=2023,month=7,day=31), dt.datetime(year=2023,month=10,day=30), freq='M')
variables = ['GDPC1']
strt=data.index[0]

dt_range= pd.date_range(dt.datetime(year=2018,month=7,day=31), dt.datetime(year=2023,month=10,day=31), freq='M')
data1=data.loc[data.index[0]:dt_range[-2]]
y_post=data.copy()
y_post.loc[dt_range[-1],'GDPC1']=np.nan
print(data1)
print(y_post)
mod=DynamicFactorMQ(endog=data1,k_endog_monthly=k_endog_monthly_,factors=factor_struct,factor_orders=lags,factor_multiplicities={factor_name:1},idiosyncratic_ar1=True,standardize=False)
res_pre=mod.fit()
point_forecasts = res_pre.forecast(steps=5)     
# Create a new results object by passing the new observations to the `append` method
print(type(res_pre))
const_post_plus1 = np.ones(len(y_post) +1)
news = res_pre.news(y_post, exog=const_post_plus1,  start='2023-10-31', end='2024-02-28')
print(news.summary())
print('newssssssssssssssssss\n',news.weights)


# Print the total impacts, computed by the `news` method
# (Note: news.total_impacts = news.revision_impacts + news.update_impacts, but
# here there are no data revisions, so total and update impacts are the same)
print('impacttttttttttt\n',news.total_impacts)

fig, ax = plt.subplots(figsize=(14, 6))
news.total_impacts.plot(kind='bar', stacked=True, width=0.3, zorder=2, ax=ax)
x = np.arange(5)
ax.plot(x, point_forecasts['GDPC1'], marker='o', color='k', markersize=7, linewidth=2)
ax.xaxis.set_ticklabels(['Dec','Jan','Feb','Mar','Apr'])
ax.xaxis.set_tick_params(size=0)
ax.set_title('Evolution of real GDP growth nowcasting: 2024Q1', fontsize=16, fontweight=600, loc='left')

#plt.plot(predict,color='red',linestyle='dotted') 
plt.show()
