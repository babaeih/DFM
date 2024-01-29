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
data[['GDPC1']]=data[['GDPC1']].pct_change(freq='Q')
data[['ICSA', 'UNRATE', 'IPFPNSS','BAAFF','IPFINAL','VIXCLS','AAAFF','IPCONGD','PAYEMS','DTB6']]=data[['ICSA', 'UNRATE', 'IPFPNSS','BAAFF','IPFINAL','VIXCLS','AAAFF','IPCONGD','PAYEMS','DTB6']].diff()
k_endog_monthly_=data.shape[1]-1

factor_name='Factor'
lags=3
pred_step=1
# F1 is the name of a block of factors. The number of distinct factors in F1 block is determined in factor_multiplicities.
keys = list(data.columns[0:].values)
#Create a dict of variables where all the values equal F1. Each variable is assumed to depend on F1 block of factors.
factor_struct = dict.fromkeys(keys,[factor_name])
print(factor_struct)
dt_range = pd.date_range(dt.datetime(year=2018,month=7,day=31), dt.datetime(year=2023,month=11,day=30), freq='M')

variables = ['GDPC1']
strt=data.index[0]
predict= data[variables]
#predict=predict.append(pd.DataFrame( columns=[variables], index=pd.DatetimeIndex(date_range(2024,2024,2,10,1))).to_period('M'))
predict.loc[strt:dt_range[0],variables]=np.nan

#for i in range(0,len(dt_range)-pred_step):        
#data1=data.loc[data.index[0]:dt_range[i]]   

mod=DynamicFactorMQ(endog=data,k_endog_monthly=k_endog_monthly_,factors=factor_struct,factor_orders=lags,factor_multiplicities={factor_name: 7},idiosyncratic_ar1=True,standardize=False)
res=mod.fit()
print(res.summary())
prediction_results = res.get_prediction(start=data.index[0], end='2024-02')
point_predictions = prediction_results.predicted_mean[variables]
ci = prediction_results.conf_int(alpha=0.05)
lower = ci[[f'lower {name}' for name in variables]]
upper = ci[[f'upper {name}' for name in variables]]

# Plot the forecasts and confidence intervals
with sns.color_palette('deep'):
    fig, ax = plt.subplots(figsize=(14, 4))

    # Plot the in-sample predictions
    point_predictions.index.names=['DATES']
    point_predictions=point_predictions.rename(columns={"GDPC1": "Prediction"})
    ax.plot(data[variables], marker='o', color='k', markersize=2, linewidth=2)
    ax.legend(['GDP'])
    print(point_predictions)
    point_predictions.loc[:'2024-02'].plot(ax=ax, color=['red', 'C1', 'C2'],
                                           legend='Prediction', linewidth=1)
  

    # Plot the out-of-sample forecasts
    """point_predictions.loc['2024-02':].plot(ax=ax, linestyle='dotted',
                                           color=['red', 'C1', 'C2'],
                                           legend='Prediction')"""

    # Confidence intervals
    for name in variables:
        ax.fill_between(ci.index,
                        lower[f'lower {name}'],
                        upper[f'upper {name}'], alpha=0.1)
        
    # Forecast period, set title
    ylim = ax.get_ylim()
    ax.vlines('2023-10', ylim[0], ylim[1], linewidth=1)
    ax.annotate(r' Forecast', ('2023-11', -0.1),rotation=90,color='red')
    ax.set(title=('US GDP growth rate prediction:'
                  ' in-sample predictions and out-of-sample forecasts, with 95% confidence intervals'), ylim=ylim)
    
    fig.tight_layout()
plt.show()

#plt.plot(predict,color='red',linestyle='dotted') 
plt.show()
