
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
import datetime as dt
from fredapi import Fred
from functools import reduce

def date_range():
    fred = Fred(api_key='54464081c8a7bac5af9fd0a27867f551')
    SP = fred.get_series('SP500')
    ICSA = fred.get_series('ICSA') #initial claims
    UNRATE= fred.get_series('UNRATE') #unemployment rate
    IPFPNSS= fred.get_series('IPFPNSS')#Industrial Production: Final Products and Nonindustrial Supplies
    BAAFF= fred.get_series('BAAFF').to_period('M')#Moody’s Baa Corporate Bond Minus FEDFUNDS
    import pandas_datareader as pdr
    levels = pdr.get_data_fred(['PCEPILFE', 'CPILFESL'], start='1999', end='2019').to_period('M')
    infl = np.log(levels).diff().iloc[1:] * 1200
    infl.columns = ['PCE', 'CPI']
    """This statement is not entirely true, because both the CPI and PCE price indexes 
    can be revised to a certain extent after the fact. As a result, the series that 
    wee re pulling are not exactly like those observed on April 14, 2017. This could 
    be fixed by pulling the archived data from ALFRED instead of FRED, but the data 
    we have is good enough for this tutorial."""
    # Remove two outliers and de-mean the series
    infl['PCE'].loc['2001-09':'2001-10'] = np.nan
    IPFINAL= fred.get_series('IPFINAL')#IP: Final Products (Market Group)
    VIXCLS= fred.get_series('VIXCLS')#CBOE Volatility Index: VIX 
    IPMANSICS= fred.get_series('IPMANSICS')#IP: Manufacturing (SIC)
    AAAFF= fred.get_series('AAAFF')#Moody’s Aaa Corporate Bond Minus FEDFUNDS
    IPCONGD= fred.get_series('IPCONGD')#IP: Consumer Goods
    PAYEMS= fred.get_series('PAYEMS')#All Employees: Total nonfarm
    DTB6= fred.get_series('DTB6')#6-Month Treasury Bill:
    GDPC1= fred.get_series('GDPC1')# Real GDP

    data_frames = [ICSA, UNRATE,IPFPNSS,BAAFF,IPFINAL,VIXCLS,IPMANSICS,AAAFF,IPCONGD,PAYEMS,DTB6,GDPC1]
    columns_= ['ICSA', 'UNRATE', 'IPFPNSS','BAAFF','IPFINAL','VIXCLS','IPMANSICS','AAAFF','IPCONGD','PAYEMS','DTB6','GDPC1']
    data_frames=[x.to_frame()  for x in data_frames]
    for i in range(0,len(data_frames)):
        data_frames[i].index.names = ['DATES']  
        data_frames[i]=data_frames[i].resample('M').mean()
        data_frames[i]=data_frames[i].reset_index(['DATES'])

    """data=pd.merge(data_frames[0],data_frames[1],on=['DATES'])
    for i in range(2,len(data_frames)):
        data=pd.merge(data,data_frames[i],on=['DATES'])
        print('000000000000000000000',columns_[i],data)"""

    data=reduce(lambda  left,right:  left.merge(right,how='inner',on="DATES"), data_frames)
    #temp=reduce(lambda  left,right: left.join(right,on="DATES", lsuffix='_left', rsuffix='_right'), data_frames)
    data=data.set_index('DATES')
    data.columns=columns_
    #data.to_excel("E:\ForecastingApp\macro-model\code\\Nowcasting\Fed_data.xlsx")  
    return data