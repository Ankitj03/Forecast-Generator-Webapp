#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import datetime
from cleandata import ts_clean,season_clean
import pmdarima.arima as ar
import matplotlib.pyplot as plt
import numpy as np
from fbprophet import Prophet
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import math


def dataclean(quantity):
    df=pd.DataFrame(quantity)
    #df=mydf.copy()
    df.columns=['Month','Total']
    today=datetime.datetime.today()
    lastday=datetime.datetime(today.year,today.month,1)
    df=df[(df['Month']<lastday)]
    df['Month']=df['Month'].apply(lambda x: x.strftime("%Y-%m"))
    df['Total']=df['Total'].astype(int)
    df.index=df.Month
    df=df.drop(['Month'],axis=1)
    moving_avg=df.rolling(12).mean()
    ts_diff=df-moving_avg
    clean_ts=season_clean(ts_diff)
    trend_ts=clean_ts+moving_avg
    return trend_ts


# In[8]:


def prophet_forecast(mydf,data_monthly,period=6):
    df=mydf.copy()
    df['ds']=df.index
    df.columns=['y','ds']
    df=df.reset_index(drop=True)
    train = df[:int(0.8*(len(df)))]
    test = df[int(0.8*(len(df))):]
    model=Prophet()
    model.fit(train)
    valid = model.make_future_dataframe(periods = len(test), freq = 'MS')  
    forecast = model.predict(valid)
    mape=mean_absolute_percentage_error(test['y'],forecast['yhat'][-len(test):])
    prophet=Prophet()
    prophet.fit(df)
    future = prophet.make_future_dataframe(periods=period,freq = 'MS')
    forecast = prophet.predict(future)
    cleanforecast=forecast[['ds','yhat']][-period:]
    cleanforecast.columns=['Month','Total']
    cleanforecast.index=cleanforecast.Month
    cleanforecast=cleanforecast.drop(['Month'],axis=1)
    final_series=data_monthly.append(cleanforecast)
    final_series.index=final_series.index.map(lambda x: x.strftime("%Y-%m"))
    return final_series,mape


# In[9]:


def arima_forecast(mydf,data_monthly,period=6):
    df=mydf.copy()
    train = df[:int(0.8*(len(df)))]
    test = df[int(0.8*(len(df))):]
    model = ar.auto_arima(train,start_p=0, start_q=0,
                           max_p=5, max_q=5, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
    model.fit(train)
    forecast = model.predict(n_periods=len(test))
    mape=mean_absolute_percentage_error(test.Total.values,forecast)
    arima = ar.auto_arima(df,start_p=0, start_q=0,
                           max_p=5, max_q=5, m=12,
                           start_P=0, seasonal=True,
                           d=1, D=1, trace=True,
                           error_action='ignore',  
                           suppress_warnings=True, 
                           stepwise=True)
    arima.fit(df)
    forecast = arima.predict(n_periods=period)
    lst=[]
    for x in range(1,period+1):
        lst.append(datetime.date(df[-1:].index.year.values[0],df[-1:].index.month.values[0],1)+datetime.timedelta(x*368/12))
    d={'Month':lst,'Total':forecast}
    future=pd.DataFrame(d)
    future.index=future['Month']
    future=future.drop(['Month'],axis=1)
    final_series=data_monthly.append(future)
    final_series.index=final_series.index.map(lambda x: x.strftime("%Y-%m"))
    return final_series,mape
    


# In[10]:


def ets_forecast(mydf,data_monthly,period=6):
    df=mydf.copy()
    train = df[:int(0.8*(len(df)))]
    test = df[int(0.8*(len(df))):]
    model = ExponentialSmoothing(train, seasonal_periods=12, trend='add', seasonal='add').fit()
    forecast = model.forecast(len(test))
    mape=mean_absolute_percentage_error(test.Total.values,forecast[-len(test):].values)
    ets = ExponentialSmoothing(df, seasonal_periods=12, trend='add', seasonal='add').fit()
    forecast1 = ets.forecast(period)
    future=pd.DataFrame(forecast1)
    future.columns=['Total']
    final_series=data_monthly.append(future)
    final_series.index=final_series.index.map(lambda x: x.strftime("%Y-%m"))
    return final_series,mape
    


# In[11]:


def mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / y_true)) * 100


# In[12]:


def iterateclean(mydf,aggregator,mastercode,startdate,enddate):
    df=mydf.copy()
    data=df[(df[aggregator] == mastercode)]
    data.Month=pd.to_datetime(data.Month)
    data=data[(data['Month']<=enddate) & (data['Month']>=startdate)]
    data=data.sort_values('Month')
    data=data[['Month','Total']]
    data.Total=data.Total.astype(int)
    data['Month']=data['Month'].apply(lambda x: x.strftime("%Y-%m"))
    data.index=data.Month
    data=data.drop(['Month'],axis=1)
    data.index=pd.to_datetime(data.index)
    if len(data)>24:
        moving_avg=data.rolling(13).mean()
    else:
        return data,data
    ts_diff=data-moving_avg
    ts_diff.dropna(inplace=True)
    clean_ts=season_clean(ts_diff)
    trend_ts=clean_ts+moving_avg
    trend_ts.dropna(inplace=True)
    return trend_ts,data



# In[11]:

def min_mape(a,b,c=math.inf):
    array=[a,b,c]
    return array.index(min(array))


