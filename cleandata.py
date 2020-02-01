#!/usr/bin/env python
# coding: utf-8

# In[1]:


from rpy2.robjects.packages import importr
#get ts object as python object
from rpy2.robjects import pandas2ri
pandas2ri.activate()
import rpy2.robjects as robjects
ts=robjects.r('ts')
import pandas as pd
import numpy as np
import dask


def ts_clean(time_series=None,freq=None,replace_missing=True,return_ts=True):
    """
    Uses R tsclean function to identify and replace outliers and missing values
    https://www.rdocumentation.org/packages/forecast/versions/7.1/topics/tsclean

    Args:
        time_series: input time series
        freq: frequency of time series
        replace_missing: if True, not only removes outliers but also interpolates missing values
        return_ts: boolean check, if True will return time series instead of class object (used for tsclean_dataframe calls)

    Returns
        cleaned_time_series: outputs cleaned time series
    """
    freq_dict = {12:'MS',1:'Y',365:'D',4:'QS',8760:'HS'}
    freq_string = freq_dict[freq]

    #find the start of the time series
    start_ts = time_series[time_series.notna()].index[0]
    #find the end of the time series
    end_ts = time_series[time_series.notna()].index[-1]
    #extract actual time series
    time_series = time_series.loc[start_ts:end_ts]
    #converts to ts object in R
    time_series_R = robjects.IntVector(time_series)
    rdata=ts(time_series_R,frequency=freq)

    if replace_missing:
        R_val = 'TRUE'
    else:
        R_val = 'FALSE'

    rstring="""
         function(rdata){
         library(forecast)
         x <- tsclean(rdata,replace.missing=%s,lambda=NULL)
         return(x)
         }
        """ % (R_val)


    rfunc=robjects.r(rstring)
    cleaned_int_vec = rfunc(rdata)
    cleaned_array = pandas2ri.ri2py(cleaned_int_vec)
    cleaned_ts = pd.Series(cleaned_array,index=pd.date_range(start=time_series[time_series.notnull()].index.min(),periods=len(time_series[time_series.notnull()]),freq=freq_string))
    #if return_ts set to True then return time series (for tsclean_dataframe calls where want the series and not the object, 
    #else return mutated class object (new class object)
    
    cleaned=pd.DataFrame(cleaned_ts)
    cleaned.columns=['cleaned']
    return cleaned
    

    
    
def season_clean(dfin):
    df=dfin.copy()
    df['month']=df.index.to_pydatetime()
    df['Date']=df.index
    df['month']=df['month'].apply(lambda x: x.month)
    average=df.groupby('month')['Total'].mean()
    average['month']=average.index
    std=df.groupby('month')['Total'].std(ddof=0)
    std['month']=std.index
    def newvalue(row):
        if row['Total']<row['Average']+row['Stdv'] and row['Total']>row['Average']-row['Stdv']:
            return row['Total']
        else:
            return row['Average']
    newdf=pd.merge(df,average,on='month')
    newdf=pd.merge(newdf,std,on='month')
    newdf.columns=['Total','month','Date','Average','Stdv']
    newdf['newvalue']=newdf.apply(newvalue,axis=1)
    newdf.index=newdf['Date']
    newdf=newdf.drop(['Total','month','Average','Stdv','Date'],axis=1)
    newdf.columns=['Total']
    #newdf=newdf.astype(int)
    newdf=newdf.sort_values('Date')
    return newdf
    


# In[ ]:




