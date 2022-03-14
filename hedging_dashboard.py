# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import yfinance as yf
#import io
import requests as re
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import quandl

@st.cache
def getData():
    
    #url='https://raw.githubusercontent.com/rounder22/cmbs/main/cleaned%20index%20data.csv'
    #s=requests.get(url).content
    #df=pd.read_csv(io.StringIO(s.decode('utf-8')))
    df=pd.read_csv('cleaned index data.csv')
    df=df.set_index('Date').T
    df.index=pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    
    #Yahoo Data
    yahoo=yf.download(['^GSPC','IVOL','XLF','BAC','JPM'],period='5y')
    df=pd.merge(df,yahoo.Close,how='outer',left_index=True,right_index=True)
    df.rename(columns={'^GSPC':'SPX'},inplace=True)
    
    #FRED Data
    api_key='73cf3bed4794ea8277f4d43358d8ce07'
    lsSeries={'UST2Y':'DGS2','UST5Y':'DGS5','UST10Y':'DGS10','UST30Y':'DGS30',
              'Real UST10Y':'DFII10'}
    endpoint='https://api.stlouisfed.org/fred/series/observations'
    for series in lsSeries:
        params={'api_key':api_key,
                'file_type':'json',
                'series_id':lsSeries[series]
                }
        dfFRED=pd.DataFrame(re.get(endpoint,params).json()['observations'])
        dfFRED['value']=dfFRED['value'].str.replace('.','')
        dfFRED['date']=pd.to_datetime(dfFRED['date'])
        dfFRED['value']=pd.to_numeric(dfFRED['value'])
        dfFRED.drop(columns=['realtime_start','realtime_end'],inplace=True)
        dfFRED.rename(columns={'date':'Date'},inplace=True)
        dfFRED.set_index('Date',inplace=True)
        dfFRED.rename(columns={'value':series},inplace=True)
        df=pd.merge(df,dfFRED,how='outer',left_index=True,right_index=True)
    
    #Quandl Data
    quandl.ApiConfig.api_key='56v4NVWgkabqfn_Xy8Rj'
    qData=quandl.get(['LBMA/GOLD','LBMA/SILVER'])
    df=pd.merge(df,qData,how='outer',left_index=True,right_index=True)
    
    return df

st.title('Hedging Analysis')
df=getData()
seriesI=st.selectbox('Select Independent Series',df.columns,index=2)
seriesD=st.selectbox('Select Dependent Series',df.columns,index=7)
df=df[[seriesI,seriesD]].dropna()
   
fig=make_subplots(specs=[[{"secondary_y":True}]])
fig.add_trace(go.Scatter(x=df.index,y=df[seriesI].values,name=seriesI)
              ,secondary_y=False)
fig.add_trace(go.Scatter(x=df.index,y=df[seriesD].values,name=seriesD)
              ,secondary_y=True)
fig.update_xaxes(rangeslider_visible=True,
                 rangeselector=dict(
        buttons=list([
            dict(count=1, label="1m", step="month", stepmode="backward"),
            dict(count=3, label="3m", step="month", stepmode="backward"),
            dict(count=6, label="6m", step="month", stepmode="backward"),
            dict(count=1, label="YTD", step="year", stepmode="todate"),
            dict(count=1, label="1y", step="year", stepmode="backward"),
            dict(step="all")
        ])
    )
)
st.plotly_chart(fig)

with st.form('rolling'):
    st.subheader('Rolling Regession Analysis')
    startDate=st.date_input('Start Date',
                            value=df.index.min(),
                            min_value=df.index.min(),
                            max_value=df.index.max()
                            )
    endDate=st.date_input('End Date',
                          value=df.index.max(),
                          min_value=df.index.min(),
                          max_value=df.index.max()
                          )
    df=df.reindex(pd.bdate_range(start=startDate,end=endDate))
    col1, col2, col3, col4=st.columns(4)
    d1=col2.number_input('Enter Number of Days',min_value=2,step=1,value=5)
    d2=col3.number_input('Enter Number of Days',min_value=2,step=1,value=10)
    d3=col4.number_input('Enter Number of Days',min_value=2,step=1,value=20)
    
    submitted1=st.form_submit_button('Submit')
    
    if submitted1:
        df1=df[[seriesI,seriesD]].dropna()
        df1=df1.astype('float')
        
        y=df1[seriesD][np.logical_not(np.isnan(df1[seriesD]))]
        x=df1[seriesI][np.logical_not(np.isnan(df1[seriesI]))]
        x=x.values.reshape(-1,1)
        linear_regressor=LinearRegression()
        linear_regressor.fit(x,y)
        x_range=np.linspace(x.min(),x.max(),100)
        y_range=linear_regressor.predict(x_range.reshape(-1,1))

        col1.write('')
        col1.write('')
        col1.write('')
        col1.write('')
        col1.write('')
        col1.write('')
        col1.markdown('**Regression Statistics**')
        col1.write('Coef: '+"{:0.2f}".format(linear_regressor.coef_[0]))
        col1.write('Intercept: '+"{:0.2f}".format(linear_regressor.intercept_))
        col1.write('Correlation: '+"{:0.2f}".format(linear_regressor.score(x,y)))
        col1.write('Observations: '+str(len(x)))
        col1.write('')
        fig1=px.scatter(df1,x=seriesI,y=seriesD)
        fig1.add_trace(go.Scatter(x=x_range,y=y_range,name='Regression Fit'))
        st.plotly_chart(fig1)
        
        col2.write(str(d1)+'d Correlations')
        s1=df1[seriesI].rolling(d1).corr(df1[seriesD])
        col2.write('Last: '+"{:0.2f}".format(s1[-1]))
        col2.write('Mean: '+"{:0.2f}".format(s1.mean()))
        col2.write('Median: '+"{:0.2f}".format(s1.median()))
        col2.write('StDev: '+"{:0.2f}".format(s1.std()))
        col2.write('Min:'+"{:0.2f}".format(s1.min()))
        col2.write('Max:'+"{:0.2f}".format(s1.max()))
        col2.write('')
        fig2=px.histogram(s1,title=str(d1)+'d Correlations')
        st.plotly_chart(fig2)
        
        col3.write(str(d2)+'d Correlations')
        s2=df1[seriesI].rolling(d2).corr(df1[seriesD])
        col3.write('Last: '+"{:0.2f}".format(s2[-1]))
        col3.write('Mean: '+"{:0.2f}".format(s2.mean()))
        col3.write('Median: '+"{:0.2f}".format(s2.median()))
        col3.write('StDev: '+"{:0.2f}".format(s2.std()))
        col3.write('Min:'+"{:0.2f}".format(s2.min()))
        col3.write('Max:'+"{:0.2f}".format(s2.max()))
        col3.write('')
        fig3=px.histogram(s2,title=str(d2)+'d Correlations')
        st.plotly_chart(fig3)
        
        col4.write(str(d2)+'d Correlations')
        s3=df1[seriesI].rolling(d3).corr(df1[seriesD])
        col4.write('Last: '+"{:0.2f}".format(s3[-1]))
        col4.write('Mean: '+"{:0.2f}".format(s3.mean()))
        col4.write('Median: '+"{:0.2f}".format(s3.median()))
        col4.write('StDev: '+"{:0.2f}".format(s3.std()))
        col4.write('Min:'+"{:0.2f}".format(s3.min()))
        col4.write('Max:'+"{:0.2f}".format(s3.max()))
        col4.write('')
        fig4=px.histogram(s3,title=str(d3)+'d Correlations')
        st.plotly_chart(fig4)