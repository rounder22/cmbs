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
import io
import requests

url='https://raw.githubusercontent.com/rounder22/cmbs/main/cleaned%20index%20data.csv'
s=requests.get(url).content
df=pd.read_csv(io.StringIO(s.decode('utf-8')))
df=df.set_index('Date').T
df.index=pd.to_datetime(df.index)
spx=yf.download('^GSPC',period='3y')
df=pd.merge(df,spx.Close.to_frame(),how='left',left_index=True,right_index=True)
df.rename(columns={'Close':'SPX'},inplace=True)

st.title('CMBS Hedging Analysis')

with st.form('Regression_Analysis'):
    seriesI=st.selectbox('Select Independent Series',df.columns)
    seriesD=st.selectbox('Select Dependent Series',df.columns)    
    submitted=st.form_submit_button('Submit')
    
    if submitted:
        df1=df[[seriesI,seriesD]].dropna()
        df1=df1.astype('float')
        y=df1[seriesD][np.logical_not(np.isnan(df1[seriesD]))]
        x=df1[seriesI][np.logical_not(np.isnan(df1[seriesI]))]
        x=x.values.reshape(-1,1)
        linear_regressor=LinearRegression()
        linear_regressor.fit(x,y)

        st.header('Regression Statistics')
        st.write(' Coef:',linear_regressor.coef_[0],'\n',
                 'Intercept:',linear_regressor.intercept_,'\n',
                 'Correlation:',linear_regressor.score(x,y),'\n',
                 'Observations:',len(x))
        
with st.form('Rolling_Correlation_Analysis'):
    seriesI=st.selectbox('Select Independent Series',df.columns)
    seriesD=st.selectbox('Select Dependent Series',df.columns)
    d=st.number_input('Enter Number of Days',min_value=0,step=1,value=5)
    
    submitted=st.form_submit_button('Submit')
    
    if submitted:
        df1=df[[seriesI,seriesD]].dropna()
        df1=df1.astype('float')
        st.header('Rolling Correlations')
        s=df1[seriesI].rolling(d).corr(df1[seriesD])
        st.write(' Last:',s[-1],'\n',
                 'Mean:',s.mean(),'\n',
                 'Median:',s.median(),'\n',
                 'StDev:',s.std(),'\n',
                 'Min:',s.min(),'\n',
                 'Max:',s.max())
