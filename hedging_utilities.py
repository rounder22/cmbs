# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 11:35:23 2023

@author: joshm
"""
import pandas as pd

def inputData(file):
    df=pd.read_csv(file)
    df=df.set_index('Date').T
    df.index=pd.to_datetime(df.index)
    old=pd.read_pickle('index_data.pkl')
    old=old.append(df)
    old.to_pickle('index_data.pkl')
  