# -*- coding: utf-8 -*-
"""
Utility functions for the hedging dashboard
"""
#from git import Repo
#import io
#url='https://raw.githubusercontent.com/rounder22/cmbs/main/cleaned%20index%20data.csv'
#s=requests.get(url).content
#df=pd.read_csv(io.StringIO(s.decode('utf-8')))

import pandas as pd

def inputData(file):
    df=pd.read_csv(file)
    df=df.set_index('Date').T
    df.index=pd.to_datetime(df.index)
    old=pd.read_pickle('index_data.pkl')
    old=old.append(df)
    old.to_pickle('index_data.pkl')
  