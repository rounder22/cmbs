# -*- coding: utf-8 -*-
"""
Utility functions for TD Ameritrade API
"""

import requests as r
import json
import pandas as pd

#GLOBALS
apiKey='M9YPA8SDNNXYJVWTC3MDQF9ZU8EO93JC'

def get_price_history(ticker,periodType,period,freq,freqType,*args):
    """
    

    Parameters
    ----------
    tickers : str
        security symbol
    periodType : string
        day,month,year, or ytd are accepted.
    period : int
        the number of periods.
    freq : int
        the number of frequency type to display for a period.
    freqType : str
        minute,daily,weekly,monthly are accepted.
    *args: list
        list of symbols
    
    Returns
    -------
    dataframe with OHLC price history for given security or a multi-index dataframe with OHLC price history
    for a list of symbols

    """ 
     
    endpoint='https://api.tdameritrade.com/v1//marketdata/%s/pricehistory'%ticker

    data=json.loads(r.get(endpoint,
                          params={'apikey':apiKey,
                                  'periodType':periodType,
                                  'period':period,
                                  'frequencyType':freqType,
                                  'frequency':freq
                                  }
                          ).content
                   )
    df=pd.DataFrame(data['candles'])
    df['datetime']=pd.to_datetime(df['datetime'],unit='ms')
    df.set_index('datetime',inplace=True)
    df.index=df.index.normalize()
    df.columns=pd.MultiIndex.from_tuples([(ticker,a) for a in df.columns])
    df.index.name=ticker
    
    return df

def get_div(ticker):
    """
    

    Parameters
    ----------
    ticker : str
        ticker

    Returns
    -------
    None.

    """
    endpoint='https://api.tdameritrade.com/v1/instruments'


    data=json.loads(r.get(endpoint,
                      params={'apikey':apiKey,
                              'symbol':ticker,
                              'projection':'fundamental'
                             }
                     ).content
               )
    
    return data[ticker]['fundamental']['dividendAmount']
    