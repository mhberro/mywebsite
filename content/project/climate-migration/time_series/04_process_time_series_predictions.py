#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import concurrent.futures
import csv
import multiprocessing
from pathlib import Path
import uuid

# import dask
from fbprophet import Prophet
import numpy as np
import pandas as pd


# In[ ]:


arima_path = Path('data/arima')
all_zip_codes_path = arima_path / 'all_zip_codes'

zip_code_paths = list(all_zip_codes_path.glob('*'))
zip_code_paths[:1]


# In[ ]:


dfs = []
first_loop_iteration = True
column_names = []
for index, zip_code_path in enumerate(zip_code_paths):
    if zip_code_path.name.startswith('zipcode='):
        for csv_path in zip_code_path.glob('*.csv'):
            dfs.append(
                pd.read_csv(
                    str(csv_path.resolve()),
                    dtype={
                        'zip': str,
                        'zipcode': str
                    }
                )
            )
df = pd.concat(dfs)
del df['zip']
print(df.shape)
df.head(1)


# In[ ]:


df.drop_duplicates(
    subset=['zipcode', 'year'],
    inplace=True
)
print(df.shape)
df.head(1)


# In[ ]:


df[df['zipcode'] == '00601']


# In[ ]:


df.to_csv(
    arima_path / 'NOAA_weather_forecast_predictions.csv',
    index=False,
    quoting=csv.QUOTE_NONNUMERIC
)


# In[ ]:


df.to_json(
    arima_path / 'NOAA_weather_forecast_predictions.json',
    orient='records',
    indent=2
)


# In[ ]:


df.to_json(
    arima_path / 'NOAA_weather_forecast_predictions.min.json',
    orient='records'
)


# In[ ]:


df_reloaded = pd.read_json(
    arima_path / 'NOAA_weather_forecast_predictions.min.json',
    dtype={'zipcode': str}
)


# In[ ]:


df_reloaded = pd.read_csv(
    arima_path / 'NOAA_weather_forecast_predictions.csv',
    dtype={'zipcode': str}
)
df_reloaded


# In[ ]:


df_reloaded[df_reloaded['zipcode'] == '55001']

