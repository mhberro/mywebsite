#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv
from pathlib import Path

import numpy as np
import pandas as pd


# In[ ]:


Path().resolve()


# In[ ]:


data_cleaned_path = Path('data/cleaned')


# In[ ]:


first_loop_iteration = True
column_names = ['date', 'zipcode', 'zip', 'PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN', 'TOBS']
for folder_path in Path('data/cleaned').iterdir():
    if folder_path.is_dir():
        if folder_path.name.startswith('zip_code_result_small'):
            if first_loop_iteration:
                df = pd.read_parquet(str(folder_path.resolve()))
                df = df[column_names].copy()
                first_loop_iteration = False
            else:
                df_next = pd.read_parquet(str(folder_path.resolve()))
                df_next = df_next[column_names].copy()
                df = pd.concat([df, df_next])
                
print(df.head(3))
print(df.tail(3))


# In[ ]:


df['zipcode'] = df['zip']


# In[ ]:


df.drop_duplicates(
    subset=['zipcode', 'date'],
    inplace=True
)


# In[ ]:


df.to_parquet(
    f"data/cleaned/all_zip_codes/",
    partition_cols=['zipcode']
)


# In[ ]:


df.to_csv(
    "data/cleaned/all_zip_codes.csv",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC
)


# In[ ]:


df_reloaded = pd.read_parquet("data/cleaned/all_zip_codes/")
df_reloaded.head()


# In[ ]:


(
    df_reloaded
    .groupby(['zip'])
    .agg({'date': 'count'})
    .sort_values(by=['date'])
    .to_csv('data/zipcode_counts.csv', quoting=csv.QUOTE_NONNUMERIC)
)


# In[ ]:


df_reloaded[df_reloaded['zip'] == '48301'].to_csv(
    "data/cleaned/all_zip_codes_48301.csv",
    index=False,
    quoting=csv.QUOTE_NONNUMERIC
)


# In[ ]:


df_from_csv = pd.read_csv(
    "data/cleaned/all_zip_codes.csv",
    quoting=csv.QUOTE_NONNUMERIC
)
df_from_csv.head()


# In[ ]:




