#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import csv

import numpy as np
import pandas as pd


# In[ ]:


# Done
# FILENAME_WITHOUT_EXTENSION = 'zip_code_result_small_001-100_801-895_new'
# FILENAME_WITHOUT_EXTENSION = 'zip_code_result_small_201-300'
# FILENAME_WITHOUT_EXTENSION = 'zip_code_result_small_301-400'
# FILENAME_WITHOUT_EXTENSION = 'zip_code_result_small_101-200'
# FILENAME_WITHOUT_EXTENSION = 'zip_code_result_small_401-500'
# FILENAME_WITHOUT_EXTENSION = 'zip_code_result_small_501-700'
# FILENAME_WITHOUT_EXTENSION = 'zip_code_result_small_701-800'

# Not Done

# Won't Do
# FILENAME_WITHOUT_EXTENSION = 'zip_code_result_small_001-100_800-895'

VARIABLES_TO_KEEP = ['PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN', 'TOBS']


# In[ ]:


df = pd.read_csv(
    f"data/raw_results/{FILENAME_WITHOUT_EXTENSION}.csv",
    header=0,
    names=['zipcode', 'date', 'variable', 'unknown_a', 'value', 'unknown_b'],
    # names=['zipcode', 'date', 'variable', 'unknown_a', 'unknown_b', 'value'],
    dtype={'zipcode': str}
)
df['date'] = df['date'].astype('datetime64')
df.head(5)


# In[ ]:


df.drop_duplicates(
    subset=['zipcode', 'date', 'variable'],
    inplace=True
)


# In[ ]:


df['new_index'] = df['zipcode'] + '_' + df['date'].astype(str)
df.head(3)


# In[ ]:


df_pivot = (
    df
    .pivot_table(
        index='new_index',
        columns='variable',
        values='value'
    )
    .reset_index()
)
df_pivot.head(3)


# In[ ]:


df_pivot['zipcode'] = df_pivot['new_index'].apply(lambda a: a.split('_')[0])
df_pivot['zip'] = df_pivot['zipcode']
df_pivot['date'] = df_pivot['new_index'].apply(lambda a: a.split('_')[1])
df_pivot['date'] = df_pivot['date'].astype('datetime64')
df_pivot = df_pivot[['zipcode', 'zip', 'date'] + VARIABLES_TO_KEEP]
df_pivot.columns.name = None
print(df_pivot.shape)
df_pivot.head(3)


# In[ ]:


all_dates = df_pivot['date'].unique()
all_dates.sort()
all_dates = (
    pd.DataFrame(all_dates)
    .rename(columns={0: 'date'})
)
all_dates.head(3)


# In[ ]:


df_pivot = all_dates.merge(
    df_pivot,
    how='left',
    on='date'
)
print(df_pivot.shape)
df_pivot.head(3)


# In[ ]:


df_pivot.sort_values(
    by=['zipcode', 'date'],
    inplace=True
)
df_pivot.head(3)


# In[ ]:


for column_name in VARIABLES_TO_KEEP:
    df_pivot[column_name] = df_pivot[column_name].fillna(method='backfill')
    df_pivot[column_name] = df_pivot[column_name].fillna(method='ffill')
df_pivot.head(3)


# In[ ]:


df_pivot.to_parquet(
    f"data/cleaned/{FILENAME_WITHOUT_EXTENSION}/",
    partition_cols=['zipcode']
)


# In[ ]:


# df_reloaded = pd.read_parquet(f"data/cleaned/{FILENAME_WITHOUT_EXTENSION}/zipcode=54001")
df_reloaded = pd.read_parquet(f"data/cleaned/{FILENAME_WITHOUT_EXTENSION}/")
df_reloaded.head()


# In[ ]:




