#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import concurrent.futures
import csv
import multiprocessing
from pathlib import Path
import uuid

from fbprophet import Prophet
import numpy as np
import pandas as pd
from sklearn.metrics import *


# ## Find zip codes with > 20,000 rows in source data

# In[ ]:


data_cleaned_path = Path('data/cleaned')

arima_path = Path('data/arima')
all_zip_codes_path = arima_path / 'all_zip_codes_test'

zip_code_paths = list(all_zip_codes_path.glob('*'))
print(zip_code_paths[:1])

ATTRIBUTES = ['PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN', 'TOBS']
PREDICTION_YEARS = ['2019']


# In[ ]:


df_reloaded = pd.read_parquet(f"{Path('data/cleaned') / 'all_zip_codes'}/")
df_reloaded.head()


# In[ ]:


df_counts = (
    df_reloaded
    .groupby(['zip'])
    .agg({'date': 'count'})
    .sort_values(by=['date'])
)
print(df_counts.head())
print(df_counts.tail())


# In[ ]:


df_filtered = (
    df_counts[df_counts['date'] >= 20_000]
    .reset_index()
    .rename(columns={'date': 'count'})
)
df_filtered.head()


# ## Train till 2009. Predict till 2019

# In[ ]:


clean_source_data_zip_code_path = Path('data/cleaned/all_zip_codes')
all_zip_codes_path = Path('data/arima/all_zip_codes_test')

clean_source_data_zip_code_paths = list(clean_source_data_zip_code_path.glob('*'))


# In[ ]:


def make_predictions(zip_code_path):
    output_folder = all_zip_codes_path / zip_code_path.name
    if output_folder.exists():
        print(f"Skipping {zip_code_path.name}")
        return
    print(f"Making predictions for {zip_code_path.name}")

    original_data = pd.read_parquet(zip_code_path.resolve())
    if original_data['zip'].iloc[0] not in df_filtered['zip'].tolist():
        return
    
    original_data = original_data[
        original_data['date'] < pd.Timestamp('2010-01-01')
    ]
    
#     print(f"zip: {original_data.iloc[0]['zip']}")
#     print(original_data.head(1))
    output_data = {
        'zip': [original_data.iloc[0]['zip']] * len(PREDICTION_YEARS),
        'zipcode': [original_data.iloc[0]['zip']] * len(PREDICTION_YEARS),
        'year': PREDICTION_YEARS
    }
    for attribute in ATTRIBUTES:
#         print(attribute)
        input_data = original_data[['date', attribute]].copy()
        input_data.rename(
            columns={
                'date': 'ds',
                attribute: 'y'
            },
            inplace=True
        )
#         print(input_data.head(1))
#         print(input_data.tail(1))

        prophet_model = (
            Prophet(
                daily_seasonality=False,
                yearly_seasonality=False
            )
            .fit(input_data)
        )

        max_date = original_data['date'].max()
#         print(max_date)
        predictions = (
            pd.date_range(
                start=pd.Timestamp('2010-01-01'),
                end=pd.Timestamp('2019-12-31'),
                name='ds'
            )
            .to_frame(index=False)
        )
#         print(predictions.head(1))
#         print(predictions.tail(1))
        predictions = prophet_model.predict(predictions)
#         print(predictions.head(1))
#         print(predictions.tail(1))

        yearly_predictions = []
        for prediction_year in PREDICTION_YEARS:
            predictions_subset = predictions.loc[
                (predictions['ds'] >= f"{prediction_year}-01-01")
                & (predictions['ds'] <= f"{prediction_year}-12-31T23:59:59")
            ]
#             print(predictions_subset['yhat'].mean())
            yearly_predictions.append(predictions_subset['yhat'].mean())
            output_data[attribute] = yearly_predictions

    output_df = pd.DataFrame(output_data)
    output_folder.mkdir()
    output_df.to_csv(
        output_folder / f"{str(uuid.uuid1())}.csv",
        index=False,
        quoting=csv.QUOTE_NONNUMERIC
    )


# In[ ]:


for index, zip_code_path in enumerate(clean_source_data_zip_code_paths):
    if index <= 0: # if zip_code_path.name == 'zipcode=37801':
        print('hi')
        make_predictions(zip_code_path)


# In[ ]:


with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
    for zip_code_path, _ in zip(
            clean_source_data_zip_code_paths,
            executor.map(make_predictions, clean_source_data_zip_code_paths)
        ):
        pass


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
df_predictions = pd.concat(dfs)
del df_predictions['zip']
print(df_predictions.shape)
df_predictions.head(3)


# ## Get accuracy for 2019

# ### Put original dataframe in yearly format

# In[ ]:


df_reloaded['year'] = df_reloaded['date'].dt.year
df_reloaded.head(3)


# In[ ]:


df_reloaded_yearly = (
    df_reloaded
    .groupby(['zip', 'year'])
    .agg({
        'PRCP': 'mean',
        'SNOW': 'mean',
        'SNWD': 'mean',
        'TMAX': 'mean',
        'TMIN': 'mean',
        'TOBS': 'mean'
    })
    .reset_index()
    .rename(columns={'zip': 'zipcode'})
)
df_reloaded_yearly = df_reloaded_yearly[df_reloaded_yearly['year'] == 2019]
print(df_reloaded_yearly.shape)
df_reloaded_yearly.head(3)


# ### Get accuracies

# In[ ]:


df_both = (
    df_predictions
    .merge(
        df_reloaded_yearly,
        on='zipcode',
        suffixes=('_pred', '')
    )
)
print(df_both.shape)
df_both.head(3)


# In[ ]:


for attribute in ATTRIBUTES:
    print(f"Attribute: {attribute}")
    rmse = mean_squared_error(
        y_true=df_both[attribute],
        y_pred=df_both[f"{attribute}_pred"],
        squared=False
    )
    mae = mean_absolute_error(
        y_true=df_both[attribute],
        y_pred=df_both[f"{attribute}_pred"]
    )
    r2 = r2_score(
        y_true=df_both[attribute],
        y_pred=df_both[f"{attribute}_pred"]
    )
    print(f"RMSE: {round(rmse, 2)}")
    print(f"MAE: {round(mae, 2)}")
    print(f"R^2: {round(r2, 2)}")
    print()
print()
print("These metrics were calculated by taking the data up to 2009 and predicting 10 years out until 2019. The 2019 predictions were compared against the 2019 yearly averages for each zip code in our actual data.")

