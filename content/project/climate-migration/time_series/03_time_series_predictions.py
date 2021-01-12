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


# In[ ]:


NUM_CPU_CORES = multiprocessing.cpu_count()


# In[ ]:


ATTRIBUTES = ['PRCP', 'SNOW', 'SNWD', 'TMAX', 'TMIN', 'TOBS']
# ATTRIBUTES = ['TMIN']
PREDICTION_YEARS = ['2030', '2040', '2050', '2060', '2070']


zip_codes_path = Path('data/cleaned/all_zip_codes')
arima_path = Path('data/arima/all_zip_codes')

zip_code_paths = list(zip_codes_path.glob('*'))


# In[ ]:


def make_predictions(zip_code_path):
    output_folder = arima_path / zip_code_path.name
    if output_folder.exists():
        # print(f"Skipping {zip_code_path.name}")
        return
    print(f"Making predictions for {zip_code_path.name}")

    original_data = pd.read_parquet(zip_code_path.resolve())
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
                start=max_date+pd.Timedelta(days=1),
                end=pd.Timestamp('2070-12-31'),
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
            print(predictions_subset['yhat'].mean())
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


for index, zip_code_path in enumerate(zip_code_paths):
    if zip_code_path.name == 'zipcode=37801': # if index <= 0:
        print('hi')
        make_predictions(zip_code_path)


# In[ ]:


with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_CPU_CORES) as executor:
    for zip_code_path, _ in zip(zip_code_paths, executor.map(make_predictions, zip_code_paths)):
        pass


# In[ ]:




