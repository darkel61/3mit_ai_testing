import datetime

import numpy as np
import pandas as pd
import plotly
import os


from greykite.algo.changepoint.adalasso.changepoint_detector import ChangepointDetector
from greykite.algo.forecast.silverkite.constants.silverkite_holiday import SilverkiteHoliday
from greykite.algo.forecast.silverkite.constants.silverkite_seasonality import SilverkiteSeasonalityEnum
from greykite.algo.forecast.silverkite.forecast_simple_silverkite_helper import cols_interact
from greykite.common import constants as cst
from greykite.common.features.timeseries_features import build_time_features_df
from greykite.common.features.timeseries_features import convert_date_to_continuous_time
from greykite.framework.benchmark.data_loader_ts import DataLoaderTS
from greykite.framework.templates.autogen.forecast_config import EvaluationPeriodParam
from greykite.framework.templates.autogen.forecast_config import ForecastConfig
from greykite.framework.templates.autogen.forecast_config import MetadataParam
from greykite.framework.templates.autogen.forecast_config import ModelComponentsParam
from greykite.framework.templates.forecaster import Forecaster
from greykite.framework.templates.model_templates import ModelTemplateEnum
from greykite.framework.utils.result_summary import summarize_grid_search_results
from greykite.framework.input.univariate_time_series import UnivariateTimeSeries

# Preparacion de Data Frame
retail_csv = pd.read_csv('retail_many.csv')
full_df = pd.DataFrame(retail_csv)
full_df.rename(columns={'fecha_venta': 'ts', 'cantidad': 'y', 'precio_venta': 'sale_price'}, inplace=True)
products = full_df['codigo_articulo'].unique()
products_name = full_df['producto'].unique()

for index, product in enumerate(products):
    df = full_df[full_df['codigo_articulo'] == product]


    df['ts'] = pd.to_datetime(df['ts']) 
    df['ts'] = df['ts'].dt.date 

    branches = df['codigo_sucursal'].unique()

    for branch in branches:
        b_df = df[df['codigo_sucursal'] == branch]
        b_df = b_df.groupby('ts').agg({
            'y': 'sum',
            'sale_price': 'last'  # Choose how to aggregate other columns
        }).reset_index()
        b_df['ts'] = pd.to_datetime(b_df['ts']) 

        b_df.set_index('ts', inplace=True)
        all_days = pd.date_range(start=b_df.index.min(), end=b_df.index.max(), freq='D')
        b_df = b_df.reindex(all_days, fill_value=20)
        b_df = b_df.reset_index().rename(columns={'index': 'ts'})



        # Anomalias
        # Calculate the Z-score
        b_df['z_score'] = (b_df['y'] - b_df['y'].mean()) / b_df['y'].std()
        # Define a threshold for identifying anomalies
        threshold = 3
        negative_threshold = 1

        # Filter anomalies
        anomalies = b_df[np.abs(b_df['z_score']) > threshold]
        anomalies = b_df[np.abs(b_df['z_score']) < negative_threshold]
        anomaly_df = pd.DataFrame({
            # start and end date are inclusive
            # each row is an anomaly interval
            cst.START_TIME_COL: list(anomalies['ts']),  # inclusive
            cst.END_TIME_COL: list(anomalies['ts']),  # inclusive
            cst.ADJUSTMENT_DELTA_COL: [int(num*-1) for num in list(anomalies['z_score'] * b_df['y'].std())],  # mask as NA
        })

        # # Creates anomaly_info dictionary.
        # # This will be fed into the template.
        anomaly_info = {
            "value_col": "y",
            "anomaly_df": anomaly_df,
            "adjustment_delta_col": cst.ADJUSTMENT_DELTA_COL,
        }

        last_price = b_df['sale_price'].iloc[-1] #<-obtenemos el ultimo precio

        # Agregar el ultimo precio al dataframe
        fechas_futuras = pd.date_range(start=b_df['ts'].max() + pd.DateOffset(1), periods=35)
        precios_futuros = [last_price]*35

        df_futuros = pd.DataFrame({'ts': fechas_futuras, 'sale_price': precios_futuros})
        b_df = pd.concat([b_df, df_futuros], ignore_index=True)

        # No es un dataframe pero pareceira un dataframe
        ts = UnivariateTimeSeries()
        ts.load_data(
            df=b_df,
            time_col="ts",
            value_col="y",
            freq="D",
            anomaly_info=anomaly_info
        )

        # Preparacion de Metadata
        metadata = MetadataParam(
            time_col="ts",  # name of the time column
            value_col="y",  # name of the value column
            freq="D"  # "H" for hourly, "D" for daily, "W" for weekly, etc.
        )

        # Runs the forecast
        try:
            forecaster = Forecaster()
            regressors = {
                "regressor_cols": ["sale_price"]
            }

            model_components = ModelComponentsParam(
                regressors=regressors, 
            )

            result = forecaster.run_forecast_config(
                df=b_df,
                config=ForecastConfig(
                    model_template=ModelTemplateEnum.SILVERKITE.name,
                    forecast_horizon=35, 
                    coverage=0.90,
                    metadata_param=metadata,
                    model_components_param=model_components,
                )
            )

            forecast = result.forecast
            fig = forecast.plot()
            fig.write_html(f"./html/{branch}-{products_name[index]}-{product}.html")



            print(f"html/{branch}-{products_name[index]}-{product}.html")
            print(pd.DataFrame(result.backtest.test_evaluation, index=["Value"]).transpose())
        except Exception as e :
            print(e)
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print(f" HAS FAILED html/{branch}-{products_name[index]}-{product}.html")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
            print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")