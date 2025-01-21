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
retail_csv = pd.read_csv('./retail_many.csv')
full_df = pd.DataFrame(retail_csv)
full_df.rename(columns={
    'fecha_venta': 'ts', 
    'cantidad': 'y', 
    'precio_venta': 'sale_price', 
    'codigo_articulo': 'article_id', 
    'producto': 'article_name'}, inplace=True)
full_df['ts'] = pd.to_datetime(full_df['ts'])

# Establece la columna 'fecha' como índice
full_df.set_index('ts', inplace=True)

# Agrupa por semanas y por producto
full_df = full_df.groupby([pd.Grouper(freq='W'), 'article_id']).agg({'y': 'sum', 'article_name': 'first', 'sale_price': 'first'}).reset_index()
print (full_df)
products = full_df['article_id'].unique()
products_name = full_df['article_name'].unique()
# silverkite defaults
# events = dict(
#     holidays_to_model_separately="auto",
#     holiday_lookup_countries=["Venezuela"],
#     holiday_pre_num_days=0,
#     holiday_post_num_days=0,
#     holiday_pre_post_num_dict=None,
#     daily_event_df_dict=None
# )

for index, product in enumerate(products):
    runtime = datetime.datetime.now()
    df = full_df[full_df['article_id'] == product]
    df = df.groupby('ts').agg({
        'y': 'sum',
        'sale_price': 'last'  # Choose how to aggregate other columns
    }).reset_index()
    
    df['ts'] = pd.to_datetime(df['ts']) 
    df['ts'] = df['ts'].dt.date 


    df.set_index('ts', inplace=True)
    all_days = pd.date_range(start=df.index.min(), end=df.index.max(), freq='W')
    df = df.reindex(all_days, fill_value=20)
    df = df.reset_index().rename(columns={'index': 'ts'})
    
    

    # Anomalias
    # Calculate the Z-score
    # TODO: Investigar normalización. Analisis logaritmico.
    df['z_score'] = (df['y'] - df['y'].mean()) / df['y'].std()
    # Define a threshold for identifying anomalies
    threshold = 1 

    print(df.describe())

    # Filter anomalies
    anomalies = df[np.abs(df['z_score']) > threshold]
    anomaly_df = pd.DataFrame({
        cst.START_TIME_COL: list(anomalies['ts']),
        cst.END_TIME_COL: list(anomalies['ts']),
        cst.ADJUSTMENT_DELTA_COL: [int(num*-1) for num in list(anomalies['z_score'] * df['y'].std())],  # mask as NA
    })

    anomaly_info = {
        "value_col": "y",
        "anomaly_df": anomaly_df,
        "adjustment_delta_col": cst.ADJUSTMENT_DELTA_COL,
    }
    
    # No es un dataframe pero pareceira un dataframe5
    ts = UnivariateTimeSeries()
    ts.load_data(
        df=df,
        time_col="ts",
        value_col="y",
        freq="W",
        #anomaly_info=anomaly_info,
        regressor_cols=["sale_price"]
    )

    # fig = ts.plot_quantiles_and_overlays(
    #     groupby_time_feature="dom",
    #     show_mean=True,
    #     show_quantiles=False,
    #     show_overlays=True,
    #     overlay_label_time_feature="month",
    #     overlay_style={"line": {"width": 1}, "opacity": 0.5},
    #     center_values=True,
    #     xlabel="day of month",
    #     ylabel=ts.original_value_col,
    #     title="monthly seasonality for each month (centered)",
    # )
    # fig.write_html(f"html/ANOMALIA{products_name[index]}-{product}.html")


    # fig = ts.plot()
    # fig.write_html(f"html/{products_name[index]}-{product}.html")


    # Comenta esta linea si queires matar la anomalia!
    df = ts.df

    # Preparacion de Metadata
    metadata = MetadataParam(
        time_col="ts",  # name of the time column
        value_col="y",  # name of the value column
        freq="W"  # "H" for hourly, "D" for daily, "W" for weekly, etc.
    )

    # Cross Validation. 
    # evaluation_period = EvaluationPeriodParam(
    #     test_horizon=90,
    #     cv_horizon=125,
    #     cv_max_splits=3,
    #     cv_min_train_periods=200
    # )

    # Runs the forecast
    try:
        forecaster = Forecaster()

        # Regresores
        last_price = df['sale_price'].iloc[-1] 

        # Agregar el ultimo precio al dataframe
        future_dates = pd.date_range(start=df['ts'].max() + pd.DateOffset(7), periods=4, freq='W')
        future_prices = [last_price]*4

        df_futuros = pd.DataFrame({'ts': future_dates, 'sale_price': future_prices})
        df = pd.concat([df, df_futuros], ignore_index=True)

        regressors = {
            "regressor_cols": ["sale_price"]
        }

        model_components = ModelComponentsParam(
            regressors=regressors, 
            # events=events,
            #Con esto comentado funciona mejor la margarina, pero en el caso contrario mejor los huevos a granel. 
            # seasonality={
            #     "yearly_seasonality": 15,
            #     "monthly_seasonality": 2,
            # },
            growth={
                "growth_term": "linear"
            },
            changepoints={
                "changepoints_dict": dict(
                    method="auto",
                    regularization_strength=0.7,
                    potential_changepoint_n=25, 
                )
            },
            custom={
                "fit_algorithm_dict": {
                    "fit_algorithm": "ridge"
                }
            }
        )

        result = forecaster.run_forecast_config(
            df=df,
            config=ForecastConfig(
                model_template=ModelTemplateEnum.SILVERKITE.name,
                forecast_horizon=35, 
                coverage=0.90,
                metadata_param=metadata,
                model_components_param=model_components,
                # evaluation_period_param=evaluation_period 
            )
        )

        # }Writing the model to a file.
        print('// Dumped //')
        forecaster.dump_forecast_result(
            destination_dir='./results',
            object_name=f'forecast-{product}',
            dump_design_info=True,
            overwrite_exist_dir=True,
        )
        print('// Dumped 2 //')

        forecast = result.forecast
        fig = forecast.plot()
        fig.write_html(f"./html/weekly-{products_name[index]}-{product}.html")


        # Crecimiento y Tendencia pero no se ha hecho nada con esto como tal.
        # model = ChangepointDetector()
        # res = model.find_trend_changepoints(
        #     df=df,  # data df
        #     time_col="ts",  # time column name
        #     value_col="y",  # value column name
        #     yearly_seasonality_order=10,  # yearly seasonality order, fit along with trend
        #     regularization_strength=0.5,  # between 0.0 and 1.0, greater values imply fewer changepoints, and 1.0 implies no changepoints
        #     resample_freq="7D",  # data aggregation frequency, eliminate small fluctuation/seasonality
        #     potential_changepoint_n=25,  # the number of potential changepoints
        #     yearly_seasonality_change_freq="365D",  # varying yearly seasonality for every year
        #     no_changepoint_distance_from_end="365D")  # the proportion of data from end where changepoints are not allowed
        # fig = model.plot(
        #     observation=True,
        #     trend_estimate=False,
        #     trend_change=True,
        #     yearly_seasonality_estimate=False,
        #     adaptive_lasso_estimate=True,
        #     plot=False)
        # fig.write_html(f"html/growth-{products_name[index]}-{product}.html")

        print(f"html/{products_name[index]}-{product}.html")
        print(pd.DataFrame(result.backtest.test_evaluation, index=["Value"]).transpose())
        print("Tarde Esto en Correr!:", datetime.datetime.now() - runtime)
        # break
    except Exception as e :
        print(e)
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print(f" HAS FAILED html/{products_name[index]}-{product}.html")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
        print("~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
