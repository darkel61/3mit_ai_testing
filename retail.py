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
retail_csv = pd.read_csv('retail.csv')
df = pd.DataFrame(retail_csv)
df = df[df['StockCode'] == "85123A"]
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']) 
df['ts'] = df['InvoiceDate'].dt.date 
df = df.groupby('ts')['Quantity'].sum().reset_index()
df['ts'] = pd.to_datetime(df['ts']) 

df.set_index('ts', inplace=True)
all_days = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
df = df.reindex(all_days, fill_value=0)
df = df.reset_index().rename(columns={'index': 'ts'})


# Anomalias
anomaly_df = pd.DataFrame({
    # start and end date are inclusive
    # each row is an anomaly interval
    cst.START_TIME_COL: ["2011-01-11", "2011-06-09", "2011-11-03"],  # inclusive
    cst.END_TIME_COL: ["2011-01-11", "2011-06-09", "2011-11-03"],  # inclusive
    cst.ADJUSTMENT_DELTA_COL: [-2300, -3000, -500],  # mask as NA
})


# Creates anomaly_info dictionary.
# This will be fed into the template.
anomaly_info = {
    "value_col": "Quantity",
    "anomaly_df": anomaly_df,
    "adjustment_delta_col": cst.ADJUSTMENT_DELTA_COL,
}

# No es un dataframe pero pareceira un dataframe
ts = UnivariateTimeSeries()
ts.load_data(
    df=df,
    time_col="ts",
    value_col="Quantity",
    freq="D",
    anomaly_info=anomaly_info
)

fig = ts.plot()
fig.write_html("html/anomaly.html")


df = ts.df

# Preparacion de Metadata
metadata = MetadataParam(
    time_col="ts",  # name of the time column
    value_col="y",  # name of the value column
    freq="D"  # "H" for hourly, "D" for daily, "W" for weekly, etc.
)

# Cross Validation. pero creo que no hice nada con esto.
# evaluation_period = EvaluationPeriodParam(
#     test_horizon=90,
#     cv_horizon=125,
#     cv_max_splits=3,
#     cv_min_train_periods=200
# )

# Runs the forecast
forecaster = Forecaster()
result = forecaster.run_forecast_config(
    df=df,
    config=ForecastConfig(
        model_template=ModelTemplateEnum.SILVERKITE.name,
        forecast_horizon=35, 
        coverage=0.90,
        metadata_param=metadata,
        # evaluation_period_param=evaluation_period #Comentable
    )
)

forecast = result.forecast
fig = forecast.plot()
fig.write_html("html/retail_forecast.html")


# Crecimiento y Tendencia pero no se ha hecho nada con esto como tal.
# model = ChangepointDetector()
# res = model.find_trend_changepoints(
#     df=df,  # data df
#     time_col="invDate",  # time column name
#     value_col="Quantity",  # value column name
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
# fig.write_html("html/growth.html")


print(pd.DataFrame(result.backtest.test_evaluation, index=["Value"]).transpose())