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

retail_csv = pd.read_csv('retail.csv')

df = pd.DataFrame(retail_csv)

df2 = df
df = df[df['StockCode'] == "85123A"]

# df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']) 
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate']) 
df['invDate'] = df['InvoiceDate'].dt.date 


df = df.groupby('invDate')['Quantity'].sum().reset_index()

# df = df.query('StockCode == "85123A"')

# df.agg({})
# # Loads dataset into UnivariateTimeSeries
# dl = DataLoaderTS()
# ts = dl.load_peyton_manning_ts()
# df = ts.df  # cleaned pandas.DataFrame

# print(ts.describe_time_col())
# print(ts.describe_value_col())


# # Preparar Grafica

# fig = df.plot()
# # # Imprimir grafica, no funciona en WSL
# # # plotly.io.show(df)
# fig.write_html("html/retail_data.html")

metadata = MetadataParam(
    time_col="invDate",  # name of the time column
    value_col="Quantity",  # name of the value column
    freq="D"  # "H" for hourly, "D" for daily, "W" for weekly, etc.
)

forecaster = Forecaster()
result = forecaster.run_forecast_config(
    df=df,
    config=ForecastConfig(
        model_template=ModelTemplateEnum.SILVERKITE.name,
        forecast_horizon=228,  
        coverage=0.95, 
        metadata_param=metadata
    )
)

forecast = result.forecast
fig = forecast.plot()
# # plotly.io.show(fig)
fig.write_html("html/retail_forecast.html")


print(pd.DataFrame(result.backtest.test_evaluation, index=["Value"]).transpose())

# # Sirve para sacar estadisticas
# pd.DataFrame(result.backtest.test_evaluation, index=["Value"]).transpose()  # formats dictionary as a pd.DataFrame
