import datetime

import numpy as np
import pandas as pd
import plotly

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


# # Loads dataset into UnivariateTimeSeries
# dl = DataLoaderTS()
# ts = dl.load_peyton_manning_ts()
# df = ts.df  # cleaned pandas.DataFrame

# print(ts.describe_time_col())
# print(ts.describe_value_col())


# # Preparar Grafica

# fig = ts.plot()
# # Imprimir grafica, no funciona en WSL
# # plotly.io.show(fig)
# fig.write_html("html/peyton_data.html")

# metadata = MetadataParam(
#     time_col="ts",  # name of the time column
#     value_col="y",  # name of the value column
#     freq="D"  # "H" for hourly, "D" for daily, "W" for weekly, etc.
# )

# forecaster = Forecaster()
# result = forecaster.run_forecast_config(
#     df=df,
#     config=ForecastConfig(
#         model_template=ModelTemplateEnum.SILVERKITE.name,
#         forecast_horizon=365,  # forecasts 365 steps ahead
#         coverage=0.95,  # 95% prediction intervals
#         metadata_param=metadata
#     )
# )

# forecast = result.forecast
# fig = forecast.plot()
# # plotly.io.show(fig)
# fig.write_html("html/peyton_forecast.html")


# # Sirve para sacar estadisticas
# pd.DataFrame(result.backtest.test_evaluation, index=["Value"]).transpose()  # formats dictionary as a pd.DataFrame

