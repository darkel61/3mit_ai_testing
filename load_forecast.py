import datetime

import numpy as np
import pandas as pd
import plotly
import os

from greykite.framework.templates.forecaster import Forecaster


forecast_codes = [386,1973,2056]

for code in forecast_codes:
    forecaster = Forecaster()
    forecaster.load_forecast_result(
        source_dir=f'./forecast-{code}',
        load_design_info=True)
    
    result = forecaster.forecast_result
    forecast = result.forecast
    fig = forecast.plot()
    fig.write_html(f"load_html/forecast-{code}.html")
    print(f"load_html/forecast-{code}.html")
    print(pd.DataFrame(result.backtest.test_evaluation, index=["Value"]).transpose())