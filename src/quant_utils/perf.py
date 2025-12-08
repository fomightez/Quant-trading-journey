# src/quant_utils/perf.py

import pandas as pd
import numpy as np

def cagr(
    returns: pd.Series | pd.DataFrame, 
    days_per_year: int = 252
) -> float | pd.Series:

    if isinstance(returns, pd.DataFrame):
        if returns.isna().all().any():
            raise ValueError("At least one column is completely empty")
    else:
        if returns.isna().all():
            raise ValueError("The series is empty")

    cum_return = (1 + returns).cumprod().iloc[-1]
    years = len(returns) / days_per_year
    return cum_return ** (1 / years) - 1

def sharpe(
    returns: pd.Series | pd.DataFrame, 
    risk_free: pd.Series |  pd.DataFrame | float,
    days_per_year: int = 252
) -> float | pd.Series:

    if isinstance(risk_free, (int, float)):
        rf_daily = risk_free / days_per_year
    else:
        rf_daily = risk_free.reindex(returns.index).ffill().bfill()
        if rf_daily.max() > 1:
            rf_daily = rf_daily / 100
        rf_daily = rf_daily / days_per_year
        
    if isinstance(rf_daily, pd.DataFrame):
        rf_daily = rf_daily.iloc[:, 0]

    excess = returns.sub(rf_daily, axis=0)

    annual_excess = excess.mean() * days_per_year
    annual_vol = returns.std() * np.sqrt(days_per_year)

    sharpe_ratio = annual_excess / annual_vol
    sharpe_ratio = sharpe_ratio.replace([np.inf, -np.inf], np.nan)

    return sharpe_ratio.round(3)

