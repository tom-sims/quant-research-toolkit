import numpy as np
import pandas as pd
from scipy.stats import norm

# === MONTE CARLO VaR ===
# returns : Series OR DataFrame (daily returns)
# weights : optional if DataFrame (will be normalised)
#
# How to call:
# r = prices.pct_change().dropna()
# VaR_monteCarlo(r, conf_int=0.95)
#
# What it does:
# fits μ and σ from history then simulates thousands of random future returns
#
# Why useful:
# flexible – can later replace normal sampling with skewed, t-dist, factor model etc
#
# How to interpret:
# same interpretation as historical:
# 95% VaR = one day loss you only expect to exceed 5% of the time


def VaR_monteCarlo(returns, weights=None, conf_int=0.95, periods_of_year = 1):

    if isinstance(returns, pd.Series):
            port = returns.dropna().copy()

    elif isinstance(returns, pd.DataFrame):
        df = returns.dropna(how="all")
        if df.empty:
            raise ValueError("DataFrame returns contains no data.")

        n = df.shape[1]
        if weights is None:
            w = np.ones(n) / n
        else:
            w = np.asarray(weights, dtype=float).ravel()
            if w.size != n:
                raise ValueError(f"weights length {w.size} != number of columns {n}")
            w = w / w.sum()

        port = (df * w).sum(axis=1).dropna()

    else:
        raise TypeError("returns must be a pandas Series or DataFrame")

    if port.empty:
        raise ValueError("Portfolio return series is empty after cleaning.")
    
    if conf_int>=1 or conf_int<=0:
        raise ValueError("Confidence interval must be between 0 and 1")
    
    rp = port.mean() * periods_of_year
    sd = port.std(ddof=1) * np.sqrt(periods_of_year)

    q = 1-conf_int
    
    simulated_returns = np.random.normal(rp,sd,10000)

    VaR = -1*np.quantile(simulated_returns, q)

    return VaR

