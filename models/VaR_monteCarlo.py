import numpy as np
import pandas as pd

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

def get_possible_returns(mean, std, rng):
    if rng is None:
        rng = np.random.default_rng()
    return rng.normal(mean, std)


def VaR_monteCarlo(returns, days, weights=None, conf_int=0.95, sims = 10000, random_state = None):
    if not isinstance(days, int) or days<=0:
        raise ValueError("Days must be a positive integer")
    if not isinstance(sims, int) or sims<=0:
        raise ValueError("Sims must be a positive integer")
    if conf_int>=1 or conf_int<=0:
        raise ValueError("Confidence interval must be between 0 and 1")

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
    
    rp = float(port.mean())
    std = float(port.std(ddof=1))

    values = []

    rng = np.random.default_rng(random_state)

    for _ in range(sims):
        value = 1
        for _ in range(days):
            r = get_possible_returns(rp, std, rng)
            value *= (1+r)
        values.append(value)

    values = np.array(values)
    losses = 1 - values
    VaR = np.quantile(losses, conf_int)
    return VaR

