import numpy as np
import pandas as pd

# === SHARPE RATIO INTERPRETATION ===
# Sharpe answers: “How much excess return do I get per unit of TOTAL risk?”
#
# Uses total volatility (standard deviation) as the risk measure.
# This measures: is the return WORTH the volatility I’m taking?
#
# Rough interpretation:
# < 0.5     poor
# 0.5–1.0   weak
# 1.0–2.0   acceptable/good
# > 2.0     exceptional
#
# This is a *historical* risk-adjusted performance measure.
#
# Sharpe is best when comparing assets/portfolios with different TOTAL volatilities.


def sharpe_ratio(rp, rf, sd):
    return (rp - rf) / sd

def sharpe(returns, weights=None, rf_annual=0.05, periods_per_year=252):

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

    rp = port.mean() * periods_per_year
    sd = port.std(ddof=1) * np.sqrt(periods_per_year)

    if np.isclose(sd, 0.0):
        raise ZeroDivisionError("Standard deviation is zero; Sharpe undefined.")

    s = sharpe_ratio(rp, rf_annual, sd)
    print(f"Sharpe: {s:.2f}")
    return s
