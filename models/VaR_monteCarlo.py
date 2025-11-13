import numpy as np
import pandas as pd

# === MONTE CARLO VaR ===
# inputs:
#   returns : pd.Series or pd.DataFrame of DAILY returns (not prices)
#             - if DataFrame, columns are assets; 'weights' are optional and auto-normalised
#   days    : int, horizon in trading days to simulate (e.g., 10, 20)
#   sims    : int, number of simulated paths (e.g., 10_000)
#   conf_int: float in (0,1), VaR confidence (e.g., 0.95)
#   random_state : optional int for reproducibility
#
# model:
#   • We model the portfolio value S_t as Geometric Brownian Motion (GBM):
#       dS_t / S_t = μ dt + σ dW_t
#   • Using DAILY estimates μ, σ from history and a 1-day step (Δt = 1),
#     each step draws Z ~ N(0,1) and updates with the exact GBM discretisation:
#       S_{t+1} = S_t * exp( (μ − 0.5 σ^2) + σ Z )
#   • This is equivalent to simulating NORMAL log-returns; prices remain strictly positive.
#
# what it does:
#   • Estimates DAILY μ and σ from the historical portfolio return series.
#   • Simulates 'sims' independent GBM price PATHS over 'days' via the exponential update above.
#   • Collects the terminal gross values S_T (starting from S_0 = 1.0).
#   • Converts to losses L = 1 − S_T and computes VaR at 'conf_int' as the loss quantile.
#
# how to call:
#   prices  = ...                       # price series/DataFrame
#   returns = prices.pct_change().dropna()
#   VaR     = VaR_monteCarlo(returns, days=20, conf_int=0.95, sims=5000, random_state=42)
#
# how to interpret:
#   • VaR(95%, N days) = the N-day loss fraction exceeded only 5% of the time under the GBM model.
#     Example: VaR = 0.04 ⇒ ≈4% potential loss over N days at 95% confidence.



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
    
    log_ret = np.log1p(port)
    mu = float(log_ret.mean())
    std = float(log_ret.std(ddof=1))
    drift = mu - 0.5*(std**2)

    values = []

    rng = np.random.default_rng(random_state)

    for _ in range(sims):
        value = 1.0
        for _ in range(days):
            z = rng.normal(0, 1)
            value *= np.exp(drift + std*z)
        values.append(value)

    values = np.array(values)
    losses = 1 - values
    VaR = np.quantile(losses, conf_int)
    return VaR


