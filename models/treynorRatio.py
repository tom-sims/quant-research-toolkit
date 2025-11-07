import numpy as np
import pandas as pd
from FF5 import FF5

# === TREYNOR RATIO INTERPRETATION ===
# Treynor answers: “How much excess return do I get per unit of SYSTEMATIC market risk?”
#
# Uses BETA (market exposure) as the risk measure.
# Beta comes from the FF5 model as the Mkt-RF coefficient.
#
# Treynor = (annual excess return) / beta
#
# If Treynor is high → portfolio earns a lot of return but with LOW market exposure.
# If Treynor is low → portfolio earns little return for the amount of market exposure.
#
# Treynor is best when comparing portfolios with similar beta targets,
# or when you want to see if a stock/portfolio is delivering return BEYOND just “market lift”.


def treynor(returns, weights=None, rf_annual=0.05, periods_per_year=252):

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

    rp_annual = port.mean() * periods_per_year

    model = FF5(port.rename("Portfolio"))
    if model is None:
        raise ValueError("FF5 returned None – likely data alignment issue.")

    if "Mkt-RF" not in model.params:
        raise KeyError("FF5 model did not return a 'Mkt-RF' coefficient.")

    beta = float(model.params["Mkt-RF"])
    if np.isclose(beta, 0.0):
        raise ZeroDivisionError("Beta is ~0; Treynor undefined.")
    return (rp_annual - rf_annual) / beta
