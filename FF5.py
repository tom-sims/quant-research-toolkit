import pandas as pd
import numpy as np
import statsmodels.api as sm

# === FF5 INPUTS ===
# returns : Series (single asset, name=ticker) or DataFrame (multiple assets, columns=tickers)
#           values must be simple daily returns, e.g. prices.pct_change().dropna()
#
# weights : only required if returns is a DataFrame (portfolio)
#           list of weights same length as columns, will be normalized automatically
#
# How to call:
#
# # single asset
# aapl = prices["AAPL"].pct_change().dropna().rename("AAPL")
# model = FF5(aapl)
#
# # portfolio
# rets = prices[["AAPL","MSFT"]].pct_change().dropna()
# model = FF5(rets, weights=[0.5,0.5])
#
# then:
# print(model.params)
# print(model.rsquared_adj)
#
# How to interpret model outputs:
#
# model.rsquared          -> R-squared
# model.rsquared_adj      -> Adjusted R-squared
#
# Alpha:
# model.params["const"]   -> alpha estimate
# model.pvalues["const"]  -> alpha p-value
#
# Factor exposures (betas):
# model.params["Mkt-RF"]  -> market beta
# model.params["SMB"]     -> size factor (small minus big)
# model.params["HML"]     -> value factor (high minus low)
# model.params["RMW"]     -> profitability factor (robust minus weak)
# model.params["CMA"]     -> investment factor (conservative minus aggressive)
#
# all parameters   : model.params.to_dict()
# all pvalues      : model.pvalues.to_dict()

# How to interpret numbers:
#
# R-squared / Adj-R2  (how much factors explain)
# <0.3 weak | 0.3–0.6 ok | 0.6–0.8 strong | >0.8 very strong
#
# Alpha ("const")
# daily 0.00004 ≈ ~1%/yr   (annualise daily alpha by ×252)
# significance: use p-value or t-stat
# p<0.05 (|t|≈2) = statistically real
# p≥0.10 = basically indistinguishable from zero
#
# Mkt-RF (market beta)
# ~1 ≈ market-like | <0.5 defensive | >1.5 aggressive
#
# SMB/HML/RMW/CMA (style tilts)
# SMB: + = small-cap tilt         / − = large-cap tilt
# HML: + = value tilt             / − = growth tilt
# RMW: + = profitable firms tilt  / − = weak profitability tilt
# CMA: + = conservative invest.   / − = aggressive invest.
# |beta| ≈ 0.1 low | 0.3 medium | 0.5+ strong
#
# Sharpe Ratio (historical, not forecast)
# <0.5 poor | 0.5–1.0 meh | 1.0–1.5 good | >1.5 very good
#
# Frequency notes:
# alpha annualise: daily ×252, monthly ×12
# sharpe annualise: daily ×√252, monthly ×√12


def get_fama_french():
    url = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_daily_CSV.zip"
    ff = pd.read_csv(url, compression="zip", skiprows=3)
    ff = ff[~ff.iloc[:, 0].astype(str).str.contains("Copyright", na=False)]
    ff["Date"] = pd.to_datetime(ff.iloc[:, 0], format="%Y%m%d", errors="coerce")
    ff = ff.dropna(subset=["Date"]).set_index("Date")
    factors = ff.iloc[:, 1:6].apply(pd.to_numeric, errors="coerce") / 100.0
    rf = ff.iloc[:, -1].astype(float) / 100.0
    rf.name = "RF"
    return factors, rf

def get_portfolio_data_from_returns(returns_df, weights):
    w = np.array(weights, dtype=float)
    w = w / w.sum()
    port = returns_df.values @ w
    return pd.Series(port.ravel(), index=returns_df.index, name="Portfolio")

def ticker_model(ret, ticker, rf, factors):
    aligned = pd.concat([ret, rf, factors], axis=1, join="inner").dropna()
    excess = aligned[ticker] - aligned["RF"]
    X = sm.add_constant(aligned[["Mkt-RF","SMB","HML","RMW","CMA"]])
    model = sm.OLS(excess, X).fit()
    return model

def portfolio_model(ret, rf, factors):
    aligned = pd.concat([ret, rf, factors], axis=1, join="inner").dropna()
    excess = aligned["Portfolio"] - aligned["RF"]
    X = sm.add_constant(aligned[["Mkt-RF","SMB","HML","RMW","CMA"]])
    model = sm.OLS(excess, X).fit()
    return model

def FF5(returns, weights=None):
    factors, rf = get_fama_french()

    if isinstance(returns, pd.Series):
        ticker = returns.name
        return ticker_model(returns, ticker, rf, factors)

    if isinstance(returns, pd.DataFrame):
        port = get_portfolio_data_from_returns(returns, weights)
        return portfolio_model(port, rf, factors)

    return None
