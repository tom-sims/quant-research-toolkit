import yfinance as yf
import pandas as pd

# -------------------------------------------------
# SETTINGS
# -------------------------------------------------
TICKERS = ["ILMN", "TMO", "DHR", "A", "QGEN", "PACB"]
OUTPUT_FILE = "peer_financials.xlsx"

# -------------------------------------------------
# DOWNLOAD STATEMENTS FOR EACH TICKER
# -------------------------------------------------
balance_sheets = {}
income_stmts = {}
cash_flows = {}

for t in TICKERS:
    print(f"Downloading data for {t}...")
    ticker = yf.Ticker(t)

    # Raw statements from yfinance
    bs = ticker.balance_sheet      # rows = line items, cols = dates
    is_ = ticker.income_stmt
    cf = ticker.cash_flow

    # Ensure they are DataFrames (yfinance sometimes gives Series)
    balance_sheets[t] = pd.DataFrame(bs)
    income_stmts[t] = pd.DataFrame(is_)
    cash_flows[t] = pd.DataFrame(cf)

# -------------------------------------------------
# CONCATENATE INTO 3 BIG TABLES
# -------------------------------------------------
# Columns will be a MultiIndex: (Ticker, Date)
# Rows are the line items (e.g. Total Revenue, Net Income, etc.)

balance_sheet_all = pd.concat(balance_sheets, axis=1)   # keys = tickers
income_stmt_all = pd.concat(income_stmts, axis=1)
cash_flow_all = pd.concat(cash_flows, axis=1)

# Optional: sort the outer column level by ticker
balance_sheet_all = balance_sheet_all.reindex(columns=sorted(balance_sheet_all.columns.levels[0]), level=0)
income_stmt_all = income_stmt_all.reindex(columns=sorted(income_stmt_all.columns.levels[0]), level=0)
cash_flow_all = cash_flow_all.reindex(columns=sorted(cash_flow_all.columns.levels[0]), level=0)

# -------------------------------------------------
# WRITE TO A SINGLE EXCEL FILE
# -------------------------------------------------
with pd.ExcelWriter(OUTPUT_FILE, engine="xlsxwriter") as writer:
    balance_sheet_all.to_excel(writer, sheet_name="Balance_Sheets")
    income_stmt_all.to_excel(writer, sheet_name="Income_Statements")
    cash_flow_all.to_excel(writer, sheet_name="Cash_Flows")

print(f"Done! Saved all statements to '{OUTPUT_FILE}'")