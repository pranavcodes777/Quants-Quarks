"""
factor_builder.py
-----------------
Loads Balance Sheet, P&L, and Cash Flow from Financial Statements DB.
Derives all meaningful factors. Saves to Hypothesis V1 Database.

Usage:
    python factor_builder.py              # all sectors
    python factor_builder.py FMCG IT     # specific sectors
"""

import os
import sys
import numpy as np
import pandas as pd

SOURCE_DB  = r"E:\Quarks&Quants\Fundamental\Financial Statements\Database"
OUTPUT_DIR = r"E:\Quarks&Quants\Non Fundamental\Hypothesis V1\Database"

MIN_MARCH_YEARS = 5
MIN_YEAR        = 2015

SECTORS: dict[str, list[str]] = {
    "FMCG":          ["HINDUNILVR", "ITC", "BRITANNIA", "DABUR", "GODREJCP", "MARICO",
                       "TATACONSUMER", "COLPAL", "NESTLEIND", "VBL"],
    "IT":            ["TCS", "INFY", "WIPRO", "HCLTECH", "TECHM", "PERSISTENT", "OFSS", "NAUKRI"],
    "Banks":         ["HDFCBANK", "ICICIBANK", "AXISBANK", "SBIN", "KOTAKBANK",
                      "INDUSINDBK", "BANDHANBNK"],
    "NBFC":          ["BAJFINANCE", "BAJAJFINSV", "CHOLAFIN", "MUTHOOTFIN", "SBICARD", "SHRIRAMFIN"],
    "Insurance":     ["HDFCLIFE", "ICICIPRULI", "SBILIFE", "LICI"],
    "Pharma":        ["SUNPHARMA", "CIPLA", "DRREDDY", "DIVISLAB", "LUPIN",
                      "AUROPHARMA", "TORNTPHARM", "ZYDUSLIFE"],
    "Auto":          ["MARUTI", "TATAMOTORS", "M&M", "BAJAJ-AUTO", "HEROMOTOCO",
                      "EICHERMOT", "TVSMOTOR", "MOTHERSON"],
    "Metals":        ["TATASTEEL", "JSWSTEEL", "HINDALCO", "COALINDIA", "JINDALSTEL"],
    "Oil & Gas":     ["RELIANCE", "BPCL", "ONGC", "GAIL"],
    "Power":         ["NTPC", "POWERGRID", "ADANIGREEN", "ADANIENSOL"],
    "Capital Goods": ["LT", "SIEMENS", "BEL", "CGPOWER", "BOSCHLTD", "HAVELLS", "POLYCAB"],
    "Cement":        ["ULTRACEMCO", "AMBUJACEM", "GRASIM"],
    "Consumer":      ["TITAN", "ASIANPAINT", "BERGEPAINT", "PIDILITIND", "PAGEIND"],
    "Telecom":       ["BHARTIARTL", "INDUSTOWER"],
    "Retail":        ["TRENT", "DMART", "IRCTC", "INDHOTEL", "ZOMATO"],
    "Infrastructure":["ADANIPORTS", "ADANIENT"],
    "Chemicals":     ["SRF"],
    "Real Estate":   ["DLF"],
}


def _clean(name: str) -> str:
    return name.replace("\xa0", " ").replace("+", "").strip()


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name] if name in df.columns else pd.Series(0.0, index=df.index)


def _load_statement(ticker: str, filename: str) -> pd.DataFrame | None:
    path = os.path.join(SOURCE_DB, ticker, filename)
    if not os.path.exists(path):
        return None
    try:
        raw = pd.read_parquet(path)
    except Exception:
        return None

    raw.columns      = ["metric"] + list(raw.columns[1:])
    raw["metric"]    = raw["metric"].apply(_clean)
    raw              = raw.set_index("metric").T
    raw.index.name   = "period"

    import re
    mask             = raw.index.str.match(r"^Mar \d{4}$")
    march            = raw[mask].copy()
    march.index      = march.index.str.replace("Mar ", "").astype(int)
    march.index.name = "year"
    march            = march[march.index >= MIN_YEAR]

    if len(march) < MIN_MARCH_YEARS:
        return None
    return march.apply(pd.to_numeric, errors="coerce")


def load_balance_sheet(ticker: str) -> pd.DataFrame | None:
    df = _load_statement(ticker, "balance_sheet.parquet")
    if df is None:
        return None
    df = df.rename(columns={"Borrowing": "Borrowings", "Other Liability": "Other Liabilities"})
    if "Deposits" in df.columns:
        dep          = pd.to_numeric(df["Deposits"], errors="coerce").fillna(0)
        brw          = pd.to_numeric(df.get("Borrowings", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
        df["Borrowings"] = dep + brw
    return df


def load_pl(ticker: str) -> pd.DataFrame | None:
    return _load_statement(ticker, "annual_pl.parquet")


def load_cf(ticker: str) -> pd.DataFrame | None:
    return _load_statement(ticker, "cash_flow.parquet")


# ── Factor derivation ──────────────────────────────────────────────────────────

def derive_bs_factors(bs: pd.DataFrame) -> pd.DataFrame:
    f       = pd.DataFrame(index=bs.index)
    eq      = _col(bs, "Equity Capital") + _col(bs, "Reserves")
    ta      = _col(bs, "Total Assets").replace(0, np.nan)
    fa      = _col(bs, "Fixed Assets")
    cw      = _col(bs, "CWIP")
    br      = _col(bs, "Borrowings").replace(0, np.nan)
    fa_cwip = (fa + cw).replace(0, np.nan)

    f["Debt_to_Equity"]     = br / eq.replace(0, np.nan)
    f["Debt_to_Assets"]     = br / ta
    f["Equity_Ratio"]       = eq / ta
    f["Financial_Leverage"] = ta / eq.replace(0, np.nan)
    f["OtherLiab_Ratio"]    = _col(bs, "Other Liabilities") / ta
    f["FixedAsset_Ratio"]   = fa / ta
    f["CWIP_Intensity"]     = cw / fa_cwip
    f["Investment_Ratio"]   = _col(bs, "Investments") / ta
    f["OtherAsset_Ratio"]   = _col(bs, "Other Assets") / ta
    f["Log_Total_Assets"]   = np.log(ta)
    f["Assets_Growth"]      = _col(bs, "Total Assets").pct_change() * 100
    f["Equity_Growth"]      = eq.pct_change() * 100
    f["Reserves_Growth"]    = _col(bs, "Reserves").pct_change() * 100
    f["Borrowings_Growth"]  = br.pct_change() * 100
    return f


def derive_pl_cf_factors(pl: pd.DataFrame, cf: pd.DataFrame,
                          bs: pd.DataFrame) -> pd.DataFrame:
    f     = pd.DataFrame(index=pl.index)
    sales = _col(pl, "Sales").replace(0, np.nan)
    op    = _col(pl, "Operating Profit")
    np_   = _col(pl, "Net Profit")
    intr  = _col(pl, "Interest").replace(0, np.nan)
    cfo   = _col(cf, "Cash from Operating Activity")
    fcf   = _col(cf, "Free Cash Flow")
    eq    = (_col(bs, "Equity Capital") + _col(bs, "Reserves")).replace(0, np.nan)
    ta    = _col(bs, "Total Assets").replace(0, np.nan)

    # Profitability
    f["Operating_Margin"]  = op  / sales * 100
    f["Net_Margin"]        = np_ / sales * 100
    f["ROE"]               = np_ / eq    * 100
    f["ROA"]               = np_ / ta    * 100
    f["Interest_Coverage"] = op  / intr          # NaN when no debt — intentional

    # Growth
    f["Revenue_Growth"]    = sales.pct_change() * 100
    f["NetProfit_Growth"]  = np_.pct_change()   * 100

    # Cash quality
    f["CFO_to_NetProfit"]  = cfo / np_.replace(0, np.nan)
    f["FCF_Margin"]        = fcf / sales * 100

    return f


# ── Master builder ─────────────────────────────────────────────────────────────

def build_sector_factors(sectors: list[str] | None = None) -> pd.DataFrame:
    target = {k: v for k, v in SECTORS.items() if sectors is None or k in sectors}
    rows   = []

    for sector, tickers in target.items():
        for ticker in tickers:
            bs = load_balance_sheet(ticker)
            if bs is None:
                continue

            pl = load_pl(ticker)
            cf = load_cf(ticker)

            bs_f = derive_bs_factors(bs)

            if pl is not None and cf is not None:
                common = bs.index.intersection(pl.index).intersection(cf.index)
                if len(common) >= MIN_MARCH_YEARS:
                    plcf_f = derive_pl_cf_factors(
                        pl.loc[common], cf.loc[common], bs.loc[common]
                    )
                    row = pd.concat([bs_f.loc[common], plcf_f], axis=1)
                else:
                    row = bs_f
            else:
                row = bs_f

            row.insert(0, "Company", ticker)
            row.insert(1, "Sector",  sector)
            rows.append(row.reset_index())

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sectors = sys.argv[1:] if len(sys.argv) > 1 else None
    label   = ", ".join(sectors) if sectors else "ALL"

    print(f"\nBuilding factors — {label}")
    print("=" * 50)

    df = build_sector_factors(sectors)
    if df.empty:
        print("No data. Check source paths.")
        sys.exit(1)

    out = os.path.join(OUTPUT_DIR, "bs_factors.parquet")
    df.to_parquet(out, index=False)

    factor_cols = [c for c in df.columns if c not in ("year", "Company", "Sector")]
    print(f"  Rows      : {len(df)}")
    print(f"  Companies : {df['Company'].nunique()}")
    print(f"  Sectors   : {df['Sector'].nunique()}")
    print(f"  Factors   : {len(factor_cols)}")
    print(f"  Saved  -> {out}")
