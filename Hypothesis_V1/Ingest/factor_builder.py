"""
factor_builder.py
-----------------
Reads annual balance sheet parquet files from the Financial Statements DB,
derives all meaningful BS factors per company-year, saves to Hypothesis V1 Database.

Usage:
    python factor_builder.py              # build all sectors
    python factor_builder.py FMCG IT     # build specific sectors only
"""

import os
import sys
import numpy as np
import pandas as pd

SOURCE_DB  = r"E:\Quarks&Quants\Fundamental\Financial Statements\Database"
OUTPUT_DIR = r"E:\Quarks&Quants\Non Fundamental\Hypothesis V1\Database"

MIN_MARCH_YEARS = 5
MIN_YEAR        = 2015   # ignore data older than this (avoids COLPAL 2006-2010 misalignment)

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


def load_balance_sheet(ticker: str) -> pd.DataFrame | None:
    path = os.path.join(SOURCE_DB, ticker, "balance_sheet.parquet")
    if not os.path.exists(path):
        return None
    try:
        raw = pd.read_parquet(path)
    except Exception:
        return None

    # First column holds metric names regardless of what it's called
    raw.columns = ["metric"] + list(raw.columns[1:])
    raw["metric"] = raw["metric"].apply(_clean)
    raw = raw.set_index("metric").T
    raw.index.name = "period"

    # Normalise column name variants across sectors
    raw = raw.rename(columns={
        "Borrowing":       "Borrowings",   # banks use singular
        "Other Liability": "Other Liabilities",
    })
    # Banks: merge Deposits into Borrowings so leverage factors are comparable
    if "Deposits" in raw.columns:
        dep = pd.to_numeric(raw["Deposits"], errors="coerce").fillna(0)
        brw = pd.to_numeric(raw.get("Borrowings", pd.Series(0, index=raw.index)), errors="coerce").fillna(0)
        raw["Borrowings"] = dep + brw

    march = raw[raw.index.str.startswith("Mar")].copy()
    march.index = march.index.str.replace("Mar ", "").astype(int)
    march.index.name = "year"
    march = march[march.index >= MIN_YEAR]

    if len(march) < MIN_MARCH_YEARS:
        return None

    return march.apply(pd.to_numeric, errors="coerce")


def _col(bs: pd.DataFrame, name: str) -> pd.Series:
    """Return column or zeros if missing."""
    return bs[name] if name in bs.columns else pd.Series(0.0, index=bs.index)


def derive_factors(bs: pd.DataFrame) -> pd.DataFrame:
    f  = pd.DataFrame(index=bs.index)
    eq = _col(bs, "Equity Capital") + _col(bs, "Reserves")
    ta = _col(bs, "Total Assets").replace(0, np.nan)
    fa = _col(bs, "Fixed Assets")
    cw = _col(bs, "CWIP")
    br = _col(bs, "Borrowings").replace(0, np.nan)
    fa_cwip = (fa + cw).replace(0, np.nan)

    # Leverage & capital structure
    f["Debt_to_Equity"]     = br / eq.replace(0, np.nan)
    f["Debt_to_Assets"]     = br / ta
    f["Equity_Ratio"]       = eq / ta
    f["Financial_Leverage"] = ta / eq.replace(0, np.nan)
    f["OtherLiab_Ratio"]    = _col(bs, "Other Liabilities") / ta

    # Asset composition
    f["FixedAsset_Ratio"]   = fa / ta
    f["CWIP_Intensity"]     = cw / fa_cwip
    f["Investment_Ratio"]   = _col(bs, "Investments") / ta
    f["OtherAsset_Ratio"]   = _col(bs, "Other Assets") / ta
    f["Tangible_Ratio"]     = fa_cwip / ta

    # Size
    f["Log_Total_Assets"]   = np.log(ta)

    # YoY growth (%)
    f["Assets_Growth"]      = _col(bs, "Total Assets").pct_change() * 100
    f["Equity_Growth"]      = eq.pct_change() * 100
    f["FixedAssets_Growth"] = fa.pct_change() * 100
    f["Reserves_Growth"]    = _col(bs, "Reserves").pct_change() * 100
    f["Borrowings_Growth"]  = br.pct_change() * 100

    return f


def build_sector_factors(sectors: list[str] | None = None) -> pd.DataFrame:
    target = {k: v for k, v in SECTORS.items() if sectors is None or k in sectors}
    rows   = []

    for sector, tickers in target.items():
        for ticker in tickers:
            bs = load_balance_sheet(ticker)
            if bs is None:
                continue
            factors = derive_factors(bs)
            factors.insert(0, "Company", ticker)
            factors.insert(1, "Sector",  sector)
            rows.append(factors.reset_index())

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sectors = sys.argv[1:] if len(sys.argv) > 1 else None
    label   = ", ".join(sectors) if sectors else "ALL"

    print(f"\nBuilding BS factors — {label}")
    print("=" * 50)

    df = build_sector_factors(sectors)

    if df.empty:
        print("No data built. Check source paths and sector names.")
        sys.exit(1)

    out = os.path.join(OUTPUT_DIR, "bs_factors.parquet")
    df.to_parquet(out, index=False)

    print(f"\n  Rows        : {len(df)}")
    print(f"  Companies   : {df['Company'].nunique()}")
    print(f"  Sectors     : {df['Sector'].nunique()}")
    print(f"  Year range  : {df['year'].min()} – {df['year'].max()}")
    print(f"\n  Saved -> {out}")
