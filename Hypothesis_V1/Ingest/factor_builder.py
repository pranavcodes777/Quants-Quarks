"""
factor_builder.py
-----------------
Loads detailed annual BS, P&L, CF from the new Database.
Falls back to old Fundamental DB for missing companies (TATAMOTORS, TATACONSUMER).
Derives all factors. Saves master parquet to Hypothesis V1 Database.

Usage:
    python factor_builder.py              # all sectors
    python factor_builder.py FMCG IT     # specific sectors
"""

import os
import sys
import re
import numpy as np
import pandas as pd
from collections import Counter

# ── Paths ──────────────────────────────────────────────────────────────────────
NEW_DB     = r"E:\Quarks&Quants\Non Fundamental\Hypothesis V1\Database"   # primary
OLD_DB     = r"E:\Quarks&Quants\Fundamental\Financial Statements\Database" # fallback
OUTPUT_DIR = NEW_DB

MIN_YEARS = 5
MIN_YEAR  = 2015

# ── Sectors ────────────────────────────────────────────────────────────────────
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


# ── Helpers ────────────────────────────────────────────────────────────────────
def _clean(name: str) -> str:
    return name.replace("\xa0", " ").replace("+", "").strip()


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    return df[name] if name in df.columns else pd.Series(0.0, index=df.index)


# ── Loaders ────────────────────────────────────────────────────────────────────
def _load_new(ticker: str, filename: str) -> pd.DataFrame | None:
    """
    Load from new detailed DB.
    Format: index=metric, columns='Mmm YYYY' period strings.
    Returns: DataFrame with index=year(int), columns=metric.
    """
    path = os.path.join(NEW_DB, ticker, filename)
    if not os.path.exists(path):
        return None
    try:
        raw = pd.read_parquet(path)
    except Exception:
        return None

    # Keep only valid period columns (any "Mmm YYYY")
    cols = [c for c in raw.columns if re.match(r"^[A-Z][a-z]{2} \d{4}$", str(c).strip())]
    if not cols:
        return None
    raw = raw[cols].copy()

    # Transpose: rows = period, cols = metric
    df = raw.T
    df.index.name = "period"

    # Detect fiscal year-end month (most frequent month in column labels)
    months     = [str(idx)[:3] for idx in df.index]
    yr_end_mon = Counter(months).most_common(1)[0][0]

    # Keep only year-end rows
    df = df[[str(idx).startswith(yr_end_mon) for idx in df.index]].copy()

    # Year = last 4 digits of period label
    df.index = [int(str(idx)[-4:]) for idx in df.index]
    df.index.name = "year"
    df = df[df.index >= MIN_YEAR]

    if len(df) < MIN_YEARS:
        return None

    df.columns = [_clean(str(c)) for c in df.columns]
    return df.apply(pd.to_numeric, errors="coerce")


def _load_old(ticker: str, filename: str) -> pd.DataFrame | None:
    """
    Load from old Fundamental DB (fallback).
    Format: first column = metric label, rest = 'Mar YYYY' period columns.
    Returns: DataFrame with index=year(int), columns=metric.
    """
    path = os.path.join(OLD_DB, ticker, filename)
    if not os.path.exists(path):
        return None
    try:
        raw = pd.read_parquet(path)
    except Exception:
        return None

    raw.columns   = ["metric"] + list(raw.columns[1:])
    raw["metric"] = raw["metric"].apply(_clean)
    raw           = raw.set_index("metric").T
    raw.index.name = "period"

    march = raw[raw.index.str.match(r"^Mar \d{4}$")].copy()
    march.index = march.index.str.replace("Mar ", "").astype(int)
    march.index.name = "year"
    march = march[march.index >= MIN_YEAR]

    if len(march) < MIN_YEARS:
        return None
    return march.apply(pd.to_numeric, errors="coerce")


def load_balance_sheet(ticker: str) -> pd.DataFrame | None:
    df = _load_new(ticker, "annual_bs.parquet")
    if df is None:
        df = _load_old(ticker, "balance_sheet.parquet")
    if df is None:
        return None

    # Bank normalisation: Deposits + Borrowings = total interest-bearing liabilities
    df = df.rename(columns={"Borrowing": "Borrowings", "Other Liability": "Other Liabilities"})
    if "Deposits" in df.columns:
        dep = pd.to_numeric(df["Deposits"], errors="coerce").fillna(0)
        brw = pd.to_numeric(df.get("Borrowings", pd.Series(0, index=df.index)), errors="coerce").fillna(0)
        df["Borrowings"] = dep + brw
    return df


def load_pl(ticker: str) -> pd.DataFrame | None:
    df = _load_new(ticker, "annual_pl.parquet")
    if df is None:
        df = _load_old(ticker, "annual_pl.parquet")
    return df


def load_cf(ticker: str) -> pd.DataFrame | None:
    df = _load_new(ticker, "annual_cf.parquet")
    if df is None:
        df = _load_old(ticker, "cash_flow.parquet")
    return df


def load_ratios(ticker: str) -> pd.DataFrame | None:
    """Ratios come from old DB only — no new equivalent."""
    path = os.path.join(OLD_DB, ticker, "ratios.parquet")
    if not os.path.exists(path):
        return None
    try:
        raw = pd.read_parquet(path)
    except Exception:
        return None

    raw.columns   = ["metric"] + list(raw.columns[1:])
    raw["metric"] = raw["metric"].apply(_clean)
    raw           = raw.set_index("metric").T
    raw.index.name = "period"

    march = raw[raw.index.str.match(r"^Mar \d{4}$")].copy()
    march.index = march.index.str.replace("Mar ", "").astype(int)
    march.index.name = "year"
    march = march[march.index >= MIN_YEAR]

    if len(march) < MIN_YEARS:
        return None

    def _to_num(s):
        cleaned = s.astype(str).str.replace("%", "").str.replace(",", "").str.strip()
        return pd.to_numeric(cleaned, errors="coerce")

    return march.apply(_to_num)


# ── Factor derivation ──────────────────────────────────────────────────────────
def derive_bs_factors(bs: pd.DataFrame) -> pd.DataFrame:
    f  = pd.DataFrame(index=bs.index)
    eq = _col(bs, "Equity Capital") + _col(bs, "Reserves")
    ta = _col(bs, "Total Assets").replace(0, np.nan)
    fa = _col(bs, "Fixed Assets")
    cw = _col(bs, "CWIP")
    br = _col(bs, "Borrowings")

    f["Debt_to_Equity"]     = br / eq.replace(0, np.nan)
    f["Debt_to_Assets"]     = br / ta
    f["Equity_Ratio"]       = eq / ta
    f["Financial_Leverage"] = ta / eq.replace(0, np.nan)
    f["OtherLiab_Ratio"]    = _col(bs, "Other Liabilities") / ta
    f["FixedAsset_Ratio"]   = fa / ta
    f["Tangible_Ratio"]     = (fa + cw) / ta
    f["CWIP_Intensity"]     = cw / (fa + cw).replace(0, np.nan)
    f["Investment_Ratio"]   = _col(bs, "Investments") / ta
    f["OtherAsset_Ratio"]   = _col(bs, "Other Assets") / ta
    f["Log_Total_Assets"]   = np.log(ta)
    f["Assets_Growth"]      = _col(bs, "Total Assets").pct_change() * 100
    f["Equity_Growth"]      = eq.pct_change() * 100
    f["FixedAssets_Growth"] = fa.pct_change() * 100
    f["Reserves_Growth"]    = _col(bs, "Reserves").pct_change() * 100
    f["Borrowings_Growth"]  = br.replace(0, np.nan).pct_change() * 100
    return f


def derive_pl_cf_factors(pl: pd.DataFrame, cf: pd.DataFrame,
                          bs: pd.DataFrame) -> pd.DataFrame:
    f      = pd.DataFrame(index=pl.index)
    sales  = _col(pl, "Sales").replace(0, np.nan)
    op     = _col(pl, "Operating Profit")
    depr   = _col(pl, "Depreciation")
    np_    = _col(pl, "Net Profit")
    intr   = _col(pl, "Interest").replace(0, np.nan)
    cfo    = _col(cf, "Cash from Operating Activity")
    fcf    = _col(cf, "Free Cash Flow")
    eq     = (_col(bs, "Equity Capital") + _col(bs, "Reserves")).replace(0, np.nan)
    ta     = _col(bs, "Total Assets").replace(0, np.nan)
    fa     = _col(bs, "Fixed Assets").replace(0, np.nan)
    ebitda = op + depr

    f["Operating_Margin"]     = op     / sales * 100
    f["EBITDA_Margin"]        = ebitda / sales * 100
    f["Net_Margin"]           = np_    / sales * 100
    f["ROE"]                  = np_    / eq    * 100
    f["ROA"]                  = np_    / ta    * 100
    f["Interest_Coverage"]    = op     / intr
    f["Asset_Turnover"]       = sales  / ta
    f["FixedAsset_Turnover"]  = sales  / fa
    f["Revenue_Growth"]       = sales.pct_change()  * 100
    f["OpProfit_Growth"]      = op.pct_change()     * 100
    f["EBITDA_Growth"]        = ebitda.pct_change() * 100
    f["NetProfit_Growth"]     = np_.pct_change()    * 100
    f["CFO_to_NetProfit"]     = cfo / np_.replace(0, np.nan)
    f["FCF_Margin"]           = fcf / sales * 100

    capex = (cfo - fcf).abs()
    f["Capex_to_Sales"]        = capex / sales * 100
    f["Capex_to_Depreciation"] = capex / depr.replace(0, np.nan)
    return f


def derive_detailed_bs_factors(bs: pd.DataFrame) -> pd.DataFrame:
    """
    Derives liquidity, debt composition, and asset mix factors.
    New annual_bs.parquet contains all sub-items — same df as load_balance_sheet output.
    """
    f = pd.DataFrame(index=bs.index)

    inv  = _col(bs, "Inventories")
    rec  = _col(bs, "Trade receivables")
    cash = _col(bs, "Cash Equivalents")
    loan = _col(bs, "Loans n Advances")
    oai  = _col(bs, "Other asset items")
    curr_assets = inv + rec + cash + loan + oai

    tp   = _col(bs, "Trade Payables")
    stb  = _col(bs, "Short term Borrowings")
    oli  = _col(bs, "Other liability items")
    curr_liab = tp + stb + oli
    curr_liab_safe = curr_liab.replace(0, np.nan)

    f["Current_Ratio"]  = curr_assets / curr_liab_safe
    f["Quick_Ratio"]    = (curr_assets - inv) / curr_liab_safe
    f["Cash_Ratio"]     = cash / curr_liab_safe

    ltb = _col(bs, "Long term Borrowings")
    f["LT_Debt_Ratio"]  = ltb / (ltb + stb).replace(0, np.nan)

    ta         = _col(bs, "Total Assets").replace(0, np.nan)
    total_debt = _col(bs, "Borrowings")
    net_debt   = (total_debt - cash).clip(lower=0)
    eq         = (_col(bs, "Equity Capital") + _col(bs, "Reserves")).replace(0, np.nan)
    f["Net_Debt_to_Equity"] = net_debt / eq

    gross   = _col(bs, "Gross Block").replace(0, np.nan)
    acc_dep = _col(bs, "Accumulated Depreciation")
    f["Asset_Age_Ratio"]         = acc_dep / gross
    f["Inventory_to_Assets"]     = inv  / ta
    f["Receivables_to_Assets"]   = rec  / ta
    f["Cash_to_Assets"]          = cash / ta
    f["TradePayables_to_Assets"] = tp   / ta
    return f


def derive_ratio_factors(ratios: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=ratios.index)

    def _pct_col(name: str) -> pd.Series:
        s = _col(ratios, name)
        if s.dtype == object:
            s = pd.to_numeric(s.astype(str).str.replace("%", "").str.strip(), errors="coerce")
        return s

    f["Debtor_Days"]           = pd.to_numeric(_col(ratios, "Debtor Days"),             errors="coerce")
    f["Inventory_Days"]        = pd.to_numeric(_col(ratios, "Inventory Days"),          errors="coerce")
    f["Days_Payable"]          = pd.to_numeric(_col(ratios, "Days Payable"),            errors="coerce")
    f["Cash_Conversion_Cycle"] = pd.to_numeric(_col(ratios, "Cash Conversion Cycle"),  errors="coerce")
    f["Working_Capital_Days"]  = pd.to_numeric(_col(ratios, "Working Capital Days"),    errors="coerce")
    f["ROCE"]                  = _pct_col("ROCE %")
    return f


# ── Master builder ─────────────────────────────────────────────────────────────
def build_sector_factors(sectors: list[str] | None = None) -> pd.DataFrame:
    target = {k: v for k, v in SECTORS.items() if sectors is None or k in sectors}
    rows   = []

    for sector, tickers in target.items():
        for ticker in tickers:
            bs = load_balance_sheet(ticker)
            if bs is None:
                print(f"  SKIP {ticker} — no balance sheet")
                continue

            pl     = load_pl(ticker)
            cf     = load_cf(ticker)
            ratios = load_ratios(ticker)

            # Base: BS factors
            bs_f   = derive_bs_factors(bs)
            common = bs.index

            # Add P&L + CF factors
            if pl is not None and cf is not None:
                c3 = bs.index.intersection(pl.index).intersection(cf.index)
                if len(c3) >= MIN_YEARS:
                    plcf_f = derive_pl_cf_factors(pl.loc[c3], cf.loc[c3], bs.loc[c3])
                    row    = pd.concat([bs_f.loc[c3], plcf_f], axis=1)
                    common = c3
                else:
                    row = bs_f
            else:
                row = bs_f

            # Add ratio factors (Debtor Days, ROCE, etc.)
            if ratios is not None:
                cr = common.intersection(ratios.index)
                if len(cr) >= MIN_YEARS:
                    rat_f  = derive_ratio_factors(ratios.loc[cr])
                    row    = row.loc[cr].join(rat_f)
                    common = cr

            # Add detailed BS factors (liquidity, asset composition)
            cd = common.intersection(bs.index)
            if len(cd) >= MIN_YEARS:
                dbs_f = derive_detailed_bs_factors(bs.loc[cd])
                row   = row.loc[cd].join(dbs_f)

            row.insert(0, "Company", ticker)
            row.insert(1, "Sector",  sector)
            rows.append(row.reset_index())

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def merge_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    ret_path = os.path.join(OUTPUT_DIR, "returns.parquet")
    if not os.path.exists(ret_path):
        print("  Warning: returns.parquet not found — skipping forward returns")
        return df

    returns            = pd.read_parquet(ret_path).copy()
    returns["year"]    = returns["year"] - 1
    returns            = returns.rename(columns={"Return_1Y": "Return_1Y_Fwd"})
    merged             = df.merge(returns, on=["Company", "year"], how="left")
    n_fwd              = merged["Return_1Y_Fwd"].notna().sum()
    print(f"  Forward returns matched: {n_fwd} of {len(merged)} rows")
    return merged


# ── Entry point ────────────────────────────────────────────────────────────────
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

    df  = merge_forward_returns(df)
    out = os.path.join(OUTPUT_DIR, "bs_factors.parquet")
    df.to_parquet(out, index=False)

    factor_cols = [c for c in df.columns if c not in ("year", "Company", "Sector", "Return_1Y_Fwd")]
    print(f"\n  Rows      : {len(df)}")
    print(f"  Companies : {df['Company'].nunique()}")
    print(f"  Sectors   : {df['Sector'].nunique()}")
    print(f"  Factors   : {len(factor_cols)}")
    print(f"  Factors   : {factor_cols}")
    print(f"\n  Saved  -> {out}")
