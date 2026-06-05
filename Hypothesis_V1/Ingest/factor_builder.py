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


def load_ratios(ticker: str) -> pd.DataFrame | None:
    path = os.path.join(SOURCE_DB, ticker, "ratios.parquet")
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
    march            = raw[raw.index.str.match(r"^Mar \d{4}$")].copy()
    march.index      = march.index.str.replace("Mar ", "").astype(int)
    march.index.name = "year"
    march            = march[march.index >= MIN_YEAR]

    if len(march) < MIN_MARCH_YEARS:
        return None

    # Strip % and commas before numeric conversion
    def _to_num(s):
        cleaned = s.astype(str).str.replace("%", "").str.replace(",", "").str.strip()
        return pd.to_numeric(cleaned, errors="coerce")

    return march.apply(_to_num)


def load_detailed_bs(ticker: str) -> pd.DataFrame | None:
    """Load the Playwright-scraped expanded balance sheet."""
    path = os.path.join(SOURCE_DB, ticker, "balance_sheet_detailed.parquet")
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None

    # Index is already metric names; columns are period strings ("Mar 2015" …)
    march = df[[c for c in df.columns if str(c).startswith("Mar")]].copy()
    march.columns = [str(c).replace("Mar ", "") for c in march.columns]
    march.columns = march.columns.astype(int)
    march = march.loc[:, march.columns >= MIN_YEAR]
    if march.shape[1] < MIN_MARCH_YEARS:
        return None
    # Transpose: rows = year, columns = metric
    out = march.T
    out.index.name = "year"
    return out.apply(pd.to_numeric, errors="coerce")


# ── Factor derivation ──────────────────────────────────────────────────────────

def derive_bs_factors(bs: pd.DataFrame) -> pd.DataFrame:
    f       = pd.DataFrame(index=bs.index)
    eq      = _col(bs, "Equity Capital") + _col(bs, "Reserves")
    ta      = _col(bs, "Total Assets").replace(0, np.nan)
    fa      = _col(bs, "Fixed Assets")
    cw      = _col(bs, "CWIP")
    br      = _col(bs, "Borrowings")               # keep 0 — zero debt is valid data
    fa_cwip = (fa + cw).replace(0, np.nan)

    f["Debt_to_Equity"]     = br / eq.replace(0, np.nan)   # 0/eq = 0 for debt-free cos
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
    f["Borrowings_Growth"]  = br.replace(0, np.nan).pct_change() * 100  # growth undefined when no debt
    return f


def derive_pl_cf_factors(pl: pd.DataFrame, cf: pd.DataFrame,
                          bs: pd.DataFrame) -> pd.DataFrame:
    f     = pd.DataFrame(index=pl.index)
    sales = _col(pl, "Sales").replace(0, np.nan)
    op    = _col(pl, "Operating Profit")
    depr  = _col(pl, "Depreciation")
    np_   = _col(pl, "Net Profit")
    intr  = _col(pl, "Interest").replace(0, np.nan)
    cfo   = _col(cf, "Cash from Operating Activity")
    fcf   = _col(cf, "Free Cash Flow")
    eq    = (_col(bs, "Equity Capital") + _col(bs, "Reserves")).replace(0, np.nan)
    ta    = _col(bs, "Total Assets").replace(0, np.nan)
    fa    = _col(bs, "Fixed Assets").replace(0, np.nan)
    ebitda = op + depr

    # Profitability
    f["Operating_Margin"]  = op     / sales * 100
    f["EBITDA_Margin"]     = ebitda / sales * 100
    f["Net_Margin"]        = np_    / sales * 100
    f["ROE"]               = np_    / eq    * 100
    f["ROA"]               = np_    / ta    * 100
    f["Interest_Coverage"] = op     / intr

    # Efficiency / Turnover
    f["Asset_Turnover"]       = sales / ta
    f["FixedAsset_Turnover"]  = sales / fa

    # Growth
    f["Revenue_Growth"]       = sales.pct_change()  * 100
    f["OpProfit_Growth"]      = op.pct_change()     * 100
    f["EBITDA_Growth"]        = ebitda.pct_change() * 100
    f["NetProfit_Growth"]     = np_.pct_change()    * 100

    # Cash quality
    f["CFO_to_NetProfit"]     = cfo / np_.replace(0, np.nan)
    f["FCF_Margin"]           = fcf / sales * 100

    # Capex
    capex = (cfo - fcf).abs()                        # Capex = CFO - FCF
    f["Capex_to_Sales"]       = capex / sales * 100
    f["Capex_to_Depreciation"]= capex / depr.replace(0, np.nan)

    return f


def derive_detailed_bs_factors(dbs: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=dbs.index)

    def g(name):
        return dbs[name] if name in dbs.columns else pd.Series(0.0, index=dbs.index)

    # Current assets (from Other Assets expansion)
    inv  = g("Inventories")
    rec  = g("Trade receivables")
    cash = g("Cash Equivalents")
    loans= g("Loans n Advances")
    oai  = g("Other asset items")
    curr_assets = inv + rec + cash + loans + oai

    # Current liabilities (from Other Liabilities expansion)
    tp   = g("Trade Payables")
    stb  = g("Short term Borrowings")
    afc  = g("Advance from Customers")
    oli  = g("Other liability items")
    curr_liab = tp + stb + afc + oli

    curr_liab_safe = curr_liab.replace(0, np.nan)
    inv_safe       = inv.replace(0, np.nan)

    # Liquidity ratios
    f["Current_Ratio"]  = curr_assets / curr_liab_safe
    f["Quick_Ratio"]    = (curr_assets - inv) / curr_liab_safe
    f["Cash_Ratio"]     = cash / curr_liab_safe

    # Debt composition
    ltb = g("Long term Borrowings")
    f["LT_Debt_Ratio"]  = ltb / (ltb + stb).replace(0, np.nan)  # % of debt that is long-term

    # Net Debt
    total_debt = g("Borrowings")
    net_debt   = (total_debt - cash).clip(lower=0)   # negative net debt = net cash
    eq         = g("Equity Capital") + g("Reserves")
    f["Net_Debt_to_Equity"] = net_debt / eq.replace(0, np.nan)

    # Asset quality
    gross = g("Gross Block").replace(0, np.nan)
    acc_dep = g("Accumulated Depreciation")
    f["Asset_Age_Ratio"]    = acc_dep / gross  # higher = older assets, nearing replacement

    # Absolute working capital components (normalised by total assets)
    ta = g("Total Assets").replace(0, np.nan)
    f["Inventory_to_Assets"]    = inv  / ta
    f["Receivables_to_Assets"]  = rec  / ta
    f["Cash_to_Assets"]         = cash / ta
    f["TradePayables_to_Assets"]= tp   / ta

    return f


def derive_ratio_factors(ratios: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=ratios.index)

    def _pct_col(name: str) -> pd.Series:
        s = _col(ratios, name)
        if s.dtype == object:
            s = s.astype(str).str.replace("%", "").str.strip()
            s = pd.to_numeric(s, errors="coerce")
        return s

    f["Debtor_Days"]          = pd.to_numeric(_col(ratios, "Debtor Days"),    errors="coerce")
    f["Inventory_Days"]       = pd.to_numeric(_col(ratios, "Inventory Days"), errors="coerce")
    f["Days_Payable"]         = pd.to_numeric(_col(ratios, "Days Payable"),   errors="coerce")
    f["Cash_Conversion_Cycle"]= pd.to_numeric(_col(ratios, "Cash Conversion Cycle"), errors="coerce")
    f["Working_Capital_Days"] = pd.to_numeric(_col(ratios, "Working Capital Days"),  errors="coerce")
    f["ROCE"]                 = _pct_col("ROCE %")

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

            pl     = load_pl(ticker)
            cf     = load_cf(ticker)
            ratios = load_ratios(ticker)
            dbs    = load_detailed_bs(ticker)

            bs_f   = derive_bs_factors(bs)
            common = bs.index

            if pl is not None and cf is not None:
                c3 = bs.index.intersection(pl.index).intersection(cf.index)
                if len(c3) >= MIN_MARCH_YEARS:
                    plcf_f = derive_pl_cf_factors(pl.loc[c3], cf.loc[c3], bs.loc[c3])
                    row    = pd.concat([bs_f.loc[c3], plcf_f], axis=1)
                    common = c3
                else:
                    row = bs_f
            else:
                row = bs_f

            if ratios is not None:
                cr = common.intersection(ratios.index)
                if len(cr) >= MIN_MARCH_YEARS:
                    rat_f = derive_ratio_factors(ratios.loc[cr])
                    row   = row.loc[cr].join(rat_f)
                    common = cr

            if dbs is not None:
                cd = common.intersection(dbs.index)
                if len(cd) >= MIN_MARCH_YEARS:
                    dbs_f = derive_detailed_bs_factors(dbs.loc[cd])
                    row   = row.loc[cd].join(dbs_f)

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
