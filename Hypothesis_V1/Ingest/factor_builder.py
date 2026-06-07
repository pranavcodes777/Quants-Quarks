"""
factor_builder.py — Comprehensive Factor Derivation
====================================================
Derives ~100 fundamental factors from 4 detailed financial statements:
  Annual BS, Annual P&L, Annual CF, Quarterly P&L

Factor categories:
  1.  BS Fundamentals          (16) — leverage, asset mix, size, growth
  2.  P&L + CF Fundamentals    (16) — margins, returns, efficiency, momentum
  3.  Cost Structure            (5) — cost breakdown from detailed PL
  4.  Earnings Quality         (12) — other income, exceptionals, tax, dividends, accruals
  5.  Cash Flow Quality        (16) — WC changes, capex, M&A, debt coverage, CF momentum
  6.  Capital Allocation        (5) — ROIC, reinvestment, FCF conversion
  7.  Detailed Asset Mix        (9) — intangibles, leases, land, asset freshness
  8.  Consistency & Trend       (9) — CAGR, stability, margin trends, FCF reliability
  9.  Ratio factors             (6) — Screener pre-computed (old DB)
  10. Detailed Liquidity/Debt  (10) — current ratio, quick ratio, sub-item leverage
  11. Quarterly Dynamics        (4) — seasonality, momentum from quarterly PL
"""

import os
import sys
import re
import numpy as np
import pandas as pd
from collections import Counter
def _linregress_sp(x, y):
    """Minimal OLS using numpy — only slope is used by factor_builder."""
    x, y  = np.asarray(x, float), np.asarray(y, float)
    n     = len(x)
    if n < 2:
        raise ValueError("need at least 2 points")
    mx, my = x.mean(), y.mean()
    ss_xx  = ((x - mx) ** 2).sum()
    if ss_xx == 0:
        raise ValueError("all x values identical")
    slope     = ((x - mx) * (y - my)).sum() / ss_xx
    intercept = my - slope * mx
    return slope, intercept, np.nan, np.nan, np.nan

# ── Paths ──────────────────────────────────────────────────────────────────────
NEW_DB     = r"E:\Quarks&Quants\Non Fundamental\Hypothesis V1\Database"
OLD_DB     = r"E:\Quarks&Quants\Fundamental\Financial Statements\Database"
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
    return df[name].copy() if name in df.columns else pd.Series(0.0, index=df.index)

def _linreg(x, y):
    try:
        return _linregress_sp(x, y)
    except Exception:
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

def _rolling_slope(s: pd.Series, window: int = 4) -> pd.Series:
    result = pd.Series(np.nan, index=s.index)
    arr    = s.values
    for i in range(window - 1, len(arr)):
        chunk = arr[i - window + 1 : i + 1]
        valid = [(j, v) for j, v in enumerate(chunk) if not np.isnan(v)]
        if len(valid) >= max(2, window // 2):
            xs, ys = zip(*valid)
            slope, *_ = _linreg(xs, ys)
            result.iloc[i] = slope
    return result

def _cagr(s: pd.Series, n: int = 5) -> pd.Series:
    result = pd.Series(np.nan, index=s.index)
    vals   = s.values
    for i in range(n, len(vals)):
        start, end = vals[i - n], vals[i]
        if pd.notna(start) and pd.notna(end) and start > 0 and end > 0:
            result.iloc[i] = ((end / start) ** (1 / n) - 1) * 100
    return result


# ── Loaders ────────────────────────────────────────────────────────────────────
def _load_new(ticker: str, filename: str) -> pd.DataFrame | None:
    path = os.path.join(NEW_DB, ticker, filename)
    if not os.path.exists(path):
        return None
    try:
        raw = pd.read_parquet(path)
    except Exception:
        return None

    cols = [c for c in raw.columns if re.match(r"^[A-Z][a-z]{2} \d{4}$", str(c).strip())]
    if not cols:
        return None
    raw = raw[cols].copy()

    df = raw.T
    df.index.name = "period"

    months     = [str(idx)[:3] for idx in df.index]
    yr_end_mon = Counter(months).most_common(1)[0][0]
    df = df[[str(idx).startswith(yr_end_mon) for idx in df.index]].copy()
    df.index = [int(str(idx)[-4:]) for idx in df.index]
    df.index.name = "year"
    df = df[df.index >= MIN_YEAR]

    if len(df) < MIN_YEARS:
        return None

    df.columns = [_clean(str(c)) for c in df.columns]
    return df.apply(pd.to_numeric, errors="coerce")


def _load_new_quarterly(ticker: str) -> pd.DataFrame | None:
    path = os.path.join(NEW_DB, ticker, "quarterly_pl.parquet")
    if not os.path.exists(path):
        return None
    try:
        raw = pd.read_parquet(path)
    except Exception:
        return None
    if raw.empty:
        return None
    raw.columns = [_clean(str(c)) for c in raw.columns]
    df = raw.T.copy()
    df.columns = [_clean(str(c)) for c in df.columns]
    return df.apply(pd.to_numeric, errors="coerce")


def _load_old(ticker: str, filename: str) -> pd.DataFrame | None:
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


# ══════════════════════════════════════════════════════════════════════════════
# FACTOR DERIVATION
# ══════════════════════════════════════════════════════════════════════════════

def derive_bs_factors(bs: pd.DataFrame) -> pd.DataFrame:
    """16 factors — leverage, asset mix, size, structural growth."""
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


def derive_pl_cf_factors(pl: pd.DataFrame, cf: pd.DataFrame, bs: pd.DataFrame) -> pd.DataFrame:
    """16 factors — core margins, returns, turnover, momentum."""
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

    f["Operating_Margin"]      = op     / sales * 100
    f["EBITDA_Margin"]         = ebitda / sales * 100
    f["Net_Margin"]            = np_    / sales * 100
    f["ROE"]                   = np_    / eq    * 100
    f["ROA"]                   = np_    / ta    * 100
    f["Interest_Coverage"]     = op     / intr
    f["Asset_Turnover"]        = sales  / ta
    f["FixedAsset_Turnover"]   = sales  / fa
    f["Revenue_Growth"]        = sales.pct_change()  * 100
    f["OpProfit_Growth"]       = op.pct_change()     * 100
    f["EBITDA_Growth"]         = ebitda.pct_change() * 100
    f["NetProfit_Growth"]      = np_.pct_change()    * 100
    f["CFO_to_NetProfit"]      = cfo / np_.replace(0, np.nan)
    f["FCF_Margin"]            = fcf / sales * 100
    capex = (cfo - fcf).abs()
    f["Capex_to_Sales"]        = capex / sales * 100
    f["Capex_to_Depreciation"] = capex / depr.replace(0, np.nan)
    return f


def derive_cost_structure(pl: pd.DataFrame) -> pd.DataFrame:
    """5 factors — cost breakdown only possible from expanded P&L sub-items."""
    f     = pd.DataFrame(index=pl.index)
    sales = _col(pl, "Sales").replace(0, np.nan)

    f["Material_Cost_Pct"]      = _col(pl, "Material Cost %")
    f["Employee_Cost_Pct"]      = _col(pl, "Employee Cost %")
    f["Manufacturing_Cost_Pct"] = _col(pl, "Manufacturing Cost %")
    f["Other_Cost_Pct"]         = _col(pl, "Other Cost %")

    emp_abs = _col(pl, "Employee Cost %") * sales / 100
    f["Employee_Productivity"]  = sales / emp_abs.replace(0, np.nan)  # revenue per ₹ of employee cost
    return f


def derive_earnings_quality(pl: pd.DataFrame, cf: pd.DataFrame) -> pd.DataFrame:
    """
    12 factors — separates clean recurring earnings from noise.
    Other income, exceptionals, tax quality, dividends, Sloan accrual.
    """
    f      = pd.DataFrame(index=pl.index)
    sales  = _col(pl, "Sales").replace(0, np.nan)
    op     = _col(pl, "Operating Profit").replace(0, np.nan)
    np_    = _col(pl, "Net Profit")
    np_s   = np_.replace(0, np.nan)
    pbt    = _col(pl, "Profit before tax").replace(0, np.nan)
    cfo    = _col(cf, "Cash from Operating Activity")

    oi = _col(pl, "Other Income")
    f["Other_Income_to_OP"]     = oi / op * 100         # other income as % of operating profit
    f["Other_Income_to_NP"]     = oi / np_s * 100       # what % of net profit is non-core
    f["Core_EBIT_Purity"]       = op / pbt * 100        # % of PBT from core operations

    excep_at    = _col(pl, "Exceptional items AT")
    profit_excl = _col(pl, "Profit excl Excep").replace(0, np.nan)
    f["Profit_Quality"]         = profit_excl / np_s    # clean / reported (1.0 = no exceptionals)
    f["Exceptional_to_NP"]      = excep_at.abs() / np_s * 100

    f["Effective_Tax_Rate"]     = _col(pl, "Tax %")
    f["Dividend_Payout"]        = _col(pl, "Dividend Payout %")

    eps = _col(pl, "EPS in Rs").replace(0, np.nan)
    f["EPS_Growth"]             = eps.pct_change() * 100

    minority = _col(pl, "Minority share").abs()
    f["Minority_Earnings_Drag"] = minority / pbt * 100

    f["Recurring_NP_Margin"]    = profit_excl / sales * 100

    # CFO/OP — Screener already computes this; operating cash conversion efficiency
    f["CFO_to_OP"]              = _col(cf, "CFO/OP")

    # Sloan Accrual Ratio: (Net Income - CFO) / Sales
    # Positive = earnings run ahead of cash (accrual inflation). Empirically predicts reversal.
    f["Accrual_Ratio"]          = (np_ - cfo) / sales * 100
    return f


def derive_cf_quality(cf: pd.DataFrame, pl: pd.DataFrame, bs: pd.DataFrame) -> pd.DataFrame:
    """
    16 factors — decomposes all 3 CF statements into investable signals.
    WC quality, capex vs disposals, M&A intensity, debt coverage, CF momentum.
    """
    f      = pd.DataFrame(index=cf.index)
    sales  = _col(pl, "Sales").replace(0, np.nan)
    op     = _col(pl, "Operating Profit")
    depr   = _col(pl, "Depreciation").replace(0, np.nan)
    ebitda = (op + depr).replace(0, np.nan)
    cfo    = _col(cf, "Cash from Operating Activity")
    fcf    = _col(cf, "Free Cash Flow")
    cfo_s  = cfo.replace(0, np.nan)

    # Operating CF decomposition
    wc_chg = _col(cf, "Working capital changes")
    f["WC_Change_to_CFO"]       = wc_chg / cfo_s              # WC drag; neg = WC consumed cash
    f["Receivables_Drag"]       = _col(cf, "Receivables") / sales * 100
    f["Inventory_Drag"]         = _col(cf, "Inventory")   / sales * 100
    f["Payables_Support"]       = _col(cf, "Payables")    / sales * 100

    # Investing CF: true capex vs disposals
    fa_pur = _col(cf, "Fixed assets purchased").abs()
    fa_sol = _col(cf, "Fixed assets sold").abs()
    f["Net_Capex_Ratio"]        = (fa_pur - fa_sol) / sales * 100   # true economic capex
    f["Asset_Disposal_Rate"]    = fa_sol / fa_pur.replace(0, np.nan) # recycling old assets

    # Financial investments and M&A
    inv_pur = _col(cf, "Investments purchased").abs()
    inv_sol = _col(cf, "Investments sold").abs()
    f["Net_Fin_Investment"]     = (inv_pur - inv_sol) / sales * 100  # net treasury/portfolio activity
    f["Acquisition_Intensity"]  = _col(cf, "Acquisition of companies").abs() / sales * 100

    # Return on financial investments
    int_rec  = _col(cf, "Interest received")
    div_rec  = _col(cf, "Dividends received")
    inv_base = _col(bs, "Investments").replace(0, np.nan)
    f["Investment_Income_Yield"] = (int_rec + div_rec) / inv_base * 100

    # Financing CF
    cff = _col(cf, "Cash from Financing Activity")
    f["Financing_Dependency"]   = cff / sales * 100   # neg = returning cash to providers

    buyback = _col(cf, "Redemp n Canc of Shares").abs()
    eq      = (_col(bs, "Equity Capital") + _col(bs, "Reserves")).replace(0, np.nan)
    f["Buyback_Intensity"]      = buyback / eq * 100

    # Debt coverage
    debt    = _col(bs, "Borrowings").replace(0, np.nan)
    cash_eq = _col(bs, "Cash Equivalents")
    net_dbt = (_col(bs, "Borrowings") - cash_eq).clip(lower=0)
    f["FCF_to_Debt"]            = fcf / debt
    f["Net_Debt_to_EBITDA"]     = net_dbt / ebitda
    f["Debt_to_EBITDA"]         = _col(bs, "Borrowings") / ebitda
    f["Interest_to_CFO"]        = _col(pl, "Interest").abs() / cfo_s  # real cash interest burden

    # Cash vs book tax — quality of tax accruals
    tax_paid = _col(cf, "Direct taxes").abs()
    pbt      = _col(pl, "Profit before tax").replace(0, np.nan)
    f["Cash_Tax_Rate"]          = tax_paid / pbt * 100

    # Depreciation coverage: can operating cash fund asset replacement?
    f["Depreciation_Coverage"]  = cfo / depr

    # CF momentum
    f["CFO_Growth"]             = cfo.pct_change() * 100
    f["FCF_Growth"]             = fcf.pct_change() * 100
    return f


def derive_capital_allocation(pl: pd.DataFrame, cf: pd.DataFrame, bs: pd.DataFrame) -> pd.DataFrame:
    """5 factors — ROIC and reinvestment quality; the engine of long-run value creation."""
    f       = pd.DataFrame(index=pl.index)
    op      = _col(pl, "Operating Profit")
    tax_pct = (_col(pl, "Tax %") / 100).clip(0, 0.50)
    nopat   = op * (1 - tax_pct)

    eq       = _col(bs, "Equity Capital") + _col(bs, "Reserves")
    debt     = _col(bs, "Borrowings")
    cash     = _col(bs, "Cash Equivalents")
    net_debt = (debt - cash).clip(lower=0)
    ic       = (eq + net_debt).replace(0, np.nan)

    f["ROIC"]             = nopat / ic * 100   # best single return metric for non-banks

    cfo     = _col(cf, "Cash from Operating Activity")
    fcf     = _col(cf, "Free Cash Flow")
    capex   = (cfo - fcf).abs()
    wc_chg  = _col(cf, "Working capital changes").abs()
    nopat_s = nopat.replace(0, np.nan)
    sales   = _col(pl, "Sales").replace(0, np.nan)

    f["Reinvestment_Rate"] = (capex + wc_chg) / nopat_s   # fraction of NOPAT reinvested
    f["FCF_Conversion"]    = fcf / nopat_s * 100           # NOPAT→FCF efficiency; 100% = perfect
    f["Cash_Accumulation"] = _col(cf, "Net Cash Flow") / sales * 100  # net cash build per unit of revenue
    f["ROIC_Growth"]       = f["ROIC"].pct_change() * 100
    return f


def derive_asset_detail(bs: pd.DataFrame, pl: pd.DataFrame) -> pd.DataFrame:
    """9 factors — sub-item asset composition unlocked by expanded BS."""
    f  = pd.DataFrame(index=bs.index)
    ta = _col(bs, "Total Assets").replace(0, np.nan)
    gb = _col(bs, "Gross Block").replace(0, np.nan)
    eq = (_col(bs, "Equity Capital") + _col(bs, "Reserves")).replace(0, np.nan)

    f["Intangible_Ratio"]        = _col(bs, "Intangible Assets") / ta * 100   # brand/patent/goodwill intensity
    f["Lease_Intensity"]         = _col(bs, "Lease Liabilities") / ta * 100   # Ind-AS-116 hidden leverage
    f["Lease_to_Equity"]         = _col(bs, "Lease Liabilities") / eq

    f["Land_to_GrossBlock"]      = _col(bs, "Land") / gb * 100     # land = no depreciation, hidden value
    f["Plant_to_GrossBlock"]     = _col(bs, "Plant Machinery") / gb * 100

    acc_dep = _col(bs, "Accumulated Depreciation")
    f["Net_Block_Freshness"]     = 1 - (acc_dep / gb)    # 1=brand new assets, 0=fully depreciated
    f["Gross_Block_Growth"]      = _col(bs, "Gross Block").pct_change() * 100

    # Customer advances: paying upfront = pricing power / recurring demand signal
    sales = _col(pl, "Sales").replace(0, np.nan)
    f["Advance_from_Customers"]  = _col(bs, "Advance from Customers") / sales * 100

    f["Minority_Interest_Ratio"] = _col(bs, "Non controlling int").abs() / eq * 100
    return f


def derive_consistency(pl: pd.DataFrame, cf: pd.DataFrame, bs: pd.DataFrame) -> pd.DataFrame:
    """9 factors — multi-year stability, CAGR, trend slopes, FCF reliability."""
    f = pd.DataFrame(index=pl.index)

    sales = _col(pl, "Sales").replace(0, np.nan)
    op    = _col(pl, "Operating Profit")
    np_   = _col(pl, "Net Profit")
    eps   = _col(pl, "EPS in Rs")
    eq    = _col(bs, "Equity Capital") + _col(bs, "Reserves")
    fcf   = _col(cf, "Free Cash Flow")

    opm = op / sales.replace(0, np.nan) * 100
    roe = np_ / eq.replace(0, np.nan) * 100

    # Compounding power — 5-year CAGRs
    f["Revenue_CAGR_5Y"]     = _cagr(sales, 5)
    f["NP_CAGR_5Y"]          = _cagr(np_.where(np_ > 0, np.nan), 5)
    f["EPS_CAGR_5Y"]         = _cagr(eps.where(eps > 0, np.nan), 5)

    # Stability: negated rolling std — higher value = more stable = quality premium
    f["OPM_Stability"]       = -opm.rolling(3, min_periods=2).std()
    f["ROE_Stability"]       = -roe.rolling(3, min_periods=2).std()

    # Trend direction — positive = improving
    f["OPM_Trend"]           = _rolling_slope(opm, 4)
    f["ROE_Trend"]           = _rolling_slope(roe, 4)

    # Margin expansion over the prior 3 years
    f["Margin_Expansion_3Y"] = opm.diff(3)

    # FCF reliability: % of last 5 years with positive FCF
    f["FCF_Reliability"]     = (
        fcf.rolling(5, min_periods=3)
           .apply(lambda x: (x > 0).sum() / len(x) * 100, raw=True)
    )
    return f


def derive_detailed_bs_factors(bs: pd.DataFrame) -> pd.DataFrame:
    """10 factors — liquidity ratios and sub-item leverage from expanded BS."""
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
    curr_liab   = tp + stb + oli
    curr_liab_s = curr_liab.replace(0, np.nan)

    f["Current_Ratio"]       = curr_assets / curr_liab_s
    f["Quick_Ratio"]         = (curr_assets - inv) / curr_liab_s
    f["Cash_Ratio"]          = cash / curr_liab_s

    ltb = _col(bs, "Long term Borrowings")
    f["LT_Debt_Ratio"]       = ltb / (ltb + stb).replace(0, np.nan)

    eq       = (_col(bs, "Equity Capital") + _col(bs, "Reserves")).replace(0, np.nan)
    net_debt = (_col(bs, "Borrowings") - cash).clip(lower=0)
    f["Net_Debt_to_Equity"]  = net_debt / eq

    ta  = _col(bs, "Total Assets").replace(0, np.nan)
    gb  = _col(bs, "Gross Block").replace(0, np.nan)
    acc = _col(bs, "Accumulated Depreciation")
    f["Asset_Age_Ratio"]          = acc / gb
    f["Inventory_to_Assets"]      = inv  / ta
    f["Receivables_to_Assets"]    = rec  / ta
    f["Cash_to_Assets"]           = cash / ta
    f["TradePayables_to_Assets"]  = tp   / ta
    return f


def derive_ratio_factors(ratios: pd.DataFrame) -> pd.DataFrame:
    """6 factors — Screener pre-computed ratios (only from old DB)."""
    f = pd.DataFrame(index=ratios.index)

    def _pct_col(name):
        s = _col(ratios, name)
        if s.dtype == object:
            s = pd.to_numeric(s.astype(str).str.replace("%", "").str.strip(), errors="coerce")
        return s

    f["Debtor_Days"]           = pd.to_numeric(_col(ratios, "Debtor Days"),            errors="coerce")
    f["Inventory_Days"]        = pd.to_numeric(_col(ratios, "Inventory Days"),         errors="coerce")
    f["Days_Payable"]          = pd.to_numeric(_col(ratios, "Days Payable"),           errors="coerce")
    f["Cash_Conversion_Cycle"] = pd.to_numeric(_col(ratios, "Cash Conversion Cycle"),  errors="coerce")
    f["Working_Capital_Days"]  = pd.to_numeric(_col(ratios, "Working Capital Days"),   errors="coerce")
    f["ROCE"]                  = _pct_col("ROCE %")
    return f


def derive_quarterly_factors(qpl: pd.DataFrame) -> dict[str, float]:
    """
    4 scalar factors from quarterly P&L (only 13 quarters / ~3 years available).
    Broadcast as constants across all years for the company — represents current character.
    """
    out = {}
    if qpl is None or qpl.empty:
        return out

    sales = _col(qpl, "Sales").replace(0, np.nan).dropna()
    opm   = _col(qpl, "OPM %").dropna()
    np_   = _col(qpl, "Net Profit").dropna()

    if len(sales) >= 4:
        mean_s = sales.mean()
        if pd.notna(mean_s) and mean_s != 0:
            out["Revenue_Seasonality"]    = sales.std() / mean_s * 100   # CV of quarterly revenues

        if len(opm) >= 4:
            out["OPM_Quarterly_Vol"]      = float(opm.std())             # intra-year margin volatility

        if len(np_) >= 4:
            out["Qtr_Profit_Consistency"] = float((np_ > 0).sum() / len(np_) * 100)  # % profitable quarters

        if len(sales) >= 5:
            recent, prior = sales.iloc[-1], sales.iloc[-5]
            if pd.notna(prior) and prior > 0:
                out["Recent_Revenue_Momentum"] = float((recent / prior - 1) * 100)  # YoY recent quarter

    return out


# ══════════════════════════════════════════════════════════════════════════════
# MASTER BUILDER
# ══════════════════════════════════════════════════════════════════════════════
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
            qpl    = _load_new_quarterly(ticker)

            # Common year index across all annual statements
            common = bs.index
            if pl is not None:
                common = common.intersection(pl.index)
            if cf is not None:
                common = common.intersection(cf.index)

            if len(common) < MIN_YEARS:
                print(f"  SKIP {ticker} — only {len(common)} common years")
                continue

            row = derive_bs_factors(bs.loc[common])

            if pl is not None and cf is not None:
                row = row.join(derive_pl_cf_factors(pl.loc[common], cf.loc[common], bs.loc[common]))

            if pl is not None:
                row = row.join(derive_cost_structure(pl.loc[common]))

            if pl is not None and cf is not None:
                row = row.join(derive_earnings_quality(pl.loc[common], cf.loc[common]))
                row = row.join(derive_cf_quality(cf.loc[common], pl.loc[common], bs.loc[common]))
                row = row.join(derive_capital_allocation(pl.loc[common], cf.loc[common], bs.loc[common]))
                row = row.join(derive_consistency(pl.loc[common], cf.loc[common], bs.loc[common]))

            if pl is not None:
                row = row.join(derive_asset_detail(bs.loc[common], pl.loc[common]))

            row = row.join(derive_detailed_bs_factors(bs.loc[common]))

            if ratios is not None:
                cr = common.intersection(ratios.index)
                if len(cr) >= MIN_YEARS:
                    row = row.loc[cr].join(derive_ratio_factors(ratios.loc[cr]))

            # Quarterly factors — scalar, broadcast across all years
            for k, v in derive_quarterly_factors(qpl).items():
                row[k] = v

            row.insert(0, "Company", ticker)
            row.insert(1, "Sector",  sector)
            rows.append(row.reset_index())

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def merge_forward_returns(df: pd.DataFrame) -> pd.DataFrame:
    ret_path = os.path.join(OUTPUT_DIR, "returns.parquet")
    if not os.path.exists(ret_path):
        print("  Warning: returns.parquet not found — skipping forward returns")
        return df

    returns         = pd.read_parquet(ret_path).copy()
    returns["year"] = returns["year"] - 1
    returns         = returns.rename(columns={"Return_1Y": "Return_1Y_Fwd"})
    merged          = df.merge(returns, on=["Company", "year"], how="left")
    n_fwd           = merged["Return_1Y_Fwd"].notna().sum()
    print(f"  Forward returns matched: {n_fwd} of {len(merged)} rows")
    return merged


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    sectors = sys.argv[1:] if len(sys.argv) > 1 else None
    label   = ", ".join(sectors) if sectors else "ALL"

    print(f"\nBuilding factors — {label}")
    print("=" * 55)

    df = build_sector_factors(sectors)
    if df.empty:
        print("No data built. Check source paths.")
        sys.exit(1)

    df  = merge_forward_returns(df)
    out = os.path.join(OUTPUT_DIR, "bs_factors.parquet")
    df.to_parquet(out, index=False)

    factor_cols = [c for c in df.columns
                   if c not in ("year", "Company", "Sector", "Return_1Y_Fwd")]
    print(f"\n  Rows      : {len(df)}")
    print(f"  Companies : {df['Company'].nunique()}")
    print(f"  Sectors   : {df['Sector'].nunique()}")
    print(f"  Factors   : {len(factor_cols)}")
    print(f"\nFactor list:")
    for i, fc in enumerate(factor_cols, 1):
        print(f"  {i:3d}. {fc}")
    print(f"\n  Saved -> {out}")
