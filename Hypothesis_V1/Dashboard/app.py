"""
Hypothesis V1 — Factor Analysis
Phase 1: Factor Correlation
"""

import os
import sys
import colorsys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats
from scipy.cluster.hierarchy import linkage, leaves_list
from scipy.spatial.distance import squareform
from scipy.stats import linregress as _linregress

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Ingest"))
from factor_builder import SECTORS

_DB_PATH = os.path.join(os.path.dirname(__file__), "..", "Database", "bs_factors.parquet")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Hypothesis V1 · Factor Research",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS — uses Streamlit CSS variables so dark/light toggle (⋮ → Settings) works
st.markdown("""
<style>
  .block-container { padding: 1.2rem 2rem 2rem; }
  div[data-testid="stTabs"] button { font-size: 0.78rem; letter-spacing: 0.06em; }

  .kpi-wrap { display: flex; gap: 12px; margin-bottom: 1.4rem; }
  .kpi {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 7px; padding: 12px 18px; flex: 1; min-width: 0;
  }
  .kpi .num { font-size: 1.75rem; font-weight: 700; color: var(--text-color); line-height: 1; }
  .kpi .lbl { font-size: 0.65rem; color: var(--text-color); opacity: 0.45;
              text-transform: uppercase; letter-spacing: 0.09em; margin-top: 4px; }

  .section-hd {
    font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.1em;
    color: var(--text-color); opacity: 0.35; margin: 1.2rem 0 0.6rem;
  }
  .chip {
    display: inline-block; padding: 2px 9px; border-radius: 10px;
    font-size: 0.7rem; font-family: 'SF Mono', 'Fira Code', monospace; margin: 2px 3px;
  }
  .chip-r { background: #3d1a1a; border: 1px solid #da3633; color: #f85149; }
  .chip-g { background: #0d2818; border: 1px solid #2ea043; color: #3fb950; }
  .chip-b { background: #0c1e32; border: 1px solid #1f6feb; color: #58a6ff; }

  .insight-box {
    background: var(--secondary-background-color);
    border: 1px solid rgba(128,128,128,0.2);
    border-radius: 7px; padding: 14px 16px; margin-top: 8px;
  }
  .insight-box p { margin: 0 0 4px; font-size: 0.78rem; color: var(--text-color); opacity: 0.7; line-height: 1.6; }
  .insight-box b { color: var(--text-color); opacity: 1; }

  .p2-bar-track { background: rgba(128,128,128,0.15); border-radius:4px; height:8px; width:100%; margin:4px 0 10px; }
  .p2-bar-fill  { height:8px; border-radius:4px; }
  .p2-mid-wrap  { position:relative; height:8px; background:rgba(128,128,128,0.15); border-radius:4px; margin:4px 0 10px; }
  .p2-interp    { background:var(--secondary-background-color); border:1px solid rgba(128,128,128,0.2); border-radius:8px; padding:16px 20px; margin-top:14px; }
  .p2-interp p  { margin:0 0 8px; font-size:0.82rem; color:var(--text-color); opacity:0.7; line-height:1.7; }
  .p2-interp b  { color:var(--text-color); opacity:1; }
  .p2-stat-row  { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:10px; }

  .ac-hd   { font-size:0.65rem; text-transform:uppercase; letter-spacing:.1em; color:var(--text-color); opacity:0.35; margin:1.2rem 0 .5rem; }
  .ac-card { background:var(--secondary-background-color); border:1px solid rgba(128,128,128,0.2); border-radius:8px; padding:14px 18px; margin-bottom:8px; }
  .ac-card p { margin:0 0 4px; font-size:.79rem; color:var(--text-color); opacity:0.7; line-height:1.65; }
  .ac-card b { color:var(--text-color); opacity:1; }

  /* Shrink st.metric values site-wide — default is too large */
  div[data-testid="stMetricValue"] { font-size: 1.25rem !important; }
  div[data-testid="stMetricLabel"] { font-size: 0.68rem !important; }
  div[data-testid="stMetricDelta"] { font-size: 0.68rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Factor metadata ────────────────────────────────────────────────────────────
FACTOR_META: dict[str, dict] = {
    "Debt_to_Equity":     {"label": "Debt / Equity",          "group": "Leverage",     "desc": "Total borrowings relative to shareholders equity. Higher = more leveraged."},
    "Debt_to_Assets":     {"label": "Debt / Assets",          "group": "Leverage",     "desc": "Fraction of assets funded by debt."},
    "Equity_Ratio":       {"label": "Equity Ratio",           "group": "Leverage",     "desc": "Fraction of assets funded by equity. Inverse of Financial Leverage."},
    "Financial_Leverage": {"label": "Financial Leverage",     "group": "Leverage",     "desc": "Total Assets / Equity. Equity multiplier — how many rupees of assets per rupee of equity."},
    "OtherLiab_Ratio":    {"label": "Other Liab / Assets",   "group": "Leverage",     "desc": "Non-debt obligations (trade payables, provisions) as a share of assets. Working-capital proxy."},
    "FixedAsset_Ratio":   {"label": "Fixed Asset Ratio",      "group": "Asset Mix",    "desc": "Tangible fixed assets as a fraction of total assets. Higher in capital-intensive businesses."},
    "CWIP_Intensity":     {"label": "CWIP Intensity",         "group": "Asset Mix",    "desc": "Capital work-in-progress as a fraction of gross block. Elevated = expansion underway."},
    "Investment_Ratio":   {"label": "Investment Ratio",       "group": "Asset Mix",    "desc": "Financial investments as a fraction of total assets. Strategic or treasury deployment."},
    "OtherAsset_Ratio":   {"label": "Other Asset Ratio",     "group": "Asset Mix",    "desc": "Current & other assets as a fraction of total assets. Includes working capital."},
    "Tangible_Ratio":     {"label": "Tangible Ratio",         "group": "Asset Mix",    "desc": "Fixed Assets + CWIP as a share of total assets."},
    "Log_Total_Assets":   {"label": "Log(Total Assets)",      "group": "Size",         "desc": "Natural log of total assets. Controls for company size effects."},
    "Assets_Growth":      {"label": "Asset Growth %",         "group": "Growth",       "desc": "Year-on-year % growth in total assets. Reflects expansion pace."},
    "Equity_Growth":      {"label": "Equity Growth %",        "group": "Growth",       "desc": "Year-on-year % growth in total equity (capital + reserves)."},
    "FixedAssets_Growth": {"label": "Fixed Asset Growth %",   "group": "Growth",       "desc": "Year-on-year % growth in net fixed assets."},
    "Reserves_Growth":    {"label": "Reserves Growth %",      "group": "Growth",       "desc": "Year-on-year % growth in reserves. Strong proxy for retained earnings build-up."},
    "Borrowings_Growth":  {"label": "Borrowings Growth %",    "group": "Growth",       "desc": "Year-on-year % change in total borrowings. Positive = debt rising."},
    "Operating_Margin":  {"label": "Operating Margin %",  "group": "Profitability", "desc": "Operating Profit / Sales. Core business profitability before interest and tax."},
    "Net_Margin":        {"label": "Net Margin %",        "group": "Profitability", "desc": "Net Profit / Sales. How much of each revenue rupee becomes profit after everything."},
    "ROE":               {"label": "ROE %",               "group": "Profitability", "desc": "Net Profit / Equity. Return generated on shareholders capital. The most watched profitability metric."},
    "ROA":               {"label": "ROA %",               "group": "Profitability", "desc": "Net Profit / Total Assets. Returns on every rupee of assets deployed regardless of funding source."},
    "Interest_Coverage": {"label": "Interest Coverage",   "group": "Profitability", "desc": "Operating Profit / Interest. Times earnings cover debt payments. NaN means zero debt — not a risk."},
    "Revenue_Growth":    {"label": "Revenue Growth %",    "group": "Growth",        "desc": "Year-on-year sales growth. Top-line momentum."},
    "NetProfit_Growth":  {"label": "Net Profit Growth %", "group": "Growth",        "desc": "Year-on-year net profit growth. Bottom-line momentum."},
    "CFO_to_NetProfit":  {"label": "CFO / Net Profit",    "group": "Cash Quality",  "desc": "Cash from Operations / Net Profit. >1 = profits backed by real cash. The single best earnings quality test."},
    "FCF_Margin":        {"label": "FCF Margin %",        "group": "Cash Quality",  "desc": "Free Cash Flow / Sales. True economic profitability after all capex. What the business actually generates."},
    "EBITDA_Margin":          {"label": "EBITDA Margin %",        "group": "Profitability", "desc": "(Operating Profit + Depreciation) / Sales. Profitability before interest, tax, and accounting charges."},
    "Asset_Turnover":         {"label": "Asset Turnover",         "group": "Efficiency",    "desc": "Sales / Total Assets. How productively the company uses its asset base to generate revenue."},
    "FixedAsset_Turnover":    {"label": "Fixed Asset Turnover",   "group": "Efficiency",    "desc": "Sales / Fixed Assets. Revenue generated per rupee of fixed asset. Higher = sweating assets harder."},
    "OpProfit_Growth":        {"label": "Op. Profit Growth %",    "group": "Growth",        "desc": "Year-on-year operating profit growth."},
    "EBITDA_Growth":          {"label": "EBITDA Growth %",        "group": "Growth",        "desc": "Year-on-year EBITDA growth."},
    "Capex_to_Sales":         {"label": "Capex / Sales %",        "group": "Capex",         "desc": "Capital expenditure as % of sales. Higher = capital-intensive; reinvesting heavily."},
    "Capex_to_Depreciation":  {"label": "Capex / Depreciation",   "group": "Capex",         "desc": ">1 = growing asset base (growth capex). <1 = just maintaining assets (maintenance capex)."},
    "Debtor_Days":            {"label": "Debtor Days",            "group": "Efficiency",    "desc": "Days to collect from customers. Lower = faster cash collection."},
    "Inventory_Days":         {"label": "Inventory Days",         "group": "Efficiency",    "desc": "Days inventory sits before being sold. Lower = leaner operations."},
    "Days_Payable":           {"label": "Days Payable",           "group": "Efficiency",    "desc": "Days taken to pay suppliers. Higher = company has more supplier financing."},
    "Cash_Conversion_Cycle":  {"label": "Cash Conversion Cycle",  "group": "Efficiency",    "desc": "Debtor Days + Inventory Days - Days Payable. Negative = suppliers fund the business. HUL runs deeply negative."},
    "Working_Capital_Days":   {"label": "Working Capital Days",   "group": "Efficiency",    "desc": "Days of sales tied up in net working capital."},
    "ROCE":                   {"label": "ROCE %",                 "group": "Profitability", "desc": "Return on Capital Employed. Measures returns on both debt and equity capital together."},
    "Current_Ratio":       {"label": "Current Ratio",        "group": "Liquidity",     "desc": "Current Assets / Current Liabilities. >1 means short-term assets cover short-term obligations."},
    "Quick_Ratio":         {"label": "Quick Ratio",          "group": "Liquidity",     "desc": "(Current Assets - Inventory) / Current Liabilities. Tougher test — excludes inventory which may not sell quickly."},
    "Cash_Ratio":          {"label": "Cash Ratio",           "group": "Liquidity",     "desc": "Cash / Current Liabilities. The strictest liquidity test — can the company cover liabilities with cash alone."},
    "Net_Debt_to_Equity":  {"label": "Net Debt / Equity",    "group": "Liquidity",     "desc": "(Total Debt - Cash) / Equity. Negative = net cash company (more cash than debt)."},
    "LT_Debt_Ratio":       {"label": "LT Debt Ratio",        "group": "Leverage",      "desc": "Long-term Borrowings / Total Borrowings. Higher = debt is long duration, less refinancing risk."},
    "Asset_Age_Ratio":     {"label": "Asset Age Ratio",      "group": "Asset Mix",     "desc": "Accumulated Depreciation / Gross Block. Higher = older assets, capex cycle may be due soon."},
    "Inventory_to_Assets":      {"label": "Inventory / Assets",      "group": "Efficiency",    "desc": "Inventories as % of total assets. High = capital tied up in stock."},
    "Receivables_to_Assets":    {"label": "Receivables / Assets",    "group": "Efficiency",    "desc": "Trade receivables as % of total assets. High = customers slow to pay."},
    "Cash_to_Assets":           {"label": "Cash / Assets",           "group": "Liquidity",     "desc": "Cash & equivalents as % of total assets. High = strong liquidity buffer."},
    "TradePayables_to_Assets":  {"label": "Trade Payables / Assets", "group": "Efficiency",    "desc": "Trade payables as % of total assets. High = company relies heavily on supplier credit."},

    # ── Cost Structure (from expanded P&L) ──────────────────────────────────────
    "Material_Cost_Pct":      {"label": "Material Cost %",       "group": "Cost Structure",  "desc": "Raw material cost as % of sales. Directly from expanded P&L. High = input-cost sensitive; margins compress when commodity prices rise."},
    "Employee_Cost_Pct":      {"label": "Employee Cost %",       "group": "Cost Structure",  "desc": "Employee cost as % of sales. High = labour-intensive business model. IT services companies typically run 50-60%."},
    "Manufacturing_Cost_Pct": {"label": "Manufacturing Cost %",  "group": "Cost Structure",  "desc": "Manufacturing overhead (power, consumables, rent) as % of sales. Represents fixed cost intensity."},
    "Other_Cost_Pct":         {"label": "Other Cost %",          "group": "Cost Structure",  "desc": "SG&A and miscellaneous costs as % of sales. Includes distribution, marketing, admin."},
    "Employee_Productivity":  {"label": "Employee Productivity", "group": "Cost Structure",  "desc": "Revenue per rupee of employee cost. Higher = more revenue generated per unit of labour spend."},

    # ── Earnings Quality (PL + CF cross) ────────────────────────────────────────
    "Other_Income_to_OP":     {"label": "Other Income / Op Profit %",  "group": "Earnings Quality", "desc": "Other (non-operating) income as % of operating profit. High = profits rely on interest/dividends/asset sales, not core business."},
    "Other_Income_to_NP":     {"label": "Other Income / Net Profit %", "group": "Earnings Quality", "desc": "What % of net profit comes from non-core sources. >30% is a quality red flag."},
    "Core_EBIT_Purity":       {"label": "Core EBIT Purity %",          "group": "Earnings Quality", "desc": "Operating profit as % of PBT. Measures how much of pre-tax profit is from core operations vs other income."},
    "Profit_Quality":         {"label": "Profit Quality",               "group": "Earnings Quality", "desc": "Profit excl. exceptionals / Reported net profit. 1.0 = clean earnings. <1 = exceptional items dragged profit; >1 = exceptional items inflated it."},
    "Exceptional_to_NP":      {"label": "Exceptional Items / NP %",    "group": "Earnings Quality", "desc": "Size of exceptional items relative to net profit. High and recurring = noisy earnings; management using exceptionals to manage expectations."},
    "Effective_Tax_Rate":     {"label": "Effective Tax Rate %",         "group": "Earnings Quality", "desc": "Actual tax rate paid as % of PBT. Below statutory rate = deferred tax benefits, MAT credits, or tax-efficient structure."},
    "Dividend_Payout":        {"label": "Dividend Payout %",            "group": "Earnings Quality", "desc": "Dividends paid as % of net profit. High = mature, cash-generative business returning capital. Very high (>100%) = paying from reserves."},
    "EPS_Growth":             {"label": "EPS Growth %",                 "group": "Earnings Quality", "desc": "Year-on-year growth in EPS (dilution-adjusted). Better than raw net profit growth as it accounts for share issuance/buybacks."},
    "Minority_Earnings_Drag": {"label": "Minority Earnings Drag %",     "group": "Earnings Quality", "desc": "Minority interest as % of PBT. High = significant earnings leak to minority shareholders of subsidiaries."},
    "Recurring_NP_Margin":    {"label": "Recurring NP Margin %",        "group": "Earnings Quality", "desc": "Net profit excluding exceptional items / sales. The cleanest measure of true recurring profitability."},
    "CFO_to_OP":              {"label": "CFO / Operating Profit",       "group": "Earnings Quality", "desc": "Cash from operations / Operating profit. >1 = operating cash exceeds accounting profit (conservative accounting). Best near-term earnings quality test."},
    "Accrual_Ratio":          {"label": "Accrual Ratio %",              "group": "Earnings Quality", "desc": "Sloan Accrual Ratio: (Net Profit - CFO) / Sales. Positive = earnings running ahead of cash (accrual inflation). Empirically one of the strongest reversal predictors."},

    # ── Cash Flow Quality ────────────────────────────────────────────────────────
    "WC_Change_to_CFO":       {"label": "WC Change / CFO",          "group": "CF Quality", "desc": "Working capital changes / CFO. Negative = WC consumed cash (receivables/inventory grew faster than payables). HUL famously has strongly positive WC (negative WC company)."},
    "Receivables_Drag":       {"label": "Receivables Change / Sales %", "group": "CF Quality", "desc": "Change in trade receivables as % of sales (from CF working capital section). Rising = customers paying slower, credit risk building."},
    "Inventory_Drag":         {"label": "Inventory Change / Sales %",   "group": "CF Quality", "desc": "Change in inventory as % of sales. Positive = inventory building up, cash absorbed. Warning sign if persistent."},
    "Payables_Support":       {"label": "Payables Change / Sales %",    "group": "CF Quality", "desc": "Change in trade payables as % of sales. Positive = stretching suppliers, freeing cash. Negative = paying suppliers faster (possible coercion or opportunity cost)."},
    "Net_Capex_Ratio":        {"label": "Net Capex / Sales %",          "group": "CF Quality", "desc": "Fixed assets purchased minus sold, as % of sales. True economic capex excluding asset recycling. More accurate than gross capex."},
    "Asset_Disposal_Rate":    {"label": "Asset Disposal Rate",          "group": "CF Quality", "desc": "Fixed assets sold / Fixed assets purchased. >0 = actively recycling old assets. Near 1 = replacement capex cycle; very high = asset-light pivot or distress."},
    "Net_Fin_Investment":     {"label": "Net Financial Investment / Sales %", "group": "CF Quality", "desc": "Net investments purchased (financial, not capex) as % of sales. High positive = building treasury / M&A. High negative = liquidating investments."},
    "Acquisition_Intensity":  {"label": "Acquisition Intensity / Sales %",   "group": "CF Quality", "desc": "Cash paid for acquisitions / sales. Measures inorganic growth activity. High = serial acquirer; quality depends on whether ROIC improves."},
    "Investment_Income_Yield":{"label": "Investment Income Yield %",    "group": "CF Quality", "desc": "(Interest + Dividends received) / Total investments. Return on the financial investment portfolio. Useful for conglomerates and cash-rich companies."},
    "Financing_Dependency":   {"label": "Financing CF / Sales %",       "group": "CF Quality", "desc": "Cash from financing activities / sales. Negative = company returned cash to providers (dividends + buybacks > new debt). Positive = net fundraiser."},
    "Buyback_Intensity":      {"label": "Buyback / Equity %",           "group": "CF Quality", "desc": "Cash returned via share buybacks / total equity. Rising buybacks often signal management confidence and reduce dilution."},
    "FCF_to_Debt":            {"label": "FCF / Total Debt",             "group": "CF Quality", "desc": "Free cash flow coverage of total debt. How many years of FCF to repay all debt. >0.25 = strong debt coverage."},
    "Net_Debt_to_EBITDA":     {"label": "Net Debt / EBITDA",            "group": "CF Quality", "desc": "Most widely used credit leverage metric. <2x = conservatively financed; >4x = highly levered. Negative = net cash company."},
    "Debt_to_EBITDA":         {"label": "Debt / EBITDA",                "group": "CF Quality", "desc": "Gross debt / EBITDA. Similar to Net Debt/EBITDA but ignores cash. Used when cash is restricted or deployment is uncertain."},
    "Interest_to_CFO":        {"label": "Interest / CFO",               "group": "CF Quality", "desc": "Cash interest burden as % of operating cash flow. >0.3 = significant portion of operating cash servicing debt."},
    "Cash_Tax_Rate":          {"label": "Cash Tax Rate %",              "group": "CF Quality", "desc": "Taxes actually paid in cash (from CF) / PBT. Much lower than book tax rate = large deferred tax assets being utilised. Higher = limited tax shield."},
    "Depreciation_Coverage":  {"label": "Depreciation Coverage",        "group": "CF Quality", "desc": "CFO / Depreciation. >1 = operating cash covers asset wear-and-tear. <1 = company can't fund replacement capex from operations alone."},
    "CFO_Growth":             {"label": "CFO Growth %",                 "group": "CF Quality", "desc": "Year-on-year growth in cash from operations. More reliable than net profit growth as it strips accounting adjustments."},
    "FCF_Growth":             {"label": "FCF Growth %",                 "group": "CF Quality", "desc": "Year-on-year growth in free cash flow. The cleanest measure of value creation momentum."},

    # ── Capital Allocation ───────────────────────────────────────────────────────
    "ROIC":             {"label": "ROIC %",              "group": "Capital Allocation", "desc": "Return on Invested Capital = NOPAT / (Equity + Net Debt). The best single metric for long-run value creation. Companies with ROIC > WACC destroy less value compounding at reinvestment."},
    "Reinvestment_Rate":{"label": "Reinvestment Rate",   "group": "Capital Allocation", "desc": "(Capex + WC investment) / NOPAT. How much of after-tax operating profit is reinvested. High + high ROIC = compounding machine. High + low ROIC = value destruction."},
    "FCF_Conversion":   {"label": "FCF Conversion %",    "group": "Capital Allocation", "desc": "FCF / NOPAT × 100. How efficiently NOPAT converts to free cash. 100% = perfect; <50% = heavy reinvestment or working capital drag; >100% = releasing WC."},
    "Cash_Accumulation":{"label": "Cash Accumulation / Sales %", "group": "Capital Allocation", "desc": "Net cash flow / sales. Positive = building cash reserves (either conservative or waiting to deploy). Persistent negative = cash-burning business model."},
    "ROIC_Growth":      {"label": "ROIC Growth %",       "group": "Capital Allocation", "desc": "Year-on-year change in ROIC. Rising ROIC = improving capital efficiency — the compounding driver of long-run returns."},

    # ── Consistency & Trend ──────────────────────────────────────────────────────
    "Revenue_CAGR_5Y":    {"label": "Revenue CAGR 5Y %",      "group": "Consistency", "desc": "5-year compound annual growth in revenue. Removes one-year volatility; shows the sustained growth trajectory."},
    "NP_CAGR_5Y":         {"label": "Net Profit CAGR 5Y %",   "group": "Consistency", "desc": "5-year CAGR in net profit. Only computed when profit is positive throughout. Shows compounding of the bottom line."},
    "EPS_CAGR_5Y":        {"label": "EPS CAGR 5Y %",          "group": "Consistency", "desc": "5-year CAGR in earnings per share. Better than net profit CAGR as it reflects dilution from share issuance."},
    "OPM_Stability":      {"label": "OPM Stability",          "group": "Consistency", "desc": "Negated 3-year rolling std of operating margin. Higher = more stable margins = pricing power or cost discipline. Low = volatile/cyclical."},
    "ROE_Stability":      {"label": "ROE Stability",          "group": "Consistency", "desc": "Negated 3-year rolling std of ROE. Higher = more consistent returns. Buffett's key quality screen — stable high ROE compounds extremely well."},
    "OPM_Trend":          {"label": "OPM Trend (slope)",      "group": "Consistency", "desc": "OLS slope of operating margin over last 4 years. Positive = margins expanding. One of the strongest momentum signals in fundamental analysis."},
    "ROE_Trend":          {"label": "ROE Trend (slope)",      "group": "Consistency", "desc": "OLS slope of ROE over last 4 years. Rising ROE = improving capital efficiency — stock market eventually reprices this."},
    "Margin_Expansion_3Y":{"label": "Margin Expansion 3Y pp", "group": "Consistency", "desc": "Operating margin now minus operating margin 3 years ago (in percentage points). Positive = structural margin expansion underway."},
    "FCF_Reliability":    {"label": "FCF Reliability %",      "group": "Consistency", "desc": "% of last 5 years with positive FCF. 100% = always generated free cash. <60% = FCF generation is lumpy or unreliable."},

    # ── Detailed Asset Mix ───────────────────────────────────────────────────────
    "Intangible_Ratio":       {"label": "Intangible / Assets %",        "group": "Asset Detail", "desc": "Intangible assets (brand, patents, goodwill) as % of total assets. High for IT/Pharma/FMCG = significant intellectual property on books."},
    "Lease_Intensity":        {"label": "Lease Liabilities / Assets %", "group": "Asset Detail", "desc": "Lease liabilities (Ind AS 116) as % of total assets. High = significant operational leverage from leases not captured in traditional debt ratios."},
    "Lease_to_Equity":        {"label": "Lease / Equity",               "group": "Asset Detail", "desc": "Lease liabilities relative to equity. Adds hidden leverage to D/E for capital-intensive leasing businesses (retail, airlines, telecom)."},
    "Land_to_GrossBlock":     {"label": "Land / Gross Block %",         "group": "Asset Detail", "desc": "Land as % of gross block. Land doesn't depreciate — high land ratio = hidden asset value often understated on books."},
    "Plant_to_GrossBlock":    {"label": "Plant & Mach / Gross Block %", "group": "Asset Detail", "desc": "Plant & machinery as % of gross block. High = heavy manufacturing intensity. Useful for identifying capital-intensive within-sector leaders."},
    "Net_Block_Freshness":    {"label": "Asset Freshness",              "group": "Asset Detail", "desc": "1 - (Accumulated Depreciation / Gross Block). Near 1 = brand new assets. Near 0 = fully depreciated, capex cycle imminent. Low freshness often precedes heavy capex."},
    "Gross_Block_Growth":     {"label": "Gross Block Growth %",         "group": "Asset Detail", "desc": "Year-on-year growth in gross block. Directly measures the pace of physical asset accumulation — raw proxy for capacity expansion."},
    "Advance_from_Customers": {"label": "Customer Advances / Sales %",  "group": "Asset Detail", "desc": "Advance payments from customers / sales. High = strong pricing power (customers pay upfront). Creates negative working capital — companies like ITC, IRCTC have this."},
    "Minority_Interest_Ratio":{"label": "Minority Interest / Equity %", "group": "Asset Detail", "desc": "Non-controlling interests as % of total equity. High = significant subsidiary exposure; consolidated profits partially owned by others."},

    # ── Quarterly Dynamics ───────────────────────────────────────────────────────
    "Revenue_Seasonality":      {"label": "Revenue Seasonality CV %",    "group": "Quarterly", "desc": "Coefficient of variation of quarterly revenues (std/mean × 100). High = business is highly seasonal (IRCTC, festive-driven). Low = steady demand."},
    "OPM_Quarterly_Vol":        {"label": "Quarterly OPM Volatility",    "group": "Quarterly", "desc": "Standard deviation of quarterly operating margins over ~3 years. High = margins fluctuate significantly quarter to quarter."},
    "Qtr_Profit_Consistency":   {"label": "Quarterly Profit Consistency %","group": "Quarterly", "desc": "% of recent quarters with positive net profit. 100% = never had a loss quarter. Useful for screening cyclical vs defensive businesses."},
    "Recent_Revenue_Momentum":  {"label": "Recent Revenue Momentum %",   "group": "Quarterly", "desc": "Year-on-year revenue growth in the most recent quarter. Near-term momentum signal — captures current business velocity before annual data is available."},

    # ── Valuation ────────────────────────────────────────────────────────────────
    "PE_Ratio":       {"label": "P/E Ratio",               "group": "Valuation", "desc": "Market Cap / Net Profit. How much the market pays per rupee of earnings. Low P/E = cheaper (value). Negative for loss-making companies."},
    "PB_Ratio":       {"label": "P/B Ratio",               "group": "Valuation", "desc": "Market Cap / Book Value (Equity Capital + Reserves). Low P/B = trading near or below asset value. Asset-heavy sectors (banks, metals) typically have lower P/B than capital-light (IT, FMCG)."},
    "EV_EBITDA":      {"label": "EV / EBITDA",             "group": "Valuation", "desc": "Enterprise Value divided by (Operating Profit + Depreciation). Capital-structure-neutral valuation; preferred for comparing companies with different debt levels. Lower = cheaper."},
    "EV_Sales":       {"label": "EV / Sales",              "group": "Valuation", "desc": "Enterprise Value divided by Revenue. Useful when profits are thin or negative. Low = market assigns low value to each rupee of revenue (value opportunity or structural concern)."},
    "Mkt_Cap_Log":    {"label": "Log Market Cap",          "group": "Valuation", "desc": "Log10 of market capitalisation in crore. Pure size factor — log scale normalises the wide range (hundreds to lakhs of crore). Small-cap effect: historically smaller stocks outperform large-caps over long periods."},
    "Div_Yield":      {"label": "Dividend Yield %",        "group": "Valuation", "desc": "Annual dividend per share / price × 100. High yield = income return; may also signal value (cheap price) or distress. Computed as (Dividend Payout % x EPS) / Price."},
    "Price_to_CFO":   {"label": "Price / CFO",             "group": "Valuation", "desc": "Market Cap / Cash from Operations. Lower = company generates more operating cash relative to its price. More conservative than P/E because CFO is harder to manipulate than reported profit."},
    "Price_to_FCF":   {"label": "Price / FCF",             "group": "Valuation", "desc": "Market Cap / Free Cash Flow (CFO minus capex). Cleanest valuation metric — values only the cash left for shareholders after maintaining and growing the business. Low = attractive."},

    # ── Momentum ─────────────────────────────────────────────────────────────────
    "Mom_12m":        {"label": "Momentum 12m %",          "group": "Momentum", "desc": "12-month price return ending March 31 of this FY (Apr to Mar). Captures the full-year trend. Historically, high 12m momentum predicts further outperformance in the near term."},
    "Mom_6m":         {"label": "Momentum 6m %",           "group": "Momentum", "desc": "6-month price return: Oct 1 to Mar 31. Medium-term trend signal. Less noisy than 3m, more recent than 12m."},
    "Mom_3m":         {"label": "Momentum 3m %",           "group": "Momentum", "desc": "3-month price return: Jan 1 to Mar 31. Short-term trend. High sensitivity to recent catalysts (earnings, guidance). Can revert more quickly than longer windows."},
    "Mom_12m_1m":     {"label": "Momentum 12m-1m %",       "group": "Momentum", "desc": "12-month return skipping the last month (Apr to Feb). The classic Jegadeesh-Titman (1993) momentum definition. Skipping last month reduces short-term reversal noise, giving a purer intermediate-trend signal."},
    "Realized_Vol_90d":{"label": "Realized Volatility 90d %","group": "Momentum", "desc": "Annualised standard deviation of daily log-returns over the 90 days before March 31. Measures recent price riskiness. High vol = uncertain path; historically associated with lower risk-adjusted returns (low-volatility anomaly)."},
}

FACTOR_COLS  = list(FACTOR_META.keys())

GROUP_COLORS = {
    "Leverage":          "#f85149",
    "Asset Mix":         "#58a6ff",
    "Profitability":     "#f0b429",
    "Growth":            "#3fb950",
    "Cash Quality":      "#79c0ff",
    "Size":              "#d2a8ff",
    "Efficiency":        "#a5d6ff",
    "Capex":             "#ffa657",
    "Liquidity":         "#56d364",
    "Cost Structure":    "#ff9500",
    "Earnings Quality":  "#c3e88d",
    "CF Quality":        "#4fc3f7",
    "Capital Allocation":"#ce93d8",
    "Consistency":       "#80cbc4",
    "Asset Detail":      "#90caf9",
    "Quarterly":         "#ffcc02",
    "Valuation":         "#f97583",
    "Momentum":          "#b39ddb",
}


# ── Data loading ───────────────────────────────────────────────────────────────
def _parquet_hash() -> str:
    import hashlib, os
    size = os.path.getsize(_DB_PATH)
    with open(_DB_PATH, "rb") as f:
        h = hashlib.md5(f.read()).hexdigest()
    return f"{size}_{h}"

@st.cache_data(show_spinner="Loading data…")
def _load_all(file_hash: str) -> pd.DataFrame:
    return pd.read_parquet(_DB_PATH)

def load_data(sector: str) -> pd.DataFrame:
    df = _load_all(_parquet_hash())
    return df[df["Sector"] == sector].copy()


# ── Correlation helpers ────────────────────────────────────────────────────────
# NOTE: parameters starting with _ are NOT hashed by st.cache_data.
# The hash strings (df_hash / corr_hash) must NOT start with _ so Streamlit
# includes them in the cache key — otherwise every call hits the same cache entry.
@st.cache_data(show_spinner=False)
def compute_corr(df_hash: str, cols: tuple, method: str, _df: pd.DataFrame) -> pd.DataFrame:
    return _df[list(cols)].corr(method=method.lower())


@st.cache_data(show_spinner=False)
def corr_pvalues(df_hash: str, cols: tuple, _df: pd.DataFrame) -> pd.DataFrame:
    col_list = list(cols)
    pmat = pd.DataFrame(1.0, index=col_list, columns=col_list)
    for i, c1 in enumerate(col_list):
        for j, c2 in enumerate(col_list):
            if i >= j:
                continue
            sub = _df[[c1, c2]].dropna()
            if len(sub) > 3:
                _, p = stats.pearsonr(sub[c1], sub[c2])
                pmat.loc[c1, c2] = p
                pmat.loc[c2, c1] = p
    return pmat


@st.cache_data(show_spinner=False)
def cluster_order(corr_hash: str, _corr: pd.DataFrame) -> list[str]:
    dist = (1 - _corr.abs()).clip(0, 1).to_numpy(copy=True)
    np.fill_diagonal(dist, 0)
    try:
        Z     = linkage(squareform(dist), method="average")
        order = leaves_list(Z)
        return _corr.index[order].tolist()
    except Exception:
        return _corr.index.tolist()


def find_redundant_groups(corr: pd.DataFrame, threshold: float) -> dict[str, list[str]]:
    visited, groups = set(), {}
    for f in corr.index:
        if f in visited:
            continue
        dupes = [c for c in corr.columns
                 if c != f and abs(corr.loc[f, c]) >= threshold and c not in visited]
        if dupes:
            groups[f] = dupes
            visited.update(dupes)
        visited.add(f)
    return groups


@st.cache_data(show_spinner="Computing composites…")
def compute_composite_stats(sector: str, companies: tuple, yr_lo: int, yr_hi: int,
                             threshold: float, method: str, _file_hash: str):
    """
    Returns (retained_per_group, group_comp_stats, factor_icir_map).
    All heavy IC + composite work runs once and is cached per filter combination.
    """
    df_c = _load_all(_file_hash)
    df_c = df_c[
        (df_c["Sector"] == sector) &
        (df_c["Company"].isin(set(companies))) &
        (df_c["year"].between(yr_lo, yr_hi))
    ].copy()

    avail_c  = [f for f in FACTOR_COLS if f in df_c.columns and df_c[f].notna().sum() > 5]
    df_fwd_c = df_c[df_c["Return_1Y_Fwd"].notna()].copy()
    if df_fwd_c.empty or not avail_c:
        return {}, {}, {}

    # 1. Within-group deduplication using the redundancy threshold
    all_grps_c = list(dict.fromkeys(FACTOR_META[f]["group"] for f in avail_c if f in FACTOR_META))
    retained_per_group: dict[str, list[str]] = {}
    for grp in all_grps_c:
        grp_fs = [f for f in avail_c if FACTOR_META.get(f, {}).get("group") == grp]
        if len(grp_fs) < 2:
            retained_per_group[grp] = grp_fs
            continue
        try:
            sub = df_c[grp_fs].dropna(how="all")
            gc  = sub.corr(method=method.lower())
            gg  = find_redundant_groups(gc, threshold)
            gr  = {d for v in gg.values() for d in v}
            retained_per_group[grp] = [f for f in grp_fs if f not in gr]
        except Exception:
            retained_per_group[grp] = grp_fs

    # 2. IC-IR for every retained factor (year-by-year Spearman)
    all_retained   = [f for fs in retained_per_group.values() for f in fs]
    factor_icir_map: dict[str, float] = {}
    for f in all_retained:
        if f not in df_fwd_c.columns:
            continue
        ic_vals: list[float] = []
        for yr in sorted(df_fwd_c["year"].unique()):
            yr_df = df_fwd_c[df_fwd_c["year"] == yr][[f, "Return_1Y_Fwd"]].dropna()
            if len(yr_df) < 5 or yr_df[f].nunique() < 3:
                continue
            try:
                r, _ = stats.spearmanr(yr_df[f], yr_df["Return_1Y_Fwd"])
                if not np.isnan(r):
                    ic_vals.append(float(r))
            except Exception:
                pass
        if len(ic_vals) >= 2:
            mic = float(np.mean(ic_vals))
            sic = float(np.std(ic_vals, ddof=1))
            factor_icir_map[f] = mic / sic if sic > 0 else 0.0

    # 3. Build group composite scores + test IC
    group_comp_stats: dict[str, dict] = {}
    for grp, grp_fs in retained_per_group.items():
        usable  = [f for f in grp_fs if f in factor_icir_map]
        weights = {f: factor_icir_map[f] for f in usable}
        if not usable or all(abs(w) < 1e-9 for w in weights.values()):
            continue

        comp_ic_by_year: dict[int, float] = {}
        scores_by_year:  dict[int, dict]  = {}

        for yr in sorted(df_fwd_c["year"].unique()):
            yr_data = df_fwd_c[df_fwd_c["year"] == yr].copy()
            need    = [f for f in usable if f in yr_data.columns] + ["Return_1Y_Fwd", "Company"]
            yr_sub  = yr_data[need].dropna(subset=["Return_1Y_Fwd"])
            if len(yr_sub) < 5:
                continue

            composite = pd.Series(0.0, index=yr_sub.index)
            n_contrib = 0
            for f in usable:
                if f not in yr_sub.columns:
                    continue
                col  = yr_sub[f]
                std_ = col.std()
                if col.dropna().nunique() < 2 or std_ < 1e-10:
                    continue
                z = (col - col.mean()) / std_
                composite += weights[f] * z.fillna(0.0)
                n_contrib += 1

            if n_contrib == 0:
                continue
            mask = yr_sub["Return_1Y_Fwd"].notna() & composite.notna()
            cs, rs = composite[mask], yr_sub.loc[mask, "Return_1Y_Fwd"]
            if len(cs) < 5 or cs.nunique() < 3:
                continue
            try:
                r, _ = stats.spearmanr(cs, rs)
                if not np.isnan(r):
                    comp_ic_by_year[yr] = float(r)
                    scores_by_year[yr]  = dict(zip(yr_sub.loc[mask, "Company"], cs.values))
            except Exception:
                pass

        if len(comp_ic_by_year) < 2:
            continue

        ic_vals  = list(comp_ic_by_year.values())
        mic      = float(np.mean(ic_vals))
        sic      = float(np.std(ic_vals, ddof=1))
        n        = len(ic_vals)
        ic_ir    = mic / sic if sic > 0 else 0.0
        hit_rate = sum(1 for v in ic_vals if v > 0) / n * 100
        t_stat   = mic / (sic / np.sqrt(n)) if sic > 0 else np.nan
        try:
            p_val = float(2 * stats.t.sf(abs(t_stat), df=n - 1))
        except Exception:
            p_val = float("nan")

        # R² = mean(IC²) × 100 — average fraction of return variance explained per year.
        # NOT mean(IC)² which collapses to near-zero when IC alternates sign year to year.
        r2_pct_g = round(float(np.mean([v ** 2 for v in ic_vals])) * 100, 2)
        _beta_by_yr: dict[int, float] = {}
        for _yr, _co_scores in scores_by_year.items():
            _yr_ret = df_fwd_c[df_fwd_c["year"] == _yr][["Company", "Return_1Y_Fwd"]].dropna()
            _aln    = _yr_ret[_yr_ret["Company"].isin(_co_scores)].copy()
            _aln["_sc"] = _aln["Company"].map(_co_scores)
            _aln = _aln.dropna(subset=["_sc"])
            if len(_aln) < 5:
                continue
            try:
                _slope = float(np.polyfit(_aln["_sc"].values, _aln["Return_1Y_Fwd"].values, 1)[0])
                if not np.isnan(_slope):
                    _beta_by_yr[_yr] = _slope
            except Exception:
                pass
        mean_beta_g = round(float(np.mean(list(_beta_by_yr.values()))), 4) if _beta_by_yr else float("nan")

        best_single_factor = max(usable, key=lambda f: abs(factor_icir_map.get(f, 0.0)), default=None)
        best_single        = abs(factor_icir_map.get(best_single_factor, 0.0)) if best_single_factor else 0.0
        # top 3 factors by |IC-IR| with labels
        top3 = sorted(usable, key=lambda f: abs(factor_icir_map.get(f, 0.0)), reverse=True)[:3]
        top3_list = [(FACTOR_META[f]["label"], round(factor_icir_map[f], 3)) for f in top3 if f in FACTOR_META]

        group_comp_stats[grp] = {
            "n_factors":           len(usable),
            "mean_ic":             round(mic, 4),
            "ic_ir":               round(ic_ir, 3),
            "hit_rate":            round(hit_rate, 1),
            "p_val":               round(p_val, 4) if not np.isnan(p_val) else float("nan"),
            "n_years":             n,
            "best_single_icir":    round(best_single, 3),
            "best_single_factor":  FACTOR_META.get(best_single_factor, {}).get("label", best_single_factor or "—"),
            "top3_factors":        top3_list,
            "improvement":         round(abs(ic_ir) - best_single, 3),
            "ic_by_year":          comp_ic_by_year,
            "scores_by_year":      scores_by_year,
            "factor_icir":         {f: round(factor_icir_map[f], 3) for f in usable},
            "r2_pct":              r2_pct_g,
            "mean_beta":           mean_beta_g,
        }

    return retained_per_group, group_comp_stats, factor_icir_map


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ◈ Hypothesis V1")
    st.divider()

    sector = st.selectbox("Sector", list(SECTORS.keys()))

    df_raw = load_data(sector)
    if df_raw.empty:
        st.error("No data available for this sector.")
        st.stop()

    all_cos  = sorted(df_raw["Company"].unique())
    sel_cos  = st.multiselect("Companies", all_cos, default=all_cos)

    yrs = sorted(df_raw["year"].unique())
    if len(yrs) >= 2:
        yr_range = st.select_slider("Year Range", options=yrs, value=(yrs[0], yrs[-1]))
    else:
        yr_range = (yrs[0], yrs[-1])

    st.divider()
    method    = st.radio("Correlation Method", ["Pearson", "Spearman"], horizontal=True)
    threshold = st.slider("Redundancy  |r| ≥", 0.70, 0.99, 0.85, 0.05)

    st.divider()
    st.caption("Data: screener.in · Prices: Yahoo Finance")


# ── Filter ─────────────────────────────────────────────────────────────────────
df = df_raw[df_raw["Company"].isin(sel_cos) & df_raw["year"].between(*yr_range)].copy()
if df.empty:
    st.warning("No data for the current selection.")
    st.stop()

avail  = [f for f in FACTOR_COLS if f in df.columns and df[f].notna().sum() > 5]
df_fwd = df[df["Return_1Y_Fwd"].notna()].copy()

# Pre-compute composites (cached — runs once per sector/filter/threshold combo)
retained_per_group, group_comp_stats, factor_icir_map = compute_composite_stats(
    sector, tuple(sorted(sel_cos)),
    int(yr_range[0]), int(yr_range[1]),
    threshold, method, _parquet_hash()
)


# neutral Plotly defaults that work on both dark and light Streamlit themes
_PLY = dict(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#888888", size=10))
_GRID = "rgba(128,128,128,0.2)"
_LINE = "rgba(128,128,128,0.4)"


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown(f"## Factor Analysis · {sector}", unsafe_allow_html=True)

n_co  = df["Company"].nunique()
n_yr  = df["year"].nunique()
n_obs = len(df)
n_f   = len(avail)

st.markdown(f"""
<div class="kpi-wrap">
  <div class="kpi"><div class="num">{n_co}</div><div class="lbl">Companies</div></div>
  <div class="kpi"><div class="num">{n_yr}</div><div class="lbl">Fiscal Years</div></div>
  <div class="kpi"><div class="num">{n_obs}</div><div class="lbl">Observations</div></div>
  <div class="kpi"><div class="num">{n_f}</div><div class="lbl">Factors</div></div>
</div>
""", unsafe_allow_html=True)


# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["  Correlation Matrix  ", "  Factor Explorer  ", "  Company Snapshot  ", "  Factor Predictability  ", "  Alpha Composite  ", "  Quintile Backtest  "])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Correlation Matrix
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    all_groups = list(dict.fromkeys(FACTOR_META[f]["group"] for f in avail if f in FACTOR_META))
    _grp_ver   = "v3"   # bump this string whenever new groups are added to force a fresh default

    # ── Heatmap toggle + controls ─────────────────────────────────────────────
    hm_ctrl1, hm_ctrl2, hm_ctrl3 = st.columns([1, 5, 1])
    with hm_ctrl1:
        show_heatmap = st.toggle("Show Heatmap", value=False, key="hm_show",
                                  help="The heatmap is heavy — toggle on only when you need it. Everything else on this tab works without it.")
    with hm_ctrl2:
        sel_groups_hm = st.multiselect(
            "hm_label", all_groups, default=all_groups, key=f"hm_groups_{_grp_ver}",
            label_visibility="collapsed", disabled=not show_heatmap,
        )
    with hm_ctrl3:
        inc_ret = st.checkbox(
            "Include Return", value=False, key="hm_inc_ret", disabled=not show_heatmap,
            help="Append 1-year forward return to the heatmap."
        )

    if show_heatmap:
        hm_factors = [f for f in avail if FACTOR_META.get(f, {}).get("group") in sel_groups_hm]
        if inc_ret and "Return_1Y_Fwd" in df.columns and df["Return_1Y_Fwd"].notna().sum() > 5:
            hm_factors = hm_factors + ["Return_1Y_Fwd"]

        if len(hm_factors) < 2:
            st.info("Select at least 2 factor groups.")
        else:
            def _label(f):
                if f == "Return_1Y_Fwd": return "Fwd Return 1Y"
                return FACTOR_META[f]["label"]

            _hm_key  = f"{_parquet_hash()}_{tuple(hm_factors)}_{method}"
            corr     = compute_corr(_hm_key, tuple(hm_factors), method, df)
            pval     = corr_pvalues(_hm_key, tuple(hm_factors), df)
            order    = cluster_order(str(corr.values.tobytes()), corr)
            corr_o   = corr.loc[order, order]

            z    = corr_o.values
            mask = np.triu(np.ones_like(z, dtype=bool), k=1)
            z_lo = np.where(mask, np.nan, z)
            labels = [_label(f) for f in order]
            annot  = []
            for i in range(len(order)):
                row = []
                for j in range(len(order)):
                    if mask[i, j]:   row.append("")
                    elif i == j:     row.append("1.00")
                    else:
                        v = z[i, j]
                        sig = "**" if abs(v) >= threshold else ("*" if abs(v) >= 0.6 else "")
                        row.append(f"{v:.2f}{sig}")
                annot.append(row)

            fig_hm = go.Figure(go.Heatmap(
                z=z_lo, x=labels, y=labels,
                text=annot, texttemplate="%{text}",
                textfont={"size": 8},
                colorscale="RdBu_r", zmid=0, zmin=-1, zmax=1,
                showscale=True,
                colorbar=dict(title=dict(text="r", font=dict(size=11)), tickfont=dict(size=10), len=0.55),
                hoverongaps=False,
                hovertemplate="<b>%{y}</b>  x  <b>%{x}</b><br>r = %{z:.3f}<extra></extra>",
            ))
            fig_hm.update_layout(**_PLY, height=530, margin=dict(l=0, r=0, t=10, b=0),
                xaxis=dict(tickangle=-40, tickfont=dict(size=8.5), showgrid=False),
                yaxis=dict(tickfont=dict(size=8.5), showgrid=False),
            )
            st.plotly_chart(fig_hm, use_container_width=True)
            st.caption(f"** = |r| >= {threshold:.2f} (redundant)   * = |r| >= 0.60   Method: {method}   n = {n_obs} observations")
    else:
        st.markdown(
            '<div style="padding:18px 20px;border-radius:8px;background:rgba(128,128,128,0.07);'
            'border:1px solid rgba(128,128,128,0.15);color:#888;font-size:0.82rem;">'
            'Heatmap is off — toggle it on above when you need it. '
            'The redundancy analysis below runs instantly without it.'
            '</div>',
            unsafe_allow_html=True,
        )
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-hd">Analysis — Factor Groups</div>', unsafe_allow_html=True)
    sel_groups_an = st.multiselect(
        "an_label", all_groups, default=all_groups, key=f"an_groups_{_grp_ver}",
        label_visibility="collapsed"
    )
    an_factors = [f for f in avail if FACTOR_META.get(f, {}).get("group") in sel_groups_an]
    if len(an_factors) < 2:
        st.info("Select at least 2 factor groups for the analysis.")
        st.stop()
    _an_key   = f"{_parquet_hash()}_{tuple(an_factors)}_{method}"
    an_corr   = compute_corr(_an_key, tuple(an_factors), method, df)
    an_order  = cluster_order(str(an_corr.values.tobytes()), an_corr)
    an_corr_o = an_corr.loc[an_order, an_order]
    groups        = find_redundant_groups(an_corr_o, threshold)
    redundant_set = {d for v in groups.values() for d in v}
    retained      = [f for f in an_order if f not in redundant_set]

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<div class="section-hd">Redundant Groups</div>', unsafe_allow_html=True)
        if groups:
            for rep, dupes in groups.items():
                rep_lbl  = FACTOR_META[rep]["label"]
                rep_grp  = FACTOR_META[rep]["group"]
                grp_col  = GROUP_COLORS.get(rep_grp, "#8b949e")
                dupe_str = "".join(
                    f'<span class="chip chip-r">{FACTOR_META[d]["label"]}</span>'
                    for d in dupes
                )
                r_vals = [f"{an_corr_o.loc[rep, d]:.2f}" for d in dupes]
                st.markdown(
                    f'<div class="insight-box">'
                    f'<p><b>{rep_lbl}</b> &nbsp;is redundant with</p>'
                    f'{dupe_str}'
                    '<p style="margin-top:6px;font-size:0.68rem;color:rgba(128,128,128,0.6)">'
                    f'r = {", ".join(r_vals)}</p></div>',
                    unsafe_allow_html=True,
                )
        else:
            st.info("No redundant pairs at this threshold.")

    with col_r:
        st.markdown('<div class="section-hd">Independent Signals to Retain</div>', unsafe_allow_html=True)
        groups_by_grp: dict[str, list[str]] = {}
        for f in retained:
            g = FACTOR_META[f]["group"]
            groups_by_grp.setdefault(g, []).append(f)

        for grp, factors in groups_by_grp.items():
            chips = "".join(
                f'<span class="chip chip-g">{FACTOR_META[f]["label"]}</span>'
                for f in factors
            )
            st.markdown(
                f'<div class="insight-box">'
                f'<p><b>{grp}</b></p>{chips}</div>',
                unsafe_allow_html=True,
            )
        st.markdown(
            f"<br><small style='color:rgba(128,128,128,0.5)'>{len(retained)} of {len(an_order)} factors retained</small>",
            unsafe_allow_html=True,
        )

    if show_heatmap and 'corr_o' in dir():
        def _label_exp(f):
            if f == "Return_1Y_Fwd": return "Fwd Return 1Y"
            return FACTOR_META.get(f, {}).get("label", f)
        st.download_button(
            "Export Correlation Matrix (CSV)",
            corr_o.rename(columns={f: _label_exp(f) for f in corr_o.columns},
                          index={f: _label_exp(f) for f in corr_o.index}).to_csv(),
            file_name=f"corr_{sector}_{method}.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Factor Explorer
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    col_fa, col_fb = st.columns([2, 3])
    with col_fa:
        f_choice = st.selectbox(
            "Factor",
            avail,
            format_func=lambda x: FACTOR_META[x]["label"],
            key="fe_factor",
        )
    with col_fb:
        show_avg = st.toggle("Show sector average", value=True, key="fe_avg")

    meta = FACTOR_META[f_choice]
    st.caption(f"**{meta['group']}** — {meta['desc']}")

    chart_df = df[["Company", "year", f_choice]].dropna()

    fig_line = px.line(
        chart_df, x="year", y=f_choice,
        color="Company", markers=True,
        labels={"year": "Fiscal Year", f_choice: meta["label"]},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )

    if show_avg:
        avg = chart_df.groupby("year")[f_choice].mean().reset_index()
        fig_line.add_scatter(
            x=avg["year"], y=avg[f_choice],
            mode="lines", name="Sector Avg",
            line=dict(color="#f0b429", width=2.5, dash="dot"),
        )

    fig_line.update_layout(**_PLY, height=360, showlegend=True, margin=dict(l=0, r=0, t=20, b=0),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        xaxis=dict(showgrid=False, tickformat="d"),
        yaxis=dict(showgrid=True, gridcolor=_GRID, zeroline=False),
        hovermode="closest",
    )
    st.plotly_chart(fig_line, use_container_width=True)

    fig_box = px.box(
        chart_df, x="Company", y=f_choice,
        color="Company",
        labels={f_choice: meta["label"]},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_box.update_layout(**_PLY, height=280, margin=dict(l=0, r=0, t=10, b=0),
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor=_GRID, zeroline=False),
    )
    st.plotly_chart(fig_box, use_container_width=True)

    st.markdown('<div class="section-hd">Summary Statistics by Company</div>', unsafe_allow_html=True)
    stats_tbl = (
        df.groupby("Company")[f_choice]
        .agg(Mean="mean", Std="std", Min="min", Max="max", Years="count")
        .sort_values("Mean", ascending=False)
        .round(3)
    )
    st.dataframe(stats_tbl, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — Company Snapshot
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    col_ca, col_cb = st.columns([2, 2])
    with col_ca:
        snap_co = st.selectbox("Company", sorted(df["Company"].unique()), key="snap_co")
    with col_cb:
        _co_yrs = sorted(df[df["Company"] == snap_co]["year"].unique(), reverse=True)
        snap_yr = st.selectbox("Year", _co_yrs, key="snap_yr")

    all_groups_snap = list(dict.fromkeys(FACTOR_META[f]["group"] for f in avail if f in FACTOR_META))
    sel_groups_snap = st.multiselect(
        "Factor Groups", all_groups_snap, default=all_groups_snap, key="snap_groups",
        help="Filter which factor groups to display"
    )
    snap_row = df[(df["Company"] == snap_co) & (df["year"] == snap_yr)]
    if snap_row.empty:
        st.warning(f"No data for {snap_co} in {snap_yr}.")
        st.stop()

    snap_avail = [
        f for f in avail
        if FACTOR_META.get(f, {}).get("group") in sel_groups_snap
        and pd.notna(snap_row[f].values[0])
    ]

    snap_vals = snap_row[snap_avail].iloc[0]
    peer_df   = df[df["year"] == snap_yr][snap_avail]

    pct_ranks = {}
    for f in snap_avail:
        valid = peer_df[f].dropna()
        if len(valid) > 1 and pd.notna(snap_vals[f]):
            pct_ranks[f] = stats.percentileofscore(valid, snap_vals[f], kind="rank")
        else:
            pct_ranks[f] = np.nan

    bar_labels = [FACTOR_META[f]["label"] for f in snap_avail]
    bar_vals   = [pct_ranks.get(f, np.nan) for f in snap_avail]
    bar_colors = ["#3fb950" if (v or 0) >= 50 else "#f85149" for v in bar_vals]

    fig_snap = go.Figure(go.Bar(
        x=bar_labels, y=bar_vals,
        marker_color=bar_colors,
        hovertemplate="<b>%{x}</b><br>Percentile: %{y:.1f}<extra></extra>",
    ))
    fig_snap.add_hline(y=50, line_dash="dot", line_color=_LINE,
                       annotation_text="50th", annotation_font_color=_LINE)
    fig_snap.update_layout(**_PLY,
        title=f"{snap_co}  ·  FY{snap_yr}  ·  Percentile rank vs {sector} peers",
        height=360, margin=dict(l=0, r=0, t=50, b=0),
        xaxis=dict(tickangle=-35, tickfont=dict(size=8.5), showgrid=False),
        yaxis=dict(range=[0, 100], title="Percentile", showgrid=True, gridcolor=_GRID, zeroline=False),
        showlegend=False,
    )
    st.plotly_chart(fig_snap, use_container_width=True)

    st.markdown('<div class="section-hd">Factor Detail</div>', unsafe_allow_html=True)
    snap_tbl = pd.DataFrame({
        "Factor":       [FACTOR_META[f]["label"] for f in snap_avail],
        "Group":        [FACTOR_META[f]["group"]  for f in snap_avail],
        "Value":        [round(snap_vals[f], 4) if pd.notna(snap_vals[f]) else None for f in snap_avail],
        "Percentile":   [f"{pct_ranks[f]:.0f}th" if pd.notna(pct_ranks[f]) else None for f in snap_avail],
        "Description":  [FACTOR_META[f]["desc"]   for f in snap_avail],
    })
    st.dataframe(snap_tbl.set_index("Factor"), use_container_width=True)

    st.markdown('<div class="section-hd">All Factors — Historical</div>', unsafe_allow_html=True)
    hist = df[df["Company"] == snap_co][["year"] + snap_avail].set_index("year").T
    hist.index = [FACTOR_META[f]["label"] for f in hist.index]
    hist = hist.round(3)
    st.dataframe(hist, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Factor Predictability  (Cross-Sectional Rank IC)
# ══════════════════════════════════════════════════════════════════════════════
with tab4:

    pass  # Tab 4 CSS is now in the global themed block above

    # ── Factor group selector ─────────────────────────────────────────────────
    st.markdown('<div class="section-hd">Factor Groups to Analyse</div>', unsafe_allow_html=True)
    all_grps_p2 = list(dict.fromkeys(FACTOR_META[f]["group"] for f in avail if f in FACTOR_META))
    sel_grps_p2 = st.multiselect("p2g", all_grps_p2, default=all_grps_p2,
                                  label_visibility="collapsed", key="p2_groups_v3")
    p2_factors  = [f for f in avail if FACTOR_META.get(f, {}).get("group") in sel_grps_p2]

    # ── Filters + column toggles ──────────────────────────────────────────────
    r2a, r2b = st.columns([3, 2])
    with r2a:
        st.markdown('<div class="section-hd">Filters</div>', unsafe_allow_html=True)
        f1, f2, f3 = st.columns(3)
        with f1:
            min_ic   = st.slider("Min |Mean IC|", 0.00, 0.40, 0.00, 0.01, key="p2_min_ic",
                                  help="|Mean IC| across years. >0.05 = meaningful, >0.10 = strong.")
        with f2:
            max_pval = st.slider("Max p-value",   0.01, 0.20, 0.10, 0.01, key="p2_max_p",
                                  help="t-test on IC time series. p<0.05 = signal is real.")
        with f3:
            min_n    = st.slider("Min Years",      2,    12,   3,    1,    key="p2_min_n",
                                  help="Minimum years with valid cross-sectional data.")
    with r2b:
        st.markdown('<div class="section-hd">Table Columns</div>', unsafe_allow_html=True)
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            show_mean_ic  = st.checkbox("Mean IC",  value=True,  key="p2_show_ic")
            show_ic_ir    = st.checkbox("IC-IR",    value=True,  key="p2_show_icir")
            show_r2       = st.checkbox("R² (%)",   value=False, key="p2_show_r2")
        with tc2:
            show_hit_rate = st.checkbox("Hit Rate", value=True,  key="p2_show_hr")
            show_pval     = st.checkbox("p-value",  value=True,  key="p2_show_pval")
            show_beta     = st.checkbox("Beta",     value=False, key="p2_show_beta")
        with tc3:
            show_nyears   = st.checkbox("N Years",  value=True,  key="p2_show_n")
            show_range    = st.checkbox("IC Range", value=False, key="p2_show_range")

    st.divider()

    # ── Data check ────────────────────────────────────────────────────────────
    df_ic = df[df["Return_1Y_Fwd"].notna()].copy() if "Return_1Y_Fwd" in df.columns else pd.DataFrame()
    if df_ic.empty:
        st.warning("No forward return data available for this sector / year range.")
        st.stop()

    # ── Compute year-by-year cross-sectional Rank IC ──────────────────────────
    ic_rows = []
    for f in p2_factors:
        if f not in df_ic.columns or f not in FACTOR_META:
            continue
        ic_by_year   = {}
        beta_by_year = {}
        for yr in sorted(df_ic["year"].unique()):
            yr_df = df_ic[df_ic["year"] == yr][[f, "Return_1Y_Fwd"]].dropna()
            # Require ≥5 companies and ≥3 distinct factor values — below this IC/Beta are unreliable
            if len(yr_df) < 5 or yr_df[f].nunique() < 3:
                continue
            try:
                corr, _ = stats.spearmanr(yr_df[f], yr_df["Return_1Y_Fwd"])
                if not np.isnan(corr):
                    ic_by_year[yr] = float(corr)
            except Exception:
                pass
            try:
                _fv = yr_df[f].values.astype(float)
                _rv = yr_df["Return_1Y_Fwd"].values.astype(float)
                # Skip if factor has no meaningful spread (e.g. Material Cost for NBFCs — mostly 0)
                _fstd = float(np.std(_fv))
                _fmean = float(np.mean(np.abs(_fv)))
                if _fstd < 1e-6:
                    continue
                # Coefficient of variation check: skip when >90% of values are the same
                _cv = _fstd / (_fmean + 1e-10)
                if _cv < 0.05 and yr_df[f].nunique() < 5:
                    continue
                # Winsorize at 5th–95th percentile to kill single-outlier leverage on OLS
                _lo = float(np.percentile(_fv, 5))
                _hi = float(np.percentile(_fv, 95))
                if _lo < _hi:
                    _fv = np.clip(_fv, _lo, _hi)
                slope = float(np.polyfit(_fv, _rv, 1)[0])
                if not np.isnan(slope):
                    beta_by_year[yr] = slope
            except Exception:
                pass

        if len(ic_by_year) < 2:
            continue

        ic_vals   = list(ic_by_year.values())
        mean_ic   = float(np.mean(ic_vals))
        std_ic    = float(np.std(ic_vals, ddof=1))
        n_years   = len(ic_vals)
        ic_ir     = mean_ic / std_ic if std_ic > 0 else 0.0
        hit_rate  = sum(1 for v in ic_vals if v > 0) / n_years * 100
        t_stat    = mean_ic / (std_ic / np.sqrt(n_years)) if std_ic > 0 else np.nan
        # R² = mean(IC²) × 100 — average R² per year, not (mean IC)² which collapses for alternating-sign IC
        r2_pct    = round(float(np.mean([v ** 2 for v in ic_vals])) * 100, 2)
        mean_beta = round(float(np.mean(list(beta_by_year.values()))), 4) if beta_by_year else np.nan
        try:
            p_val = float(2 * stats.t.sf(abs(t_stat), df=n_years - 1))
        except Exception:
            p_val = np.nan

        ic_rows.append({
            "_key":        f,
            "Factor":      FACTOR_META[f]["label"],
            "Group":       FACTOR_META[f]["group"],
            "Mean IC":     round(mean_ic,  4),
            "IC-IR":       round(ic_ir,    3),
            "Hit Rate":    round(hit_rate, 1),
            "N Years":     n_years,
            "p-value":     round(p_val, 4) if not np.isnan(p_val) else np.nan,
            "R² (%)":      r2_pct,
            "Beta":        mean_beta,
            "Min IC":      round(min(ic_vals), 4),
            "Max IC":      round(max(ic_vals), 4),
            "_ic_by_year": ic_by_year,
        })

    ic_all = (pd.DataFrame(ic_rows).sort_values("Mean IC", key=abs, ascending=False)
              if ic_rows else pd.DataFrame())

    if ic_all.empty:
        st.info("No factors with enough cross-sectional data. Try expanding year range or factor groups.")
        st.stop()

    ic_filt = ic_all[
        (ic_all["Mean IC"].abs() >= min_ic)  &
        (ic_all["p-value"]       <= max_pval) &
        (ic_all["N Years"]       >= min_n)
    ].copy()

    # ── KPI strip ─────────────────────────────────────────────────────────────
    kc1, kc2, kc3, kc4, kc5 = st.columns(5)
    kc1.metric("Factors analysed",  len(ic_all))
    kc2.metric("Pass filters",      len(ic_filt))
    kc3.metric("Positive Mean IC",  int((ic_filt["Mean IC"] > 0).sum()))
    kc4.metric("IC-IR > 1.0",       int((ic_filt["IC-IR"].abs() > 1.0).sum()))
    kc5.metric("Hit Rate > 60%",    int((ic_filt["Hit Rate"] > 60).sum()))

    # ── Table ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-hd">Factor Rank IC Table — sorted by |Mean IC|</div>',
                unsafe_allow_html=True)

    if ic_filt.empty:
        st.info("No factors pass filters. Relax Min |Mean IC|, Max p-value, or Min Years.")
    else:
        dcols = ["Factor", "Group"]
        if show_mean_ic:  dcols.append("Mean IC")
        if show_ic_ir:    dcols.append("IC-IR")
        if show_hit_rate: dcols.append("Hit Rate")
        if show_pval:     dcols.append("p-value")
        if show_nyears:   dcols.append("N Years")
        if show_r2:       dcols.append("R² (%)")
        if show_beta:     dcols.append("Beta")
        if show_range:    dcols += ["Min IC", "Max IC"]

        def _s_ic(v):
            if not isinstance(v, (int, float)): return ""
            if v >  0.10: return "color:#3fb950;font-weight:700"
            if v >  0.05: return "color:#56d364"
            if v < -0.10: return "color:#f85149;font-weight:700"
            if v < -0.05: return "color:#ff7b72"
            return "color:#8b949e"

        def _s_icir(v):
            if not isinstance(v, (int, float)): return ""
            if abs(v) > 1.50: return "color:#3fb950;font-weight:700"
            if abs(v) > 0.75: return "color:#56d364"
            return "color:#8b949e"

        def _s_hr(v):
            if not isinstance(v, (int, float)): return ""
            if v >= 70: return "color:#3fb950;font-weight:700"
            if v >= 55: return "color:#56d364"
            if v <= 35: return "color:#f85149"
            return "color:#8b949e"

        def _s_pv(v):
            if not isinstance(v, (int, float)): return ""
            if v < 0.05: return "color:#3fb950"
            if v < 0.10: return "color:#f0b429"
            return "color:#8b949e"

        def _s_r2(v):
            if not isinstance(v, (int, float)): return ""
            if v >= 10: return "color:#3fb950;font-weight:700"
            if v >=  4: return "color:#56d364"
            if v >=  1: return "color:#8b949e"
            return "color:#484f58"

        def _s_beta(v):
            if not isinstance(v, (int, float)) or np.isnan(v): return ""
            if v >  5: return "color:#3fb950;font-weight:700"
            if v >  0: return "color:#56d364"
            if v < -5: return "color:#f85149;font-weight:700"
            return "color:#ff7b72"

        styled = ic_filt[dcols].set_index("Factor").style
        if show_mean_ic:  styled = styled.map(_s_ic,   subset=["Mean IC"])
        if show_ic_ir:    styled = styled.map(_s_icir, subset=["IC-IR"])
        if show_hit_rate: styled = styled.map(_s_hr,   subset=["Hit Rate"])
        if show_pval:     styled = styled.map(_s_pv,   subset=["p-value"])
        if show_r2:       styled = styled.map(_s_r2,   subset=["R² (%)"])
        if show_beta:     styled = styled.map(_s_beta, subset=["Beta"])
        st.dataframe(styled, use_container_width=True)

    st.divider()

    # ── Deep Dive ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-hd">Factor Deep Dive</div>', unsafe_allow_html=True)

    sel_label    = st.selectbox("Select factor", ic_all["Factor"].tolist(), key="p2_deep")
    sel_row      = ic_all[ic_all["Factor"] == sel_label].iloc[0]
    sel_key      = sel_row["_key"]
    dd_mean_ic   = sel_row["Mean IC"]
    dd_ic_ir     = sel_row["IC-IR"]
    dd_hr        = sel_row["Hit Rate"]
    dd_pval      = sel_row["p-value"]
    dd_n         = sel_row["N Years"]
    dd_min_ic    = sel_row["Min IC"]
    dd_max_ic    = sel_row["Max IC"]
    dd_r2        = sel_row["R² (%)"]
    dd_beta      = sel_row["Beta"]
    dd_ic_by_yr  = sel_row["_ic_by_year"]

    # significance
    if   dd_pval < 0.01: sig_str, sig_cls = "p < 0.01 · 99% confident", "chip-g"
    elif dd_pval < 0.05: sig_str, sig_cls = "p < 0.05 · 95% confident", "chip-g"
    elif dd_pval < 0.10: sig_str, sig_cls = "p < 0.10 · 90% confident", "chip-b"
    else:                sig_str, sig_cls = "Not significant",            "chip-r"

    # IC-IR quality label
    if   dd_ic_ir >  1.5: icir_cls, icir_q = "chip-g", "strong & consistent"
    elif dd_ic_ir >  0.5: icir_cls, icir_q = "chip-b", "moderate"
    elif dd_ic_ir > -0.5: icir_cls, icir_q = "chip-r", "weak"
    else:                  icir_cls, icir_q = "chip-r", "negative & inconsistent"

    # Hit rate quality label
    if   dd_hr >= 65: hr_cls, hr_q = "chip-g", "reliable"
    elif dd_hr >= 50: hr_cls, hr_q = "chip-b", "mixed"
    else:              hr_cls, hr_q = "chip-r", "unreliable"

    # Stat chips
    r2_cls   = "chip-g" if dd_r2 >= 10 else ("chip-b" if dd_r2 >= 4 else "chip-r")
    beta_str = f"{dd_beta:+.2f}% / unit" if not (isinstance(dd_beta, float) and np.isnan(dd_beta)) else "n/a"
    beta_cls = "chip-g" if (not (isinstance(dd_beta, float) and np.isnan(dd_beta))) and dd_beta > 0 else "chip-r"

    st.markdown(
        f'<div class="p2-stat-row">'
        f'<span class="chip {"chip-g" if dd_mean_ic > 0.05 else ("chip-b" if dd_mean_ic > 0 else "chip-r")}">Mean IC {dd_mean_ic:+.4f}</span>'
        f'<span class="chip {icir_cls}">IC-IR {dd_ic_ir:+.3f} · {icir_q}</span>'
        f'<span class="chip {hr_cls}">Hit Rate {dd_hr:.0f}% · {hr_q}</span>'
        f'<span class="chip chip-b">{dd_n} years</span>'
        f'<span class="chip {sig_cls}">{sig_str}</span>'
        f'<span class="chip {r2_cls}">R² {dd_r2:.1f}%</span>'
        f'<span class="chip {beta_cls}">β {beta_str}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # ── Three visual bars ─────────────────────────────────────────────────────
    # IC-IR bar (centred at 0, range -3 to +3)
    icir_clamped = max(-3.0, min(3.0, dd_ic_ir))
    icir_pct     = (icir_clamped + 3.0) / 6.0 * 100
    icir_col     = "#3fb950" if dd_ic_ir >= 0 else "#f85149"

    # Hit rate bar (0–100%)
    hr_col = "#3fb950" if dd_hr >= 65 else ("#f0b429" if dd_hr >= 50 else "#f85149")

    # IC range bar (maps [-1,+1] → [0%,100%])
    rng_left  = max(0.0, (dd_min_ic + 1) / 2 * 100)
    rng_width = max(0.0, min(100.0 - rng_left, (dd_max_ic - dd_min_ic) / 2 * 100))
    rng_mean  = max(0.0, min(100.0, (dd_mean_ic + 1) / 2 * 100))

    _td = "rgba(128,128,128,0.5)"
    _ln = "rgba(128,128,128,0.4)"
    st.markdown(f'''
<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:24px;margin-bottom:12px;">
  <div>
    <div style="font-size:0.65rem;color:{_td};text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px;">
      IC-IR (Factor Sharpe) &mdash; {dd_ic_ir:+.3f}
    </div>
    <div class="p2-mid-wrap">
      <div style="position:absolute;top:0;left:50%;width:1px;height:8px;background:{_ln};"></div>
      <div style="position:absolute;top:0;left:{min(50,icir_pct):.1f}%;width:{abs(icir_pct-50):.1f}%;height:8px;border-radius:3px;background:{icir_col};"></div>
    </div>
  </div>
  <div>
    <div style="font-size:0.65rem;color:{_td};text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px;">
      Hit Rate &mdash; {dd_hr:.0f}% of years positive
    </div>
    <div class="p2-bar-track">
      <div class="p2-bar-fill" style="width:{dd_hr:.1f}%;background:{hr_col};"></div>
    </div>
  </div>
  <div>
    <div style="font-size:0.65rem;color:{_td};text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px;">
      IC Range &mdash; {dd_min_ic:+.3f} to {dd_max_ic:+.3f}
    </div>
    <div class="p2-mid-wrap">
      <div style="position:absolute;top:0;left:50%;width:1px;height:8px;background:{_ln};"></div>
      <div style="position:absolute;top:2px;height:4px;border-radius:2px;background:#58a6ff;left:{rng_left:.1f}%;width:{rng_width:.1f}%;"></div>
      <div style="position:absolute;top:0;left:{rng_mean:.1f}%;width:2px;height:8px;background:#f0b429;border-radius:1px;"></div>
    </div>
  </div>
</div>''', unsafe_allow_html=True)

    # ── Year-by-year Rank IC chart ─────────────────────────────────────────────
    if dd_ic_by_yr:
        yy_df   = pd.DataFrame([{"Year": str(int(y)), "IC": round(v, 4)}
                                 for y, v in sorted(dd_ic_by_yr.items())])
        mean_ic = yy_df["IC"].mean()
        std_ic  = yy_df["IC"].std()
        pct_pos = (yy_df["IC"] > 0).mean() * 100

        bar_colors = []
        for v in yy_df["IC"]:
            if v >= 0.10:  bar_colors.append("#3fb950")
            elif v >= 0:   bar_colors.append("#56d364")
            elif v >= -0.10: bar_colors.append("#ff7b72")
            else:          bar_colors.append("#f85149")

        fig_yy = go.Figure()
        # ±1σ band
        fig_yy.add_hrect(
            y0=mean_ic - std_ic, y1=mean_ic + std_ic,
            fillcolor="rgba(240,180,41,0.07)", line_width=0,
            annotation_text="+/-1 SD band", annotation_position="top left",
            annotation_font=dict(color="rgba(240,180,41,0.5)", size=9),
        )
        fig_yy.add_trace(go.Bar(
            x=yy_df["Year"], y=yy_df["IC"],
            marker_color=bar_colors,
            text=[f"{v:+.3f}" for v in yy_df["IC"]],
            textposition="outside",
            textfont=dict(size=9, color="#888888"),
            hovertemplate=(
                "<b>FY%{x}</b><br>"
                "Rank IC = <b>%{y:.4f}</b><br>"
                "<span style='color:#aaa'>Positive IC = higher-ranked stocks beat lower-ranked</span>"
                "<extra></extra>"
            ),
            cliponaxis=False,
        ))
        fig_yy.add_hline(y=0,       line_color=_LINE,     line_dash="dot",   line_width=1)
        fig_yy.add_hline(y=mean_ic, line_color="#f0b429", line_dash="solid", line_width=2,
                         annotation_text=f"Mean = {mean_ic:+.4f}",
                         annotation_font_color="#f0b429", annotation_position="top right")
        # Threshold reference lines
        fig_yy.add_hline(y=0.10,  line_color="rgba(63,185,80,0.25)",  line_dash="dot", line_width=1)
        fig_yy.add_hline(y=-0.10, line_color="rgba(248,81,73,0.25)",  line_dash="dot", line_width=1)

        fig_yy.update_layout(**_PLY,
            title=dict(
                text=f"Year-by-Year Cross-Sectional Rank IC  |  Hit Rate {pct_pos:.0f}%  |  IC-IR {dd_ic_ir:+.3f}",
                font=dict(size=13),
            ),
            height=420,
            xaxis=dict(type="category", showgrid=False, tickfont=dict(size=11)),
            yaxis=dict(
                showgrid=True, gridcolor=_GRID, zeroline=False,
                tickformat=".2f", range=[min(-0.05, yy_df["IC"].min() * 1.35),
                                          max(0.05,  yy_df["IC"].max() * 1.35)],
            ),
            margin=dict(l=10, r=10, t=50, b=10),
        )
        st.plotly_chart(fig_yy, use_container_width=True)

    # ── Scatter: all years pooled — color-coded by year ────────────────────────
    scat = df_ic[[sel_key, "Return_1Y_Fwd", "Company", "year"]].dropna().copy()
    if len(scat) >= 5:
        try:
            _fv_sc = scat[sel_key].values.astype(float)
            _rv_sc = scat["Return_1Y_Fwd"].values.astype(float)
            # Winsorize pooled factor at 2nd–98th percentile before OLS to prevent
            # a single multi-year outlier from dominating the trend line
            _p2, _p98 = float(np.percentile(_fv_sc, 2)), float(np.percentile(_fv_sc, 98))
            if _p2 < _p98:
                _fv_sc = np.clip(_fv_sc, _p2, _p98)
            _sl, _ic_v, _, _, _ = _linregress(_fv_sc, _rv_sc)
            xlo, xhi = scat[sel_key].min(), scat[sel_key].max()  # draw across full visual range
            ols_ok = True
        except Exception:
            ols_ok = False

        # Colour points by year (spectral gradient)
        years_sorted = sorted(scat["year"].unique())
        yr_idx       = {yr: i for i, yr in enumerate(years_sorted)}
        n_yrs        = max(len(years_sorted) - 1, 1)
        def _yr_color(yr):
            t   = yr_idx[yr] / n_yrs
            r, g, b = colorsys.hsv_to_rgb(0.67 - 0.55 * t, 0.75, 0.90)
            return f"rgb({int(r*255)},{int(g*255)},{int(b*255)})"

        scat["_color"] = scat["year"].map(_yr_color)
        scat["_size"]  = 8

        # Flag outliers (|return| > 2σ or extreme factor value) for label
        ret_std = scat["Return_1Y_Fwd"].std()
        fac_std = scat[sel_key].std()
        scat["_outlier"] = (
            (scat["Return_1Y_Fwd"].abs() > 2 * ret_std) |
            (scat[sel_key].abs()          > 2 * fac_std)
        )

        fig_sc = go.Figure()

        # Render one trace per year for the legend
        for yr in years_sorted:
            yr_data = scat[scat["year"] == yr]
            col     = _yr_color(yr)
            fig_sc.add_trace(go.Scatter(
                x=yr_data[sel_key], y=yr_data["Return_1Y_Fwd"],
                mode="markers",
                name=f"FY{yr}",
                marker=dict(color=col, size=7, opacity=0.80,
                            line=dict(color="rgba(0,0,0,0.25)", width=0.5)),
                customdata=yr_data[["Company", "year"]].values,
                hovertemplate=(
                    "<b>%{customdata[0]}</b>  FY%{customdata[1]}<br>"
                    + sel_label + ": <b>%{x:.3f}</b><br>"
                    "Fwd Return: <b>%{y:.1f}%</b><extra></extra>"
                ),
            ))

        # Outlier text labels
        out = scat[scat["_outlier"]]
        if len(out):
            fig_sc.add_trace(go.Scatter(
                x=out[sel_key], y=out["Return_1Y_Fwd"],
                mode="text",
                text=out["Company"] + " '" + out["year"].astype(str).str[-2:],
                textfont=dict(size=8, color="rgba(160,160,160,0.8)"),
                textposition="top center",
                showlegend=False,
                hoverinfo="skip",
            ))

        if ols_ok:
            fig_sc.add_trace(go.Scatter(
                x=[xlo, xhi],
                y=[_ic_v + _sl * xlo, _ic_v + _sl * xhi],
                mode="lines",
                name="OLS trend",
                line=dict(color="#f0b429", width=2, dash="dash"),
                showlegend=True,
            ))

        fig_sc.add_hline(y=0, line_color=_LINE, line_dash="dot", line_width=1)
        fig_sc.update_layout(**_PLY,
            title=dict(
                text=f"{sel_label}  vs  Next-Year Return  -  all years pooled  (each colour = one year)",
                font=dict(size=12),
            ),
            height=500,
            xaxis=dict(title=sel_label, showgrid=True, gridcolor=_GRID, title_font=dict(size=11)),
            yaxis=dict(title="Forward Return %", showgrid=True, gridcolor=_GRID, zeroline=False),
            legend=dict(
                orientation="v", yanchor="middle", y=0.5, xanchor="left", x=1.01,
                font=dict(size=9), itemsizing="constant",
                bgcolor="rgba(0,0,0,0)", borderwidth=0,
            ),
            margin=dict(l=10, r=110, t=50, b=10),
        )
        st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # ── Interpretation box — verdict-first, scannable ─────────────────────────
    direction   = "positively" if dd_mean_ic > 0 else "negatively"
    strength    = "strongly" if abs(dd_mean_ic) > 0.15 else ("meaningfully" if abs(dd_mean_ic) > 0.07 else "weakly")
    ic_col      = "#3fb950" if dd_mean_ic > 0.05 else ("#f0b429" if dd_mean_ic > 0 else "#f85149")
    icir_col_h  = "#3fb950" if abs(dd_ic_ir) > 1.5 else ("#f0b429" if abs(dd_ic_ir) > 0.5 else "#f85149")
    hr_col_h    = "#3fb950" if dd_hr >= 65 else ("#f0b429" if dd_hr >= 50 else "#f85149")
    sig_col     = "#3fb950" if dd_pval < 0.05 else ("#f0b429" if dd_pval < 0.10 else "#f85149")
    r2_col      = "#3fb950" if dd_r2 >= 10 else ("#f0b429" if dd_r2 >= 4 else "#8b949e")

    # overall verdict
    is_usable = (abs(dd_mean_ic) >= 0.05 and dd_pval <= 0.10 and dd_hr >= 50)
    if abs(dd_mean_ic) > 0.12 and dd_ic_ir > 1.0 and dd_pval < 0.05:
        verdict_label, verdict_bg = "STRONG SIGNAL", "#1a4731"
        verdict_border, verdict_text = "#3fb950", "#3fb950"
    elif is_usable:
        verdict_label, verdict_bg = "USABLE SIGNAL", "#1a3a20"
        verdict_border, verdict_text = "#56d364", "#56d364"
    elif dd_pval > 0.10:
        verdict_label, verdict_bg = "NOT SIGNIFICANT", "#3d1f1f"
        verdict_border, verdict_text = "#f85149", "#f85149"
    else:
        verdict_label, verdict_bg = "WEAK / NOISY", "#2d2510"
        verdict_border, verdict_text = "#f0b429", "#f0b429"

    _ic_strength_label = "Strong (&gt;0.10)" if abs(dd_mean_ic) > 0.10 else ("Meaningful (0.05-0.10)" if abs(dd_mean_ic) > 0.05 else "Weak (&lt;0.05)")
    _interp_html = (
        f'<div class="p2-interp">'
        f'<div style="display:flex;align-items:center;gap:12px;margin-bottom:14px;padding-bottom:12px;border-bottom:1px solid rgba(128,128,128,0.2);">'
        f'<div style="padding:4px 14px;border-radius:4px;border:1px solid {verdict_border};background:{verdict_bg};font-size:0.72rem;font-weight:700;letter-spacing:.12em;color:{verdict_text};">{verdict_label}</div>'
        f'<div style="font-size:0.95rem;font-weight:500;"><b>{sel_label}</b> {strength} correlates <b style="color:{ic_col};">{direction}</b> with next-year returns in <b>{sector}</b></div>'
        f'</div>'
        f'<div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;margin-bottom:14px;">'
        f'<div style="background:rgba(128,128,128,0.07);border-radius:6px;padding:10px 12px;">'
        f'<div style="font-size:0.62rem;color:#888;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px;">Signal Strength</div>'
        f'<div style="font-size:1.1rem;font-weight:700;color:{ic_col};">IC {dd_mean_ic:+.4f}</div>'
        f'<div style="font-size:0.72rem;color:#888;margin-top:3px;">{_ic_strength_label} &nbsp; range {dd_min_ic:+.3f} to {dd_max_ic:+.3f}</div>'
        f'</div>'
        f'<div style="background:rgba(128,128,128,0.07);border-radius:6px;padding:10px 12px;">'
        f'<div style="font-size:0.62rem;color:#888;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px;">Consistency (IC-IR)</div>'
        f'<div style="font-size:1.1rem;font-weight:700;color:{icir_col_h};">IC-IR {dd_ic_ir:+.3f}</div>'
        f'<div style="font-size:0.72rem;color:#888;margin-top:3px;">{icir_q.title()} &nbsp; Hit Rate <span style="color:{hr_col_h};font-weight:600;">{dd_hr:.0f}%</span> of years positive</div>'
        f'</div>'
        f'<div style="background:rgba(128,128,128,0.07);border-radius:6px;padding:10px 12px;">'
        f'<div style="font-size:0.62rem;color:#888;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px;">Statistical Confidence</div>'
        f'<div style="font-size:1.1rem;font-weight:700;color:{sig_col};">{sig_str.split("·")[0].strip()}</div>'
        f'<div style="font-size:0.72rem;color:#888;margin-top:3px;">{dd_n} years of data &nbsp; R2 <span style="color:{r2_col};font-weight:600;">{dd_r2:.1f}%</span> return variance explained</div>'
        f'</div>'
        f'</div>'
        f'<div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;font-size:0.78rem;color:#aaa;line-height:1.6;">'
        f'<div><span style="color:#666;font-size:0.68rem;text-transform:uppercase;letter-spacing:.07em;">What IC means &nbsp;</span>'
        f'Each year: companies ranked by <i>{sel_label}</i>, then by next-year return. IC = Spearman rank correlation between those two rankings. Averaged across <b style="color:#ccc;">{dd_n} independent years</b>.</div>'
        f'<div><span style="color:#666;font-size:0.68rem;text-transform:uppercase;letter-spacing:.07em;">Economic size &nbsp;</span>'
        f'Beta <b style="color:#ccc;">{beta_str}</b> — for each 1-unit increase in this factor, next-year return shifts by that amount on average. IC-IR is the Sharpe ratio of this factor: <b style="color:#ccc;">&gt;1.5 = consistent, &lt;0.5 = noisy</b>.</div>'
        f'</div>'
        f'</div>'
    )
    st.markdown(_interp_html, unsafe_allow_html=True)

    st.caption(
        "Bars = one year's cross-sectional Rank IC (Spearman) across companies in this sector. "
        "Scatter is all years pooled — visual only; statistics derive from the bars. "
        "Green dashed thresholds on bar chart = IC ±0.10 reference lines."
    )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5 — Alpha Composite Score
# ══════════════════════════════════════════════════════════════════════════════
with tab5:

    pass  # Tab 5 CSS is in the global themed block

    if not group_comp_stats:
        st.warning("Not enough data to compute group composites. Try expanding the year range or selecting more companies.")
        st.stop()

    # ── KPI strip ─────────────────────────────────────────────────────────────
    n_grps    = len(group_comp_stats)
    n_imp     = sum(1 for g in group_comp_stats.values() if g["improvement"] > 0)
    best_g    = max(group_comp_stats.items(), key=lambda x: abs(x[1]["ic_ir"]))
    best_name, best_stat = best_g

    kc1, kc2, kc3, kc4, kc5 = st.columns(5)
    kc1.metric("Groups tested",          n_grps)
    kc2.metric("Composite > best single", f"{n_imp} / {n_grps}")
    kc3.metric("Best group",             best_name)
    kc4.metric("Best IC-IR",             f"{best_stat['ic_ir']:+.3f}")
    kc5.metric("Best hit rate",          f"{best_stat['hit_rate']:.0f}%")

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 1 — Group Composite IC Table
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="ac-hd">Section 1 — Group Composite IC (each group combined into one score)</div>',
                unsafe_allow_html=True)

    # ── Column toggles — single horizontal row ───────────────────────────────
    st.markdown('<div class="section-hd">Show / Hide Columns</div>', unsafe_allow_html=True)
    _tc = st.columns(8)
    with _tc[0]: t5_show_mic   = st.checkbox("Mean IC",     value=True, key="t5_show_mic")
    with _tc[1]: t5_show_icir  = st.checkbox("IC-IR",       value=True, key="t5_show_icir")
    with _tc[2]: t5_show_hr    = st.checkbox("Hit Rate",    value=True, key="t5_show_hr")
    with _tc[3]: t5_show_pval  = st.checkbox("p-value",     value=True, key="t5_show_pval")
    with _tc[4]: t5_show_r2    = st.checkbox("R² (%)",      value=True, key="t5_show_r2")
    with _tc[5]: t5_show_beta  = st.checkbox("Beta",        value=True, key="t5_show_beta")
    with _tc[6]: t5_show_nyrs  = st.checkbox("N Years",     value=True, key="t5_show_nyrs")
    with _tc[7]: t5_show_delta = st.checkbox("Δ vs Single", value=True, key="t5_show_delta")

    tbl_rows = []
    for grp, gs in sorted(group_comp_stats.items(), key=lambda x: abs(x[1]["ic_ir"]), reverse=True):
        tbl_rows.append({
            "Group":              grp,
            "Indep. Factors":     gs["n_factors"],
            "Composite IC-IR":    gs["ic_ir"],
            "Best Single IC-IR":  gs["best_single_icir"],
            "Δ vs Single":        gs["improvement"],
            "Hit Rate %":         gs["hit_rate"],
            "Mean IC":            gs["mean_ic"],
            "p-value":            gs["p_val"],
            "N Years":            gs["n_years"],
            "R² (%)":             gs.get("r2_pct", float("nan")),
            "Beta":               gs.get("mean_beta", float("nan")),
        })

    tdf_full = pd.DataFrame(tbl_rows).set_index("Group")

    # Build the displayed column list based on toggles
    _t5_dcols = ["Indep. Factors", "Composite IC-IR", "Best Single IC-IR"]
    if t5_show_delta: _t5_dcols.append("Δ vs Single")
    if t5_show_hr:    _t5_dcols.append("Hit Rate %")
    if t5_show_mic:   _t5_dcols.append("Mean IC")
    if t5_show_pval:  _t5_dcols.append("p-value")
    if t5_show_nyrs:  _t5_dcols.append("N Years")
    if t5_show_r2:    _t5_dcols.append("R² (%)")
    if t5_show_beta:  _t5_dcols.append("Beta")
    tdf = tdf_full[_t5_dcols]

    def _sc_icir(v):
        if not isinstance(v, (int, float)): return ""
        if abs(v) > 1.5: return "color:#3fb950;font-weight:700"
        if abs(v) > 0.75: return "color:#56d364"
        return "color:#8b949e"

    def _sc_delta(v):
        if not isinstance(v, (int, float)): return ""
        if v > 0.1:  return "color:#3fb950;font-weight:700"
        if v > 0:    return "color:#56d364"
        if v < -0.1: return "color:#f85149"
        return "color:#8b949e"

    def _sc_mic(v):
        if not isinstance(v, (int, float)): return ""
        if v >  0.10: return "color:#3fb950;font-weight:700"
        if v >  0.05: return "color:#56d364"
        if v < -0.05: return "color:#f85149"
        return "color:#8b949e"

    def _sc_pv(v):
        if not isinstance(v, (int, float)): return ""
        if v < 0.05: return "color:#3fb950"
        if v < 0.10: return "color:#f0b429"
        return "color:#8b949e"

    def _sc_r2(v):
        if not isinstance(v, (int, float)) or np.isnan(v): return ""
        if v >= 10: return "color:#3fb950;font-weight:700"
        if v >=  4: return "color:#56d364"
        if v >=  1: return "color:#8b949e"
        return "color:#484f58"

    def _sc_beta_t5(v):
        if not isinstance(v, (int, float)) or np.isnan(v): return ""
        if v >  5: return "color:#3fb950;font-weight:700"
        if v >  0: return "color:#56d364"
        if v < -5: return "color:#f85149;font-weight:700"
        return "color:#ff7b72"

    styled_tdf = (tdf.style
                  .map(_sc_icir,    subset=[c for c in ["Composite IC-IR", "Best Single IC-IR"] if c in tdf.columns])
                  .map(_sc_delta,   subset=[c for c in ["Δ vs Single"]  if c in tdf.columns])
                  .map(_sc_mic,     subset=[c for c in ["Mean IC"]      if c in tdf.columns])
                  .map(_sc_pv,      subset=[c for c in ["p-value"]      if c in tdf.columns])
                  .map(_sc_r2,      subset=[c for c in ["R² (%)"]       if c in tdf.columns])
                  .map(_sc_beta_t5, subset=[c for c in ["Beta"]         if c in tdf.columns]))
    st.dataframe(styled_tdf, use_container_width=True)

    # ── Composite vs best-single bar chart ────────────────────────────────────
    grps_sorted = [r["Group"] for r in tbl_rows]
    grp_colors  = [GROUP_COLORS.get(g, "#8b949e") for g in grps_sorted]

    # Build rich hover text for composite bar
    cmp_hover = []
    for g in grps_sorted:
        gs   = group_comp_stats[g]
        sign = 1 if gs["ic_ir"] >= 0 else -1
        top3_lines = "<br>".join(
            f"  {i+1}. {lbl} ({icir:+.3f})"
            for i, (lbl, icir) in enumerate(gs["top3_factors"])
        )
        imp_str = f"+{gs['improvement']:.3f}" if gs["improvement"] >= 0 else f"{gs['improvement']:.3f}"
        cmp_hover.append(
            f"<b>{g}</b> — Composite<br>"
            f"IC-IR: <b>{gs['ic_ir']:+.3f}</b>  |  Hit Rate: {gs['hit_rate']:.0f}%  |  Mean IC: {gs['mean_ic']:+.4f}<br>"
            f"Improvement over best single: <b>{imp_str}</b><br>"
            f"<br><b>Top factors in this group:</b><br>{top3_lines}"
            f"<extra></extra>"
        )

    # Build rich hover text for best-single bar
    single_hover = []
    for g in grps_sorted:
        gs = group_comp_stats[g]
        single_hover.append(
            f"<b>{g}</b> — Best Single Factor<br>"
            f"Factor: <b>{gs['best_single_factor']}</b><br>"
            f"IC-IR: <b>{gs['best_single_icir']:+.3f}</b><br>"
            f"<br><i>Use composite instead? {'Yes ✓' if gs['improvement'] > 0 else 'No — single factor wins'}</i>"
            f"<extra></extra>"
        )

    fig_cmp = go.Figure()
    # Threshold reference bands
    fig_cmp.add_hrect(y0=0.75, y1=1.5,  fillcolor="rgba(63,185,80,0.04)",  line_width=0)
    fig_cmp.add_hrect(y0=1.5,  y1=3.5,  fillcolor="rgba(63,185,80,0.08)",  line_width=0)
    fig_cmp.add_hline(y=0.75, line_color="rgba(63,185,80,0.3)",  line_dash="dot", line_width=1,
                      annotation_text="IC-IR 0.75", annotation_font=dict(color="rgba(63,185,80,0.5)", size=8),
                      annotation_position="top right")
    fig_cmp.add_hline(y=1.50, line_color="rgba(63,185,80,0.5)",  line_dash="dot", line_width=1,
                      annotation_text="IC-IR 1.50 (strong)", annotation_font=dict(color="rgba(63,185,80,0.6)", size=8),
                      annotation_position="top right")
    fig_cmp.add_trace(go.Bar(
        name="Composite IC-IR",
        x=grps_sorted,
        y=[group_comp_stats[g]["ic_ir"] for g in grps_sorted],
        marker_color=grp_colors,
        opacity=0.9,
        text=[f"{group_comp_stats[g]['ic_ir']:+.3f}" for g in grps_sorted],
        textposition="outside",
        textfont=dict(size=9, color="#cccccc"),
        hovertemplate=cmp_hover,
        cliponaxis=False,
    ))
    fig_cmp.add_trace(go.Bar(
        name="Best Single Factor",
        x=grps_sorted,
        y=[group_comp_stats[g]["best_single_icir"] * (1 if group_comp_stats[g]["ic_ir"] >= 0 else -1)
           for g in grps_sorted],
        marker_color="#484f58",
        opacity=0.75,
        text=[group_comp_stats[g]["best_single_factor"] for g in grps_sorted],
        textposition="outside",
        textfont=dict(size=8, color="#8b949e"),
        hovertemplate=single_hover,
        cliponaxis=False,
    ))
    fig_cmp.add_hline(y=0, line_color=_LINE, line_dash="dot", line_width=1)
    _cmp_max = max((abs(group_comp_stats[g]["ic_ir"]) for g in grps_sorted), default=1.0)
    fig_cmp.update_layout(**_PLY,
        title=dict(text="Composite IC-IR vs Best Single Factor — hover for factor names & top-3 breakdown", font=dict(size=13)),
        height=480, showlegend=True,
        barmode="group",
        margin=dict(l=0, r=10, t=50, b=90),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10), orientation="h", x=0.02, y=1.06),
        xaxis=dict(tickangle=-32, showgrid=False, tickfont=dict(size=10)),
        yaxis=dict(showgrid=True, gridcolor=_GRID, zeroline=False, tickformat=".2f",
                   range=[-0.1, _cmp_max * 1.30]),
        hoverlabel=dict(bgcolor="#1c2128", font_size=11, font_family="monospace"),
        bargap=0.25,
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

    # ── R² and Beta mini charts ────────────────────────────────────────────────
    st.markdown('<div class="ac-hd">Variance Explained (R²) and Economic Size (Beta) by Group</div>',
                unsafe_allow_html=True)

    _r2_grps   = sorted(group_comp_stats.keys(), key=lambda g: group_comp_stats[g].get("r2_pct", 0), reverse=True)
    _beta_grps = sorted(
        [g for g in group_comp_stats if not (isinstance(group_comp_stats[g].get("mean_beta"), float)
                                              and np.isnan(group_comp_stats[g]["mean_beta"]))],
        key=lambda g: abs(group_comp_stats[g].get("mean_beta", 0)), reverse=True
    )

    _ch1, _ch2 = st.columns(2)

    with _ch1:
        _r2_vals   = [group_comp_stats[g]["r2_pct"]   for g in _r2_grps]
        _r2_colors = [
            "#3fb950" if v >= 10 else ("#56d364" if v >= 4 else ("#f0b429" if v >= 1 else "#484f58"))
            for v in _r2_vals
        ]
        fig_r2 = go.Figure(go.Bar(
            x=_r2_vals, y=_r2_grps,
            orientation="h",
            marker_color=_r2_colors,
            text=[f"{v:.1f}%" for v in _r2_vals],
            textposition="outside",
            textfont=dict(size=9, color="#aaaaaa"),
            hovertemplate="<b>%{y}</b><br>R² = %{x:.2f}%<br>"
                          "<span style='color:#aaa;font-size:10px'>% of return variance the composite explains on average</span>"
                          "<extra></extra>",
            cliponaxis=False,
        ))
        fig_r2.add_vline(x=4,  line_color="rgba(86,211,100,0.3)", line_dash="dot", line_width=1)
        fig_r2.add_vline(x=10, line_color="rgba(63,185,80,0.5)",  line_dash="dot", line_width=1,
                         annotation_text="10% = strong", annotation_position="top right",
                         annotation_font=dict(color="rgba(63,185,80,0.5)", size=8))
        fig_r2.update_layout(**_PLY,
            title=dict(text="R² (%) — Return Variance Explained", font=dict(size=12)),
            height=max(260, len(_r2_grps) * 28 + 60),
            margin=dict(l=0, r=60, t=40, b=10),
            xaxis=dict(showgrid=True, gridcolor=_GRID, zeroline=False, ticksuffix="%"),
            yaxis=dict(autorange="reversed", showgrid=False, tickfont=dict(size=10)),
        )
        st.plotly_chart(fig_r2, use_container_width=True)
        st.caption(
            "R² = Mean IC² × 100. Measures what fraction of next-year return variance "
            "the composite factor score explains on average. ≥10% = strong, 4–10% = meaningful, <1% = negligible."
        )

    with _ch2:
        _beta_vals   = [group_comp_stats[g].get("mean_beta", float("nan")) for g in _beta_grps]
        _beta_colors = [
            "#3fb950" if v > 5 else ("#56d364" if v > 0 else ("#f85149" if v < -5 else "#ff7b72"))
            for v in _beta_vals
        ]
        fig_beta = go.Figure(go.Bar(
            x=_beta_vals, y=_beta_grps,
            orientation="h",
            marker_color=_beta_colors,
            text=[f"{v:+.2f}" for v in _beta_vals],
            textposition="outside",
            textfont=dict(size=9, color="#aaaaaa"),
            hovertemplate="<b>%{y}</b><br>Beta = %{x:+.3f}%/unit<br>"
                          "<span style='color:#aaa;font-size:10px'>Avg return shift per 1-unit move in composite score</span>"
                          "<extra></extra>",
            cliponaxis=False,
        ))
        fig_beta.add_vline(x=0, line_color=_LINE, line_dash="dot", line_width=1)
        fig_beta.update_layout(**_PLY,
            title=dict(text="Beta — Economic Size of Composite", font=dict(size=12)),
            height=max(260, len(_beta_grps) * 28 + 60),
            margin=dict(l=0, r=70, t=40, b=10),
            xaxis=dict(showgrid=True, gridcolor=_GRID, zeroline=False, ticksuffix="%"),
            yaxis=dict(autorange="reversed", showgrid=False, tickfont=dict(size=10)),
        )
        st.plotly_chart(fig_beta, use_container_width=True)
        st.caption(
            "Beta = average OLS slope of (composite score → next-year return) across years. "
            "A Beta of +5 means a 1-unit higher composite score is associated with +5% more return. "
            "Positive = factor scores in the right direction."
        )

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # SECTION 2 — Style Builder
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="ac-hd">Section 2 — Style Composite Builder (combine groups into one style score)</div>',
                unsafe_allow_html=True)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

    avail_grps = list(group_comp_stats.keys())
    default_quality = [g for g in ["Earnings Quality", "CF Quality", "Capital Allocation"] if g in avail_grps]
    if not default_quality:
        default_quality = avail_grps[:3]

    sel_style = st.multiselect(
        "Select groups to combine",
        avail_grps, default=default_quality, key="t5_style",
        help="Each group's composite score is weighted by its IC-IR and summed into one Style Score per company."
    )

    if len(sel_style) < 2:
        st.info("Select at least 2 groups to build a style composite.")
    else:
        # Combine group scores year-by-year (weighted by each group's IC-IR magnitude)
        style_ic_by_yr:     dict[int, float] = {}
        style_scores_by_yr: dict[int, dict]  = {}

        all_yrs = sorted(set(yr for g in sel_style for yr in group_comp_stats[g]["scores_by_year"]))
        for yr in all_yrs:
            yr_companies: dict[str, float] = {}
            total_weight = 0.0
            for grp in sel_style:
                if yr not in group_comp_stats[grp]["scores_by_year"]:
                    continue
                w    = abs(group_comp_stats[grp]["ic_ir"])
                sign = 1.0 if group_comp_stats[grp]["mean_ic"] >= 0 else -1.0
                for co, sc in group_comp_stats[grp]["scores_by_year"][yr].items():
                    yr_companies[co] = yr_companies.get(co, 0.0) + sign * w * sc
                total_weight += w

            if total_weight < 1e-9 or len(yr_companies) < 5:
                continue

            # Normalize by total weight
            yr_companies = {co: v / total_weight for co, v in yr_companies.items()}

            # Get returns for this year
            yr_ret = df_fwd[df_fwd["year"] == yr][["Company", "Return_1Y_Fwd"]].dropna()
            aligned = yr_ret[yr_ret["Company"].isin(yr_companies)].copy()
            aligned["style_score"] = aligned["Company"].map(yr_companies)
            aligned = aligned.dropna()
            if len(aligned) < 5 or aligned["style_score"].nunique() < 3:
                continue

            try:
                r, _ = stats.spearmanr(aligned["style_score"], aligned["Return_1Y_Fwd"])
                if not np.isnan(r):
                    style_ic_by_yr[yr]     = float(r)
                    style_scores_by_yr[yr] = dict(zip(aligned["Company"], aligned["style_score"]))
            except Exception:
                pass

        if len(style_ic_by_yr) < 2:
            st.info("Not enough overlapping years with data for all selected groups.")
        else:
            sic_vals  = list(style_ic_by_yr.values())
            s_mic     = float(np.mean(sic_vals))
            s_sic     = float(np.std(sic_vals, ddof=1))
            s_n       = len(sic_vals)
            s_icir    = s_mic / s_sic if s_sic > 0 else 0.0
            s_hr      = sum(1 for v in sic_vals if v > 0) / s_n * 100
            s_tstat   = s_mic / (s_sic / np.sqrt(s_n)) if s_sic > 0 else np.nan
            try:
                s_pval = float(2 * stats.t.sf(abs(s_tstat), df=s_n - 1))
            except Exception:
                s_pval = float("nan")

            # Style KPI strip
            sk1, sk2, sk3, sk4, sk5 = st.columns(5)
            sk1.metric("Style Mean IC",  f"{s_mic:+.4f}")
            sk2.metric("Style IC-IR",    f"{s_icir:+.3f}",
                       delta="strong" if abs(s_icir) > 1.5 else ("moderate" if abs(s_icir) > 0.75 else "weak"),
                       delta_color="normal")
            sk3.metric("Hit Rate",       f"{s_hr:.0f}%")
            sk4.metric("p-value",        f"{s_pval:.4f}" if not np.isnan(s_pval) else "—")
            sk5.metric("N Years",        s_n)

            # Year-by-year style IC chart
            yy_sdf = pd.DataFrame([{"Year": str(int(y)), "IC": round(v, 4)}
                                    for y, v in sorted(style_ic_by_yr.items())])
            st_colors     = ["#3fb950" if v >= 0 else "#f85149" for v in yy_sdf["IC"]]
            st_above_mean = [abs(v) > abs(s_mic) for v in yy_sdf["IC"]]
            st_opacity    = [0.95 if a else 0.6 for a in st_above_mean]

            fig_st = go.Figure()
            fig_st.add_trace(go.Bar(
                x=yy_sdf["Year"], y=yy_sdf["IC"],
                marker_color=st_colors,
                marker_opacity=st_opacity,
                marker_line_width=0,
                text=[f"{v:+.3f}" for v in yy_sdf["IC"]],
                textposition="outside",
                textfont=dict(size=9),
                customdata=[[
                    "above" if v >= 0 else "below",
                    "stronger" if abs(v) > abs(s_mic) else "weaker"
                ] for v in yy_sdf["IC"]],
                hovertemplate=(
                    "<b>FY %{x}</b><br>"
                    "Style IC = <b>%{y:+.4f}</b><br>"
                    "vs Mean (%{customdata[1]} than avg)<br>"
                    "<extra></extra>"
                ),
            ))
            fig_st.add_hline(y=0,     line_color=_LINE,    line_dash="dot",   line_width=1)
            fig_st.add_hline(y=s_mic, line_color="#f0b429", line_dash="solid", line_width=1.5,
                             annotation_text=f"Mean {s_mic:+.4f}",
                             annotation_font_color="#f0b429", annotation_position="top right",
                             annotation_font_size=10)
            # +1 std band
            if s_sic > 0:
                fig_st.add_hrect(y0=s_mic - s_sic, y1=s_mic + s_sic,
                                 fillcolor="rgba(240,180,41,0.06)", line_width=0,
                                 annotation_text="±1σ", annotation_position="top left",
                                 annotation_font_color="#f0b429", annotation_font_size=9)
            fig_st.update_layout(**_PLY,
                title=f"Style Composite — {' + '.join(sel_style)}  |  IC-IR {s_icir:+.3f}  |  Hit Rate {s_hr:.0f}%",
                height=360,
                margin=dict(l=0, r=0, t=45, b=0),
                xaxis=dict(type="category", showgrid=False, tickfont=dict(size=10)),
                yaxis=dict(showgrid=True, gridcolor=_GRID, zeroline=False,
                           tickformat=".3f", nticks=8, tickfont=dict(size=9)),
                bargap=0.25,
                hovermode="x unified",
            )
            st.plotly_chart(fig_st, use_container_width=True)

            # Interpretation
            s_dir = "positive" if s_mic > 0 else "negative"
            s_str = "strong" if abs(s_icir) > 1.5 else ("moderate" if abs(s_icir) > 0.75 else "weak")
            st.markdown(f'''<div class="ac-card">
              <p>The <b>{" + ".join(sel_style)}</b> style composite has a <b>{s_str}</b> {s_dir} signal
                 (IC-IR {s_icir:+.3f}) with a hit rate of <b>{s_hr:.0f}%</b> across {s_n} years in <b>{sector}</b>.</p>
              <p><b>How it works:</b> each company is scored on each group's composite, groups are weighted by their IC-IR,
                 then summed into one style score. The ranking of companies by this score is then tested against
                 next-year returns using Spearman rank correlation — each bar above is one independent year.</p>
            </div>''', unsafe_allow_html=True)

            st.divider()

            # ══════════════════════════════════════════════════════════════════
            # SECTION 3 — Company Leaderboard
            # ══════════════════════════════════════════════════════════════════
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
            st.markdown('<div class="ac-hd">Section 3 — Company Leaderboard (who scores highest right now?)</div>',
                        unsafe_allow_html=True)
            st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

            avail_lb_yrs = sorted(style_scores_by_yr.keys(), reverse=True)
            sel_lb_yr    = st.selectbox("Leaderboard year", avail_lb_yrs,
                                        format_func=lambda y: f"FY {int(y)}", key="t5_lb_yr")

            yr_sc = style_scores_by_yr[sel_lb_yr]
            lb_rows = []
            for rank, (co, sc) in enumerate(
                sorted(yr_sc.items(), key=lambda x: x[1], reverse=True), start=1
            ):
                row = {"Rank": rank, "Company": co, "Style Score": round(sc, 3)}
                for grp in sel_style:
                    gsc = group_comp_stats[grp]["scores_by_year"].get(sel_lb_yr, {}).get(co, np.nan)
                    row[grp] = round(gsc, 3) if not np.isnan(gsc) else np.nan
                lb_rows.append(row)

            lb_df = pd.DataFrame(lb_rows).set_index("Rank")

            # Style leaderboard table
            def _sc_style(v):
                if not isinstance(v, (int, float)) or np.isnan(v): return ""
                if v > 0.5:  return "color:#3fb950;font-weight:700"
                if v > 0:    return "color:#56d364"
                if v < -0.5: return "color:#f85149;font-weight:700"
                return "color:#ff7b72"

            lb_styled = lb_df.style.map(_sc_style, subset=["Style Score"] + sel_style)
            st.dataframe(lb_styled, use_container_width=True)

            # Horizontal bar chart — color intensity by score magnitude
            lb_sorted = lb_df.sort_values("Style Score")
            _sc_max = max(lb_sorted["Style Score"].abs().max(), 0.01)

            def _lb_color(v):
                t = min(abs(v) / _sc_max, 1.0)
                if v > 0:
                    g = int(100 + 85 * t)
                    return f"rgb(30,{g},50)"
                else:
                    r = int(120 + 88 * t)
                    return f"rgb({r},40,40)"

            lb_bar_colors = [_lb_color(v) for v in lb_sorted["Style Score"]]

            # Build richer hover: style score + each group score
            lb_hover = []
            for co in lb_sorted["Company"]:
                sc_val = yr_sc.get(co, 0)
                lines  = [f"<b>{co}</b>", f"Style Score: <b>{sc_val:+.3f}</b>", ""]
                for grp in sel_style:
                    gval = group_comp_stats[grp]["scores_by_year"].get(sel_lb_yr, {}).get(co, float("nan"))
                    if not np.isnan(gval):
                        bar  = "+" * min(int(abs(gval) * 4), 5) if gval > 0 else "-" * min(int(abs(gval) * 4), 5)
                        lines.append(f"{grp}: {gval:+.3f}  {bar}")
                lb_hover.append("<br>".join(lines) + "<extra></extra>")

            fig_lb = go.Figure()
            fig_lb.add_trace(go.Bar(
                y=lb_sorted["Company"],
                x=lb_sorted["Style Score"],
                orientation="h",
                marker_color=lb_bar_colors,
                marker_line_color="rgba(255,255,255,0.08)",
                marker_line_width=1,
                text=[f"{v:+.3f}" for v in lb_sorted["Style Score"]],
                textposition="outside",
                textfont=dict(size=9, color="#aaaaaa"),
                hovertemplate=lb_hover,
                cliponaxis=False,
            ))
            fig_lb.add_vline(x=0, line_color=_LINE, line_dash="dot", line_width=1)
            fig_lb.update_layout(**_PLY,
                title=dict(
                    text=f"{sector}  -  FY{int(sel_lb_yr)}  -  Style: {' + '.join(sel_style)}",
                    font=dict(size=12),
                ),
                height=max(300, len(lb_df) * 34 + 80),
                margin=dict(l=0, r=80, t=45, b=10),
                yaxis=dict(autorange="reversed", showgrid=False, tickfont=dict(size=10)),
                xaxis=dict(showgrid=True, gridcolor=_GRID, zeroline=False),
                hoverlabel=dict(bgcolor="#1c2128", font_size=11, font_family="monospace"),
            )
            st.plotly_chart(fig_lb, use_container_width=True)

            st.caption(
                f"Score = IC-IR-weighted sum of group composites. "
                f"Groups used: {', '.join(sel_style)}. "
                f"Redundancy threshold: |r| >= {threshold:.2f} — "
                f"only independent signals (after within-group deduplication) contribute. "
                f"Hover a bar to see each group's contribution."
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6 — Quintile Backtest
# ══════════════════════════════════════════════════════════════════════════════
with tab6:

    if not group_comp_stats:
        st.warning("Not enough data. Expand year range or select more companies.")
        st.stop()

    st.markdown("""
    <div class="ac-card">
      <p><b>What this does:</b> Every year, every company gets a style score based on the groups you select.
         Companies are split into 5 equal buckets — Q1 = top 20% (best-scored), Q5 = bottom 20% (worst-scored).
         Then we look at what actually happened to their returns. No predictions. Pure historical evidence.</p>
      <p><b>The conclusion:</b> if Q1 consistently beats Q5, the scoring system has a real edge and the top bucket is where you want to be.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    # ── Group selector ─────────────────────────────────────────────────────────
    st.markdown('<div class="ac-hd">Select Groups for the Style Composite</div>', unsafe_allow_html=True)

    _avail_t6 = list(group_comp_stats.keys())
    # Smart default: top groups that are statistically significant and positively signed
    _default_t6 = [
        g for g in sorted(_avail_t6,
                           key=lambda g: abs(group_comp_stats[g]["ic_ir"]), reverse=True)
        if group_comp_stats[g]["p_val"] < 0.10 and group_comp_stats[g]["ic_ir"] > 0
    ][:5]
    if not _default_t6:
        _default_t6 = _avail_t6[:3]

    sel_t6 = st.multiselect(
        "t6_grp_label",
        _avail_t6,
        default=_default_t6,
        key="t6_groups",
        label_visibility="collapsed",
        format_func=lambda g: (
            f"{g}   ·   IC-IR {group_comp_stats[g]['ic_ir']:+.2f}"
            f"   ·   p={group_comp_stats[g]['p_val']:.3f}"
            f"   ·   Hit {group_comp_stats[g]['hit_rate']:.0f}%"
        ),
    )
    if len(sel_t6) < 1:
        st.info("Select at least one group above.")
        st.stop()

    st.divider()

    # ── Build style scores year-by-year ────────────────────────────────────────
    _qt_scores: dict[int, dict] = {}
    for _yr in sorted(set(y for g in sel_t6 for y in group_comp_stats[g]["scores_by_year"])):
        _co_sc: dict[str, float] = {}
        _tw = 0.0
        for _g in sel_t6:
            if _yr not in group_comp_stats[_g]["scores_by_year"]:
                continue
            _w    = abs(group_comp_stats[_g]["ic_ir"])
            _sign = 1.0 if group_comp_stats[_g]["mean_ic"] >= 0 else -1.0
            for _co, _sc in group_comp_stats[_g]["scores_by_year"][_yr].items():
                _co_sc[_co] = _co_sc.get(_co, 0.0) + _sign * _w * _sc
            _tw += _w
        if _tw > 1e-9 and len(_co_sc) >= 5:
            _qt_scores[_yr] = {c: v / _tw for c, v in _co_sc.items()}

    if not _qt_scores:
        st.warning("Not enough overlapping data across selected groups.")
        st.stop()

    # ── Compute quintile returns ────────────────────────────────────────────────
    _qt_rows:    list[dict] = []
    _spread_yr:  dict[int, float] = {}
    _ew_yr:      dict[int, float] = {}
    _q1_cos_yr:  dict[int, list]  = {}

    for _yr, _scores in _qt_scores.items():
        _ret = df_fwd[df_fwd["year"] == _yr][["Company", "Return_1Y_Fwd"]].dropna()
        _aln = _ret[_ret["Company"].isin(_scores)].copy()
        _aln["score"] = _aln["Company"].map(_scores)
        _aln = _aln.dropna(subset=["score"])
        if len(_aln) < 5:
            continue

        _ew_yr[_yr] = _aln["Return_1Y_Fwd"].mean()

        try:
            _aln["Q"] = pd.qcut(_aln["score"].rank(method="first"), 5,
                                 labels=["Q5", "Q4", "Q3", "Q2", "Q1"])
        except Exception:
            continue

        _yr_q: dict[str, float] = {}
        for _q in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
            _sub = _aln[_aln["Q"] == _q]
            if _sub.empty:
                continue
            _avg = _sub["Return_1Y_Fwd"].mean()
            _yr_q[_q] = _avg
            _qt_rows.append({"year": _yr, "Quintile": _q,
                             "AvgReturn": _avg, "N": len(_sub),
                             "Companies": list(_sub["Company"])})
            if _q == "Q1":
                _q1_cos_yr[_yr] = list(_sub.sort_values("score", ascending=False)["Company"])

        if "Q1" in _yr_q and "Q5" in _yr_q:
            _spread_yr[_yr] = _yr_q["Q1"] - _yr_q["Q5"]

    if not _qt_rows or not _spread_yr:
        st.warning("Not enough data for quintile analysis. Try expanding the year range.")
        st.stop()

    _qt_df  = pd.DataFrame(_qt_rows)
    _qt_avg = (_qt_df.groupby("Quintile")["AvgReturn"]
               .agg(Mean="mean", Std="std", Count="count")
               .reindex(["Q1","Q2","Q3","Q4","Q5"])
               .round(2))

    _q1_avg      = _qt_avg.loc["Q1", "Mean"]
    _q5_avg      = _qt_avg.loc["Q5", "Mean"]
    _spread_avg  = _q1_avg - _q5_avg
    _n_yrs_t6    = len(_spread_yr)
    _win_rate    = sum(1 for v in _spread_yr.values() if v > 0) / _n_yrs_t6 * 100
    _sp_vals     = list(_spread_yr.values())
    _sp_icir     = (float(np.mean(_sp_vals)) / float(np.std(_sp_vals, ddof=1))
                    if len(_sp_vals) >= 2 and np.std(_sp_vals, ddof=1) > 0 else 0.0)
    _ew_avg      = float(np.mean(list(_ew_yr.values()))) if _ew_yr else 0.0

    # ── KPI strip ─────────────────────────────────────────────────────────────
    _k1, _k2, _k3, _k4, _k5, _k6 = st.columns(6)
    _k1.metric("Q1 Avg Return",  f"{_q1_avg:+.1f}%",
               delta=f"{_q1_avg - _ew_avg:+.1f}% vs sector avg")
    _k2.metric("Q5 Avg Return",  f"{_q5_avg:+.1f}%",
               delta=f"{_q5_avg - _ew_avg:+.1f}% vs sector avg", delta_color="inverse")
    _k3.metric("Q1 − Q5 Spread", f"{_spread_avg:+.1f}%/yr")
    _k4.metric("Win Rate",       f"{_win_rate:.0f}%",
               help="% of years Q1 beat Q5")
    _k5.metric("Spread IC-IR",   f"{_sp_icir:+.2f}",
               help="Mean spread / Std spread — consistency of the alpha")
    _k6.metric("Years Tested",   _n_yrs_t6)

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # CHART 1 — Average return by quintile
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="ac-hd">Chart 1 — Average Annual Return by Quintile</div>',
                unsafe_allow_html=True)

    _qc = {"Q1":"#3fb950","Q2":"#56d364","Q3":"#f0b429","Q4":"#ff7b72","Q5":"#f85149"}
    _ql = ["Q1 — Top 20%","Q2","Q3 — Middle","Q4","Q5 — Bottom 20%"]
    _qk = ["Q1","Q2","Q3","Q4","Q5"]
    _qv = [_qt_avg.loc[q, "Mean"] for q in _qk]

    fig_q1 = go.Figure()
    fig_q1.add_hline(y=_ew_avg, line_dash="dot", line_color="#f0b429", line_width=1.5,
                     annotation_text=f"Sector avg {_ew_avg:+.1f}%",
                     annotation_font=dict(color="#f0b429", size=10),
                     annotation_position="top right")
    fig_q1.add_hline(y=0, line_color=_LINE, line_dash="dot", line_width=1)
    fig_q1.add_trace(go.Bar(
        x=_ql, y=_qv,
        marker_color=[_qc[q] for q in _qk],
        text=[f"{v:+.1f}%" for v in _qv],
        textposition="outside", textfont=dict(size=14, color="#cccccc"),
        hovertemplate="<b>%{x}</b><br>Avg Return: <b>%{y:+.2f}%</b><extra></extra>",
        cliponaxis=False, width=0.5,
    ))
    _qmax = max(abs(max(_qv)), abs(min(_qv)), 5) * 1.4
    fig_q1.update_layout(**_PLY,
        height=400, margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(showgrid=False, tickfont=dict(size=12)),
        yaxis=dict(showgrid=True, gridcolor=_GRID, zeroline=False,
                   ticksuffix="%", range=[-_qmax, _qmax]),
        bargap=0.35,
    )
    st.plotly_chart(fig_q1, use_container_width=True)
    st.caption(
        f"Each bar = average across {_n_yrs_t6} independent years. "
        f"A clean staircase from Q1 down to Q5 is the proof the scoring system works. "
        f"Yellow line = equal-weight sector average."
    )

    # ══════════════════════════════════════════════════════════════════════════
    # CHART 2 — Year-by-year Q1 vs Q5 spread
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="ac-hd">Chart 2 — Q1 − Q5 Spread by Year (consistency check)</div>',
                unsafe_allow_html=True)

    _sp_df_rows = []
    for _y, _sp in sorted(_spread_yr.items()):
        _q1r = _qt_df[(_qt_df["year"]==_y)&(_qt_df["Quintile"]=="Q1")]["AvgReturn"]
        _q5r = _qt_df[(_qt_df["year"]==_y)&(_qt_df["Quintile"]=="Q5")]["AvgReturn"]
        _sp_df_rows.append({
            "Year": str(int(_y)), "Spread": round(_sp, 1),
            "Q1": round(_q1r.values[0], 1) if len(_q1r) else 0,
            "Q5": round(_q5r.values[0], 1) if len(_q5r) else 0,
        })
    _sp_df = pd.DataFrame(_sp_df_rows)

    fig_q2 = go.Figure()
    _sp_mean_v = float(np.mean(_sp_df["Spread"]))
    _sp_std_v  = float(np.std(_sp_df["Spread"]))
    if _sp_std_v > 0:
        fig_q2.add_hrect(y0=_sp_mean_v - _sp_std_v, y1=_sp_mean_v + _sp_std_v,
                         fillcolor="rgba(240,180,41,0.06)", line_width=0,
                         annotation_text="±1σ band", annotation_position="top left",
                         annotation_font=dict(color="rgba(240,180,41,0.4)", size=9))
    fig_q2.add_trace(go.Bar(
        x=_sp_df["Year"], y=_sp_df["Spread"],
        marker_color=["#3fb950" if v >= 0 else "#f85149" for v in _sp_df["Spread"]],
        text=[f"{v:+.1f}%" for v in _sp_df["Spread"]],
        textposition="outside", textfont=dict(size=9, color="#aaaaaa"),
        customdata=np.column_stack([_sp_df["Q1"], _sp_df["Q5"]]),
        hovertemplate=(
            "<b>FY%{x}</b><br>"
            "Spread: <b>%{y:+.1f}%</b><br>"
            "Q1 avg: %{customdata[0]:+.1f}%<br>"
            "Q5 avg: %{customdata[1]:+.1f}%<extra></extra>"
        ),
        cliponaxis=False,
    ))
    fig_q2.add_hline(y=0,           line_color=_LINE,     line_dash="dot",   line_width=1)
    fig_q2.add_hline(y=_sp_mean_v,  line_color="#f0b429", line_dash="solid", line_width=1.5,
                     annotation_text=f"Avg {_sp_mean_v:+.1f}%",
                     annotation_font=dict(color="#f0b429", size=10),
                     annotation_position="top right")
    fig_q2.update_layout(**_PLY,
        height=340, margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(type="category", showgrid=False, tickfont=dict(size=11)),
        yaxis=dict(showgrid=True, gridcolor=_GRID, zeroline=False, ticksuffix="%"),
        bargap=0.25,
    )
    st.plotly_chart(fig_q2, use_container_width=True)
    st.caption(
        f"Green = Q1 beat Q5 that year. Red = it didn't. "
        f"Win rate: {_win_rate:.0f}% of {_n_yrs_t6} years. "
        f"Spread IC-IR {_sp_icir:+.2f} — the Sharpe ratio of the alpha."
    )

    # ══════════════════════════════════════════════════════════════════════════
    # CHART 3 — Cumulative wealth
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown('<div class="ac-hd">Chart 3 — Cumulative Wealth · ₹100 at start</div>',
                unsafe_allow_html=True)

    _sorted_yrs = sorted(_spread_yr.keys())
    _c_q1 = [100.0]; _c_q5 = [100.0]; _c_ew = [100.0]
    for _y in _sorted_yrs:
        _r1 = _qt_df[(_qt_df["year"]==_y)&(_qt_df["Quintile"]=="Q1")]["AvgReturn"]
        _r5 = _qt_df[(_qt_df["year"]==_y)&(_qt_df["Quintile"]=="Q5")]["AvgReturn"]
        _c_q1.append(_c_q1[-1] * (1 + (_r1.values[0] if len(_r1) else 0) / 100))
        _c_q5.append(_c_q5[-1] * (1 + (_r5.values[0] if len(_r5) else 0) / 100))
        _c_ew.append(_c_ew[-1] * (1 + _ew_yr.get(_y, 0) / 100))

    _x_cum = ["Start"] + [str(int(y)) for y in _sorted_yrs]

    fig_q3 = go.Figure()
    fig_q3.add_trace(go.Scatter(
        x=_x_cum, y=_c_q1, mode="lines+markers", name="Q1 — Top 20%",
        line=dict(color="#3fb950", width=3),
        marker=dict(size=6, color="#3fb950"),
        hovertemplate="FY%{x}  Q1: ₹<b>%{y:.0f}</b><extra></extra>",
    ))
    fig_q3.add_trace(go.Scatter(
        x=_x_cum, y=_c_ew, mode="lines+markers", name="Sector Equal-Weight",
        line=dict(color="#f0b429", width=1.5, dash="dot"),
        marker=dict(size=5, color="#f0b429"),
        hovertemplate="FY%{x}  EW: ₹<b>%{y:.0f}</b><extra></extra>",
    ))
    fig_q3.add_trace(go.Scatter(
        x=_x_cum, y=_c_q5, mode="lines+markers", name="Q5 — Bottom 20%",
        line=dict(color="#f85149", width=3),
        marker=dict(size=6, color="#f85149"),
        hovertemplate="FY%{x}  Q5: ₹<b>%{y:.0f}</b><extra></extra>",
    ))
    fig_q3.add_hline(y=100, line_color=_LINE, line_dash="dot", line_width=1,
                     annotation_text="₹100 start", annotation_position="top left",
                     annotation_font=dict(color="#666", size=9))
    fig_q3.update_layout(**_PLY,
        height=420, margin=dict(l=10, r=10, t=20, b=10),
        xaxis=dict(type="category", showgrid=False, tickfont=dict(size=11)),
        yaxis=dict(showgrid=True, gridcolor=_GRID, zeroline=False,
                   tickprefix="₹", tickformat=".0f"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11),
                    orientation="h", x=0.01, y=1.06),
        hovermode="x unified",
    )
    st.plotly_chart(fig_q3, use_container_width=True)
    st.caption(
        f"Hypothetical annual rebalance — no transaction costs. "
        f"Q1 → ₹{_c_q1[-1]:.0f}  ·  Equal-weight → ₹{_c_ew[-1]:.0f}  ·  Q5 → ₹{_c_q5[-1]:.0f}. "
        f"The gap between Q1 and Q5 is the total edge of this scoring system over the period."
    )

    st.divider()

    # ══════════════════════════════════════════════════════════════════════════
    # VERDICT + THE ANSWER
    # ══════════════════════════════════════════════════════════════════════════
    if   _spread_avg > 20 and _win_rate >= 70 and _sp_icir > 1.0:
        _vl, _vbg, _vb, _vt = "STRONG EDGE",   "#1a4731", "#3fb950", "#3fb950"
        _vd = (f"Q1 beat Q5 by <b>{_spread_avg:+.1f}%/yr</b> on average, winning "
               f"<b>{_win_rate:.0f}%</b> of years with a Spread IC-IR of <b>{_sp_icir:+.2f}</b>. "
               "This scoring system has a statistically reliable edge in this sector. "
               "The top-quintile companies below are the data-backed recommendation.")
    elif _spread_avg > 8 and _win_rate >= 60:
        _vl, _vbg, _vb, _vt = "SOLID EDGE",    "#1a3a20", "#56d364", "#56d364"
        _vd = (f"Q1 beat Q5 by <b>{_spread_avg:+.1f}%/yr</b>, winning <b>{_win_rate:.0f}%</b> of years. "
               "Signal is meaningful but not dominant — use alongside other conviction signals. "
               "Top-quintile below has a higher historical base rate than the rest of the sector.")
    elif _spread_avg > 0 and _win_rate >= 50:
        _vl, _vbg, _vb, _vt = "WEAK EDGE",     "#2d2510", "#f0b429", "#f0b429"
        _vd = (f"The signal direction is right ({_spread_avg:+.1f}%/yr spread) but inconsistent "
               f"(only {_win_rate:.0f}% win rate). "
               "Add more groups, tighten IC-IR filters, or use a broader universe before acting on this.")
    else:
        _vl, _vbg, _vb, _vt = "NO EDGE",       "#3d1f1f", "#f85149", "#f85149"
        _vd = ("Q1 did not reliably beat Q5 in this sector with this group combination. "
               "The scoring system does not have a historical edge here. "
               "Try different groups or a different sector.")

    st.markdown(f"""
    <div style="border:1px solid {_vb};background:{_vbg};border-radius:8px;padding:18px 22px;margin-bottom:20px;">
      <div style="display:flex;align-items:center;gap:14px;margin-bottom:10px;">
        <div style="padding:5px 16px;border-radius:4px;border:1px solid {_vb};
                    font-size:0.75rem;font-weight:700;letter-spacing:.14em;color:{_vt};">{_vl}</div>
        <div style="font-size:0.88rem;color:#ccc;">
          Spread <b style="color:{_vt};">{_spread_avg:+.1f}%/yr</b> &nbsp;·&nbsp;
          Win Rate <b style="color:{_vt};">{_win_rate:.0f}%</b> &nbsp;·&nbsp;
          Spread IC-IR <b style="color:{_vt};">{_sp_icir:+.2f}</b> &nbsp;·&nbsp;
          {_n_yrs_t6} years tested
        </div>
      </div>
      <div style="font-size:0.82rem;color:#aaa;line-height:1.7;">{_vd}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── THE ANSWER: ranked company table ─────────────────────────────────────
    _ans_yr = max(_qt_scores.keys())
    st.markdown(
        f'<div class="ac-hd">The Answer — FY{int(_ans_yr)} Rankings  '
        f'(most recent year with validated return data)</div>',
        unsafe_allow_html=True
    )

    _ans_scores = _qt_scores[_ans_yr]
    _ans_df = pd.DataFrame([{"Company": c, "Score": round(s, 4)}
                             for c, s in _ans_scores.items()])
    _ans_df = _ans_df.sort_values("Score", ascending=False).reset_index(drop=True)
    _n_ans   = len(_ans_df)
    _q1_n    = max(1, _n_ans // 5)
    _q5_n    = max(1, _n_ans // 5)

    def _assign_label(i):
        if i < _q1_n:              return "Q1 — BUY"
        if i < _q1_n * 2:          return "Q2"
        if i < _n_ans - _q5_n * 2: return "Q3"
        if i < _n_ans - _q5_n:     return "Q4"
        return                             "Q5 — AVOID"

    _ans_df["Signal"] = [_assign_label(i) for i in range(_n_ans)]

    for _g in sel_t6:
        _g_sc = group_comp_stats[_g]["scores_by_year"].get(_ans_yr, {})
        _ans_df[_g[:18]] = _ans_df["Company"].map(_g_sc).round(3)

    _ans_df.index = range(1, len(_ans_df) + 1)
    _ans_df.index.name = "Rank"

    def _sc_sig(v):
        if v == "Q1 — BUY":   return "background-color:#0d2818;color:#3fb950;font-weight:700"
        if v == "Q5 — AVOID": return "background-color:#2a0d0d;color:#f85149;font-weight:700"
        if v == "Q2":          return "color:#56d364"
        if v == "Q4":          return "color:#ff7b72"
        return "color:#8b949e"

    st.dataframe(
        _ans_df.style.map(_sc_sig, subset=["Signal"]),
        use_container_width=True,
        height=min(680, (_n_ans + 1) * 36 + 40),
    )

    st.caption(
        f"FY{int(_ans_yr)} fundamentals, groups: {', '.join(sel_t6)}. "
        f"Q1 = top {_q1_n} companies ({100//5}% of {_n_ans} in sector). "
        f"Score = IC-IR-weighted composite. Higher = historically better return predictor. "
        f"This is a statistical ranking — not investment advice. Apply your own due diligence."
    )
