"""
Hypothesis V1 — Factor Analysis
Phase 1: Factor Correlation
"""

import os
import sys
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

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  [data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #21262d; }
  .block-container { padding: 1.2rem 2rem 2rem; }
  div[data-testid="stTabs"] button { font-size: 0.78rem; letter-spacing: 0.06em; }

  .kpi-wrap { display: flex; gap: 12px; margin-bottom: 1.4rem; }
  .kpi {
    background: #161b22; border: 1px solid #21262d; border-radius: 7px;
    padding: 12px 18px; flex: 1; min-width: 0;
  }
  .kpi .num { font-size: 1.75rem; font-weight: 700; color: #e6edf3; line-height: 1; }
  .kpi .lbl { font-size: 0.65rem; color: #6e7681; text-transform: uppercase;
              letter-spacing: 0.09em; margin-top: 4px; }

  .section-hd {
    font-size: 0.65rem; text-transform: uppercase; letter-spacing: 0.1em;
    color: #484f58; margin: 1.2rem 0 0.6rem;
  }
  .chip {
    display: inline-block; padding: 2px 9px; border-radius: 10px;
    font-size: 0.7rem; font-family: 'SF Mono', 'Fira Code', monospace; margin: 2px 3px;
  }
  .chip-r { background: #3d1a1a; border: 1px solid #da3633; color: #f85149; }
  .chip-g { background: #0d2818; border: 1px solid #2ea043; color: #3fb950; }
  .chip-b { background: #0c1e32; border: 1px solid #1f6feb; color: #58a6ff; }

  .insight-box {
    background: #161b22; border: 1px solid #21262d; border-radius: 7px;
    padding: 14px 16px; margin-top: 8px;
  }
  .insight-box p { margin: 0 0 4px; font-size: 0.78rem; color: #8b949e; line-height: 1.6; }
  .insight-box b { color: #e6edf3; }
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
}

FACTOR_COLS  = list(FACTOR_META.keys())
GROUP_COLORS = {
    "Leverage":     "#f85149",
    "Asset Mix":    "#58a6ff",
    "Profitability":"#f0b429",
    "Growth":       "#3fb950",
    "Cash Quality": "#79c0ff",
    "Size":         "#d2a8ff",
    "Efficiency":   "#a5d6ff",
    "Capex":        "#ffa657",
    "Liquidity":    "#56d364",
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
def compute_corr(df: pd.DataFrame, method: str) -> pd.DataFrame:
    return df.corr(method=method.lower())


def corr_pvalues(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.columns.tolist()
    pmat = pd.DataFrame(1.0, index=cols, columns=cols)
    for i, c1 in enumerate(cols):
        for j, c2 in enumerate(cols):
            if i >= j:
                continue
            sub = df[[c1, c2]].dropna()
            if len(sub) > 3:
                _, p = stats.pearsonr(sub[c1], sub[c2])
                pmat.loc[c1, c2] = p
                pmat.loc[c2, c1] = p
    return pmat


def cluster_order(corr: pd.DataFrame) -> list[str]:
    dist = (1 - corr.abs()).clip(0, 1).to_numpy(copy=True)
    np.fill_diagonal(dist, 0)
    try:
        Z     = linkage(squareform(dist), method="average")
        order = leaves_list(Z)
        return corr.index[order].tolist()
    except Exception:
        return corr.index.tolist()


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

avail = [f for f in FACTOR_COLS if f in df.columns and df[f].notna().sum() > 5]


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
tab1, tab2, tab3, tab4 = st.tabs(["  Correlation Matrix  ", "  Factor Explorer  ", "  Company Snapshot  ", "  Factor Predictability  "])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Correlation Matrix
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    all_groups = list(dict.fromkeys(FACTOR_META[f]["group"] for f in avail if f in FACTOR_META))
    st.markdown('<div class="section-hd">Heatmap — Factor Groups</div>', unsafe_allow_html=True)
    sel_groups_hm = st.multiselect(
        "hm_label", all_groups, default=all_groups, key="hm_groups",
        label_visibility="collapsed"
    )
    hm_col1, hm_col2 = st.columns([5, 1])
    with hm_col1:
        hm_factors = [f for f in avail if FACTOR_META.get(f, {}).get("group") in sel_groups_hm]
    with hm_col2:
        inc_ret = st.checkbox(
            "Include Return",
            value=False,
            key="hm_inc_ret",
            help="Append 1-year forward stock return to the heatmap so you can see which factors correlate with actual price performance."
        )

    if inc_ret and "Return_1Y_Fwd" in df.columns and df["Return_1Y_Fwd"].notna().sum() > 5:
        hm_factors = hm_factors + ["Return_1Y_Fwd"]

    if len(hm_factors) < 2:
        st.info("Select at least 2 factor groups.")
        st.stop()

    # Label lookup that handles Return_1Y_Fwd (not in FACTOR_META)
    def _label(f):
        if f == "Return_1Y_Fwd":
            return "Fwd Return 1Y"
        return FACTOR_META[f]["label"]

    corr   = compute_corr(df[hm_factors], method)
    pval   = corr_pvalues(df[hm_factors])
    order  = cluster_order(corr)
    corr_o = corr.loc[order, order]
    pval_o = pval.loc[order, order]

    z    = corr_o.values
    mask = np.triu(np.ones_like(z, dtype=bool), k=1)
    z_lo = np.where(mask, np.nan, z)

    labels = [_label(f) for f in order]
    annot  = []
    for i in range(len(order)):
        row = []
        for j in range(len(order)):
            if mask[i, j]:
                row.append("")
            elif i == j:
                row.append("1.00")
            else:
                v   = z[i, j]
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
        hovertemplate="<b>%{y}</b>  ×  <b>%{x}</b><br>r = %{z:.3f}<extra></extra>",
    ))
    fig_hm.update_layout(
        height=530,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8b949e", size=10),
        xaxis=dict(tickangle=-40, tickfont=dict(size=8.5), showgrid=False),
        yaxis=dict(tickfont=dict(size=8.5), showgrid=False),
    )
    st.plotly_chart(fig_hm, use_container_width=True)
    st.caption(f"** = |r| ≥ {threshold:.2f} (redundant)   * = |r| ≥ 0.60   Method: {method}   n = {n_obs} observations")

    st.divider()
    st.markdown('<div class="section-hd">Analysis — Factor Groups</div>', unsafe_allow_html=True)
    sel_groups_an = st.multiselect(
        "an_label", all_groups, default=all_groups, key="an_groups",
        label_visibility="collapsed"
    )
    an_factors = [f for f in avail if FACTOR_META.get(f, {}).get("group") in sel_groups_an]
    if len(an_factors) < 2:
        st.info("Select at least 2 factor groups for the analysis.")
        st.stop()
    an_corr   = compute_corr(df[an_factors], method)
    an_order  = cluster_order(an_corr)
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
                    f'<p style="margin-top:6px;font-size:0.68rem;color:#484f58">'
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
            f"<br><small style='color:#484f58'>{len(retained)} of {len(order)} factors retained</small>",
            unsafe_allow_html=True,
        )

    st.download_button(
        "↓ Export Correlation Matrix (CSV)",
        corr_o.rename(columns={f: _label(f) for f in corr_o.columns},
                      index={f: _label(f) for f in corr_o.index}).to_csv(),
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

    fig_line.update_layout(
        height=360,
        margin=dict(l=0, r=0, t=20, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8b949e"),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=10)),
        xaxis=dict(showgrid=False, tickformat="d"),
        yaxis=dict(showgrid=True, gridcolor="#21262d"),
        hovermode="closest",
    )
    st.plotly_chart(fig_line, use_container_width=True)

    fig_box = px.box(
        chart_df, x="Company", y=f_choice,
        color="Company",
        labels={f_choice: meta["label"]},
        color_discrete_sequence=px.colors.qualitative.Set2,
    )
    fig_box.update_layout(
        height=280,
        margin=dict(l=0, r=0, t=10, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8b949e", size=10),
        showlegend=False,
        xaxis=dict(showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#21262d"),
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
        snap_yr = st.selectbox("Year", sorted(df["year"].unique(), reverse=True), key="snap_yr")

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
    fig_snap.add_hline(y=50, line_dash="dot", line_color="#484f58",
                       annotation_text="50th", annotation_font_color="#484f58")
    fig_snap.update_layout(
        title=dict(
            text=f"{snap_co}  ·  FY{snap_yr}  ·  Percentile rank vs {sector} peers",
            font=dict(size=12, color="#8b949e"),
        ),
        height=360,
        margin=dict(l=0, r=0, t=50, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8b949e", size=10),
        xaxis=dict(tickangle=-35, tickfont=dict(size=8.5), showgrid=False),
        yaxis=dict(range=[0, 100], title="Percentile", showgrid=True, gridcolor="#21262d"),
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

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — Factor Predictability  (Phase 2)
# ══════════════════════════════════════════════════════════════════════════════
with tab4:

    st.markdown("""<style>
    .p2-bar-track{background:#21262d;border-radius:4px;height:8px;width:100%;margin:4px 0 10px;}
    .p2-bar-fill{height:8px;border-radius:4px;}
    .p2-beta-wrap{position:relative;height:8px;background:#21262d;border-radius:4px;margin:4px 0 10px;}
    .p2-interp{background:#161b22;border:1px solid #21262d;border-radius:8px;padding:16px 20px;margin-top:14px;}
    .p2-interp p{margin:0 0 8px;font-size:0.82rem;color:#8b949e;line-height:1.7;}
    .p2-interp b{color:#e6edf3;}
    .p2-stat-row{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:10px;}
    </style>""", unsafe_allow_html=True)

    # ── Row 1: Factor groups + stability toggle ───────────────────────────────
    r1a, r1b = st.columns([4, 1])
    with r1a:
        st.markdown('<div class="section-hd">Factor Groups to Analyse</div>', unsafe_allow_html=True)
        all_grps_p2 = list(dict.fromkeys(FACTOR_META[f]["group"] for f in avail if f in FACTOR_META))
        sel_grps_p2 = st.multiselect("p2g", all_grps_p2, default=all_grps_p2,
                                      label_visibility="collapsed", key="p2_groups")
    with r1b:
        st.markdown('<div class="section-hd">Stability Chart</div>', unsafe_allow_html=True)
        show_yy = st.checkbox("Year-by-Year IC", value=True, key="p2_show_yy")

    p2_factors = [f for f in avail if FACTOR_META.get(f, {}).get("group") in sel_grps_p2]

    # ── Row 2: Filters | Column toggles ──────────────────────────────────────
    r2a, r2b = st.columns([3, 2])
    with r2a:
        st.markdown('<div class="section-hd">Filters</div>', unsafe_allow_html=True)
        f1, f2, f3 = st.columns(3)
        with f1:
            min_ic   = st.slider("Min |IC|",    0.00, 0.30, 0.00, 0.01, key="p2_min_ic",
                                  help="|IC| = correlation strength. 0.10+ is meaningful.")
        with f2:
            max_pval = st.slider("Max p-value", 0.01, 0.20, 0.10, 0.01, key="p2_max_p",
                                  help="0.05 = 95% confidence the signal is real.")
        with f3:
            min_n    = st.slider("Min N",       10,   300,  30,   10,   key="p2_min_n",
                                  help="Min observations. Larger N = more trustworthy.")
    with r2b:
        st.markdown('<div class="section-hd">Table Columns</div>', unsafe_allow_html=True)
        tc1, tc2, tc3 = st.columns(3)
        with tc1:
            show_ic   = st.checkbox("IC",      value=True,  key="p2_show_ic")
            show_r2   = st.checkbox("R2",      value=True,  key="p2_show_r2")
        with tc2:
            show_beta = st.checkbox("Beta",    value=True,  key="p2_show_beta")
            show_pval = st.checkbox("P-Value", value=True,  key="p2_show_pval")
        with tc3:
            show_n    = st.checkbox("N",       value=True,  key="p2_show_n")

    st.divider()

    # ── Data & computation ────────────────────────────────────────────────────
    df_ic = df[df["Return_1Y_Fwd"].notna()].copy() if "Return_1Y_Fwd" in df.columns else pd.DataFrame()
    if df_ic.empty:
        st.warning("No forward return data available for this sector / year range.")
        st.stop()

    ic_rows = []
    for f in p2_factors:
        valid = df_ic[[f, "Return_1Y_Fwd"]].dropna()
        n = len(valid)
        if n < 5 or valid[f].nunique() < 2:
            continue
        try:
            slope, intercept, r, pval, _ = _linregress(valid[f], valid["Return_1Y_Fwd"])
        except Exception:
            continue
        ic_rows.append({
            "_key": f,
            "Factor":  FACTOR_META[f]["label"],
            "Group":   FACTOR_META[f]["group"],
            "IC":      round(r,       4),
            "R2":      round(r ** 2,  4),
            "Beta":    round(slope,   6),
            "P-Value": round(pval,    4),
            "N":       n,
            "_intercept": intercept,
        })

    ic_all  = pd.DataFrame(ic_rows).sort_values("IC", key=abs, ascending=False)
    ic_filt = ic_all[
        (ic_all["IC"].abs()  >= min_ic)   &
        (ic_all["P-Value"]   <= max_pval) &
        (ic_all["N"]         >= min_n)
    ].copy()

    # ── KPI strip ─────────────────────────────────────────────────────────────
    kc1, kc2, kc3, kc4, kc5 = st.columns(5)
    kc1.metric("Factors analysed",  len(ic_all))
    kc2.metric("Pass filters",      len(ic_filt))
    kc3.metric("Positive IC",       int((ic_filt["IC"] > 0).sum()))
    kc4.metric("Negative IC",       int((ic_filt["IC"] < 0).sum()))
    kc5.metric("Observations",      len(df_ic))

    # ── Table ─────────────────────────────────────────────────────────────────
    st.markdown('<div class="section-hd">Factor Predictability Table — sorted by |IC|</div>',
                unsafe_allow_html=True)
    if ic_filt.empty:
        st.info("No factors pass the current filters. Relax Min |IC|, Max p-value, or Min N.")
    else:
        dcols = ["Factor", "Group"]
        if show_ic:   dcols.append("IC")
        if show_r2:   dcols.append("R2")
        if show_beta: dcols.append("Beta")
        if show_pval: dcols.append("P-Value")
        if show_n:    dcols.append("N")

        def _ic_style(v):
            if not isinstance(v, float): return ""
            if v >  0.10: return "color:#3fb950;font-weight:600"
            if v < -0.10: return "color:#f85149;font-weight:600"
            return "color:#8b949e"

        styled = ic_filt[dcols].set_index("Factor").style
        if show_ic:
            styled = styled.map(_ic_style, subset=["IC"])
        st.dataframe(styled, use_container_width=True)

    st.divider()

    # ── Deep Dive ─────────────────────────────────────────────────────────────
    st.markdown('<div class="section-hd">Factor Deep Dive</div>', unsafe_allow_html=True)
    if ic_all.empty:
        st.info("No factors to analyse. Expand factor groups or year range.")
        st.stop()

    sel_label = st.selectbox("Select factor to deep dive", ic_all["Factor"].tolist(), key="p2_deep")
    sel_row   = ic_all[ic_all["Factor"] == sel_label].iloc[0]
    sel_key   = sel_row["_key"]
    dd_ic, dd_r2, dd_beta, dd_pval, dd_n, dd_int = (
        sel_row["IC"], sel_row["R2"], sel_row["Beta"],
        sel_row["P-Value"], sel_row["N"], sel_row["_intercept"]
    )

    if   dd_pval < 0.01: sig_str, sig_cls = "p < 0.01  (99% confidence)", "chip-g"
    elif dd_pval < 0.05: sig_str, sig_cls = "p < 0.05  (95% confidence)", "chip-g"
    elif dd_pval < 0.10: sig_str, sig_cls = "p < 0.10  (90% confidence)", "chip-b"
    else:                sig_str, sig_cls = "Not statistically significant", "chip-r"

    # Stat chips
    st.markdown(
        f'<div class="p2-stat-row">'
        f'<span class="chip chip-g">IC = {dd_ic:+.4f}</span>'
        f'<span class="chip chip-g">R2 = {dd_r2:.4f}</span>'
        f'<span class="chip chip-g">Beta = {dd_beta:+.6f}</span>'
        f'<span class="chip chip-b">N = {dd_n}</span>'
        f'<span class="chip {sig_cls}">{sig_str}</span>'
        f'</div>',
        unsafe_allow_html=True,
    )

    # R2 bar + Beta bar
    r2_pct   = min(dd_r2 * 100, 100)
    r2_col   = "#f0b429" if r2_pct < 10 else "#3fb950"
    beta_ref = max(ic_all["Beta"].abs().max(), 0.001)
    beta_pct = min(abs(dd_beta) / beta_ref * 50, 50)
    beta_col = "#3fb950" if dd_beta >= 0 else "#f85149"
    if dd_beta >= 0:
        b_style = f"position:absolute;top:0;left:50%;height:8px;border-radius:0 3px 3px 0;width:{beta_pct:.1f}%;background:{beta_col};"
    else:
        b_style = f"position:absolute;top:0;left:{50-beta_pct:.1f}%;height:8px;border-radius:3px 0 0 3px;width:{beta_pct:.1f}%;background:{beta_col};"

    st.markdown(
        f'''<div style="display:grid;grid-template-columns:1fr 1fr;gap:24px;margin-bottom:8px;">
      <div>
        <div style="font-size:0.65rem;color:#6e7681;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px;">
          R2 Variance Explained &mdash; {r2_pct:.1f}%
        </div>
        <div class="p2-bar-track">
          <div class="p2-bar-fill" style="width:{r2_pct:.1f}%;background:{r2_col};"></div>
        </div>
      </div>
      <div>
        <div style="font-size:0.65rem;color:#6e7681;text-transform:uppercase;letter-spacing:.08em;margin-bottom:4px;">
          Beta Direction &amp; Magnitude &mdash; {dd_beta:+.4f}
        </div>
        <div class="p2-beta-wrap">
          <div style="position:absolute;top:0;left:50%;width:1px;height:8px;background:#484f58;"></div>
          <div style="{b_style}"></div>
        </div>
      </div>
    </div>''',
        unsafe_allow_html=True,
    )

    # Scatter plot
    scat = df_ic[[sel_key, "Return_1Y_Fwd", "Company", "year"]].dropna()
    xmin, xmax = scat[sel_key].min(), scat[sel_key].max()
    fig_sc = go.Figure()
    fig_sc.add_trace(go.Scatter(
        x=scat[sel_key], y=scat["Return_1Y_Fwd"],
        mode="markers",
        marker=dict(color="#3fb950", size=6, opacity=0.65),
        customdata=scat[["Company", "year"]].values,
        hovertemplate=(
            "<b>%{customdata[0]}</b>  FY%{customdata[1]}<br>"
            + sel_label + ": %{x:.3f}<br>Fwd Return: %{y:.1f}%<extra></extra>"
        ),
    ))
    fig_sc.add_trace(go.Scatter(
        x=[xmin, xmax],
        y=[dd_int + dd_beta * xmin, dd_int + dd_beta * xmax],
        mode="lines", line=dict(color="#f0b429", width=2),
    ))
    fig_sc.add_hline(y=0, line_color="#484f58", line_dash="dot", line_width=1)
    fig_sc.update_layout(
        title=f"{sel_label}  vs  Next-Year Return  |  IC {dd_ic:+.4f}  |  R2 {dd_r2:.4f}",
        height=420, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#8b949e", size=10), showlegend=False,
        xaxis=dict(title=sel_label, showgrid=True, gridcolor="#21262d"),
        yaxis=dict(title="Forward Return (%)", showgrid=True, gridcolor="#21262d"),
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    # Interpretation box — below scatter
    direction = "positively" if dd_ic > 0 else "negatively"
    strength  = "strongly" if abs(dd_ic) > 0.20 else ("moderately" if abs(dd_ic) > 0.10 else "weakly")
    st.markdown(
        f'''<div class="p2-interp">
      <p><b>{sel_label}</b> {strength} correlates {direction} with next-year returns in <b>{sector}</b>.</p>
      <p><b>IC {dd_ic:+.4f}</b> &mdash; Direction and strength of factor-return relationship.
         Benchmark: |IC| &gt; 0.05 = meaningful, &gt; 0.10 = strong signal worth trading.</p>
      <p><b>R2 {dd_r2:.4f}</b> &mdash; This factor alone explains <b>{dd_r2*100:.1f}%</b> of
         next-year return variance. Single factors rarely exceed 10-15% in practice.</p>
      <p><b>Beta {dd_beta:+.6f}</b> &mdash; For every +1 unit in {sel_label},
         next-year return changes by <b>{dd_beta:+.4f}%</b> on average (OLS regression slope).</p>
      <p><b>p-value {dd_pval:.4f}</b> &mdash; {sig_str}. Based on <b>{dd_n}</b> company-year observations.</p>
    </div>''',
        unsafe_allow_html=True,
    )

    # Year-by-year IC stability
    if show_yy:
        st.markdown('<div class="section-hd">IC Stability Across Years</div>', unsafe_allow_html=True)
        yy_rows = []
        for yr in sorted(df_ic["year"].unique()):
            yd = df_ic[df_ic["year"] == yr][[sel_key, "Return_1Y_Fwd"]].dropna()
            if len(yd) >= 4 and yd[sel_key].nunique() >= 2:
                try:
                    _, _, r_y, _, _ = _linregress(yd[sel_key], yd["Return_1Y_Fwd"])
                    yy_rows.append({"Year": str(int(yr)), "IC": round(r_y, 4)})
                except Exception:
                    pass

        if yy_rows:
            yy_df   = pd.DataFrame(yy_rows)
            mean_ic = yy_df["IC"].mean()
            pct_pos = (yy_df["IC"] > 0).mean() * 100
            fig_yy  = go.Figure()
            fig_yy.add_trace(go.Bar(
                x=yy_df["Year"], y=yy_df["IC"],
                marker_color=["#3fb950" if v >= 0 else "#f85149" for v in yy_df["IC"]],
                hovertemplate="FY%{x}<br>IC = %{y:.4f}<extra></extra>",
            ))
            fig_yy.add_hline(y=0,       line_color="#484f58", line_dash="dot",   line_width=1)
            fig_yy.add_hline(y=mean_ic, line_color="#f0b429", line_dash="solid", line_width=2,
                             annotation_text=f"Mean IC = {mean_ic:+.4f}",
                             annotation_font_color="#f0b429",
                             annotation_position="top right")
            fig_yy.update_layout(
                title=f"Year-by-Year IC  |  Positive in {pct_pos:.0f}% of years  |  Mean = {mean_ic:+.4f}",
                height=300, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#8b949e", size=10), showlegend=False,
                xaxis=dict(type="category", showgrid=False),
                yaxis=dict(showgrid=True, gridcolor="#21262d", zeroline=False),
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_yy, use_container_width=True)

            if pct_pos >= 70 and abs(mean_ic) >= 0.05:
                verd, vcls = "Consistent signal — positive IC in most years.", "chip-g"
            elif pct_pos <= 35 or abs(mean_ic) < 0.02:
                verd, vcls = "Unreliable — IC flips sign. Not tradeable on its own.", "chip-r"
            else:
                verd, vcls = "Mixed — use alongside other factors.", "chip-b"
            st.markdown(f'<span class="chip {vcls}">{verd}</span><br><br>', unsafe_allow_html=True)

    st.caption(
        "IC = Pearson correlation (factor year T vs return year T+1). "
        "R2 = variance explained. Beta = OLS slope. Yellow line = mean IC across all years."
    )
