"""
Hypothesis V1 — Balance Sheet Factor Research
Phase 1: Factor Correlation Analysis
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
}

FACTOR_COLS  = list(FACTOR_META.keys())
GROUP_COLORS = {"Leverage": "#f85149", "Asset Mix": "#58a6ff", "Size": "#d2a8ff", "Growth": "#3fb950"}


# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data…")
def _load_all() -> pd.DataFrame:
    return pd.read_parquet(_DB_PATH)

def load_data(sector: str) -> pd.DataFrame:
    return _load_all()[_load_all()["Sector"] == sector].copy()


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
    st.caption("Balance Sheet · Phase 1 · Factor Research")
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
st.markdown(f"## Balance Sheet Factor Analysis &nbsp;·&nbsp; {sector}", unsafe_allow_html=True)

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
tab1, tab2, tab3 = st.tabs(["  Correlation Matrix  ", "  Factor Explorer  ", "  Company Snapshot  "])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Correlation Matrix
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    corr   = compute_corr(df[avail], method)
    pval   = corr_pvalues(df[avail])
    order  = cluster_order(corr)
    corr_o = corr.loc[order, order]
    pval_o = pval.loc[order, order]
    groups = find_redundant_groups(corr_o, threshold)
    redundant_set = {d for v in groups.values() for d in v}
    retained      = [f for f in order if f not in redundant_set]

    z    = corr_o.values
    mask = np.triu(np.ones_like(z, dtype=bool), k=1)
    z_lo = np.where(mask, np.nan, z)

    labels = [FACTOR_META[f]["label"] for f in order]
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

    # ── Analysis cards ─────────────────────────────────────────────────────────
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
                r_vals = [f"{corr_o.loc[rep, d]:.2f}" for d in dupes]
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

    # Download
    st.download_button(
        "↓ Export Correlation Matrix (CSV)",
        corr_o.rename(columns={f: FACTOR_META[f]["label"] for f in corr_o.columns},
                      index={f: FACTOR_META[f]["label"] for f in corr_o.index}).to_csv(),
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
        hovermode="x unified",
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # Box plot distribution
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

    # Stats table
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

    snap_row = df[(df["Company"] == snap_co) & (df["year"] == snap_yr)]
    if snap_row.empty:
        st.warning(f"No data for {snap_co} in {snap_yr}.")
        st.stop()

    snap_vals = snap_row[avail].iloc[0]
    peer_df   = df[df["year"] == snap_yr][avail]

    pct_ranks = {}
    for f in avail:
        valid = peer_df[f].dropna()
        if len(valid) > 1 and pd.notna(snap_vals[f]):
            pct_ranks[f] = stats.percentileofscore(valid, snap_vals[f], kind="rank")
        else:
            pct_ranks[f] = np.nan

    # Percentile bar chart
    bar_labels = [FACTOR_META[f]["label"] for f in avail]
    bar_vals   = [pct_ranks.get(f, np.nan) for f in avail]
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

    # Detail table
    st.markdown('<div class="section-hd">Factor Detail</div>', unsafe_allow_html=True)
    snap_tbl = pd.DataFrame({
        "Factor":       [FACTOR_META[f]["label"] for f in avail],
        "Group":        [FACTOR_META[f]["group"]  for f in avail],
        "Value":        [round(snap_vals[f], 4) if pd.notna(snap_vals[f]) else "—" for f in avail],
        "Percentile":   [f"{pct_ranks[f]:.0f}th" if pd.notna(pct_ranks[f]) else "—" for f in avail],
        "Description":  [FACTOR_META[f]["desc"]   for f in avail],
    })
    st.dataframe(snap_tbl.set_index("Factor"), use_container_width=True)

    # Multi-year trend for this company
    st.markdown('<div class="section-hd">All Factors — Historical</div>', unsafe_allow_html=True)
    hist = df[df["Company"] == snap_co][["year"] + avail].set_index("year").T
    hist.index = [FACTOR_META[f]["label"] for f in hist.index]
    hist = hist.round(3)
    st.dataframe(hist, use_container_width=True)
