"""Portfolio-first risk monitoring and AI investigation dashboard.

Install: python -m pip install streamlit plotly pandas
Run from repository root: python -m streamlit run app/investigation.py
Read-only: no live API calls or modification of risk data.
"""
from __future__ import annotations

import json
import re
from html import escape
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Risk Monitoring & Investigation", page_icon="📊",
                   layout="wide", initial_sidebar_state="expanded")
ROOT = Path.cwd()
SAMPLE = ROOT / "data" / "sample"
OUTPUTS = ROOT / "data" / "outputs"

st.markdown("""
<style>
:root {color-scheme:dark}
html, body, [data-testid="stAppViewContainer"], .stApp,
[data-testid="stHeader"] {background:#090d17 !important;color:#e9eefb}
[data-testid="stSidebar"] {background:#0d1320 !important;
 border-right:1px solid #24304a}
[data-testid="stSidebar"] * {color:#dce6f7}
[data-testid="stHeader"] {background:transparent !important}
.block-container {max-width:1530px;padding-top:1.45rem;padding-bottom:5rem}
h1,h2,h3 {letter-spacing:-.035em;color:#f3f7ff !important}
h1 {font-size:2rem !important} h2 {font-size:1.45rem !important}
div[data-testid="stVerticalBlockBorderWrapper"] > div {border-color:#25324a}
[data-testid="stMetric"] {background:#101827;border:1px solid #25324a;
 padding:15px 18px;border-radius:14px}
[data-testid="stMetricLabel"] {color:#9eaec8}
[data-testid="stMetricValue"] {color:#f2f6ff}
div[data-baseweb="select"] > div, input {background:#111a2a !important;
 color:#f4f7ff !important;border-color:#30405c !important}
[data-testid="stDataFrame"] {border:1px solid #283651;border-radius:12px}
.stTabs [data-baseweb="tab-list"] {gap:8px;border-bottom:1px solid #25324a}
.stTabs [data-baseweb="tab"] {color:#9eb0d0}
.stTabs [aria-selected="true"] {color:#7caaff !important}
.stButton button {border-radius:10px;border-color:#324362;
 background:#17243b;color:#edf3ff}
.stButton button:hover {border-color:#6599ff;color:#fff}

.risk-kpi {
 position:relative;overflow:hidden;
 background:linear-gradient(135deg,var(--kpi-bg,#152033),#111827 76%);
 border:1px solid color-mix(in srgb,var(--kpi-color,#608dff) 32%,#25324a);
 border-radius:18px;padding:20px 23px 18px;min-height:158px;
 margin-bottom:9px;box-shadow:0 8px 24px rgba(0,0,0,.11);
 transition:transform .18s ease,border-color .18s ease;
}
.risk-kpi:hover {transform:translateY(-2px);
 border-color:color-mix(in srgb,var(--kpi-color) 65%,#25324a)}
.risk-kpi::before {content:"";position:absolute;left:0;top:0;bottom:0;
 width:4px;background:var(--kpi-color,#608dff)}
.risk-kpi::after {content:"";position:absolute;right:-45px;top:-75px;
 width:185px;height:185px;border-radius:50%;
 background:radial-gradient(circle,color-mix(in srgb,var(--kpi-color) 14%,transparent),transparent 72%);
 pointer-events:none}
.risk-kpi-head {display:flex;align-items:center;justify-content:space-between;gap:10px}
.risk-kpi-label {font-size:.96rem;font-weight:620;color:#b7c7df}
.risk-kpi-icon {display:inline-flex;align-items:center;justify-content:center;
 width:34px;height:34px;border-radius:10px;font-size:1.1rem;
 background:color-mix(in srgb,var(--kpi-color) 15%,transparent);
 color:var(--kpi-color)}
.risk-kpi-value {font-size:2.18rem;font-weight:780;letter-spacing:-.045em;
 color:#f5f8ff;line-height:1.12;margin-top:15px}
.risk-kpi-value.loss {color:#ff6b80}
.risk-kpi-detail {font-size:.89rem;color:#9fb4d3;margin-top:9px}
.risk-kpi-footer {margin-top:12px;font-size:.77rem;color:#8399b9;
 display:flex;align-items:center;gap:6px}
.risk-kpi-dot {width:6px;height:6px;border-radius:50%;background:var(--kpi-color)}
.portfolio-hero {background:linear-gradient(115deg,#192a47 0%,#131e31 58%,#11192a 100%);
 border:1px solid #31466b;border-radius:18px;padding:20px 24px;
 min-height:96px;position:relative;overflow:hidden}
.portfolio-hero::after {content:"";position:absolute;width:200px;height:200px;
 right:-80px;top:-120px;border-radius:50%;
 background:radial-gradient(circle,rgba(83,142,255,.22),transparent 70%)}
.portfolio-hero-title {color:#f5f8ff;font-size:1.08rem;font-weight:760}
.portfolio-hero-sub {color:#a6bad8;font-size:.87rem;margin-top:7px}
.portfolio-hero-tag {display:inline-block;margin-top:12px;
 background:rgba(59,197,162,.12);border:1px solid rgba(59,197,162,.35);
 color:#6ce0bb;border-radius:999px;padding:4px 10px;font-size:.73rem;font-weight:700}
.snapshot-strip {display:flex;gap:11px;flex-wrap:wrap;margin:3px 0 20px}
.snapshot-chip {background:#111c2d;border:1px solid #293b58;border-radius:999px;
 color:#aebfda;padding:7px 13px;font-size:.81rem}
.snapshot-chip strong {color:#e9f1ff}
@media(max-width:800px){.risk-kpi{min-height:130px;padding:16px}
.risk-kpi-value{font-size:1.7rem}}
.panel {background:#111a2a;border:1px solid #273750;border-radius:16px;
 padding:18px 20px;margin:6px 0 13px}
.panel-title {font-size:1rem;font-weight:700;color:#f0f4ff;margin-bottom:5px}
.panel-sub {font-size:.86rem;color:#91a5c4}
.hero {background:linear-gradient(115deg,#172441,#11182a 72%);
 border:1px solid #2a3e63;border-radius:19px;padding:22px 26px;
 margin:6px 0 18px}
.hero-eyebrow {color:#80aafa;font-size:.78rem;font-weight:700;
 letter-spacing:.09em;text-transform:uppercase}
.hero-title {color:#f7f9ff;font-size:1.7rem;font-weight:750;
 letter-spacing:-.035em;margin:8px 0}
.hero-meta {color:#9fb1d1;font-size:.9rem}
.finding {background:linear-gradient(105deg,#301c2a,#151d2d);
 border:1px solid #704153;border-radius:15px;padding:17px 20px;
 margin:8px 0 16px}
.finding-head {font-weight:750;color:#ff9eaa;margin-bottom:7px}
.finding-body {color:#f1e7ec;line-height:1.6}
.small-tag {display:inline-block;padding:4px 10px;border-radius:999px;
 background:#32212c;border:1px solid #844453;color:#ff8f9e;
 font-size:.75rem;font-weight:750}
.section-kicker {color:#7ba6f8;font-weight:750;font-size:.78rem;
 text-transform:uppercase;letter-spacing:.11em;margin:16px 0 5px}
.info-line {color:#a1b1ce;font-size:.9rem}
@media(max-width:800px){.risk-kpi{min-height:115px;padding:14px}
.risk-kpi-value{font-size:1.55rem}.hero-title{font-size:1.4rem}}

.monitor-grid {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));
 gap:13px;margin:14px 0 22px}
.monitor-tile {background:#121d30;border:1px solid #2a3a56;border-radius:15px;
 padding:16px 18px;min-height:112px;border-top:3px solid var(--tone)}
.monitor-tile-label {color:#a9b9d4;font-size:.82rem;font-weight:700;
 letter-spacing:.045em}
.monitor-tile-value {color:var(--tone);font-size:2.1rem;font-weight:800;
 margin:9px 0 0;line-height:1.1}
.monitor-tile-foot {font-size:.77rem;color:#8397b7;margin-top:6px}
.threshold-grid {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));
 gap:1px;border:1px solid #2a3b59;border-radius:15px;overflow:hidden;
 background:#2a3b59;margin:13px 0 17px}
.threshold-tile {background:#141e30;padding:18px 18px;min-height:104px}
.threshold-value {font-size:1.9rem;font-weight:800;color:var(--tone)}
.threshold-label {font-size:.84rem;color:#a5b6d1;margin-top:6px}
.monitor-table-wrap {overflow-x:auto;border:1px solid #2c3b56;
 border-radius:14px;background:#111827;margin:12px 0 20px}
.monitor-table {width:100%;border-collapse:collapse;min-width:820px;
 font-size:.87rem}
.monitor-table th {background:#1b263a;color:#a7b9d6;font-weight:650;
 text-align:left;padding:14px 13px;border-bottom:1px solid #33435d}
.monitor-table td {padding:13px;border-bottom:1px solid #253148;
 color:#dce5f4;vertical-align:middle}
.monitor-table tr:last-child td {border-bottom:0}
.monitor-table tr:hover td {background:#1b2940}
.monitor-table tr.row-BREACH td:first-child {border-left:3px solid #ff657b}
.monitor-table tr.row-WARNING td:first-child {border-left:3px solid #ffbf69}
.monitor-table tr.row-PASS td:first-child {border-left:3px solid #39cba4}
.monitor-table tr.row-NOT_EVALUATED td:first-child {border-left:3px solid #8395b6}
.status-pill {display:inline-block;border-radius:999px;padding:4px 9px;
 font-size:.73rem;font-weight:800;background:color-mix(in srgb,var(--tone) 14%,transparent);
 color:var(--tone);border:1px solid color-mix(in srgb,var(--tone) 40%,transparent)}
.rule-sub {font-size:.75rem;color:#8095b5;margin-top:4px}
.limit-track {height:6px;background:#28374f;border-radius:8px;
 margin-top:7px;overflow:hidden;min-width:90px}
.limit-fill {height:100%;border-radius:8px;background:var(--tone)}
@media(max-width:800px){.monitor-grid,.threshold-grid{
 grid-template-columns:repeat(2,minmax(0,1fr))}
 .threshold-value{font-size:1.5rem}}

/* Exception cards: intentionally distinct from the policy table. */
.exception-eyebrow {font-size:.72rem;letter-spacing:.1em;text-transform:uppercase;
 color:#93a8ca;font-weight:750;margin-bottom:7px}
.exception-heading {font-size:1.15rem;color:#f3f6ff;font-weight:780;
 letter-spacing:-.02em;margin:2px 0 8px}
.exception-meta {color:#94a7c6;font-size:.8rem;line-height:1.7}
.exception-number {color:#f5f7ff;font-size:1.48rem;font-weight:800;
 letter-spacing:-.02em;margin-top:6px}
.exception-number-sub {font-size:.78rem;color:#9db0cc;margin-top:3px}
.exception-severity {display:inline-block;border:1px solid var(--severity);
 color:var(--severity);border-radius:999px;padding:5px 10px;
 background:color-mix(in srgb,var(--severity) 11%,transparent);
 font-size:.73rem;font-weight:800;letter-spacing:.02em}
.exception-state {color:#ff8192;font-size:.75rem;font-weight:800;
 letter-spacing:.08em;text-transform:uppercase;margin-bottom:5px}
.exception-legend {color:#a8bad4;font-size:.84rem;margin:-3px 0 17px}
/* Streamlit containers for each exception are the cards. */
[data-testid="stVerticalBlockBorderWrapper"]:has(.exception-heading) {
 background:linear-gradient(105deg,#1b2035 0%,#131d30 65%,#131b2b 100%);
 border:1px solid #384563 !important;border-left:4px solid #ff657c !important;
 border-radius:17px !important;margin:0 0 14px;
 box-shadow:0 7px 20px rgba(0,0,0,.12);
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.exception-heading):hover {
 border-color:#657ba6 !important;border-left-color:#ff657c !important;
}
[data-testid="stVerticalBlockBorderWrapper"]:has(.exception-heading) .stButton button {
 background:linear-gradient(90deg,#2861d8,#397ff6) !important;
 color:white !important;border:1px solid #619aff !important;
 font-weight:700;min-height:42px;
}

/* Refined navigation rail */
[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {gap:.65rem}
.sidebar-brand {padding:13px 2px 8px}
.sidebar-brand-icon {width:40px;height:40px;display:inline-flex;align-items:center;
 justify-content:center;border-radius:12px;font-size:22px;
 background:linear-gradient(135deg,#24457a,#413277);
 border:1px solid #4764a0;box-shadow:0 7px 22px #091124}
.sidebar-brand-title {color:#f5f8ff;font-size:1.12rem;font-weight:800;
 letter-spacing:-.03em;margin:12px 0 3px}
.sidebar-brand-sub {font-size:.77rem;color:#94a9ca;line-height:1.45}
.sidebar-label {font-size:.69rem;letter-spacing:.13em;font-weight:800;
 color:#799bd6;margin:12px 0 3px;text-transform:uppercase}
.sidebar-nav-item {display:flex;align-items:center;gap:11px;
 padding:11px 12px;border:1px solid #263853;border-radius:11px;
 background:#111d30;margin:6px 0}
.sidebar-nav-icon {font-size:1rem;color:#7ca9ff;width:21px;text-align:center}
.sidebar-nav-name {font-size:.86rem;font-weight:650;color:#dce7fb}
.sidebar-nav-sub {font-size:.7rem;color:#8ea4c5;margin-top:3px}
.sidebar-case {border:1px solid #2b3d59;border-left:3px solid #759cff;
 background:linear-gradient(110deg,#15243a,#101a2b);border-radius:11px;
 padding:12px 13px;margin:8px 0}
.sidebar-case-date {font-weight:800;color:#eaf2ff;font-size:.87rem}
.sidebar-case-title {font-size:.78rem;color:#adc0df;margin-top:4px}
.sidebar-case-meta {font-size:.7rem;color:#8198b8;margin-top:7px}
.sidebar-demo {background:rgba(245,182,90,.08);border:1px solid rgba(245,182,90,.32);
 border-radius:12px;padding:12px 13px;margin-top:10px}
.sidebar-demo-title {color:#ffcc82;font-size:.79rem;font-weight:800}
.sidebar-demo-text {color:#b7c5dc;font-size:.75rem;line-height:1.55;margin-top:6px}
.sidebar-footer {border-top:1px solid #2a3851;padding-top:13px;margin-top:14px;
 color:#8ea2bf;font-size:.73rem;line-height:1.7}
.sidebar-live-dot {display:inline-block;width:7px;height:7px;
 border-radius:50%;background:#43cba2;margin-right:7px}

.portfolio-intro {background:linear-gradient(112deg,#182944,#111b2e 70%,#101827);
 border:1px solid #33496d;border-radius:20px;padding:23px 26px;margin:7px 0 15px}
.portfolio-intro-head {display:flex;align-items:center;justify-content:space-between;
 gap:12px;flex-wrap:wrap}
.portfolio-intro-name {font-size:1.28rem;font-weight:780;color:#f4f8ff}
.portfolio-intro-pill {background:rgba(67,211,167,.11);color:#62dcb6;
 border:1px solid rgba(67,211,167,.32);border-radius:999px;
 padding:6px 11px;font-size:.72rem;font-weight:750}
.portfolio-intro-desc {color:#afc0dc;line-height:1.7;font-size:.92rem;
 max-width:1000px;margin:11px 0 16px}
.portfolio-stat-row {display:grid;grid-template-columns:repeat(4,minmax(0,1fr));
 gap:10px;margin-top:12px}
.portfolio-stat {background:#101a2c;border:1px solid #2b3c58;
 border-radius:13px;padding:13px 15px}
.portfolio-stat-label {font-size:.72rem;color:#92a7c9}
.portfolio-stat-value {font-size:1.15rem;font-weight:750;color:#f3f7ff;margin-top:6px}
.portfolio-stat-sub {font-size:.7rem;color:#8a9ebc;margin-top:4px}
.initial-weight-row {display:flex;justify-content:space-between;align-items:center;
 padding:10px 1px;border-bottom:1px solid #253550;gap:8px}
.initial-weight-label {color:#c9d7ed;font-size:.87rem;font-weight:650}
.initial-weight-value {color:#eaf1ff;font-size:.89rem;font-weight:760}
.initial-weight-sub {color:#8fa4c5;font-size:.74rem;margin-top:3px}
.initial-weight-track {height:7px;background:#25334b;border-radius:8px;overflow:hidden;margin-top:7px}
.initial-weight-fill {height:100%;border-radius:8px;background:var(--fill,#729cff)}
.transaction-panel {background:#111c2e;border:1px solid #2c3e5d;
 border-radius:15px;padding:17px 19px;margin-top:7px}
.transaction-kicker {font-size:.72rem;letter-spacing:.08em;text-transform:uppercase;
 color:#82aaff;font-weight:800}
.transaction-value {font-size:1.55rem;color:#f4f7ff;font-weight:800;margin-top:5px}
.transaction-sub {font-size:.8rem;color:#9cafcb;margin-top:5px}
.date-bar {background:#111b2c;border:1px solid #2b3b56;border-radius:14px;
 padding:12px 16px;margin-top:3px}
.date-bar-title {color:#eaf2ff;font-weight:700;font-size:.9rem}
.date-bar-sub {color:#8fa5c6;font-size:.78rem;margin-top:3px}
@media(max-width:850px){.portfolio-stat-row{grid-template-columns:repeat(2,minmax(0,1fr))}}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def read_csv(path):
    try:
        return pd.read_csv(path)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def read_json(path):
    try:
        with open(path, encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

def date_column(df):
    if df.empty or "date" not in df:
        return df
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return df

def filter_data(df, portfolio, date=None):
    if df.empty:
        return df
    result = df.copy()
    if "portfolio_id" in result:
        result = result[result["portfolio_id"].astype(str) == portfolio]
    if date is not None and "date" in result:
        result = result[result["date"] == date]
    return result

def value(row, *keys):
    for key in keys:
        if key in row:
            try:
                number = float(row[key])
                if pd.notna(number):
                    return number
            except (ValueError, TypeError):
                pass
    return None

def dollars(v):
    if v is None:
        return "—"
    return f"{'-' if v < 0 else ''}${abs(v):,.0f}"

def percent(v):
    return f"{v:.2%}" if v is not None else "—"

def millions(v):
    """USD metric in millions, preserving the negative sign."""
    if v is None:
        return "—"
    sign = "−" if v < 0 else ""
    return f"{sign}${abs(v) / 1_000_000:,.2f}M"

def pnl_millions(v):
    """Use K for small daily P&L; M for larger daily P&L."""
    if v is None:
        return "—"
    sign = "−" if v < 0 else "+"
    amount = abs(v)
    if amount >= 1_000_000:
        return f"{sign}${amount / 1_000_000:,.2f}M"
    return f"{sign}${amount / 1_000:,.1f}K"

def kpi_card(label, display, detail=None, loss=False,
             tone="#679aff", icon="◈", footer="Portfolio snapshot"):
    """Dark financial KPI tile with semantic accent and supporting context."""
    detail_html = (
        f'<div class="risk-kpi-detail">{escape(detail)}</div>'
        if detail else ""
    )
    value_class = "risk-kpi-value loss" if loss else "risk-kpi-value"
    st.markdown(
        f'<div class="risk-kpi" style="--kpi-color:{tone}">'
        '<div class="risk-kpi-head">'
        f'<div class="risk-kpi-label">{escape(label)}</div>'
        f'<div class="risk-kpi-icon">{escape(icon)}</div></div>'
        f'<div class="{value_class}">{escape(display)}</div>'
        f'{detail_html}'
        '<div class="risk-kpi-footer"><span class="risk-kpi-dot"></span>'
        f'{escape(footer)}</div></div>',
        unsafe_allow_html=True,
    )

def textval(v, fallback="—"):
    return str(v) if v is not None and not pd.isna(v) else fallback

def line_chart(df, key, title, pct=False, zero=False):
    if key not in df or df.empty:
        st.info(f"{title}: source series unavailable.")
        return
    sub = df[["date", key]].copy()
    sub[key] = pd.to_numeric(sub[key], errors="coerce")
    sub = sub.dropna().sort_values("date")
    sub["date"] = pd.to_datetime(sub["date"], errors="coerce")
    sub = sub.dropna(subset=["date"])
    if sub.empty:
        st.info(f"{title}: no observations.")
        return
    fig = go.Figure(go.Scatter(x=sub["date"], y=sub[key], mode="lines",
                               line=dict(width=2.4, color="#36a9b8"),
                               fill="tozeroy" if pct and title == "Drawdown" else None))
    if zero:
        fig.add_hline(y=0, line_color="gray", opacity=.5)
    fig.update_layout(title=title, height=300, margin=dict(l=4,r=4,t=44,b=4),
                      xaxis_title="", yaxis_title="", showlegend=False,
                      yaxis_tickformat=".1%" if pct else ",.0f")
    st.plotly_chart(fig, use_container_width=True)

def evidence_result(entry):
    if not isinstance(entry, dict):
        return {}
    for k in ("result", "data", "output"):
        if isinstance(entry.get(k), dict):
            return entry[k]
    return entry

def evidence_tool(entry):
    return str(entry.get("tool_name", entry.get("tool", "Tool"))) if isinstance(entry, dict) else "Tool"

def pnl_attribution(evidence):
    for entry in evidence:
        if evidence_tool(entry) != "get_pnl_attribution":
            continue
        result = evidence_result(entry)
        for key in ("instrument_attribution", "instrument_pnl",
                    "pnl_by_instrument", "instrument_contributions"):
            arr = result.get(key)
            if isinstance(arr, dict):
                arr = [{"instrument_id": k, "pnl_usd": v} for k, v in arr.items()]
            if not isinstance(arr, list):
                continue
            rows = []
            for x in arr:
                if not isinstance(x, dict):
                    continue
                sym = x.get("instrument_id", x.get("symbol"))
                amt = value(x, "pnl_usd", "daily_pnl_usd", "pnl_contribution_usd", "pnl")
                if sym and amt is not None:
                    rows.append((str(sym), amt))
            if rows:
                return pd.DataFrame(rows, columns=["Instrument", "P&L"])
    return pd.DataFrame()

def saved_investigation(portfolio, date, rule):
    """Use verified metadata where available; a narrowly scoped legacy filename fallback."""
    for path in sorted(OUTPUTS.glob("live_investigation_*.json")):
        doc = read_json(str(path))
        if not doc:
            continue
        context = doc.get("exception", {})
        if not isinstance(context, dict):
            context = {}
        p = str(context.get("portfolio_id", doc.get("portfolio_id", "")))
        d = str(context.get("date", context.get("as_of_date", doc.get("as_of_date", ""))))
        r = str(context.get("rule_id", doc.get("rule_id", "")))
        if (p, d, r) == (portfolio, date, rule):
            return doc, path
        if (path.name == "live_investigation_pnl_20250404.json"
                and (portfolio, date, rule) == ("PORT_001", "2025-04-04", "PNL_VAR_EXCEEDANCE")
                and (p, d) == (portfolio, date)):
            return doc, path
    return {}, None

def split_report(report):
    """Parse numbered and unnumbered Markdown headings from saved reports."""
    if not isinstance(report, str) or not report.strip():
        return []
    pattern = re.compile(
        r"(?m)^[ \t]{0,3}#{1,4}[ \t]+"
        r"(?:(\d+)[.)][ \t]+)?"
        r"(Executive Summary|Exception and Observed Risk|"
        r"Exception and observed risk|Supporting Evidence|Supporting evidence|"
        r"Potential Drivers|Potential drivers|Limitations|"
        r"Recommended Human Review|Recommended human review)"
        r"[ \t]*$",
        re.IGNORECASE,
    )
    matches=list(pattern.finditer(report))
    sections=[]
    for i,match in enumerate(matches):
        stop=matches[i+1].start() if i+1<len(matches) else len(report)
        sections.append((int(match.group(1)) if match.group(1) else i+1,
                         match.group(2).strip(),report[match.end():stop].strip()))
    return sections

def report_summary(report):
    for _,heading,body in split_report(report):
        if heading.casefold()=="executive summary":
            return body
    return ""

def render_report_markdown(text):
    """Render AI Markdown without treating financial USD amounts as LaTeX.

    Preserve report wording and evidence references; formatting-only changes.
    """
    if not text:
        return ""
    text=str(text).replace("\r\n","\n")
    # Streamlit Markdown interprets unescaped $...$ as inline math.
    # Risk reports use dollar signs for USD, not TeX.
    text=re.sub(r"(?<!\\)\$", r"\\$", text)
    # Some saved model outputs place bold markers between a number and
    # its unit; leave these in place rather than rewriting financial facts.
    return text

def evidence_based_overview(rule, row, evaluation):
    """Deterministic UI context when a saved AI draft lacks its own summary.

    Not represented as an AI-generated Executive Summary.
    """
    matching=evaluation[evaluation["rule_id"].astype(str)==rule] \
        if not evaluation.empty and "rule_id" in evaluation else pd.DataFrame()
    if matching.empty:
        return "The saved AI report has no Executive Summary. Review the detailed findings below."
    record=matching.iloc[0].to_dict()
    metric=str(record.get("metric",""))
    observed=pd.to_numeric(record.get("metric_value"),errors="coerce")
    breach=pd.to_numeric(record.get("breach_threshold"),errors="coerce")
    status=str(record.get("status","UNKNOWN"))
    if pd.isna(observed):
        return f"The recorded risk-limit evaluation is {status} for {rule}."
    unit="×" if metric=="pnl_loss_to_prior_var" else "%"
    shown=(f"{observed:.3f}{unit}" if unit=="×" else f"{100*observed:.2f}%")
    result=f"The recorded {rule.replace('_',' ').lower()} evaluation is {status}, with an observed value of {shown}."
    if pd.notna(breach):
        limit=f"{breach:.3f}×" if unit=="×" else f"{100*breach:.2f}%"
        result+=f" The configured breach threshold is {limit}."
        if metric!="pnl_loss_to_prior_var" and float(observed)>float(breach):
            result+=f" The observed value is {(observed-breach)*100:.2f} percentage points above that threshold."
    return result


def plot_style(fig, height=310):
    fig.update_layout(
        paper_bgcolor="#111a2a", plot_bgcolor="#111a2a",
        font=dict(color="#c8d5ea", size=12), height=height,
        margin=dict(l=10,r=12,t=45,b=12),
        xaxis=dict(gridcolor="#23324a", zerolinecolor="#36435a"),
        yaxis=dict(gridcolor="#23324a", zerolinecolor="#36435a"),
        legend=dict(orientation="h", y=-.22),
    )
    return fig

def bar_chart(df, x, y, title, pct=False, horizontal=False):
    if df.empty or x not in df or y not in df:
        st.info(f"{title}: data unavailable.")
        return
    data=df[[x,y]].copy()
    data[y]=pd.to_numeric(data[y],errors="coerce")
    data=data.dropna(subset=[y])
    if data.empty:
        st.info(f"{title}: data unavailable.")
        return
    if horizontal:
        data=data.sort_values(y)
        fig=go.Figure(go.Bar(x=data[y],y=data[x],orientation="h",
                            marker_color="#6495ff"))
    else:
        fig=go.Figure(go.Bar(x=data[x],y=data[y],
                            marker_color=["#fb6477" if v<0 else "#39c5a0"
                                          for v in data[y]]))
    fig.update_layout(title=title,showlegend=False)
    if pct:
        fig.update_xaxes(tickformat=".0%") if horizontal else fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(plot_style(fig),use_container_width=True)

def fmt_ratio(v):
    return f"{v:.3f}×" if v is not None else "—"

def safe_str(v):
    return escape(str(v)) if v is not None else "—"

def rule_status(row):
    return str(row.get("status", "UNKNOWN")).upper()

def chart_line(df, field, title, pct=False, color="#6c9fff"):
    if df.empty or field not in df:
        st.info(f"{title}: data unavailable.")
        return
    t=df[["date",field]].copy()
    t["date"]=pd.to_datetime(t["date"],errors="coerce")
    t[field]=pd.to_numeric(t[field],errors="coerce")
    t=t.dropna()
    if t.empty:
        st.info(f"{title}: insufficient history.")
        return
    fig=go.Figure(go.Scatter(x=t["date"],y=t[field],mode="lines",
                             line=dict(color=color,width=2.5),
                             fill="tozeroy" if field=="drawdown" else None,
                             fillcolor="rgba(255,103,124,.12)" if field=="drawdown" else None))
    fig.update_layout(title=title,showlegend=False)
    fig.update_yaxes(tickformat=".1%" if pct else ",.0f")
    st.plotly_chart(plot_style(fig),use_container_width=True)


# ---------------------- Read-only source integration ----------------------
portfolio_df = date_column(read_csv(str(SAMPLE / "daily_portfolio.csv")))
risk_df = date_column(read_csv(str(OUTPUTS / "daily_risk_metrics.csv")))
positions_df = date_column(read_csv(str(SAMPLE / "daily_positions.csv")))
transactions_df = date_column(read_csv(str(SAMPLE / "transactions.csv")))
exceptions_df = date_column(read_csv(str(OUTPUTS / "risk_exceptions.csv")))
evaluations_df = date_column(read_csv(str(OUTPUTS / "risk_limit_evaluations.csv")))

if portfolio_df.empty or risk_df.empty:
    st.error("Required source missing: daily_portfolio.csv or daily_risk_metrics.csv")
    st.stop()

join_keys = ["portfolio_id", "date"]
for name, frame in (("Accounting", portfolio_df), ("Risk", risk_df)):
    if not set(join_keys).issubset(frame.columns):
        st.error(f"{name} data must contain portfolio_id and date.")
        st.stop()

# Use accounting as the authoritative NAV/P&L source, risk output for risk fields.
risk_cols = [c for c in risk_df.columns if c not in portfolio_df.columns]
try:
    dashboard = portfolio_df.merge(
        risk_df[join_keys + risk_cols], on=join_keys,
        how="left", validate="one_to_one", indicator=True,
    )
except pd.errors.MergeError as exc:
    st.error(f"Data integration error: {exc}")
    st.stop()
unmatched = int((dashboard["_merge"] != "both").sum())
dashboard = dashboard.drop(columns=["_merge"])

portfolios = sorted(dashboard["portfolio_id"].dropna().astype(str).unique())
with st.sidebar:
    st.markdown(
        '<div class="sidebar-brand">'
        
        '<div class="sidebar-brand-title">AI Risk Investigator</div>'
        '<div class="sidebar-brand-sub">SilverStone Risk Intelligence<br>'
        'Portfolio risk &amp; evidence-led investigations</div>'
        '</div>', unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-label">Portfolio workspace</div>',
                unsafe_allow_html=True)
    portfolio = st.selectbox(
        "Select portfolio", portfolios, label_visibility="collapsed",
    )
    st.markdown('<div class="sidebar-label">Explore</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-nav-item"><span class="sidebar-nav-icon">◈</span>'
        '<div><div class="sidebar-nav-name">Portfolio Overview</div>'
        '<div class="sidebar-nav-sub">Performance &amp; risk metrics</div></div></div>'
        '<div class="sidebar-nav-item"><span class="sidebar-nav-icon">◉</span>'
        '<div><div class="sidebar-nav-name">Risk Limit Monitoring</div>'
        '<div class="sidebar-nav-sub">Policy checks &amp; exceptions</div></div></div>'
        '<div class="sidebar-nav-item"><span class="sidebar-nav-icon">◇</span>'
        '<div><div class="sidebar-nav-name">AI Investigations</div>'
        '<div class="sidebar-nav-sub">Saved reports, evidence &amp; trace</div></div></div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-label">Illustrative investigations</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="sidebar-case">'
        '<div class="sidebar-case-date">Sep 30, 2026</div>'
        '<div class="sidebar-case-title">Equity Exposure Limit</div>'
        '<div class="sidebar-case-meta">PORT_001 · Example case 01</div></div>'
        '<div class="sidebar-case" style="border-left-color:#f1ad65">'
        '<div class="sidebar-case-date">Apr 04, 2025</div>'
        '<div class="sidebar-case-title">P&amp;L / VaR Exceedance</div>'
        '<div class="sidebar-case-meta">PORT_001 · Example case 02</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sidebar-demo">'
        '<div class="sidebar-demo-title">ⓘ &nbsp; Demonstration scope</div>'
        '<div class="sidebar-demo-text">Only these two investigation cases '
        'are illustrative examples at present. Historical portfolio metrics '
        'and risk monitoring cover other available dates, but saved AI reports '
        'are not implied for every exception.</div></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sidebar-footer">'
        '<span class="sidebar-live-dot"></span>Read-only dashboard<br>'
        'Opening reports or changing dates does not trigger AI calls.<br>'
        'AI findings require human review.</div>',
        unsafe_allow_html=True,
    )

hist_all = filter_data(dashboard, portfolio).sort_values("date")
dates = sorted(hist_all["date"].dropna().unique(), reverse=True)
if not dates:
    st.error("No dates available for selected portfolio.")
    st.stop()

st.markdown('<div class="section-kicker">Portfolio monitoring / Risk intelligence</div>',
            unsafe_allow_html=True)
st.title("Portfolio Risk Dashboard")
st.caption("Daily risk snapshot, historical analytics, policy monitoring, and saved AI investigations")


# Portfolio composition is derived from the earliest available position
# snapshot. This is explicitly labelled as an opening snapshot, rather
# than asserting these weights were the original trade instructions.
portfolio_positions = filter_data(positions_df, portfolio).sort_values("date") \
    if not positions_df.empty else pd.DataFrame()
portfolio_trades = (filter_data(transactions_df, portfolio).copy()
                    if not transactions_df.empty else pd.DataFrame())
# Transaction files may use trade_date / execution_date rather than date.
# Keep the original columns for display, but normalize a date for analysis.
if not portfolio_trades.empty:
    transaction_date_field = next(
        (field for field in (
            "date", "trade_date", "execution_date", "transaction_date",
            "trade_timestamp", "execution_timestamp", "timestamp",
        ) if field in portfolio_trades.columns), None
    )
    if transaction_date_field:
        normalized_dates = pd.to_datetime(
            portfolio_trades[transaction_date_field], errors="coerce", utc=True
        )
        if transaction_date_field != "date":
            portfolio_trades["date"] = normalized_dates.dt.strftime("%Y-%m-%d")
        else:
            portfolio_trades["date"] = normalized_dates.dt.strftime("%Y-%m-%d")
        portfolio_trades = portfolio_trades.sort_values("date", na_position="last")

first_date = dates[-1]
first_record = filter_data(hist_all, portfolio, first_date).iloc[0].to_dict()
first_nav = value(first_record, "nav_usd")
last_date = dates[0]

opening = pd.DataFrame()
if not portfolio_positions.empty:
    opening_date = str(portfolio_positions["date"].iloc[0])
    opening = portfolio_positions[
        portfolio_positions["date"] == opening_date
    ].copy()
else:
    opening_date = first_date

symbol_col = next((c for c in ("symbol","instrument_id","ticker")
                   if c in opening.columns),None)
value_col = next((c for c in ("market_value_usd","position_value_usd",
                             "market_value","value_usd")
                  if c in opening.columns),None)
weight_col = next((c for c in ("weight_pct_nav","position_weight",
                              "weight","weight_pct")
                   if c in opening.columns),None)
opening_items=[]
if not opening.empty and symbol_col:
    for _,item in opening.iterrows():
        symbol=str(item.get(symbol_col,""))
        amt=pd.to_numeric(item.get(value_col),errors="coerce") if value_col else float("nan")
        w=pd.to_numeric(item.get(weight_col),errors="coerce") if weight_col else float("nan")
        if pd.isna(w) and pd.notna(amt) and first_nav and first_nav>0:
            w=float(amt)/first_nav
        if pd.notna(w) and weight_col and "pct" in weight_col and abs(w)>1:
            w=float(w)/100
        opening_items.append({
            "symbol":symbol,
            "amount":None if pd.isna(amt) else float(amt),
            "weight":None if pd.isna(w) else float(w),
        })
opening_items.sort(key=lambda x: x["weight"] if x["weight"] is not None else -1,
                   reverse=True)

trade_count=len(portfolio_trades)
trade_dates = (portfolio_trades["date"].dropna()
               if "date" in portfolio_trades.columns else pd.Series(dtype=str))
trade_start = str(trade_dates.min()) if not trade_dates.empty else None
trade_end = str(trade_dates.max()) if not trade_dates.empty else None
side_col=next((c for c in ("side","trade_side","action","transaction_type")
               if c in portfolio_trades.columns),None)
buy_count=sell_count=None
if side_col:
    side=portfolio_trades[side_col].astype(str).str.upper().str.strip()
    buy_count=int(side.isin(["BUY","B","PURCHASE"]).sum())
    sell_count=int(side.isin(["SELL","S","SALE"]).sum())
notional_col=next((c for c in ("notional_usd","trade_notional_usd",
                              "gross_notional_usd","notional","amount_usd")
                   if c in portfolio_trades.columns),None)
gross_notional=None
if notional_col and trade_count:
    gross_notional=float(pd.to_numeric(
        portfolio_trades[notional_col],errors="coerce"
    ).abs().sum())
elif trade_count and {"quantity","execution_price"}.issubset(portfolio_trades.columns):
    gross_notional=float((
        pd.to_numeric(portfolio_trades["quantity"],errors="coerce").abs()
        *pd.to_numeric(portfolio_trades["execution_price"],errors="coerce").abs()
    ).sum())

assets=sorted({x["symbol"] for x in opening_items if x["symbol"]})
assets_text=", ".join(assets) if assets else "Equity, rates and credit instruments"
st.markdown(
    '<div class="portfolio-intro">'
    '<div class="portfolio-intro-head">'
    f'<div class="portfolio-intro-name">◈ &nbsp; {safe_str(portfolio)}'
    ' &nbsp; · &nbsp; Multi-Asset Portfolio</div>'
    '<span class="portfolio-intro-pill">● &nbsp; Risk monitoring active</span></div>'
    '<div class="portfolio-intro-desc">'
    'A multi-asset portfolio designed to monitor risk and performance '
    'across equity, rates, and credit ETF exposures. '
    f'The recorded holdings include <strong>{escape(assets_text)}</strong>. '
    'Portfolio values, holdings, and transactions are based on the project '
    'datasets; the portfolio is illustrative and not a live managed account.'
    '</div>'
    '<div class="portfolio-stat-row">'
    f'<div class="portfolio-stat"><div class="portfolio-stat-label">Opening NAV</div>'
    f'<div class="portfolio-stat-value">{escape(millions(first_nav))}</div>'
    f'<div class="portfolio-stat-sub">As of {escape(first_date)}</div></div>'
    f'<div class="portfolio-stat"><div class="portfolio-stat-label">Portfolio universe</div>'
    f'<div class="portfolio-stat-value">{len(assets) if assets else "—"} ETFs</div>'
    '<div class="portfolio-stat-sub">Equity · Rates · Credit</div></div>'
    f'<div class="portfolio-stat"><div class="portfolio-stat-label">Recorded transactions</div>'
    f'<div class="portfolio-stat-value">{trade_count:,}</div>'
    '<div class="portfolio-stat-sub">Available transaction history</div></div>'
    f'<div class="portfolio-stat"><div class="portfolio-stat-label">History through</div>'
    f'<div class="portfolio-stat-value">{escape(last_date)}</div>'
    f'<div class="portfolio-stat-sub">{len(hist_all):,} daily snapshots</div></div>'
    '</div></div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="section-kicker">Snapshot controls</div>',
            unsafe_allow_html=True)
date_left,date_right=st.columns([1,2.6],vertical_alignment="bottom")
with date_left:
    date=st.selectbox("Snapshot date",dates,index=0,
                      help="Choose any available trading date. Defaults to latest.")
with date_right:
    st.markdown(
        '<div class="date-bar"><div class="date-bar-title">Historical risk snapshot</div>'
        '<div class="date-bar-sub">Select a date to update portfolio metrics, '
        'charts, policy evaluations and saved investigation availability.</div></div>',
        unsafe_allow_html=True,
    )


# As-of holdings: use the exact selected-date positions, not the opening snapshot.
selected_positions = filter_data(portfolio_positions, portfolio, date) \
    if not portfolio_positions.empty else pd.DataFrame()
selected_nav = value(filter_data(hist_all, portfolio, date).iloc[-1].to_dict(), "nav_usd")
selected_items = []
if not selected_positions.empty:
    symbol_field = next((c for c in ("symbol", "instrument_id", "ticker")
                         if c in selected_positions.columns), None)
    mv_field = next((c for c in ("market_value_usd", "position_value_usd",
                               "market_value", "value_usd")
                     if c in selected_positions.columns), None)
    weight_field = next((c for c in ("weight_pct_nav", "position_weight",
                                   "weight", "weight_pct")
                         if c in selected_positions.columns), None)
    if symbol_field:
        for _, holding in selected_positions.iterrows():
            amount = pd.to_numeric(holding.get(mv_field), errors="coerce") if mv_field else float("nan")
            weight = pd.to_numeric(holding.get(weight_field), errors="coerce") if weight_field else float("nan")
            if pd.isna(weight) and pd.notna(amount) and selected_nav and selected_nav > 0:
                weight = float(amount) / selected_nav
            if pd.notna(weight) and weight_field and "pct" in weight_field and abs(weight) > 1:
                weight = float(weight) / 100
            selected_items.append({
                "symbol": str(holding.get(symbol_field, "")),
                "amount": None if pd.isna(amount) else float(amount),
                "weight": None if pd.isna(weight) else float(weight),
            })
selected_items.sort(key=lambda x: x["weight"] if x["weight"] is not None else -1,
                    reverse=True)

# As-of trading history: do not include future transactions or undated rows.
# A missing date column is not evidence that all transactions occurred already.
if not portfolio_trades.empty and "date" in portfolio_trades.columns:
    selected_trades = portfolio_trades.loc[
        portfolio_trades["date"].notna()
        & (portfolio_trades["date"] <= date)
    ].copy()
else:
    selected_trades = pd.DataFrame(columns=portfolio_trades.columns)
selected_trade_count = len(selected_trades)
selected_trade_dates = selected_trades["date"].dropna() if "date" in selected_trades else pd.Series(dtype=str)
selected_trade_start = str(selected_trade_dates.min()) if not selected_trade_dates.empty else None
selected_trade_end = str(selected_trade_dates.max()) if not selected_trade_dates.empty else None
selected_side_col = next((c for c in ("side", "trade_side", "action", "transaction_type")
                          if c in selected_trades.columns), None)
selected_buys = selected_sells = None
if selected_side_col:
    sides = selected_trades[selected_side_col].astype(str).str.upper().str.strip()
    selected_buys = int(sides.isin(["BUY", "B", "PURCHASE"]).sum())
    selected_sells = int(sides.isin(["SELL", "S", "SALE"]).sum())
selected_notional_col = next((c for c in (
    "notional_usd", "trade_notional_usd", "gross_notional_usd", "notional", "amount_usd"
) if c in selected_trades.columns), None)
selected_gross = None
if selected_notional_col and selected_trade_count:
    selected_gross = float(pd.to_numeric(
        selected_trades[selected_notional_col], errors="coerce"
    ).abs().sum())
elif selected_trade_count and {"quantity", "execution_price"}.issubset(selected_trades.columns):
    selected_gross = float((
        pd.to_numeric(selected_trades["quantity"], errors="coerce").abs()
        * pd.to_numeric(selected_trades["execution_price"], errors="coerce").abs()
    ).sum())

st.markdown(f"#### Portfolio Construction & Trading History · {date}")
st.caption("Holdings as of the selected date and recorded trades executed on or before that date.")
construction_col, transactions_col=st.columns([1.25,1],gap="medium")
with construction_col:
    with st.container(border=True):
        st.markdown("**Portfolio Holdings & Weights**")
        st.caption(f"Position snapshot: {date}. Weights are based on the selected-date NAV.")
        if selected_items:
            colors=["#639aff","#a58bff","#38cba5","#f2b95f","#fa8298","#60c4d9"]
            for i,item in enumerate(selected_items):
                weight=item["weight"]
                amount=item["amount"]
                pct_text=f"{weight*100:.2f}%" if weight is not None else "N/A"
                amt_text=millions(amount) if amount is not None else "N/A"
                width=min(100,max(0,100*weight)) if weight is not None else 0
                color=colors[i%len(colors)]
                st.markdown(
                    '<div class="initial-weight-row">'
                    f'<div><div class="initial-weight-label">{escape(item["symbol"])}</div>'
                    f'<div class="initial-weight-sub">Market value: {escape(amt_text)}</div></div>'
                    f'<div class="initial-weight-value">{escape(pct_text)}</div></div>'
                    '<div class="initial-weight-track">'
                    f'<div class="initial-weight-fill" style="--fill:{color};width:{width:.2f}%"></div>'
                    '</div>',unsafe_allow_html=True,
                )
        else:
            st.info("No instrument-level positions available on the selected date.")
with transactions_col:
    with st.container(border=True):
        st.markdown("**Transaction Activity**")
        st.caption(f"Recorded transactions through {date}, inclusive; not inferred trades.")
        st.markdown(
            '<div class="transaction-panel">'
            '<div class="transaction-kicker">Recorded trades</div>'
            f'<div class="transaction-value">{selected_trade_count:,}</div>'
            '<div class="transaction-sub">'
            + (f'{escape(selected_trade_start)} to {escape(selected_trade_end)}'
               if selected_trade_start and selected_trade_end else "No dated transactions through this snapshot")
            + '</div></div>',
            unsafe_allow_html=True,
        )
        a,b=st.columns(2)
        with a:
            st.metric("Buys",str(selected_buys) if selected_buys is not None else "N/A")
        with b:
            st.metric("Sells",str(selected_sells) if selected_sells is not None else "N/A")
        if selected_gross is not None:
            st.metric("Gross traded notional",millions(selected_gross))
        st.caption("Gross notional sums absolute amounts of recorded trades through the selected date.")
        with st.expander("View transactions through selected date"):
            if selected_trades.empty:
                st.info("No recorded transactions through the selected date.")
            else:
                st.dataframe(selected_trades,hide_index=True,use_container_width=True)


if unmatched:
    st.warning(f"{unmatched} accounting observations have no matching risk records.")

row = filter_data(hist_all, portfolio, date).iloc[-1].to_dict()
history = hist_all[hist_all["date"] <= date].copy().sort_values("date")
pos = filter_data(positions_df, portfolio, date)
day_exc = filter_data(exceptions_df, portfolio, date)
day_eval = filter_data(evaluations_df, portfolio, date)

nav=value(row,"nav_usd")
pnl=value(row,"daily_pnl_usd")
var=value(row,"var_99_usd")
es=value(row,"es_99_usd")
vol=value(row,"annualized_volatility")
dd=value(row,"drawdown")
ratio=value(row,"pnl_loss_to_prior_var")

st.markdown('<div class="section-kicker">01 / Portfolio snapshot</div>',
            unsafe_allow_html=True)
st.subheader("Key Risk Metrics")
st.caption("Six core measures · USD amounts in millions · Selected-date risk snapshot")

# Use the actual risk output for all indicators, never example values.
metric_rows = [
    [
        dict(label="Portfolio NAV", display=millions(nav),
             detail="Total portfolio net asset value",
             tone="#6a9eff", icon="◈", footer="Portfolio valuation"),
        dict(label="Daily P&L", display=pnl_millions(pnl),
             detail=(f"Daily return: {percent(value(row, 'daily_return'))}"
                     if value(row, "daily_return") is not None else None),
             loss=pnl is not None and pnl < 0,
             tone="#ff6e83" if pnl is not None and pnl < 0 else "#38cba5",
             icon="↗" if pnl is not None and pnl >= 0 else "↘",
             footer="One-day portfolio performance"),
    ],
    [
        dict(label="99% Value at Risk", display=millions(var),
             detail=(f"{100*var/nav:.3f}% of NAV"
                     if var is not None and nav and nav > 0 else None),
             tone="#ac8aff", icon="◇", footer="1-day historical simulation"),
        dict(label="99% Expected Shortfall", display=millions(es),
             detail=(f"{100*es/nav:.3f}% of NAV"
                     if es is not None and nav and nav > 0 else None),
             tone="#f6b85e", icon="◎", footer="Average modeled tail loss"),
    ],
    [
        dict(label="Annualized Volatility", display=percent(vol),
             detail="63-day rolling return volatility",
             tone="#43c9ca", icon="〰", footer="Annualized portfolio risk"),
        dict(label="Drawdown", display=percent(dd),
             detail="Relative to historical NAV peak",
             loss=dd is not None and dd < 0,
             tone="#fa789b", icon="↘", footer="Peak-to-current decline"),
    ],
]
for pair in metric_rows:
    ca, cb = st.columns(2, gap="medium")
    with ca: kpi_card(**pair[0])
    with cb: kpi_card(**pair[1])

st.markdown(
    '<div class="snapshot-strip">'
    f'<span class="snapshot-chip">◷ &nbsp; As of <strong>{escape(date)}</strong></span>'
    f'<span class="snapshot-chip">◈ &nbsp; Loss / prior VaR: '
    f'<strong>{escape(fmt_ratio(ratio))}</strong></span>'
    f'<span class="snapshot-chip">◉ &nbsp; Largest holding: '
    f'<strong>{escape(str(row.get("largest_position_symbol") or "—"))}</strong></span>'
    '</div>',
    unsafe_allow_html=True,
)

st.markdown('<div class="section-kicker">02 / Portfolio analytics</div>',
            unsafe_allow_html=True)
st.subheader("Performance & Risk Trends")
perf, alloc, risk_tab = st.tabs([
    "Performance & Drawdown","Allocation & Concentration","VaR / ES / Volatility",
])
with perf:
    c1,c2=st.columns(2)
    with c1: chart_line(history,"nav_usd","Portfolio NAV (USD)")
    with c2: chart_line(history,"drawdown","Portfolio Drawdown",pct=True,color="#ff7489")
    bar_chart(history.tail(40),"date","daily_pnl_usd","Daily P&L · Last 40 trading days")
with alloc:
    allocations=[]
    for label,col,color in [
        ("Equity","equity_exposure_pct","#568dff"),
        ("Rates","rates_exposure_pct","#36c7a0"),
        ("Credit","credit_exposure_pct","#ffbd66"),
    ]:
        v=value(row,col)
        if v is not None:
            allocations.append((label,max(0,v),color))
    c1,c2=st.columns([1,1.1])
    with c1:
        if allocations:
            fig=go.Figure(go.Pie(
                labels=[x[0] for x in allocations],
                values=[x[1] for x in allocations],
                hole=.64,
                marker=dict(colors=[x[2] for x in allocations]),
                textinfo="label+percent",
            ))
            fig.update_layout(title="Asset-class allocation",
                              annotations=[dict(text=f"{millions(nav)}<br>NAV",
                                                x=.5,y=.5,showarrow=False,
                                                font=dict(color="#e7efff",size=14))])
            st.plotly_chart(plot_style(fig,340),use_container_width=True)
        else: st.info("Allocation not available for selected date.")
    with c2:
        if allocations:
            st.markdown("#### Exposure breakdown")
            st.dataframe(pd.DataFrame(
                [{"Asset class":a,"Weight":f"{b:.2%}",
                  "Exposure (USD)":dollars(nav*b) if nav is not None else "—"}
                 for a,b,_ in allocations]),hide_index=True,use_container_width=True)
        st.caption("Exposure percentages are based on the selected date's risk output.")
    if not pos.empty:
        st.markdown("#### Instrument positions")
        columns=[c for c in (
            "instrument_id","symbol","asset_class","market_value_usd",
            "weight","weight_pct_nav","position_weight","quantity"
        ) if c in pos.columns]
        st.dataframe(pos[columns] if columns else pos,
                     use_container_width=True,hide_index=True)
with risk_tab:
    c1,c2=st.columns(2)
    with c1: chart_line(history,"var_99_usd","99% VaR (USD)")
    with c2: chart_line(history,"es_99_usd","99% Expected Shortfall (USD)",color="#b08bff")
    c3,c4=st.columns(2)
    with c3: chart_line(history,"annualized_volatility",
                        "Rolling Annualized Volatility",pct=True,color="#38c9b1")
    with c4: chart_line(history,"pnl_loss_to_prior_var",
                        "P&L Loss / Prior-day VaR",color="#ffbd66")


def status_color(status):
    return {
        "PASS":"#36caa0","WARNING":"#ffbf65",
        "BREACH":"#ff647b","NOT_EVALUATED":"#9aa9c5",
    }.get(str(status).upper(),"#a6b5ce")

def monitoring_pct(metric, v):
    """Display policy percentages in percentage points, ratios as multiples."""
    if v is None or pd.isna(v):
        return "—"
    metric=str(metric).lower()
    if metric=="pnl_loss_to_prior_var":
        return f"{float(v):.3f}×"
    return f"{float(v)*100:.2f}%"

def monitoring_tile(label, count, color, subtitle):
    return (
        f'<div class="monitor-tile" style="--tone:{color}">'
        f'<div class="monitor-tile-label">{escape(label)}</div>'
        f'<div class="monitor-tile-value">{int(count)}</div>'
        f'<div class="monitor-tile-foot">{escape(subtitle)}</div></div>'
    )

def risk_limit_table(frame):
    """Present policy results without changing or recalculating decisions."""
    if frame.empty:
        return ""
    headers=("Risk rule","Observed","Warning limit","Breach limit","Status","Coverage")
    rows=[]
    for _,r in frame.iterrows():
        status=str(r.get("status","UNKNOWN")).upper()
        color=status_color(status)
        metric=str(r.get("metric",""))
        rule=str(r.get("rule_id","UNKNOWN"))
        observed=pd.to_numeric(r.get("metric_value"),errors="coerce")
        warning=pd.to_numeric(r.get("warning_threshold"),errors="coerce")
        breach=pd.to_numeric(r.get("breach_threshold"),errors="coerce")
        obs=None if pd.isna(observed) else float(observed)
        warn=None if pd.isna(warning) else float(warning)
        limit=None if pd.isna(breach) else float(breach)
        obs_text=monitoring_pct(metric,obs)
        warn_text=monitoring_pct(metric,warn)
        limit_text=monitoring_pct(metric,limit)
        # A visual comparison only: width is scaled to largest absolute
        # threshold/value. Actual status comes exclusively from backend.
        max_mag=max((abs(x) for x in (obs,warn,limit) if x is not None),default=0)
        width=(min(100,100*abs(obs)/max_mag) if obs is not None and max_mag else 0)
        available=r.get("available_observations")
        coverage=(f'{int(float(available))} observations'
                  if pd.notna(available) else "—")
        reason=r.get("reason_code")
        if status=="NOT_EVALUATED" and pd.notna(reason) and str(reason)!="None":
            coverage=escape(str(reason))
        else:
            coverage=escape(coverage)
        rows.append(
            f'<tr class="row-{escape(status)}">'
            f'<td><strong>{escape(rule.replace("_"," ").title())}</strong>'
            f'<div class="rule-sub">{escape(metric)}</div></td>'
            f'<td style="--tone:{color}"><strong style="color:{color}">{escape(obs_text)}</strong>'
            f'<div class="limit-track"><div class="limit-fill" '
            f'style="width:{width:.1f}%"></div></div></td>'
            f'<td>{escape(warn_text)}</td><td>{escape(limit_text)}</td>'
            f'<td><span class="status-pill" style="--tone:{color}">'
            f'{escape(status.replace("_"," "))}</span></td>'
            f'<td>{coverage}</td></tr>'
        )
    header="".join(f"<th>{h}</th>" for h in headers)
    return ('<div class="monitor-table-wrap"><table class="monitor-table">'
            f'<thead><tr>{header}</tr></thead><tbody>'
            + "".join(rows) + '</tbody></table></div>')

def threshold_strip(rule_row):
    """Show observed, warning, breach and excess for a selected policy."""
    metric=str(rule_row.get("metric",""))
    obs=pd.to_numeric(rule_row.get("metric_value"),errors="coerce")
    warn=pd.to_numeric(rule_row.get("warning_threshold"),errors="coerce")
    limit=pd.to_numeric(rule_row.get("breach_threshold"),errors="coerce")
    if pd.isna(obs) or pd.isna(limit):
        return ""
    # The direction of drawdown thresholds is reversed. For the highlighted
    # policy strip use upper-bound rules only.
    if float(limit) <= 0:
        return ""
    is_ratio=(metric=="pnl_loss_to_prior_var")
    excess=float(obs)-float(limit)
    def display(v):
        return monitoring_pct(metric,v)
    excess_label=(f'{excess:+.3f}×' if is_ratio else f'{excess*100:+.2f} pp')
    cards=[
        (display(float(obs)),"Observed", "#ff657b"),
        (display(float(warn)) if pd.notna(warn) else "—","Warning limit","#ffbf69"),
        (display(float(limit)),"Breach limit","#7ba5ff"),
        (excess_label,"Above breach limit",
         "#ff657b" if excess>0 else "#36caa0"),
    ]
    return ('<div class="threshold-grid">'+''.join(
        f'<div class="threshold-tile" style="--tone:{c}">'
        f'<div class="threshold-value">{escape(v)}</div>'
        f'<div class="threshold-label">{escape(l)}</div></div>'
        for v,l,c in cards
    )+'</div>')

st.markdown('<div class="section-kicker">03 / Policy monitoring</div>',
            unsafe_allow_html=True)
st.subheader("Risk Limit Monitoring")
st.caption(f"Risk limit evaluations for {portfolio} on {date}. "
           "All PASS, WARNING, BREACH and NOT_EVALUATED outcomes are retained.")

statuses=day_eval["status"].astype(str).str.upper().value_counts() \
    if not day_eval.empty and "status" in day_eval else pd.Series(dtype=int)
status_specs=[
    ("PASS","#36caa0","Within configured limits"),
    ("WARNING","#ffbf65","Approaching a limit"),
    ("BREACH","#ff647b","Requires investigation"),
    ("NOT_EVALUATED","#9aa9c5","Not enough valid evidence"),
]
st.markdown(
    '<div class="monitor-grid">'+''.join(
        monitoring_tile(label,statuses.get(label,0),color,subtitle)
        for label,color,subtitle in status_specs
    )+'</div>',
    unsafe_allow_html=True,
)

if day_eval.empty:
    st.info("No limit evaluation records for this date.")
else:
    # Highlight an actual BREACH, with priority to equity exposure if present.
    breaches=day_eval[day_eval["status"].astype(str).str.upper()=="BREACH"]
    if not breaches.empty:
        preferred=breaches[breaches["rule_id"].astype(str)=="EQUITY_EXPOSURE_LIMIT"]
        featured=(preferred.iloc[0] if not preferred.empty else breaches.iloc[0])
        st.markdown(
            f'#### <span style="color:#ff7b8c">●</span> '
            f'Featured breach · {escape(str(featured.get("rule_id","")).replace("_"," ").title())}',
            unsafe_allow_html=True,
        )
        strip=threshold_strip(featured)
        if strip:
            st.markdown(strip,unsafe_allow_html=True)
        st.caption("Thresholds and status come directly from the risk-limit evaluations. "
                   "The excess is shown in percentage points for percentage-based limits.")
    st.markdown("#### All Policy Evaluations")
    st.markdown(risk_limit_table(day_eval),unsafe_allow_html=True)

st.markdown("### Detected Exceptions")
st.markdown(
    '<div class="exception-legend">Risk-limit breaches and warnings requiring '
    'investigation · Select an exception to view its saved AI analysis.</div>',
    unsafe_allow_html=True,
)
if day_exc.empty:
    st.success("No exception records for this date.")
else:
    for i,ex in day_exc.reset_index(drop=True).iterrows():
        rule=str(ex.get("rule_id","UNKNOWN"))
        status=rule_status(ex)
        sev=str(ex.get("severity","")).upper()
        exc_id=str(ex.get("exception_id",f"{date}_{rule}"))
        item_key=f"{portfolio}|{date}|{rule}|{exc_id}"
        matches=day_eval[day_eval["rule_id"].astype(str)==rule] \
            if not day_eval.empty and "rule_id" in day_eval else pd.DataFrame()
        ev=matches.iloc[0] if not matches.empty else ex
        metric=str(ev.get("metric",ex.get("metric","")))
        observed=pd.to_numeric(ev.get("metric_value",ex.get("metric_value")),errors="coerce")
        threshold=pd.to_numeric(ev.get("breach_threshold"),errors="coerce")
        observed_txt=monitoring_pct(metric,None if pd.isna(observed) else observed)
        threshold_txt=monitoring_pct(metric,None if pd.isna(threshold) else threshold)
        severity_color={"HIGH":"#ff7589","MEDIUM":"#ffbd6a","LOW":"#79b6ff"}.get(sev,"#9caecb")
        title=rule.replace("_"," ").title()
        with st.container(border=True):
            left,mid,right=st.columns([4.4,2.4,1.5],gap="large",vertical_alignment="center")
            with left:
                st.markdown(
                    f'<div class="exception-eyebrow">Risk exception · {escape(date)}</div>'
                    f'<div class="exception-heading">{escape(title)}</div>'
                    f'<span class="exception-severity" style="--severity:{severity_color}">'
                    f'{escape(sev or status)} SEVERITY</span>'
                    f'<div class="exception-meta" style="margin-top:10px">'
                    f'{escape(portfolio)} &nbsp;·&nbsp; {escape(exc_id)}</div>',
                    unsafe_allow_html=True,
                )
            with mid:
                st.markdown(
                    f'<div class="exception-state">● {escape(status)}</div>'
                    f'<div class="exception-number">{escape(observed_txt)}</div>'
                    f'<div class="exception-number-sub">Observed · '
                    f'Breach limit {escape(threshold_txt)}</div>',
                    unsafe_allow_html=True,
                )
            with right:
                if st.button("View investigation →",key=f"open_{item_key}",
                             use_container_width=True):
                    st.session_state["selected_investigation"]=item_key

    valid={}
    for i,ex in day_exc.reset_index(drop=True).iterrows():
        rule=str(ex.get("rule_id","UNKNOWN"))
        exc_id=str(ex.get("exception_id",f"{date}_{rule}"))
        valid[f"{portfolio}|{date}|{rule}|{exc_id}"]=ex.to_dict()
    active=st.session_state.get("selected_investigation")
    if active in valid:
        ex=valid[active]
        rule=str(ex.get("rule_id","UNKNOWN"))
        status=rule_status(ex)
        saved,path=saved_investigation(portfolio,date,rule)
        st.markdown('<div class="section-kicker">04 / AI investigation</div>',
                    unsafe_allow_html=True)
        a,b=st.columns([5,1],vertical_alignment="center")
        with a:
            st.markdown(
                '<div class="hero"><div class="hero-eyebrow">'
                'AI Risk Investigator · Saved Investigation</div>'
                f'<div class="hero-title">{safe_str(rule.replace("_"," ").title())}'
                f' <span class="small-tag">{safe_str(status)}</span></div>'
                f'<div class="hero-meta">{safe_str(portfolio)} · {safe_str(date)}'
                f' · Rule: {safe_str(rule)}</div></div>',
                unsafe_allow_html=True,
            )
        with b:
            if st.button("Close report",use_container_width=True):
                st.session_state.pop("selected_investigation",None)
                st.rerun()
        if not saved:
            st.info("No matching saved investigation found. This dashboard does not "
                    "start live investigations or spend API credits.")
        else:
            report=saved.get("draft_report","") or ""
            evidence=saved.get("evidence",[])
            trace=saved.get("trace",[])
            summary=report_summary(report)
            report_status=str(saved.get("status","UNKNOWN"))
            st.caption(f"Report: DRAFT  ·  Review: {report_status}  ·  "
                       f"Evidence records: {len(evidence)}  ·  "
                       f"Trace events: {len(trace)}")
            summary_tab,findings_tab,evidence_tab,trace_tab,gov_tab=st.tabs([
                "Executive Summary","Detailed Findings","Data & Evidence",
                "Trace & Workflow","Governance",
            ])
            with summary_tab:
                if summary:
                    st.markdown("### Executive Summary")
                    st.markdown(render_report_markdown(summary))
                else:
                    st.info("This saved AI report has no Executive Summary section. "
                            "The following is a deterministic overview from the "
                            "risk-limit evaluation, not an AI-authored summary.")
                    st.markdown(render_report_markdown(
                        evidence_based_overview(rule,row,day_eval)
                    ))
                    st.caption("To add an AI-authored Executive Summary, regenerate "
                               "the saved report using the updated report instructions.")
                observed=value(ex,"metric_value")
                warning=value(ex,"warning_threshold")
                breach=value(ex,"breach_threshold")
                if "EQUITY_EXPOSURE" in rule:
                    observed=value(row,"equity_exposure_pct") if observed is None else observed
                if observed is not None:
                    st.markdown("#### Observed Risk vs. Policy Thresholds")
                    if warning is not None and breach is not None:
                        c1,c2,c3=st.columns(3)
                        c1.metric("Observed",f"{observed:.2%}")
                        c2.metric("Warning limit",f"{warning:.2%}")
                        c3.metric("Breach limit",f"{breach:.2%}")
                        fig=go.Figure(go.Bar(
                            x=["Observed","Warning","Breach"],
                            y=[observed,warning,breach],
                            marker_color=["#ff657b","#ffbd66","#639aff"],
                            text=[f"{v:.2%}" for v in [observed,warning,breach]],
                            textposition="outside"))
                        fig.update_yaxes(tickformat=".0%")
                        fig.update_layout(title="Observed metric vs. policy limits",
                                          showlegend=False)
                        st.plotly_chart(plot_style(fig,310),use_container_width=True)
                    else:
                        st.metric("Observed exception metric",f"{observed:.4f}")
                        st.caption("Policy thresholds are not present in the exception row.")
                if allocations:
                    st.markdown("#### Portfolio Risk Context")
                    x,y=st.columns(2)
                    with x:
                        fig=go.Figure(go.Pie(
                            labels=[z[0] for z in allocations],
                            values=[z[1] for z in allocations],hole=.6,
                            marker=dict(colors=[z[2] for z in allocations])))
                        fig.update_layout(title="Asset allocation")
                        st.plotly_chart(plot_style(fig,300),use_container_width=True)
                    with y:
                        st.markdown("**Selected-date risk metrics**")
                        st.dataframe(pd.DataFrame([
                            {"Metric":"NAV","Value":millions(nav)},
                            {"Metric":"Equity exposure","Value":percent(value(row,"equity_exposure_pct"))},
                            {"Metric":"99% VaR","Value":millions(var)},
                            {"Metric":"99% ES","Value":millions(es)},
                            {"Metric":"Annualized volatility","Value":percent(vol)},
                            {"Metric":"Drawdown","Value":percent(dd)},
                        ]),hide_index=True,use_container_width=True)
                st.caption("AI-generated draft; findings require human validation.")
            with findings_tab:
                sections=split_report(report)
                if sections:
                    for n,heading,body in sections:
                        if heading.casefold()=="executive summary":
                            continue
                        with st.container(border=True):
                            st.markdown(f"#### {heading}")
                            st.markdown(render_report_markdown(body))
                else:
                    st.markdown(render_report_markdown(report) if report
                                else "No report text available.")
                st.download_button("Export complete report (.md)",report,
                                   file_name=f"investigation_{portfolio}_{date}_{rule}.md",
                                   mime="text/markdown")
            with evidence_tab:
                pnl_df=pnl_attribution(evidence)
                if not pnl_df.empty:
                    bar_chart(pnl_df,"Instrument","P&L",
                              "Instrument-level P&L attribution",horizontal=True)
                if not pos.empty:
                    st.markdown("#### Instrument concentration")
                    pos_weight=next((c for c in (
                        "weight_pct_nav","position_weight","weight"
                    ) if c in pos),None)
                    sym=next((c for c in ("instrument_id","symbol") if c in pos),None)
                    if pos_weight and sym:
                        bar_chart(pos,sym,pos_weight,
                                  "Position weights",horizontal=True)
                st.markdown(f"#### Collected Evidence ({len(evidence)})")
                if not evidence:
                    st.info("No structured evidence records in saved JSON.")
                for j,item in enumerate(evidence):
                    with st.expander(f"Evidence {j} · {evidence_tool(item)}"):
                        st.json(item,expanded=False)
                st.download_button("Export evidence (.json)",
                                   json.dumps(evidence,indent=2,default=str),
                                   file_name=f"evidence_{portfolio}_{date}.json",
                                   mime="application/json")
            with trace_tab:
                st.markdown("#### Investigation Trace")
                if trace:
                    rows=[]
                    for j,event in enumerate(trace):
                        if isinstance(event,dict):
                            rows.append({
                                "Step":j+1,
                                "Event":event.get("event_type",""),
                                "Timestamp":event.get("timestamp_utc",""),
                                "Details":json.dumps(event.get("details",{}),
                                                     default=str,ensure_ascii=False)[:500],
                            })
                    if rows:
                        st.dataframe(pd.DataFrame(rows),hide_index=True,
                                     use_container_width=True)
                else: st.info("No investigation trace recorded.")
                st.markdown("#### Execution Budget")
                st.json(saved.get("budget",{}))
                if saved.get("api_audit"):
                    with st.expander("API audit"):
                        st.dataframe(pd.DataFrame(saved["api_audit"]),
                                     hide_index=True,use_container_width=True)
            with gov_tab:
                st.markdown("#### Governance & Human Review")
                st.markdown(f"**Investigation status:** `{report_status}`")
                st.markdown("**Report classification:** AI-generated draft, not approval")
                st.markdown("**Human review:** Required before operational conclusions")
                limitations=saved.get("limitations",[])
                if limitations:
                    st.markdown("#### Tool and evidence limitations")
                    for item in limitations:
                        st.warning(str(item))
                st.markdown("#### Recommended human review")
                for n,heading,body in split_report(report):
                    if heading.lower()=="recommended human review":
                        st.markdown(render_report_markdown(body))
                if path:
                    st.caption(f"Saved source: {path.relative_to(ROOT)}")

st.divider()
st.caption("SilverStone Risk Intelligence · Read-only risk analytics · "
           "No API calls are made by this dashboard.")
