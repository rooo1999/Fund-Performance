"""
Mutual Fund Basket Comparison - Streamlit app

Define several baskets of mutual funds (scheme code + weight), pick a start date and
a start amount, and see how the money would have grown in each basket.

NAV data comes from the free mfapi.in API (AMFI data), keyed by scheme code.
Run locally:   streamlit run app.py
"""

from datetime import date, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

st.set_page_config(page_title="Basket Comparison", page_icon="📊", layout="wide")

API = "https://api.mfapi.in/mf"

# --------------------------------------------------------------------------
# Default baskets (EDIT THESE or use the table / CSV upload in the app).
# Scheme codes are from AMFI - verify the resolved fund name shown in the app.
# --------------------------------------------------------------------------
DEFAULT_BASKETS = pd.DataFrame(
    [
        ["Basket A", 120716, 50.0],
        ["Basket A", 118989, 30.0],
        ["Basket A", 125354, 20.0],
        ["Basket B", 122639, 60.0],
        ["Basket B", 118989, 40.0],
    ],
    columns=["Basket", "Scheme Code", "Weight %"],
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def inr(x: float) -> str:
    """Format a number in Indian digit grouping with the rupee sign."""
    sign = "-" if x < 0 else ""
    s = f"{abs(x):.0f}"
    if len(s) > 3:
        head, tail = s[:-3], s[-3:]
        parts = []
        while len(head) > 2:
            parts.insert(0, head[-2:])
            head = head[:-2]
        if head:
            parts.insert(0, head)
        s = ",".join(parts + [tail])
    return f"{sign}₹{s}"


@st.cache_data(ttl=6 * 3600, show_spinner=False)
def fetch_nav(code: int):
    """Return (name, NAV series indexed by date) for a scheme code."""
    r = requests.get(f"{API}/{int(code)}", timeout=30)
    r.raise_for_status()
    js = r.json()
    data = js.get("data", [])
    if not data:
        raise ValueError("no NAV data returned")
    df = pd.DataFrame(data)
    df["date"] = pd.to_datetime(df["date"], format="%d-%m-%Y")
    df["nav"] = pd.to_numeric(df["nav"], errors="coerce")
    s = df.dropna().drop_duplicates("date").set_index("date")["nav"].sort_index()
    s = s[s > 0]
    return js.get("meta", {}).get("scheme_name", str(code)), s


@st.cache_data(ttl=3600, show_spinner=False)
def search_funds(q: str) -> pd.DataFrame:
    r = requests.get(f"{API}/search", params={"q": q}, timeout=30)
    r.raise_for_status()
    return pd.DataFrame(r.json())


def simulate(prices: np.ndarray, index: pd.DatetimeIndex, w: np.ndarray,
             amount: float, freq: str) -> np.ndarray:
    """Daily portfolio value for one basket, with optional periodic rebalancing."""
    n = len(index)
    if freq == "Monthly":
        keys = index.year * 12 + index.month
    elif freq == "Quarterly":
        keys = index.year * 4 + index.quarter
    elif freq == "Annually":
        keys = index.year
    else:
        keys = None

    units = amount * w / prices[0]
    vals = np.empty(n)
    vals[0] = amount
    for i in range(1, n):
        if keys is not None and keys[i] != keys[i - 1]:
            total = (units * prices[i]).sum()
            units = total * w / prices[i]
        vals[i] = (units * prices[i]).sum()
    return vals


def metrics(values: pd.Series, amount: float) -> dict:
    days = (values.index[-1] - values.index[0]).days
    final = values.iloc[-1]
    cagr = (final / amount) ** (365.25 / days) - 1 if days > 0 else np.nan
    rets = values.pct_change().dropna()
    vol = rets.std() * np.sqrt(252) if len(rets) > 1 else np.nan
    dd = values / values.cummax() - 1
    return {
        "Final Value": final,
        "Gain": final - amount,
        "Absolute Return": final / amount - 1,
        "CAGR": cagr,
        "Volatility (ann.)": vol,
        "Max Drawdown": dd.min(),
        "Return / Risk": cagr / vol if vol and vol > 0 else np.nan,
    }, dd


# --------------------------------------------------------------------------
# Sidebar - inputs
# --------------------------------------------------------------------------
st.title("📊 Mutual Fund Basket Comparison")
st.caption("Lump-sum investment on a chosen start date, compared across your baskets.")

with st.sidebar:
    st.header("Inputs")
    start_date = st.date_input("Start date", value=date.today() - timedelta(days=3 * 365),
                               min_value=date(2005, 1, 1), max_value=date.today())
    end_date = st.date_input("End date", value=date.today(),
                             min_value=date(2005, 1, 1), max_value=date.today())
    amount = st.number_input("Start amount (₹)", min_value=1000.0, value=1_000_000.0,
                             step=50_000.0, format="%.0f")
    rebalance = st.selectbox("Rebalancing", ["None (buy & hold)", "Monthly", "Quarterly", "Annually"],
                             index=0)
    rebalance = "None" if rebalance.startswith("None") else rebalance
    auto_shift = st.checkbox(
        "Auto-shift start date if a fund is too new",
        value=False,
        help="If any fund has no NAV on the start date, move the common start date forward "
             "to the date all funds have data, so every basket is compared fairly.",
    )

# --------------------------------------------------------------------------
# Basket definition
# --------------------------------------------------------------------------
st.subheader("1. Define your baskets")

c1, c2 = st.columns([3, 1])
with c2:
    st.download_button("Download template CSV", DEFAULT_BASKETS.to_csv(index=False),
                       "baskets_template.csv", "text/csv", use_container_width=True)
    up = st.file_uploader("Upload baskets CSV", type="csv", label_visibility="collapsed")

if up is not None:
    try:
        base_df = pd.read_csv(up)
        base_df = base_df[["Basket", "Scheme Code", "Weight %"]]
    except Exception as e:
        st.error(f"Could not read CSV (need columns Basket, Scheme Code, Weight %): {e}")
        base_df = DEFAULT_BASKETS
else:
    base_df = st.session_state.get("baskets_df", DEFAULT_BASKETS)

with c1:
    edited = st.data_editor(
        base_df,
        num_rows="dynamic",
        use_container_width=True,
        key="basket_editor",
        column_config={
            "Basket": st.column_config.TextColumn("Basket", required=True),
            "Scheme Code": st.column_config.NumberColumn("Scheme Code", format="%d", required=True),
            "Weight %": st.column_config.NumberColumn("Weight %", min_value=0.0, max_value=100.0,
                                                      format="%.2f", required=True),
        },
    )
st.session_state["baskets_df"] = edited

with st.expander("🔎 Find a scheme code"):
    q = st.text_input("Search fund name (e.g. 'parag parikh flexi direct growth')")
    if q and len(q) >= 3:
        try:
            res = search_funds(q)
            st.dataframe(res.rename(columns={"schemeCode": "Scheme Code", "schemeName": "Fund Name"}),
                         use_container_width=True, hide_index=True)
        except Exception as e:
            st.error(f"Search failed: {e}")

run = st.button("Run comparison", type="primary")
if not run:
    st.info("Set your baskets and inputs, then click **Run comparison**.")
    st.stop()

# --------------------------------------------------------------------------
# Validate input table
# --------------------------------------------------------------------------
tbl = edited.dropna(subset=["Basket", "Scheme Code", "Weight %"]).copy()
tbl["Basket"] = tbl["Basket"].astype(str).str.strip()
tbl["Scheme Code"] = tbl["Scheme Code"].astype(int)
tbl = tbl[(tbl["Basket"] != "") & (tbl["Weight %"] > 0)]
if tbl.empty:
    st.error("Add at least one basket with funds and weights.")
    st.stop()
if start_date >= end_date:
    st.error("Start date must be before end date.")
    st.stop()

# merge duplicate funds inside a basket
tbl = tbl.groupby(["Basket", "Scheme Code"], as_index=False)["Weight %"].sum()
wsum = tbl.groupby("Basket")["Weight %"].sum()
for b, s in wsum.items():
    if abs(s - 100) > 0.01:
        st.warning(f"**{b}**: weights add up to {s:.2f}%, not 100%. Normalised to 100%.")
tbl["w"] = tbl["Weight %"] / tbl.groupby("Basket")["Weight %"].transform("sum")

# --------------------------------------------------------------------------
# Fetch NAVs
# --------------------------------------------------------------------------
codes = sorted(tbl["Scheme Code"].unique())
names, series = {}, {}
with st.spinner(f"Fetching NAV history for {len(codes)} funds..."):
    for code in codes:
        try:
            names[code], series[code] = fetch_nav(code)
        except Exception as e:
            st.error(f"Scheme {code}: could not fetch data ({e}). Check the scheme code.")
            st.stop()

navs = pd.DataFrame(series).sort_index().ffill()
first_dt = {c: series[c].index[0] for c in codes}
last_dt = {c: series[c].index[-1] for c in codes}

# end date: clip to latest date all funds have data
end_eff = min(pd.Timestamp(end_date), min(last_dt.values()))
if end_eff < pd.Timestamp(end_date) - pd.Timedelta(days=5):
    st.warning(f"Some fund's latest NAV is {end_eff.date()}; comparison ends there.")

# start date: first trading date on/after chosen start
req_start = pd.Timestamp(start_date)
if auto_shift:
    req_start = max(req_start, max(first_dt.values()))
cand = navs.index[navs.index >= req_start]
if len(cand) == 0:
    st.error("No NAV data on or after the start date.")
    st.stop()
s0 = cand[0]
if s0 != pd.Timestamp(start_date):
    st.caption(f"Start date used: **{s0.date()}** (first available NAV on/after your chosen date).")

window = navs.loc[s0:end_eff]
if len(window) < 2:
    st.error("Not enough data in the selected window.")
    st.stop()

# --------------------------------------------------------------------------
# Simulate
# --------------------------------------------------------------------------
growth, dds, rows, skipped = {}, {}, [], []
for b, g in tbl.groupby("Basket", sort=False):
    too_new = [names[c] for c in g["Scheme Code"] if first_dt[c] > s0]
    if too_new:
        skipped.append((b, too_new))
        continue
    prices = window[list(g["Scheme Code"])].values
    vals = pd.Series(simulate(prices, window.index, g["w"].values, amount, rebalance),
                     index=window.index)
    m, dd = metrics(vals, amount)
    growth[b], dds[b] = vals, dd
    rows.append({"Basket": b, **m})

for b, too_new in skipped:
    st.error(f"**{b}** skipped - no NAV on {s0.date()} for: {', '.join(too_new)}. "
             "Tick 'Auto-shift start date' in the sidebar to compare from a later common date.")
if not growth:
    st.stop()

summary = pd.DataFrame(rows).set_index("Basket")

# --------------------------------------------------------------------------
# Results
# --------------------------------------------------------------------------
st.subheader("2. Results")
st.caption(f"{inr(amount)} invested on {s0.date()} · evaluated to {end_eff.date()} · "
           f"rebalancing: {rebalance}")

best = summary["Final Value"].idxmax()
cols = st.columns(len(summary))
for col, (b, r) in zip(cols, summary.iterrows()):
    col.metric(("🏆 " if b == best else "") + b, inr(r["Final Value"]),
               f"{r['Absolute Return']:+.1%}  ·  CAGR {r['CAGR']:.1%}")

tab1, tab2, tab3, tab4 = st.tabs(["Growth", "Drawdown", "Summary table", "Fund details"])

with tab1:
    fig = go.Figure()
    for b, v in growth.items():
        fig.add_trace(go.Scatter(x=v.index, y=v.values, name=b, mode="lines"))
    fig.add_hline(y=amount, line_dash="dot", line_color="grey", annotation_text="Start amount")
    fig.update_layout(height=480, hovermode="x unified", yaxis_title="Portfolio value (₹)",
                      legend=dict(orientation="h", y=1.08), margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

    # relative performance: indexed to 100
    fig2 = go.Figure()
    for b, v in growth.items():
        fig2.add_trace(go.Scatter(x=v.index, y=v / amount * 100, name=b, mode="lines"))
    fig2.update_layout(height=320, hovermode="x unified", yaxis_title="Indexed (start = 100)",
                       showlegend=False, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig2, use_container_width=True)

with tab2:
    fig = go.Figure()
    for b, d in dds.items():
        fig.add_trace(go.Scatter(x=d.index, y=d.values * 100, name=b, mode="lines"))
    fig.update_layout(height=420, hovermode="x unified", yaxis_title="Drawdown (%)",
                      legend=dict(orientation="h", y=1.08), margin=dict(l=10, r=10, t=30, b=10))
    st.plotly_chart(fig, use_container_width=True)

with tab3:
    show = summary.copy()
    show["Final Value"] = show["Final Value"].map(inr)
    show["Gain"] = show["Gain"].map(inr)
    for c in ["Absolute Return", "CAGR", "Volatility (ann.)", "Max Drawdown"]:
        show[c] = show[c].map(lambda x: f"{x:.2%}")
    show["Return / Risk"] = show["Return / Risk"].map(lambda x: f"{x:.2f}")
    st.dataframe(show, use_container_width=True)
    st.download_button("Download growth data (CSV)",
                       pd.DataFrame(growth).round(2).to_csv(),
                       "basket_growth.csv", "text/csv")
    st.caption("CAGR for periods under 1 year is annualised and can look extreme. "
               "Volatility is based on daily NAV returns. Direct/Regular and Growth/IDCW "
               "plan choice affects returns - check the fund names below.")

with tab4:
    det = tbl.copy()
    det["Fund"] = det["Scheme Code"].map(names)
    det["Fund return (period)"] = det["Scheme Code"].map(
        lambda c: window[c].iloc[-1] / window[c].iloc[0] - 1 if first_dt[c] <= s0 else np.nan)
    det["Weight %"] = det["Weight %"].map(lambda x: f"{x:.1f}")
    det["Fund return (period)"] = det["Fund return (period)"].map(
        lambda x: "n/a (too new)" if pd.isna(x) else f"{x:.2%}")
    st.dataframe(det[["Basket", "Scheme Code", "Fund", "Weight %", "Fund return (period)"]],
                 use_container_width=True, hide_index=True)

st.caption("Returns are based on NAV (dividends assumed reinvested in Growth plans). "
           "No taxes, exit loads or rebalancing costs are included. Past performance does not "
           "indicate future returns.")
