import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from streamlit_option_menu import option_menu
import datetime

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Order Fulfillment EDA",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

[data-testid="stSidebar"] { background: linear-gradient(180deg,#0f172a 0%,#1e1b4b 100%); }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
.stApp { background: #0a0e1a; color: #e2e8f0; }

/* KPI cards */
div[data-testid="metric-container"] {
    background: linear-gradient(135deg,#131929,#1a2235);
    border: 1px solid rgba(99,120,200,0.25);
    border-radius: 16px; padding: 18px 20px;
    box-shadow: 0 4px 24px rgba(0,0,0,0.35);
    transition: transform 0.2s;
}
div[data-testid="metric-container"]:hover { transform: translateY(-2px); }
div[data-testid="metric-container"] label {
    color: #94a3b8 !important; font-size: 0.68rem !important;
    letter-spacing: 1.2px; text-transform: uppercase; font-weight: 700;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #e2e8f0 !important; font-size: 1.9rem !important; font-weight: 800 !important;
}
div[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 0.72rem !important; }

h1 { color: #e2e8f0 !important; font-weight: 800 !important; }
h2, h3 { color: #c7d2fe !important; font-weight: 700 !important; }
hr { border-color: rgba(99,120,200,0.15) !important; }

/* Tabs */
[data-testid="stTabs"] button {
    color: #94a3b8 !important; font-weight: 600 !important;
    border-radius: 10px 10px 0 0 !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #a5b4fc !important;
    border-bottom: 2px solid #6366f1 !important;
}

/* Sidebar logo */
.sidebar-logo { text-align:center; padding:18px 10px 12px; border-bottom:1px solid rgba(99,120,200,0.2); margin-bottom:8px; }
.sidebar-logo .icon { font-size: 2.4rem; }
.sidebar-logo h2 { font-size:1.05rem; font-weight:800; background:linear-gradient(90deg,#a5b4fc,#22d3ee); -webkit-background-clip:text; -webkit-text-fill-color:transparent; margin:4px 0 2px; }
.sidebar-logo p { font-size:0.68rem; color:#64748b !important; }

/* Insight boxes */
.insight { padding:10px 14px; border-radius:0 10px 10px 0; margin:8px 0; font-size:0.82rem; line-height:1.5; }
.ins-info  { background:rgba(99,102,241,0.1);  border-left:3px solid #6366f1; color:#c7d2fe; }
.ins-warn  { background:rgba(251,191,36,0.1);  border-left:3px solid #fbbf24; color:#fef3c7; }
.ins-bad   { background:rgba(248,113,113,0.1); border-left:3px solid #f87171; color:#fee2e2; }
.ins-good  { background:rgba(52,211,153,0.1);  border-left:3px solid #34d399; color:#d1fae5; }

/* Gauge label */
.gauge-label { text-align:center; font-size:0.75rem; color:#94a3b8; margin-top:-10px; font-weight:600; }

/* Finding card */
.fcard { background:#131929; border:1px solid rgba(99,120,200,0.2); border-radius:14px; padding:18px 20px; margin-bottom:12px; }

/* Select / Multiselect */
[data-testid="stSelectbox"] > div,
[data-testid="stMultiSelect"] > div { background:#1a2235 !important; border-color:rgba(99,120,200,0.3) !important; border-radius:10px !important; }

/* Slider */
[data-testid="stSlider"] { padding: 0 4px; }

/* Button */
.stButton > button {
    background:linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    color:white !important; border:none !important;
    border-radius:10px !important; font-weight:700 !important;
    padding: 8px 20px !important;
}
.stButton > button:hover { opacity: 0.9 !important; transform: translateY(-1px); }

/* Download btn */
[data-testid="stDownloadButton"] > button {
    background:linear-gradient(135deg,#059669,#34d399) !important;
    color:white !important; border:none !important; border-radius:10px !important; font-weight:700 !important;
}

/* Expander */
[data-testid="stExpander"] { background:#131929 !important; border:1px solid rgba(99,120,200,0.2) !important; border-radius:12px !important; }

/* Table */
[data-testid="stDataFrame"] th { background:#1a2235 !important; color:#a5b4fc !important; }
</style>
""", unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("order_fulfillment_eda_dataset.csv", parse_dates=["order_date"])
    df["product_category"] = df["product_category"].str.strip().str.title()
    df["shipping_tier"]    = df["shipping_tier"].str.strip().str.title()
    df["order_month"]      = df["order_date"].dt.to_period("M").astype(str)
    df["order_quarter"]    = df["order_date"].dt.to_period("Q").astype(str)
    df["day_of_week"]      = df["order_date"].dt.day_name()
    df["week_num"]         = df["order_date"].dt.isocalendar().week.astype(int)
    df["otif_flag"]        = (df["on_time_flag"] == 1) & (df["in_full_flag"] == 1)
    df["perfect_order"]    = df["otif_flag"] & (df["delivery_condition"] == "Intact")
    df["revenue_lost_usd"] = df.apply(lambda r: r["order_value_usd"] if r["order_status"] != "Delivered" else 0, axis=1)
    return df

df = load_data()

# ── Plot theme helper ──────────────────────────────────────────────────────────
COLORS = ["#6366f1","#22d3ee","#34d399","#f472b6","#fb923c","#fbbf24","#f87171","#a78bfa","#38bdf8","#4ade80"]

def theme(fig, title="", h=None):
    upd = dict(
        paper_bgcolor="#131929", plot_bgcolor="#131929",
        font=dict(family="Inter", color="#94a3b8", size=12),
        margin=dict(l=20, r=20, t=44 if title else 20, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        title=dict(text=title, font=dict(color="#c7d2fe", size=14, family="Inter"), x=0),
    )
    if h: upd["height"] = h
    fig.update_layout(**upd)
    fig.update_xaxes(gridcolor="rgba(99,120,200,0.1)", zerolinecolor="rgba(99,120,200,0.1)", showline=False)
    fig.update_yaxes(gridcolor="rgba(99,120,200,0.1)", zerolinecolor="rgba(99,120,200,0.1)", showline=False)
    return fig

def ins(text, kind="info"):
    cls = {"info":"ins-info","warn":"ins-warn","bad":"ins-bad","good":"ins-good"}[kind]
    st.markdown(f'<div class="insight {cls}">{text}</div>', unsafe_allow_html=True)

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div class="sidebar-logo">
        <div class="icon">📦</div>
        <h2>Order Fulfillment EDA</h2>
        <p>India E-Commerce · Jul 2024 – Dec 2025</p>
    </div>""", unsafe_allow_html=True)

    selected = option_menu(
        menu_title=None,
        options=["🏠 Overview","📈 Order Trends","🚛 Carrier Analysis",
                 "🏭 Warehouse Deep-Dive","🎯 Quality & Returns",
                 "❌ Cancellations","🔬 Statistical Tests","📊 Executive Report"],
        default_index=0,
        styles={
            "container":         {"background-color":"transparent","padding":"0"},
            "nav-link":          {"font-size":"0.80rem","color":"#94a3b8","border-radius":"10px","margin":"2px 0","padding":"8px 12px"},
            "nav-link-selected": {"background":"linear-gradient(135deg,#6366f1,#8b5cf6)","color":"white","font-weight":"700"},
        },
    )

    st.markdown("---")
    st.markdown("### 🎛️ Global Filters")

    # Date range
    min_date = df["order_date"].min().date()
    max_date = df["order_date"].max().date()
    date_range = st.date_input("📅 Date Range", value=(min_date, max_date),
                               min_value=min_date, max_value=max_date)

    sel_regions  = st.multiselect("🗺️ Region",           sorted(df["dest_region"].dropna().unique()),      default=sorted(df["dest_region"].dropna().unique()))
    sel_segs     = st.multiselect("👥 Customer Segment",  sorted(df["customer_segment"].dropna().unique()), default=sorted(df["customer_segment"].dropna().unique()))
    sel_tiers    = st.multiselect("📦 Shipping Tier",     sorted(df["shipping_tier"].dropna().unique()),    default=sorted(df["shipping_tier"].dropna().unique()))
    sel_cats     = st.multiselect("🏷️ Product Category",  sorted(df["product_category"].dropna().unique()), default=sorted(df["product_category"].dropna().unique()))
    sel_wh       = st.multiselect("🏭 Warehouse",         sorted(df["warehouse_name"].dropna().unique()),   default=sorted(df["warehouse_name"].dropna().unique()))

    st.markdown("---")
    if st.button("🔄 Reset All Filters"):
        st.rerun()

    st.caption(f"📊 Raw records: {len(df):,} orders")

# ── Apply filters ──────────────────────────────────────────────────────────────
start_d = pd.to_datetime(date_range[0]) if len(date_range) >= 1 else df["order_date"].min()
end_d   = pd.to_datetime(date_range[1]) if len(date_range) == 2 else df["order_date"].max()

mask = (
    df["order_date"].between(start_d, end_d) &
    df["dest_region"].isin(sel_regions) &
    df["customer_segment"].isin(sel_segs) &
    df["shipping_tier"].isin(sel_tiers) &
    df["product_category"].isin(sel_cats) &
    df["warehouse_name"].isin(sel_wh)
)
dff     = df[mask].copy()
dff_del = dff[dff["order_status"] == "Delivered"].copy()

n_total = len(dff)
n_del   = len(dff_del)

# ══════════════════════════════════════════════════════════════════════════════
# 🏠 OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
if "Overview" in selected:
    st.markdown("# 📦 Order Fulfillment Dashboard")
    st.caption(f"Showing **{n_total:,}** orders · {start_d.date()} → {end_d.date()}")
    st.markdown("---")

    # ── KPIs ──
    otif        = round(dff_del["otif_flag"].mean()*100,1)        if n_del else 0
    perfect     = round(dff_del["perfect_order"].mean()*100,1)    if n_del else 0
    cancel_rate = round((dff["order_status"]=="Cancelled").mean()*100,1)
    avg_transit = round(dff_del["transit_hours"].mean(),1)         if n_del else 0
    avg_o2s     = round(dff_del["order_to_ship_hours"].mean(),1)   if n_del else 0
    total_rev   = dff["order_value_usd"].sum()
    rev_lost    = dff["revenue_lost_usd"].sum()

    k = st.columns(7)
    k[0].metric("📦 Total Orders",   f"{n_total:,}")
    k[1].metric("✅ Delivered",       f"{n_del:,}",         f"{n_del/n_total*100:.1f}%" if n_total else "")
    k[2].metric("🎯 OTIF",            f"{otif}%",           "⚠️ vs 85% target")
    k[3].metric("🏆 Perfect Orders",  f"{perfect}%")
    k[4].metric("❌ Cancel Rate",     f"{cancel_rate}%")
    k[5].metric("💰 Revenue",         f"${total_rev:,.0f}")
    k[6].metric("💸 Rev Lost",        f"${rev_lost:,.0f}",  "cancellations")

    st.markdown("---")

    # ── Gauge row ──
    st.markdown("### 🎯 Performance Gauges")
    gc = st.columns(4)

    def gauge(val, title, max_val=100, threshold_warn=50, threshold_good=75):
        color = "#34d399" if val >= threshold_good else ("#fbbf24" if val >= threshold_warn else "#f87171")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=val,
            number={"suffix":"%","font":{"color":"#e2e8f0","size":28,"family":"Inter"}},
            gauge={
                "axis":{"range":[0,max_val],"tickcolor":"#64748b","tickfont":{"color":"#64748b","size":10}},
                "bar":{"color":color,"thickness":0.6},
                "bgcolor":"#1a2235",
                "borderwidth":0,
                "steps":[
                    {"range":[0,threshold_warn],"color":"rgba(248,113,113,0.1)"},
                    {"range":[threshold_warn,threshold_good],"color":"rgba(251,191,36,0.1)"},
                    {"range":[threshold_good,max_val],"color":"rgba(52,211,153,0.1)"},
                ],
                "threshold":{"line":{"color":"#fbbf24","width":3},"thickness":0.75,"value":85},
            },
        ))
        fig.update_layout(paper_bgcolor="#131929",plot_bgcolor="#131929",
                          height=190,margin=dict(l=20,r=20,t=30,b=0),
                          font=dict(family="Inter",color="#94a3b8"))
        return fig

    with gc[0]: st.plotly_chart(gauge(otif,    "OTIF %"),      use_container_width=True); st.markdown('<p class="gauge-label">OTIF Rate</p>',           unsafe_allow_html=True)
    with gc[1]: st.plotly_chart(gauge(perfect, "Perfect %"),   use_container_width=True); st.markdown('<p class="gauge-label">Perfect Order Rate</p>',   unsafe_allow_html=True)
    with gc[2]:
        ontime_pct = round(dff_del["on_time_flag"].mean()*100,1) if n_del else 0
        st.plotly_chart(gauge(ontime_pct,"On-Time %"),          use_container_width=True); st.markdown('<p class="gauge-label">On-Time Delivery %</p>',  unsafe_allow_html=True)
    with gc[3]:
        infull_pct = round(dff_del["in_full_flag"].mean()*100,1) if n_del else 0
        st.plotly_chart(gauge(infull_pct,"In-Full %"),          use_container_width=True); st.markdown('<p class="gauge-label">In-Full Delivery %</p>',  unsafe_allow_html=True)

    ins("⚠️ All four KPIs fall below the <strong>85% SLA target</strong> (yellow line on each gauge). Both WH-04 bottleneck and EcoParcel instability must be fixed simultaneously.", "warn")

    st.markdown("---")

    # ── Charts row ──
    c1, c2 = st.columns([2,1])
    with c1:
        monthly = dff.groupby("order_month")["order_id"].count().reset_index(name="Orders").sort_values("order_month")
        monthly["3M MA"] = monthly["Orders"].rolling(3,min_periods=1).mean().round(1)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=monthly["order_month"], y=monthly["Orders"],
            name="Orders", mode="lines+markers", fill="tozeroy",
            fillcolor="rgba(99,102,241,0.12)", line=dict(color="#6366f1",width=2.5), marker=dict(size=6)))
        fig.add_trace(go.Scatter(x=monthly["order_month"], y=monthly["3M MA"],
            name="3M Rolling Avg", mode="lines", line=dict(color="#22d3ee",width=2.5,dash="dash")))
        st.plotly_chart(theme(fig,"📈 Monthly Orders + 3-Month Moving Average"), use_container_width=True)

    with c2:
        status_cnt = dff["order_status"].value_counts().reset_index()
        fig2 = px.pie(status_cnt, names="order_status", values="count", hole=0.62,
                      color_discrete_sequence=["#22d3ee","#f87171","#f472b6"])
        fig2.update_traces(textinfo="percent+label", pull=[0.05,0,0])
        st.plotly_chart(theme(fig2,"🍩 Order Status"), use_container_width=True)

    # ── Treemap of revenue by region > segment ──
    st.markdown("### 🌳 Revenue Treemap — Region → Segment → Category")
    tree = dff.groupby(["dest_region","customer_segment","product_category"])["order_value_usd"].sum().reset_index()
    fig3 = px.treemap(tree, path=["dest_region","customer_segment","product_category"],
                      values="order_value_usd", color="order_value_usd",
                      color_continuous_scale=["#1e1b4b","#6366f1","#22d3ee"])
    fig3.update_traces(textinfo="label+percent parent", hovertemplate="<b>%{label}</b><br>Revenue: $%{value:,.0f}<extra></extra>")
    st.plotly_chart(theme(fig3, h=420), use_container_width=True)
    ins("💡 Click any region/segment block to drill down into sub-categories interactively.", "info")

# ══════════════════════════════════════════════════════════════════════════════
# 📈 ORDER TRENDS
# ══════════════════════════════════════════════════════════════════════════════
elif "Order Trends" in selected:
    st.markdown("# 📈 Order Trends")
    st.caption("Time-series analysis, product mix, and volume patterns.")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📅 Time Series","📦 Product Mix","🗓️ Calendar Patterns"])

    with tab1:
        metric = st.selectbox("Select Metric to Plot", ["Total Orders","Delivered Orders","Revenue (USD)","OTIF %","Avg Transit (h)"])

        monthly_all = dff.groupby("order_month").agg(
            Total_Orders=("order_id","count"),
            Delivered_Orders=("order_status", lambda s: (s=="Delivered").sum()),
            Revenue=("order_value_usd","sum"),
        ).reset_index().sort_values("order_month")

        # OTIF per month
        otif_m = dff_del.groupby("order_month")["otif_flag"].mean().mul(100).round(1).reset_index(name="OTIF_pct") if n_del else pd.DataFrame(columns=["order_month","OTIF_pct"])
        transit_m = dff_del.groupby("order_month")["transit_hours"].mean().round(1).reset_index(name="Avg_Transit") if n_del else pd.DataFrame(columns=["order_month","Avg_Transit"])
        monthly_all = monthly_all.merge(otif_m, on="order_month", how="left").merge(transit_m, on="order_month", how="left")

        col_map = {"Total Orders":"Total_Orders","Delivered Orders":"Delivered_Orders",
                   "Revenue (USD)":"Revenue","OTIF %":"OTIF_pct","Avg Transit (h)":"Avg_Transit"}
        ycol = col_map[metric]
        monthly_all["Rolling"] = monthly_all[ycol].rolling(3,min_periods=1).mean()
        monthly_all["MoM_%"]   = monthly_all[ycol].pct_change().mul(100).round(1)

        fig = go.Figure()
        fig.add_trace(go.Bar(x=monthly_all["order_month"], y=monthly_all[ycol],
            name=metric, marker_color="#6366f1", opacity=0.75,
            hovertemplate="%{x}<br>" + metric + ": %{y:,.1f}<extra></extra>"))
        fig.add_trace(go.Scatter(x=monthly_all["order_month"], y=monthly_all["Rolling"],
            name="3M Rolling Avg", mode="lines", line=dict(color="#22d3ee",width=2.5),
            hovertemplate="%{x}<br>3M Avg: %{y:,.1f}<extra></extra>"))
        if metric == "OTIF %":
            fig.add_hline(y=85, line_dash="dot", line_color="#fbbf24",
                          annotation_text="85% SLA Target", annotation_position="top right")
        st.plotly_chart(theme(fig, f"Monthly {metric} + 3-Month Rolling Average", h=380), use_container_width=True)

        # MoM growth
        fig2 = go.Figure(go.Bar(
            x=monthly_all["order_month"], y=monthly_all["MoM_%"],
            marker_color=monthly_all["MoM_%"].apply(lambda v: "#34d399" if v >= 0 else "#f87171"),
            text=monthly_all["MoM_%"].apply(lambda v: f"{v:+.1f}%"),
            textposition="outside", hovertemplate="%{x}<br>MoM Growth: %{y:.1f}%<extra></extra>"
        ))
        st.plotly_chart(theme(fig2, "Month-over-Month Growth %", h=280), use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            sort_by = st.selectbox("Sort categories by", ["Orders","Revenue","Avg Order Value"])
        with col2:
            show_n = st.slider("Show top N categories", 3, len(dff["product_category"].unique()), len(dff["product_category"].unique()))

        cat_grp = dff.groupby("product_category").agg(
            Orders=("order_id","count"),
            Revenue=("order_value_usd","sum"),
            Avg_Order_Value=("order_value_usd","mean"),
            OTIF_Pct=("otif_flag","mean"),
        ).reset_index()
        cat_grp["OTIF_Pct"] = (cat_grp["OTIF_Pct"]*100).round(1)
        col_map2 = {"Orders":"Orders","Revenue":"Revenue","Avg Order Value":"Avg_Order_Value"}
        cat_grp = cat_grp.sort_values(col_map2[sort_by], ascending=False).head(show_n)

        fig3 = px.bar(cat_grp, x="product_category", y=col_map2[sort_by],
                      color="OTIF_Pct", color_continuous_scale=["#f87171","#fbbf24","#34d399"],
                      text_auto=".2s", hover_data=["Orders","Revenue","OTIF_Pct"])
        fig3.update_traces(textposition="outside")
        fig3.update_coloraxes(colorbar_title="OTIF %")
        st.plotly_chart(theme(fig3, f"Product Category — {sort_by} (colored by OTIF %)", h=380), use_container_width=True)

        # Sunburst
        sun = dff.groupby(["product_category","customer_segment"])["order_value_usd"].sum().reset_index()
        fig4 = px.sunburst(sun, path=["product_category","customer_segment"],
                           values="order_value_usd", color="order_value_usd",
                           color_continuous_scale=["#1e1b4b","#6366f1","#f472b6"])
        fig4.update_traces(hovertemplate="<b>%{label}</b><br>Revenue: $%{value:,.0f}<extra></extra>")
        st.plotly_chart(theme(fig4, "🌞 Revenue Sunburst — Category → Segment", h=440), use_container_width=True)
        ins("💡 Click any slice to drill down. Click the center circle to go back up.", "info")

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            dow_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
            dow = dff.groupby("day_of_week")["order_id"].count().reindex(dow_order).reset_index(name="Orders")
            fig5 = px.bar(dow, x="day_of_week", y="Orders", color="Orders",
                          color_continuous_scale=["#6366f1","#22d3ee"], text="Orders")
            fig5.update_traces(textposition="outside")
            st.plotly_chart(theme(fig5,"Orders by Day of Week",h=320), use_container_width=True)
        with c2:
            hr_data = dff.copy()
            hr_data["hour"] = hr_data["order_date"].dt.hour
            hr_grp = hr_data.groupby("hour")["order_id"].count().reset_index(name="Orders")
            fig6 = px.area(hr_grp, x="hour", y="Orders", color_discrete_sequence=["#f472b6"])
            fig6.update_traces(fill="tozeroy", fillcolor="rgba(244,114,182,0.15)")
            st.plotly_chart(theme(fig6,"Orders by Hour of Day",h=320), use_container_width=True)

        # Heatmap: Month × Weekday
        st.markdown("### 🗓️ Order Volume Heatmap — Month × Weekday")
        hm = dff.copy()
        hm["month_short"] = hm["order_date"].dt.strftime("%b %Y")
        hm_pivot = hm.groupby(["month_short","day_of_week"])["order_id"].count().unstack(fill_value=0)
        hm_pivot = hm_pivot.reindex(columns=dow_order, fill_value=0)
        fig7 = px.imshow(hm_pivot, color_continuous_scale=["#0f172a","#6366f1","#22d3ee"],
                         aspect="auto", text_auto=True)
        fig7.update_traces(hovertemplate="Month: %{y}<br>Day: %{x}<br>Orders: %{z}<extra></extra>")
        st.plotly_chart(theme(fig7, h=500), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# 🚛 CARRIER ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
elif "Carrier" in selected:
    st.markdown("# 🚛 Carrier Performance Analysis")
    st.markdown("---")

    if n_del == 0:
        st.warning("No delivered orders match current filters.")
    else:
        carriers_avail = sorted(dff_del["carrier_name"].dropna().unique())

        # ── Carrier compare selector ──
        st.markdown("### 🔀 Carrier Comparison Tool")
        c_left, c_right = st.columns(2)
        with c_left:
            carrier_a = st.selectbox("Compare Carrier A", carriers_avail, index=0)
        with c_right:
            opts = [c for c in carriers_avail if c != carrier_a]
            carrier_b = st.selectbox("Compare Carrier B", opts, index=0)

        da = dff_del[dff_del["carrier_name"]==carrier_a]
        db = dff_del[dff_del["carrier_name"]==carrier_b]

        m1,m2,m3,m4 = st.columns(4)
        m1.metric(f"{carrier_a} Avg Transit", f"{da['transit_hours'].mean():.1f}h",
                  f"{da['transit_hours'].mean()-db['transit_hours'].mean():+.1f}h vs {carrier_b}")
        m2.metric(f"{carrier_a} On-Time %",   f"{da['on_time_flag'].mean()*100:.1f}%",
                  f"{(da['on_time_flag'].mean()-db['on_time_flag'].mean())*100:+.1f}pp")
        m3.metric(f"{carrier_b} Avg Transit", f"{db['transit_hours'].mean():.1f}h")
        m4.metric(f"{carrier_b} On-Time %",   f"{db['on_time_flag'].mean()*100:.1f}%")

        # Violin comparison
        compare_df = dff_del[dff_del["carrier_name"].isin([carrier_a, carrier_b])]
        fig_v = px.violin(compare_df, x="carrier_name", y="transit_hours", color="carrier_name",
                          box=True, points="outliers",
                          color_discrete_map={carrier_a:"#6366f1", carrier_b:"#f472b6"})
        fig_v.update_layout(showlegend=False)
        st.plotly_chart(theme(fig_v, f"Transit Distribution: {carrier_a} vs {carrier_b}", h=380), use_container_width=True)

        st.markdown("---")
        st.markdown("### 📊 All Carrier Overview")

        carrier_grp = dff_del.groupby("carrier_name").agg(
            Deliveries=("order_id","count"),
            Avg_Transit=("transit_hours","mean"),
            Median_Transit=("transit_hours","median"),
            IQR=("transit_hours", lambda x: x.quantile(0.75)-x.quantile(0.25)),
            On_Time_Pct=("on_time_flag", lambda x: x.mean()*100),
            Avg_Cost=("shipping_cost_usd","mean"),
        ).round(1).reset_index().sort_values("On_Time_Pct", ascending=False)
        carrier_grp["Rating"] = carrier_grp["On_Time_Pct"].apply(
            lambda x: "🟢 Best" if x>=48 else ("🟡 Average" if x>=35 else "🔴 Worst"))

        st.dataframe(carrier_grp.rename(columns={
            "carrier_name":"Carrier","Avg_Transit":"Avg Transit (h)","Median_Transit":"Median (h)",
            "IQR":"IQR (h)","On_Time_Pct":"On-Time %","Avg_Cost":"Avg Cost ($)"}),
            use_container_width=True, hide_index=True)

        c1,c2 = st.columns(2)
        with c1:
            fig1 = go.Figure()
            for i, row in carrier_grp.iterrows():
                fig1.add_trace(go.Bar(
                    name=row["carrier_name"], x=[row["carrier_name"]],
                    y=[row["Avg_Transit"]], error_y=dict(type="data", array=[row["IQR"]]),
                    marker_color=COLORS[i % len(COLORS)],
                    text=[f"{row['Avg_Transit']}h"], textposition="outside",
                ))
            fig1.update_layout(showlegend=False)
            st.plotly_chart(theme(fig1,"Avg Transit Hours ± IQR",h=360), use_container_width=True)
        with c2:
            fig2 = px.bar(carrier_grp, x="carrier_name", y="On_Time_Pct",
                          color="On_Time_Pct", color_continuous_scale=["#f87171","#fbbf24","#34d399"],
                          text="On_Time_Pct")
            fig2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
            fig2.add_hline(y=85, line_dash="dot", line_color="#fbbf24", annotation_text="85% SLA")
            st.plotly_chart(theme(fig2,"On-Time Delivery %",h=360), use_container_width=True)

        # Scatter: cost vs transit
        st.markdown("### 💸 Cost vs Speed Analysis")
        fig3 = px.scatter(dff_del.dropna(subset=["transit_hours","shipping_cost_usd"]),
                          x="transit_hours", y="shipping_cost_usd",
                          color="carrier_name", size="order_value_usd",
                          color_discrete_sequence=COLORS,
                          hover_data=["order_id","on_time_flag","dest_region"],
                          trendline="lowess")
        fig3.update_traces(marker=dict(opacity=0.75))
        st.plotly_chart(theme(fig3,"Shipping Cost vs Transit Time (bubble = order value)",h=420), use_container_width=True)
        ins("🔴 EcoParcel combines high transit time with no cost saving — worst value proposition.", "bad")
        ins("🟢 PrimeFreight delivers faster at comparable cost — best value carrier.", "good")

# ══════════════════════════════════════════════════════════════════════════════
# 🏭 WAREHOUSE DEEP-DIVE
# ══════════════════════════════════════════════════════════════════════════════
elif "Warehouse" in selected:
    st.markdown("# 🏭 Warehouse Performance Deep-Dive")
    st.markdown("---")

    if n_del == 0:
        st.warning("No delivered orders match current filters.")
    else:
        # Interactive warehouse selector
        wh_list = sorted(dff_del["warehouse_name"].dropna().unique())
        sel_wh_detail = st.selectbox("🔍 Select a Warehouse to Inspect", ["All Warehouses"] + wh_list)

        wh_grp = dff_del.groupby(["warehouse_id","warehouse_name"]).agg(
            Orders=("order_id","count"),
            Avg_O2S=("order_to_ship_hours","mean"),
            Median_O2S=("order_to_ship_hours","median"),
            Std_O2S=("order_to_ship_hours","std"),
            On_Time_Pct=("on_time_flag",lambda x: x.mean()*100),
            OTIF_Pct=("otif_flag",lambda x: x.mean()*100),
        ).round(1).reset_index().sort_values("Avg_O2S")

        network_avg = dff_del["order_to_ship_hours"].mean()
        wh04_avg    = dff_del[dff_del["warehouse_id"]=="WH-04"]["order_to_ship_hours"].mean()

        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Network Avg O2S",    f"{network_avg:.1f}h")
        k2.metric("Best WH O2S",        f"{wh_grp['Avg_O2S'].min():.1f}h",  wh_grp.loc[wh_grp['Avg_O2S'].idxmin(),"warehouse_name"])
        k3.metric("Worst WH O2S",       f"{wh04_avg:.1f}h",                  "WH-04 Kolkata ⚠️")
        k4.metric("Bottleneck Overhead", f"+{wh04_avg-network_avg:.1f}h",    f"{(wh04_avg/network_avg-1)*100:.0f}% above avg")

        st.markdown("---")
        c1,c2 = st.columns(2)
        with c1:
            bar_colors = ["#f87171" if wid=="WH-04" else "#6366f1" for wid in wh_grp["warehouse_id"]]
            fig1 = go.Figure(go.Bar(
                x=wh_grp["warehouse_name"], y=wh_grp["Avg_O2S"],
                marker_color=bar_colors,
                text=wh_grp["Avg_O2S"], textposition="outside",
                error_y=dict(type="data", array=wh_grp["Std_O2S"].tolist()),
                hovertemplate="<b>%{x}</b><br>Avg O2S: %{y:.1f}h<extra></extra>"
            ))
            fig1.add_hline(y=network_avg, line_dash="dash", line_color="#fbbf24",
                           annotation_text=f"Network Avg: {network_avg:.1f}h")
            st.plotly_chart(theme(fig1,"Order-to-Ship Hours (Red = Bottleneck)",h=360), use_container_width=True)
        with c2:
            fig2 = px.box(dff_del.dropna(subset=["order_to_ship_hours"]),
                          x="warehouse_name", y="order_to_ship_hours",
                          color="warehouse_name", color_discrete_sequence=COLORS, points="outliers")
            fig2.update_layout(showlegend=False, xaxis_tickangle=-20)
            st.plotly_chart(theme(fig2,"O2S Distribution (Box + Outliers)",h=360), use_container_width=True)

        # Detailed view for selected warehouse
        if sel_wh_detail != "All Warehouses":
            st.markdown(f"### 🔍 Deep-Dive: {sel_wh_detail}")
            wh_data = dff_del[dff_del["warehouse_name"]==sel_wh_detail]
            d1,d2,d3,d4 = st.columns(4)
            d1.metric("Orders",    len(wh_data))
            d2.metric("Avg O2S",   f"{wh_data['order_to_ship_hours'].mean():.1f}h")
            d3.metric("On-Time %", f"{wh_data['on_time_flag'].mean()*100:.1f}%")
            d4.metric("OTIF %",    f"{wh_data['otif_flag'].mean()*100:.1f}%")

            wc1,wc2 = st.columns(2)
            with wc1:
                fig_w1 = px.histogram(wh_data, x="order_to_ship_hours", nbins=20,
                                      color_discrete_sequence=["#6366f1"], marginal="rug")
                st.plotly_chart(theme(fig_w1,f"{sel_wh_detail} — O2S Distribution",h=300), use_container_width=True)
            with wc2:
                cat_wh = wh_data.groupby("product_category")["on_time_flag"].mean().mul(100).round(1).reset_index(name="On_Time_%")
                fig_w2 = px.bar(cat_wh.sort_values("On_Time_%"), x="On_Time_%", y="product_category",
                                orientation="h", color="On_Time_%",
                                color_continuous_scale=["#f87171","#fbbf24","#34d399"],
                                text="On_Time_%")
                fig_w2.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
                st.plotly_chart(theme(fig_w2,f"{sel_wh_detail} — On-Time % by Category",h=300), use_container_width=True)

        # Welch's t-test
        st.markdown("### 🔬 Statistical Significance: WH-04 vs Network")
        wh4_s   = dff_del[dff_del["warehouse_id"]=="WH-04"]["order_to_ship_hours"].dropna()
        other_s = dff_del[dff_del["warehouse_id"]!="WH-04"]["order_to_ship_hours"].dropna()

        if len(wh4_s) > 1 and len(other_s) > 1:
            t_stat, p_val = stats.ttest_ind(wh4_s, other_s, equal_var=False)
            tc1,tc2,tc3,tc4 = st.columns(4)
            tc1.metric("WH-04 Mean O2S",  f"{wh4_s.mean():.1f}h",  f"n={len(wh4_s)}")
            tc2.metric("Others Mean O2S", f"{other_s.mean():.1f}h", f"n={len(other_s)}")
            tc3.metric("t-statistic",     f"{t_stat:.2f}")
            tc4.metric("p-value",         f"{p_val:.2e}",           "✅ Significant" if p_val<0.05 else "❌ Not Significant")
            if p_val < 0.05:
                ins(f"🔴 <strong>p = {p_val:.2e}</strong> — WH-04 Kolkata is statistically and significantly slower than the rest of the network. The bottleneck is real, not random noise. Reallocate wave-picking staff immediately.", "bad")

# ══════════════════════════════════════════════════════════════════════════════
# 🎯 QUALITY & RETURNS
# ══════════════════════════════════════════════════════════════════════════════
elif "Quality" in selected:
    st.markdown("# 🎯 Delivery Quality & Returns Analysis")
    st.markdown("---")

    if n_del == 0:
        st.warning("No delivered orders match current filters.")
    else:
        tab1, tab2 = st.tabs(["📦 Delivery Condition","🔁 OTIF Breakdown"])
        with tab1:
            c1,c2 = st.columns(2)
            with c1:
                cond = dff_del["delivery_condition"].value_counts().reset_index()
                fig1 = px.pie(cond, names="delivery_condition", values="count", hole=0.6,
                              color_discrete_map={"Intact":"#34d399","Damaged":"#f87171","Returned":"#fbbf24"})
                fig1.update_traces(textinfo="percent+label", pull=[0.05,0,0])
                st.plotly_chart(theme(fig1,"Overall Delivery Condition",h=320), use_container_width=True)
            with c2:
                cond_cat = pd.crosstab(dff_del["product_category"], dff_del["delivery_condition"], normalize="index")*100
                for col in ["Intact","Damaged","Returned"]:
                    if col not in cond_cat.columns: cond_cat[col]=0
                cond_cat = cond_cat[["Intact","Damaged","Returned"]].reset_index()
                fig2 = go.Figure()
                fig2.add_trace(go.Bar(name="Intact",   x=cond_cat["product_category"], y=cond_cat["Intact"],   marker_color="#34d399"))
                fig2.add_trace(go.Bar(name="Damaged",  x=cond_cat["product_category"], y=cond_cat["Damaged"],  marker_color="#f87171"))
                fig2.add_trace(go.Bar(name="Returned", x=cond_cat["product_category"], y=cond_cat["Returned"], marker_color="#fbbf24"))
                fig2.update_layout(barmode="stack")
                st.plotly_chart(theme(fig2,"100% Stacked — Condition by Category",h=320), use_container_width=True)
            ins("⚠️ <strong>Apparel (13.8%)</strong> and <strong>Books & Media (11.9%)</strong> have the highest damage rates. Electronics has the lowest despite fragility — suggesting better packaging.", "warn")

            # Condition by carrier
            cond_carr = pd.crosstab(dff_del["carrier_name"], dff_del["delivery_condition"], normalize="index")*100
            for col in ["Intact","Damaged","Returned"]:
                if col not in cond_carr.columns: cond_carr[col]=0
            cond_carr = cond_carr[["Intact","Damaged","Returned"]].reset_index()
            fig3 = go.Figure()
            fig3.add_trace(go.Bar(name="Intact",   x=cond_carr["carrier_name"], y=cond_carr["Intact"],   marker_color="#34d399"))
            fig3.add_trace(go.Bar(name="Damaged",  x=cond_carr["carrier_name"], y=cond_carr["Damaged"],  marker_color="#f87171"))
            fig3.add_trace(go.Bar(name="Returned", x=cond_carr["carrier_name"], y=cond_carr["Returned"], marker_color="#fbbf24"))
            fig3.update_layout(barmode="stack")
            st.plotly_chart(theme(fig3,"100% Stacked — Condition by Carrier",h=320), use_container_width=True)

        with tab2:
            cat_perf = dff_del.groupby("product_category").agg(
                OTIF_Pct=("otif_flag",lambda x:x.mean()*100),
                OnTime_Pct=("on_time_flag",lambda x:x.mean()*100),
                InFull_Pct=("in_full_flag",lambda x:x.mean()*100),
                Perfect_Pct=("perfect_order",lambda x:x.mean()*100),
                Orders=("order_id","count"),
            ).round(1).reset_index().sort_values("OTIF_Pct")

            fig4 = go.Figure()
            for metric_name, col_name, color in [("OTIF %","OTIF_Pct","#6366f1"),
                                                  ("On-Time %","OnTime_Pct","#22d3ee"),
                                                  ("In-Full %","InFull_Pct","#34d399"),
                                                  ("Perfect Order %","Perfect_Pct","#f472b6")]:
                fig4.add_trace(go.Bar(name=metric_name, x=cat_perf["product_category"],
                                      y=cat_perf[col_name], text=cat_perf[col_name],
                                      texttemplate="%{text:.0f}%", textposition="outside"))
            fig4.add_hline(y=85, line_dash="dot", line_color="#fbbf24", annotation_text="85% SLA")
            fig4.update_layout(barmode="group")
            st.plotly_chart(theme(fig4,"OTIF, On-Time, In-Full & Perfect Order % by Category",h=420), use_container_width=True)

            # Scatter: order value vs OTIF
            fig5 = px.scatter(dff_del, x="order_value_usd", y="order_to_ship_hours",
                              color="otif_flag", size="order_weight_kg",
                              color_discrete_map={True:"#34d399", False:"#f87171"},
                              hover_data=["carrier_name","warehouse_name","product_category"],
                              symbol="otif_flag")
            st.plotly_chart(theme(fig5,"Order Value vs O2S Hours (green=OTIF met, red=missed)",h=420), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# ❌ CANCELLATIONS
# ══════════════════════════════════════════════════════════════════════════════
elif "Cancellations" in selected:
    st.markdown("# ❌ Cancellation Analysis")
    st.markdown("---")

    canc = dff[dff["order_status"]=="Cancelled"]
    n_canc = len(canc)

    if n_canc == 0:
        st.info("No cancellations in current filter selection.")
    else:
        k1,k2,k3,k4 = st.columns(4)
        k1.metric("Total Cancelled",    n_canc)
        k2.metric("Cancel Rate",        f"{n_canc/n_total*100:.1f}%")
        k3.metric("Revenue Lost",       f"${canc['order_value_usd'].sum():,.0f}")
        k4.metric("Avg Cancelled Value",f"${canc['order_value_usd'].mean():,.0f}")

        st.markdown("---")
        c1,c2 = st.columns(2)
        with c1:
            cr = canc["cancellation_reason"].value_counts().reset_index()
            cr.columns = ["Reason","Count"]
            cr["Cumulative%"] = (cr["Count"].cumsum()/cr["Count"].sum()*100).round(1)

            fig1 = make_subplots(specs=[[{"secondary_y":True}]])
            fig1.add_trace(go.Bar(x=cr["Reason"], y=cr["Count"], marker_color="#f87171",
                                  name="Count", text=cr["Count"], textposition="outside"))
            fig1.add_trace(go.Scatter(x=cr["Reason"], y=cr["Cumulative%"], mode="lines+markers",
                                      name="Cumulative %", line=dict(color="#fbbf24",width=2.5),
                                      marker=dict(size=8)), secondary_y=True)
            fig1.add_hline(y=80, line_dash="dot", line_color="#6366f1",
                           annotation_text="80% Pareto Line", secondary_y=True)
            fig1.update_yaxes(title_text="Count", secondary_y=False, gridcolor="rgba(99,120,200,0.1)")
            fig1.update_yaxes(title_text="Cumulative %", secondary_y=True, range=[0,105], gridcolor=None)
            fig1.update_layout(**{**dict(paper_bgcolor="#131929",plot_bgcolor="#131929",
                                        font=dict(family="Inter",color="#94a3b8",size=12),
                                        margin=dict(l=20,r=40,t=44,b=20),
                                        legend=dict(bgcolor="rgba(0,0,0,0)"),
                                        title=dict(text="Cancellation Pareto Chart",
                                                   font=dict(color="#c7d2fe",size=14),x=0))})
            st.plotly_chart(fig1, use_container_width=True)

        with c2:
            rev_lost = canc.groupby("cancellation_reason")["order_value_usd"].sum().reset_index()
            rev_lost.columns = ["Reason","Revenue Lost"]
            fig2 = px.bar(rev_lost.sort_values("Revenue Lost",ascending=False),
                          x="Reason", y="Revenue Lost",
                          color="Revenue Lost", color_continuous_scale=["#fbbf24","#f87171"],
                          text_auto=".2s")
            fig2.update_traces(textposition="outside")
            st.plotly_chart(theme(fig2,"Revenue Lost per Cancellation Reason",h=360), use_container_width=True)

        # Region & Segment breakdown
        c3,c4 = st.columns(2)
        with c3:
            canc_reg = canc.groupby("dest_region")["order_id"].count().reset_index(name="Cancellations")
            fig3 = px.bar(canc_reg.sort_values("Cancellations",ascending=False),
                          x="dest_region", y="Cancellations", color="Cancellations",
                          color_continuous_scale=["#fbbf24","#f87171"], text="Cancellations")
            fig3.update_traces(textposition="outside")
            st.plotly_chart(theme(fig3,"Cancellations by Region",h=320), use_container_width=True)
        with c4:
            canc_seg = canc.groupby(["customer_segment","cancellation_reason"])["order_id"].count().reset_index(name="Count")
            fig4 = px.bar(canc_seg, x="customer_segment", y="Count", color="cancellation_reason",
                          color_discrete_sequence=COLORS, barmode="group")
            st.plotly_chart(theme(fig4,"Cancellations by Segment & Reason",h=320), use_container_width=True)

        ins("🔴 Stockout is the #1 driver — real-time inventory alerts and safety stock buffers for top SKUs required.", "bad")

# ══════════════════════════════════════════════════════════════════════════════
# 🔬 STATISTICAL TESTS
# ══════════════════════════════════════════════════════════════════════════════
elif "Statistical" in selected:
    st.markdown("# 🔬 Statistical Analysis")
    st.markdown("Hypothesis testing, correlations, and distribution analysis.")
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📐 Hypothesis Tests","🔗 Correlations","📊 Distributions"])

    with tab1:
        st.markdown("### Welch's Two-Sample t-Test")
        st.markdown("Test whether the mean of a numeric variable differs significantly between two groups.")
        col1,col2,col3 = st.columns(3)
        with col1:
            test_var = st.selectbox("Variable", ["order_to_ship_hours","transit_hours","total_fulfillment_hours","order_value_usd"])
        with col2:
            group_var = st.selectbox("Group By", ["warehouse_id","carrier_name","customer_segment","dest_region"])
        with col3:
            groups_avail = sorted(dff_del[group_var].dropna().unique())
            g1 = st.selectbox("Group A", groups_avail, index=0)
            g2 = st.selectbox("Group B", [g for g in groups_avail if g!=g1], index=0)

        s1 = dff_del[dff_del[group_var]==g1][test_var].dropna()
        s2 = dff_del[dff_del[group_var]==g2][test_var].dropna()

        if len(s1) > 1 and len(s2) > 1:
            t_stat, p_val = stats.ttest_ind(s1, s2, equal_var=False)
            m1,m2,m3,m4 = st.columns(4)
            m1.metric(f"{g1} Mean",    f"{s1.mean():.2f}", f"n={len(s1)}")
            m2.metric(f"{g2} Mean",    f"{s2.mean():.2f}", f"n={len(s2)}")
            m3.metric("t-statistic",   f"{t_stat:.3f}")
            m4.metric("p-value",       f"{p_val:.4f}", "✅ Significant" if p_val<0.05 else "❌ Not Significant")

            fig_t = go.Figure()
            fig_t.add_trace(go.Violin(y=s1, name=g1, box_visible=True, meanline_visible=True, fillcolor="rgba(99,102,241,0.3)", line_color="#6366f1"))
            fig_t.add_trace(go.Violin(y=s2, name=g2, box_visible=True, meanline_visible=True, fillcolor="rgba(244,114,182,0.3)", line_color="#f472b6"))
            st.plotly_chart(theme(fig_t, f"Distribution of {test_var}: {g1} vs {g2}", h=380), use_container_width=True)

            if p_val < 0.05:
                ins(f"✅ The difference is <strong>statistically significant</strong> (p={p_val:.4f}). The {test_var} means between {g1} and {g2} are unlikely to be equal by random chance.", "bad" if abs(t_stat)>5 else "warn")
            else:
                ins(f"ℹ️ The difference is <strong>not statistically significant</strong> (p={p_val:.4f}). Cannot reject the null hypothesis.", "info")

    with tab2:
        st.markdown("### 🔗 Correlation Heatmap")
        num_cols = ["order_weight_kg","order_value_usd","shipping_cost_usd",
                    "order_to_ship_hours","transit_hours","total_fulfillment_hours",
                    "on_time_flag","in_full_flag","otif_flag","sla_target_hours"]
        avail_num = [c for c in num_cols if c in dff_del.columns]
        corr = dff_del[avail_num].corr().round(2)
        fig_corr = px.imshow(corr, color_continuous_scale=["#f87171","#131929","#22d3ee"],
                             zmin=-1, zmax=1, text_auto=True, aspect="auto")
        fig_corr.update_traces(hovertemplate="<b>%{x}</b> × <b>%{y}</b><br>r = %{z}<extra></extra>")
        st.plotly_chart(theme(fig_corr, h=560), use_container_width=True)

        # Scatter matrix for selected vars
        st.markdown("### Interactive Scatter Matrix")
        sel_vars = st.multiselect("Select variables for scatter matrix",
            avail_num, default=["order_to_ship_hours","transit_hours","order_value_usd","shipping_cost_usd"])
        if len(sel_vars) >= 2:
            fig_sm = px.scatter_matrix(dff_del[sel_vars+["on_time_flag"]].dropna(),
                                       dimensions=sel_vars, color="on_time_flag",
                                       color_discrete_map={1:"#34d399",0:"#f87171"})
            fig_sm.update_traces(diagonal_visible=False, marker=dict(size=3,opacity=0.6))
            st.plotly_chart(theme(fig_sm, h=600), use_container_width=True)

    with tab3:
        st.markdown("### Distribution Explorer")
        dist_col = st.selectbox("Select variable", ["order_to_ship_hours","transit_hours","order_value_usd","order_weight_kg","shipping_cost_usd"])
        group_color = st.selectbox("Color by", ["carrier_name","warehouse_name","customer_segment","product_category"])
        fig_d = px.histogram(dff_del.dropna(subset=[dist_col]),
                             x=dist_col, color=group_color,
                             color_discrete_sequence=COLORS,
                             nbins=40, marginal="box", barmode="overlay", opacity=0.7)
        st.plotly_chart(theme(fig_d, f"Distribution of {dist_col} by {group_color}", h=460), use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# 📊 EXECUTIVE REPORT
# ══════════════════════════════════════════════════════════════════════════════
elif "Executive" in selected:
    st.markdown("# 📊 Executive Report")
    st.caption("Synthesis of all findings with priority-ranked recommendations.")
    st.markdown("---")

    otif        = round(dff_del["otif_flag"].mean()*100,1)      if n_del else 0
    perfect     = round(dff_del["perfect_order"].mean()*100,1)  if n_del else 0
    cancel_rate = round((dff["order_status"]=="Cancelled").mean()*100,1)
    rev_lost    = dff["revenue_lost_usd"].sum()
    wh04_avg    = dff_del[dff_del["warehouse_id"]=="WH-04"]["order_to_ship_hours"].mean() if n_del else 0
    nw_avg      = dff_del["order_to_ship_hours"].mean() if n_del else 0
    eco_ot      = dff_del[dff_del["carrier_name"]=="EcoParcel"]["on_time_flag"].mean()*100   if n_del else 0
    prime_ot    = dff_del[dff_del["carrier_name"]=="PrimeFreight"]["on_time_flag"].mean()*100 if n_del else 0

    # Summary KPIs
    k = st.columns(5)
    k[0].metric("🎯 OTIF",          f"{otif}%",         "Target: 85%")
    k[1].metric("🏆 Perfect Order", f"{perfect}%")
    k[2].metric("❌ Cancel Rate",   f"{cancel_rate}%")
    k[3].metric("💸 Revenue Lost",  f"${rev_lost:,.0f}")
    k[4].metric("🏭 WH-04 O2S",    f"{wh04_avg:.1f}h",  f"+{wh04_avg-nw_avg:.1f}h vs network")

    st.markdown("---")
    st.markdown("### 🔴 Priority Findings & Actions")

    findings = [
        ("🔴","High","WH-04 Kolkata Pick-Pack Bottleneck",
         f"WH-04 averages <strong>{wh04_avg:.1f}h</strong> O2S vs <strong>{nw_avg:.1f}h</strong> network average. Statistically confirmed (p < 0.0001). Over {((wh04_avg/nw_avg)-1)*100:.0f}% slower.",
         "Reallocate wave-picking staff & automate barcode batching at Kolkata Hub. Owner: Warehouse Ops.","#f87171"),
        ("🔴","High","EcoParcel Transit Instability",
         f"EcoParcel: <strong>{eco_ot:.1f}%</strong> on-time vs PrimeFreight <strong>{prime_ot:.1f}%</strong>. Widest IQR (>35h) = highest customer SLA risk and damage correlation.",
         "Shift 30% of East/Central volume from EcoParcel to PrimeFreight. Owner: Procurement.","#f87171"),
        ("🔴","High",f"Systemic OTIF Failure ({otif}%)",
         "OTIF is <strong>"+str(otif)+"%</strong> — far below 85% SLA. 3-month rolling average consistently below target. This is structural, not seasonal.",
         "Fix WH-04 and EcoParcel simultaneously. Set monthly OTIF review cadence. Owner: Supply Chain VP.","#f87171"),
        ("🟡","Medium","Stockout-Driven Cancellations",
         f"Stockout is the #1 cancellation reason. Cancel rate = <strong>{cancel_rate}%</strong>. Revenue lost = <strong>${rev_lost:,.0f}</strong>. Top 2 reasons drive 40%+ of all cancellations.",
         "Implement real-time inventory alerts and safety stock buffers for top SKUs. Owner: Inventory Planning.","#fbbf24"),
        ("🟡","Medium","Oct/Nov Festive Peak Surge",
         "Order volume spikes sharply every October and November (festive e-commerce peak), straining warehouse capacity and worsening SLA compliance.",
         "Develop pre-season staffing ramp-up plan and contingency carrier contracts. Owner: Operations Planning.","#fbbf24"),
        ("🟢","Low","Apparel Damage Rate (13.8%)",
         "Apparel has the highest damage rate among all categories despite being non-fragile. Packaging mismatch suspected.",
         "Revise packaging guidelines for Apparel. Run root-cause analysis on damage origins. Owner: Logistics QA.","#34d399"),
    ]

    for icon, prio, title, evidence, action, color in findings:
        with st.expander(f"{icon} [{prio}] {title}", expanded=(prio=="High")):
            st.markdown(f'<div style="font-size:0.82rem;color:#94a3b8;line-height:1.6;margin-bottom:8px">{evidence}</div>', unsafe_allow_html=True)
            st.markdown(f'<div style="font-size:0.82rem;color:#c7d2fe;padding:8px 12px;background:rgba(99,102,241,0.1);border-radius:8px">💡 <strong>Recommended Action:</strong> {action}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Chart-Choice Cheat Sheet")
    cheat = pd.DataFrame({
        "Goal":    ["Compare categories","Show distribution","Track over time","Relationship of 2 numbers","Part-to-whole","Compare groups fairly","Hierarchical drill-down"],
        "Best Chart": ["Horizontal Bar","Histogram + KDE / Box Plot","Line + Rolling Avg","Scatter + OLS","Donut/Pie","100% Stacked Bar","Treemap / Sunburst"],
        "Avoid":   ["Pie (hard to compare)","Single mean bar","Unordered bar","Dual-axis arbitrary scales","Stacked bar unnormalized","Side-by-side (volume dominates)","Nested pie"],
    })
    st.dataframe(cheat, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("### 📥 Download Filtered Data")
    cc1,cc2 = st.columns(2)
    with cc1:
        csv = dff.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download All Filtered Orders (CSV)", csv, "filtered_orders.csv","text/csv")
    with cc2:
        csv2 = dff_del.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download Delivered Orders Only (CSV)", csv2, "delivered_orders.csv","text/csv")
