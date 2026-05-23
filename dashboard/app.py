"""
UPI Fraud Detector – Streamlit Dashboard
Run: streamlit run dashboard/app.py   (from project root)
  OR: cd dashboard && streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
from pathlib import Path
from sklearn.preprocessing import LabelEncoder

# ── Paths ──────────────────────────────────────────────────
DASHBOARD_DIR = Path(__file__).resolve().parent
BASE_DIR      = DASHBOARD_DIR.parent
DATA_PATH     = BASE_DIR / "data"   / "upi_transactions.csv"
MODEL_PATH    = BASE_DIR / "src"    / "rf_model.pkl"
FEATURES_PATH = BASE_DIR / "src"    / "features.pkl"
METRICS_PATH  = BASE_DIR / "outputs"/ "model_metrics.csv"

# ── Page config ────────────────────────────────────────────
st.set_page_config(page_title="UPI Fraud Detector", page_icon="🔍",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    .main, .stApp { background-color: #0f0f1a; }
    h1, h2, h3 { color: #7b5ea7 !important; }
    .fraud-alert {
        background: linear-gradient(135deg,#4a0010,#800020);
        border: 2px solid #ff4d6d; border-radius:12px;
        padding:20px; text-align:center;
    }
    .safe-alert {
        background: linear-gradient(135deg,#003a2f,#005f4b);
        border: 2px solid #00c9a7; border-radius:12px;
        padding:20px; text-align:center;
    }
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg,#1a1a2e,#16213e);
        border: 1px solid #444466; border-radius:10px; padding:15px;
    }
</style>""", unsafe_allow_html=True)

# ── Load data & model ──────────────────────────────────────
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH, parse_dates=["timestamp"])

@st.cache_resource
def load_model():
    with open(MODEL_PATH,    "rb") as f: model    = pickle.load(f)
    with open(FEATURES_PATH, "rb") as f: features = pickle.load(f)
    return model, features

df = load_data()
model, FEATURES = load_model()

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 UPI Fraud Detector")
    st.markdown("---")
    page = st.radio("Navigate", ["📊 Dashboard", "🔮 Predict Transaction", "📁 Raw Data"])
    st.markdown("---")
    st.markdown(f"**Total Transactions:** {len(df):,}")
    st.markdown(f"**Fraud Cases:** {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.1f}%)")
    st.markdown("**Date Range:** Jan – Dec 2023")

# ═══════════════════════════════════════════════════════════
# PAGE 1 – DASHBOARD
# ═══════════════════════════════════════════════════════════
if page == "📊 Dashboard":
    st.markdown("# 📊 UPI Transaction Fraud Analysis Dashboard")
    st.markdown("Real-time monitoring of UPI transaction patterns and fraud detection")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("Total Transactions", f"{len(df):,}")
    with col2: st.metric("Fraud Detected",     f"{df['is_fraud'].sum():,}", delta=f"{df['is_fraud'].mean()*100:.1f}% rate", delta_color="inverse")
    with col3: st.metric("Avg Normal Amount",  f"₹{df[df.is_fraud==0]['amount'].mean():,.0f}")
    with col4: st.metric("Avg Fraud Amount",   f"₹{df[df.is_fraud==1]['amount'].mean():,.0f}", delta_color="inverse")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("💰 Amount Distribution")
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df[df.is_fraud==0]["amount"], name="Normal",
                                   marker_color="#00c9a7", opacity=0.7, nbinsx=60))
        fig.add_trace(go.Histogram(x=df[df.is_fraud==1]["amount"], name="Fraud",
                                   marker_color="#ff4d6d", opacity=0.8, nbinsx=40))
        fig.update_layout(barmode="overlay", paper_bgcolor="#1a1a2e",
                          plot_bgcolor="#1a1a2e", font_color="#ccccee")
        fig.add_vline(x=15000, line_dash="dash", line_color="yellow",
                      annotation_text="₹15k threshold")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🕐 Fraud by Hour of Day")
        hourly = df.groupby(["hour","is_fraud"]).size().reset_index(name="count")
        fig = px.line(hourly, x="hour", y="count", color="is_fraud",
                      color_discrete_map={0:"#00c9a7",1:"#ff4d6d"}, markers=True)
        fig.update_layout(paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e", font_color="#ccccee")
        fig.add_vrect(x0=0,  x1=6,  fillcolor="red", opacity=0.08, annotation_text="Risk Zone")
        fig.add_vrect(x0=22, x1=24, fillcolor="red", opacity=0.08)
        st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("🏙️ Fraud Rate by City")
        city_fraud = df.groupby("sender_city")["is_fraud"].mean().reset_index()
        city_fraud.columns = ["City","Fraud Rate"]
        city_fraud["Fraud Rate %"] = city_fraud["Fraud Rate"] * 100
        city_fraud = city_fraud.sort_values("Fraud Rate %", ascending=False)
        fig = px.bar(city_fraud, x="City", y="Fraud Rate %",
                     color="Fraud Rate %", color_continuous_scale="RdYlGn_r")
        fig.update_layout(paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e", font_color="#ccccee")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("📅 Monthly Fraud Trend")
        monthly = df.groupby("month").agg(total=("is_fraud","count"),fraud=("is_fraud","sum")).reset_index()
        monthly["fraud_rate"] = monthly["fraud"] / monthly["total"] * 100
        monthly["month_name"] = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
        fig = go.Figure()
        fig.add_bar(x=monthly["month_name"], y=monthly["total"], name="Total", marker_color="#7b5ea7", opacity=0.5)
        fig.add_bar(x=monthly["month_name"], y=monthly["fraud"],  name="Fraud", marker_color="#ff4d6d")
        fig.add_trace(go.Scatter(x=monthly["month_name"], y=monthly["fraud_rate"],
                                 name="Fraud Rate %", yaxis="y2",
                                 line=dict(color="yellow", width=2.5), mode="lines+markers"))
        fig.update_layout(barmode="overlay",
                          yaxis2=dict(overlaying="y", side="right", color="yellow"),
                          paper_bgcolor="#1a1a2e", plot_bgcolor="#1a1a2e", font_color="#ccccee")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🏆 Model Performance")
    try:
        metrics_df = pd.read_csv(METRICS_PATH).round(4)
        st.dataframe(metrics_df, use_container_width=True)
    except:
        st.info("Run src/model.py first to generate metrics.")

# ═══════════════════════════════════════════════════════════
# PAGE 2 – PREDICT
# ═══════════════════════════════════════════════════════════
elif page == "🔮 Predict Transaction":
    st.markdown("# 🔮 Real-Time Transaction Fraud Predictor")
    st.markdown("---")

    CITIES    = ["Mumbai","Delhi","Bengaluru","Hyderabad","Chennai",
                 "Pune","Kolkata","Ahmedabad","Jaipur","Lucknow"]
    BANKS     = ["SBI","HDFC","ICICI","Axis","Kotak","PNB","BOB","Yes Bank"]
    MERCHANTS = ["Swiggy","Zomato","Amazon","Flipkart","PhonePe","Paytm",
                 "BigBasket","Myntra","BookMyShow","MakeMyTrip",
                 "Individual","Electricity Board","LIC","Rent Payment"]
    TX_TYPES  = ["P2P","P2M","Bill Payment","Recharge","Investment"]

    col1, col2, col3 = st.columns(3)
    with col1:
        amount          = st.number_input("💰 Amount (₹)", min_value=1.0, max_value=500000.0, value=5000.0, step=100.0)
        sender_city     = st.selectbox("📍 Sender City",   CITIES)
        receiver_city   = st.selectbox("📍 Receiver City", CITIES)
        sender_bank     = st.selectbox("🏦 Sender Bank",   BANKS)
    with col2:
        receiver_bank     = st.selectbox("🏦 Receiver Bank",     BANKS)
        merchant_category = st.selectbox("🏪 Merchant Category", MERCHANTS)
        transaction_type  = st.selectbox("📋 Transaction Type",  TX_TYPES)
        device_type       = st.selectbox("📱 Device",            ["Android","iOS"])
    with col3:
        hour              = st.slider("🕐 Hour",           0, 23, 14)
        month             = st.slider("📅 Month",          1, 12,  6)
        is_new_recipient  = st.radio("👤 New Recipient?",  [0,1], format_func=lambda x: "Yes" if x else "No")
        failed_attempts   = st.slider("❌ Failed Attempts",0,  5,  0)

    if st.button("🔍 ANALYZE TRANSACTION", use_container_width=True, type="primary"):
        all_vals = {
            "sender_city": CITIES, "receiver_city": CITIES,
            "sender_bank": BANKS,  "receiver_bank": BANKS,
            "merchant_category": MERCHANTS,
            "transaction_type":  TX_TYPES,
            "device_type": ["Android","iOS"],
        }
        def encode(col, val):
            le = LabelEncoder()
            le.fit(all_vals[col])
            return le.transform([val])[0]

        is_odd_hour = 1 if hour < 6 or hour >= 22 else 0
        cross_city  = 1 if sender_city != receiver_city else 0

        sample = pd.DataFrame([{
            "amount": amount, "hour": hour, "month": month,
            "is_new_recipient": is_new_recipient, "failed_attempts": failed_attempts,
            "is_odd_hour": is_odd_hour, "cross_city": cross_city,
            "sender_city":       encode("sender_city",       sender_city),
            "receiver_city":     encode("receiver_city",     receiver_city),
            "sender_bank":       encode("sender_bank",       sender_bank),
            "receiver_bank":     encode("receiver_bank",     receiver_bank),
            "merchant_category": encode("merchant_category", merchant_category),
            "transaction_type":  encode("transaction_type",  transaction_type),
            "device_type":       encode("device_type",       device_type),
        }])

        pred  = model.predict(sample)[0]
        proba = model.predict_proba(sample)[0][1]

        st.markdown("---")
        _, col, _ = st.columns([1,2,1])
        with col:
            if pred == 1:
                st.markdown(f"""<div class="fraud-alert">
                    <h1>🚨 FRAUD DETECTED</h1>
                    <h2>Probability: {proba*100:.1f}%</h2>
                    <p>This transaction looks suspicious. Please verify.</p>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="safe-alert">
                    <h1>✅ TRANSACTION SAFE</h1>
                    <h2>Fraud Probability: {proba*100:.1f}%</h2>
                    <p>This transaction appears legitimate.</p>
                </div>""", unsafe_allow_html=True)

        st.markdown("---")
        st.subheader("📊 Risk Factors Detected")
        risks = []
        if amount > 15000:             risks.append("🔴 **High Amount** — ₹{:,.0f} exceeds ₹15k threshold".format(amount))
        if is_odd_hour:                risks.append(f"🔴 **Odd Hour** — Transaction at {hour}:00 (midnight risk zone)")
        if is_new_recipient:           risks.append("🟡 **New Recipient** — First-time transfer to this person")
        if failed_attempts > 0:        risks.append(f"🟡 **Failed Attempts** — {failed_attempts} failed attempt(s)")
        if cross_city:                 risks.append(f"🟡 **Cross-City** — {sender_city} → {receiver_city}")
        if merchant_category=="Individual": risks.append("🟡 **P2P Transfer** — Individual transfers carry higher risk")
        for r in risks: st.markdown(r)
        if not risks: st.success("✅ No significant risk factors detected.")

# ═══════════════════════════════════════════════════════════
# PAGE 3 – RAW DATA
# ═══════════════════════════════════════════════════════════
elif page == "📁 Raw Data":
    st.markdown("# 📁 Raw Transaction Data")
    col1, col2, col3 = st.columns(3)
    with col1: fraud_filter = st.selectbox("Filter", ["All","Fraud Only","Normal Only"])
    with col2: city_filter  = st.selectbox("City",   ["All"] + sorted(df["sender_city"].unique().tolist()))
    with col3: amount_max   = st.number_input("Max Amount", value=int(df["amount"].max()), step=1000)

    filtered = df.copy()
    if fraud_filter == "Fraud Only":  filtered = filtered[filtered.is_fraud==1]
    if fraud_filter == "Normal Only": filtered = filtered[filtered.is_fraud==0]
    if city_filter  != "All":         filtered = filtered[filtered.sender_city==city_filter]
    filtered = filtered[filtered.amount <= amount_max]

    st.markdown(f"Showing **{len(filtered):,}** transactions")
    show_cols = ["transaction_id","timestamp","amount","sender_city","receiver_city",
                 "merchant_category","transaction_type","failed_attempts","is_fraud"]
    st.dataframe(filtered[show_cols].head(500), use_container_width=True)
    st.download_button("⬇️ Download CSV",
                       data=filtered.to_csv(index=False).encode("utf-8"),
                       file_name="filtered_transactions.csv", mime="text/csv")
