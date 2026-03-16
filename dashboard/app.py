"""
dashboard/app.py
----------------
Streamlit dashboard for the AI Financial Transaction Anomaly Detection System.

Run:
    streamlit run dashboard/app.py
"""

import os
import random
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Path setup ───────────────────────────────────────────────────────────────
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from anomaly_detection import AnomalyDetector          # noqa: E402  # type: ignore
from data_preprocessing import DataPreprocessor        # noqa: E402  # type: ignore
from feature_engineering import FeatureEngineer        # noqa: E402  # type: ignore

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Financial Anomaly Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
        .main-header {
            font-size: 2.4rem;
            font-weight: 800;
            color: #1a3c6e;
            text-align: center;
            padding: 0.6rem 0 0.2rem 0;
            letter-spacing: -0.5px;
        }
        .sub-header {
            text-align: center;
            color: #555;
            margin-bottom: 1.2rem;
            font-size: 1rem;
        }
        div[data-testid="metric-container"] {
            background: #f7f9fc;
            border-radius: 10px;
            padding: 0.6rem 0.8rem;
            border: 1px solid #e1e8f0;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────────────────────────────────────
# Helper: generate synthetic sample data (used when no CSV is uploaded)
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data
def _generate_sample_data(n_normal: int = 600, n_anomalous: int = 30) -> pd.DataFrame:
    """Return a small demo dataset without touching the filesystem."""
    np.random.seed(42)
    random.seed(42)

    TX_TYPES    = ["purchase", "transfer", "withdrawal", "deposit", "payment"]
    MERCHANTS   = ["grocery", "electronics", "restaurant", "gas_station",
                   "clothing", "healthcare", "entertainment", "travel", "online"]
    LOCATIONS   = ["New York", "Los Angeles", "Chicago", "Houston", "Phoenix",
                   "Philadelphia", "San Antonio", "San Diego", "Dallas", "San Jose"]
    base        = datetime(2024, 1, 1)

    rows: list[dict] = []

    for i in range(1, n_normal + 1):
        cid    = f"CUST_{np.random.randint(1, 151):04d}"
        amount = round(min(abs(np.random.normal(150, 80)), 800), 2)
        ts     = base + timedelta(days=np.random.randint(0, 365),
                                  hours=np.random.randint(8, 22),
                                  minutes=np.random.randint(0, 60))
        rows.append(dict(
            transaction_id=f"TXN_{i:06d}",
            customer_id=cid,
            transaction_amount=amount,
            transaction_time=ts.strftime("%Y-%m-%d %H:%M:%S"),
            transaction_type=random.choice(TX_TYPES[:4]),
            merchant_category=random.choice(MERCHANTS[:6]),
            location=random.choice(LOCATIONS),
            account_balance=round(np.random.uniform(500, 15_000), 2),
        ))

    anomaly_patterns = ["high_amount", "odd_hours", "foreign"]
    for j in range(1, n_anomalous + 1):
        cid     = f"CUST_{np.random.randint(1, 151):04d}"
        pattern = random.choice(anomaly_patterns)
        if pattern == "high_amount":
            amount, hour, loc = round(np.random.uniform(5_000, 30_000), 2), np.random.randint(8, 22),  random.choice(LOCATIONS)
        elif pattern == "odd_hours":
            amount, hour, loc = round(np.random.uniform(200, 2_000), 2),   np.random.randint(0, 5),    random.choice(LOCATIONS)
        else:
            amount, hour, loc = round(np.random.uniform(1_000, 8_000), 2), np.random.randint(0, 24), "Foreign"

        ts = base + timedelta(days=np.random.randint(0, 365), hours=hour, minutes=np.random.randint(0, 60))
        rows.append(dict(
            transaction_id=f"TXN_{n_normal + j:06d}",
            customer_id=cid,
            transaction_amount=amount,
            transaction_time=ts.strftime("%Y-%m-%d %H:%M:%S"),
            transaction_type=random.choice(TX_TYPES),
            merchant_category=random.choice(MERCHANTS),
            location=loc,
            account_balance=round(np.random.uniform(0, 500), 2),
        ))

    return pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# Core detection pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_detection(df_raw: pd.DataFrame, model_type: str, contamination: float) -> pd.DataFrame:
    """
    Full pipeline: preprocess → engineer features → fit model → detect.
    Returns a DataFrame with 'risk_score' and 'is_anomaly_detected' columns.
    """
    preprocessor = DataPreprocessor()
    df = preprocessor.preprocess(df_raw, fit=True)

    engineer = FeatureEngineer()
    df = engineer.engineer_features(df)

    feature_cols      = engineer.get_feature_columns()
    available_features = [c for c in feature_cols if c in df.columns]

    detector = AnomalyDetector(model_type=model_type, contamination=contamination)
    detector.fit(df[available_features].fillna(0).values)

    df = detector.detect(df, available_features)

    # Preserve original readable columns
    keep_orig = ["transaction_id", "customer_id", "transaction_amount",
                 "transaction_time", "transaction_type", "merchant_category",
                 "location", "account_balance"]
    keep_orig = [c for c in keep_orig if c in df.columns]
    keep_eng  = ["hour", "day_of_week", "spending_deviation",
                 "amount_to_balance_ratio", "tx_frequency",
                 "risk_score", "is_anomaly_detected"]
    keep_eng  = [c for c in keep_eng if c in df.columns]

    return df[keep_orig + keep_eng].copy()


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Header ───────────────────────────────────────────────────────────
    st.markdown(
        '<div class="main-header">🔍 AI Financial Anomaly Detector</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">Machine-learning powered detection of suspicious transactions</p>',
        unsafe_allow_html=True,
    )

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("⚙️ Configuration")
        st.markdown("---")

        st.subheader("Model Selection")
        model_type = st.selectbox(
            "Detection Algorithm",
            options=["isolation_forest", "one_class_svm", "local_outlier_factor"],
            format_func=lambda x: {
                "isolation_forest":    "🌲 Isolation Forest",
                "one_class_svm":       "🔵 One-Class SVM",
                "local_outlier_factor":"📊 Local Outlier Factor",
            }[x],
        )

        st.markdown("---")
        st.subheader("Sensitivity")
        contamination = st.slider(
            "Contamination Rate",
            min_value=0.01, max_value=0.20, value=0.05, step=0.01,
            help="Expected fraction of anomalies in the dataset.",
        )
        st.caption(f"Expected anomalies ≈ **{contamination * 100:.0f}%** of all transactions")

        st.markdown("---")
        st.subheader("Risk Threshold")
        risk_threshold = st.slider("Flag as high-risk when score ≥", 40, 95, 70)

        st.markdown("---")
        st.markdown(
            """
            **Algorithms at a glance**
            - **Isolation Forest** — builds random trees; rare points need fewer splits to isolate.
            - **One-Class SVM** — learns a tight boundary around normal data; points outside = anomalous.
            - **LOF** — compares local density; low-density points relative to neighbours = anomalous.
            """
        )

    # ── Tabs ─────────────────────────────────────────────────────────────
    tab_upload, tab_overview, tab_anomalies, tab_charts, tab_report = st.tabs([
        "📁 Data Upload",
        "📊 Overview",
        "🚨 Suspicious Transactions",
        "📈 Charts",
        "📋 Report",
    ])

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 1 – Data Upload
    # ═══════════════════════════════════════════════════════════════════════
    with tab_upload:
        st.header("Load Transaction Data")

        col_left, col_right = st.columns([3, 2])

        with col_left:
            uploaded = st.file_uploader(
                "Upload a transaction CSV file",
                type=["csv"],
                help="File must contain the 8 required columns listed below.",
            )
            with st.expander("📋 Required Column Schema", expanded=False):
                schema = {
                    "transaction_id":     "Unique identifier for each transaction",
                    "customer_id":        "Customer identifier",
                    "transaction_amount": "Dollar amount (float)",
                    "transaction_time":   "Timestamp — YYYY-MM-DD HH:MM:SS",
                    "transaction_type":   "purchase / transfer / withdrawal / deposit / payment",
                    "merchant_category":  "grocery / electronics / restaurant / …",
                    "location":           "City name or 'Foreign'",
                    "account_balance":    "Balance at time of transaction (float)",
                }
                st.table(pd.DataFrame(schema.items(), columns=["Column", "Description"]))

        with col_right:
            st.info(
                "**No data?**\n\n"
                "Generate a sample dataset by running:\n\n"
                "```\npython data/generate_data.py\n```\n\n"
                "Then upload `data/transactions.csv`, "
                "or use the built-in demo data below."
            )

        use_demo = st.checkbox("✅ Use built-in demo data (630 transactions, ~30 anomalies)")

        df_raw: pd.DataFrame | None = None

        if use_demo:
            df_raw = _generate_sample_data()
            st.success(f"Demo data loaded — {len(df_raw):,} transactions.")
        elif uploaded is not None:
            df_raw = pd.read_csv(uploaded)
            st.success(f"File uploaded — {len(df_raw):,} rows, {len(df_raw.columns)} columns.")

        if df_raw is not None:
            st.dataframe(df_raw.head(8), use_container_width=True)

            # Data quality summary
            st.markdown("#### Data Quality")
            q1, q2, q3, q4 = st.columns(4)
            q1.metric("Rows",            f"{len(df_raw):,}")
            q2.metric("Columns",         len(df_raw.columns))
            q3.metric("Missing values",  int(df_raw.isnull().sum().sum()))
            q4.metric("Duplicate IDs",   int(df_raw.duplicated(subset=["transaction_id"]).sum())
                                         if "transaction_id" in df_raw.columns else "N/A")

            st.markdown("---")
            if st.button("🚀 Run Anomaly Detection", type="primary", use_container_width=True):
                with st.spinner(f"Running {model_type.replace('_', ' ').title()} …"):
                    try:
                        result = run_detection(df_raw, model_type, contamination)
                        st.session_state["result"]    = result
                        st.session_state["model_type"] = model_type
                        st.success("✅ Detection complete! Explore the other tabs.")
                        st.balloons()
                    except Exception as exc:
                        st.error(f"❌ Detection failed: {exc}")
                        st.exception(exc)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 2 – Overview
    # ═══════════════════════════════════════════════════════════════════════
    with tab_overview:
        st.header("Transaction Overview")
        if "result" not in st.session_state:
            st.info("Upload data and run detection first (Tab 1).")
        else:
            df = st.session_state["result"]
            total    = len(df)
            n_anom   = int(df["is_anomaly_detected"].sum())
            n_norm   = total - n_anom
            avg_risk = df["risk_score"].mean()
            hi_risk  = int((df["risk_score"] >= risk_threshold).sum())

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Total Transactions", f"{total:,}")
            c2.metric("Normal",             f"{n_norm:,}",  f"{n_norm/total*100:.1f}%")
            c3.metric("Anomalies",          f"{n_anom:,}",  f"−{n_anom/total*100:.1f}%", delta_color="inverse")
            c4.metric("Avg Risk Score",     f"{avg_risk:.1f}")
            c5.metric(f"High Risk (≥ {risk_threshold})", f"{hi_risk:,}")

            st.markdown("---")
            col_a, col_b = st.columns(2)

            with col_a:
                if "transaction_type" in df.columns:
                    st.subheader("By Transaction Type")
                    summary = (
                        df.groupby("transaction_type")
                        .agg(Total=("is_anomaly_detected", "count"),
                             Anomalies=("is_anomaly_detected", "sum"),
                             Avg_Risk=("risk_score", "mean"))
                        .rename(columns={"Avg_Risk": "Avg Risk"})
                        .assign(Normal=lambda d: d["Total"] - d["Anomalies"])
                        .assign(**{"Anomaly %": lambda d: (d["Anomalies"] / d["Total"] * 100).round(1)})
                        .reset_index()
                    )
                    st.dataframe(summary, use_container_width=True, hide_index=True)

            with col_b:
                if "merchant_category" in df.columns:
                    st.subheader("Avg Risk by Merchant Category")
                    cat_risk = (
                        df.groupby("merchant_category")["risk_score"]
                        .mean()
                        .sort_values(ascending=False)
                        .reset_index()
                        .rename(columns={"risk_score": "Avg Risk Score"})
                    )
                    fig = px.bar(cat_risk, x="merchant_category", y="Avg Risk Score",
                                 color="Avg Risk Score", color_continuous_scale="RdYlGn_r",
                                 labels={"merchant_category": "Category"})
                    fig.update_layout(height=360, margin=dict(t=20, b=20))
                    st.plotly_chart(fig, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 3 – Suspicious Transactions
    # ═══════════════════════════════════════════════════════════════════════
    with tab_anomalies:
        st.header("🚨 Suspicious Transactions")
        if "result" not in st.session_state:
            st.info("Upload data and run detection first (Tab 1).")
        else:
            df   = st.session_state["result"]
            anom = df[df["is_anomaly_detected"] == 1].sort_values("risk_score", ascending=False)

            st.error(f"**{len(anom):,} suspicious transaction(s) detected**")

            # Filters
            f1, f2, f3 = st.columns(3)
            min_risk = f1.slider("Min Risk Score", 0, 100, 0)
            sel_type = f2.selectbox(
                "Transaction Type",
                ["All"] + (sorted(df["transaction_type"].unique().tolist()) if "transaction_type" in df.columns else []),
            )
            sel_loc = f3.selectbox(
                "Location",
                ["All"] + (sorted(df["location"].unique().tolist()) if "location" in df.columns else []),
            )

            mask = anom["risk_score"] >= min_risk
            if sel_type != "All" and "transaction_type" in anom.columns:
                mask &= anom["transaction_type"] == sel_type
            if sel_loc != "All" and "location" in anom.columns:
                mask &= anom["location"] == sel_loc
            filtered = anom[mask]

            st.markdown(f"**Showing {len(filtered):,} transaction(s)**")

            disp_cols = [c for c in ["transaction_id", "customer_id", "transaction_amount",
                                     "transaction_time", "transaction_type", "location",
                                     "account_balance", "risk_score"] if c in filtered.columns]

            def _risk_color(val):
                if val >= 80:
                    return "background-color: #ff4c4c; color: white"
                if val >= 60:
                    return "background-color: #ff9a00; color: white"
                return "background-color: #ffd200; color: black"

            styled = (
                filtered[disp_cols]
                .style
                .applymap(_risk_color, subset=["risk_score"])
                .format({"transaction_amount": "${:.2f}",
                         "account_balance":    "${:.2f}",
                         "risk_score":         "{:.1f}"})
            )
            st.dataframe(styled, use_container_width=True, height=420)

            csv_bytes = filtered[disp_cols].to_csv(index=False).encode()
            st.download_button(
                "⬇️ Download Suspicious Transactions (CSV)",
                data=csv_bytes,
                file_name="suspicious_transactions.csv",
                mime="text/plain",
                use_container_width=True,
            )

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 4 – Charts
    # ═══════════════════════════════════════════════════════════════════════
    with tab_charts:
        st.header("📈 Visualisations")
        if "result" not in st.session_state:
            st.info("Upload data and run detection first (Tab 1).")
        else:
            df        = st.session_state["result"]
            norm_df   = df[df["is_anomaly_detected"] == 0]
            anom_df   = df[df["is_anomaly_detected"] == 1]
            COLOR_MAP = {0: "#2980b9", 1: "#e74c3c"}
            type_map  = {0: "Normal",  1: "Anomaly"}

            # 1. Amount distribution
            st.subheader("Transaction Amount Distribution")
            fig1 = go.Figure()
            fig1.add_trace(go.Histogram(x=norm_df["transaction_amount"], name="Normal",
                                        marker_color="#2980b9", opacity=0.75, nbinsx=50))
            fig1.add_trace(go.Histogram(x=anom_df["transaction_amount"], name="Anomaly",
                                        marker_color="#e74c3c", opacity=0.85, nbinsx=30))
            fig1.update_layout(barmode="overlay", xaxis_title="Amount ($)",
                               yaxis_title="Count", height=380, margin=dict(t=20))
            st.plotly_chart(fig1, use_container_width=True)

            ch1, ch2 = st.columns(2)

            with ch1:
                # 2. Risk score distribution
                st.subheader("Risk Score Distribution")
                fig2 = px.histogram(
                    df, x="risk_score", color=df["is_anomaly_detected"].map(type_map),
                    nbins=50, barmode="overlay",
                    color_discrete_map={"Normal": "#2980b9", "Anomaly": "#e74c3c"},
                    labels={"color": "Type"},
                )
                fig2.add_vline(x=risk_threshold, line_dash="dash", line_color="#e74c3c",
                               annotation_text=f"Threshold ({risk_threshold})",
                               annotation_position="top right")
                fig2.update_layout(height=360, margin=dict(t=20))
                st.plotly_chart(fig2, use_container_width=True)

            with ch2:
                # 3. By hour of day
                if "hour" in df.columns:
                    st.subheader("Transactions by Hour of Day")
                    hourly = (
                        df.assign(Type=df["is_anomaly_detected"].map(type_map))
                        .groupby(["hour", "Type"])
                        .size()
                        .reset_index(name="count")
                    )
                    fig3 = px.bar(hourly, x="hour", y="count", color="Type",
                                  color_discrete_map={"Normal": "#2980b9", "Anomaly": "#e74c3c"},
                                  barmode="stack")
                    fig3.update_layout(height=360, margin=dict(t=20))
                    st.plotly_chart(fig3, use_container_width=True)

            ch3, ch4 = st.columns(2)

            with ch3:
                # 4. Amount vs Risk
                st.subheader("Amount vs Risk Score")
                fig4 = px.scatter(
                    df, x="transaction_amount", y="risk_score",
                    color=df["is_anomaly_detected"].map(type_map),
                    color_discrete_map={"Normal": "#2980b9", "Anomaly": "#e74c3c"},
                    opacity=0.55, size_max=8,
                    hover_data=[c for c in ["transaction_id", "customer_id"] if c in df.columns],
                )
                fig4.add_hline(y=risk_threshold, line_dash="dash", line_color="#e74c3c")
                fig4.update_layout(height=360, margin=dict(t=20),
                                   xaxis_title="Amount ($)", yaxis_title="Risk Score")
                st.plotly_chart(fig4, use_container_width=True)

            with ch4:
                # 5. Balance vs Amount
                st.subheader("Account Balance vs Transaction Amount")
                fig5 = px.scatter(
                    df, x="account_balance", y="transaction_amount",
                    color=df["is_anomaly_detected"].map(type_map),
                    color_discrete_map={"Normal": "#2980b9", "Anomaly": "#e74c3c"},
                    opacity=0.55,
                )
                fig5.update_layout(height=360, margin=dict(t=20),
                                   xaxis_title="Balance ($)", yaxis_title="Amount ($)")
                st.plotly_chart(fig5, use_container_width=True)

            # 6. Anomaly rate by location
            if "location" in df.columns:
                st.subheader("Anomaly Rate by Location")
                loc_stats = (
                    df.groupby("location")
                    .agg(total=("is_anomaly_detected", "count"),
                         anomalies=("is_anomaly_detected", "sum"))
                    .assign(anomaly_rate=lambda d: d["anomalies"] / d["total"] * 100)
                    .sort_values("anomaly_rate", ascending=False)
                    .reset_index()
                )
                fig6 = px.bar(loc_stats, x="location", y="anomaly_rate",
                              color="anomaly_rate", color_continuous_scale="RdYlGn_r",
                              labels={"anomaly_rate": "Anomaly Rate (%)", "location": "Location"})
                fig6.update_layout(height=380, margin=dict(t=20))
                st.plotly_chart(fig6, use_container_width=True)

    # ═══════════════════════════════════════════════════════════════════════
    # TAB 5 – Report
    # ═══════════════════════════════════════════════════════════════════════
    with tab_report:
        st.header("📋 Detection Report")
        if "result" not in st.session_state:
            st.info("Upload data and run detection first (Tab 1).")
        else:
            df       = st.session_state["result"]
            m_type   = st.session_state.get("model_type", model_type)
            total    = len(df)
            n_anom   = int(df["is_anomaly_detected"].sum())

            # Executive summary table
            st.subheader("Executive Summary")
            anom_amount = (
                df[df["is_anomaly_detected"] == 1]["transaction_amount"].mean()
                if n_anom else None
            )
            summary_rows = [
                ("Report Generated",               pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")),
                ("Total Transactions Analysed",    f"{total:,}"),
                ("Normal Transactions",            f"{total - n_anom:,}"),
                ("Suspicious Transactions",        f"{n_anom:,}"),
                ("Detection Rate",                 f"{n_anom/total*100:.2f}%"),
                ("Avg Transaction Amount (All)",   f"${df['transaction_amount'].mean():.2f}"),
                ("Avg Transaction Amount (Anomaly)",
                 f"${anom_amount:.2f}" if anom_amount else "N/A"),
                ("Avg Risk Score (Normal)",
                 f"{df[df['is_anomaly_detected']==0]['risk_score'].mean():.1f}"),
                ("Avg Risk Score (Anomaly)",
                 f"{df[df['is_anomaly_detected']==1]['risk_score'].mean():.1f}" if n_anom else "N/A"),
                ("Model Used",                     m_type.replace("_", " ").title()),
                ("Contamination Rate",             f"{contamination*100:.0f}%"),
                ("High-Risk Threshold",            f"{risk_threshold}"),
            ]
            st.dataframe(
                pd.DataFrame(summary_rows, columns=["Metric", "Value"]),
                use_container_width=True, hide_index=True,
            )

            st.markdown("---")
            rp_col1, rp_col2 = st.columns(2)

            with rp_col1:
                # Pie: risk buckets
                st.subheader("Risk Level Breakdown")
                buckets = pd.cut(
                    df["risk_score"],
                    bins=[0, 30, 60, 80, 100],
                    labels=["Low (0–30)", "Medium (30–60)", "High (60–80)", "Critical (80–100)"],
                    include_lowest=True,
                )
                counts = buckets.value_counts()
                fig_pie = px.pie(
                    values=counts.values, names=counts.index,
                    color_discrete_sequence=["#27ae60", "#f39c12", "#e74c3c", "#8e44ad"],
                )
                fig_pie.update_layout(height=340, margin=dict(t=10))
                st.plotly_chart(fig_pie, use_container_width=True)

            with rp_col2:
                # Top risky customers
                if "customer_id" in df.columns:
                    st.subheader("Top High-Risk Customers")
                    top = (
                        df.groupby("customer_id")
                        .agg(
                            Anomalies=("is_anomaly_detected", "sum"),
                            Max_Risk=("risk_score", "max"),
                            Avg_Risk=("risk_score", "mean"),
                            Total_Amount=("transaction_amount", "sum"),
                        )
                        .sort_values("Anomalies", ascending=False)
                        .head(10)
                        .reset_index()
                        .rename(columns={"customer_id": "Customer ID",
                                         "Max_Risk": "Max Risk",
                                         "Avg_Risk": "Avg Risk",
                                         "Total_Amount": "Total Spent ($)"})
                    )
                    top["Max Risk"] = top["Max Risk"].round(1)
                    top["Avg Risk"] = top["Avg Risk"].round(1)
                    top["Total Spent ($)"] = top["Total Spent ($)"].round(2)
                    st.dataframe(top, use_container_width=True, hide_index=True)

            # Download buttons
            st.markdown("---")
            st.subheader("Download Results")
            dl1, dl2 = st.columns(2)
            with dl1:
                st.download_button(
                    "⬇️ Full Results (all transactions)",
                    data=df.to_csv(index=False).encode(),
                    file_name="full_detection_results.csv",
                    mime="text/plain",
                    use_container_width=True,
                )
            with dl2:
                suspicious = df[df["is_anomaly_detected"] == 1].sort_values("risk_score", ascending=False)
                st.download_button(
                    "⬇️ Suspicious Transactions Only",
                    data=suspicious.to_csv(index=False).encode(),
                    file_name="suspicious_transactions.csv",
                    mime="text/plain",
                    use_container_width=True,
                )


if __name__ == "__main__":
    main()
