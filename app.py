"""Streamlit Dashboard for Call Center Insights.

Visualizes call center performance metrics, sentiment shifts, resolution times,
and per-call evidence quotes with timestamps.
"""

import json
import os
import pandas as pd
import streamlit as st

# Configure page
st.set_page_config(
    page_title="Call Center Insights | Local QA Analytics",
    page_icon="📞",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown("""
<style>
    .reportview-container {
        background: #0e1117;
    }
    .metric-card {
        background-color: #1e222d;
        border: 1px solid #2e3440;
        border-radius: 8px;
        padding: 18px;
        margin-bottom: 12px;
    }
    .badge-simulated {
        background-color: #e67e22;
        color: white;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
    }
    .quote-box {
        background-color: #262c3a;
        border-left: 4px solid #3498db;
        padding: 10px 14px;
        margin: 6px 0;
        border-radius: 0 6px 6px 0;
        font-family: monospace;
    }
    .grounded-tag {
        color: #2ecc71;
        font-weight: bold;
    }
    .ungrounded-tag {
        color: #e74c3c;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


def load_data():
    """Load report data from CSV or fallback to simulated eval dataset."""
    default_csv = "reports/calls_report.csv"
    if os.path.exists(default_csv):
        df = pd.read_csv(default_csv)
        source = "reports/calls_report.csv"
        return df, source

    # Fallback to dev dataset if report not yet generated
    dev_json = "evals/data/dev_transcripts.json"
    if os.path.exists(dev_json):
        with open(dev_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        rows = []
        for d in data:
            gt = d.get("ground_truth", {})
            rows.append({
                "call_id": d["call_id"],
                "audio_file": d.get("audio_file", f"{d['call_id']}.wav"),
                "duration_seconds": d.get("duration_seconds", 60.0),
                "resolution_time_minutes": gt.get("resolution_time_minutes"),
                "resolution": gt.get("resolution", "unclear"),
                "customer_sentiment_start": gt.get("customer_sentiment_start", "neutral"),
                "customer_sentiment_end": gt.get("customer_sentiment_end", "neutral"),
                "sentiment_shift": f"{gt.get('customer_sentiment_start')} -> {gt.get('customer_sentiment_end')}",
                "agent_handled_well": gt.get("agent_handled_well", True),
                "grounded": True,
                "reason": gt.get("reason", "Simulated demo ground-truth record."),
                "evidence": "; ".join(f"[{e.get('timestamp', 0.0)}s] \"{e.get('quote')}\"" for e in gt.get("evidence", [])),
                "provider": "simulated_ground_truth",
                "is_stereo": True,
                "diarization_confidence": 1.0,
            })
        return pd.DataFrame(rows), "evals/data/dev_transcripts.json (simulated reference)"

    return pd.DataFrame(), "None"


def main():
    st.title("📞 Call Center Insights Local QA Analytics")
    st.markdown('<span class="badge-simulated">SIMULATED DATASET</span> *All bundled evaluation and demo data are strictly synthetic/simulated.*', unsafe_allow_html=True)
    st.write("")

    df, source_label = load_data()

    # Sidebar
    st.sidebar.header("Controls & Filters")
    st.sidebar.caption(f"Active data source: `{source_label}`")

    if df.empty:
        st.warning("No call analysis report found. Run `python -m callinsight.cli analyze samples/ --out reports/` to generate reports.")
        return

    # Sidebar Filters
    resolutions = ["All"] + sorted(list(df["resolution"].dropna().unique()))
    selected_res = st.sidebar.selectbox("Filter by Resolution", resolutions)

    sentiments = ["All"] + sorted(list(df["customer_sentiment_start"].dropna().unique()))
    selected_sent = st.sidebar.selectbox("Filter by Start Sentiment", sentiments)

    filtered_df = df.copy()
    if selected_res != "All":
        filtered_df = filtered_df[filtered_df["resolution"] == selected_res]
    if selected_sent != "All":
        filtered_df = filtered_df[filtered_df["customer_sentiment_start"] == selected_sent]

    # Metrics Overview
    total_calls = len(filtered_df)
    resolved_count = (filtered_df["resolution"] == "resolved").sum()
    res_rate = round((resolved_count / total_calls * 100), 1) if total_calls > 0 else 0.0

    valid_res_times = filtered_df[filtered_df["resolution"] == "resolved"]["resolution_time_minutes"].dropna()
    avg_res_time = round(valid_res_times.mean(), 2) if not valid_res_times.empty else "N/A"

    handled_well_rate = round((filtered_df["agent_handled_well"].sum() / total_calls * 100), 1) if total_calls > 0 else 0.0
    grounded_rate = round((filtered_df["grounded"].sum() / total_calls * 100), 1) if total_calls > 0 else 0.0

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Calls", f"{total_calls}")
    col2.metric("Resolved Rate", f"{res_rate}%", f"{resolved_count} calls")
    col3.metric("Avg. Resolution Time", f"{avg_res_time} min" if avg_res_time != "N/A" else "N/A")
    col4.metric("Agent Handled Well", f"{handled_well_rate}%")
    col5.metric("Grounding Rate", f"{grounded_rate}%")

    st.write("---")

    # Visualizations
    tab_overview, tab_inspector, tab_raw = st.tabs(["📊 Analytics Overview", "🔍 Call Inspector", "📋 Data Table"])

    with tab_overview:
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Customer Sentiment Shifts")
            shift_counts = filtered_df["sentiment_shift"].value_counts().reset_index()
            shift_counts.columns = ["Shift", "Count"]
            st.bar_chart(shift_counts.set_index("Shift"))

        with c2:
            st.subheader("Resolution Breakdown")
            res_counts = filtered_df["resolution"].value_counts().reset_index()
            res_counts.columns = ["Resolution", "Count"]
            st.bar_chart(res_counts.set_index("Resolution"))

    with tab_inspector:
        st.subheader("Deep-Dive Call Inspector")
        call_ids = list(filtered_df["call_id"].unique())
        if call_ids:
            selected_call_id = st.selectbox("Select Call ID", call_ids)
            call_row = filtered_df[filtered_df["call_id"] == selected_call_id].iloc[0]

            col_a, col_b, col_c = st.columns(3)
            col_a.write(f"**Resolution:** `{call_row['resolution']}`")
            col_a.write(f"**Duration:** `{call_row['duration_seconds']}s`")
            col_b.write(f"**Sentiment Shift:** `{call_row['sentiment_shift']}`")
            col_b.write(f"**Resolution Time:** `{call_row['resolution_time_minutes']} min`")
            col_c.write(f"**Agent Handled Well:** `{call_row['agent_handled_well']}`")
            is_gr = call_row['grounded']
            col_c.write(f"**Grounding:** {'<span class=\"grounded-tag\">✓ Grounded</span>' if is_gr else '<span class=\"ungrounded-tag\">✗ Ungrounded</span>'}", unsafe_allow_html=True)

            st.markdown(f"**Reason:** *{call_row.get('reason', 'N/A')}*")

            st.write("### Supporting Evidence Quotes")
            evidence_val = call_row.get("evidence", "")
            if pd.notna(evidence_val) and str(evidence_val).strip():
                for quote_item in str(evidence_val).split("; "):
                    st.markdown(f'<div class="quote-box">{quote_item}</div>', unsafe_allow_html=True)
            else:
                st.caption("No evidence quotes attached to this call.")
        else:
            st.info("No calls matching filter criteria.")

    with tab_raw:
        st.subheader("Call Records Table")
        st.dataframe(filtered_df, use_container_width=True)


if __name__ == "__main__":
    main()
