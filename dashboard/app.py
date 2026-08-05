import sys
sys.path.insert(0, "D:\\Projects\\PFA\\gps-digital-twin")

import os
import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
import pandas as pd
import plotly.express as px
from datetime import datetime

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="iDIRTECH Fleet Monitor",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .stApp { background-color: #0f1117; }
    .block-container { padding: 1.5rem 2rem 1rem 2rem !important; }
    #MainMenu, footer, header { visibility: hidden; }

    div[data-testid="metric-container"] {
        background: #161b27;
        border: 1px solid #1e2130;
        border-radius: 10px;
        padding: 1rem 1.25rem;
    }
    div[data-testid="metric-container"] label {
        font-size: 0.7rem !important;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #6b7280 !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #ffffff !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricDelta"] {
        font-size: 0.72rem !important;
    }

    .stTabs [data-baseweb="tab-list"] {
        background: transparent;
        border-bottom: 1px solid #1e2130;
        gap: 0;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 0.85rem;
        font-weight: 500;
        color: #6b7280;
        padding: 0.6rem 1.5rem;
        border-radius: 0;
        background: transparent;
    }
    .stTabs [aria-selected="true"] {
        color: #ffffff !important;
        font-weight: 700 !important;
        border-bottom: 3px solid #ef4444 !important;
        background: transparent !important;
    }

    .stButton > button {
        background: #ef4444 !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        font-size: 0.8rem !important;
    }
    .stButton > button:hover { background: #dc2626 !important; }

    [data-testid="stExpander"] {
        background: #161b27 !important;
        border: 1px solid #1e2130 !important;
        border-radius: 8px !important;
        margin-bottom: 0.5rem;
    }
    [data-testid="stExpander"] summary {
        color: #e5e7eb !important;
        font-size: 0.8rem !important;
    }

    .stTextInput input {
        background: #161b27 !important;
        border: 1px solid #1e2130 !important;
        color: #e5e7eb !important;
        border-radius: 8px !important;
    }

    p, .stMarkdown, label { color: #9ca3af !important; }
    h1, h2, h3, h4, h5, h6 { color: #ffffff !important; }

    hr { border-color: #1e2130 !important; }

    .stInfo { background: #161b27 !important; border-color: #1e2130 !important; }
    .stCaption { color: #6b7280 !important; font-size: 0.72rem !important; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=30)
def fetch(endpoint):
    try:
        return requests.get(f"{API_URL}/{endpoint}", timeout=5).json()
    except:
        return []

vehicles     = fetch("vehicles")
anomaly_list = fetch("anomalies")
diag_list    = fetch("diagnoses")

total_taxis      = len(vehicles)
active_anomalies = sum(1 for v in vehicles if v.get("anomaly"))
avg_speed        = sum(v.get("speed", 0) for v in vehicles) / total_taxis if total_taxis > 0 else 0
anomaly_rate     = round(active_anomalies / total_taxis * 100, 1) if total_taxis > 0 else 0
now              = datetime.now().strftime("%H:%M:%S")

# ── Header ────────────────────────────────────────────────────────────────
col_title, col_badge = st.columns([5, 1])
with col_title:
    st.markdown(f"""
    <div style="padding:0.25rem 0 1rem 0">
        <span style="font-size:1.3rem;font-weight:800;color:#ffffff;letter-spacing:-0.02em">
            🚕 iDIRTECH Fleet Monitor
        </span><br>
        <span style="font-size:0.75rem;color:#4b5563">
            GPS Digital Twin &nbsp;·&nbsp; Real-time Fleet Intelligence &nbsp;·&nbsp; Updated {now}
        </span>
    </div>
    """, unsafe_allow_html=True)
with col_badge:
    st.markdown("""
    <div style="text-align:right;padding-top:0.5rem">
        <span style="background:#052e16;color:#22c55e;font-size:0.72rem;font-weight:700;
                     padding:0.3rem 0.8rem;border-radius:20px;border:1px solid #166534">
            ● LIVE
        </span>
    </div>
    """, unsafe_allow_html=True)

# ── KPI Row ────────────────────────────────────────────────────────────────
k1, k2, k3, k4 = st.columns(4)
with k1:
    st.metric("🚗 Active Vehicles", total_taxis)
with k2:
    st.metric("⚠️ Active Anomalies", active_anomalies,
              delta=f"{anomaly_rate}% of fleet" if active_anomalies > 0 else None,
              delta_color="inverse")
with k3:
    st.metric("⚡ Avg Speed", f"{avg_speed:.1f} m/s")
with k4:
    st.metric("🤖 AI Diagnoses", len(diag_list))

st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

# ── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2 = st.tabs(["🗺️   Live Monitor", "🤖   Fleet Assistant"])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — Live Monitor
# ══════════════════════════════════════════════════════════════════════════
with tab1:
    map_col, right_col = st.columns([3, 2], gap="medium")

    with map_col:
        st.markdown("##### 🗺️ Live Fleet Map — Porto, Portugal")

        m = folium.Map(
            location=[41.15, -8.61],
            zoom_start=13,
            tiles="OpenStreetMap"
        )

        for v in vehicles:
            if "lat" not in v or "lon" not in v:
                continue
            is_anom = v.get("anomaly", False)
            color   = "#ef4444" if is_anom else "#3b82f6"
            speed   = v.get("speed", 0)
            atype   = v.get("anomaly_type", "normal")
            tid     = v.get("taxi_id", "")

            folium.CircleMarker(
                location=[v["lat"], v["lon"]],
                radius=9 if is_anom else 6,
                color=color, fill=True, fill_color=color,
                fill_opacity=0.9, weight=2,
                popup=folium.Popup(
                    f"<b>Taxi {tid}</b><br>Speed: {speed:.1f} m/s<br>"
                    f"{'⚠️ ' + atype if is_anom else '✅ Normal'}",
                    max_width=200
                ),
                tooltip=f"🚕 {tid}"
            ).add_to(m)

        st_folium(m, use_container_width=True, height=460, returned_objects=[])

        if anomaly_list:
            st.markdown("##### 📊 Anomaly Distribution by Vehicle")
            adf = pd.DataFrame(anomaly_list)
            if "taxi_id" in adf.columns and "anomaly_type" in adf.columns:
                chart = adf.groupby(["taxi_id", "anomaly_type"]).size().reset_index(name="count")
                fig = px.bar(
                    chart, x="taxi_id", y="count", color="anomaly_type",
                    color_discrete_map={
                        "speed": "#ef4444",
                        "position_jump": "#f59e0b",
                        "uncertainty": "#3b82f6"
                    },
                    template="plotly_dark", height=220
                )
                fig.update_layout(
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                    margin=dict(l=0, r=0, t=10, b=0),
                    legend=dict(orientation="h", y=-0.4, font=dict(size=10, color="#6b7280")),
                    xaxis_title="", yaxis_title="Count",
                    font=dict(size=10, color="#6b7280"),
                    bargap=0.3
                )
                fig.update_xaxes(showgrid=False, tickangle=45, tickfont=dict(size=8))
                fig.update_yaxes(showgrid=True, gridcolor="#1e2130", dtick=1)
                st.plotly_chart(fig, use_container_width=True)

    with right_col:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

        st.markdown("##### ⚠️ Recent Anomalies")
        if not anomaly_list:
            st.info("No anomalies detected yet.")
        else:
            for a in anomaly_list[:6]:
                speed = a.get("speed", 0)
                atype = a.get("anomaly_type", "unknown")
                tid   = a.get("taxi_id", "N/A")
                icon  = "🔴" if speed > 100 else "🟡"
                st.markdown(f"""
                <div style="padding:0.6rem 0;border-bottom:1px solid #1e2130">
                    <div style="font-size:0.8rem;font-weight:600;color:#e5e7eb">
                        {icon} Taxi {tid}
                    </div>
                    <div style="font-size:0.7rem;color:#6b7280;margin-top:0.1rem">
                        {atype} &nbsp;·&nbsp; {speed:.1f} m/s
                    </div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
        st.markdown("##### 🤖 AI Diagnoses")

        if not diag_list:
            st.info("No diagnoses yet — pipeline running...")
        else:
            for d in diag_list[:4]:
                diagnosis = d.get("diagnosis", {})
                tid   = d.get("taxi_id", "N/A")
                atype = d.get("anomaly_type", "")

                if isinstance(diagnosis, dict):
                    severity  = diagnosis.get("severity", "HIGH")
                    diag_text = diagnosis.get("diagnosis", "N/A")
                    rec_text  = diagnosis.get("recommendation", "N/A")
                    badge_color  = "#ef4444" if severity == "HIGH" else "#f59e0b" if severity == "MEDIUM" else "#3b82f6"
                    border_color = badge_color
                else:
                    severity     = "HIGH"
                    diag_text    = str(diagnosis)[:120]
                    rec_text     = "—"
                    badge_color  = "#ef4444"
                    border_color = "#ef4444"

                st.markdown(f"""
                <div style="
                    background:#161b27;
                    border:1px solid #1e2130;
                    border-left:4px solid {border_color};
                    border-radius:8px;
                    padding:0.75rem 1rem;
                    margin-bottom:0.6rem;
                ">
                    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:0.35rem">
                        <span style="font-size:0.8rem;font-weight:600;color:#e5e7eb">
                            Taxi {tid} · {atype}
                        </span>
                        <span style="font-size:0.62rem;font-weight:700;
                                     background:{badge_color}22;color:{badge_color};
                                     padding:0.15rem 0.5rem;border-radius:4px">
                            {severity}
                        </span>
                    </div>
                    <div style="font-size:0.73rem;color:#9ca3af;margin-bottom:0.2rem">
                        📋 {diag_text}
                    </div>
                    <div style="font-size:0.73rem;color:#38bdf8;font-weight:500">
                        → {rec_text}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — Fleet Assistant
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    left, right = st.columns([3, 2], gap="medium")

    with left:
        st.markdown("##### 💬 Ask the Fleet AI")
        st.caption("Queries vehicle history, maintenance records, and past anomalies to give actionable insights.")

        st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)

        question = st.text_input(
            "question",
            placeholder="e.g. Which taxi needs attention most urgently?",
            label_visibility="collapsed"
        )
        st.caption("💡 Try: *What is the maintenance status of taxi 20000154?* · *Summarize the fleet health* · *Find anomalies similar to high speed detection*")

        st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)

        if st.button("Ask Agent", use_container_width=True):
            if question:
                with st.spinner("Agent reasoning — this may take 15–30 seconds..."):
                    try:
                        response = requests.post(
                            f"{API_URL}/agent",
                            json={"question": question},
                            timeout=60
                        )
                        answer = response.json().get("answer", "No answer received.")
                        st.markdown("**Agent response:**")
                        st.markdown(f"""
                        <div style="
                            background:#161b27;border:1px solid #1e2130;
                            border-radius:10px;padding:1.25rem;
                            font-size:0.82rem;color:#d1d5db;
                            line-height:1.75;margin-top:0.5rem;
                            white-space:pre-wrap;
                        ">{answer}</div>
                        """, unsafe_allow_html=True)
                    except requests.exceptions.Timeout:
                        st.error("⏱️ Agent timed out. Try a simpler question.")
                    except Exception as e:
                        st.error(f"Error: {e}")
            else:
                st.warning("Please enter a question.")

    with right:
        st.markdown("##### 📋 Recent Auto-Diagnoses")
        st.caption("Automatically generated when anomalies are detected.")

        st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)

        if not diag_list:
            st.info("No diagnoses yet.")
        else:
            for d in diag_list[:8]:
                diagnosis = d.get("diagnosis", {})
                tid   = d.get("taxi_id", "N/A")
                atype = d.get("anomaly_type", "unknown")
                label = f"Taxi {tid} — {atype}"

                with st.expander(label):
                    if isinstance(diagnosis, dict):
                        severity = diagnosis.get("severity", "N/A")
                        color    = "#ef4444" if severity == "HIGH" else "#f59e0b" if severity == "MEDIUM" else "#3b82f6"
                        st.markdown(f"**Severity:** <span style='color:{color};font-weight:700'>{severity}</span>", unsafe_allow_html=True)
                        st.markdown(f"**Diagnosis:** {diagnosis.get('diagnosis', 'N/A')}")
                        st.markdown(f"**Recommendation:** {diagnosis.get('recommendation', 'N/A')}")
                        st.markdown(f"**Maintenance:** {diagnosis.get('maintenance_status', 'N/A')}")
                    else:
                        st.write(str(diagnosis))