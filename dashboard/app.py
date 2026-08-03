import sys
sys.path.insert(0, "D:\\Projects\\PFA\\gps-digital-twin")
import os
import streamlit as st
import folium
from streamlit_folium import st_folium
import requests
import time

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(
    page_title="GPS Digital Twin",
    page_icon="🚕",
    layout="wide"
)

st.title("🚕 GPS Digital Twin — Fleet Monitor")

# Sidebar statistics
st.sidebar.title("📊 Fleet Statistics")

vehicles_response = requests.get(f"{API_URL}/vehicles")
vehicles = vehicles_response.json()

anomalies_response = requests.get(f"{API_URL}/anomalies")
anomaly_list = anomalies_response.json()

total_taxis = len(vehicles)
active_anomalies = sum(1 for v in vehicles if v.get("anomaly"))
avg_speed = sum(v.get("speed", 0) for v in vehicles) / total_taxis if total_taxis > 0 else 0

st.sidebar.metric("🚕 Active Taxis", total_taxis)
st.sidebar.metric("⚠️ Anomalies", active_anomalies)
st.sidebar.metric("⚡ Avg Speed", f"{avg_speed:.1f} m/s")

st.subheader("🗺️ Live Fleet Map")


m = folium.Map(location=[41.15, -8.61], zoom_start=13)

for v in vehicles:
    if "lat" not in v or "lon" not in v:
        continue
    color = "red" if v.get("anomaly") else "blue"
    folium.CircleMarker(
        location=[v["lat"], v["lon"]],
        radius=6,
        color=color,
        fill=True,
        popup=f"Taxi {v['taxi_id']} | Speed: {v.get('speed', 0):.1f} m/s"
    ).add_to(m)

st_folium(m, width=1200, height=500, returned_objects=[])

st.subheader("⚠️ Recent Anomalies")


if not anomaly_list:
    st.info("No anomalies detected yet.")
else:
    for a in anomaly_list[:10]:
        st.error(
            f"🚨 Taxi **{a['taxi_id']}** | "
            f"Speed: {a.get('speed', 0):.1f} m/s | "
            f"Uncertainty: {a.get('uncertainty', 0):.2f} | "
            f"Type: {a.get('anomaly_type', 'unknown')}"
        )

st.subheader("🤖 Fleet Assistant")

question = st.text_input("Ask the agent:", placeholder="e.g. What is the maintenance status of taxi 20000154?")

if st.button("Ask"):
    if question:
        with st.spinner("Thinking..."):
            response = requests.post(
                f"{API_URL}/agent",
                json={"question": question}
            )
            answer = response.json().get("answer", "No answer")
        st.success(answer)

if st.button("🔄 Refresh"):
    st.rerun()