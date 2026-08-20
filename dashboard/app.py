"""
iDIRTECH Fleet Monitor — GPS Digital Twin dashboard.

Surfaces two independent anomaly detectors side by side:
  · Rule-based thresholds (speed > 80 m/s, jump > 800 m)
  · Isolation Forest on per-vehicle normalised features

Consumes the FastAPI layer; no direct database access.
"""

import sys
sys.path.insert(0, "D:\\Projects\\PFA\\gps-digital-twin")

import os
import json
import re
from datetime import datetime

import folium
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_folium import st_folium

API_URL = os.getenv("API_URL", "http://localhost:8000")

# Detection state → colour. Purple marks ML-only detections: anomalies the
# threshold rules are structurally unable to express.
C_NORMAL = "#3b82f6"
C_RULE   = "#f59e0b"
C_ML     = "#a855f7"
C_BOTH   = "#ef4444"

SEV_COLOUR = {"HIGH": "#ef4444", "MEDIUM": "#f59e0b", "LOW": "#3b82f6"}

st.set_page_config(
    page_title="iDIRTECH Fleet Monitor",
    page_icon="🚕",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .stApp { background: #0b0e14; }
    .block-container { padding: 1.25rem 2rem 2rem 2rem !important; max-width: 100% !important; }
    #MainMenu, footer, header { visibility: hidden; }

    * { font-feature-settings: "tnum" 1, "cv05" 1; }

    /* ── masthead ────────────────────────────────────────────────── */
    .masthead {
        display: flex; align-items: baseline; justify-content: space-between;
        padding-bottom: .85rem; margin-bottom: 1.1rem;
        border-bottom: 1px solid #1b2130;
    }
    .brand { font-size: 1.05rem; font-weight: 700; color: #f3f4f6; letter-spacing: -.015em; }
    .brand span { color: #6b7280; font-weight: 400; }
    .sub { font-size: .7rem; color: #4b5563; margin-top: .15rem; letter-spacing: .02em; }
    .status { display: flex; align-items: center; gap: .9rem; }
    .pill {
        display: inline-flex; align-items: center; gap: .4rem;
        font-size: .66rem; font-weight: 600; letter-spacing: .07em;
        padding: .28rem .7rem; border-radius: 3px;
        background: rgba(34,197,94,.09); color: #22c55e;
        border: 1px solid rgba(34,197,94,.22);
    }
    .dot { width: 5px; height: 5px; border-radius: 50%; background: #22c55e;
           animation: blink 2s ease-in-out infinite; }
    @keyframes blink { 0%,100% { opacity: 1 } 50% { opacity: .25 } }
    .clock { font-size: .7rem; color: #4b5563; font-variant-numeric: tabular-nums; }

    /* ── metric strip ────────────────────────────────────────────── */
    .strip { display: grid; grid-template-columns: repeat(5, 1fr); gap: 1px;
             background: #1b2130; border: 1px solid #1b2130; border-radius: 4px;
             overflow: hidden; margin-bottom: 1.25rem; }
    .cell { background: #10141d; padding: .85rem 1.1rem; }
    .cell-k { font-size: .62rem; color: #4b5563; text-transform: uppercase;
              letter-spacing: .11em; font-weight: 600; margin-bottom: .35rem; }
    .cell-v { font-size: 1.65rem; font-weight: 650; color: #e5e7eb;
              line-height: 1; letter-spacing: -.03em; font-variant-numeric: tabular-nums; }
    .cell-u { font-size: .78rem; font-weight: 400; color: #4b5563; margin-left: .12rem; }
    .cell-n { font-size: .64rem; color: #4b5563; margin-top: .3rem; }
    .v-warn { color: #f59e0b; }
    .v-crit { color: #ef4444; }
    .v-ml   { color: #a855f7; }

    /* ── section rule ────────────────────────────────────────────── */
    .rule { display: flex; align-items: center; gap: .7rem;
            margin: 0 0 .7rem 0; }
    .rule-t { font-size: .66rem; font-weight: 650; color: #6b7280;
              text-transform: uppercase; letter-spacing: .12em; white-space: nowrap; }
    .rule-l { flex: 1; height: 1px; background: #1b2130; }
    .rule-c { font-size: .64rem; color: #374151; font-variant-numeric: tabular-nums; }

    /* ── feed rows ───────────────────────────────────────────────── */
    .row { display: flex; gap: .7rem; padding: .5rem 0;
           border-bottom: 1px solid #151a24; }
    .row:last-child { border-bottom: none; }
    .bar { width: 2px; border-radius: 1px; flex-shrink: 0; }
    .row-b { flex: 1; min-width: 0; }
    .row-t { display: flex; justify-content: space-between; align-items: baseline; }
    .row-id { font-size: .74rem; font-weight: 600; color: #d1d5db;
              font-variant-numeric: tabular-nums; }
    .row-v { font-size: .7rem; color: #6b7280; font-variant-numeric: tabular-nums; }
    .row-m { font-size: .64rem; color: #4b5563; margin-top: .1rem;
             display: flex; gap: .45rem; align-items: center; }
    .tag { font-size: .58rem; font-weight: 600; padding: .05rem .32rem;
           border-radius: 2px; letter-spacing: .04em; }

    /* ── diagnosis ───────────────────────────────────────────────── */
    .diag { background: #10141d; border: 1px solid #1b2130;
            border-left: 2px solid #ef4444; border-radius: 3px;
            padding: .65rem .8rem; margin-bottom: .5rem; }
    .diag-h { display: flex; justify-content: space-between;
              align-items: center; margin-bottom: .35rem; }
    .diag-id { font-size: .73rem; font-weight: 600; color: #d1d5db; }
    .diag-s { font-size: .58rem; font-weight: 700; padding: .1rem .4rem;
              border-radius: 2px; letter-spacing: .05em; }
    .diag-b { font-size: .69rem; color: #9ca3af; line-height: 1.45; }
    .diag-r { font-size: .69rem; color: #38bdf8; margin-top: .28rem; line-height: 1.45; }

    /* ── health ──────────────────────────────────────────────────── */
    .hp { margin-bottom: .55rem; }
    .hp-t { display: flex; justify-content: space-between; margin-bottom: .22rem; }
    .hp-id { font-size: .71rem; color: #9ca3af; font-variant-numeric: tabular-nums; }
    .hp-v { font-size: .71rem; font-weight: 650; font-variant-numeric: tabular-nums; }
    .hp-track { background: #151a24; border-radius: 1px; height: 3px; overflow: hidden; }
    .hp-fill { height: 3px; border-radius: 1px; }

    /* ── legend ──────────────────────────────────────────────────── */
    .leg { display: flex; gap: 1.15rem; flex-wrap: wrap;
           font-size: .64rem; color: #4b5563; margin-top: .6rem;
           padding-top: .55rem; border-top: 1px solid #151a24; }
    .leg-i { display: flex; align-items: center; gap: .35rem; }
    .leg-d { width: 7px; height: 7px; border-radius: 50%; }

    .empty { font-size: .7rem; color: #374151; padding: 1.1rem 0;
             text-align: center; font-style: italic; }

    /* ── streamlit overrides ─────────────────────────────────────── */
    .stTabs [data-baseweb="tab-list"] { gap: 0; background: transparent;
        border-bottom: 1px solid #1b2130; margin-bottom: 1.1rem; }
    .stTabs [data-baseweb="tab"] { font-size: .76rem; font-weight: 500;
        color: #4b5563; padding: .5rem 1.15rem; background: transparent; border-radius: 0; }
    .stTabs [aria-selected="true"] { color: #e5e7eb !important; font-weight: 650 !important;
        border-bottom: 2px solid #a855f7 !important; }

    .stButton > button { background: #151a24 !important; color: #9ca3af !important;
        border: 1px solid #1b2130 !important; border-radius: 3px !important;
        font-size: .7rem !important; font-weight: 600 !important;
        letter-spacing: .04em !important; padding: .38rem .9rem !important; }
    .stButton > button:hover { background: #1b2130 !important; color: #e5e7eb !important;
        border-color: #2a3244 !important; }

    .stTextInput input { background: #10141d !important; border: 1px solid #1b2130 !important;
        color: #e5e7eb !important; border-radius: 3px !important; font-size: .78rem !important; }
    .stTextInput input:focus { border-color: #a855f7 !important; box-shadow: none !important; }

    [data-testid="stExpander"] { background: #10141d !important;
        border: 1px solid #1b2130 !important; border-radius: 3px !important;
        margin-bottom: .4rem !important; }
    [data-testid="stExpander"] summary { color: #9ca3af !important; font-size: .73rem !important; }

    p, .stMarkdown, label { color: #9ca3af; }
    hr { border-color: #1b2130 !important; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=30, show_spinner=False)
def fetch(endpoint: str):
    """GET an API endpoint, returning [] on any failure."""
    try:
        return requests.get(f"{API_URL}/{endpoint}", timeout=5).json()
    except Exception:
        return []


vehicles     = fetch("vehicles")
anomaly_list = fetch("anomalies")
diag_list    = fetch("diagnoses")
health_list  = fetch("health")

api_up = bool(vehicles or anomaly_list or diag_list)

# Freeze map data in session state so typing in the agent box does not
# force Folium to re-render the whole map.
if "map_data" not in st.session_state or st.session_state.get("refresh_map"):
    st.session_state["map_data"] = vehicles
    st.session_state["refresh_map"] = False


def classify(v: dict) -> tuple[str, str]:
    """Return (colour, label) for a vehicle's detection state."""
    r, m = v.get("anomaly", False), v.get("ml_anomaly", False)
    if r and m:  return C_BOTH,   "Both detectors"
    if r:        return C_RULE,   "Rule-based only"
    if m:        return C_ML,     "ML only"
    return C_NORMAL, "Normal"


n_total  = len(vehicles)
n_rule   = sum(1 for v in vehicles if v.get("anomaly"))
n_ml     = sum(1 for v in vehicles if v.get("ml_anomaly"))
n_both   = sum(1 for v in vehicles if v.get("anomaly") and v.get("ml_anomaly"))
n_mlonly = n_ml - n_both
n_warm   = sum(1 for v in vehicles if str(v.get("ml_status", "")).startswith("warmup"))
avg_spd  = sum(v.get("speed", 0) for v in vehicles) / n_total if n_total else 0
rate     = round(n_rule / n_total * 100, 1) if n_total else 0
now      = datetime.now().strftime("%H:%M:%S")


# ─────────────────────────────────────────────────────────────────────
# Masthead
# ─────────────────────────────────────────────────────────────────────
badge = ('<span class="pill"><span class="dot"></span>LIVE</span>' if api_up else
         '<span class="pill" style="background:rgba(239,68,68,.09);color:#ef4444;'
         'border-color:rgba(239,68,68,.22)">API OFFLINE</span>')

st.markdown(f"""
<div class="masthead">
  <div>
    <div class="brand">iDIRTECH <span>/ Fleet Monitor</span></div>
    <div class="sub">GPS Digital Twin · Kalman filtering · Isolation Forest anomaly detection</div>
  </div>
  <div class="status">{badge}<span class="clock">{now}</span></div>
</div>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────
# Metric strip
# ─────────────────────────────────────────────────────────────────────
rule_cls = "v-crit" if n_rule > 5 else "v-warn" if n_rule else ""
warm_note = f"{n_warm} in warm-up" if n_warm else "all models active"

st.markdown(f"""
<div class="strip">
  <div class="cell">
    <div class="cell-k">Vehicles</div>
    <div class="cell-v">{n_total}</div>
    <div class="cell-n">tracked live</div>
  </div>
  <div class="cell">
    <div class="cell-k">Rule-based</div>
    <div class="cell-v {rule_cls}">{n_rule}</div>
    <div class="cell-n">{rate}% of fleet</div>
  </div>
  <div class="cell">
    <div class="cell-k">Isolation Forest</div>
    <div class="cell-v v-ml">{n_ml}</div>
    <div class="cell-n">{n_mlonly} beyond rules</div>
  </div>
  <div class="cell">
    <div class="cell-k">Mean speed</div>
    <div class="cell-v">{avg_spd:.1f}<span class="cell-u">m/s</span></div>
    <div class="cell-n">{warm_note}</div>
  </div>
  <div class="cell">
    <div class="cell-k">AI diagnoses</div>
    <div class="cell-v">{len(diag_list)}</div>
    <div class="cell-n">auto-generated</div>
  </div>
</div>
""", unsafe_allow_html=True)


def rule_header(title: str, count: str = "") -> None:
    st.markdown(
        f'<div class="rule"><span class="rule-t">{title}</span>'
        f'<span class="rule-l"></span>'
        f'<span class="rule-c">{count}</span></div>',
        unsafe_allow_html=True,
    )


tab_live, tab_agent, tab_model = st.tabs(
    ["Live Monitor", "Fleet Assistant", "Detector Comparison"]
)

# ═════════════════════════════════════════════════════════════════════
# Live Monitor
# ═════════════════════════════════════════════════════════════════════
with tab_live:
    left, right = st.columns([1.55, 1], gap="medium")

    with left:
        rule_header("Fleet position", "Porto, Portugal · UTM 29N")

        fmap = folium.Map(
            location=[41.15, -8.61], zoom_start=13,
            tiles="CartoDB positron", control_scale=True,
        )

        for v in st.session_state["map_data"]:
            if "lat" not in v or "lon" not in v:
                continue
            colour, label = classify(v)
            flagged = colour != C_NORMAL
            score = v.get("ml_score")
            score_txt = f"{score:.3f}" if isinstance(score, (int, float)) else "warm-up"

            folium.CircleMarker(
                location=[v["lat"], v["lon"]],
                radius=7 if flagged else 4.5,
                color=colour, fill=True, fill_color=colour,
                fill_opacity=.85, weight=1.6 if flagged else 1,
                popup=folium.Popup(
                    f'<div style="font-family:system-ui;font-size:12px;line-height:1.55">'
                    f'<b>Taxi {v.get("taxi_id","")}</b><br>'
                    f'{label}<br>'
                    f'Speed &nbsp;{v.get("speed",0):.1f} m/s<br>'
                    f'Health &nbsp;{v.get("health_score","—")}/100<br>'
                    f'IF score &nbsp;{score_txt}</div>',
                    max_width=210,
                ),
                tooltip=str(v.get("taxi_id", "")),
            ).add_to(fmap)

        st_folium(fmap, use_container_width=True, height=452, returned_objects=[])

        st.markdown(f"""
        <div class="leg">
          <span class="leg-i"><span class="leg-d" style="background:{C_NORMAL}"></span>Normal</span>
          <span class="leg-i"><span class="leg-d" style="background:{C_RULE}"></span>Rule-based only</span>
          <span class="leg-i"><span class="leg-d" style="background:{C_ML}"></span>Isolation Forest only</span>
          <span class="leg-i"><span class="leg-d" style="background:{C_BOTH}"></span>Both detectors</span>
        </div>
        """, unsafe_allow_html=True)

        # ── detector overlap ─────────────────────────────────────────
        if n_rule or n_ml:
            st.markdown("<div style='height:1.1rem'></div>", unsafe_allow_html=True)
            rule_header("Detector overlap", f"{n_total} vehicles")

            n_ruleonly = n_rule - n_both
            n_clean = n_total - n_rule - n_mlonly

            fig = go.Figure(go.Bar(
                x=[n_clean, n_ruleonly, n_mlonly, n_both],
                y=["Normal", "Rule only", "IF only", "Both"],
                orientation="h",
                marker=dict(color=[C_NORMAL, C_RULE, C_ML, C_BOTH]),
                text=[n_clean, n_ruleonly, n_mlonly, n_both],
                textposition="outside",
                textfont=dict(size=10, color="#6b7280"),
                hovertemplate="%{y}: %{x} vehicles<extra></extra>",
            ))
            fig.update_layout(
                height=158, template="plotly_dark",
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=28, t=4, b=4),
                font=dict(size=10, color="#6b7280"),
                xaxis=dict(showgrid=True, gridcolor="#151a24", zeroline=False, title=""),
                yaxis=dict(showgrid=False, title="", autorange="reversed"),
                bargap=.42, showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True,
                            config={"displayModeBar": False})

    with right:
        if st.button("Refresh", use_container_width=True):
            st.cache_data.clear()
            st.session_state["refresh_map"] = True
            st.rerun()

        st.markdown("<div style='height:.85rem'></div>", unsafe_allow_html=True)

        # ── anomaly feed ─────────────────────────────────────────────
        rule_header("Anomaly feed", f"{len(anomaly_list)} recent · historical")
        if not anomaly_list:
            st.markdown('<div class="empty">No anomalies detected</div>',
                        unsafe_allow_html=True)
        else:
            for a in anomaly_list[:7]:
                spd  = a.get("speed", 0)
                atyp = a.get("anomaly_type", "—")
                by_ml, by_rule = a.get("ml_anomaly", False), a.get("anomaly", False)

                if by_ml and by_rule:
                    bar, tag, tag_c = C_BOTH, "BOTH", C_BOTH
                elif by_ml:
                    bar, tag, tag_c = C_ML, "IF", C_ML
                else:
                    bar, tag, tag_c = C_RULE, "RULE", C_RULE

                st.markdown(f"""
                <div class="row">
                  <div class="bar" style="background:{bar}"></div>
                  <div class="row-b">
                    <div class="row-t">
                      <span class="row-id">{a.get('taxi_id','—')}</span>
                      <span class="row-v">{spd:.1f} m/s</span>
                    </div>
                    <div class="row-m">
                      <span class="tag" style="background:{tag_c}1a;color:{tag_c}">{tag}</span>
                      <span>{atyp.replace('_',' ')}</span>
                    </div>
                  </div>
                </div>
                """, unsafe_allow_html=True)

        # ── diagnoses ────────────────────────────────────────────────
        st.markdown("<div style='height:1.1rem'></div>", unsafe_allow_html=True)
        rule_header("Agent diagnoses", f"{len(diag_list)} total")

        if not diag_list:
            st.markdown('<div class="empty">Awaiting anomalies</div>',
                        unsafe_allow_html=True)
        else:
            for d in diag_list[:4]:
                dg = d.get("diagnosis", {})
                if isinstance(dg, dict):
                    sev  = dg.get("severity", "HIGH")
                    body = dg.get("diagnosis", "—")
                    rec  = dg.get("recommendation", "—")
                else:
                    sev, body, rec = "HIGH", str(dg)[:110], "—"
                col = SEV_COLOUR.get(sev, "#ef4444")

                st.markdown(f"""
                <div class="diag" style="border-left-color:{col}">
                  <div class="diag-h">
                    <span class="diag-id">{d.get('taxi_id','—')}
                      <span style="color:#4b5563;font-weight:400">
                        · {d.get('anomaly_type','').replace('_',' ')}</span></span>
                    <span class="diag-s" style="background:{col}1a;color:{col}">{sev}</span>
                  </div>
                  <div class="diag-b">{body}</div>
                  <div class="diag-r">→ {rec}</div>
                </div>
                """, unsafe_allow_html=True)

        # ── health ───────────────────────────────────────────────────
        st.markdown("<div style='height:1.1rem'></div>", unsafe_allow_html=True)
        rule_header("Vehicle health", "lowest first")

        if not health_list:
            st.markdown('<div class="empty">No scores yet</div>',
                        unsafe_allow_html=True)
        else:
            for h in sorted(health_list, key=lambda x: x.get("score", 100))[:7]:
                s = h.get("score", 100)
                c = "#ef4444" if s < 40 else "#f59e0b" if s < 70 else "#22c55e"
                st.markdown(f"""
                <div class="hp">
                  <div class="hp-t">
                    <span class="hp-id">{h.get('taxi_id','—')}</span>
                    <span class="hp-v" style="color:{c}">{s}</span>
                  </div>
                  <div class="hp-track">
                    <div class="hp-fill" style="background:{c};width:{s}%"></div>
                  </div>
                </div>
                """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════
# Fleet Assistant
# ═════════════════════════════════════════════════════════════════════
with tab_agent:
    rule_header("Fleet assistant", "LangGraph ReAct · 4 tools")
    st.markdown("""
    <div style="font-size:.73rem;color:#6b7280;line-height:1.6;margin-bottom:.9rem">
      Queries vehicle history, maintenance records and semantically similar past
      anomalies before answering. Responses typically take 15–30 seconds.
    </div>
    """, unsafe_allow_html=True)

    q_col, b_col = st.columns([4, 1])
    with q_col:
        question = st.text_input(
            "q", placeholder="Which vehicle needs attention most urgently?",
            label_visibility="collapsed",
        )
    with b_col:
        ask = st.button("Ask", use_container_width=True)

    st.markdown("""
    <div style="font-size:.64rem;color:#374151;margin-top:.45rem">
      Maintenance status of taxi 20000154 &nbsp;·&nbsp;
      Summarise fleet health &nbsp;·&nbsp;
      Find anomalies similar to a high-speed spike
    </div>
    """, unsafe_allow_html=True)

    if ask and question:
        with st.spinner("Reasoning…"):
            try:
                resp = requests.post(f"{API_URL}/agent",
                                     json={"question": question}, timeout=90)
                answer = resp.json().get("answer", "No answer received.")

                parsed = None
                match = re.search(r"\{.*\}", answer, re.DOTALL)
                if match:
                    try:
                        parsed = json.loads(match.group())
                    except json.JSONDecodeError:
                        parsed = None

                st.markdown("<div style='height:.9rem'></div>", unsafe_allow_html=True)

                if parsed:
                    sev = parsed.get("severity", "—")
                    col = SEV_COLOUR.get(sev, "#6b7280")
                    st.markdown(f"""
                    <div class="diag" style="border-left-color:{col};padding:.9rem 1.05rem">
                      <div class="diag-h">
                        <span class="diag-id">{parsed.get('taxi_id','—')}
                          <span style="color:#4b5563;font-weight:400">
                            · {str(parsed.get('anomaly_type','')).replace('_',' ')}</span></span>
                        <span class="diag-s" style="background:{col}1a;color:{col}">{sev}</span>
                      </div>
                      <div class="diag-b" style="margin-bottom:.4rem">
                        {parsed.get('diagnosis','—')}</div>
                      <div class="diag-r">→ {parsed.get('recommendation','—')}</div>
                      <div style="font-size:.63rem;color:#4b5563;margin-top:.55rem;
                                  padding-top:.45rem;border-top:1px solid #1b2130">
                        Maintenance: {parsed.get('maintenance_status','—')}
                        &nbsp;·&nbsp; Confidence: {parsed.get('confidence','—')}
                      </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(
                        f'<div class="diag" style="border-left-color:#3b82f6;'
                        f'padding:1rem 1.15rem"><div class="diag-b" '
                        f'style="white-space:pre-wrap">{answer}</div></div>',
                        unsafe_allow_html=True,
                    )
            except requests.exceptions.Timeout:
                st.error("Agent timed out — try a narrower question.")
            except Exception as exc:
                st.error(f"Request failed: {exc}")
    elif ask:
        st.warning("Enter a question first.")

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)
    rule_header("Diagnosis log", f"{len(diag_list)} entries")

    if not diag_list:
        st.markdown('<div class="empty">No diagnoses yet</div>', unsafe_allow_html=True)
    else:
        for d in diag_list:
            dg = d.get("diagnosis", {})
            sev = dg.get("severity", "—") if isinstance(dg, dict) else "—"
            label = (f"{d.get('taxi_id','—')}  ·  "
                     f"{str(d.get('anomaly_type','')).replace('_',' ')}  ·  {sev}")
            with st.expander(label):
                if isinstance(dg, dict):
                    for k in ("diagnosis", "recommendation",
                              "maintenance_status", "confidence"):
                        st.markdown(f"**{k.replace('_',' ').title()}** — {dg.get(k,'—')}")
                else:
                    st.write(str(dg))


# ═════════════════════════════════════════════════════════════════════
# Detector Comparison
# ═════════════════════════════════════════════════════════════════════
with tab_model:
    rule_header("Detector comparison", "4.31M pings · 443 taxis · 5-fold GroupKFold")

    st.markdown("""
    <div style="font-size:.73rem;color:#6b7280;line-height:1.65;margin-bottom:1rem">
      Both detectors run in parallel on every ping. Threshold rules stay as an
      interpretable safety net; Isolation Forest evaluates all six features jointly
      and surfaces multivariate anomalies no single threshold can express.
    </div>
    """, unsafe_allow_html=True)

    comparison = pd.DataFrame([
        {"Detector": "Rule-based", "Features": "speed, jump_distance",
         "F1 (CV)": "1.000 ref", "Speed confound r": "+0.879",
         "Signature": "threshold crossing", "Status": "active"},
        {"Detector": "IF · absolute", "Features": "6 raw",
         "F1 (CV)": "0.738 ± 0.003", "Speed confound r": "+0.873",
         "Signature": "fleet-relative outlier", "Status": "superseded"},
        {"Detector": "IF · normalised v1", "Features": "4 scaled + vx, vy",
         "F1 (CV)": "0.731 ± 0.009", "Speed confound r": "+0.477",
         "Signature": "partial correction", "Status": "superseded"},
        {"Detector": "IF · normalised v2", "Features": "6 per-vehicle scaled",
         "F1 (CV)": "0.718 ± 0.011", "Speed confound r": "−0.002",
         "Signature": "3.27× own baseline", "Status": "deployed"},
        {"Detector": "LSTM autoencoder", "Features": "10-ping sequences",
         "F1 (CV)": "—", "Speed confound r": "−0.167",
         "Signature": "0.648× own baseline", "Status": "rejected"},
    ])
    st.dataframe(comparison, use_container_width=True, hide_index=True)

    st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2, gap="medium")

    with c1:
        rule_header("Speed confound", "0 = independent of driving speed")
        names = ["Rule-based", "IF absolute", "IF norm v1", "IF norm v2", "LSTM"]
        vals  = [0.879, 0.873, 0.477, -0.002, -0.167]
        cols  = ["#4b5563", C_NORMAL, C_RULE, "#22c55e", C_BOTH]

        f = go.Figure(go.Bar(
            x=names, y=vals, marker=dict(color=cols),
            text=[f"{v:+.3f}" for v in vals], textposition="outside",
            textfont=dict(size=10, color="#6b7280"),
            hovertemplate="%{x}: r = %{y:+.3f}<extra></extra>",
        ))
        f.add_hline(y=0, line_color="#374151", line_width=1)
        f.update_layout(
            height=252, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=18, b=0),
            font=dict(size=9.5, color="#6b7280"),
            yaxis=dict(range=[-.32, 1.05], showgrid=True,
                       gridcolor="#151a24", zeroline=False,
                       title=dict(text="correlation with mean speed",
                                  font=dict(size=9))),
            xaxis=dict(showgrid=False, title=""),
            bargap=.45, showlegend=False,
        )
        st.plotly_chart(f, use_container_width=True, config={"displayModeBar": False})

    with c2:
        rule_header("Detection signature", "ratio to vehicle's own average")

        f2 = go.Figure(go.Bar(
            x=["IF norm v2", "LSTM"], y=[3.27, 0.648],
            marker=dict(color=["#22c55e", C_BOTH]),
            text=["3.27×", "0.648×"], textposition="outside",
            textfont=dict(size=11, color="#9ca3af"),
            width=[.42, .42],
            hovertemplate="%{x}: %{y:.2f}× baseline<extra></extra>",
        ))
        f2.add_hline(y=1.0, line_dash="dash", line_color="#4b5563", line_width=1.2,
                     annotation_text="own baseline",
                     annotation_font=dict(size=9, color="#4b5563"),
                     annotation_position="right")
        f2.update_layout(
            height=252, template="plotly_dark",
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=18, b=0),
            font=dict(size=9.5, color="#6b7280"),
            yaxis=dict(range=[0, 4.1], showgrid=True, gridcolor="#151a24",
                       zeroline=False,
                       title=dict(text="median speed ratio", font=dict(size=9))),
            xaxis=dict(showgrid=False, title=""),
            showlegend=False,
        )
        st.plotly_chart(f2, use_container_width=True, config={"displayModeBar": False})

    st.markdown("""
    <div style="font-size:.7rem;color:#6b7280;line-height:1.7;
                border-left:2px solid #1b2130;padding-left:.9rem;margin-top:.4rem">
      <b style="color:#9ca3af">Why v2 is deployed.</b>
      Rule-based detection and the absolute-feature model both correlate with mean
      driving speed at r ≈ 0.87 — they largely surface fast vehicles rather than
      faulty sensors, because <code>jump_distance ≈ speed × 15 s</code>. Expressing
      every feature relative to each vehicle's own baseline removes the correlation
      (r = −0.002) for 0.020 F1.<br><br>
      <b style="color:#9ca3af">Why the LSTM was rejected.</b>
      It reconstructs known anomalies 43× worse than normal pings, but its exclusive
      detections sit at 0.648× each vehicle's own average — vehicles decelerating,
      not malfunctioning. An autoencoder flags statistical rarity, and transitions are
      rare in a dataset dominated by steady cruising. Rarity is not abnormality.
    </div>
    """, unsafe_allow_html=True)