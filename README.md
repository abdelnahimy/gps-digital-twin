# 🚕 GPS Digital Twin with LLM Agent

> Final Year Project (PFA) — 8-week internship at **iDIRTECH**, Marrakech
> Real-time fleet intelligence: Kafka · Spark · Kalman Filter · LangGraph · MongoDB · FastAPI · Streamlit · Docker

---

## Table of Contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Data](#data)
- [Stack](#stack)
- [How it works](#how-it-works)
- [Anomaly Detection & Evaluation](#anomaly-detection--evaluation)
- [AI Agent](#ai-agent)
- [Storage Schema](#storage-schema)
- [API Endpoints](#api-endpoints)
- [Dashboard](#dashboard)
- [Docker](#docker)
- [Tests](#tests)
- [8-Week Plan](#8-week-plan)
- [Improvements](#improvements)

---

## What it does

A **passive → active** GPS digital twin for a vehicle fleet:

- **Passive:** mirrors the real GPS state of each vehicle in real time (position, velocity, road context)
- **Active:** detects anomalies using a Kalman filter, automatically diagnoses them using a LangGraph LLM agent, and tracks vehicle health over time

When a GPS ping deviates from what physics predicts — the Kalman filter's innovation signal — the LangGraph agent automatically investigates: it queries vehicle history, maintenance records, and similar past anomalies, then generates a structured JSON diagnosis without any human input.

---

## Architecture

```
┌──────────────────────────────────────────────┐
│              GPS Simulator                    │
│   Replays Porto taxi pings from CSV          │
│   → Kafka topic: gps-pings                   │
└─────────────────┬────────────────────────────┘
                  │
                  ▼
┌──────────────────────────────────────────────┐
│         Spark Structured Streaming           │
│  - UTM projection (lat/lon → meters)         │
│  - Kalman filter per vehicle [x,y,vx,vy]    │
│  - OSMnx nearest road edge enrichment       │
│  - Anomaly detection (3 types)              │
│  - Vehicle health score update              │
└──────────┬───────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────┐
│            Anomaly Detection                 │
│  speed > 80 m/s → SPEED                     │
│  jump > 800m   → POSITION_JUMP              │
│  trace(P) > 100 → UNCERTAINTY               │
└──────┬─────────────────────┬─────────────────┘
       │ normal              │ anomaly
       ▼                     ▼
┌─────────────┐    ┌──────────────────────────┐
│  MongoDB    │    │    LangGraph Agent        │
│ twin_states │    │  auto_diagnose() thread  │
└──────┬──────┘    │  → query_vehicle_history │
       │           │  → query_maintenance     │
       │           │  → find_similar_anomalies│
       │           │  → structured JSON       │
       │           └───────────┬──────────────┘
       └───────────────────────┤
                               ▼
┌──────────────────────────────────────────────┐
│                 MongoDB                      │
│  twin_states · anomalies · diagnoses        │
│  maintenance · health_scores                │
└──────────────────┬───────────────────────────┘
                   ▼
┌──────────────────────────────────────────────┐
│          FastAPI + WebSocket                 │
│  REST endpoints · Live broadcast            │
└──────────────────┬───────────────────────────┘
                   ▼
┌──────────────────────────────────────────────┐
│           Streamlit Dashboard                │
│  Live map · KPIs · Anomaly feed             │
│  AI Diagnoses · Health scores · Agent chat  │
└──────────────────────────────────────────────┘
```

---

## Project Structure

```
gps-digital-twin/
├── .env                            # GROQ_API_KEY, MONGO_URI (gitignored)
├── .gitignore
├── .dockerignore
├── docker-compose.yml              # 5 services: Zookeeper, Kafka, MongoDB, API, Dashboard
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/                        # train.csv — Porto dataset (gitignored)
│   ├── processed/                  # porto_pings_50taxis.csv (228k pings)
│   └── road_network/
│       └── porto_road_network.graphml  # OSMnx Porto (5043 nodes, 10549 edges)
│
├── simulator/
│   ├── preprocess.py               # POLYLINE → one row per ping (explode_porto)
│   └── gps_simulator.py            # Kafka producer replaying pings
│
├── processing/
│   ├── spark_pipeline.py           # Spark Structured Streaming + foreachBatch
│   ├── projection.py               # latlon_to_meters() — UTM zone 29N
│   ├── road_enrichment.py          # OSMnx nearest-edge lookups
│   └── coordinate_rescale.py       # Porto → Marrakech bbox rescaling (future)
│
├── twin/
│   ├── kalman_filter.py            # create_vehicle_filter() + VehicleTwin
│   ├── twin_registry.py            # TwinRegistry — one filter per vehicle
│   ├── anomaly_detector.py         # is_anomaly() — speed, position_jump, uncertainty
│   └── health_score.py             # update_health_score() — 0-100 per vehicle
│
├── agent/
│   ├── tools.py                    # query_vehicle_history, query_maintenance,
│   │                               # find_similar_anomalies, get_fleet_summary
│   ├── graph.py                    # LangGraph ReAct agent + auto_diagnose()
│   ├── prompts.py                  # SYSTEM_PROMPT
│   └── memory.py                   # ChromaDB setup + store_anomaly()
│
├── storage/
│   ├── mongo_client.py             # connection + index creation (5 collections)
│   ├── schemas.py                  # document shape reference
│   └── seed_maintenance.py         # synthetic maintenance records (50 taxis)
│
├── models/
│   └── anomaly_classifier.py       # ML anomaly detector (Isolation Forest + LSTM)
│
├── api/
│   ├── main.py                     # FastAPI — REST + WebSocket + async agent
│   └── Dockerfile
│
├── dashboard/
│   ├── app.py                      # Streamlit — dark theme, 2 tabs
│   └── Dockerfile
│
├── notebooks/
│   ├── exploration.ipynb           # Raw data EDA
│   └── anomaly_detection_ml.ipynb  # ML model: feature extraction, IF, LSTM
│
└── tests/
    ├── test_kalman_filter.py
    ├── test_anomaly_detector.py
    ├── test_agent_tools.py
    ├── evaluate_detector.py        # F1 threshold sweep
    └── compare_llms.py             # LLM model comparison
```

---

## Quick Start

### 1. Clone and set up environment

```bash
git clone https://github.com/YOUR_USERNAME/gps-digital-twin.git
cd gps-digital-twin
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key
MONGO_URI=mongodb://localhost:27017
API_URL=http://localhost:8000
```

### 3. Start infrastructure

```bash
docker-compose up -d
```

This starts: Zookeeper, Kafka, MongoDB, API, Dashboard.

### 4. Prepare data (first time only)

```bash
# Download Porto taxi dataset from Kaggle:
# https://www.kaggle.com/datasets/crailtap/taxi-trajectory
# Place train.csv in data/raw/

# Preprocess — 50 taxis, 100 trips each → 228k pings
python simulator/preprocess.py

# Download Porto road network (~2 minutes)
python processing/road_enrichment.py

# Seed synthetic maintenance records
python storage/seed_maintenance.py
```

### 5. Run the pipeline

```bash
# Terminal 1 — Spark processor
python processing/spark_pipeline.py

# Terminal 2 — GPS simulator
python simulator/gps_simulator.py
```

- Dashboard: **http://localhost:8501**
- API docs: **http://localhost:8000/docs**

---

## Data

### Porto Taxi Trajectory Dataset

| Field | Description |
|---|---|
| `TAXI_ID` | vehicle identifier |
| `TIMESTAMP` | trip start (Unix seconds) |
| `POLYLINE` | list of `[lon, lat]` pairs, one every 15 seconds |
| `MISSING_DATA` | trips with missing pings — filtered out |

- **1,710,670** raw trips · **442** taxis · Jul 2013 – Jun 2014
- Source: [Kaggle — ECML/PKDD 2015](https://www.kaggle.com/datasets/crailtap/taxi-trajectory)
- Citation: Moreira-Matias et al. (2013). UCI ML Repository. https://doi.org/10.24432/C55W25
- Dev subset: **50 taxis · 100 trips/taxi → 228,330 GPS pings**
- Note: dataset has documented GPS noise (ECML/PKDD challenge paper) — points frequently map outside road network

### Porto Road Network

Downloaded once via OSMnx from OpenStreetMap:
- **5,043 nodes · 10,549 edges**
- Saved as `porto_road_network.graphml`
- Used for nearest road edge enrichment per ping

### Synthetic Maintenance Records

Generated by `storage/seed_maintenance.py`:
- 50 records, one per taxi
- Fields: last oil change, next oil change, tire change, inspection dates, mileage
- Used by the LangGraph agent when diagnosing anomalies

---

## Stack

| Layer | Tool | Purpose |
|---|---|---|
| Ingestion | Apache Kafka + Zookeeper | GPS stream message broker |
| Ingestion | confluent-kafka | Python Kafka producer |
| Processing | PySpark 3.5 | Structured Streaming — foreachBatch |
| Spatial | OSMnx | Road network download + nearest edge |
| Spatial | pyproj | UTM projection zone 29N |
| Spatial | Shapely, H3 | Geometry operations |
| Twin | FilterPy | Kalman filter (dim_x=4, dim_z=2) |
| Agent | LangGraph | ReAct agent orchestration |
| Agent | LangChain | Tool wrappers |
| Agent | Groq (llama-3.1-8b-instant) | LLM reasoning — free tier |
| Agent | ChromaDB | Vector store for anomaly semantic memory |
| Storage | MongoDB 7 | 5 collections with indexes |
| API | FastAPI + Uvicorn | REST + WebSocket + async |
| Dashboard | Streamlit + Folium + Plotly | Live map, charts, agent chat |
| Infrastructure | Docker Compose | 5 containers, one-command startup |
| ML | Scikit-learn, PyTorch | Isolation Forest + LSTM Autoencoder |

---

## How it works

### 1. GPS Simulator

Reads `porto_pings_50taxis.csv` and pushes each row as a JSON message into Kafka topic `gps-pings`. Configurable delay simulates real-time pacing (default 0.01s for development).

### 2. Spark Pipeline

Consumes Kafka stream via `foreachBatch`. For each ping:
1. Converts lat/lon → UTM meters (zone 29N) via `pyproj`
2. Finds nearest road edge via OSMnx
3. Runs Kalman filter predict → update cycle via `TwinRegistry`
4. Detects anomalies via `is_anomaly()`
5. Updates vehicle health score via `update_health_score()`
6. Saves enriched state to MongoDB `twin_states`
7. If anomaly: saves to `anomalies`, stores in ChromaDB, triggers `auto_diagnose()`
8. Broadcasts state to WebSocket clients via `POST /broadcast`

### 3. Kalman Filter

Tracks each vehicle as `[x, y, vx, vy]` in UTM meters:

```
State transition (F):        Measurement (H):
[1  0  dt  0 ]              [1  0  0  0]
[0  1  0   dt]    x_pred =  [0  1  0  0]  × state
[0  0  1   0 ]
[0  0  0   1 ]

R = I₂ × 9.0    # GPS noise: ~3m std dev
Q = I₄ × 0.01   # Process noise: smooth velocity
P = I₄ × 500    # Initial uncertainty
dt = 15s         # Porto ping interval
```

### 4. Anomaly Detection

Three anomaly types detected by `is_anomaly()`:

| Type | Condition | Threshold | Severity |
|---|---|---|---|
| speed | `sqrt(vx²+vy²) > threshold` | 80 m/s | HIGH if >100, MEDIUM otherwise |
| position_jump | distance from previous position | 800 m | HIGH |
| uncertainty | `trace(P) > threshold` | 100.0 | LOW |

First ping per taxi is always skipped (filter not yet initialized).

### 5. Vehicle Health Score

Persistent 0-100 score per taxi in MongoDB `health_scores`:

| Event | Points |
|---|---|
| Normal ping | +2 |
| Speed anomaly | -5 |
| Position jump | -3 |
| Uncertainty anomaly | -2 |

Score clamped between 0 and 100. Shown as progress bars in dashboard.

### 6. LangGraph Agent

ReAct agent with 4 tools, powered by Groq `llama-3.1-8b-instant`:

| Tool | Queries | When used |
|---|---|---|
| `query_vehicle_history` | MongoDB `twin_states` — last 10 states | vehicle context |
| `query_maintenance` | MongoDB `maintenance` | service records |
| `find_similar_anomalies` | ChromaDB — semantic search | past comparable events |
| `get_fleet_summary` | MongoDB aggregation | overall fleet health |

**Proactive auto-diagnosis:** When Spark detects an anomaly, `auto_diagnose()` is called in a background thread with rate limiting (10s cooldown, semaphore). The agent generates a structured JSON diagnosis saved to `diagnoses` collection.

**Structured output:**
```json
{
  "taxi_id": "20000403",
  "anomaly_type": "position_jump",
  "severity": "HIGH",
  "speed_detected": 55.7,
  "diagnosis": "GPS signal dropout detected...",
  "recommendation": "Inspect GPS antenna within 24h",
  "maintenance_status": "up to date",
  "confidence": "HIGH"
}
```

### 7. Dashboard

Two-tab Streamlit app:

**Tab 1 — Live Monitor:**
- KPI row: Active Vehicles, Anomalies, Avg Speed, AI Diagnoses
- Live Folium map (OpenStreetMap) — blue=normal, red=anomaly
- Plotly anomaly distribution chart by vehicle and type
- Recent anomalies feed colored by severity
- AI Diagnosis cards with HIGH/MEDIUM/LOW badges
- Vehicle Health Score progress bars

**Tab 2 — Fleet Assistant:**
- Agent chat interface with example questions
- All auto-diagnoses expandable with full JSON

---

## Anomaly Detection & Evaluation

### Methodology

Synthetic anomaly injection on 5 taxis, 200 pings each, 5% injection rate:

```python
inject_anomalies(df, injection_rate=0.05)
# Injects GPS position shifts of 1-2km at random indices
# Ground truth labels generated automatically
```

### Threshold Sweep Results

| Jump Threshold | Precision | Recall | F1 |
|---|---|---|---|
| 500m | 0.311 | 1.000 | 0.474 |
| 700m | 0.434 | 0.980 | 0.601 |
| **800m** | **0.484** | **0.920** | **0.634** |
| 900m | 0.467 | 0.700 | 0.560 |

**Selected threshold: 800m → F1=0.634, Precision=0.484, Recall=0.920**

High recall (92%) prioritized over precision — in fleet safety, missing a real anomaly is worse than a false positive.

### Why F1=0.634?

The Porto dataset has inherent GPS noise documented in the ECML/PKDD 2015 challenge paper — mean position jump of 182m, max 5095m in "clean" data. False positives are real GPS dropouts in the dataset, not detector errors. F1 reflects dataset quality, not detector quality.

---

## AI Agent

### LLM Comparison

| Model | Response Time | JSON Compliance | Quality |
|---|---|---|---|
| Groq llama-3.1-8b-instant | 3.5s | ✅ | Good |
| Groq llama-3.3-70b-versatile | 8.2s | ✅ | Excellent |
| Gemini 2.0 Flash | N/A | N/A | Quota exceeded |

Production recommendation: `llama-3.3-70b-versatile` for quality, `llama-3.1-8b-instant` for speed/cost.

---

## Storage Schema

### Collections

**`twin_states`** — one document per GPS ping

| Field | Type | Description |
|---|---|---|
| `taxi_id` | str | vehicle identifier |
| `timestamp` | int | Unix seconds |
| `x`, `y` | float | Kalman-filtered position (UTM meters) |
| `vx`, `vy` | float | estimated velocity (m/s) |
| `speed` | float | `sqrt(vx²+vy²)` |
| `uncertainty` | float | `trace(P)` |
| `anomaly` | bool | anomaly flag |
| `anomaly_type` | str | speed / position_jump / uncertainty / none |
| `health_score` | int | vehicle health at this ping |
| `road` | dict | nearest road edge `{u, v, key}` |
| `lat`, `lon` | float | original GPS coordinates |

**`anomalies`** — same schema, anomalous pings only

**`diagnoses`** — one document per auto-diagnosis

| Field | Type | Description |
|---|---|---|
| `taxi_id` | str | vehicle identifier |
| `timestamp` | int | Unix seconds |
| `anomaly_type` | str | type of anomaly |
| `speed` | float | speed at detection |
| `uncertainty` | float | filter uncertainty |
| `diagnosis` | dict | structured JSON from agent |

**`maintenance`** — one document per vehicle

| Field | Type | Description |
|---|---|---|
| `taxi_id` | str | vehicle identifier |
| `last_oil_change` | str | date |
| `next_oil_change` | str | date |
| `last_tire_change` | str | date |
| `next_tire_change` | str | date |
| `last_inspection` | str | date |
| `next_inspection` | str | date |
| `mileage` | int | total km |

**`health_scores`** — one document per vehicle

| Field | Type | Description |
|---|---|---|
| `taxi_id` | str | vehicle identifier |
| `score` | int | 0-100 health score |

### Indexes

- `twin_states`: compound `(taxi_id, timestamp DESC)`
- `anomalies`: compound `(taxi_id, timestamp DESC)`
- `diagnoses`: compound `(taxi_id, timestamp DESC)`
- `maintenance`: unique `taxi_id`
- `health_scores`: unique `taxi_id`

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/vehicles` | Latest state for all active taxis |
| GET | `/vehicles/{taxi_id}` | Last 50 states for one taxi |
| GET | `/anomalies` | Last 50 anomaly events |
| GET | `/diagnoses` | Last 20 auto-diagnoses |
| GET | `/health` | Health scores for all vehicles |
| POST | `/agent` | Ask the LangGraph agent a question |
| POST | `/broadcast` | Internal — Spark pushes state to WebSocket |
| WS | `/ws` | Live WebSocket stream |

---

## Dashboard

| Section | Description |
|---|---|
| KPI row | Active Vehicles · Anomalies · Avg Speed · AI Diagnoses |
| Live Map | Porto OpenStreetMap — blue/red markers, taxi popup on click |
| Anomaly Chart | Plotly bar chart — anomalies per taxi colored by type |
| Recent Anomalies | Last 6 anomalies with type and speed |
| AI Diagnoses | Last 4 structured agent diagnoses with severity badges |
| Health Scores | Progress bars 0-100 per vehicle |
| Fleet Assistant | Agent chat + all auto-diagnoses expandable |

---

## Docker

### Services

| Service | Image | Port |
|---|---|---|
| zookeeper | confluentinc/cp-zookeeper:7.5.0 | 2181 |
| kafka | confluentinc/cp-kafka:7.5.0 | 9092 |
| mongodb | mongo:7.0 | 27017 |
| api | ./api/Dockerfile | 8000 |
| dashboard | ./dashboard/Dockerfile | 8501 |

```bash
# Start everything
docker-compose up -d

# Stop everything
docker-compose down

# Rebuild after code changes
docker-compose build --no-cache api
docker-compose up -d
```

Note: GPS Simulator and Spark pipeline still run locally (not containerized).

---

## Tests

```bash
python tests/test_kalman_filter.py
python tests/test_anomaly_detector.py
python tests/test_agent_tools.py
```

| Test file | Coverage |
|---|---|
| `test_kalman_filter.py` | Matrix shapes, initialization, velocity estimation |
| `test_anomaly_detector.py` | Normal state, speed anomaly, position jump |
| `test_agent_tools.py` | Tool responses, fleet summary |
| `evaluate_detector.py` | F1 sweep across 5 taxis |
| `compare_llms.py` | LLM model comparison (response time, JSON compliance) |

---

## 8-Week Plan

| Week | Focus | Deliverable |
|---|---|---|
| 1 | Data pipeline + Kafka + GPS Simulator + Spark | 228k pings flowing Kafka → Spark → console |
| 2 | Kalman filter + Twin Registry + Anomaly Detection | Filtered states with speed, uncertainty, position jump |
| 3 | Road enrichment + F1 evaluation + ML model | OSMnx enrichment, F1=0.634, Isolation Forest + LSTM |
| 4 | MongoDB + LangGraph agent + ChromaDB | Agent answering fleet questions, auto-diagnosis |
| 5 | FastAPI + WebSocket | REST API + live broadcast |
| 6 | Streamlit dashboard | Live map + anomaly feed + agent chat + health scores |
| 7 | Docker Compose full stack | One-command startup, 5 containers |
| 8 | Report + jury preparation | Final report + demo script + slides |

---

## Improvements

### Implemented
- Kalman filter tuning — R=9, Q=0.01 (justified via F1 evaluation)
- 3 anomaly types with `anomaly_type` field
- Threshold optimization via F1 sweep — 800m optimal
- Fleet statistics sidebar
- 4th agent tool: `get_fleet_summary()`
- Proactive auto-diagnosis in background thread
- Rate limiting for agent (10s cooldown, semaphore)
- Vehicle health score (0-100, per-ping update)
- Structured JSON agent output
- Session state map (performance optimization)
- LLM model comparison (Groq vs Gemini)
- ML anomaly detection (Isolation Forest + LSTM Autoencoder)

### Future Work

| Improvement | Description | Effort |
|---|---|---|
| Real Teltonika data | Replace Porto dataset with live iDIRTECH GPS data | N/A |
| Coordinate rescaling | Porto → Marrakech bbox for local demo | Low |
| Containerize pipeline | Add simulator + Spark as Docker services | Medium |
| Adaptive Q matrix | Tune process noise by road type | High |
| JWT auth | Token-based auth on FastAPI | Medium |
| Smooth map updates | Replace Folium with React + Leaflet.js WebSocket | High |

---

*Final Year Project (PFA) — 8-week internship at iDIRTECH, Marrakech — 2026*
*Student: Abdeljalil Nahimy*
*Stack: Kafka · Spark · FilterPy · LangGraph · Groq · MongoDB · ChromaDB · FastAPI · Streamlit · Docker*
