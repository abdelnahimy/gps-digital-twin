# 🚗 GPS Digital Twin with LLM Agent

> Internship project at **iDIRTECH** — real-time fleet vehicle tracking using Kalman filtering
> and an LLM agent that automatically diagnoses GPS anomalies in natural language.

---

## Table of Contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Data](#data)
- [Stack](#stack)
- [How it works](#how-it-works)
- [Storage Schema](#storage-schema)
- [API Endpoints](#api-endpoints)
- [Docker](#docker)
- [8-Week Plan](#8-week-plan)
- [Improvements & Future Work](#improvements--future-work)

---

## What it does

A **passive → active** digital twin for a vehicle fleet:

- **Passive:** mirrors the real GPS state of each vehicle in real time (position, velocity)
- **Active:** detects when reality deviates from what physics predicts, then reasons about *why* using an LLM agent

A Kalman filter predicts where a vehicle *should* be next. When the raw GPS ping arrives and it's far from the prediction — that spike is the anomaly signal. The LLM agent then investigates: it queries vehicle maintenance history, road context at that location, and similar past events, then writes a plain-language French diagnosis for the fleet manager.

---

## Architecture

```
┌──────────────────────────────────────┐
│           GPS Simulator              │
│   Replays Porto taxi pings           │
└─────────────┬────────────────────────┘
              │  Kafka topic: gps-raw
              ▼
┌──────────────────────────────────────┐
│      Spark Structured Streaming      │
│  - UTM projection (lat/lon → meters) │
│  - OSMnx road context enrichment     │
│  - Kalman filter per vehicle         │
│  - Computes innovation_norm          │
└─────────────┬────────────────────────┘
              │
              ▼
┌──────────────────────────────────────┐
│           Anomaly Check              │
│    innovation_norm > threshold ?     │
└──────┬───────────────────┬───────────┘
       │ normal            │ anomaly
       ▼                   ▼
┌─────────────┐   ┌────────────────────┐
│ Twin Store  │   │     LLM Agent      │
│  MongoDB    │   │ LangGraph+GPT-4o   │
└──────┬──────┘   │ → French diagnosis │
       │          └────────┬───────────┘
       └──────────┬─────────┘
                  ▼
┌──────────────────────────────────────┐
│              MongoDB                 │
│  Twin states · Diagnoses · Maintenance│
└──────────────┬───────────────────────┘
               ▼
┌──────────────────────────────────────┐
│     FastAPI + WebSocket + Streamlit  │
│   Live map · Twin state · Agent feed │
└──────────────────────────────────────┘
```

---

## Project Structure

```
gps-digital-twin/
├── .env                            # credentials (gitignored)
├── .gitignore
├── docker-compose.yml
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/                        # original train.csv (gitignored)
│   ├── processed/                  # porto_pings_50taxis.csv
│   └── road_network/
│       └── porto_road_network.graphml
│
├── simulator/
│   ├── preprocess.py               # POLYLINE → one row per ping
│   └── gps_simulator.py            # Kafka producer replaying pings
│
├── processing/
│   ├── spark_pipeline.py           # Spark Structured Streaming job
│   ├── projection.py               # lat/lon → UTM meters
│   └── road_enrichment.py          # OSMnx nearest-edge lookups
│
├── twin/
│   ├── kalman_filter.py            # VehicleTwin, create_vehicle_filter()
│   ├── twin_registry.py            # one filter per vehicle
│   └── anomaly_detector.py         # is_anomaly(), inject_anomaly()
│
├── agent/
│   ├── tools.py                    # query_vehicle_history, query_road_context, find_similar_anomalies
│   ├── graph.py                    # LangGraph agent definition
│   ├── prompts.py                  # system prompt
│   └── memory.py                   # ChromaDB setup
│
├── storage/
│   ├── mongo_client.py             # connection + index creation
│   ├── schemas.py                  # document shape reference
│   └── seed_maintenance.py         # generate synthetic maintenance history
│
├── api/
│   ├── main.py                     # FastAPI — REST + WebSocket
│   └── Dockerfile
│
├── dashboard/
│   ├── app.py                      # Streamlit dashboard
│   └── Dockerfile
│
├── notebooks/
│   └── exploration.ipynb           # EDA, threshold tuning
│
└── tests/
    ├── test_kalman_filter.py
    ├── test_anomaly_detector.py
    └── test_agent_tools.py
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

Copy `.env.example` to `.env` and fill in your OpenAI key:

```
OPENAI_API_KEY=sk-...
MONGO_URI=mongodb://localhost:27017
KAFKA_BOOTSTRAP_SERVERS=localhost:9092
INNOVATION_THRESHOLD=50.0
```

### 3. Start infrastructure

```bash
docker-compose up -d zookeeper kafka mongodb
```

### 4. Prepare data

```bash
# preprocess Porto dataset into one row per ping
python simulator/preprocess.py

# download Porto road network (one time, ~few minutes)
python processing/road_enrichment.py --download

# seed synthetic maintenance history into MongoDB
python storage/seed_maintenance.py
```

### 5. Run the pipeline

```bash
# terminal 1 — Spark processor
python processing/spark_pipeline.py

# terminal 2 — GPS simulator
python simulator/gps_simulator.py

# terminal 3 — API
uvicorn api.main:app --reload

# terminal 4 — Dashboard
streamlit run dashboard/app.py
```

Dashboard at **http://localhost:8501**

---

## Data

### Porto Taxi Trajectory dataset

| Field | Description |
|---|---|
| `TAXI_ID` | vehicle identifier |
| `TIMESTAMP` | trip start (Unix seconds) |
| `POLYLINE` | list of `[lon, lat]` pairs, one every 15 seconds |
| `MISSING_DATA` | trips with missing pings — filtered out |

- **1,710,670** raw trips · **442** taxis · Jul 2013 – Jun 2014
- Download: [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/339/taxi+service+trajectory+prediction+challenge+ecml+pkdd+2015)
- Dev subset: **50 taxis** → **228,330 individual GPS pings**

### Porto Road Network

Downloaded once via OSMnx and saved as a `.graphml` file. Used by the Spark pipeline for road type and speed limit enrichment, and by the LLM agent for location context.

### Synthetic Maintenance History

Generated once and stored in MongoDB. Each of the 50 vehicles gets a realistic record including vehicle model, mileage, last oil change, brake inspection, and last known anomaly. The LLM agent queries this when diagnosing.

---

## Stack

| Layer | Tool | Purpose |
|---|---|---|
| Ingestion | Apache Kafka | GPS stream message broker |
| Ingestion | confluent-kafka | Python Kafka producer/consumer |
| Processing | PySpark | Structured Streaming pipeline |
| Spatial | OSMnx | Road network queries |
| Spatial | pyproj | UTM coordinate projection |
| Spatial | Shapely, H3 | Geometry operations |
| Twin | FilterPy | Kalman filter |
| Agent | LangGraph | Agent graph orchestration |
| Agent | LangChain | Tool wrappers, prompt templates |
| Agent | OpenAI GPT-4o-mini | LLM reasoning |
| Agent | ChromaDB | Vector store for agent memory |
| Storage | MongoDB | Twin states, diagnoses, maintenance |
| Storage | Motor | Async MongoDB driver |
| API | FastAPI + Uvicorn | REST endpoints + WebSocket |
| Dashboard | Streamlit + Folium | Live map and agent feed |
| Infrastructure | Docker Compose | Full stack orchestration |

---

## How it works

### 1. GPS Simulator

Reads `porto_pings_50taxis.csv` and replays each ping into Kafka at real-time speed (respecting the original 15-second interval between pings). Each Kafka message contains: `vehicle_id`, `timestamp`, `latitude`, `longitude`.

### 2. Spark Pipeline

Consumes the Kafka stream and for each ping:
- Projects coordinates from lat/lon into UTM meters (Kalman filter needs Euclidean space)
- Looks up the nearest road via OSMnx (road type, speed limit)
- Runs the Kalman filter predict → update cycle for that vehicle
- Computes the `innovation_norm` — the distance in meters between where the filter predicted the vehicle to be and where it actually is
- Writes the enriched twin state to MongoDB

### 3. Kalman Filter

Tracks each vehicle's state as `[x, y, vx, vy]` — position and velocity in meters. The filter maintains a running prediction of where the vehicle should be based on its last known velocity. The **innovation** is the gap between that prediction and the actual GPS reading. Under normal driving this gap is small (5–20m, just GPS noise). A spike above the threshold means something physically unexpected happened.

### 4. Anomaly Detection

If `innovation_norm` exceeds the tuned threshold, the ping is flagged as anomalous and the full context is sent to the LLM agent. Anomaly types the system can detect: GPS jumps, frozen position (repeated coordinates), impossible speed (teleportation), and route deviation.

### 5. LLM Agent

A LangGraph ReAct agent with 3 tools:

| Tool | What it queries | Why |
|---|---|---|
| `query_vehicle_history` | MongoDB maintenance + recent twin states | vehicle context |
| `query_road_context` | OSMnx nearest edge at anomaly location | road type, speed limit |
| `find_similar_anomalies` | ChromaDB past diagnoses | comparable past events |

After calling all 3 tools, GPT-4o-mini synthesizes the results into a 2–3 sentence French diagnosis with a confidence level (HIGH / MEDIUM / LOW). The diagnosis is saved to the `diagnoses` MongoDB collection and stored in ChromaDB for future memory.

### 6. Dashboard

Streamlit app with two panels:
- **Live map** — Porto city map with vehicle markers (blue = normal, red = anomaly)
- **Anomaly feed** — expandable list of recent anomalies with their LLM diagnosis

---

## Storage Schema

### Collections

**`twin_states`** — one document per GPS ping

| Field | Description |
|---|---|
| `vehicle_id` | which vehicle |
| `timestamp` | Unix seconds |
| `x_est`, `y_est` | Kalman-filtered position (meters, UTM) |
| `vx_est`, `vy_est` | estimated velocity |
| `innovation_norm` | anomaly signal in meters |
| `road_type` | from OSMnx |
| `speed_limit` | from OSMnx |
| `is_anomaly` | boolean flag |
| `created_at` | for TTL auto-deletion |

**`diagnoses`** — one document per anomaly event

| Field | Description |
|---|---|
| `vehicle_id` | which vehicle |
| `timestamp` | when it happened |
| `innovation_norm` | severity in meters |
| `diagnosis` | French text from LLM |
| `road_type` | context at anomaly location |
| `agent_steps` | how many tool calls the agent made |

**`maintenance`** — one document per vehicle (synthetic)

| Field | Description |
|---|---|
| `vehicle_id` | which vehicle |
| `vehicle_model` | e.g. Renault Kangoo |
| `mileage_km` | total mileage |
| `oil_change` | date of last service |
| `last_anomaly` | date and type of last known event |

### Indexes

- `twin_states`: compound index on `(vehicle_id, timestamp)` for fast per-vehicle queries
- `diagnoses`: compound index on `(vehicle_id, timestamp)`
- `maintenance`: unique index on `vehicle_id`
- `twin_states`: TTL index on `created_at` — auto-deletes documents older than 7 days

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/vehicles` | list all active vehicle IDs |
| GET | `/twins/{vehicle_id}` | latest twin state for a vehicle |
| GET | `/alerts?limit=20` | most recent anomaly events |
| GET | `/diagnoses/{vehicle_id}` | LLM diagnoses for a vehicle |
| WS | `/ws/live` | live stream of new twin states |

---

## Docker

### Services

| Service | Image | Port |
|---|---|---|
| zookeeper | confluentinc/cp-zookeeper:7.4.0 | 2181 |
| kafka | confluentinc/cp-kafka:7.4.0 | 9092 |
| mongodb | mongo:7 | 27017 |
| api | ./api | 8000 |
| dashboard | ./dashboard | 8501 |

### AWS EC2 deployment

Minimum recommended: `t3.medium` (Ubuntu 22.04). Install Docker, clone the repo, add your `.env`, then `docker-compose up -d`. Dashboard accessible at `http://YOUR_EC2_IP:8501`.

---

## 8-Week Plan

| Week | Focus | Deliverable |
|---|---|---|
| 1 | Data pipeline + GPS simulator | Kafka → Spark working end-to-end |
| 2 | Kalman filter + twin state | MongoDB filling with `innovation_norm` |
| 3 | Road enrichment + anomaly detection | Anomaly events + F1 score documented |
| 4 | LLM agent v1 + rule-based baseline | Diagnoses in MongoDB + comparison table |
| 5 | Agent memory + ChromaDB | Agent querying past similar anomalies |
| 6 | FastAPI + WebSocket + Streamlit | Dashboard live at localhost:8501 |
| 7 | Docker + AWS + demo script | Public URL + demo video |
| 8 | Report + presentation + buffer | GitHub repo, report PDF, slides |

---

## Improvements & Future Work

### ✅ Must-Do (Week 2–3)

**F1 Evaluation — Anomaly Detector Quality Metric**
Synthetic anomalies are injected into the clean dataset (GPS jumps, frozen positions, impossible speeds). The Kalman filter runs over them and its predictions are compared against the known ground truth labels. A threshold sweep from 10m to 200m produces a precision/recall/F1 curve, and the best threshold is selected from that curve. This gives a concrete, defensible metric for the jury rather than a subjective "it works."

**Unit Tests — Kalman Filter**
Three pytest tests covering: filter matrix shapes, convergence to near-zero innovation for a stationary vehicle, and large innovation output for a sudden 500m GPS jump. Located in `tests/test_kalman_filter.py`. Run with `pytest tests/ -v`.

---

### 🔶 Should-Do (Week 4–5)

**Vehicle Health Score**
A persistent `health_score` (0–100) per vehicle stored in a new `vehicle_health` MongoDB collection. Decremented on each anomaly, weighted by severity (a 50m anomaly costs fewer points than a 200m jump). Surfaces in the dashboard and gives the LLM agent additional context when diagnosing repeat offenders.

**Rule-Based Baseline Explainer**
A simple if/else explainer that runs alongside the LLM agent. Takes the same anomaly context and applies fixed rules: spike above 200m → GPS dropout, speed above 1.5x limit → speeding, etc. Running both on the same 10 test scenarios and comparing outputs side-by-side in the report proves the LLM adds genuine value over a naive fallback.

**MongoDB TTL Index**
Auto-deletes `twin_states` documents older than 7 days. One index creation at startup prevents storage from growing unbounded during long demo runs.

**Credentials in `.env`**
All API keys and connection strings loaded from a `.env` file via `python-dotenv`. Never hardcoded in source. `.env` is gitignored. This is a basic security requirement for any production-adjacent project.

---

### 💡 Nice-to-Have (Week 6–8, if time allows)

| Improvement | Description | Effort |
|---|---|---|
| Alert lifecycle | Acknowledge / dismiss anomaly alerts in the dashboard | Medium |
| Adaptive Q matrix | Tune process noise dynamically by road type — fewer false positives | High |
| JWT auth on API | Token-based auth on FastAPI endpoints | Medium |
| Pre-load OSMnx graph | Cache road network in memory at startup instead of per-query loading | Low |
| Multi-hypothesis tracking | Track multiple possible vehicle states simultaneously | Very High |

---

*Built during 8-week internship at iDIRTECH — July 2026*
*Stack: Kafka · Spark · FilterPy · LangGraph · GPT-4o-mini · MongoDB · FastAPI · Streamlit*
