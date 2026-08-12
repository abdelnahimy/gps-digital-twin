# 🚕 GPS Digital Twin with LLM Agent

> Final Year Project (PFA) — 8-week internship at **iDIRTECH**, Marrakech
> Real-time fleet intelligence: Kafka · Spark · Kalman Filter · Isolation Forest · LangGraph · MongoDB · FastAPI · Streamlit · Docker

---

## Table of Contents

- [What it does](#what-it-does)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Data](#data)
- [Stack](#stack)
- [How it works](#how-it-works)
- [Anomaly Detection — Rule-Based](#anomaly-detection--rule-based)
- [Anomaly Detection — Machine Learning](#anomaly-detection--machine-learning)
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
- **Active:** detects anomalies using a Kalman filter and unsupervised ML, automatically diagnoses them via a LangGraph LLM agent, and tracks vehicle health over time

When a GPS ping deviates from what physics predicts — the Kalman filter's innovation signal — the LangGraph agent automatically investigates: it queries vehicle history, maintenance records, and similar past anomalies, then generates a structured JSON diagnosis without any human input.

Detection runs through **two complementary paths**: interpretable threshold rules for the live pipeline, and an Isolation Forest trained on 4.3M pings that catches multivariate anomalies the rules cannot express.

---

## Architecture

```
┌───────────────────────────────────────────────────────────┐
│                     GPS Simulator                          │
│         Replays Porto taxi pings from CSV                 │
│              → Kafka topic: gps-pings                     │
└───────────────────────────┬───────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────┐
│              Spark Structured Streaming                    │
│  · UTM projection (lat/lon → meters, zone 29N)            │
│  · Kalman filter per vehicle → [x, y, vx, vy]             │
│  · OSMnx nearest road edge enrichment                     │
│  · Feature extraction (6 features per ping)               │
│  · Vehicle health score update                            │
└───────────────────────────┬───────────────────────────────┘
                            │
                            ▼
┌───────────────────────────────────────────────────────────┐
│                  ANOMALY DETECTION                         │
│                                                            │
│  ┌──────────────────────┐   ┌──────────────────────────┐  │
│  │   Rule-Based (live)  │   │  Isolation Forest (ML)   │  │
│  │  speed  > 80 m/s     │   │  6 features, joint space │  │
│  │  jump   > 800 m      │   │  contamination = 0.03    │  │
│  │  trace(P) > 100      │   │  F1 = 0.738 ± 0.003      │  │
│  │  interpretable, fast │   │  +62.8% coverage         │  │
│  └──────────┬───────────┘   └────────────┬─────────────┘  │
│             └───────────┬────────────────┘                 │
└─────────────────────────┼─────────────────────────────────┘
                          │
           ┌──────────────┴──────────────┐
           │ normal                      │ anomaly
           ▼                             ▼
┌─────────────────────┐   ┌──────────────────────────────────┐
│      MongoDB        │   │        LangGraph Agent            │
│    twin_states      │   │    auto_diagnose() — threaded    │
└──────────┬──────────┘   │  → query_vehicle_history         │
           │              │  → query_maintenance             │
           │              │  → find_similar_anomalies        │
           │              │  → get_fleet_summary             │
           │              │  → structured JSON diagnosis     │
           │              └───────────────┬──────────────────┘
           └──────────────────┬───────────┘
                              ▼
┌───────────────────────────────────────────────────────────┐
│                        MongoDB                             │
│   twin_states · anomalies · diagnoses                     │
│   maintenance · health_scores                             │
└───────────────────────────┬───────────────────────────────┘
                            ▼
┌───────────────────────────────────────────────────────────┐
│                 FastAPI + WebSocket                        │
│           REST endpoints · Live broadcast                 │
└───────────────────────────┬───────────────────────────────┘
                            ▼
┌───────────────────────────────────────────────────────────┐
│                  Streamlit Dashboard                       │
│   Live map · KPIs · Anomaly feed                          │
│   AI Diagnoses · Health scores · Agent chat               │
└───────────────────────────────────────────────────────────┘
```

### ML Model Architecture

```
                    Raw GPS ping (lat, lon)
                              │
                              ▼
                 UTM projection — zone 29N
                              │
                              ▼
              ┌───────────────────────────────┐
              │   Kalman Filter (per vehicle) │
              │   state = [x, y, vx, vy]      │
              │   F, H, R=9, Q=0.01, P=500    │
              └───────────────┬───────────────┘
                              │
                              ▼
              ┌───────────────────────────────┐
              │      Feature Extraction        │
              │  speed, vx, vy, jump_distance │
              │  rolling_mean, rolling_std     │
              └───────────────┬───────────────┘
                              │
              ┌───────────────┴───────────────┐
              ▼                               ▼
    ┌──────────────────┐          ┌──────────────────────┐
    │ ABSOLUTE features│          │ NORMALISED features  │
    │ speed            │          │ speed_z              │
    │ vx, vy           │          │ jump_z               │
    │ jump_distance    │          │ speed_ratio          │
    │ rolling_mean     │          │ rolling_std_ratio    │
    │ rolling_std      │          │ vx, vy               │
    └────────┬─────────┘          └──────────┬───────────┘
             │                               │
             ▼                               ▼
    ┌──────────────────┐          ┌──────────────────────┐
    │ StandardScaler   │          │ StandardScaler       │
    └────────┬─────────┘          └──────────┬───────────┘
             ▼                               ▼
    ┌──────────────────┐          ┌──────────────────────┐
    │ Isolation Forest │          │ Isolation Forest     │
    │ 100 trees        │          │ 100 trees            │
    │ contamination.03 │          │ contamination .03    │
    └────────┬─────────┘          └──────────┬───────────┘
             ▼                               ▼
      F1 = 0.738 ± 0.003            confound r = 0.477
      confound r = 0.873            (−45.4% vs absolute)
             │                               │
             └───────────────┬───────────────┘
                             ▼
              5-fold GroupKFold CV (by taxi)
              355 train taxis / 88 test taxis
```

---

## Project Structure

```
gps-digital-twin/
├── .env                            # GROQ_API_KEY, MONGO_URI (gitignored)
├── .gitignore
├── .dockerignore
├── docker-compose.yml              # 5 services
├── requirements.txt
├── README.md
│
├── data/
│   ├── raw/                        # train.csv — Porto dataset (gitignored)
│   ├── processed/                  # porto_pings_50taxis.csv (228k pings)
│   └── road_network/
│       └── porto_road_network.graphml
│
├── simulator/
│   ├── preprocess.py               # POLYLINE → one row per ping
│   └── gps_simulator.py            # Kafka producer
│
├── processing/
│   ├── spark_pipeline.py           # Spark Structured Streaming
│   ├── projection.py               # latlon_to_meters() — UTM 29N
│   ├── road_enrichment.py          # OSMnx nearest-edge
│   └── coordinate_rescale.py       # Porto → Marrakech (future)
│
├── twin/
│   ├── kalman_filter.py            # create_vehicle_filter() + VehicleTwin
│   ├── twin_registry.py            # TwinRegistry
│   ├── anomaly_detector.py         # is_anomaly() — 3 types
│   └── health_score.py             # update_health_score() — 0-100
│
├── agent/
│   ├── tools.py                    # 4 LangChain tools
│   ├── graph.py                    # ReAct agent + auto_diagnose()
│   ├── prompts.py                  # SYSTEM_PROMPT
│   └── memory.py                   # ChromaDB + store_anomaly()
│
├── storage/
│   ├── mongo_client.py             # 5 collections + indexes
│   ├── schemas.py
│   └── seed_maintenance.py
│
├── models/                         # trained ML artefacts
│   ├── isolation_forest.pkl        # absolute-feature model
│   ├── isolation_forest_normalised.pkl
│   ├── scaler.pkl
│   └── scaler_normalised.pkl
│
├── api/
│   ├── main.py                     # FastAPI REST + WebSocket
│   └── Dockerfile
│
├── dashboard/
│   ├── app.py                      # Streamlit — dark theme, 2 tabs
│   └── Dockerfile
│
├── notebooks/
│   ├── exploration.ipynb           # Raw data EDA
│   └── anomaly_detection_ml.ipynb  # Full ML study (46 cells)
│
└── tests/
    ├── test_kalman_filter.py
    ├── test_anomaly_detector.py
    ├── test_agent_tools.py
    ├── evaluate_detector.py        # F1 threshold sweep
    └── compare_llms.py             # LLM comparison
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

Starts Zookeeper, Kafka, MongoDB, API, Dashboard.

### 4. Prepare data (first time only)

```bash
# Download Porto taxi dataset from Kaggle:
# https://www.kaggle.com/datasets/crailtap/taxi-trajectory
# Place train.csv in data/raw/

python simulator/preprocess.py        # 50 taxis, 100 trips → 228k pings
python processing/road_enrichment.py  # OSMnx Porto network (~2 min)
python storage/seed_maintenance.py    # synthetic maintenance records
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

| Field          | Description                                      |
| -------------- | ------------------------------------------------ |
| `TAXI_ID`      | vehicle identifier                               |
| `TIMESTAMP`    | trip start (Unix seconds)                        |
| `POLYLINE`     | list of `[lon, lat]` pairs, one every 15 seconds |
| `MISSING_DATA` | trips with missing pings — filtered out          |

- **1,704,759** clean trips · **444** taxis · Jul 2013 – Jun 2014
- Source: [Kaggle — ECML/PKDD 2015](https://www.kaggle.com/datasets/crailtap/taxi-trajectory)
- Citation: Moreira-Matias et al. (2013). UCI ML Repository. https://doi.org/10.24432/C55W25

**Two scopes used:**

| Purpose            | Scope                 | Pings     | Runtime |
| ------------------ | --------------------- | --------- | ------- |
| Live pipeline demo | 50 taxis × 100 trips  | 228,330   | 1 min   |
| ML study           | 444 taxis × 200 trips | 4,311,098 | 6 min   |

Full extraction (~83M pings) requires ~2h because the Kalman filter is sequential and cannot be vectorised — each ping's prediction depends on the previous state. Cross-validation std of ±0.003 indicates the estimate had already converged at 4.3M.

**Documented GPS noise:** the ECML/PKDD challenge paper notes many points map outside the road network. Observed in our extraction: max speed 11,513 m/s (~41,000 km/h), max jump 172 km in one 15-second interval. Median values are realistic (6.3 m/s, 96 m).

### Porto Road Network

OSMnx / OpenStreetMap — **5,043 nodes · 10,549 edges**, saved as `porto_road_network.graphml`.

### Synthetic Maintenance Records

50 records generated by `storage/seed_maintenance.py` — oil change, tire change, inspection dates, mileage. Consumed by the LangGraph agent during diagnosis.

---

## Stack

| Layer          | Tool                        | Purpose                                          |
| -------------- | --------------------------- | ------------------------------------------------ |
| Ingestion      | Apache Kafka + Zookeeper    | GPS stream message broker                        |
| Ingestion      | confluent-kafka             | Python Kafka producer                            |
| Processing     | PySpark 3.5                 | Structured Streaming — foreachBatch              |
| Spatial        | OSMnx                       | Road network + nearest edge                      |
| Spatial        | pyproj                      | UTM projection zone 29N                          |
| Spatial        | Shapely, H3                 | Geometry operations                              |
| Twin           | FilterPy                    | Kalman filter (dim_x=4, dim_z=2)                 |
| **ML**         | **scikit-learn**            | **Isolation Forest, StandardScaler, GroupKFold** |
| Agent          | LangGraph                   | ReAct agent orchestration                        |
| Agent          | LangChain                   | Tool wrappers                                    |
| Agent          | Groq (llama-3.1-8b-instant) | LLM reasoning — free tier                        |
| Agent          | ChromaDB                    | Vector store for anomaly memory                  |
| Storage        | MongoDB 7                   | 5 collections with indexes                       |
| API            | FastAPI + Uvicorn           | REST + WebSocket + async                         |
| Dashboard      | Streamlit + Folium + Plotly | Live map, charts, agent chat                     |
| Infrastructure | Docker Compose              | 5 containers                                     |

---

## How it works

### 1. GPS Simulator

Reads `porto_pings_50taxis.csv` and pushes each row as JSON into Kafka topic `gps-pings`. Configurable delay simulates real-time pacing.

### 2. Spark Pipeline

Consumes the Kafka stream via `foreachBatch`. Per ping:

1. lat/lon → UTM meters (zone 29N)
2. Nearest road edge via OSMnx
3. Kalman predict → update via `TwinRegistry`
4. Anomaly detection via `is_anomaly()`
5. Health score update via `update_health_score()`
6. Save to MongoDB `twin_states`
7. If anomaly: save to `anomalies`, store in ChromaDB, trigger `auto_diagnose()`
8. Broadcast to WebSocket clients

### 3. Kalman Filter

```
State transition (F):          Measurement (H):
[1  0  dt  0 ]                 [1  0  0  0]
[0  1  0   dt]   x_pred =      [0  1  0  0]  × state
[0  0  1   0 ]
[0  0  0   1 ]

R = I₂ × 9.0    # GPS noise ~3m std dev
Q = I₄ × 0.01   # process noise, smooth velocity
P = I₄ × 500    # initial uncertainty
dt = 15s        # Porto ping interval
```

### 4. Vehicle Health Score

Persistent 0-100 per taxi in MongoDB:

| Event               | Points |
| ------------------- | ------ |
| Normal ping         | +2     |
| Speed anomaly       | -5     |
| Position jump       | -3     |
| Uncertainty anomaly | -2     |

### 5. LangGraph Agent

ReAct agent, 4 tools, Groq `llama-3.1-8b-instant`:

| Tool                     | Queries                                |
| ------------------------ | -------------------------------------- |
| `query_vehicle_history`  | MongoDB `twin_states` — last 10 states |
| `query_maintenance`      | MongoDB `maintenance`                  |
| `find_similar_anomalies` | ChromaDB semantic search               |
| `get_fleet_summary`      | MongoDB aggregation                    |

**Proactive auto-diagnosis:** anomalies trigger `auto_diagnose()` in a background thread with rate limiting (10s cooldown, semaphore) to respect Groq quota.

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

---

## Anomaly Detection — Rule-Based

Three types detected by `is_anomaly()`:

| Type          | Condition                   | Threshold | Severity                  |
| ------------- | --------------------------- | --------- | ------------------------- |
| speed         | `sqrt(vx²+vy²)`             | 80 m/s    | HIGH if >100, else MEDIUM |
| position_jump | distance from previous ping | 800 m     | HIGH                      |
| uncertainty   | `trace(P)`                  | 100.0     | LOW                       |

First ping per taxi is skipped — the filter is still initialising.

### Threshold Optimisation

Synthetic anomaly injection, 5 taxis × 200 pings, 5% injection rate:

| Jump Threshold | Precision | Recall    | F1        |
| -------------- | --------- | --------- | --------- |
| 500m           | 0.311     | 1.000     | 0.474     |
| 700m           | 0.434     | 0.980     | 0.601     |
| **800m**       | **0.484** | **0.920** | **0.634** |
| 900m           | 0.467     | 0.700     | 0.560     |

High recall prioritised — missing a real anomaly is worse than a false positive in fleet safety.

---

## Anomaly Detection — Machine Learning

Full study in `notebooks/anomaly_detection_ml.ipynb` (46 cells, executed on Kaggle).

### Dataset & Features

**4,311,098 pings · 444 taxis · 6 features**

| Feature              | Source                      | Signal                 |
| -------------------- | --------------------------- | ---------------------- |
| `speed`              | `sqrt(vx²+vy²)`             | velocity magnitude     |
| `vx`, `vy`           | Kalman state                | directional velocity   |
| `jump_distance`      | distance from previous ping | position teleportation |
| `rolling_mean_speed` | mean speed, window 5        | driving baseline       |
| `rolling_std_speed`  | std speed, window 5         | erratic driving        |

`uncertainty` was **dropped after EDA** — std 0.13 across 4.3M rows, correlation ≈ 0.00 with every other feature. The Kalman covariance converges to an identical steady state for every vehicle, so it carries no discriminative information.

### Isolation Forest — Results

**5-fold GroupKFold, grouped by taxi** (355 train / 88 test taxis per fold). Rows from the same vehicle are highly correlated; random row splitting would leak information between folds. The scaler is fitted on training folds only.

| Fold | Test pings | Precision | Recall | F1    |
| ---- | ---------- | --------- | ------ | ----- |
| 1    | 862,343    | 0.600     | 0.950  | 0.736 |
| 2    | 862,380    | 0.604     | 0.958  | 0.741 |
| 3    | 862,116    | 0.603     | 0.961  | 0.741 |
| 4    | 862,381    | 0.599     | 0.948  | 0.734 |
| 5    | 861,878    | 0.603     | 0.954  | 0.739 |

**Mean F1 = 0.738 ± 0.003 · Precision = 0.602 ± 0.002 · Recall = 0.954 ± 0.005**

A standard deviation of 0.003 across five folds of unseen vehicles indicates the model learned a general property of GPS data rather than memorising specific taxis.

### Agreement with Rule-Based Detector

|                           | Count      |
| ------------------------- | ---------- |
| Both flagged              | 78,195     |
| Only rule-based           | 3,229      |
| **Only Isolation Forest** | **51,138** |
| Neither                   | 4,178,536  |

**96.0%** of rule-based detections are independently rediscovered by Isolation Forest, which was never shown the thresholds. Convergence between an unsupervised model and hand-tuned physical rules is evidence both track a real signal.

The 51,138 IF-only detections are a **+62.8% increase in coverage**. Their features versus normal pings:

| Feature                 | IF-only (mean) | Normal (mean) | Ratio     |
| ----------------------- | -------------- | ------------- | --------- |
| `speed`                 | 46.80          | 8.32          | 5.6×      |
| `rolling_mean_speed`    | 71.72          | 9.09          | 7.9×      |
| **`rolling_std_speed`** | **53.43**      | **3.51**      | **15.2×** |
| `jump_distance`         | 299.51         | 124.89        | 2.4×      |

`rolling_std_speed` at 15× normal means violently erratic speed over the preceding five pings — a signature no single threshold can express.

### Feature Importance

Permutation importance (shuffle each feature, measure anomaly-score shift):

| Rank | Feature              | Importance |
| ---- | -------------------- | ---------- |
| 1    | `speed`              | 0.0128     |
| 2    | `rolling_mean_speed` | 0.0119     |
| 3    | `jump_distance`      | 0.0118     |
| 4    | `rolling_std_speed`  | 0.0067     |
| 5    | `vy`                 | 0.0061     |
| 6    | `vx`                 | 0.0043     |

Magnitude features dominate; directional components rank lowest — physically coherent, since heading alone is not anomalous.

### The Speed Confound — Discovery and Fix

Per-taxi analysis revealed a problem in **both** detectors:

```
correlation(mean_speed, anomaly_rate):
  Rule-based            r = +0.879
  Isolation Forest      r = +0.873
```

87% of a vehicle's anomaly rate is explained by how fast it habitually drives. The mechanism is arithmetic: `jump_distance ≈ speed × 15s`, so `mean_speed` ↔ `mean_jump` correlate at **0.992**. Both detectors were largely surfacing **fast-driving vehicles** rather than **faulty sensors**.

This is not an ML failure — Isolation Forest inherited the bias from the features, and the rule-based detector is marginally worse.

**Fix — per-vehicle normalisation.** Replace absolute features with relative ones:

| Absolute            | Normalised          | Meaning                                 |
| ------------------- | ------------------- | --------------------------------------- |
| `speed`             | `speed_z`           | σ from this taxi's own mean             |
| `jump_distance`     | `jump_z`            | σ from this taxi's own mean             |
| —                   | `speed_ratio`       | current ÷ recent 5-ping average         |
| `rolling_std_speed` | `rolling_std_ratio` | variance ÷ this taxi's typical variance |

`rolling_std_ratio` is included because raw variance is itself speed-dependent — leaving it absolute reintroduces the confound.

**Result:**

| Method                       | Speed confound r |
| ---------------------------- | ---------------- |
| Rule-based                   | +0.879           |
| IF — absolute features       | +0.873           |
| **IF — normalised features** | **+0.477**       |

**Confound reduction: 45.4%**

### The Normalisation Trade

`contamination=0.03` fixes the total count, so both models flagged 129,333 pings with **82.6% overlap**. The 22,480-ping symmetric difference shows what changed:

|                             | Absolute-only | Normalised-only |
| --------------------------- | ------------- | --------------- |
| Median speed                | 30.7 m/s      | 45.5 m/s        |
| Median recent average       | 74.1 m/s      | 14.1 m/s        |
| **Median `speed_ratio`**    | **0.30×**     | **3.51×**       |
| Above 80 m/s rule threshold | —             | **0**           |
| Above 800 m jump threshold  | —             | 830 (3.7%)      |

The absolute model flagged vehicles at 30 m/s that had just been travelling at 74 m/s — **decelerating normally**, at 0.30× their own baseline. False alarms produced by comparing against a fleet median of 6.3 m/s.

The normalised model instead flags vehicles averaging 14 m/s that suddenly read 45 m/s — **3.51× their own baseline**. A 15-second jump from 50 km/h to 164 km/h is a sensor fault, not acceleration. **Zero** cross the speed threshold and only 3.7% cross the jump threshold, so ~96% are invisible to the rule-based system.

### Final Comparison

| Method                        | Anomalies | Rate  | F1 (CV) | Confound r | Validation                      |
| ----------------------------- | --------- | ----- | ------- | ---------- | ------------------------------- |
| Rule-based (threshold)        | 81,424    | 1.89% | 1.000   | 0.879      | reference (pseudo ground truth) |
| Isolation Forest (absolute)   | 129,333   | 3.00% | 0.738   | 0.873      | 5-fold GroupKFold by taxi       |
| Isolation Forest (normalised) | 129,333   | 3.00% | —       | **0.477**  | confound diagnostic             |

> Rule-based scores F1 = 1.000 because it **defines** the reference labels, not because it is perfect.
>
> The normalised model has no F1 because its detections deliberately diverge from those labels — that divergence _is_ the finding. Evaluating it against a reference it was designed to improve upon would be circular.

### Limitations

- **No human-labelled ground truth.** Metrics quantify _agreement between methods_, not absolute correctness. IF-only detections count as false positives by construction, so reported precision is a lower bound.
- **Residual confound (r = 0.477) has two inseparable components.** Part is genuine physics — high speeds imply highways, tunnels and overpasses where multipath error degrades GPS. Part is residual leakage: `vx`/`vy` remain absolute and scale with speed. Normalising velocity by heading variance is the next step.
- **The normalised model was not cross-validated** for the reason stated above.

### Production Recommendation

Run both detectors in parallel. The rule-based path stays as a fast, fully interpretable safety net explainable to a fleet operator in one sentence. Isolation Forest runs alongside to surface multivariate cases the rules cannot reach, feeding the LLM agent for diagnosis.

---

## AI Agent

### LLM Comparison

| Model                        | Response Time | JSON Compliance | Quality                  |
| ---------------------------- | ------------- | --------------- | ------------------------ |
| Groq llama-3.1-8b-instant    | 3.5s          | ✅              | Good                     |
| Groq llama-3.3-70b-versatile | 8.2s          | ✅              | Excellent                |
| Gemini 2.0 Flash             | —             | —               | Free-tier quota exceeded |

Production recommendation: `llama-3.3-70b-versatile` for quality, `llama-3.1-8b-instant` for speed and cost. The architecture is model-agnostic via LangChain's interface.

---

## Storage Schema

### Collections

**`twin_states`** — one document per GPS ping

| Field          | Type  | Description                                |
| -------------- | ----- | ------------------------------------------ |
| `taxi_id`      | str   | vehicle identifier                         |
| `timestamp`    | int   | Unix seconds                               |
| `x`, `y`       | float | Kalman-filtered position (UTM meters)      |
| `vx`, `vy`     | float | estimated velocity (m/s)                   |
| `speed`        | float | `sqrt(vx²+vy²)`                            |
| `uncertainty`  | float | `trace(P)`                                 |
| `anomaly`      | bool  | anomaly flag                               |
| `anomaly_type` | str   | speed / position_jump / uncertainty / none |
| `health_score` | int   | vehicle health at this ping                |
| `road`         | dict  | nearest road edge `{u, v, key}`            |
| `lat`, `lon`   | float | original GPS coordinates                   |

**`anomalies`** — same schema, anomalous pings only

**`diagnoses`** — one document per auto-diagnosis

| Field          | Type  | Description                |
| -------------- | ----- | -------------------------- |
| `taxi_id`      | str   | vehicle identifier         |
| `timestamp`    | int   | Unix seconds               |
| `anomaly_type` | str   | type of anomaly            |
| `speed`        | float | speed at detection         |
| `uncertainty`  | float | filter uncertainty         |
| `diagnosis`    | dict  | structured JSON from agent |

**`maintenance`** — one per vehicle: oil change, tire change, inspection dates, mileage

**`health_scores`** — one per vehicle: `taxi_id`, `score` (0-100)

### Indexes

- `twin_states`, `anomalies`, `diagnoses`: compound `(taxi_id, timestamp DESC)`
- `maintenance`, `health_scores`: unique `taxi_id`

---

## API Endpoints

| Method | Endpoint              | Description                       |
| ------ | --------------------- | --------------------------------- |
| GET    | `/vehicles`           | Latest state for all active taxis |
| GET    | `/vehicles/{taxi_id}` | Last 50 states for one taxi       |
| GET    | `/anomalies`          | Last 50 anomaly events            |
| GET    | `/diagnoses`          | Last 20 auto-diagnoses            |
| GET    | `/health`             | Health scores for all vehicles    |
| POST   | `/agent`              | Ask the LangGraph agent           |
| POST   | `/broadcast`          | Internal — Spark → WebSocket      |
| WS     | `/ws`                 | Live WebSocket stream             |

---

## Dashboard

| Section          | Description                                            |
| ---------------- | ------------------------------------------------------ |
| KPI row          | Active Vehicles · Anomalies · Avg Speed · AI Diagnoses |
| Live Map         | Porto OpenStreetMap — blue/red markers, popup on click |
| Anomaly Chart    | Plotly bar chart per taxi, coloured by type            |
| Recent Anomalies | Last 6 with type and speed                             |
| AI Diagnoses     | Structured agent output with severity badges           |
| Health Scores    | Progress bars 0-100 per vehicle                        |
| Fleet Assistant  | Agent chat + all auto-diagnoses expandable             |

---

## Docker

| Service   | Image                           | Port  |
| --------- | ------------------------------- | ----- |
| zookeeper | confluentinc/cp-zookeeper:7.5.0 | 2181  |
| kafka     | confluentinc/cp-kafka:7.5.0     | 9092  |
| mongodb   | mongo:7.0                       | 27017 |
| api       | ./api/Dockerfile                | 8000  |
| dashboard | ./dashboard/Dockerfile          | 8501  |

```bash
docker-compose up -d                      # start everything
docker-compose down                       # stop everything
docker-compose build --no-cache api       # rebuild after code changes
```

GPS Simulator and Spark pipeline run locally (not yet containerised).

---

## Tests

```bash
python tests/test_kalman_filter.py
python tests/test_anomaly_detector.py
python tests/test_agent_tools.py
```

| File                       | Coverage                                           |
| -------------------------- | -------------------------------------------------- |
| `test_kalman_filter.py`    | Matrix shapes, initialisation, velocity estimation |
| `test_anomaly_detector.py` | Normal state, speed anomaly, position jump         |
| `test_agent_tools.py`      | Tool responses, fleet summary                      |
| `evaluate_detector.py`     | F1 threshold sweep, synthetic injection            |
| `compare_llms.py`          | LLM response time and JSON compliance              |

---

## 8-Week Plan

| Week | Focus                                             | Deliverable                                          |
| ---- | ------------------------------------------------- | ---------------------------------------------------- |
| 1    | Data pipeline + Kafka + Simulator + Spark         | 228k pings Kafka → Spark → console                   |
| 2    | Kalman filter + Twin Registry + Anomaly Detection | Filtered states, 3 anomaly types                     |
| 3    | Road enrichment + F1 evaluation + **ML study**    | OSMnx, F1=0.634 rules, **F1=0.738 Isolation Forest** |
| 4    | MongoDB + LangGraph agent + ChromaDB              | Agent Q&A, proactive auto-diagnosis                  |
| 5    | FastAPI + WebSocket                               | REST API + live broadcast                            |
| 6    | Streamlit dashboard                               | Live map, anomaly feed, agent chat, health scores    |
| 7    | Docker Compose full stack                         | One-command startup, 5 containers                    |
| 8    | Report + jury preparation                         | Final report, demo script, slides                    |

---

## Improvements

### Implemented

- Kalman filter tuning — R=9, Q=0.01, justified via F1 evaluation
- 3 anomaly types with `anomaly_type` field
- Threshold optimisation via F1 sweep — 800m optimal
- **Isolation Forest on 4.3M pings — F1 = 0.738 ± 0.003 (5-fold GroupKFold)**
- **Speed confound discovered (r = 0.873) and reduced to 0.477 via per-vehicle normalisation**
- **Permutation feature importance analysis**
- **Per-taxi anomaly rate analysis**
- 4th agent tool: `get_fleet_summary()`
- Proactive auto-diagnosis in background thread with rate limiting
- Vehicle health score (0-100, per-ping update)
- Structured JSON agent output
- LLM model comparison (Groq vs Gemini)
- Session state map optimisation

### Future Work

| Improvement                 | Description                                                                    | Effort |
| --------------------------- | ------------------------------------------------------------------------------ | ------ |
| LSTM Autoencoder            | Sequence modelling over 10-ping windows; reconstruction error as anomaly score | High   |
| Normalise `vx`/`vy`         | Remove residual speed leakage from directional components                      | Low    |
| Deploy ML model to pipeline | Load `isolation_forest.pkl` into Spark as second detector                      | Medium |
| Real Teltonika data         | Replace Porto dataset with live iDIRTECH GPS                                   | N/A    |
| Coordinate rescaling        | Porto → Marrakech bbox for local demo                                          | Low    |
| Containerise pipeline       | Simulator + Spark as Docker services                                           | Medium |
| Adaptive Q matrix           | Tune process noise by road type                                                | High   |
| JWT auth                    | Token-based auth on FastAPI                                                    | Medium |
| Smooth map updates          | React + Leaflet.js over WebSocket                                              | High   |

---

_Final Year Project (PFA) — 8-week internship at iDIRTECH, Marrakech — 2026_
_Student: Abdeljalil Nahimy_
_Stack: Kafka · Spark · FilterPy · scikit-learn · LangGraph · Groq · MongoDB · ChromaDB · FastAPI · Streamlit · Docker_
