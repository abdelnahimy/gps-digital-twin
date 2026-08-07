import sys
sys.path.insert(0, "D:\\Projects\\PFA\\gps-digital-twin")

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from storage.mongo_client import twin_states, anomalies, diagnoses, health_scores
from agent.graph import run_agent
from pydantic import BaseModel
from typing import List
import os

app = FastAPI(title="GPS Digital Twin API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            await connection.send_json(message)

manager = ConnectionManager()

@app.get("/vehicles")
def get_vehicles():
    pipeline = [
        {"$sort": {"_id": -1}},
        {"$group": {
            "_id": "$taxi_id",
            "last_state": {"$first": "$$ROOT"}
        }},
        {"$replaceRoot": {"newRoot": "$last_state"}}
    ]
    results = list(twin_states.aggregate(pipeline, allowDiskUse=True))

    # mark taxis with recent anomalies as red on map
    recent_anomaly_ids = set(
        a["taxi_id"] for a in anomalies.find(
            {}, {"taxi_id": 1, "_id": 0}
        ).sort("_id", -1).limit(20)
    )

    for r in results:
        r.pop("_id", None)
        if r.get("taxi_id") in recent_anomaly_ids:
            r["anomaly"] = True

    return results

@app.get("/vehicles/{taxi_id}")
def get_vehicle_history(taxi_id: str):
    results = list(twin_states.find(
        {"taxi_id": taxi_id},
        {"_id": 0}
    ).sort("_id", -1).limit(50))
    if not results:
        return {"error": f"No history found for taxi {taxi_id}"}
    return results

@app.get("/anomalies")
def get_anomalies():
    results = list(anomalies.find(
        {}, {"_id": 0}
    ).sort("_id", -1).limit(50))
    return results


@app.get("/health")
def get_health_scores():
    results = list(health_scores.find({}, {"_id": 0}).sort("score", 1))
    return results

class AgentQuery(BaseModel):
    question: str
    
@app.post("/agent")
async def ask_agent(query: AgentQuery):
    import asyncio
    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(None, run_agent, query.question)
    return {"question": query.question, "answer": answer}

@app.get("/diagnoses")
def get_diagnoses():
    results = list(diagnoses.find(
        {}, {"_id": 0}
    ).sort("_id", -1).limit(20))
    return results

@app.post("/broadcast")
async def broadcast(state: dict):
    await manager.broadcast(state)
    return {"status": "ok"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except:
            manager.disconnect(websocket)