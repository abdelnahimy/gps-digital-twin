import sys
sys.path.insert(0, "D:\\Projects\\PFA\\gps-digital-twin")

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from storage.mongo_client import twin_states, anomalies
from agent.graph import run_agent
from pydantic import BaseModel
from typing import List

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
        {"$sort": {"timestamp": -1}},
        {"$group": {"_id": "$taxi_id", "last_state": {"$first": "$$ROOT"}}},
        {"$replaceRoot": {"newRoot": "$last_state"}}
    ]
    results = list(twin_states.aggregate(pipeline))
    for r in results:
        r.pop("_id", None)
    return results

@app.get("/vehicles/{taxi_id}")
def get_vehicle_history(taxi_id: str):
    results = list(twin_states.find(
        {"taxi_id": taxi_id},
        {"_id": 0}
    ).sort("timestamp", -1).limit(50))
    if not results:
        return {"error": f"No history found for taxi {taxi_id}"}
    return results

@app.get("/anomalies")
def get_anomalies():
    results = list(anomalies.find(
        {}, {"_id": 0}
    ).sort("timestamp", -1).limit(50))
    return results

class AgentQuery(BaseModel):
    question: str

@app.post("/agent")
def ask_agent(query: AgentQuery):
    answer = run_agent(query.question)
    return {"question": query.question, "answer": answer}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except:
        manager.disconnect(websocket)

@app.post("/broadcast")
async def broadcast(state: dict):
    await manager.broadcast(state)
    return {"status": "ok"}