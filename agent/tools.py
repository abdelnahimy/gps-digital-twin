import sys
sys.path.insert(0, "D:\\Projects\\PFA\\gps-digital-twin")

from langchain.tools import tool
from storage.mongo_client import twin_states, anomalies, maintenance

@tool
def query_vehicle_history(taxi_id: str) -> str:
    """Query the last 10 GPS states for a specific vehicle."""
    results = list(twin_states.find(
        {"taxi_id": taxi_id},
        {"_id": 0}
    ).sort("timestamp", -1).limit(10))

    if not results:
        return f"No history found for vehicle with taxi_id: {taxi_id}"
    return str(results)

@tool
def query_maintenance(taxi_id: str) -> str:
    """Query the maintenance records for a specific vehicle."""
    results = maintenance.find_one({"taxi_id" : taxi_id}, {"_id": 0})

    if not results:
        return f"No maintenance records found for vehicle with taxi_id: {taxi_id}"
    return str(results)

@tool
def find_similar_anomalies(description: str) -> str:
    """Find similar anomalies from the past based on a text description."""
    from agent.memory import anomaly_collection
    results = anomaly_collection.query(
        query_texts=[description],
        n_results=3
    )
    if not results["documents"][0]:
        return "No similar anomalies found"
    return str(results["documents"][0])
