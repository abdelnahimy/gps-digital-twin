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

@tool
def get_fleet_summary() -> str:
    """Get an overall summary of the fleet including total vehicles, anomalies, and average speed."""
    total = twin_states.distinct("taxi_id")
    recent_anomalies = list(anomalies.find({}, {"_id": 0}).sort("timestamp", -1).limit(5))
    
    all_states_pipeline = [
        {"$sort": {"timestamp": -1}},
        {"$group": {"_id": "$taxi_id", "last_state": {"$first": "$$ROOT"}}},
        {"$replaceRoot": {"newRoot": "$last_state"}}
    ]
    latest_states = list(twin_states.aggregate(all_states_pipeline))
    
    avg_speed = sum(s.get("speed", 0) for s in latest_states) / len(latest_states) if latest_states else 0
    anomaly_count = sum(1 for s in latest_states if s.get("anomaly"))
    
    return (
        f"Fleet summary: {len(total)} total vehicles tracked, "
        f"{anomaly_count} currently showing anomalies, "
        f"average speed {avg_speed:.1f} m/s. "
        f"Recent anomalies: {[a.get('taxi_id') for a in recent_anomalies]}"
    )