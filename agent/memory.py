import chromadb 

client = chromadb.Client()
anomaly_collection = client.get_or_create_collection("anomalies")

def store_anomaly(state: dict) -> None:
    text = (f"Taxi {state['taxi_id']} anomaly: "
            f"speed={state['speed']:.1f} m/s, "
            f"uncertainty={state['uncertainty']:.2f}, "
            f"position=({state['x']:.1f}, {state['y']:.1f})")
    
    anomaly_collection.add(
        documents=[text],
        ids=[f"{state['taxi_id']}_{state['timestamp']}"]
    )