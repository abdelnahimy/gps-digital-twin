import pandas as pd
import json
import time
from confluent_kafka import Producer

TOPIC = "gps-pings"
CSV_PATH = "data\\processed\\porto_pings_50taxis.csv"

producer = Producer({"bootstrap.servers": "localhost:9092"})

def delivery_report(err, msg):
    if err is not None:
        print(f"Delivery failed for record {msg.key()}: {err}")
    else:
        print(f"Record {msg.key()} successfully produced to {msg.topic()} [{msg.partition()}] at offset {msg.offset()}")

def simulate(csv_path, delay=0.01):
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        ping = {
            "trip_id": row["TRIP_ID"],
            "taxi_id": row["TAXI_ID"],
            "timestamp": row["TIMESTAMP"],
            "lon": row["lon"],
            "lat": row["lat"],
            "ping_index": row["ping_index"]
        }
        producer.produce(
            TOPIC,
            key=str(int(row["TRIP_ID"])),
            value=json.dumps(ping),
            callback=delivery_report
        )
        producer.poll(0)
        time.sleep(delay)
    producer.flush()
if __name__ == "__main__":
    simulate(CSV_PATH, delay=0.01)