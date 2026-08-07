from pymongo import MongoClient
import os

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = "gps_digital_twin"

client = MongoClient(MONGO_URI)
db = client[DB_NAME]

twin_states = db["twin_states"]
anomalies = db["anomalies"]
maintenance = db["maintenance"]
health_scores = db["health_scores"]
diagnoses = db["diagnoses"]

diagnoses.create_index([("taxi_id", 1), ("timestamp", -1)])
twin_states.create_index([("taxi_id", 1), ("timestamp", -1)])
anomalies.create_index([("taxi_id", 1), ("timestamp", -1)])
maintenance.create_index([("taxi_id", 1)])
health_scores.create_index([("taxi_id", 1), ("timestamp", -1)])