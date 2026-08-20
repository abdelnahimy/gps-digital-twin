import os
os.environ["HADOOP_HOME"] = "C:\\hadoop"
os.environ["PATH"] = os.environ["PATH"] + ";C:\\hadoop\\bin"
import sys
sys.path.insert(0, "D:\\Projects\\PFA\\gps-digital-twin")

import numpy as np

from twin.twin_registry import TwinRegistry
from twin.anomaly_detector import is_anomaly
from twin.health_score import update_health_score

from processing.projection import latlon_to_meters
from processing.road_enrichment import get_nearest_road

from storage.mongo_client import twin_states, anomalies

from agent.memory import store_anomaly
from agent.graph import auto_diagnose

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, IntegerType

from models.ml_detector import ml_is_anomaly

TOPIC = "gps-pings"

ping_schema = StructType([
    StructField("trip_id", LongType()),
    StructField("taxi_id", LongType()),
    StructField("timestamp", LongType()),
    StructField("lon", DoubleType()),
    StructField("lat", DoubleType()),
    StructField("ping_index", LongType())
])

spark = SparkSession.builder \
    .appName("GPSDigitalTwin") \
    .master("local[*]") \
    .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
    .getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

registry = TwinRegistry()

def process_batch(batch_df, batch_id):
    pdf = batch_df.toPandas()
    for _, row in pdf.iterrows():
        twin = registry.get_or_create(str(int(float(row["taxi_id"]))))
        x, y = latlon_to_meters(row["lat"], row["lon"])
        state = twin.update(x, y)

        # rule-based detection (unchanged)
        anomaly, anomaly_type = is_anomaly(state)
        state["anomaly"] = anomaly
        state["anomaly_type"] = anomaly_type

        # ML detection
        jump = 0.0
        if state["prev_x"] is not None:
            jump = float(np.sqrt((state["x"] - state["prev_x"])**2 +
                                 (state["y"] - state["prev_y"])**2))

        norm_feats = twin.normalised_features(
            state["speed"], jump, state["vx"], state["vy"]
        )

        if norm_feats is None:
            state["ml_anomaly"] = False
            state["ml_score"] = None
            state["ml_status"] = f"warmup ({twin.n_pings}/20)"
        else:
            ml_flag, ml_score = ml_is_anomaly(norm_feats)
            state["ml_anomaly"] = ml_flag
            state["ml_score"] = ml_score
            state["ml_status"] = "active"

        state["road"] = get_nearest_road(row["lat"], row["lon"])
        state["timestamp"] = int(row["timestamp"])
        state["lat"] = float(row["lat"])
        state["lon"] = float(row["lon"])
        state["health_score"] = update_health_score(
            str(int(float(row["taxi_id"]))),
            state["anomaly"] or state["ml_anomaly"]
        )

        twin_states.insert_one(dict(state))

        # trigger diagnosis when EITHER detector fires
        if state["anomaly"] or state["ml_anomaly"]:
            if not state["anomaly"] and state["ml_anomaly"]:
                state["anomaly_type"] = "ml_detected"
            anomalies.insert_one(dict(state))
        print(state)

raw_stream = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "localhost:9092") \
    .option("subscribe", TOPIC) \
    .option("startingOffsets", "latest") \
    .load()

parsed_stream = raw_stream.select(
    from_json(col("value").cast("string"), ping_schema).alias("data")
    ).select("data.*")

query = parsed_stream.writeStream \
    .foreachBatch(process_batch) \
    .start()

query.awaitTermination()