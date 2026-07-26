import os
os.environ["HADOOP_HOME"] = "C:\\hadoop"
os.environ["PATH"] = os.environ["PATH"] + ";C:\\hadoop\\bin"
import sys
sys.path.insert(0, "D:\\Projects\\PFA\\gps-digital-twin")

from twin.twin_registry import TwinRegistry
from processing.projection import latlon_to_meters
from twin.anomaly_detector import is_anomaly
from processing.road_enrichment import get_nearest_road
from storage.mongo_client import twin_states, anomalies
from agent.memory import store_anomaly

from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType, IntegerType

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
    """
    Process a batch of GPS pings and update the digital twin registry.

    Args:
        batch_df (DataFrame): A DataFrame containing the batch of GPS pings.
        batch_id (int): The unique identifier for the batch.
    """
    pdf = batch_df.toPandas()
    for _, row in pdf.iterrows():
        twin = registry.get_or_create(str(int(float(row["taxi_id"]))))
        x, y = latlon_to_meters(row["lat"], row["lon"])
        state = twin.update(x, y)
        state["anomaly"] = is_anomaly(state)
        state["road"] = get_nearest_road(row["lat"], row["lon"])
        state["timestamp"] = int(row["timestamp"])
        state["lat"] = float(row["lat"])
        state["lon"] = float(row["lon"])
        
        twin_states.insert_one(dict(state))

        if state["anomaly"]:
            anomalies.insert_one(dict(state))
            store_anomaly(state)
            
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