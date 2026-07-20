import os
os.environ["HADOOP_HOME"] = "C:\\hadoop"
os.environ["PATH"] = os.environ["PATH"] + ";C:\\hadoop\\bin"

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
    .outputMode("append") \
    .format("console") \
    .start()
query.awaitTermination()