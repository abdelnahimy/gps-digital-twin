import sys
sys.path.insert(0, "D:\\Projects\\PFA\\gps-digital-twin")

import pandas as pd
import random
from datetime import datetime, timedelta
from storage.mongo_client import maintenance

def random_date(start_days_ago: int, end_days_ago: int) -> str:
    days = random.randint(min(start_days_ago, end_days_ago), max(start_days_ago, end_days_ago))
    date = datetime.now() - timedelta(days=days)
    return date.strftime("%Y-%m-%d")

def generate_maintenance():
    df = pd.read_csv("data/processed/porto_pings_50taxis.csv")
    taxi_ids = df["TAXI_ID"].unique()

    records = []
    for taxi_id in taxi_ids:
        record = {
            "taxi_id": str(int(taxi_id)),
            "last_oil_change": random_date(180, 90),
            "next_oil_change": random_date(-30, -90),  # future date
            "last_tire_change": random_date(365, 180),
            "next_tire_change": random_date(-60 , -120),  # future date
            "last_inspection": random_date(200, 100),
            "next_inspection": random_date(-90, -180),  # future date
            "mileage": random.randint(20000, 150000)
        }
        records.append(record)

    maintenance.insert_many(records)
    print(f"Inserted {len(records)} maintenance records")

if __name__ == "__main__":
    generate_maintenance()