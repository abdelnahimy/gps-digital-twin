import sys
sys.path.insert(0, "D:\\Projects\\PFA\\gps-digital-twin")

from storage.mongo_client import health_scores

WINDOW = 100  # rolling window length, in pings


def update_health_score(taxi_id: str, is_anomaly: bool) -> int:
    """
    Update and return a vehicle's health score.

    Score is the percentage of that vehicle's last WINDOW pings that were
    anomaly-free. Unlike a weighted-penalty scheme this needs no arbitrary
    constants — a score of 94 means precisely that 94% of recent pings were
    clean.

    Parameters:
        taxi_id (str): Vehicle identifier.
        is_anomaly (bool): Whether this ping was flagged by any detector.

    Returns:
        int: Health score, 0-100.
    """
    doc = health_scores.find_one({"taxi_id": taxi_id}) or {"clean": 0, "total": 0}

    clean = doc.get("clean", 0) + (0 if is_anomaly else 1)
    total = doc.get("total", 0) + 1

    # decay both counters once the window is full, so old pings fade out
    if total > WINDOW:
        clean = int(clean * WINDOW / total)
        total = WINDOW

    score = int(clean / total * 100)

    health_scores.update_one(
        {"taxi_id": taxi_id},
        {"$set": {"clean": clean, "total": total, "score": score}},
        upsert=True,
    )
    return score