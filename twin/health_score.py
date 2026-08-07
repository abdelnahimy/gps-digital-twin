import sys
sys.path.insert(0, "D:\\Projects\\PFA\\gps-digital-twin")

from storage.mongo_client import health_scores

ANOMALY_PENALTIES = {
    "speed": -5,
    "position_jump": -3,
    "uncertainty": -2,
    "none": +2  # reward for normal ping
}

def update_health_score(taxi_id: str, anomaly_type: str) -> int:
    """
    Update the health score for a taxi and return the new score.
    """
    penalty = ANOMALY_PENALTIES.get(anomaly_type, 0)

    result = health_scores.find_one_and_update(
        {"taxi_id": taxi_id},
        {"$inc": {"score": penalty}},
        upsert=True,
        return_document=True
    )

    # clamp between 0 and 100
    score = result.get("score", 100) if result else 100
    score = max(0, min(100, score))

    # save clamped value
    health_scores.update_one(
        {"taxi_id": taxi_id},
        {"$set": {"score": score}},
        upsert=True
    )

    return score