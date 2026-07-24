import numpy as np

SPEED_THRESHOLD = 50.0      # meters/second (~180 km/h)
UNCERTAINTY_THRESHOLD = 100.0

def is_anomaly(state: dict) -> bool:
    vx = state.get("vx", 0.0)
    vy = state.get("vy", 0.0)
    speed = np.sqrt(vx**2 + vy**2)
    uncertainty = state.get("uncertainty", 0.0)

    if speed > SPEED_THRESHOLD or uncertainty > UNCERTAINTY_THRESHOLD:
        return True
    return False