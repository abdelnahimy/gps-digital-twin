import numpy as np

SPEED_THRESHOLD = 50.0      # meters/second (~180 km/h)
UNCERTAINTY_THRESHOLD = 100.0
POSITION_JUMP_THRESHOLD = 500.0     # meters — max realistic in 15 seconds

def is_anomaly(state: dict) -> bool:
    vx = state.get("vx", 0.0)
    vy = state.get("vy", 0.0)
    speed = np.sqrt(vx**2 + vy**2)
    uncertainty = state.get("uncertainty", 0.0)

    if speed > SPEED_THRESHOLD : 
        return True , "speed"
    
    if uncertainty > UNCERTAINTY_THRESHOLD:
        return True , "Uncertainty"
        # position jump anomaly
    prev_x = state.get("prev_x")
    prev_y = state.get("prev_y")
    if prev_x is not None and prev_y is not None:
        distance = np.sqrt((state["x"] - prev_x)**2 + (state["y"] - prev_y)**2)
        if distance > POSITION_JUMP_THRESHOLD:
            return True , "position_jump"

    return False , "none"