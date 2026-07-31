import sys
sys.path.insert(0, "D:\\Projects\\PFA\\gps-digital-twin")

from twin.anomaly_detector import is_anomaly, SPEED_THRESHOLD, UNCERTAINTY_THRESHOLD, POSITION_JUMP_THRESHOLD

def test_normal_state():
    state = {
        "x": 532681.0, "y": 4554796.0,
        "vx": 5.0, "vy": 3.0,
        "speed": 5.8,
        "uncertainty": 9.0,
        "prev_x": 532670.0, "prev_y": 4554780.0
    }
    anomaly, anomaly_type = is_anomaly(state)
    assert anomaly == False
    assert anomaly_type == "none"
    print("✅ test_normal_state passed")

def test_speed_anomaly():
    state = {
        "x": 532681.0, "y": 4554796.0,
        "vx": 60.0, "vy": 40.0,
        "speed": 72.0,
        "uncertainty": 9.0,
        "prev_x": None, "prev_y": None
    }
    anomaly, anomaly_type = is_anomaly(state)
    assert anomaly == True
    assert anomaly_type == "speed"
    print("✅ test_speed_anomaly passed")

def test_position_jump():
    state = {
        "x": 533681.0, "y": 4555796.0,
        "vx": 5.0, "vy": 3.0,
        "speed": 5.8,
        "uncertainty": 9.0,
        "prev_x": 532681.0, "prev_y": 4554796.0
    }
    anomaly, anomaly_type = is_anomaly(state)
    print(f"anomaly={anomaly}, type={anomaly_type}")
    assert anomaly == True
    assert anomaly_type == "position_jump"

if __name__ == "__main__":
    test_normal_state()
    test_speed_anomaly()
    test_position_jump()
    print("All tests passed!")