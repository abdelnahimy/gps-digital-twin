import sys
sys.path.insert(0, "D:\\Projects\\PFA\\gps-digital-twin")

import numpy as np
from twin.kalman_filter import create_vehicle_filter, VehicleTwin

def test_filter_creation():
    kf = create_vehicle_filter()
    assert kf.F.shape == (4, 4)
    assert kf.H.shape == (2, 4)
    assert kf.R.shape == (2, 2)
    assert kf.P.shape == (4, 4)
    print("✅ test_filter_creation passed")

def test_vehicle_twin_initialization():
    twin = VehicleTwin("test_taxi")
    state = twin.update(532681.0, 4554796.0)
    assert state["x"] == 532681.0
    assert state["y"] == 4554796.0
    assert state["vx"] == 0.0
    assert state["vy"] == 0.0
    print("✅ test_vehicle_twin_initialization passed")

def test_vehicle_twin_update():
    twin = VehicleTwin("test_taxi")
    twin.update(532681.0, 4554796.0)
    state = twin.update(532700.0, 4554820.0)
    assert state["speed"] > 0
    assert state["uncertainty"] < 500.0
    print("✅ test_vehicle_twin_update passed")

if __name__ == "__main__":
    test_filter_creation()
    test_vehicle_twin_initialization()
    test_vehicle_twin_update()
    print("All tests passed!")