import sys
sys.path.insert(0, "D:\\Projects\\PFA\\gps-digital-twin")

from agent.tools import query_vehicle_history, query_maintenance, get_fleet_summary

def test_query_vehicle_history_no_data():
    result = query_vehicle_history.invoke({"taxi_id": "nonexistent_taxi"})
    assert "No history" in result
    print("✅ test_query_vehicle_history_no_data passed")

def test_query_maintenance_no_data():
    result = query_maintenance.invoke({"taxi_id": "nonexistent_taxi"})
    assert "No maintenance" in result
    print("✅ test_query_maintenance_no_data passed")

def test_get_fleet_summary():
    result = get_fleet_summary.invoke({})
    assert "Fleet summary" in result
    print("✅ test_get_fleet_summary passed")

if __name__ == "__main__":
    test_query_vehicle_history_no_data()
    test_query_maintenance_no_data()
    test_get_fleet_summary()
    print("All tests passed!")