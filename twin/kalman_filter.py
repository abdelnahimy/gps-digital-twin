import numpy as np
from filterpy.kalman import KalmanFilter

def create_vehicle_filter(dt: float = 15.0) -> KalmanFilter:
    """
    Create a Kalman filter for vehicle tracking.

    Parameters:
    dt (float): Time step in seconds. Default is 15.0 seconds.

    Returns:
    KalmanFilter: Configured Kalman filter instance.
    """
    # State vector: [x, y, vx, vy]
    kf = KalmanFilter(dim_x=4, dim_z=2)

    # State transition matrix
    kf.F = np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

    kf.H = np.array([
                    [1, 0, 0, 0],
                    [0, 1, 0, 0]])
    kf.R = np.eye(2) * 5.0      # measurement noise
    kf.P = np.eye(4) * 500.0    # initial uncertainty
    kf.Q = np.eye(4) * 0.1      # process noise
    kf.x = np.zeros((4, 1))  # initial state
    return kf

class VehicleTwin:
    def __init__(self, taxi_id: str):
        self.taxi_id = taxi_id
        self.kf = create_vehicle_filter()
        self.initialized = False
    def update(self, x: float, y: float) -> dict:
        """
        Update the Kalman filter with new GPS coordinates.

        Parameters:
        x (float): Longitude of the vehicle.
        y (float): Latitude of the vehicle.

        Returns:
        dict: Dictionary containing the estimated state and covariance.
        """
        if not self.initialized:
            self.kf.x = np.array([[x], [y], [0], [0]])
            self.initialized = True
        else:
            self.kf.predict()
            self.kf.update(np.array([[x], [y]]))

        return {
            "taxi_id": self.taxi_id,
            "x": float(self.kf.x[0]),
            "y": float(self.kf.x[1]),
            "vx": float(self.kf.x[2]),
            "vy": float(self.kf.x[3]),
            "speed": float(np.sqrt(self.kf.x[2]**2 + self.kf.x[3]**2)),
            "uncertainty": float(np.trace(self.kf.P))
        }