import numpy as np
from collections import deque
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

    # State transition matrix — constant velocity model
    kf.F = np.array([[1, 0, dt, 0],
                     [0, 1, 0, dt],
                     [0, 0, 1, 0],
                     [0, 0, 0, 1]])

    # Measurement matrix — GPS observes position only
    kf.H = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0]])

    kf.R = np.eye(2) * 9.0      # GPS noise ~3m std dev → variance 9
    kf.P = np.eye(4) * 500.0    # initial uncertainty
    kf.Q = np.eye(4) * 0.01     # process noise — smoother velocity
    kf.x = np.zeros((4, 1))     # initial state
    return kf


class VehicleTwin:
    """Digital twin for one vehicle — Kalman filter plus running statistics."""

    def __init__(self, taxi_id: str):
        self.taxi_id = taxi_id
        self.kf = create_vehicle_filter()
        self.initialized = False
        self.last_x = None
        self.last_y = None

        # running per-vehicle statistics for normalised ML features
        self.n_pings = 0
        self.speed_sum = 0.0
        self.speed_sq_sum = 0.0
        self.jump_sum = 0.0
        self.jump_sq_sum = 0.0
        self.vx_sum = 0.0
        self.vx_sq_sum = 0.0
        self.vy_sum = 0.0
        self.vy_sq_sum = 0.0
        self.speed_window = deque(maxlen=5)

    def update(self, x: float, y: float) -> dict:
        """
        Update the Kalman filter with new GPS coordinates.

        Parameters:
            x (float): Easting in UTM meters.
            y (float): Northing in UTM meters.

        Returns:
            dict: Filtered state estimate.
        """
        if not self.initialized:
            self.kf.x = np.array([[x], [y], [0], [0]])
            self.initialized = True
        else:
            self.kf.predict()
            self.kf.update(np.array([[x], [y]]))

        prev_x = self.last_x
        prev_y = self.last_y
        self.last_x = float(self.kf.x[0])
        self.last_y = float(self.kf.x[1])

        return {
            "taxi_id": self.taxi_id,
            "x": float(self.kf.x[0]),
            "y": float(self.kf.x[1]),
            "vx": float(self.kf.x[2]),
            "vy": float(self.kf.x[3]),
            "speed": float(np.sqrt(self.kf.x[2]**2 + self.kf.x[3]**2)),
            "uncertainty": float(np.trace(self.kf.P)),
            "prev_x": prev_x,
            "prev_y": prev_y
        }

    def normalised_features(self, speed: float, jump: float,
                            vx: float, vy: float,
                            warmup: int = 20) -> dict | None:
        """
        Update running per-vehicle statistics and return the 6 normalised
        features the ML model expects.

        Uses expanding-window statistics — only past pings contribute — so this
        is causal and deployable. Returns None until `warmup` pings have been
        seen, because z-scores computed from very few samples are unstable.

        Parameters:
            speed (float): Current filtered speed in m/s.
            jump (float): Distance from previous position in meters.
            vx (float): Filtered velocity, x component.
            vy (float): Filtered velocity, y component.
            warmup (int): Minimum pings before features are returned.

        Returns:
            dict | None: Six normalised features, or None during warm-up.
        """
        self.n_pings += 1
        self.speed_sum    += speed
        self.speed_sq_sum += speed ** 2
        self.jump_sum     += jump
        self.jump_sq_sum  += jump ** 2
        self.vx_sum       += vx
        self.vx_sq_sum    += vx ** 2
        self.vy_sum       += vy
        self.vy_sq_sum    += vy ** 2
        self.speed_window.append(speed)

        if self.n_pings < warmup:
            return None

        n = self.n_pings
        eps = 1e-6

        def mean_std(total, sq_total):
            mean = total / n
            var = max(sq_total / n - mean ** 2, 0.0)
            return mean, np.sqrt(var)

        speed_mean, speed_std = mean_std(self.speed_sum, self.speed_sq_sum)
        jump_mean,  jump_std  = mean_std(self.jump_sum,  self.jump_sq_sum)
        vx_mean,    vx_std    = mean_std(self.vx_sum,    self.vx_sq_sum)
        vy_mean,    vy_std    = mean_std(self.vy_sum,    self.vy_sq_sum)

        roll_mean = float(np.mean(self.speed_window))
        roll_std  = float(np.std(self.speed_window))

        return {
            "speed_z":           (speed - speed_mean) / (speed_std + eps),
            "jump_z":            (jump - jump_mean) / (jump_std + eps),
            "speed_ratio":       min(speed / (roll_mean + eps), 5.0),
            "rolling_std_ratio": roll_std / (speed_std + eps),
            "vx_z":              (vx - vx_mean) / (vx_std + eps),
            "vy_z":              (vy - vy_mean) / (vy_std + eps),
        }