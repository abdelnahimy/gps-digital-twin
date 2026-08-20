"""
Machine learning anomaly detector.

Wraps the Isolation Forest trained on per-vehicle normalised features
(4.3M pings, 443 taxis). See notebooks/anomaly_detection_ml.ipynb.

Model: isolation_forest_normalised_v2.pkl
  · Cross-validated F1 = 0.718 ± 0.011 (5-fold GroupKFold by vehicle)
  · Speed confound r = -0.002 (vs +0.873 for absolute features)
  · Fault signature: 3.27x the vehicle's own recent average speed
"""
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

MODEL_DIR = Path(__file__).parent

# order must match the training feature order exactly
FEATURE_ORDER = ["speed_z", "jump_z", "speed_ratio",
                 "rolling_std_ratio", "vx_z", "vy_z"]

_model = joblib.load(MODEL_DIR / "isolation_forest_normalised_v2.pkl")
_scaler = joblib.load(MODEL_DIR / "scaler_normalised_v2.pkl")


def ml_is_anomaly(features: dict) -> tuple[bool, float]:
    """
    Score a single ping with the trained Isolation Forest.

    Parameters:
        features (dict): Output of VehicleTwin.normalised_features().

    Returns:
        tuple[bool, float]: (is_anomaly, anomaly_score).
            Score is negative for anomalies, positive for normal pings —
            more negative means more anomalous.
    """
    x = pd.DataFrame([[features[f] for f in FEATURE_ORDER]],
                     columns=FEATURE_ORDER, dtype=float)
    x = x.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    x_scaled = _scaler.transform(x)
    is_anom = bool(_model.predict(x_scaled)[0] == -1)
    score = float(_model.score_samples(x_scaled)[0])
    return is_anom, score