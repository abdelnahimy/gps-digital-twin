import sys
sys.path.insert(0, "D:\\Projects\\PFA\\gps-digital-twin")

import pandas as pd
import numpy as np
from twin.kalman_filter import VehicleTwin
from twin.anomaly_detector import is_anomaly

from processing.projection import latlon_to_meters

# load clean data
df = pd.read_csv("data/processed/porto_pings_50taxis.csv")

# take one taxi for evaluation
EVAL_TAXI = str(int(df["TAXI_ID"].unique()[0]))
taxi_df = df[df["TAXI_ID"] == df["TAXI_ID"].unique()[0]].head(200).reset_index(drop=True)

print(f"Evaluating on taxi {EVAL_TAXI} — {len(taxi_df)} pings")

def inject_anomalies(df, injection_rate=0.05):
    """
    Inject synthetic anomalies into clean pings.
    Returns modified dataframe and ground truth labels.
    """
    df = df.copy()
    labels = [0] * len(df)  # 0 = normal, 1 = anomaly
    
    n_anomalies = int(len(df) * injection_rate)
    anomaly_indices = np.random.choice(
        range(10, len(df)),  # skip first 10 pings (filter warmup)
        size=n_anomalies,
        replace=False
    )
    
    for idx in anomaly_indices:
        anomaly_type = np.random.choice(["speed_spike", "position_jump"])
        
        if anomaly_type == "speed_spike":
            # teleport 0.01 degrees (~1km)
            df.at[idx, "lat"] += np.random.choice([-0.01, 0.01])
            df.at[idx, "lon"] += np.random.choice([-0.01, 0.01])
        else:
            # large position jump ~2km
            df.at[idx, "lat"] += np.random.choice([-0.02, 0.02])
            df.at[idx, "lon"] += np.random.choice([-0.02, 0.02])
        
        labels[idx] = 1
    
    return df, labels, sorted(anomaly_indices)

df_injected, ground_truth, injected_at = inject_anomalies(taxi_df)
print(f"Injected {sum(ground_truth)} anomalies at indices: {injected_at[:5]}...")

def evaluate(df, labels):
    twin = VehicleTwin(EVAL_TAXI)
    predictions = []

    for _, row in df.iterrows():
        x, y = latlon_to_meters(row["lat"], row["lon"])
        state = twin.update(x, y)
        anomaly, _ = is_anomaly(state)
        predictions.append(1 if anomaly else 0)

    # compute metrics
    tp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(predictions, labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 1)
    tn = sum(1 for p, l in zip(predictions, labels) if p == 0 and l == 0)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print(f"\n--- Evaluation Results ---")
    print(f"TP: {tp} | FP: {fp} | FN: {fn} | TN: {tn}")
    print(f"Precision: {precision:.3f}")
    print(f"Recall:    {recall:.3f}")
    print(f"F1 Score:  {f1:.3f}")
    return precision, recall, f1

evaluate(df_injected, ground_truth)
def threshold_sweep():
    print("\n--- Threshold Sweep ---")
    print(f"{'Jump Threshold':>15} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    
    best_f1 = 0
    best_threshold = 0
    
    for threshold in range(100, 1500, 100):
        twin = VehicleTwin(EVAL_TAXI)
        predictions = []
        
        for _, row in df_injected.iterrows():
            x, y = latlon_to_meters(row["lat"], row["lon"])
            state = twin.update(x, y)
            
            # override threshold for sweep
            vx, vy = state["vx"], state["vy"]
            speed = np.sqrt(vx**2 + vy**2)
            prev_x, prev_y = state.get("prev_x"), state.get("prev_y")
            
            anomaly = False
            if prev_x is not None:
                distance = np.sqrt((state["x"] - prev_x)**2 + (state["y"] - prev_y)**2)
                if distance > threshold or speed > 80:
                    anomaly = True
            
            predictions.append(1 if anomaly else 0)
        
        tp = sum(1 for p, l in zip(predictions, ground_truth) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(predictions, ground_truth) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(predictions, ground_truth) if p == 0 and l == 1)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{threshold:>15} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\nBest threshold: {best_threshold}m → F1: {best_f1:.3f}")
    return best_threshold

threshold_sweep()

def analyze_taxi(df):
    print("\n--- Raw Data Analysis ---")
    
    twin = VehicleTwin(EVAL_TAXI)
    speeds = []
    jumps = []
    
    for _, row in df.iterrows():
        x, y = latlon_to_meters(row["lat"], row["lon"])
        state = twin.update(x, y)
        
        speeds.append(state["speed"])
        
        if state.get("prev_x") is not None:
            distance = np.sqrt((state["x"] - state.get("prev_x", 0))**2 + 
                             (state["y"] - state.get("prev_y", 0))**2)
            jumps.append(distance)
    
    print(f"Speed stats (m/s):")
    print(f"  Mean:  {np.mean(speeds):.2f}")
    print(f"  Max:   {np.max(speeds):.2f}")
    print(f"  Std:   {np.std(speeds):.2f}")
    print(f"  >80:   {sum(1 for s in speeds if s > 80)} pings")
    
    print(f"\nPosition jump stats (meters):")
    print(f"  Mean:  {np.mean(jumps):.2f}")
    print(f"  Max:   {np.max(jumps):.2f}")
    print(f"  Std:   {np.std(jumps):.2f}")
    print(f"  >500:  {sum(1 for j in jumps if j > 500)} pings")
    print(f"  >800:  {sum(1 for j in jumps if j > 800)} pings")

analyze_taxi(taxi_df)

def multi_taxi_sweep(n_taxis=5):
    print("\n--- Multi-Taxi Threshold Sweep ---")
    print(f"{'Threshold':>10} {'Precision':>10} {'Recall':>10} {'F1':>10}")
    
    taxis = df["TAXI_ID"].unique()[:n_taxis]
    best_f1 = 0
    best_threshold = 0
    
    for threshold in range(100, 1500, 100):
        all_preds, all_labels = [], []
        
        for taxi_id in taxis:
            taxi_data = df[df["TAXI_ID"] == taxi_id].head(200).reset_index(drop=True)
            taxi_injected, labels, _ = inject_anomalies(taxi_data)
            
            twin = VehicleTwin(str(int(taxi_id)))
            preds = []
            
            for _, row in taxi_injected.iterrows():
                x, y = latlon_to_meters(row["lat"], row["lon"])
                state = twin.update(x, y)
                vx, vy = state["vx"], state["vy"]
                speed = np.sqrt(vx**2 + vy**2)
                prev_x, prev_y = state.get("prev_x"), state.get("prev_y")
                
                anomaly = False
                if prev_x is not None:
                    distance = np.sqrt((state["x"] - prev_x)**2 + (state["y"] - prev_y)**2)
                    if distance > threshold or speed > 80:
                        anomaly = True
                preds.append(1 if anomaly else 0)
            
            all_preds.extend(preds)
            all_labels.extend(labels)
        
        tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
        fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"{threshold:>10} {precision:>10.3f} {recall:>10.3f} {f1:>10.3f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\nBest threshold across {n_taxis} taxis: {best_threshold}m → F1: {best_f1:.3f}")
    return best_threshold

best = multi_taxi_sweep(n_taxis=5)