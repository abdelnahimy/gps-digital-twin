import pandas as pd
from ast import literal_eval

def explode_porto(input_path: str, output_path: str, n_taxis: int = 50) -> pd.DataFrame:
    df = pd.read_csv(input_path,
                     usecols=["TRIP_ID", "TAXI_ID", "TIMESTAMP", "MISSING_DATA", "POLYLINE"])

    df = df[df["MISSING_DATA"] == False]
    df = df[df["POLYLINE"] != "[]"]
    df = df.drop(columns=["MISSING_DATA"])

    # keep only 50 taxis
    top_taxis = df["TAXI_ID"].unique()[:n_taxis]
    df = df[df["TAXI_ID"].isin(top_taxis)]
    df = df.groupby("TAXI_ID").head(100)  # add this line right after

    df["POLYLINE"] = df["POLYLINE"].apply(literal_eval)
    df = df.explode("POLYLINE")
    df["ping_index"] = df.groupby("TRIP_ID").cumcount()
    df[["lon", "lat"]] = pd.DataFrame(df["POLYLINE"].tolist(), index=df.index)
    df = df.drop(columns=["POLYLINE"])

    df.to_csv(output_path, index=False)
    print(f"Saved {len(df):,} pings → {output_path}")
    return df


if __name__ == "__main__":
    explode_porto(
        input_path="data/raw/train.csv",
        output_path="data/processed/porto_pings_50taxis.csv"
    )