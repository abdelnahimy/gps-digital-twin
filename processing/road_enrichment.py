import osmnx as ox
import numpy as np
from pyproj import Proj
import time

def download_road_network(place: str, save_path: str) -> None:
    print(f"[START] Downloading road network for: {place}")
    
    # Enable OSMnx logging
    ox.settings.log_console = True
    ox.settings.use_cache = True  # speeds up repeated runs

    start_time = time.time()

    print("[STEP 1] Requesting data from OpenStreetMap...")
    G = ox.graph_from_place(place, network_type="drive")

    print("[STEP 2] Saving graph to file...")
    ox.save_graphml(G, save_path)

    elapsed = time.time() - start_time
    print(f"[DONE] Road network saved → {save_path}")
    print(f"[TIME] Total duration: {elapsed:.2f} seconds")


if __name__ == "__main__":
    download_road_network(
        place="Porto, Portugal",
        save_path="data/road_network/porto_road_network.graphml"
    )