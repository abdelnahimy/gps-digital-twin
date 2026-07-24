import osmnx as ox
import numpy as np
from pyproj import Proj

GRAPH_PATH = "data/road_network/porto_road_network.graphml"
G = ox.load_graphml(GRAPH_PATH)

def get_nearest_road(lat: float, lon: float) -> dict:
    nearest_edge = ox.nearest_edges(G, lon, lat)
    return {
        "u": nearest_edge[0],
        "v": nearest_edge[1],
        "key": nearest_edge[2]
    }

def download_road_network(place: str, save_path: str) -> None:
    G = ox.graph_from_place(place, network_type="drive")
    ox.save_graphml(G, save_path)
    print(f"Road network saved → {save_path}")

if __name__ == "__main__":
    download_road_network(
        place="Porto, Portugal",
        save_path="data/road_network/porto_road_network.graphml"
    )