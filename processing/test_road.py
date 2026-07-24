import osmnx as ox

G = ox.load_graphml("data/road_network/porto_road_network.graphml")

ox.plot_graph(G)