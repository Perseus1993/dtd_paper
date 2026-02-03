"""
OSM Network Loader for DTD Simulation.

Downloads and processes OpenStreetMap road networks via OSMnx,
adding required attributes (free_flow_time, capacity) for traffic simulation.
"""

import hashlib
import os
import pickle
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

try:
    import osmnx as ox
except ImportError:
    ox = None


# Highway type to capacity (vehicles per hour per lane)
HIGHWAY_CAPACITY_MAP: Dict[str, float] = {
    "motorway": 2200.0,
    "motorway_link": 1800.0,
    "trunk": 2000.0,
    "trunk_link": 1600.0,
    "primary": 1800.0,
    "primary_link": 1400.0,
    "secondary": 1500.0,
    "secondary_link": 1200.0,
    "tertiary": 1200.0,
    "tertiary_link": 1000.0,
    "residential": 800.0,
    "living_street": 400.0,
    "unclassified": 600.0,
    "service": 400.0,
}

DEFAULT_CAPACITY_PER_LANE = 1000.0


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _parse_lanes(lanes_attr) -> int:
    """Parse lanes attribute which may be int, str, or list."""
    if lanes_attr is None:
        return 1
    if isinstance(lanes_attr, int):
        return max(1, lanes_attr)
    if isinstance(lanes_attr, str):
        try:
            return max(1, int(lanes_attr.split("|")[0].split(";")[0].strip()))
        except (ValueError, IndexError):
            return 1
    if isinstance(lanes_attr, list):
        try:
            return max(1, int(lanes_attr[0]))
        except (ValueError, IndexError, TypeError):
            return 1
    return 1


def _parse_highway_type(highway_attr) -> str:
    """Parse highway attribute which may be str or list."""
    if highway_attr is None:
        return "unclassified"
    if isinstance(highway_attr, str):
        return highway_attr
    if isinstance(highway_attr, list):
        return str(highway_attr[0]) if highway_attr else "unclassified"
    return "unclassified"


def load_osm_network(
    place_name: Optional[str] = None,
    bbox: Optional[Tuple[float, float, float, float]] = None,
    network_type: str = "drive",
    simplify: bool = True,
    cache_dir: Optional[str] = None,
) -> nx.MultiDiGraph:
    """
    Download or load cached OSM network.

    Args:
        place_name: Place name for OSMnx query (e.g., "Pudong, Shanghai, China")
        bbox: Bounding box (west, south, east, north) as alternative to place_name
        network_type: OSMnx network type ("drive", "walk", "bike", "all")
        simplify: Whether to simplify graph topology
        cache_dir: Directory for caching downloaded networks

    Returns:
        NetworkX MultiDiGraph with edge attributes
    """
    if ox is None:
        raise ImportError("osmnx is required. Install with: pip install osmnx")

    if place_name is None and bbox is None:
        raise ValueError("Either place_name or bbox must be provided")

    # Check cache
    cache_path = None
    if cache_dir is not None:
        _ensure_dir(cache_dir)
        if place_name:
            cache_key = place_name
            cache_key = cache_key.replace(" ", "_").replace(",", "").replace(".", "_")
            cache_path = os.path.join(cache_dir, f"{cache_key}_{network_type}.graphml")
        else:
            # Use a short hash for bbox to avoid extremely long filenames on Windows.
            west, south, east, north = bbox
            bbox_str = f"{float(west):.6f},{float(south):.6f},{float(east):.6f},{float(north):.6f}"
            bbox_hash = hashlib.sha1(bbox_str.encode("utf-8")).hexdigest()[:12]
            cache_key = f"bbox_{bbox_hash}"
            cache_path = os.path.join(cache_dir, f"{cache_key}_{network_type}.graphml")

            # Backward compatibility: check legacy verbose bbox key first.
            legacy_key = f"bbox_{west}_{south}_{east}_{north}"
            legacy_key = legacy_key.replace(" ", "_").replace(",", "").replace(".", "_")
            legacy_path = os.path.join(cache_dir, f"{legacy_key}_{network_type}.graphml")
            if os.path.exists(legacy_path):
                print(f"Loading cached network from {legacy_path}")
                return ox.load_graphml(legacy_path)

        if os.path.exists(cache_path):
            print(f"Loading cached network from {cache_path}")
            return ox.load_graphml(cache_path)

    # Download network
    print(f"Downloading OSM network for: {place_name or bbox}")
    if place_name:
        G = ox.graph_from_place(place_name, network_type=network_type, simplify=simplify)
    else:
        west, south, east, north = bbox
        # osmnx>=2.0 expects bbox=(left, bottom, right, top) = (west, south, east, north)
        G = ox.graph_from_bbox(bbox=(west, south, east, north), network_type=network_type, simplify=simplify)

    # Save to cache
    if cache_path:
        print(f"Caching network to {cache_path}")
        ox.save_graphml(G, cache_path)

    return G


def add_travel_times(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """
    Add free_flow_time attribute to edges based on speed limits.

    Uses OSMnx's speed imputation, then converts travel_time (seconds)
    to free_flow_time (minutes).
    """
    if ox is None:
        raise ImportError("osmnx is required")

    # Add speed data (imputes from highway type if not available)
    G = ox.add_edge_speeds(G)
    # Add travel time in seconds
    G = ox.add_edge_travel_times(G)

    # Convert to minutes and store as free_flow_time
    for u, v, k, data in G.edges(keys=True, data=True):
        travel_time_sec = data.get("travel_time", data.get("length", 100) / 10.0)
        # Convert to minutes
        free_flow_time = float(travel_time_sec) / 60.0
        # Ensure minimum travel time
        G[u][v][k]["free_flow_time"] = max(0.1, free_flow_time)

    return G


def estimate_edge_capacity(
    G: nx.MultiDiGraph,
    highway_capacity_map: Optional[Dict[str, float]] = None,
    default_capacity: float = DEFAULT_CAPACITY_PER_LANE,
) -> nx.MultiDiGraph:
    """
    Estimate edge capacity from highway type and number of lanes.

    Args:
        G: Network graph
        highway_capacity_map: Dict mapping highway type to capacity per lane
        default_capacity: Default capacity per lane if highway type unknown

    Returns:
        Graph with capacity attribute added to edges
    """
    if highway_capacity_map is None:
        highway_capacity_map = HIGHWAY_CAPACITY_MAP

    for u, v, k, data in G.edges(keys=True, data=True):
        highway_type = _parse_highway_type(data.get("highway"))
        lanes = _parse_lanes(data.get("lanes"))

        capacity_per_lane = highway_capacity_map.get(highway_type, default_capacity)
        capacity = capacity_per_lane * lanes

        G[u][v][k]["capacity"] = capacity

    return G


def assign_edge_ids(G: nx.MultiDiGraph) -> nx.MultiDiGraph:
    """Assign sequential edge IDs for shock targeting."""
    edge_id = 0
    for u, v, k in G.edges(keys=True):
        G[u][v][k]["edge"] = edge_id
        edge_id += 1
    return G


def nodes_from_coordinates(
    G: nx.MultiDiGraph,
    coords: List[Tuple[float, float]],
) -> List[int]:
    """
    Map lat/lon coordinates to nearest network nodes.

    Args:
        G: Network graph
        coords: List of (longitude, latitude) tuples

    Returns:
        List of node IDs
    """
    if ox is None:
        raise ImportError("osmnx is required")

    if not coords:
        return []

    lons = [c[0] for c in coords]
    lats = [c[1] for c in coords]

    nodes = ox.nearest_nodes(G, X=lons, Y=lats)

    if isinstance(nodes, (int, np.integer)):
        return [int(nodes)]
    return [int(n) for n in nodes]


def subgraph_for_od(
    G: nx.MultiDiGraph,
    origins: List[int],
    destinations: List[int],
    buffer_hops: int = 2,
) -> nx.MultiDiGraph:
    """
    Extract subgraph containing paths between OD pairs plus buffer.

    This is critical for scalability with large networks.

    Args:
        G: Full network graph
        origins: List of origin node IDs
        destinations: List of destination node IDs
        buffer_hops: Number of hops to include beyond shortest paths

    Returns:
        Subgraph containing relevant nodes and edges
    """
    # Collect nodes on shortest paths
    path_nodes = set()

    # Build a simple directed graph by selecting the fastest multiedge per (u, v).
    # This matches the semantics used elsewhere (e.g., shortest-path route generation)
    # and avoids arbitrary edge selection when converting MultiDiGraph -> DiGraph.
    G_simple = nx.DiGraph()
    for u, v, k, data in G.edges(keys=True, data=True):
        t0 = float(data.get("free_flow_time", data.get("travel_time", data.get("length", 1.0))))
        if G_simple.has_edge(u, v):
            if t0 < float(G_simple[u][v].get("free_flow_time", float("inf"))):
                G_simple[u][v]["free_flow_time"] = t0
        else:
            G_simple.add_edge(u, v, free_flow_time=t0)

    for o in origins:
        for d in destinations:
            if o == d:
                continue
            try:
                path = nx.shortest_path(G_simple, o, d, weight="free_flow_time")
                path_nodes.update(path)
            except nx.NetworkXNoPath:
                continue

    if not path_nodes:
        raise ValueError("No paths found between any OD pairs")

    # Add buffer: include neighbors within buffer_hops
    buffer_nodes = set(path_nodes)
    current_frontier = set(path_nodes)

    for _ in range(buffer_hops):
        next_frontier = set()
        for node in current_frontier:
            if node in G:
                next_frontier.update(G.predecessors(node))
                next_frontier.update(G.successors(node))

        # Only expand to nodes not already included.
        next_frontier = next_frontier - buffer_nodes
        if not next_frontier:
            break
        buffer_nodes.update(next_frontier)
        current_frontier = next_frontier

    # Extract subgraph
    subgraph = G.subgraph(buffer_nodes).copy()

    print(f"Extracted subgraph: {subgraph.number_of_nodes()} nodes, {subgraph.number_of_edges()} edges")
    print(f"  (from full graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges)")

    return subgraph


def prepare_network_for_simulation(
    G: nx.MultiDiGraph,
    cache_dir: Optional[str] = None,
) -> nx.MultiDiGraph:
    """
    Full pipeline to prepare OSM network for DTD simulation.

    Adds: free_flow_time, capacity, edge (ID)
    """
    G = add_travel_times(G)
    G = estimate_edge_capacity(G)
    G = assign_edge_ids(G)
    return G


def get_network_stats(G: nx.MultiDiGraph) -> Dict[str, float]:
    """Get basic network statistics."""
    return {
        "nodes": G.number_of_nodes(),
        "edges": G.number_of_edges(),
        "avg_degree": sum(dict(G.degree()).values()) / max(1, G.number_of_nodes()),
    }
