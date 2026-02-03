"""
City Data Loader for DTD Simulation.

Provides unified interface for loading city-specific network and OD demand data.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from dtd_srdt.osm_network import (
    load_osm_network,
    prepare_network_for_simulation,
    subgraph_for_od,
    nodes_from_coordinates,
)
from worldmove.od_extractor import (
    load_worldmove_data,
    extract_od_demand,
    aggregate_od_to_network,
)


# City configurations
# Use district-level queries for manageable network sizes
CITY_CONFIGS: Dict[str, Dict] = {
    "shanghai": {
        "worldmove_npz": "396_CN_Shanghai.npz",
        "place_query": "Pudong New Area, Shanghai, China",
        "peak_slots": (14, 20),  # 7:00-10:00 AM
        "top_k_od": 30,
        "min_demand": 10,
        "bbox_coverage": 0.98,
        "bbox_buffer_km": 5.0,
    },
    "singapore": {
        "worldmove_npz": "539_SG_Singapore.npz",
        "place_query": "Central Region, Singapore",
        "peak_slots": (14, 20),
        "top_k_od": 40,
        "min_demand": 5,
        "bbox_coverage": 0.98,
        "bbox_buffer_km": 5.0,
    },
    "hiroshima": {
        "worldmove_npz": "hzy-558_JPN_Hiroshima.npz",
        "place_query": "Naka-ku, Hiroshima, Japan",
        "peak_slots": (14, 20),
        "top_k_od": 25,
        "min_demand": 3,
        "bbox_coverage": 0.98,
        "bbox_buffer_km": 5.0,
    },
}


def _bbox_from_lonlat(
    lons: np.ndarray,
    lats: np.ndarray,
    buffer_km: float,
) -> Tuple[float, float, float, float]:
    west = float(np.min(lons))
    east = float(np.max(lons))
    south = float(np.min(lats))
    north = float(np.max(lats))

    if buffer_km > 0:
        lat0 = 0.5 * (south + north)
        dlat = float(buffer_km) / 111.0
        cos_lat = float(np.cos(np.deg2rad(lat0)))
        cos_lat = max(cos_lat, 1e-6)
        dlon = float(buffer_km) / (111.0 * cos_lat)
        west -= dlon
        east += dlon
        south -= dlat
        north += dlat

    return (west, south, east, north)


def compute_city_od_bbox(
    city: str,
    use_peak_hours: bool = True,
    coverage: float = 0.98,
    buffer_km: float = 5.0,
) -> Tuple[float, float, float, float]:
    if city not in CITY_CONFIGS:
        raise ValueError(f"Unknown city: {city}")

    config = CITY_CONFIGS[city]
    npz_path = os.path.join(get_worldmove_root(), config["worldmove_npz"])
    traj, pop, grid_coords = load_worldmove_data(npz_path)
    time_filter = config["peak_slots"] if use_peak_hours else None
    origin_cells, dest_cells, _ = extract_od_demand(traj, grid_coords, min_dwell=2, time_slot_filter=time_filter)

    cells = np.concatenate([origin_cells, dest_cells])
    if cells.size == 0:
        raise ValueError(f"No OD cells extracted for {city}")

    unique, counts = np.unique(cells, return_counts=True)
    order = np.argsort(counts)[::-1]
    unique = unique[order]
    counts = counts[order]
    cum = np.cumsum(counts, dtype=float) / float(np.sum(counts))

    keep = cum <= float(coverage)
    if not np.any(keep):
        keep = np.zeros_like(unique, dtype=bool)
        keep[0] = True

    kept_cells = unique[keep]
    coords = [grid_coords[int(c)] for c in kept_cells if int(c) in grid_coords]
    if not coords:
        raise ValueError(f"No coordinates for OD cells in {city}")

    lons = np.array([c[0] for c in coords], dtype=float)
    lats = np.array([c[1] for c in coords], dtype=float)
    return _bbox_from_lonlat(lons, lats, float(buffer_km))


def get_city_od_points(
    city: str,
    use_peak_hours: bool = True,
    coverage: float = 0.98,
    max_points: Optional[int] = 2000,
    seed: int = 7,
) -> Tuple[np.ndarray, np.ndarray]:
    if city not in CITY_CONFIGS:
        raise ValueError(f"Unknown city: {city}")

    config = CITY_CONFIGS[city]
    npz_path = os.path.join(get_worldmove_root(), config["worldmove_npz"])
    traj, pop, grid_coords = load_worldmove_data(npz_path)
    time_filter = config["peak_slots"] if use_peak_hours else None
    origin_cells, dest_cells, _ = extract_od_demand(traj, grid_coords, min_dwell=2, time_slot_filter=time_filter)

    cells = np.concatenate([origin_cells, dest_cells])
    if cells.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    unique, counts = np.unique(cells, return_counts=True)
    order = np.argsort(counts)[::-1]
    unique = unique[order]
    counts = counts[order]
    cum = np.cumsum(counts, dtype=float) / float(np.sum(counts))

    keep = cum <= float(coverage)
    if not np.any(keep):
        keep = np.zeros_like(unique, dtype=bool)
        keep[0] = True

    kept_cells = unique[keep]
    coords = [grid_coords[int(c)] for c in kept_cells if int(c) in grid_coords]
    if not coords:
        return np.array([], dtype=float), np.array([], dtype=float)

    lons = np.array([c[0] for c in coords], dtype=float)
    lats = np.array([c[1] for c in coords], dtype=float)

    if max_points is not None and int(max_points) > 0 and lons.size > int(max_points):
        rng = np.random.default_rng(int(seed))
        idx = rng.choice(lons.size, size=int(max_points), replace=False)
        lons = lons[idx]
        lats = lats[idx]

    return lons, lats


def get_project_root() -> str:
    """Get project root directory."""
    return os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))


def get_cache_dir() -> str:
    """Get cache directory for networks."""
    cache_dir = os.path.join(get_project_root(), "cache", "networks")
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def get_worldmove_root() -> str:
    """Get WorldMove data directory."""
    return os.path.join(get_project_root(), "worldmove", "raw_data")


def load_city_network(
    city: str,
    simplify: bool = True,
    use_cache: bool = True,
    area_mode: str = "place",
    bbox_buffer_km: Optional[float] = None,
    bbox_coverage: Optional[float] = None,
) -> nx.MultiDiGraph:
    """
    Load OSM network for specified city.

    Args:
        city: City name (shanghai, singapore, hiroshima)
        simplify: Whether to simplify graph topology
        use_cache: Whether to use cached network

    Returns:
        Prepared network graph with free_flow_time, capacity, edge attributes
    """
    if city not in CITY_CONFIGS:
        raise ValueError(f"Unknown city: {city}. Available: {list(CITY_CONFIGS.keys())}")

    config = CITY_CONFIGS[city]
    cache_dir = get_cache_dir() if use_cache else None

    if area_mode == "place":
        place_name = config["place_query"]
        bbox = None
    elif area_mode == "od_bbox":
        cov = float(bbox_coverage) if bbox_coverage is not None else float(config.get("bbox_coverage", 0.98))
        buf = float(bbox_buffer_km) if bbox_buffer_km is not None else float(config.get("bbox_buffer_km", 5.0))
        bbox = compute_city_od_bbox(city, use_peak_hours=True, coverage=cov, buffer_km=buf)
        place_name = None
        print(f"Using OD-derived bbox for {city}: {bbox}")
    else:
        raise ValueError(f"Unknown area_mode: {area_mode}")

    G = load_osm_network(
        place_name=place_name,
        bbox=bbox,
        network_type="drive",
        simplify=simplify,
        cache_dir=cache_dir,
    )

    G = prepare_network_for_simulation(G)

    print(f"Loaded {city} network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    return G


def load_city_od_demand(
    city: str,
    G: nx.MultiDiGraph,
    use_peak_hours: bool = True,
    top_k_od: Optional[int] = None,
    min_demand: Optional[int] = None,
) -> List[Tuple[int, int, int]]:
    """
    Load WorldMove OD demand mapped to network nodes.

    Args:
        city: City name
        G: Network graph
        use_peak_hours: Whether to filter to peak hours only

    Returns:
        List of (origin_node, dest_node, demand_count)
    """
    if city not in CITY_CONFIGS:
        raise ValueError(f"Unknown city: {city}")

    config = CITY_CONFIGS[city]
    npz_path = os.path.join(get_worldmove_root(), config["worldmove_npz"])

    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"WorldMove data not found: {npz_path}")

    traj, pop, grid_coords = load_worldmove_data(npz_path)

    time_filter = config["peak_slots"] if use_peak_hours else None

    origin_cells, dest_cells, time_slots = extract_od_demand(
        traj, grid_coords, min_dwell=2, time_slot_filter=time_filter
    )

    top_k_eff = config["top_k_od"] if top_k_od is None else int(top_k_od)
    min_demand_eff = config["min_demand"] if min_demand is None else int(min_demand)

    od_demand = aggregate_od_to_network(
        origin_cells,
        dest_cells,
        grid_coords,
        G,
        top_k=top_k_eff,
        min_demand=min_demand_eff,
    )

    return od_demand


def prepare_city_simulation_inputs(
    city: str,
    use_subgraph: bool = True,
    subgraph_buffer: int = 2,
    area_mode: str = "place",
    bbox_buffer_km: Optional[float] = None,
    bbox_coverage: Optional[float] = None,
    top_k_od: Optional[int] = None,
    min_demand: Optional[int] = None,
) -> Tuple[nx.MultiDiGraph, List[Tuple[int, int]], np.ndarray]:
    """
    Full pipeline: load network, load demand, prepare for simulation.

    Args:
        city: City name
        use_subgraph: Whether to extract OD-relevant subgraph
        subgraph_buffer: Buffer hops for subgraph extraction

    Returns:
        Tuple of (G_multi, od_pairs, demand_per_od)
        - G_multi: Network graph
        - od_pairs: List of (origin, destination) tuples
        - demand_per_od: Array of demand counts per OD pair
    """
    # Load full network
    G = load_city_network(
        city,
        area_mode=area_mode,
        bbox_buffer_km=bbox_buffer_km,
        bbox_coverage=bbox_coverage,
    )

    # Load OD demand
    od_demand = load_city_od_demand(city, G, top_k_od=top_k_od, min_demand=min_demand)

    if not od_demand:
        raise ValueError(f"No OD demand extracted for {city}")

    # Extract od_pairs and demand array
    od_pairs = [(o, d) for o, d, _ in od_demand]
    demand_per_od = np.array([cnt for _, _, cnt in od_demand], dtype=np.int32)

    # Extract subgraph if requested
    if use_subgraph:
        origins = [o for o, d in od_pairs]
        destinations = [d for o, d in od_pairs]
        G = subgraph_for_od(G, origins, destinations, buffer_hops=subgraph_buffer)

        # Re-assign edge IDs after subgraph extraction
        from dtd_srdt.osm_network import assign_edge_ids
        G = assign_edge_ids(G)

    print(f"Prepared {city} simulation inputs:")
    print(f"  Network: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    print(f"  OD pairs: {len(od_pairs)}")
    print(f"  Total demand: {demand_per_od.sum()}")

    return G, od_pairs, demand_per_od


def validate_od_connectivity(
    G: nx.MultiDiGraph,
    od_pairs: List[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """
    Validate that OD pairs have paths in the network.

    Returns only valid OD pairs with connectivity.
    """
    G_simple = nx.DiGraph()
    for u, v, k, data in G.edges(keys=True, data=True):
        t0 = float(data.get("free_flow_time", data.get("length", 1.0)))
        if G_simple.has_edge(u, v):
            if t0 < float(G_simple[u][v].get("free_flow_time", float("inf"))):
                G_simple[u][v]["free_flow_time"] = t0
        else:
            G_simple.add_edge(u, v, free_flow_time=t0)
    valid_pairs = []

    for o, d in od_pairs:
        if o not in G_simple or d not in G_simple:
            continue
        if nx.has_path(G_simple, o, d):
            valid_pairs.append((o, d))

    if len(valid_pairs) < len(od_pairs):
        print(f"Warning: {len(od_pairs) - len(valid_pairs)} OD pairs have no path")

    return valid_pairs


def list_available_cities() -> List[str]:
    """Return list of available city names."""
    return list(CITY_CONFIGS.keys())


def get_city_info(city: str) -> Dict:
    """Get configuration info for a city."""
    if city not in CITY_CONFIGS:
        raise ValueError(f"Unknown city: {city}")
    return CITY_CONFIGS[city].copy()
