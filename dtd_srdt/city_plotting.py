"""
City-scale spatial visualization for DTD simulation.

Provides map-based visualizations of:
A. Link flow distribution
B. Link travel time / congestion
C. Shock impact spatial distribution
"""

import os
from typing import Dict, List, Optional, Tuple, Union

import networkx as nx
import numpy as np

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    import matplotlib.ticker as mticker
    from matplotlib.collections import LineCollection
    from matplotlib.colorbar import ColorbarBase
    from matplotlib.cm import ScalarMappable
except ImportError:
    plt = None

try:
    import osmnx as ox
except ImportError:
    ox = None


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _get_edge_coords(G: nx.MultiDiGraph) -> Dict[Tuple[int, int, int], Tuple[np.ndarray, np.ndarray]]:
    """Extract edge coordinates from graph."""
    edge_coords = {}
    for u, v, k in G.edges(keys=True):
        x0, y0 = G.nodes[u].get("x", 0), G.nodes[u].get("y", 0)
        x1, y1 = G.nodes[v].get("x", 0), G.nodes[v].get("y", 0)
        edge_coords[(u, v, k)] = (np.array([x0, x1]), np.array([y0, y1]))
    return edge_coords


def _normalize_values(values: np.ndarray, vmin: Optional[float] = None, vmax: Optional[float] = None) -> np.ndarray:
    """Normalize values to [0, 1] range."""
    values = np.asarray(values, dtype=float)
    if vmin is None:
        vmin = np.nanmin(values)
    if vmax is None:
        vmax = np.nanmax(values)
    if vmax - vmin < 1e-10:
        return np.zeros_like(values)
    return np.clip((values - vmin) / (vmax - vmin), 0, 1)


def _select_best_keys(G: nx.MultiDiGraph) -> Dict[Tuple[int, int], int]:
    best_key: Dict[Tuple[int, int], int] = {}
    best_t0: Dict[Tuple[int, int], float] = {}
    for u, v, k, data in G.edges(keys=True, data=True):
        t0 = float(data.get("free_flow_time", float("inf")))
        key = (int(u), int(v))
        if key not in best_t0 or t0 < best_t0[key]:
            best_t0[key] = t0
            best_key[key] = int(k)
    return best_key


def _edge_segment(G: nx.MultiDiGraph, u: int, v: int, k: int) -> List[Tuple[float, float]]:
    data = G[u][v][k]
    geom = data.get("geometry")
    if geom is not None:
        try:
            xs, ys = geom.xy
            return list(zip([float(x) for x in xs], [float(y) for y in ys]))
        except Exception:
            pass
    x0, y0 = G.nodes[u].get("x", 0), G.nodes[u].get("y", 0)
    x1, y1 = G.nodes[v].get("x", 0), G.nodes[v].get("y", 0)
    return [(float(x0), float(y0)), (float(x1), float(y1))]


def plot_link_flow_map(
    G: nx.MultiDiGraph,
    link_flows: Dict[Tuple[int, int], float],
    outpath: str,
    title: str = "Link Flow Distribution",
    cmap: str = "YlOrRd",
    figsize: Tuple[float, float] = (12, 10),
    show: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """
    Plot link flow distribution on city map.

    Args:
        G: Network graph with node positions
        link_flows: Dict mapping (u, v) -> flow value
        outpath: Output file path
        title: Plot title
        cmap: Colormap name
        figsize: Figure size
        show: Whether to show plot
        vmin, vmax: Color scale bounds
    """
    if plt is None:
        raise ImportError("matplotlib required")

    _ensure_dir(os.path.dirname(outpath) if os.path.dirname(outpath) else ".")

    fig, ax = plt.subplots(figsize=figsize)

    best_key = _select_best_keys(G)

    # Collect edge data
    segments0 = []
    segments_pos = []
    flows_pos = []

    for u, v, k, data in G.edges(keys=True, data=True):
        if best_key.get((int(u), int(v))) != int(k):
            continue

        flow = link_flows.get((u, v), 0.0)

        seg = _edge_segment(G, int(u), int(v), int(k))
        if float(flow) > 0.0:
            segments_pos.append(seg)
            flows_pos.append(float(flow))
        else:
            segments0.append(seg)

    if segments0:
        lc0 = LineCollection(segments0, colors="#D0D0D0", linewidths=0.2, alpha=0.25)
        ax.add_collection(lc0)

    flows_pos_arr = np.asarray(flows_pos, dtype=float)
    if flows_pos_arr.size == 0:
        ax.autoscale()
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.tight_layout()
        fig.savefig(outpath, dpi=200, bbox_inches="tight")
        if show:
            plt.show()
        plt.close(fig)
        return

    if vmax is None:
        vmax = float(np.percentile(flows_pos_arr, 99))
    if vmin is None:
        vmin = 0.0

    pos_min = float(np.min(flows_pos_arr))
    pos_max = float(np.max(flows_pos_arr))
    use_log = (pos_max / max(pos_min, 1e-9)) >= 50.0

    if use_log:
        vmin_c = float(max(1.0, np.percentile(flows_pos_arr, 5)))
        vmax_c = float(max(vmin_c * 1.01, vmax))
        norm = mcolors.LogNorm(vmin=vmin_c, vmax=vmax_c)
        widths = 0.8 + 4.5 * _normalize_values(np.log10(flows_pos_arr), np.log10(vmin_c), np.log10(vmax_c))
        tick_vals = np.unique(np.round(np.geomspace(vmin_c, vmax_c, num=5)).astype(int))
        tick_vals = [int(x) for x in tick_vals if x > 0]
    else:
        vmax_c = float(max(vmin + 1e-9, vmax))
        norm = mcolors.Normalize(vmin=float(vmin), vmax=vmax_c)
        widths = 0.8 + 4.5 * _normalize_values(flows_pos_arr, float(vmin), vmax_c)
        p50 = float(np.percentile(flows_pos_arr, 50))
        p90 = float(np.percentile(flows_pos_arr, 90))
        tick_vals = [float(vmin), p50, p90, vmax_c]

    colors = plt.cm.get_cmap(cmap)(norm(flows_pos_arr))
    lc = LineCollection(segments_pos, colors=colors, linewidths=widths, alpha=0.9)
    ax.add_collection(lc)

    # Add colorbar
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Link Flow (vehicles)", fontsize=10)
    try:
        cbar.set_ticks(tick_vals)
    except Exception:
        pass
    try:
        cbar.ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _pos: f"{x:,.0f}"))
    except Exception:
        pass

    try:
        p50 = float(np.percentile(flows_pos_arr, 50))
        p95 = float(np.percentile(flows_pos_arr, 95))
        ax.text(
            0.01,
            0.01,
            f"flow>0: min={pos_min:,.0f}, p50={p50:,.0f}, p95={p95:,.0f}, max={pos_max:,.0f}",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=9,
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )
    except Exception:
        pass

    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    try:
        ax.ticklabel_format(useOffset=False)
    except Exception:
        pass

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


def plot_od_points_map(
    G: nx.MultiDiGraph,
    lons: np.ndarray,
    lats: np.ndarray,
    outpath: str,
    title: str = "OD Points Overlay",
    figsize: Tuple[float, float] = (12, 10),
    show: bool = False,
    edge_color: str = "#D0D0D0",
    edge_width: float = 0.4,
    point_color: str = "#1f77b4",
    point_size: float = 6.0,
    point_alpha: float = 0.5,
) -> None:
    if plt is None:
        raise ImportError("matplotlib required")

    _ensure_dir(os.path.dirname(outpath) if os.path.dirname(outpath) else ".")

    fig, ax = plt.subplots(figsize=figsize)

    best_key = _select_best_keys(G)
    segments = []
    for u, v, k, data in G.edges(keys=True, data=True):
        if best_key.get((int(u), int(v))) != int(k):
            continue
        segments.append(_edge_segment(G, int(u), int(v), int(k)))

    if segments:
        lc = LineCollection(segments, colors=edge_color, linewidths=edge_width, alpha=1.0)
        ax.add_collection(lc)

    if lons.size > 0:
        ax.scatter(lons, lats, s=float(point_size), c=point_color, alpha=float(point_alpha), linewidths=0.0)

    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    try:
        ax.ticklabel_format(useOffset=False)
    except Exception:
        pass

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


def plot_congestion_map(
    G: nx.MultiDiGraph,
    link_tt: Dict[Tuple[int, int], float],
    outpath: str,
    title: str = "Link Travel Time Distribution",
    mode: str = "ratio",  # "time" or "ratio"
    cmap: str = "RdYlGn_r",
    figsize: Tuple[float, float] = (12, 10),
    show: bool = False,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> None:
    """
    Plot link travel time / congestion distribution.

    Args:
        G: Network graph
        link_tt: Dict mapping (u, v) -> travel time
        outpath: Output file path
        title: Plot title
        mode: "time" for absolute travel time, "ratio" for congestion ratio (τ/τ₀)
        cmap: Colormap name
        figsize: Figure size
        show: Whether to show plot
        vmin, vmax: Color scale bounds
    """
    if plt is None:
        raise ImportError("matplotlib required")

    _ensure_dir(os.path.dirname(outpath) if os.path.dirname(outpath) else ".")

    fig, ax = plt.subplots(figsize=figsize)

    best_key = _select_best_keys(G)

    segments = []
    values = []

    for u, v, k, data in G.edges(keys=True, data=True):
        if best_key.get((int(u), int(v))) != int(k):
            continue

        tt = link_tt.get((u, v), data.get("free_flow_time", 1.0))

        if mode == "ratio":
            ff = float(data.get("free_flow_time", 1.0))
            val = tt / max(ff, 0.01)  # Congestion ratio
        else:
            val = tt

        segments.append(_edge_segment(G, int(u), int(v), int(k)))
        values.append(val)

    values = np.array(values)

    # Set bounds
    if mode == "ratio":
        if vmin is None:
            vmin = 1.0
        if vmax is None:
            vmax = max(3.0, np.percentile(values, 95))
        label = "Congestion Ratio (τ/τ₀)"
    else:
        if vmin is None:
            vmin = 0
        if vmax is None:
            vmax = np.percentile(values, 95)
        label = "Travel Time (min)"

    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    colors = plt.cm.get_cmap(cmap)(norm(values))

    widths = 0.5 + 2.5 * _normalize_values(values, vmin, vmax)

    lc = LineCollection(segments, colors=colors, linewidths=widths, alpha=0.8)
    ax.add_collection(lc)

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(label, fontsize=10)

    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    try:
        ax.ticklabel_format(useOffset=False)
    except Exception:
        pass

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


def plot_shock_impact_map(
    G: nx.MultiDiGraph,
    link_tt_shock: Dict[Tuple[int, int], float],
    link_tt_baseline: Dict[Tuple[int, int], float],
    outpath: str,
    title: str = "Shock Impact (Travel Time Change)",
    mode: str = "ratio",  # "diff" or "ratio"
    cmap: str = "RdBu_r",
    figsize: Tuple[float, float] = (12, 10),
    show: bool = False,
    shock_edge: Optional[Tuple[int, int, int]] = None,
    shock_edges: Optional[List[Tuple[int, int, int]]] = None,
) -> None:
    """
    Plot shock impact spatial distribution.

    Shows relative change from baseline: τ_shock - τ_baseline or τ_shock / τ_baseline

    Args:
        G: Network graph
        link_tt_shock: Travel times during shock
        link_tt_baseline: Baseline travel times
        outpath: Output file path
        title: Plot title
        mode: "diff" for absolute difference, "ratio" for relative ratio
        cmap: Colormap name (diverging recommended)
        figsize: Figure size
        show: Whether to show plot
        shock_edge: Optional (u, v, k) tuple to highlight shocked edge
    """
    if plt is None:
        raise ImportError("matplotlib required")

    _ensure_dir(os.path.dirname(outpath) if os.path.dirname(outpath) else ".")

    fig, ax = plt.subplots(figsize=figsize)

    best_key = _select_best_keys(G)

    segments = []
    values = []
    shock_segments: List[List[Tuple[float, float]]] = []

    shock_set = set(tuple(map(int, e)) for e in shock_edges) if shock_edges else set()
    if shock_edge is not None:
        shock_set.add((int(shock_edge[0]), int(shock_edge[1]), int(shock_edge[2])))

    for u, v, k, data in G.edges(keys=True, data=True):
        if best_key.get((int(u), int(v))) != int(k):
            continue

        tt_shock = link_tt_shock.get((u, v), data.get("free_flow_time", 1.0))
        tt_base = link_tt_baseline.get((u, v), data.get("free_flow_time", 1.0))

        if mode == "ratio":
            val = tt_shock / max(tt_base, 0.01)
        else:
            val = tt_shock - tt_base

        segments.append(_edge_segment(G, int(u), int(v), int(k)))
        values.append(val)

        if shock_set and (int(u), int(v), int(k)) in shock_set:
            shock_segments.append(_edge_segment(G, int(u), int(v), int(k)))

    values = np.array(values)

    # Set bounds (centered for diverging colormap)
    if mode == "ratio":
        vmax = max(2.0, np.percentile(np.abs(values - 1), 95) + 1)
        vmin = 2 - vmax  # Symmetric around 1
        center = 1.0
        label = "Travel Time Ratio (shock/baseline)"
    else:
        vmax = max(np.abs(values).max(), 0.1)
        vmin = -vmax
        center = 0.0
        label = "Travel Time Change (min)"

    # Use TwoSlopeNorm for centering
    norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
    colors = plt.cm.get_cmap(cmap)(norm(values))

    widths = 0.5 + 2.5 * _normalize_values(np.abs(values - center), 0, vmax - center)

    lc = LineCollection(segments, colors=colors, linewidths=widths, alpha=0.8)
    ax.add_collection(lc)

    # Highlight shock edges
    if shock_segments:
        lc_shock = LineCollection(shock_segments, colors="black", linewidths=2.5, linestyles="--", alpha=0.95)
        ax.add_collection(lc_shock)
        ax.plot([], [], color="black", linewidth=2.5, linestyle="--", label="Shocked Region")
        ax.legend(loc="upper right")

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label(label, fontsize=10)

    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=12)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)


def plot_spatial_evolution(
    G: nx.MultiDiGraph,
    daily_link_tt: List[Dict[Tuple[int, int], float]],
    days_to_plot: List[int],
    outdir: str,
    title_prefix: str = "Day",
    mode: str = "ratio",
    cmap: str = "RdYlGn_r",
    figsize: Tuple[float, float] = (10, 8),
    show: bool = False,
) -> None:
    """
    Plot spatial distribution across multiple days.

    Args:
        G: Network graph
        daily_link_tt: List of daily link travel time dicts
        days_to_plot: Which days to plot
        outdir: Output directory
        title_prefix: Prefix for titles
        mode: "time" or "ratio"
        cmap: Colormap
        figsize: Figure size
        show: Whether to show
    """
    _ensure_dir(outdir)

    best_key = _select_best_keys(G)

    # Find global bounds
    all_values = []
    for d in days_to_plot:
        if d < len(daily_link_tt):
            for (u, v), tt in daily_link_tt[d].items():
                if mode == "ratio":
                    k = best_key.get((int(u), int(v)))
                    ff = float(G[u][v][k].get("free_flow_time", 1.0)) if (k is not None and G.has_edge(u, v, k)) else 1.0
                    all_values.append(tt / max(ff, 0.01))
                else:
                    all_values.append(tt)

    if not all_values:
        return

    if mode == "ratio":
        vmin, vmax = 1.0, max(3.0, np.percentile(all_values, 95))
    else:
        vmin, vmax = 0, np.percentile(all_values, 95)

    for d in days_to_plot:
        if d < len(daily_link_tt):
            outpath = os.path.join(outdir, f"spatial_day{d:03d}.png")
            plot_congestion_map(
                G, daily_link_tt[d], outpath,
                title=f"{title_prefix} {d+1}",
                mode=mode, cmap=cmap, figsize=figsize,
                show=show, vmin=vmin, vmax=vmax
            )


def plot_comparison_panel(
    G: nx.MultiDiGraph,
    link_tt_baseline: Dict[Tuple[int, int], float],
    link_tt_shock: Dict[Tuple[int, int], float],
    link_tt_recovery: Dict[Tuple[int, int], float],
    outpath: str,
    title: str = "Spatial Comparison: Baseline vs Shock vs Recovery",
    figsize: Tuple[float, float] = (18, 6),
    show: bool = False,
) -> None:
    """
    Plot 3-panel comparison: baseline, shock, recovery.
    """
    if plt is None:
        raise ImportError("matplotlib required")

    _ensure_dir(os.path.dirname(outpath) if os.path.dirname(outpath) else ".")

    fig, axes = plt.subplots(1, 3, figsize=figsize)

    best_key = _select_best_keys(G)

    datasets = [
        (link_tt_baseline, "Baseline", "YlGn"),
        (link_tt_shock, "Shock", "YlOrRd"),
        (link_tt_recovery, "Recovery", "YlGn"),
    ]

    # Find global bounds
    all_ratios = []
    for link_tt, _, _ in datasets:
        for (u, v), tt in link_tt.items():
            k = best_key.get((int(u), int(v)))
            if k is not None and G.has_edge(u, v, k):
                ff = float(G[u][v][k].get("free_flow_time", 1.0))
                all_ratios.append(tt / max(ff, 0.01))

    vmin, vmax = 1.0, max(3.0, np.percentile(all_ratios, 95)) if all_ratios else 3.0

    for ax, (link_tt, label, cmap) in zip(axes, datasets):
        segments = []
        values = []

        for u, v, k, data in G.edges(keys=True, data=True):
            if best_key.get((int(u), int(v))) != int(k):
                continue

            tt = link_tt.get((u, v), data.get("free_flow_time", 1.0))
            ff = data.get("free_flow_time", 1.0)
            val = tt / max(ff, 0.01)

            segments.append(_edge_segment(G, int(u), int(v), int(k)))
            values.append(val)

        values = np.array(values)
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        colors = plt.cm.get_cmap(cmap)(norm(values))
        widths = 0.5 + 2.0 * _normalize_values(values, vmin, vmax)

        lc = LineCollection(segments, colors=colors, linewidths=widths, alpha=0.8)
        ax.add_collection(lc)

        ax.autoscale()
        ax.set_aspect("equal")
        ax.set_title(label, fontsize=11)
        ax.set_xlabel("Longitude", fontsize=9)
        ax.set_ylabel("Latitude", fontsize=9)
        try:
            ax.ticklabel_format(useOffset=False)
        except Exception:
            pass

    # Shared colorbar
    sm = ScalarMappable(cmap="YlOrRd", norm=mcolors.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.02)
    cbar.set_label("Congestion Ratio (τ/τ₀)", fontsize=10)

    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    fig.savefig(outpath, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    plt.close(fig)
