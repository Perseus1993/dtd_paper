"""
City Spatial Visualization Script.

Generates spatial maps showing:
A. Link flow distribution
B. Link travel time / congestion
C. Shock impact spatial distribution
"""

import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from dtd_srdt.sioux_sim import run_sioux_simulation
from dtd_srdt.city_plotting import (
    plot_link_flow_map,
    plot_congestion_map,
    plot_shock_impact_map,
    plot_comparison_panel,
    plot_od_points_map,
)

from city_loader import (
    prepare_city_simulation_inputs,
    list_available_cities,
    validate_od_connectivity,
    get_city_od_points,
    load_city_network,
)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def main():
    available_cities = list_available_cities()

    parser = argparse.ArgumentParser(
        description="Generate spatial visualizations for city DTD simulation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--city", type=str, required=True, choices=available_cities)
    parser.add_argument("--scenario", type=str, default="S1", choices=["S0", "S1", "S2", "G", "N", "GN", "R"])
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--k_routes", type=int, default=3)
    parser.add_argument("--num_windows", type=int, default=20)
    parser.add_argument("--theta", type=float, default=3.0)
    parser.add_argument("--shock_day", type=int, default=30)
    parser.add_argument("--shock_len", type=int, default=11)
    parser.add_argument("--shock_factor", type=float, default=0.3)
    parser.add_argument("--shock_k_hops", type=int, default=0)
    parser.add_argument("--capacity_scale", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument("--fig_root", type=str, default=os.path.join("outputs", "city"))
    parser.add_argument("--no_show", action="store_true")

    parser.add_argument("--no_subgraph", action="store_true")
    parser.add_argument("--subgraph_buffer", type=int, default=2)

    parser.add_argument("--area_mode", type=str, default="place", choices=["place", "od_bbox"])
    parser.add_argument("--bbox_coverage", type=float, default=None)
    parser.add_argument("--bbox_buffer_km", type=float, default=None)

    parser.add_argument("--use_all_od", action="store_true")
    parser.add_argument("--top_k_od", type=int, default=None)
    parser.add_argument("--min_demand", type=int, default=None)

    parser.add_argument("--plot_od_points", action="store_true")
    parser.add_argument("--od_points_max", type=int, default=2000)
    parser.add_argument("--diag_only", action="store_true")

    # Visualization options
    parser.add_argument("--baseline_day", type=int, default=None,
                       help="Day to use as baseline (default: shock_day - 5)")
    parser.add_argument("--shock_peak_day", type=int, default=None,
                       help="Day during shock to visualize (default: mid-shock)")
    parser.add_argument("--recovery_day", type=int, default=None,
                       help="Day after shock to visualize (default: days - 5)")

    args = parser.parse_args()

    def _clamp_day(d: int) -> int:
        return int(max(1, min(int(d), int(args.days))))

    # Set defaults
    if args.outdir is None:
        scenario_tag = str(args.scenario)
        suffixes: List[str] = []
        if str(args.area_mode) == "od_bbox":
            suffixes.append("od_bbox")
        if bool(args.no_subgraph):
            suffixes.append("fullnet")
        if int(args.shock_k_hops) > 0:
            suffixes.append(f"shock_k{int(args.shock_k_hops)}")
        if bool(args.plot_od_points):
            suffixes.append("od_points")
        if bool(args.diag_only):
            suffixes.append("diag")

        if bool(args.use_all_od):
            suffixes.append("odall")
        else:
            if args.top_k_od is not None:
                suffixes.append(f"odk{int(args.top_k_od)}")
            if args.min_demand is not None:
                suffixes.append(f"mind{int(args.min_demand)}")

        run_name = scenario_tag + ("_" + "_".join(suffixes) if suffixes else "")
        args.outdir = os.path.join("spatial", str(args.city), run_name)

    if os.path.isabs(args.outdir):
        outdir = args.outdir
    else:
        outdir = os.path.join(args.fig_root, args.outdir)

    _ensure_dir(outdir)

    # Default days to visualize
    baseline_day = _clamp_day(args.baseline_day if args.baseline_day else (args.shock_day - 5))
    shock_peak_day = _clamp_day(args.shock_peak_day if args.shock_peak_day else (args.shock_day + args.shock_len // 2))
    recovery_day = _clamp_day(args.recovery_day if args.recovery_day else (args.shock_day + args.shock_len + 10))

    print("=" * 60)
    print(f"City Spatial Visualization")
    print(f"City: {args.city}")
    print(f"Scenario: {args.scenario}")
    print(f"Days: baseline={baseline_day}, shock_peak={shock_peak_day}, recovery={recovery_day}")
    print("=" * 60)

    # Load city data
    print("\nLoading city network and OD demand...")
    G_multi, od_pairs, demand_per_od = prepare_city_simulation_inputs(
        args.city,
        use_subgraph=not args.no_subgraph,
        subgraph_buffer=int(args.subgraph_buffer),
        area_mode=args.area_mode,
        bbox_buffer_km=args.bbox_buffer_km,
        bbox_coverage=args.bbox_coverage,
        top_k_od=(0 if args.use_all_od else args.top_k_od),
        min_demand=(1 if args.use_all_od else args.min_demand),
    )

    if args.plot_od_points:
        cov = float(args.bbox_coverage) if args.bbox_coverage is not None else 0.98
        lons, lats = get_city_od_points(
            args.city,
            use_peak_hours=True,
            coverage=cov,
            max_points=int(args.od_points_max) if args.od_points_max is not None else None,
            seed=int(args.seed),
        )
        if args.no_subgraph:
            G_plot = G_multi
        else:
            G_plot = load_city_network(
                args.city,
                area_mode=args.area_mode,
                bbox_buffer_km=args.bbox_buffer_km,
                bbox_coverage=args.bbox_coverage,
            )
        plot_od_points_map(
            G_plot,
            lons,
            lats,
            os.path.join(outdir, "od_points.png"),
            title=f"{args.city.title()} OD Points Overlay",
            show=not args.no_show,
        )

        if args.diag_only:
            print(f"\nDone! Diagnostic figure saved to: {outdir}")
            return

    # Validate connectivity
    valid_od = validate_od_connectivity(G_multi, od_pairs)
    if len(valid_od) < len(od_pairs):
        valid_indices = [i for i, od in enumerate(od_pairs) if od in valid_od]
        od_pairs = valid_od
        demand_per_od = demand_per_od[valid_indices]

    avg_demand = int(np.mean(demand_per_od))
    print(f"\nRunning simulation with link tracking...")

    results = run_sioux_simulation(
        G_multi,
        scenario=args.scenario,
        days=args.days,
        od_pairs=od_pairs,
        demand_per_od=avg_demand,
        k_routes=args.k_routes,
        num_dep_windows=args.num_windows,
        dep_window_size=1.0,
        target_arrival=25.0,
        beta_early=0.8,
        beta_late=1.8,
        theta=args.theta,
        init_var=4.0,
        info_var_base=0.2,
        exp_var=2.0,
        share_n=4.0,
        penetration=1.0,
        shock_day=args.shock_day,
        shock_len=args.shock_len,
        shock_edge_id=0,
        shock_factor=args.shock_factor,
        shock_k_hops=int(args.shock_k_hops),
        capacity_scale=args.capacity_scale,
        seed=args.seed,
        track_link_data=True,  # Enable link tracking
    )

    daily_link_flows = results.get("daily_link_flows", [])
    daily_link_tt = results.get("daily_link_tt", [])

    if not daily_link_flows or not daily_link_tt:
        print("ERROR: Link tracking data not available!")
        sys.exit(1)

    print(f"\nGenerating spatial visualizations...")

    # Convert 1-indexed days to 0-indexed
    baseline_idx = baseline_day - 1
    shock_idx = shock_peak_day - 1
    recovery_idx = recovery_day - 1

    # A. Link Flow Maps
    print("  A. Link flow maps...")
    for day_idx, day_name in [(baseline_idx, "baseline"), (shock_idx, "shock"), (recovery_idx, "recovery")]:
        if 0 <= day_idx < len(daily_link_flows):
            plot_link_flow_map(
                G_multi,
                daily_link_flows[day_idx],
                os.path.join(outdir, f"flow_{day_name}_day{day_idx+1}.png"),
                title=f"{args.city.title()} Link Flow - {day_name.title()} (Day {day_idx+1})",
                show=not args.no_show,
            )

    # B. Congestion Maps
    print("  B. Congestion maps...")
    for day_idx, day_name in [(baseline_idx, "baseline"), (shock_idx, "shock"), (recovery_idx, "recovery")]:
        if 0 <= day_idx < len(daily_link_tt):
            plot_congestion_map(
                G_multi,
                daily_link_tt[day_idx],
                os.path.join(outdir, f"congestion_{day_name}_day{day_idx+1}.png"),
                title=f"{args.city.title()} Congestion Ratio - {day_name.title()} (Day {day_idx+1})",
                mode="ratio",
                show=not args.no_show,
            )

    # C. Shock Impact Map
    print("  C. Shock impact map...")
    if 0 <= baseline_idx < len(daily_link_tt) and 0 <= shock_idx < len(daily_link_tt):
        # Find shocked edge for highlighting
        shock_edge_id = int(results.get("shock_edge_selected", -1))
        shock_edges = []
        shock_edge_ids_arr = results.get("shock_edge_ids", None)
        shock_edge_ids = set(int(x) for x in np.asarray(shock_edge_ids_arr, dtype=int).tolist()) if shock_edge_ids_arr is not None else set()
        if not shock_edge_ids and shock_edge_id >= 0:
            shock_edge_ids = {int(shock_edge_id)}

        if shock_edge_ids:
            for u, v, k, data in G_multi.edges(keys=True, data=True):
                if int(data.get("edge", -1)) in shock_edge_ids:
                    shock_edges.append((int(u), int(v), int(k)))

        plot_shock_impact_map(
            G_multi,
            daily_link_tt[shock_idx],
            daily_link_tt[baseline_idx],
            os.path.join(outdir, f"shock_impact_day{shock_idx+1}_vs_day{baseline_idx+1}.png"),
            title=f"{args.city.title()} Shock Impact (Day {shock_idx+1} vs Baseline Day {baseline_idx+1})",
            mode="ratio",
            shock_edges=shock_edges if shock_edges else None,
            show=not args.no_show,
        )

    # D. 3-Panel Comparison
    print("  D. Comparison panel...")
    if (0 <= baseline_idx < len(daily_link_tt) and
        0 <= shock_idx < len(daily_link_tt) and
        0 <= recovery_idx < len(daily_link_tt)):

        plot_comparison_panel(
            G_multi,
            daily_link_tt[baseline_idx],
            daily_link_tt[shock_idx],
            daily_link_tt[recovery_idx],
            os.path.join(outdir, "comparison_panel.png"),
            title=f"{args.city.title()}: Baseline (Day {baseline_idx+1}) vs Shock (Day {shock_idx+1}) vs Recovery (Day {recovery_idx+1})",
            show=not args.no_show,
        )

    print(f"\nDone! Figures saved to: {outdir}")
    print(f"\nGenerated files:")
    for f in sorted(os.listdir(outdir)):
        if f.endswith(".png"):
            print(f"  - {f}")


if __name__ == "__main__":
    main()
