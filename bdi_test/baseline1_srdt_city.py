"""
Real City DTD Simulation CLI.

Runs Day-to-Day traffic simulation on real city road networks
using WorldMove mobility data for OD demand.
"""

import argparse
import os
import sys
from typing import List

import numpy as np

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))

from dtd_srdt.metrics import compute_disruption_metrics
from dtd_srdt.plotting import plot_results
from dtd_srdt.sioux_sim import run_sioux_simulation

from city_loader import (
    prepare_city_simulation_inputs,
    list_available_cities,
    validate_od_connectivity,
)


def _resolve_outdir(fig_root: str, outdir: str) -> str:
    outdir = str(outdir)
    if os.path.isabs(outdir):
        return outdir
    fig_root = str(fig_root) if fig_root else ""
    if fig_root.strip() in ("", "."):
        return outdir
    return os.path.join(fig_root, outdir)


def main():
    available_cities = list_available_cities()

    parser = argparse.ArgumentParser(
        description="Run DTD simulation on real city road networks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # City selection
    parser.add_argument(
        "--city",
        type=str,
        required=True,
        choices=available_cities,
        help=f"City to simulate. Available: {available_cities}",
    )

    # Scenario
    parser.add_argument(
        "--scenario",
        type=str,
        default="S1",
        choices=["S0", "S1", "S2", "G", "N", "GN", "R"],
        help="Information scenario. S0=no info, S1/G=global, S2/R=rumor/sharing, N=social, GN=fusion",
    )

    # Simulation parameters
    parser.add_argument("--days", type=int, default=60, help="Number of simulation days")
    parser.add_argument("--k_routes", type=int, default=3, help="Number of routes per OD pair")
    parser.add_argument("--num_windows", type=int, default=20, help="Number of departure windows")
    parser.add_argument("--dt", type=float, default=1.0, help="Departure window size (minutes)")
    parser.add_argument("--target_arrival", type=float, default=25.0, help="Target arrival time")

    # Schedule delay parameters
    parser.add_argument("--beta_early", type=float, default=0.8, help="Early arrival penalty")
    parser.add_argument("--beta_late", type=float, default=1.8, help="Late arrival penalty")

    # Choice model parameters
    parser.add_argument("--theta", type=float, default=3.0, help="Logit sensitivity")
    parser.add_argument("--init_var", type=float, default=4.0, help="Initial belief variance")
    parser.add_argument("--info_var", type=float, default=0.2, help="Information variance")
    parser.add_argument("--exp_var", type=float, default=2.0, help="Experience variance")
    parser.add_argument("--share_n", type=float, default=4.0, help="Sharing power parameter")
    parser.add_argument("--penetration", type=float, default=1.0, help="Information penetration rate")

    # Belief update parameters
    parser.add_argument("--no_pretrip", action="store_true", help="Disable pre-trip update")
    parser.add_argument("--no_posttrip", action="store_true", help="Disable post-trip update")
    parser.add_argument("--update_rule", type=str, default="bayes", choices=["bayes", "exp_smooth"])
    parser.add_argument("--eta_info", type=float, default=0.3, help="Info smoothing rate")
    parser.add_argument("--eta_exp", type=float, default=0.3, help="Experience smoothing rate")
    parser.add_argument("--s2_report_var", type=float, default=None, help="S2 report variance")
    parser.add_argument("--risk_eta", type=float, default=0.0, help="Risk sensitivity")
    parser.add_argument("--s1_lag_days", type=int, default=1, help="S1 information lag")
    parser.add_argument("--s1_meas_var", type=float, default=0.0, help="S1 measurement variance")
    parser.add_argument("--s2_info_var_mode", type=str, default="share_power",
                       choices=["share_power", "report_mean_var"])

    # Social network parameters
    parser.add_argument("--social_topology", type=str, default="ring", choices=["ring", "er"])
    parser.add_argument("--social_k", type=int, default=6, help="Social network degree")
    parser.add_argument("--social_var", type=float, default=1.0, help="Social signal variance")

    # Ablation flags
    parser.add_argument("--route_only", action="store_true", help="Route choice only (no departure)")
    parser.add_argument("--no_sharing", action="store_true", help="Disable information sharing")

    # Shock parameters
    parser.add_argument("--shock_day", type=int, default=30, help="Day shock starts")
    parser.add_argument("--shock_len", type=int, default=11, help="Shock duration (days)")
    parser.add_argument("--shock_edge", type=int, default=0, help="Edge to disrupt (0=auto)")
    parser.add_argument("--shock_factor", type=float, default=0.3, help="Capacity reduction factor")
    parser.add_argument("--shock_k_hops", type=int, default=0, help="Expand shock to k-hop neighborhood (0=single edge)")

    # Network parameters
    parser.add_argument("--capacity_scale", type=float, default=0.1, help="Overall capacity scaling")
    parser.add_argument("--no_subgraph", action="store_true", help="Use full network (slower)")
    parser.add_argument("--subgraph_buffer", type=int, default=2, help="Subgraph buffer hops")

    parser.add_argument("--area_mode", type=str, default="place", choices=["place", "od_bbox"])
    parser.add_argument("--bbox_coverage", type=float, default=None)
    parser.add_argument("--bbox_buffer_km", type=float, default=None)

    parser.add_argument("--use_all_od", action="store_true")
    parser.add_argument("--top_k_od", type=int, default=None)
    parser.add_argument("--min_demand", type=int, default=None)

    # Output parameters
    parser.add_argument("--outdir", type=str, default=None, help="Output directory")
    parser.add_argument("--fig_root", type=str, default=os.path.join("outputs", "city"), help="Figure root directory")
    parser.add_argument("--no_show", action="store_true", help="Don't show plots")

    # Seed parameters
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument("--n_seeds", type=int, default=1, help="Number of seeds to run")
    parser.add_argument("--plot_each_seed", action="store_true", help="Plot each seed separately")

    # Demand scaling
    parser.add_argument("--demand_scale", type=float, default=1.0,
                       help="Scale WorldMove demand (useful for calibration)")

    args = parser.parse_args()

    # Set default outdir based on city
    if args.outdir is None:
        args.outdir = os.path.join("sim", str(args.city), str(args.scenario))

    outdir = _resolve_outdir(args.fig_root, args.outdir)

    print(f"=" * 60)
    print(f"Real City DTD Simulation")
    print(f"City: {args.city}")
    print(f"Scenario: {args.scenario}")
    print(f"=" * 60)

    # Load city data
    print("\nLoading city network and OD demand...")
    G_multi, od_pairs, demand_per_od = prepare_city_simulation_inputs(
        args.city,
        use_subgraph=not args.no_subgraph,
        subgraph_buffer=args.subgraph_buffer,
        area_mode=args.area_mode,
        bbox_buffer_km=args.bbox_buffer_km,
        bbox_coverage=args.bbox_coverage,
        top_k_od=(0 if args.use_all_od else args.top_k_od),
        min_demand=(1 if args.use_all_od else args.min_demand),
    )

    # Validate OD connectivity
    valid_od = validate_od_connectivity(G_multi, od_pairs)
    if len(valid_od) < len(od_pairs):
        print(f"Using {len(valid_od)}/{len(od_pairs)} OD pairs with valid paths")
        # Filter demand to match valid pairs
        valid_indices = [i for i, od in enumerate(od_pairs) if od in valid_od]
        od_pairs = valid_od
        demand_per_od = demand_per_od[valid_indices]

    if not od_pairs:
        print("ERROR: No valid OD pairs found!")
        sys.exit(1)

    # Scale demand
    if args.demand_scale != 1.0:
        demand_per_od = (demand_per_od * args.demand_scale).astype(np.int32)
        demand_per_od = np.maximum(demand_per_od, 1)

    # For now, use uniform demand (average of WorldMove demand)
    # This matches the sioux_sim interface
    avg_demand = int(np.mean(demand_per_od))
    print(f"\nUsing average demand per OD: {avg_demand}")

    n_seeds = max(1, args.n_seeds)
    seed_list = [args.seed + i for i in range(n_seeds)]
    do_plot_each = args.plot_each_seed and outdir is not None

    metrics_list: List[dict] = []
    shock_edge_list: List[int] = []

    for seed in seed_list:
        print(f"\n--- Running seed {seed} ---")

        results = run_sioux_simulation(
            G_multi,
            scenario=args.scenario,
            days=args.days,
            od_pairs=od_pairs,
            demand_per_od=avg_demand,
            k_routes=args.k_routes,
            num_dep_windows=args.num_windows,
            dep_window_size=args.dt,
            target_arrival=args.target_arrival,
            beta_early=args.beta_early,
            beta_late=args.beta_late,
            theta=args.theta,
            init_var=args.init_var,
            info_var_base=args.info_var,
            exp_var=args.exp_var,
            share_n=args.share_n,
            penetration=args.penetration,
            shock_day=args.shock_day,
            shock_len=args.shock_len,
            shock_edge_id=args.shock_edge,
            shock_factor=args.shock_factor,
            shock_k_hops=args.shock_k_hops,
            capacity_scale=args.capacity_scale,
            seed=seed,
            s2_report_var=args.s2_report_var,
            route_only=args.route_only,
            no_sharing=args.no_sharing,
            use_pretrip=not args.no_pretrip,
            use_posttrip=not args.no_posttrip,
            update_rule=args.update_rule,
            eta_info=args.eta_info,
            eta_exp=args.eta_exp,
            risk_eta=args.risk_eta,
            s1_lag_days=args.s1_lag_days,
            s1_meas_var=args.s1_meas_var,
            s2_info_var_mode=args.s2_info_var_mode,
            social_topology=args.social_topology,
            social_k=args.social_k,
            social_var=args.social_var,
        )

        # Compute metrics
        tail = min(10, len(results["avg_tt"]))
        avg_tt_tail = float(np.mean(results["avg_tt"][-tail:]))
        rel_gap_tail = results["rel_gap"][-tail:]
        rel_gap_tail = rel_gap_tail[np.isfinite(rel_gap_tail)]
        rel_gap_mean = float(np.mean(rel_gap_tail)) if rel_gap_tail.size > 0 else float("nan")

        metrics = compute_disruption_metrics(results, shock_day=args.shock_day, shock_len=args.shock_len)

        shock_edge = int(results.get("shock_edge_selected", -1))
        if not np.isfinite(shock_edge):
            shock_edge = -1
        shock_edge_list.append(shock_edge)

        print(f"  city={args.city} scenario={args.scenario} seed={seed}")
        print(f"  avg_tt_last{tail}={avg_tt_tail:.4f}  rel_gap_last{tail}={rel_gap_mean:.6f}")

        if metrics:
            metrics_list.append(metrics)
            print(f"  baseline_tt={metrics['baseline_tt']:.4f}  peak_tt={metrics['shock_tt_peak']:.4f}")
            print(f"  recovery_day={metrics['recovery_day']}  post_var={metrics['post_shock_tt_var']:.6f}")

        # Plot
        if do_plot_each:
            out_seed = os.path.join(outdir, f"seed_{seed}")
            plot_results(
                results,
                title=f"{args.city.title()} DTD ({args.scenario}, seed={seed})",
                outdir=out_seed,
                show=not args.no_show,
                shock_day=args.shock_day,
                shock_len=args.shock_len,
            )
        elif n_seeds == 1 and outdir:
            plot_results(
                results,
                title=f"{args.city.title()} DTD ({args.scenario})",
                outdir=outdir,
                show=not args.no_show,
                shock_day=args.shock_day,
                shock_len=args.shock_len,
            )

    # Multi-seed summary
    if n_seeds > 1 and metrics_list:
        print(f"\n{'=' * 60}")
        print(f"Multi-seed Summary (n={n_seeds})")
        print(f"{'=' * 60}")

        keys = ["baseline_tt", "shock_tt_peak", "shock_rg_peak", "recovery_day", "post_shock_tt_var"]
        for k in keys:
            arr = np.array([m.get(k, np.nan) for m in metrics_list], dtype=float)
            valid = arr[np.isfinite(arr)]
            if valid.size > 0:
                print(f"  {k}: mean={np.mean(valid):.4f} std={np.std(valid):.4f}")

    print(f"\nDone! Output saved to: {outdir}")


if __name__ == "__main__":
    main()
