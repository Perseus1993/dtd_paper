import argparse
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))
from dtd_srdt.metrics import compute_disruption_metrics
from dtd_srdt.plotting import plot_results
from dtd_srdt.sioux_sim import run_sioux_simulation

sys.path.append(os.path.dirname(__file__))
from sf import load_graph


def _parse_od_list(values: Optional[List[str]]) -> List[Tuple[int, int]]:
    if not values:
        return [(14, 4), (15, 6)]
    out: List[Tuple[int, int]] = []
    for s in values:
        s = s.replace(" ", "")
        if "-" in s:
            a, b = s.split("-", 1)
        elif "," in s:
            a, b = s.split(",", 1)
        else:
            raise ValueError("OD must be like '14-4' or '14,4'")
        out.append((int(a), int(b)))
    return out


def _resolve_outdir(fig_root: str, outdir: Optional[str]) -> Optional[str]:
    if outdir is None:
        return None
    outdir = str(outdir)
    if os.path.isabs(outdir):
        return outdir
    fig_root = str(fig_root) if fig_root is not None else ""
    if fig_root.strip() in ("", "."):
        return outdir
    return os.path.join(fig_root, outdir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=str,
        default="S1",
        choices=["S0", "S1", "S2", "G", "N", "GN", "R"],
        help="Information scenario. Aliases: S1=G (global navigation), S2=R (rumor/aggregate sharing).",
    )
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--od", action="append", default=None, help="OD pair like '14-4' (repeatable)")
    parser.add_argument("--demand_per_od", type=int, default=500)
    parser.add_argument("--k_routes", type=int, default=5)
    parser.add_argument("--num_windows", type=int, default=20)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--target_arrival", type=float, default=25.0)
    parser.add_argument("--beta_early", type=float, default=0.8)
    parser.add_argument("--beta_late", type=float, default=1.8)
    parser.add_argument("--theta", type=float, default=3.0)
    parser.add_argument(
        "--br_delta",
        type=float,
        default=0.0,
        help="Bounded rationality indifference band (keep previous-day choice if within delta; 0 disables).",
    )
    parser.add_argument("--init_var", type=float, default=4.0)
    parser.add_argument("--info_var", type=float, default=0.2)
    parser.add_argument("--exp_var", type=float, default=2.0)
    parser.add_argument(
        "--s2_report_var",
        type=float,
        default=None,
        help="Variance of S2 shared reported travel time noise. Default None uses exp_var; 0 disables report noise.",
    )
    parser.add_argument("--share_n", type=float, default=4.0)
    parser.add_argument("--penetration", type=float, default=1.0)
    parser.add_argument("--no_pretrip", action="store_true")
    parser.add_argument("--no_posttrip", action="store_true")
    parser.add_argument("--update_rule", type=str, default="bayes", choices=["bayes", "exp_smooth"])
    parser.add_argument("--eta_info", type=float, default=0.3)
    parser.add_argument("--eta_exp", type=float, default=0.3)
    parser.add_argument(
        "--risk_eta",
        type=float,
        default=0.0,
        help="Risk sensitivity coefficient multiplying posterior variance in perceived cost (default 0 disables risk term).",
    )
    parser.add_argument(
        "--s1_lag_days",
        type=int,
        default=1,
        help="Lag (in days) for S1 historical broadcast evaluation. 1 means previous day; >1 introduces information delay.",
    )
    parser.add_argument(
        "--s1_meas_var",
        type=float,
        default=0.0,
        help="Additional measurement variance added to S1 broadcast signal mean (default 0 disables).",
    )
    parser.add_argument(
        "--s1_active_only",
        action="store_true",
        help="If set, S1/G broadcast is only applied to alternatives with positive previous-day flow (coverage-matched variant).",
    )
    parser.add_argument(
        "--s2_info_var_mode",
        type=str,
        default="share_power",
        choices=["share_power", "report_mean_var"],
        help="S2 information-variance mode: share_power uses sigma0^2/(share^n); report_mean_var uses sigma0^2 + sigma_report^2/N_reports.",
    )
    parser.add_argument(
        "--s2_no_fallback",
        action="store_true",
        help="If set, S2/R disables fallback for alternatives with zero previous-day flow (treat as missing signal via infinite variance).",
    )
    parser.add_argument(
        "--social_topology",
        type=str,
        default="ring",
        choices=["ring", "er"],
        help="Social network topology for N/GN: ring (regular) or er (Erdos-Renyi).",
    )
    parser.add_argument(
        "--social_k",
        type=int,
        default=6,
        help="Average degree (ring uses exact degree rounded to even; ER uses p=k/(n-1)).",
    )
    parser.add_argument(
        "--social_var",
        type=float,
        default=1.0,
        help="Observation variance for social signal (sigma_N^2) used in Bayes update.",
    )
    parser.add_argument("--route_only", action="store_true")
    parser.add_argument("--no_sharing", action="store_true")
    parser.add_argument("--shock_day", type=int, default=30)
    parser.add_argument("--shock_len", type=int, default=11)
    parser.add_argument("--shock_edge", type=int, default=0, help="edge id to disrupt; set 0 to auto-select based on day shock_day-1")
    parser.add_argument("--shock_factor", type=float, default=0.3)
    parser.add_argument("--capacity_scale", type=float, default=0.1)
    parser.add_argument("--outdir", type=str, default="figs_sf")
    parser.add_argument(
        "--fig_root",
        type=str,
        default=os.path.join("outputs", "sioux"),
        help="If outdir is a plain folder name, it will be placed under this root (set '' or '.' to disable).",
    )
    parser.add_argument("--no_show", action="store_true")
    parser.add_argument("--no_plot", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=1,
        help="Number of random seeds to run (seed, seed+1, ..., seed+n_seeds-1).",
    )
    parser.add_argument(
        "--plot_each_seed",
        action="store_true",
        help="If set, generate plots for each seed under outdir/seed_<k>.",
    )
    parser.add_argument(
        "--trust_heterogeneity",
        type=str,
        default="none",
        choices=["none", "mixed"],
        help="Trust heterogeneity: 'none' (homogeneous) or 'mixed' (50%% high-trust + 50%% low-trust agents).",
    )
    parser.add_argument(
        "--trust_high_info_var",
        type=float,
        default=0.5,
        help="Info variance for high-trust agents (lower = more trust in information).",
    )
    parser.add_argument(
        "--trust_low_info_var",
        type=float,
        default=4.0,
        help="Info variance for low-trust agents (higher = less trust in information).",
    )
    parser.add_argument(
        "--trust_high_frac",
        type=float,
        default=0.5,
        help="Fraction of agents that are high-trust (default 0.5 = 50%%).",
    )
    args = parser.parse_args()

    outdir = _resolve_outdir(args.fig_root, args.outdir)

    od_pairs = _parse_od_list(args.od)

    G_multi, _, _, _ = load_graph(draw=False)

    n_seeds = max(1, int(args.n_seeds))
    seed_list = [int(args.seed) + i for i in range(n_seeds)]
    do_plot_each_seed = bool(args.plot_each_seed) and (outdir is not None) and (not args.no_plot)

    shock_edge_selected_list: List[int] = []
    metrics_list: List[dict] = []

    for seed in seed_list:
        results = run_sioux_simulation(
            G_multi,
            scenario=args.scenario,
            days=args.days,
            od_pairs=od_pairs,
            demand_per_od=args.demand_per_od,
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
            br_delta=args.br_delta,
            s1_active_only=bool(args.s1_active_only),
            s2_no_fallback=bool(args.s2_no_fallback),
            social_topology=args.social_topology,
            social_k=args.social_k,
            social_var=args.social_var,
            trust_heterogeneity=args.trust_heterogeneity,
            trust_high_info_var=args.trust_high_info_var,
            trust_low_info_var=args.trust_low_info_var,
            trust_high_frac=args.trust_high_frac,
        )

        tail = min(10, len(results["avg_tt"]))
        avg_tt_tail = float(np.mean(results["avg_tt"][-tail:]))
        rel_gap_tail = results["rel_gap"][-tail:]
        rel_gap_tail = rel_gap_tail[np.isfinite(rel_gap_tail)]
        rel_gap_tail_mean = float(np.mean(rel_gap_tail)) if rel_gap_tail.size > 0 else float("nan")
        metrics = compute_disruption_metrics(results, shock_day=args.shock_day, shock_len=args.shock_len)

        shock_edge_selected = (
            int(results["shock_edge_selected"]) if np.isfinite(results["shock_edge_selected"]) else -1
        )
        shock_edge_selected_list.append(shock_edge_selected)

        print(
            f"scenario={args.scenario}  OD={od_pairs}  seed={seed}  avg_tt_last{tail}={avg_tt_tail:.6f}  rel_gap_last{tail}={rel_gap_tail_mean:.6f}  "
            f"shock=(day={args.shock_day},len={args.shock_len},edge={args.shock_edge},factor={args.shock_factor})  "
            f"resolved_edge={shock_edge_selected if shock_edge_selected >= 0 else 'NA'}  cap_scale={args.capacity_scale}"
        )
        if metrics:
            metrics = dict(metrics)
            metrics["avg_tt_last10"] = float(avg_tt_tail)
            metrics["rel_gap_last10"] = float(rel_gap_tail_mean)
            metrics_list.append(metrics)
            print(
                "disruption_metrics  "
                f"baseline_tt={metrics['baseline_tt']:.6f}  "
                f"shock_tt_peak={metrics['shock_tt_peak']:.6f}  "
                f"shock_rg_peak={metrics['shock_rg_peak']:.6f}  "
                f"recovery_day={metrics['recovery_day']}  "
                f"post_shock_tt_var={metrics['post_shock_tt_var']}  "
                f"dep_mean(base/shock/post)={metrics['dep_mean_base']:.3f}/{metrics['dep_mean_shock']:.3f}/{metrics['dep_mean_post']:.3f}  "
                f"hhi(base/shock/post)={metrics['hhi_base']:.6f}/{metrics['hhi_shock']:.6f}/{metrics['hhi_post']:.6f}  "
                f"entropy(base/shock/post)={metrics['entropy_base']:.6f}/{metrics['entropy_shock']:.6f}/{metrics['entropy_post']:.6f}"
            )

        if do_plot_each_seed:
            outdir_seed = os.path.join(outdir, f"seed_{seed}")
            plot_results(
                results,
                title=f"Baseline-1 SRDT Bayes SiouxFalls ({args.scenario}, seed={seed})",
                outdir=outdir_seed,
                show=not args.no_show,
                shock_day=args.shock_day,
                shock_len=args.shock_len,
            )
        elif n_seeds == 1 and (outdir is not None) and (not args.no_plot):
            plot_results(
                results,
                title=f"Baseline-1 SRDT Bayes SiouxFalls ({args.scenario})",
                outdir=outdir,
                show=not args.no_show,
                shock_day=args.shock_day,
                shock_len=args.shock_len,
            )

    if n_seeds > 1 and metrics_list:
        keys = [
            "baseline_tt",
            "shock_tt_peak",
            "shock_rg_peak",
            "recovery_day",
            "post_shock_tt_var",
            "avg_tt_last10",
            "rel_gap_last10",
            "dep_mean_base",
            "dep_mean_shock",
            "dep_mean_post",
            "hhi_base",
            "hhi_shock",
            "hhi_post",
            "entropy_base",
            "entropy_shock",
            "entropy_post",
        ]
        print(f"multiseed_summary  n_seeds={n_seeds}  seeds={seed_list[0]}..{seed_list[-1]}")

        uniq_edges = sorted(set(shock_edge_selected_list))
        if len(uniq_edges) == 1:
            edge_str = str(uniq_edges[0]) if uniq_edges[0] >= 0 else "NA"
            print(f"shock_edge_selected  unique={edge_str}")
        else:
            edge_str = ",".join(str(e) if e >= 0 else "NA" for e in uniq_edges)
            print(f"shock_edge_selected  multiple=[{edge_str}]")

        for k in keys:
            arr = np.array([m.get(k, float("nan")) for m in metrics_list], dtype=float)
            m = np.isfinite(arr)
            if not np.any(m):
                mu = float("nan")
                sd = float("nan")
            else:
                mu = float(np.mean(arr[m]))
                sd = float(np.std(arr[m]))
            print(f"{k}  mean={mu:.6f}  std={sd:.6f}")


if __name__ == "__main__":
    main()
