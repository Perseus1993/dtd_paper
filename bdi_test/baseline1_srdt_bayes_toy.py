import argparse
import os
import random
import sys
from typing import Optional
import numpy as np

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))
from dtd_srdt.common import BayesianBelief, bayes_update, bpr_time, relative_gap, schedule_penalties, softmax_choice
from dtd_srdt.metrics import compute_disruption_metrics
from dtd_srdt.plotting import plot_results
from dtd_srdt.toy_sim import run_toy_simulation


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
    parser.add_argument("--scenario", type=str, default="S0", choices=["S0", "S1", "S2"])
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--demand", type=int, default=300)
    parser.add_argument("--num_windows", type=int, default=20)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--target_arrival", type=float, default=10.0)
    parser.add_argument("--beta_early", type=float, default=0.8)
    parser.add_argument("--beta_late", type=float, default=1.8)
    parser.add_argument("--theta", type=float, default=1.0)
    parser.add_argument(
        "--br_delta",
        type=float,
        default=0.0,
        help="Bounded rationality indifference band (keep previous-day choice if within delta; 0 disables).",
    )
    parser.add_argument("--init_var", type=float, default=4.0)
    parser.add_argument("--info_var", type=float, default=4.0)
    parser.add_argument("--exp_var", type=float, default=1.0)
    parser.add_argument(
        "--s2_report_var",
        type=float,
        default=None,
        help="Variance of S2 shared reported travel time noise. Default None uses exp_var; 0 disables report noise.",
    )
    parser.add_argument("--share_n", type=float, default=2.0)
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
        help="If set, S1 broadcast is only applied to alternatives with positive previous-day flow (coverage-matched variant).",
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
        help="If set, S2 disables fallback for alternatives with zero previous-day flow (treat as missing signal via infinite variance).",
    )
    parser.add_argument("--route_only", action="store_true")
    parser.add_argument("--no_sharing", action="store_true")
    parser.add_argument("--shock_day", type=int, default=0)
    parser.add_argument("--shock_len", type=int, default=0)
    parser.add_argument("--shock_route", type=str, default="P2")
    parser.add_argument("--shock_factor", type=float, default=0.3)
    parser.add_argument("--outdir", type=str, default=None)
    parser.add_argument(
        "--fig_root",
        type=str,
        default=os.path.join("outputs", "toy"),
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
    args = parser.parse_args()

    outdir = _resolve_outdir(args.fig_root, args.outdir)

    shock_route = args.shock_route if args.shock_day > 0 and args.shock_len > 0 else None

    n_seeds = max(1, int(args.n_seeds))
    seed_list = [int(args.seed) + i for i in range(n_seeds)]

    metrics_list = []
    avg_tt_tail_list = []
    rel_gap_tail_mean_list = []

    last_results = None
    tail = 10

    for seed in seed_list:
        results = run_toy_simulation(
            scenario=args.scenario,
            days=args.days,
            total_demand=args.demand,
            num_dep_windows=args.num_windows,
            dep_window_size=args.dt,
            target_arrival=args.target_arrival,
            beta_early=args.beta_early,
            beta_late=args.beta_late,
            theta=args.theta,
            br_delta=args.br_delta,
            init_var=args.init_var,
            info_var_base=args.info_var,
            exp_var=args.exp_var,
            share_n=args.share_n,
            penetration=args.penetration,
            shock_day=args.shock_day,
            shock_len=args.shock_len,
            shock_route=shock_route,
            shock_factor=args.shock_factor,
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
            s1_active_only=bool(args.s1_active_only),
            s2_no_fallback=bool(args.s2_no_fallback),
        )
        last_results = results

        tail = min(10, len(results["avg_tt"]))
        avg_tt_tail = float(np.mean(results["avg_tt"][-tail:]))
        rel_gap_tail = results["rel_gap"][-tail:]
        rel_gap_tail = rel_gap_tail[np.isfinite(rel_gap_tail)]
        rel_gap_tail_mean = float(np.mean(rel_gap_tail)) if rel_gap_tail.size > 0 else float("nan")
        avg_tt_tail_list.append(avg_tt_tail)
        rel_gap_tail_mean_list.append(rel_gap_tail_mean)

        m = compute_disruption_metrics(results, shock_day=args.shock_day, shock_len=args.shock_len)
        if m:
            m = dict(m)
            m["avg_tt_last10"] = float(avg_tt_tail)
            m["rel_gap_last10"] = float(rel_gap_tail_mean)
            metrics_list.append(m)

        print(
            f"scenario={args.scenario}  seed={seed}  avg_tt_last{tail}={avg_tt_tail:.6f}  rel_gap_last{tail}={rel_gap_tail_mean:.6f}  "
            f"shock=({args.shock_day},{args.shock_len},{shock_route},{args.shock_factor})"
        )

    if n_seeds > 1 and metrics_list:
        keys = [
            "baseline_tt",
            "shock_tt_peak",
            "shock_rg_peak",
            "recovery_day",
            "avg_tt_last10",
            "rel_gap_last10",
        ]
        print(f"multiseed_summary  n_seeds={n_seeds}  seeds={seed_list[0]}..{seed_list[-1]}")
        for k in keys:
            arr = np.array([m.get(k, float('nan')) for m in metrics_list], dtype=float)
            mask = np.isfinite(arr)
            if not np.any(mask):
                mu = float("nan")
                sd = float("nan")
            else:
                mu = float(np.mean(arr[mask]))
                sd = float(np.std(arr[mask]))
            print(f"{k}  mean={mu:.6f}  std={sd:.6f}")

    if n_seeds == 1 and (last_results is not None) and (outdir is not None) and (not args.no_plot):
        try:
            plot_results(
                last_results,
                title=f"Baseline-1 SRDT Bayes Toy ({args.scenario})",
                outdir=outdir,
                show=not args.no_show,
                shock_day=args.shock_day,
                shock_len=args.shock_len,
            )
        except Exception as e:
            print(f"plot_error: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
