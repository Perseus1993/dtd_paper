import argparse
import os
import sys
from typing import List, Optional, Tuple

import numpy as np

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))
from dtd_srdt.sweep import plot_heatmap, sweep_sioux_grid

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


def _parse_float_list(s: str) -> List[float]:
    s = s.strip()
    if ":" in s:
        a, b, c = s.split(":")
        start = float(a)
        end = float(b)
        step = float(c)
        if step == 0:
            raise ValueError("step must be non-zero")
        n = int(np.floor((end - start) / step)) + 1
        if n <= 0:
            return []
        return [float(start + i * step) for i in range(n)]
    parts = [p for p in s.split(",") if p.strip() != ""]
    return [float(p) for p in parts]


def _resolve_outdir(fig_root: str, outdir: Optional[str]) -> Optional[str]:
    if outdir is None:
        return None
    outdir = str(outdir)
    if os.path.isabs(outdir) or (os.sep in outdir) or (os.altsep is not None and os.altsep in outdir):
        return outdir
    fig_root = str(fig_root) if fig_root is not None else ""
    if fig_root.strip() in ("", "."):
        return outdir
    return os.path.join(fig_root, outdir)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", type=str, default="S1", choices=["S0", "S1", "S2"])
    parser.add_argument("--days", type=int, default=60)
    parser.add_argument("--od", action="append", default=None, help="OD pair like '14-4' (repeatable)")
    parser.add_argument("--demand_per_od", type=int, default=500)
    parser.add_argument("--k_routes", type=int, default=5)
    parser.add_argument("--num_windows", type=int, default=20)
    parser.add_argument("--dt", type=float, default=1.0)
    parser.add_argument("--target_arrival", type=float, default=25.0)
    parser.add_argument("--beta_early", type=float, default=0.8)
    parser.add_argument("--beta_late", type=float, default=1.8)
    parser.add_argument("--init_var", type=float, default=4.0)
    parser.add_argument("--info_var", type=float, default=0.2)
    parser.add_argument("--exp_var", type=float, default=2.0)
    parser.add_argument("--share_n", type=float, default=4.0)
    parser.add_argument("--shock_day", type=int, default=30)
    parser.add_argument("--shock_len", type=int, default=11)
    parser.add_argument("--shock_edge", type=int, default=0)
    parser.add_argument("--shock_factor", type=float, default=0.3)
    parser.add_argument("--capacity_scale", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=7)

    parser.add_argument("--p_grid", type=str, default="0,0.2,0.4,0.6,0.8,1")
    parser.add_argument("--theta_grid", type=str, default="1,2,3,4,5")
    parser.add_argument("--outdir", type=str, default="figs_sf_heatmap")
    parser.add_argument(
        "--fig_root",
        type=str,
        default=os.path.join("outputs", "sioux"),
        help="If outdir is a plain folder name, it will be placed under this root (set '' or '.' to disable).",
    )
    args = parser.parse_args()

    outdir = _resolve_outdir(args.fig_root, args.outdir)
    if outdir is None:
        raise ValueError("outdir must not be None")
    os.makedirs(outdir, exist_ok=True)

    od_pairs = _parse_od_list(args.od)
    p_vals = _parse_float_list(args.p_grid)
    th_vals = _parse_float_list(args.theta_grid)

    G_multi, _, _, _ = load_graph(draw=False)

    df = sweep_sioux_grid(
        G_multi,
        scenario=args.scenario,
        od_pairs=od_pairs,
        days=args.days,
        demand_per_od=args.demand_per_od,
        k_routes=args.k_routes,
        num_dep_windows=args.num_windows,
        dep_window_size=args.dt,
        target_arrival=args.target_arrival,
        beta_early=args.beta_early,
        beta_late=args.beta_late,
        init_var=args.init_var,
        info_var_base=args.info_var,
        exp_var=args.exp_var,
        share_n=args.share_n,
        shock_day=args.shock_day,
        shock_len=args.shock_len,
        shock_edge_id=args.shock_edge,
        shock_factor=args.shock_factor,
        capacity_scale=args.capacity_scale,
        seed=args.seed,
        penetrations=p_vals,
        thetas=th_vals,
    )

    csv_path = os.path.join(outdir, "sf_sweep.csv")
    df.to_csv(csv_path, index=False)

    plot_heatmap(
        df,
        x="penetration",
        y="theta",
        value="shock_tt_peak",
        out_path=os.path.join(outdir, "heatmap_peak_tt.png"),
        title=f"Sioux Falls peak TT (scenario={args.scenario})",
    )
    plot_heatmap(
        df,
        x="penetration",
        y="theta",
        value="shock_rg_peak",
        out_path=os.path.join(outdir, "heatmap_peak_rg.png"),
        title=f"Sioux Falls peak RG (scenario={args.scenario})",
    )
    plot_heatmap(
        df,
        x="penetration",
        y="theta",
        value="recovery_day",
        out_path=os.path.join(outdir, "heatmap_recovery_day.png"),
        title=f"Sioux Falls recovery day (scenario={args.scenario})",
    )

    print(f"saved_csv={csv_path}")


if __name__ == "__main__":
    main()
