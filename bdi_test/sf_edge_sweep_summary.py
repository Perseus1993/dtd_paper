import argparse
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"empty csv: {path}")
    return df


def _compute_delta_peak(df: pd.DataFrame) -> pd.DataFrame:
    base = float(df["baseline_tt_mean"].iloc[0])
    out = df.copy()
    out["delta_peak_tt"] = out["shock_tt_peak_mean"].astype(float) - base
    return out


def _save_cdf_plot(deltas: np.ndarray, out_path: str, title: str, threshold: float) -> None:
    x = np.array(deltas, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return

    xs = np.sort(x)
    ys = (np.arange(xs.size) + 1.0) / float(xs.size)

    fig = plt.figure(figsize=(6.8, 4.6))
    ax = fig.add_subplot(1, 1, 1)
    ax.step(xs, ys, where="post", color="#4C78A8")
    ax.axvline(float(threshold), color="#E45756", linestyle="--", linewidth=1.2)
    ax.set_xlabel(r"$\Delta$ peak TT (peakTT - baselineTT)")
    ax.set_ylabel("Empirical CDF")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=250)
    plt.close(fig)


def _top_disruptive(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = df[df["delta_peak_tt"] > float(threshold)].copy()
    out = out.sort_values("shock_tt_peak_mean", ascending=False)
    return out


def _maybe_write_top_tex(df_top: pd.DataFrame, out_path: str) -> None:
    cols = [
        "shock_edge",
        "u",
        "v",
        "base_volume",
        "base_cost",
        "shock_tt_peak_mean",
        "shock_rg_peak_mean",
        "recovery_day_mean",
        "post_shock_tt_var_mean",
    ]
    for c in cols:
        if c not in df_top.columns:
            return

    with open(out_path, "w", encoding="utf-8") as f:
        for _, r in df_top[cols].iterrows():
            eid = int(r["shock_edge"])
            u = int(r["u"])
            v = int(r["v"])
            vol = float(r["base_volume"])
            cost = float(r["base_cost"])
            peak_tt = float(r["shock_tt_peak_mean"])
            peak_rg = float(r["shock_rg_peak_mean"])
            rec = float(r["recovery_day_mean"])
            post_var = float(r["post_shock_tt_var_mean"])
            f.write(
                f" {eid} ({u}\\to{v}) & {vol:.1f} & {cost:.3f} & {peak_tt:.4f} & {peak_rg:.4f} & {rec:.1f} & {post_var:.6f} \\\\\n"
            )


def summarize_one(csv_path: str, outdir: str, tag: str, threshold: float) -> pd.DataFrame:
    df0 = _load_csv(csv_path)
    df = _compute_delta_peak(df0)

    deltas = df["delta_peak_tt"].to_numpy(dtype=float)
    cdf_path = os.path.join(outdir, f"sf_edge_sweep_delta_peak_cdf_{tag}.png")
    _save_cdf_plot(deltas, cdf_path, title=f"Sioux Falls edge sweep: {tag}", threshold=threshold)

    df_top = _top_disruptive(df, threshold=threshold)
    top_csv = os.path.join(outdir, f"sf_edge_sweep_top_disruptive_{tag}.csv")
    df_top.to_csv(top_csv, index=False)

    top_tex = os.path.join(outdir, f"sf_edge_sweep_top_disruptive_{tag}.tex")
    _maybe_write_top_tex(df_top, top_tex)

    base = float(df0["baseline_tt_mean"].iloc[0])
    n_edges = int(df0.shape[0])
    n_nontrivial = int(df_top.shape[0])

    print(f"[{tag}] csv={csv_path}")
    print(f"[{tag}] baseline_tt={base:.6f}")
    print(f"[{tag}] n_edges={n_edges}  n_nontrivial(delta_peak_tt>{threshold})={n_nontrivial}")
    print(f"[{tag}] saved_cdf={cdf_path}")
    print(f"[{tag}] saved_top_csv={top_csv}")
    print(f"[{tag}] saved_top_tex={top_tex}")

    return df_top


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--csv_all",
        type=str,
        default=os.path.join("outputs", "sioux", "figs_sf_edge_sweep", "sf_edge_S1_SWEEP_all_seed7_n1.csv"),
    )
    parser.add_argument(
        "--csv_candidate",
        type=str,
        default=os.path.join("outputs", "sioux", "figs_sf_edge_sweep", "sf_edge_S1_SWEEP_candidate_seed7_n1.csv"),
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=os.path.join("outputs", "sioux", "figs_sf_edge_sweep_summary"),
    )
    parser.add_argument("--threshold", type=float, default=0.1)
    args = parser.parse_args()

    outdir = str(args.outdir)
    _ensure_dir(outdir)

    csv_all: Optional[str] = str(args.csv_all) if args.csv_all else None
    csv_candidate: Optional[str] = str(args.csv_candidate) if args.csv_candidate else None

    if csv_all:
        summarize_one(csv_all, outdir=outdir, tag="all", threshold=float(args.threshold))

    if csv_candidate:
        summarize_one(csv_candidate, outdir=outdir, tag="candidate", threshold=float(args.threshold))


if __name__ == "__main__":
    main()
