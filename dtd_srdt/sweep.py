from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .metrics import compute_disruption_metrics
from .sioux_sim import run_sioux_simulation


def _parse_grid(values: Sequence[float]) -> np.ndarray:
    arr = np.array(list(values), dtype=float)
    if arr.size == 0:
        raise ValueError("grid must be non-empty")
    return arr


def sweep_sioux_grid(
    G_multi,
    scenario: str,
    od_pairs: List[Tuple[int, int]],
    days: int,
    demand_per_od: int,
    k_routes: int,
    num_dep_windows: int,
    dep_window_size: float,
    target_arrival: float,
    beta_early: float,
    beta_late: float,
    init_var: float,
    info_var_base: float,
    exp_var: float,
    share_n: float,
    shock_day: int,
    shock_len: int,
    shock_edge_id: int,
    shock_factor: float,
    capacity_scale: float,
    seed: int,
    penetrations: Sequence[float],
    thetas: Sequence[float],
    s2_report_var: Optional[float] = None,
    risk_eta: float = 0.0,
) -> pd.DataFrame:
    p_grid = _parse_grid(penetrations)
    th_grid = _parse_grid(thetas)

    rows: List[Dict[str, float]] = []

    for p in p_grid:
        for th in th_grid:
            results = run_sioux_simulation(
                G_multi,
                scenario=scenario,
                days=days,
                od_pairs=od_pairs,
                demand_per_od=demand_per_od,
                k_routes=k_routes,
                num_dep_windows=num_dep_windows,
                dep_window_size=dep_window_size,
                target_arrival=target_arrival,
                beta_early=beta_early,
                beta_late=beta_late,
                theta=float(th),
                init_var=init_var,
                info_var_base=info_var_base,
                exp_var=exp_var,
                share_n=share_n,
                penetration=float(p),
                shock_day=shock_day,
                shock_len=shock_len,
                shock_edge_id=shock_edge_id,
                shock_factor=shock_factor,
                capacity_scale=capacity_scale,
                seed=seed,
                s2_report_var=s2_report_var,
                risk_eta=risk_eta,
            )

            tail = min(10, len(results["avg_tt"]))
            avg_tt_tail = float(np.mean(results["avg_tt"][-tail:]))
            rel_gap_tail = results["rel_gap"][-tail:]
            rel_gap_tail = rel_gap_tail[np.isfinite(rel_gap_tail)]
            rel_gap_tail_mean = float(np.mean(rel_gap_tail)) if rel_gap_tail.size > 0 else float("nan")

            metrics = compute_disruption_metrics(results, shock_day=shock_day, shock_len=shock_len)
            row: Dict[str, float] = {
                "penetration": float(p),
                "theta": float(th),
                "avg_tt_last10": avg_tt_tail,
                "rel_gap_last10": rel_gap_tail_mean,
                "resolved_edge": float(results.get("shock_edge_selected", float("nan"))),
            }
            if metrics:
                row.update(metrics)
            rows.append(row)

    df = pd.DataFrame(rows)
    return df


def _pivot_grid(df: pd.DataFrame, x: str, y: str, v: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.sort(df[x].unique())
    ys = np.sort(df[y].unique())
    grid = np.full((ys.size, xs.size), np.nan, dtype=float)

    x_to_j = {float(val): int(j) for j, val in enumerate(xs)}
    y_to_i = {float(val): int(i) for i, val in enumerate(ys)}

    for _, r in df.iterrows():
        xv = float(r[x])
        yv = float(r[y])
        if xv not in x_to_j or yv not in y_to_i:
            continue
        grid[y_to_i[yv], x_to_j[xv]] = float(r.get(v, np.nan))

    return xs, ys, grid


def plot_heatmap(
    df: pd.DataFrame,
    x: str,
    y: str,
    value: str,
    out_path: str,
    title: Optional[str] = None,
) -> None:
    import matplotlib.pyplot as plt

    xs, ys, grid = _pivot_grid(df, x=x, y=y, v=value)

    fig = plt.figure(figsize=(7.2, 5.6))
    ax = plt.gca()

    im = ax.imshow(grid, origin="lower", aspect="auto")
    ax.set_xticks(np.arange(xs.size))
    ax.set_yticks(np.arange(ys.size))
    ax.set_xticklabels([f"{v:g}" for v in xs])
    ax.set_yticklabels([f"{v:g}" for v in ys])
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    if title is not None:
        ax.set_title(title)

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(value)

    plt.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
