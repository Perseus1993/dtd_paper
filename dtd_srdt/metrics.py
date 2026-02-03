from typing import Dict

import numpy as np

from .common import safe_mean


def compute_disruption_metrics(
    results: Dict[str, np.ndarray],
    shock_day: int,
    shock_len: int,
    baseline_days: int = 10,
    post_var_days: int = 20,
    recovery_tol: float = 0.01,
) -> Dict[str, float]:
    if shock_day <= 0 or shock_len <= 0:
        return {}

    avg_tt = results["avg_tt"]
    rel_gap_arr = results["rel_gap"]
    dep_counts = results["dep_counts"]

    days = len(avg_tt)
    shock_start = shock_day
    shock_end = min(days, shock_day + shock_len - 1)

    base_end = shock_start - 1
    base_start = max(1, base_end - baseline_days + 1)
    base_idx = np.arange(base_start - 1, base_end)
    shock_idx = np.arange(shock_start - 1, shock_end)
    post_start = shock_end + 1
    post_end = min(days, post_start + baseline_days - 1)
    post_idx = np.arange(post_start - 1, post_end)

    post_var_day = max(0, int(post_var_days))
    post_var_end = min(days, post_start + post_var_day - 1)
    post_var_idx = np.arange(post_start - 1, post_var_end) if (post_start <= days and post_var_day > 0) else np.array([], dtype=int)

    baseline_tt = safe_mean(avg_tt[base_idx])
    shock_tt_peak = float(np.max(avg_tt[shock_idx])) if shock_idx.size > 0 else float("nan")
    shock_rg_peak = float(np.nanmax(rel_gap_arr[shock_idx])) if shock_idx.size > 0 else float("nan")

    recovery_day = float("nan")
    if np.isfinite(baseline_tt) and post_start <= days:
        thresh = baseline_tt * (1.0 + recovery_tol)
        for d in range(post_start, days + 1):
            if avg_tt[d - 1] <= thresh:
                recovery_day = float(d)
                break

    dep_bins = np.arange(dep_counts.shape[1], dtype=float)
    dep_mass = np.sum(dep_counts, axis=1)
    dep_mass = np.where(dep_mass <= 0, 1.0, dep_mass)
    dep_mean = np.sum(dep_counts * dep_bins.reshape(1, -1), axis=1) / dep_mass

    dep_mean_base = safe_mean(dep_mean[base_idx])
    dep_mean_shock = safe_mean(dep_mean[shock_idx])
    dep_mean_post = safe_mean(dep_mean[post_idx]) if post_idx.size > 0 else float("nan")

    hhi_base = float("nan")
    hhi_shock = float("nan")
    hhi_post = float("nan")
    entropy_base = float("nan")
    entropy_shock = float("nan")
    entropy_post = float("nan")

    daily_flow_global = results.get("daily_flow_global", None)
    od_offsets = results.get("od_offsets", None)
    od_sizes = results.get("od_sizes", None)
    if (
        isinstance(daily_flow_global, np.ndarray)
        and daily_flow_global.ndim == 2
        and isinstance(od_offsets, np.ndarray)
        and isinstance(od_sizes, np.ndarray)
        and od_offsets.size == od_sizes.size
        and daily_flow_global.shape[0] == days
    ):
        hhi_day = np.full(days, float("nan"), dtype=float)
        ent_day = np.full(days, float("nan"), dtype=float)

        for d in range(days):
            tot_w = 0.0
            hhi_w = 0.0
            ent_w = 0.0
            for off, sz in zip(od_offsets.tolist(), od_sizes.tolist()):
                off_i = int(off)
                sz_i = int(sz)
                if sz_i <= 0:
                    continue
                f = daily_flow_global[d, off_i : off_i + sz_i]
                tot = float(np.sum(f))
                if not np.isfinite(tot) or tot <= 0.0:
                    continue

                p = f / tot
                p = np.where(p > 0.0, p, 0.0)

                hhi = float(np.sum(p * p))

                if sz_i <= 1:
                    ent = 0.0
                else:
                    m = p > 0.0
                    ent_raw = -float(np.sum(p[m] * np.log(p[m]))) if np.any(m) else 0.0
                    ent = ent_raw / float(np.log(float(sz_i)))

                hhi_w += hhi * tot
                ent_w += ent * tot
                tot_w += tot

            if tot_w > 0.0:
                hhi_day[d] = hhi_w / tot_w
                ent_day[d] = ent_w / tot_w

        hhi_base = safe_mean(hhi_day[base_idx])
        hhi_shock = safe_mean(hhi_day[shock_idx])
        hhi_post = safe_mean(hhi_day[post_idx]) if post_idx.size > 0 else float("nan")
        entropy_base = safe_mean(ent_day[base_idx])
        entropy_shock = safe_mean(ent_day[shock_idx])
        entropy_post = safe_mean(ent_day[post_idx]) if post_idx.size > 0 else float("nan")

    post_shock_tt_var = float("nan")
    if post_var_idx.size > 0:
        x = avg_tt[post_var_idx]
        x = x[np.isfinite(x)]
        if x.size > 0:
            post_shock_tt_var = float(np.var(x))

    return {
        "baseline_tt": baseline_tt,
        "shock_tt_peak": shock_tt_peak,
        "shock_rg_peak": shock_rg_peak,
        "recovery_day": recovery_day,
        "post_shock_tt_var": post_shock_tt_var,
        "dep_mean_base": dep_mean_base,
        "dep_mean_shock": dep_mean_shock,
        "dep_mean_post": dep_mean_post,
        "hhi_base": hhi_base,
        "hhi_shock": hhi_shock,
        "hhi_post": hhi_post,
        "entropy_base": entropy_base,
        "entropy_shock": entropy_shock,
        "entropy_post": entropy_post,
    }
