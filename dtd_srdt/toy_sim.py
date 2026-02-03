import random
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .common import BayesianBelief, bayes_update, bpr_time, relative_gap, schedule_penalties, softmax_choice


@dataclass
class Choice:
    route: str
    dep_idx: int


def _exp_smooth(mu: float, obs: float, eta: float) -> float:
    return (1.0 - eta) * mu + eta * obs


def run_toy_simulation(
    scenario: str,
    days: int,
    total_demand: int,
    num_dep_windows: int,
    dep_window_size: float,
    target_arrival: float,
    beta_early: float,
    beta_late: float,
    theta: float,
    init_var: float,
    info_var_base: float,
    exp_var: float,
    share_n: float,
    penetration: float,
    shock_day: int,
    shock_len: int,
    shock_route: Optional[str],
    shock_factor: float,
    seed: int,
    route_only: bool = False,
    no_sharing: bool = False,
    use_pretrip: bool = True,
    use_posttrip: bool = True,
    update_rule: str = "bayes",
    eta_info: float = 0.3,
    eta_exp: float = 0.3,
    s2_report_var: Optional[float] = None,
    risk_eta: float = 0.0,
    s1_lag_days: int = 1,
    s1_meas_var: float = 0.0,
    s2_info_var_mode: str = "share_power",
    br_delta: float = 0.0,
    s1_active_only: bool = False,
    s2_no_fallback: bool = False,
) -> Dict[str, np.ndarray]:
    paths = {
        "P1": {"free_flow_time": 3.5, "capacity": 90},
        "P2": {"free_flow_time": 3.0, "capacity": 100},
        "P3": {"free_flow_time": 2.0, "capacity": 100},
        "P4": {"free_flow_time": 3.5, "capacity": 200},
    }

    if scenario not in {"S0", "S1", "S2"}:
        raise ValueError("scenario must be one of S0/S1/S2")

    if penetration < 0.0 or penetration > 1.0:
        raise ValueError("penetration must be in [0,1]")

    if update_rule not in {"bayes", "exp_smooth"}:
        raise ValueError("update_rule must be 'bayes' or 'exp_smooth'")

    if update_rule == "exp_smooth":
        if not (0.0 <= eta_info <= 1.0):
            raise ValueError("eta_info must be in [0,1]")
        if not (0.0 <= eta_exp <= 1.0):
            raise ValueError("eta_exp must be in [0,1]")

    if s2_report_var is not None and float(s2_report_var) < 0.0:
        raise ValueError("s2_report_var must be non-negative or None")

    if not np.isfinite(float(risk_eta)):
        raise ValueError("risk_eta must be finite")

    if float(br_delta) < 0.0 or not np.isfinite(float(br_delta)):
        raise ValueError("br_delta must be finite and non-negative")

    if int(s1_lag_days) < 1:
        raise ValueError("s1_lag_days must be >= 1")

    if float(s1_meas_var) < 0.0 or not np.isfinite(float(s1_meas_var)):
        raise ValueError("s1_meas_var must be finite and non-negative")

    if s2_info_var_mode not in {"share_power", "report_mean_var"}:
        raise ValueError("s2_info_var_mode must be 'share_power' or 'report_mean_var'")

    report_var = float(exp_var) if s2_report_var is None else float(s2_report_var)

    rng = np.random.default_rng(seed)
    random.seed(seed)

    rng_report = np.random.default_rng(seed + 2000003)

    rng_s1 = np.random.default_rng(seed + 3000003)

    rng_mask = np.random.default_rng(seed + 1000003)
    if scenario == "S0" or penetration <= 0.0:
        has_info = np.zeros(total_demand, dtype=bool)
    elif penetration >= 1.0:
        has_info = np.ones(total_demand, dtype=bool)
    else:
        has_info = rng_mask.random(total_demand) < float(penetration)

    routes = list(paths.keys())
    dep_windows = list(range(num_dep_windows))
    pairs: List[Choice] = [Choice(r, d) for d in dep_windows for r in routes]

    pair_to_idx: Dict[Tuple[str, int], int] = {(c.route, int(c.dep_idx)): int(j) for j, c in enumerate(pairs)}

    dep_to_pair_idx: List[List[int]] = [[] for _ in range(num_dep_windows)]
    for j, c in enumerate(pairs):
        dep_to_pair_idx[int(c.dep_idx)].append(int(j))

    fixed_dep_idx = None
    if route_only:
        mean_ff = float(np.mean([float(paths[r]["free_flow_time"]) for r in routes]))
        dep0 = int(round((float(target_arrival) - mean_ff) / float(dep_window_size)))
        dep0 = int(np.clip(dep0, 0, num_dep_windows - 1))
        fixed_dep_idx = np.full(total_demand, dep0, dtype=int)

    beliefs: List[List[BayesianBelief]] = []
    for _ in range(total_demand):
        agent_beliefs = []
        for c in pairs:
            mu0 = float(paths[c.route]["free_flow_time"])
            agent_beliefs.append(BayesianBelief(mu=mu0, var=init_var))
        beliefs.append(agent_beliefs)

    last_day_tt = np.array([paths[c.route]["free_flow_time"] for c in pairs], dtype=float)
    last_day_I = last_day_tt.copy()
    last_day_flows = np.zeros(len(pairs), dtype=float)

    tt_hist: List[np.ndarray] = [last_day_tt.copy()]

    avg_tt_per_day: List[float] = []
    route_switch_per_day: List[int] = []
    dep_switch_per_day: List[int] = []
    rel_gap_per_day: List[float] = []
    dep_counts_per_day: List[np.ndarray] = []

    # prev_choices is None for day 1 (no previous day to compare)
    prev_choices: Optional[List[Choice]] = None

    prev_flow_vec = None
    flow_vec_per_day: List[np.ndarray] = []

    for day in range(days):
        day_no = day + 1
        flow_vec = np.zeros(len(pairs), dtype=float)
        chosen_pairs: List[Choice] = []
        chosen_pair_ids: List[int] = []

        lag = min(int(s1_lag_days), len(tt_hist))
        eval_signal = tt_hist[-lag]
        signal_mean = eval_signal if scenario != "S2" else last_day_I
        if scenario == "S1" and float(s1_meas_var) > 0.0:
            signal_mean = signal_mean + rng_s1.normal(0.0, float(np.sqrt(float(s1_meas_var))), size=len(pairs))
            signal_mean = np.maximum(signal_mean, 1e-6)

        if scenario == "S0":
            info_vars = np.full(len(pairs), float("inf"), dtype=float)
        elif scenario == "S1" or (scenario == "S2" and no_sharing):
            if scenario == "S1" and bool(s1_active_only):
                info_vars = np.full(len(pairs), float("inf"), dtype=float)
                m = last_day_flows > 0
                if np.any(m):
                    info_vars[m] = float(info_var_base)
            else:
                info_vars = np.full(len(pairs), info_var_base, dtype=float)

            if scenario == "S2" and bool(s2_no_fallback):
                m0 = last_day_flows <= 0
                if np.any(m0):
                    info_vars[m0] = float("inf")
        else:
            if s2_info_var_mode == "share_power":
                share = np.zeros(len(pairs), dtype=float)
                if np.sum(last_day_flows) > 0:
                    share = last_day_flows / float(np.sum(last_day_flows))
                share = np.clip(share, 1e-6, 1.0)
                info_vars = info_var_base / (share**share_n)
            else:
                cnt = last_day_flows
                info_vars = np.full(len(pairs), float("inf"), dtype=float)
                m_cnt = cnt > 0
                if np.any(m_cnt):
                    info_vars[m_cnt] = float(info_var_base) + float(report_var) / cnt[m_cnt]

            if bool(s2_no_fallback):
                m0 = last_day_flows <= 0
                if np.any(m0):
                    info_vars[m0] = float("inf")

        # Store pre-trip updated beliefs for later use (avoid redundant computation)
        pretrip_beliefs: List[List[Tuple[float, float]]] = []

        for i in range(total_demand):
            use_info = scenario != "S0" and bool(has_info[i]) and bool(use_pretrip)

            allowed = dep_to_pair_idx[int(fixed_dep_idx[i])] if fixed_dep_idx is not None else None
            if allowed is None:
                cand_idx = list(range(len(pairs)))
            else:
                cand_idx = allowed

            costs = np.zeros(len(cand_idx), dtype=float)
            agent_pretrip: List[Tuple[float, float]] = []
            for jj, j in enumerate(cand_idx):
                c = pairs[j]
                b = beliefs[i][j]
                mu1, var1 = b.mu, b.var
                obs_var = float(info_vars[j])
                if use_info and np.isfinite(obs_var) and obs_var > 0.0:
                    if update_rule == "bayes":
                        mu2, var2 = bayes_update(mu1, var1, float(signal_mean[j]), obs_var)
                    else:
                        mu2, var2 = _exp_smooth(mu1, float(signal_mean[j]), float(eta_info)), var1
                else:
                    mu2, var2 = mu1, var1
                agent_pretrip.append((mu2, var2))
                dep_time = c.dep_idx * dep_window_size
                arrival = dep_time + mu2
                early, late = schedule_penalties(arrival, target_arrival)
                costs[jj] = mu2 + beta_early * early + beta_late * late + float(risk_eta) * float(var2)

            pretrip_beliefs.append(agent_pretrip)

            idx = None
            if prev_choices is not None and float(br_delta) > 0.0:
                prev = prev_choices[i]
                prev_idx = pair_to_idx.get((str(prev.route), int(prev.dep_idx)))
                if prev_idx is not None and prev_idx in cand_idx:
                    jj_prev = int(cand_idx.index(int(prev_idx)))
                    min_cost = float(np.min(costs))
                    prev_cost = float(costs[jj_prev])
                    if np.isfinite(min_cost) and np.isfinite(prev_cost) and (prev_cost - min_cost) <= float(br_delta):
                        idx = int(prev_idx)

            if idx is None:
                jj_star = softmax_choice(costs, theta=theta, rng=rng)
                idx = int(cand_idx[jj_star])
            chosen = pairs[idx]
            chosen_pairs.append(chosen)
            chosen_pair_ids.append(idx)
            flow_vec[idx] += 1.0

        in_shock = (
            shock_route is not None
            and shock_len > 0
            and shock_day > 0
            and shock_day <= day_no <= (shock_day + shock_len - 1)
        )

        # FIX: Aggregate flow by route (across all departure windows) for BPR calculation
        # This ensures travel time is departure-window invariant as stated in the paper
        route_flow: Dict[str, float] = {r: 0.0 for r in routes}
        for j, c in enumerate(pairs):
            route_flow[c.route] += flow_vec[j]

        # Compute travel time using aggregated route flow
        tt_vec = np.zeros(len(pairs), dtype=float)
        route_tt: Dict[str, float] = {}
        for r in routes:
            info = paths[r]
            cap = float(info["capacity"])
            if in_shock and r == shock_route:
                cap = cap * max(float(shock_factor), 1e-6)
            route_tt[r] = bpr_time(info["free_flow_time"], route_flow[r], cap)

        # All alternatives on the same route have the same travel time
        for j, c in enumerate(pairs):
            tt_vec[j] = route_tt[c.route]

        tt_hist.append(tt_vec.copy())

        day_tt: List[float] = []
        shared_sum = np.zeros(len(pairs), dtype=float)
        shared_cnt = np.zeros(len(pairs), dtype=float)
        for i in range(total_demand):
            idx = chosen_pair_ids[i]
            agent_pretrip = pretrip_beliefs[i]

            # Determine which candidate index corresponds to idx
            allowed = dep_to_pair_idx[int(fixed_dep_idx[i])] if fixed_dep_idx is not None else None
            if allowed is None:
                cand_idx = list(range(len(pairs)))
            else:
                cand_idx = allowed

            # Use pre-computed pre-trip beliefs (avoid redundant computation)
            jj_chosen = cand_idx.index(idx)
            mu2, var2 = agent_pretrip[jj_chosen]

            t_exp = float(tt_vec[idx])
            if scenario == "S2" and report_var > 0.0:
                t_rep = float(t_exp + rng_report.normal(0.0, float(np.sqrt(report_var))))
                t_rep = max(t_rep, 1e-6)
            else:
                t_rep = t_exp
            shared_sum[idx] += t_rep
            shared_cnt[idx] += 1.0
            if use_posttrip:
                if update_rule == "bayes":
                    mu3, var3 = bayes_update(mu2, var2, t_exp, exp_var)
                else:
                    mu3, var3 = _exp_smooth(mu2, t_exp, float(eta_exp)), var2
            else:
                mu3, var3 = mu2, var2
            beliefs[i][idx] = BayesianBelief(mu=mu3, var=var3)

            # Update beliefs for non-chosen alternatives using pre-computed pre-trip values
            for jj, j in enumerate(cand_idx):
                if j == idx:
                    continue
                mu2j, var2j = agent_pretrip[jj]
                beliefs[i][j] = BayesianBelief(mu=mu2j, var=var2j)

            day_tt.append(t_exp)

        avg_tt_per_day.append(float(np.mean(day_tt)))

        # FIX: For day 1, there's no previous day to compare, so switching = 0
        if prev_choices is None:
            r_switch = 0
            d_switch = 0
        else:
            r_switch = 0
            d_switch = 0
            for i in range(total_demand):
                if chosen_pairs[i].route != prev_choices[i].route:
                    r_switch += 1
                if chosen_pairs[i].dep_idx != prev_choices[i].dep_idx:
                    d_switch += 1
        route_switch_per_day.append(r_switch)
        dep_switch_per_day.append(d_switch)

        dep_counts = np.zeros(num_dep_windows, dtype=float)
        for c in chosen_pairs:
            dep_counts[c.dep_idx] += 1.0
        dep_counts_per_day.append(dep_counts)

        flow_vec_per_day.append(flow_vec.copy())

        if prev_flow_vec is None:
            rel_gap_per_day.append(float("nan"))
        else:
            rel_gap_per_day.append(relative_gap(prev_flow_vec, flow_vec))
        prev_flow_vec = flow_vec.copy()

        last_day_tt = tt_vec
        if scenario == "S2":
            last_day_I = tt_vec.copy()
            m = shared_cnt > 0
            if np.any(m):
                last_day_I[m] = shared_sum[m] / shared_cnt[m]
        else:
            last_day_I = last_day_tt
        last_day_flows = flow_vec
        prev_choices = chosen_pairs

    return {
        "avg_tt": np.array(avg_tt_per_day),
        "route_switch": np.array(route_switch_per_day),
        "dep_switch": np.array(dep_switch_per_day),
        "rel_gap": np.array(rel_gap_per_day),
        "dep_counts": np.stack(dep_counts_per_day, axis=0),
        "daily_flow_global": np.stack(flow_vec_per_day, axis=0),
        "od_offsets": np.array([0], dtype=int),
        "od_sizes": np.array([len(pairs)], dtype=int),
    }
