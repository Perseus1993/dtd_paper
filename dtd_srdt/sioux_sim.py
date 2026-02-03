import random
from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from .common import BayesianBelief, bayes_update, bpr_time, relative_gap, schedule_penalties, softmax_choice


def _exp_smooth(mu: float, obs: float, eta: float) -> float:
    return (1.0 - eta) * mu + eta * obs


def _normalize_scenario(scenario: str) -> str:
    s = str(scenario).strip().upper()
    if s in {"S0"}:
        return "S0"
    if s in {"S1", "G"}:
        return "G"
    if s in {"N"}:
        return "N"
    if s in {"GN", "G+N", "G_N"}:
        return "GN"
    if s in {"S2", "R"}:
        return "R"
    raise ValueError("scenario must be one of S0/G/N/GN/R (S1 as alias of G; S2 as alias of R)")


def _ring_neighbors(n: int, k: int) -> List[List[int]]:
    if n <= 0:
        return []
    k = int(k)
    if k < 0:
        raise ValueError("social_k must be non-negative")
    if k == 0:
        return [[] for _ in range(n)]
    if k >= n:
        k = n - 1
    if k % 2 != 0:
        k = k - 1
    half = k // 2
    neigh = [set() for _ in range(n)]
    for i in range(n):
        for d in range(1, half + 1):
            neigh[i].add((i + d) % n)
            neigh[i].add((i - d) % n)
    return [sorted(list(s)) for s in neigh]


def _er_neighbors(n: int, k: int, rng: np.random.Generator) -> List[List[int]]:
    if n <= 0:
        return []
    k = int(k)
    if k < 0:
        raise ValueError("social_k must be non-negative")
    if n == 1 or k == 0:
        return [[] for _ in range(n)]
    p = min(1.0, float(k) / float(n - 1))
    neigh = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if float(rng.random()) < p:
                neigh[i].add(j)
                neigh[j].add(i)
    return [sorted(list(s)) for s in neigh]


def build_simple_graph(G_multi: nx.MultiDiGraph) -> nx.DiGraph:
    G = nx.DiGraph()
    for u, v, k, data in G_multi.edges(keys=True, data=True):
        t0 = float(data.get("free_flow_time", data.get("length", 1.0)))
        cap = float(data.get("capacity", 1.0))
        edge_id = int(data.get("edge", -1))
        if G.has_edge(u, v):
            if t0 < float(G[u][v].get("free_flow_time", float("inf"))):
                G[u][v].update({"free_flow_time": t0, "capacity": cap, "edge": edge_id, "key": k})
        else:
            G.add_edge(u, v, free_flow_time=t0, capacity=cap, edge=edge_id, key=k)
    return G


def k_shortest_paths(G: nx.DiGraph, o: int, d: int, k: int) -> List[List[int]]:
    gen = nx.shortest_simple_paths(G, o, d, weight="free_flow_time")
    paths: List[List[int]] = []
    for _ in range(k):
        try:
            paths.append(next(gen))
        except StopIteration:
            break
    return paths


def path_to_links(G: nx.DiGraph, nodes: Sequence[int]) -> List[Tuple[int, int]]:
    links: List[Tuple[int, int]] = []
    for a, b in zip(nodes[:-1], nodes[1:]):
        if not G.has_edge(a, b):
            raise ValueError(f"edge ({a},{b}) not in graph")
        links.append((a, b))
    return links


def route_freeflow_time(G: nx.DiGraph, links: Sequence[Tuple[int, int]]) -> float:
    return float(sum(float(G[u][v]["free_flow_time"]) for u, v in links))


def run_sioux_simulation(
    G_multi: nx.MultiDiGraph,
    scenario: str,
    days: int,
    od_pairs: List[Tuple[int, int]],
    demand_per_od: int,
    k_routes: int,
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
    shock_edge_id: int,
    shock_factor: float,
    capacity_scale: float,
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
    social_topology: str = "ring",
    social_k: int = 6,
    social_var: float = 1.0,
    shock_k_hops: int = 0,
    track_link_data: bool = False,
    trust_heterogeneity: str = "none",
    trust_high_info_var: float = 0.5,
    trust_low_info_var: float = 4.0,
    trust_high_frac: float = 0.5,
) -> Dict[str, np.ndarray]:
    scenario_mode = _normalize_scenario(scenario)

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

    if int(s1_lag_days) < 0:
        raise ValueError("s1_lag_days must be >= 0")

    if float(s1_meas_var) < 0.0 or not np.isfinite(float(s1_meas_var)):
        raise ValueError("s1_meas_var must be finite and non-negative")

    if float(br_delta) < 0.0 or not np.isfinite(float(br_delta)):
        raise ValueError("br_delta must be finite and non-negative")

    if s2_info_var_mode not in {"share_power", "report_mean_var"}:
        raise ValueError("s2_info_var_mode must be 'share_power' or 'report_mean_var'")

    if float(social_var) < 0.0 or not np.isfinite(float(social_var)):
        raise ValueError("social_var must be finite and non-negative")

    if trust_heterogeneity not in {"none", "mixed"}:
        raise ValueError("trust_heterogeneity must be 'none' or 'mixed'")

    report_var = float(exp_var) if s2_report_var is None else float(s2_report_var)

    if capacity_scale <= 0:
        raise ValueError("capacity_scale must be positive")

    rng = np.random.default_rng(seed)
    random.seed(seed)

    rng_report = np.random.default_rng(seed + 2000003)

    rng_s1 = np.random.default_rng(seed + 3000003)

    rng_social = np.random.default_rng(seed + 4000003)

    G = build_simple_graph(G_multi)

    od_routes: List[List[List[Tuple[int, int]]]] = []
    od_route_ff: List[np.ndarray] = []
    candidate_links: set[Tuple[int, int]] = set()

    for o, d in od_pairs:
        node_paths = k_shortest_paths(G, o, d, k_routes)
        if len(node_paths) == 0:
            raise ValueError(f"no path for OD ({o},{d})")
        routes_links = [path_to_links(G, p) for p in node_paths]
        for links in routes_links:
            candidate_links.update(links)
        od_routes.append(routes_links)
        od_route_ff.append(np.array([route_freeflow_time(G, links) for links in routes_links], dtype=float))

    od_fixed_dep: List[int] = []
    if route_only:
        for od_i in range(len(od_pairs)):
            mean_ff = float(np.mean(od_route_ff[od_i]))
            dep0 = int(round((float(target_arrival) - mean_ff) / float(dep_window_size)))
            dep0 = int(np.clip(dep0, 0, num_dep_windows - 1))
            od_fixed_dep.append(dep0)

    od_alt_maps: List[List[Tuple[int, int]]] = []
    od_dep_to_alt: List[List[List[int]]] = []
    for od_i in range(len(od_pairs)):
        alts: List[Tuple[int, int]] = []
        for dep_idx in range(num_dep_windows):
            for r_idx in range(len(od_routes[od_i])):
                alts.append((r_idx, dep_idx))
        od_alt_maps.append(alts)

        dep_to_alt = [[] for _ in range(num_dep_windows)]
        for j, (r_idx, dep_idx) in enumerate(alts):
            dep_to_alt[int(dep_idx)].append(int(j))
        od_dep_to_alt.append(dep_to_alt)

    agents_od: List[int] = []
    agents_beliefs: List[List[BayesianBelief]] = []
    agents_prev_choice: List[Tuple[int, int, int]] = []
    agents_has_info: List[bool] = []
    agents_fixed_dep: List[int] = []
    agents_info_var: List[float] = []  # per-agent info variance for trust heterogeneity

    rng_mask = np.random.default_rng(seed + 1000003)
    rng_trust = np.random.default_rng(seed + 5000003)

    for od_i in range(len(od_pairs)):
        n_routes = len(od_routes[od_i])
        alts = od_alt_maps[od_i]
        for _ in range(demand_per_od):
            agents_od.append(od_i)

            if route_only:
                agents_fixed_dep.append(int(od_fixed_dep[od_i]))
            else:
                agents_fixed_dep.append(-1)

            if scenario_mode == "S0" or penetration <= 0.0:
                agents_has_info.append(False)
            elif penetration >= 1.0:
                agents_has_info.append(True)
            else:
                agents_has_info.append(bool(rng_mask.random() < float(penetration)))

            b: List[BayesianBelief] = []
            for (r_idx, dep_idx) in alts:
                mu0 = float(od_route_ff[od_i][r_idx])
                b.append(BayesianBelief(mu=mu0, var=init_var))
            agents_beliefs.append(b)
            r0 = int(rng.integers(0, n_routes))
            t0 = int(rng.integers(0, num_dep_windows))
            agents_prev_choice.append((od_i, r0, t0))

            # Assign trust type (info variance) for heterogeneity
            if trust_heterogeneity == "mixed":
                if rng_trust.random() < float(trust_high_frac):
                    agents_info_var.append(float(trust_high_info_var))
                else:
                    agents_info_var.append(float(trust_low_info_var))
            else:
                agents_info_var.append(float(info_var_base))

    total_agents = len(agents_od)

    social_neighbors: Optional[List[List[int]]] = None
    if scenario_mode in {"N", "GN"}:
        topo = str(social_topology).strip().lower()
        social_neighbors = [[] for _ in range(total_agents)]
        for od_i in range(len(od_pairs)):
            start = int(od_i) * int(demand_per_od)
            n = int(demand_per_od)
            if topo in {"ring", "small_world", "small-world"}:
                neigh_local = _ring_neighbors(n, int(social_k))
            elif topo in {"er", "erdos", "erdos_renyi", "erdos-renyi"}:
                neigh_local = _er_neighbors(n, int(social_k), rng_social)
            else:
                raise ValueError("social_topology must be 'ring' or 'er'")
            for local_i in range(n):
                i_global = start + int(local_i)
                social_neighbors[i_global] = [start + int(j) for j in neigh_local[local_i]]

    last_day_tt_od: List[np.ndarray] = []
    last_day_I_od: List[np.ndarray] = []
    last_day_flow_od: List[np.ndarray] = []
    for od_i in range(len(od_pairs)):
        alt_count = len(od_alt_maps[od_i])
        tmp = np.zeros(alt_count, dtype=float)
        for j, (r_idx, dep_idx) in enumerate(od_alt_maps[od_i]):
            tmp[j] = float(od_route_ff[od_i][r_idx])
        last_day_tt_od.append(tmp)
        last_day_I_od.append(tmp.copy())
        last_day_flow_od.append(np.zeros(alt_count, dtype=float))

    tt_hist_od: List[List[np.ndarray]] = [[last_day_tt_od[od_i].copy()] for od_i in range(len(od_pairs))]

    avg_tt_per_day: List[float] = []
    route_switch_per_day: List[int] = []
    dep_switch_per_day: List[int] = []
    rel_gap_per_day: List[float] = []
    dep_counts_per_day: List[np.ndarray] = []
    flow_global_per_day: List[np.ndarray] = []

    prev_flow_vec: Optional[np.ndarray] = None

    od_offsets: List[int] = []
    s = 0
    for od_i in range(len(od_pairs)):
        od_offsets.append(s)
        s += len(od_alt_maps[od_i])
    total_alt = s

    shock_edge_selected: Optional[int] = None if shock_edge_id == 0 else int(shock_edge_id)

    if shock_edge_id == 0:
        link_score: Dict[Tuple[int, int], float] = {lk: 0.0 for lk in candidate_links}
        for od_i in range(len(od_pairs)):
            for links in od_routes[od_i]:
                for lk in links:
                    if lk in link_score:
                        link_score[lk] += 1.0
        if len(link_score) > 0:
            best_link = max(link_score.items(), key=lambda kv: kv[1])[0]
            shock_edge_selected = int(G[best_link[0]][best_link[1]].get("edge", -1))

    shock_edge_ids: set[int] = set()
    shock_base_link: Optional[Tuple[int, int]] = None
    if shock_edge_selected is not None and np.isfinite(float(shock_edge_selected)):
        shock_edge_selected = int(shock_edge_selected)
        for (u, v) in candidate_links:
            if int(G[u][v].get("edge", -1)) == int(shock_edge_selected):
                shock_base_link = (int(u), int(v))
                break
        if shock_base_link is None:
            for u, v, data in G.edges(data=True):
                if int(data.get("edge", -1)) == int(shock_edge_selected):
                    shock_base_link = (int(u), int(v))
                    break

        k_hops = int(shock_k_hops)
        if k_hops <= 0 or shock_base_link is None:
            shock_edge_ids.add(int(shock_edge_selected))
        else:
            H = nx.Graph()
            for (a, b) in candidate_links:
                H.add_edge(int(a), int(b))

            buffer_nodes = {int(shock_base_link[0]), int(shock_base_link[1])}
            frontier = set(buffer_nodes)
            for _ in range(k_hops):
                next_frontier: set[int] = set()
                for n in frontier:
                    if H.has_node(n):
                        next_frontier.update(int(x) for x in H.neighbors(n))
                next_frontier = next_frontier - buffer_nodes
                if not next_frontier:
                    break
                buffer_nodes.update(next_frontier)
                frontier = next_frontier

            for (a, b) in candidate_links:
                ia, ib = int(a), int(b)
                if ia in buffer_nodes and ib in buffer_nodes:
                    eid = int(G[ia][ib].get("edge", -1))
                    if eid >= 0:
                        shock_edge_ids.add(eid)

    # Track link-level data if requested
    daily_link_flows: List[Dict[Tuple[int, int], float]] = [] if track_link_data else []
    daily_link_tt: List[Dict[Tuple[int, int], float]] = [] if track_link_data else []

    for day in range(days):
        day_no = day + 1

        flow_od = [np.zeros(len(od_alt_maps[od_i]), dtype=float) for od_i in range(len(od_pairs))]
        chosen_alt_idx: List[int] = []

        lag = min(max(1, int(s1_lag_days)), len(tt_hist_od[0]))
        eval_tt_od = [tt_hist_od[od_i][-lag] for od_i in range(len(od_pairs))]

        global_signal_od = eval_tt_od
        if scenario_mode in {"G", "GN"} and float(s1_meas_var) > 0.0:
            global_signal_od = [
                np.maximum(sig + rng_s1.normal(0.0, float(np.sqrt(float(s1_meas_var))), size=sig.shape), 1e-6)
                for sig in global_signal_od
            ]

        rumor_signal_od = last_day_I_od
        rumor_vars_od: Optional[List[np.ndarray]] = None
        if scenario_mode == "R":
            rumor_vars_od = []
            for od_i in range(len(od_pairs)):
                alt_count = len(od_alt_maps[od_i])
                if bool(no_sharing):
                    v = np.full(alt_count, info_var_base, dtype=float)
                    if bool(s2_no_fallback):
                        m0 = last_day_flow_od[od_i] <= 0
                        if np.any(m0):
                            v[m0] = float("inf")
                    rumor_vars_od.append(v)
                else:
                    if s2_info_var_mode == "share_power":
                        share = np.zeros(alt_count, dtype=float)
                        denom = float(np.sum(last_day_flow_od[od_i]))
                        if denom > 0:
                            share = last_day_flow_od[od_i] / denom
                        if bool(s2_no_fallback):
                            m0 = share <= 0
                            share2 = np.clip(share, 1e-6, 1.0)
                            v = info_var_base / (share2**share_n)
                            if np.any(m0):
                                v[m0] = float("inf")
                            rumor_vars_od.append(v)
                        else:
                            share = np.clip(share, 1e-6, 1.0)
                            rumor_vars_od.append(info_var_base / (share**share_n))
                    else:
                        cnt = last_day_flow_od[od_i]
                        vars_ = np.full(alt_count, float("inf"), dtype=float)
                        m_cnt = cnt > 0
                        if np.any(m_cnt):
                            vars_[m_cnt] = float(info_var_base) + float(report_var) / cnt[m_cnt]
                        rumor_vars_od.append(vars_)

        social_sum: Optional[List[DefaultDict[int, float]]] = None
        social_cnt: Optional[List[DefaultDict[int, int]]] = None
        if scenario_mode in {"N", "GN"}:
            if social_neighbors is None:
                raise RuntimeError("social_neighbors not initialized")
            social_sum = [defaultdict(float) for _ in range(total_agents)]
            social_cnt = [defaultdict(int) for _ in range(total_agents)]
            for i in range(total_agents):
                od_i = int(agents_od[i])
                n_routes = len(od_routes[od_i])
                for nb in social_neighbors[i]:
                    od_nb, r_prev, t_prev = agents_prev_choice[int(nb)]
                    if int(od_nb) != int(od_i):
                        continue
                    alt_nb = int(t_prev) * n_routes + int(r_prev)
                    obs = float(last_day_tt_od[od_i][alt_nb])
                    social_sum[i][alt_nb] += obs
                    social_cnt[i][alt_nb] += 1

        def _pretrip_update(i: int, od_i: int, alt_idx: int, mu: float, var: float, use_info: bool) -> Tuple[float, float]:
            if not use_info:
                return mu, var

            if scenario_mode == "G":
                if bool(s1_active_only) and float(last_day_flow_od[od_i][alt_idx]) <= 0.0:
                    return mu, var
                agent_info_var = float(agents_info_var[i])
                if update_rule == "bayes":
                    return bayes_update(mu, var, float(global_signal_od[od_i][alt_idx]), agent_info_var)
                return _exp_smooth(mu, float(global_signal_od[od_i][alt_idx]), float(eta_info)), var

            if scenario_mode == "R":
                if rumor_vars_od is None:
                    raise RuntimeError("rumor_vars_od not initialized")
                obs_var = float(rumor_vars_od[od_i][alt_idx])
                if not np.isfinite(obs_var) or obs_var <= 0.0:
                    return mu, var
                if update_rule == "bayes":
                    return bayes_update(mu, var, float(rumor_signal_od[od_i][alt_idx]), obs_var)
                return _exp_smooth(mu, float(rumor_signal_od[od_i][alt_idx]), float(eta_info)), var

            if scenario_mode == "N":
                if social_sum is None or social_cnt is None:
                    raise RuntimeError("social signal not initialized")
                cnt = int(social_cnt[i].get(int(alt_idx), 0))
                if cnt <= 0:
                    return mu, var
                obs = float(social_sum[i][int(alt_idx)]) / float(cnt)
                if update_rule == "bayes":
                    return bayes_update(mu, var, obs, float(social_var))
                return _exp_smooth(mu, obs, float(eta_info)), var

            if scenario_mode == "GN":
                mu2, var2 = mu, var
                agent_info_var = float(agents_info_var[i])
                if not (bool(s1_active_only) and float(last_day_flow_od[od_i][alt_idx]) <= 0.0):
                    if update_rule == "bayes":
                        mu2, var2 = bayes_update(mu2, var2, float(global_signal_od[od_i][alt_idx]), agent_info_var)
                    else:
                        mu2 = _exp_smooth(mu2, float(global_signal_od[od_i][alt_idx]), float(eta_info))

                if social_sum is None or social_cnt is None:
                    raise RuntimeError("social signal not initialized")
                cnt = int(social_cnt[i].get(int(alt_idx), 0))
                if cnt > 0:
                    obs = float(social_sum[i][int(alt_idx)]) / float(cnt)
                    if update_rule == "bayes":
                        mu2, var2 = bayes_update(mu2, var2, obs, float(social_var))
                    else:
                        mu2 = _exp_smooth(mu2, obs, float(eta_info))
                return mu2, var2

            return mu, var

        link_flow: Dict[Tuple[int, int], float] = {}
        dep_counts = np.zeros(num_dep_windows, dtype=float)

        for i in range(total_agents):
            od_i = agents_od[i]
            alts = od_alt_maps[od_i]
            if route_only:
                cand_idx = od_dep_to_alt[od_i][int(agents_fixed_dep[i])]
            else:
                cand_idx = list(range(len(alts)))

            costs = np.zeros(len(cand_idx), dtype=float)

            use_info = scenario_mode != "S0" and bool(agents_has_info[i]) and bool(use_pretrip)

            for jj, j in enumerate(cand_idx):
                r_idx, dep_idx = alts[j]
                b = agents_beliefs[i][j]
                mu1, var1 = b.mu, b.var

                mu2, var2 = _pretrip_update(i, int(od_i), int(j), float(mu1), float(var1), bool(use_info))

                dep_time = dep_idx * dep_window_size
                arrival = dep_time + mu2
                early, late = schedule_penalties(arrival, target_arrival)
                costs[jj] = mu2 + beta_early * early + beta_late * late + float(risk_eta) * float(var2)

            j_star = None
            if float(br_delta) > 0.0:
                od_prev, r_prev, t_prev = agents_prev_choice[i]
                if int(od_prev) == int(od_i):
                    n_routes = len(od_routes[od_i])
                    j_prev = int(t_prev) * int(n_routes) + int(r_prev)
                    if j_prev in cand_idx:
                        jj_prev = int(cand_idx.index(int(j_prev)))
                        min_cost = float(np.min(costs))
                        prev_cost = float(costs[jj_prev])
                        if np.isfinite(min_cost) and np.isfinite(prev_cost) and (prev_cost - min_cost) <= float(br_delta):
                            j_star = int(j_prev)

            if j_star is None:
                jj_star = softmax_choice(costs, theta=theta, rng=rng)
                j_star = int(cand_idx[jj_star])
            chosen_alt_idx.append(j_star)
            flow_od[od_i][j_star] += 1.0

            r_idx, dep_idx = alts[j_star]
            dep_counts[dep_idx] += 1.0

            for u, v in od_routes[od_i][r_idx]:
                link_flow[(u, v)] = link_flow.get((u, v), 0.0) + 1.0

        in_shock = shock_day > 0 and shock_len > 0 and shock_day <= day_no <= (shock_day + shock_len - 1)

        link_tt: Dict[Tuple[int, int], float] = {}
        for (u, v) in candidate_links:
            vol = float(link_flow.get((u, v), 0.0))
            t0 = float(G[u][v]["free_flow_time"])
            cap = float(G[u][v]["capacity"]) * float(capacity_scale)
            if in_shock and shock_edge_ids and int(G[u][v].get("edge", -1)) in shock_edge_ids:
                cap = cap * max(float(shock_factor), 1e-6)
            link_tt[(u, v)] = bpr_time(t0, vol, cap)

        # Track link-level data
        if track_link_data:
            daily_link_flows.append(dict(link_flow))
            daily_link_tt.append(dict(link_tt))

        tt_od: List[np.ndarray] = []
        for od_i in range(len(od_pairs)):
            n_alts = len(od_alt_maps[od_i])
            tt = np.zeros(n_alts, dtype=float)
            for j, (r_idx, dep_idx) in enumerate(od_alt_maps[od_i]):
                tt[j] = float(sum(link_tt[(u, v)] for (u, v) in od_routes[od_i][r_idx]))
            tt_od.append(tt)

        day_tt_vals: List[float] = []
        new_prev: List[Tuple[int, int, int]] = []
        shared_sum_od = [np.zeros(len(od_alt_maps[od_i]), dtype=float) for od_i in range(len(od_pairs))]
        shared_cnt_od = [np.zeros(len(od_alt_maps[od_i]), dtype=float) for od_i in range(len(od_pairs))]

        for i in range(total_agents):
            od_i = agents_od[i]
            alts = od_alt_maps[od_i]
            j_star = chosen_alt_idx[i]
            r_idx_star, dep_idx_star = alts[j_star]

            use_info = scenario_mode != "S0" and bool(agents_has_info[i]) and bool(use_pretrip)

            b = agents_beliefs[i][j_star]
            mu1, var1 = b.mu, b.var
            mu2, var2 = _pretrip_update(i, int(od_i), int(j_star), float(mu1), float(var1), bool(use_info))

            t_exp = float(tt_od[od_i][j_star])
            if scenario_mode == "R" and report_var > 0.0:
                t_rep = float(t_exp + rng_report.normal(0.0, float(np.sqrt(report_var))))
                t_rep = max(t_rep, 1e-6)
            else:
                t_rep = t_exp
            shared_sum_od[od_i][j_star] += t_rep
            shared_cnt_od[od_i][j_star] += 1.0
            if use_posttrip:
                if update_rule == "bayes":
                    mu3, var3 = bayes_update(mu2, var2, t_exp, exp_var)
                else:
                    mu3, var3 = _exp_smooth(mu2, t_exp, float(eta_exp)), var2
            else:
                mu3, var3 = mu2, var2
            agents_beliefs[i][j_star] = BayesianBelief(mu=mu3, var=var3)

            for j in range(len(alts)):
                if j == j_star:
                    continue
                bj = agents_beliefs[i][j]
                mu1j, var1j = bj.mu, bj.var
                mu2j, var2j = _pretrip_update(i, int(od_i), int(j), float(mu1j), float(var1j), bool(use_info))
                agents_beliefs[i][j] = BayesianBelief(mu=mu2j, var=var2j)

            day_tt_vals.append(t_exp)
            new_prev.append((od_i, r_idx_star, dep_idx_star))

        r_switch = 0
        d_switch = 0
        for i in range(total_agents):
            _, r_prev, t_prev = agents_prev_choice[i]
            _, r_cur, t_cur = new_prev[i]
            if r_prev != r_cur:
                r_switch += 1
            if t_prev != t_cur:
                d_switch += 1

        agents_prev_choice = new_prev

        flow_global = np.zeros(total_alt, dtype=float)
        for od_i in range(len(od_pairs)):
            off = od_offsets[od_i]
            flow_global[off : off + len(flow_od[od_i])] = flow_od[od_i]

        flow_global_per_day.append(flow_global.copy())

        if prev_flow_vec is None:
            rg = float("nan")
        else:
            rg = relative_gap(prev_flow_vec, flow_global)
        prev_flow_vec = flow_global.copy()

        avg_tt_per_day.append(float(np.mean(day_tt_vals)))
        route_switch_per_day.append(r_switch)
        dep_switch_per_day.append(d_switch)
        rel_gap_per_day.append(rg)
        dep_counts_per_day.append(dep_counts)

        for od_i in range(len(od_pairs)):
            last_day_tt_od[od_i] = tt_od[od_i]
            if scenario_mode == "R":
                last_day_I = tt_od[od_i].copy()
                m = shared_cnt_od[od_i] > 0
                if np.any(m):
                    last_day_I[m] = shared_sum_od[od_i][m] / shared_cnt_od[od_i][m]
                last_day_I_od[od_i] = last_day_I
            else:
                last_day_I_od[od_i] = last_day_tt_od[od_i]
            last_day_flow_od[od_i] = flow_od[od_i]
            tt_hist_od[od_i].append(tt_od[od_i].copy())

    result = {
        "avg_tt": np.array(avg_tt_per_day),
        "route_switch": np.array(route_switch_per_day),
        "dep_switch": np.array(dep_switch_per_day),
        "rel_gap": np.array(rel_gap_per_day),
        "dep_counts": np.stack(dep_counts_per_day, axis=0),
        "daily_flow_global": np.stack(flow_global_per_day, axis=0),
        "od_offsets": np.array(od_offsets, dtype=int),
        "od_sizes": np.array([len(od_alt_maps[od_i]) for od_i in range(len(od_pairs))], dtype=int),
        "shock_edge_selected": float(shock_edge_selected) if shock_edge_selected is not None else float("nan"),
        "shock_edge_count": np.array([int(len(shock_edge_ids))], dtype=int),
        "shock_edge_ids": np.array(sorted(list(shock_edge_ids)), dtype=int),
    }

    if track_link_data:
        result["daily_link_flows"] = daily_link_flows
        result["daily_link_tt"] = daily_link_tt

    return result
