import argparse
import os
import sys
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import networkx as nx
import numpy as np

sys.path.append(os.path.normpath(os.path.join(os.path.dirname(__file__), "..")))
from dtd_srdt.metrics import compute_disruption_metrics
from dtd_srdt.sioux_sim import run_sioux_simulation

sys.path.append(os.path.dirname(__file__))
from sf import load_graph


def _draw_errorbar(ax, x, y, yerr, color, label):
    y = np.array(y, dtype=float)
    yerr = np.array(yerr, dtype=float)
    m = np.isfinite(y)
    if np.any(m):
        ax.errorbar(
            np.array(x, dtype=float)[m],
            y[m],
            yerr=yerr[m],
            fmt="o",
            capsize=3,
            markersize=5.2,
            color=color,
            label=label,
        )


def _disable_offset(ax) -> None:
    fmt = mticker.ScalarFormatter(useOffset=False)
    fmt.set_scientific(False)
    ax.yaxis.set_major_formatter(fmt)


def _set_fixed_decimals(ax, decimals: int = 3) -> None:
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter(f"%.{int(decimals)}f"))


def draw_main_table_figures(outdir: str) -> None:
    plt.rcParams.update({
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
    })
    scenarios = ["S0", "S1", "S2"]
    sc_labels = ["S0", "S1", "S2"]
    colors = {"S0": "#4C78A8", "S1": "#F58518", "S2": "#54A24B"}

    metrics = [
        ("baseline TT", "baseline_tt"),
        ("peak TT", "peak_tt"),
        ("peak RG", "peak_rg"),
        ("recovery day", "recovery_day"),
        ("last10 TT", "tt_last10"),
        ("last10 RG", "rg_last10"),
    ]

    toy = {
        "S0": {"baseline_tt": (3.073, 0.013), "peak_tt": (10.572, 2.866), "peak_rg": (0.372, 0.036), "recovery_day": (47.3, 5.7), "tt_last10": (3.172, 0.023), "rg_last10": (0.285, 0.013)},
        "S1": {"baseline_tt": (3.090, 0.014), "peak_tt": (7.628, 2.455), "peak_rg": (0.385, 0.061), "recovery_day": (51.1, 5.6), "tt_last10": (3.194, 0.029), "rg_last10": (0.269, 0.019)},
        "S2": {"baseline_tt": (3.085, 0.015), "peak_tt": (10.043, 1.716), "peak_rg": (0.365, 0.028), "recovery_day": (47.6, 4.8), "tt_last10": (3.175, 0.022), "rg_last10": (0.280, 0.020)},
    }
    sf = {
        "S0": {"baseline_tt": (13.635, 0.001), "peak_tt": (33.455, 0.132), "peak_rg": (0.486, 0.040), "recovery_day": (float("nan"), float("nan")), "tt_last10": (13.939, 0.020), "rg_last10": (0.073, 0.005)},
        "S1": {"baseline_tt": (13.637, 0.001), "peak_tt": (33.397, 0.190), "peak_rg": (0.976, 0.015), "recovery_day": (41.4, 0.7), "tt_last10": (13.546, 0.006), "rg_last10": (0.059, 0.005)},
        "S2": {"baseline_tt": (13.634, 0.001), "peak_tt": (33.437, 0.153), "peak_rg": (0.759, 0.033), "recovery_day": (51.3, 1.5), "tt_last10": (13.691, 0.026), "rg_last10": (0.068, 0.005)},
    }

    def _draw_combined_summary(outpath: str, title: str, src: dict) -> None:
        xs = np.arange(len(scenarios), dtype=float)
        fig, axes = plt.subplots(2, 3, figsize=(11.0, 5.6))
        for j, (m_title, m_key) in enumerate(metrics):
            ax = axes[j // 3, j % 3]
            for i, sc in enumerate(scenarios):
                mu, sd = src[sc][m_key]
                _draw_errorbar(ax, [xs[i]], [mu], [sd], color=colors[sc], label=(sc_labels[i] if j == 0 else None))
                if m_key == "recovery_day" and (not np.isfinite(float(mu))):
                    ax.annotate(
                        "NA",
                        xy=(xs[i], 0.0),
                        xycoords=("data", "axes fraction"),
                        xytext=(0, 6),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=10,
                        color=colors[sc],
                    )
            ax.set_title(m_title)
            ax.set_xticks(xs)
            ax.set_xticklabels(sc_labels)
            ax.grid(True, alpha=0.25)
            _disable_offset(ax)

            # Baseline TT panels can show tiny differences; keep them readable.
            if m_key == "baseline_tt":
                _set_fixed_decimals(ax, decimals=3)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.08))
        fig.suptitle(title, y=0.99)
        fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.94])
        fig.savefig(outpath, dpi=300)
        plt.close(fig)

    _draw_combined_summary(
        os.path.join(outdir, "toy_combined_summary_multiseed.png"),
        "Toy network: combined summary (mean±std, seeds 7-16)",
        toy,
    )
    _draw_combined_summary(
        os.path.join(outdir, "sf_combined_summary_multiseed.png"),
        "Sioux Falls: combined summary (mean±std, seeds 7-16)",
        sf,
    )

    phases = ["base", "shock", "post"]
    x = np.arange(len(phases), dtype=float)
    conc = {
        "S1": {
            "hhi": ([0.6708, 0.5784, 0.5316], [0.0063, 0.0047, 0.0034]),
            "entropy": ([0.1425, 0.1739, 0.1974], [0.0022, 0.0022, 0.0016]),
        },
        "S2": {
            "hhi": ([0.5702, 0.3990, 0.3824], [0.0211, 0.0119, 0.0111]),
            "entropy": ([0.1972, 0.3335, 0.3614], [0.0058, 0.0047, 0.0074]),
        },
    }

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
    for ax, key, title in [(axes[0], "hhi", "HHI"), (axes[1], "entropy", "Normalized entropy")]:
        for sc, style in [("S1", "-o"), ("S2", "-s")]:
            y, yerr = conc[sc][key]
            ax.errorbar(x, y, yerr=yerr, fmt=style, capsize=3, markersize=4.0, linewidth=1.4, label=sc)
        ax.set_xticks(x)
        ax.set_xticklabels(phases)
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "sf_concentration_phases.png"), dpi=300)
    plt.close(fig)

    trust = {
        "Homogeneous": {"baseline_tt": (13.637, 0.002), "peak_tt": (33.42, 0.15), "peak_rg": (0.977, 0.012), "recovery_day": (41.6, 0.8), "post_var": (0.0071, 0.0011), "hhi_post": (0.532, 0.005)},
        "Mixed": {"baseline_tt": (13.637, 0.002), "peak_tt": (33.42, 0.15), "peak_rg": (0.892, 0.025), "recovery_day": (41.6, 0.8), "post_var": (0.0083, 0.0016), "hhi_post": (0.523, 0.008)},
    }
    trust_metrics = [
        ("baseline TT", "baseline_tt"),
        ("peak TT", "peak_tt"),
        ("peak RG", "peak_rg"),
        ("recovery day", "recovery_day"),
        ("post-shock TT var", "post_var"),
        ("HHI (post)", "hhi_post"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 5.6))
    xs = np.arange(2, dtype=float)
    xlabels = ["Homogeneous", "Mixed"]
    for j, (t_title, t_key) in enumerate(trust_metrics):
        ax = axes[j // 3, j % 3]
        ys = [trust[k][t_key][0] for k in xlabels]
        es = [trust[k][t_key][1] for k in xlabels]
        ax.errorbar(xs, ys, yerr=es, fmt="o", capsize=3, markersize=4.5, color="#4C78A8")
        ax.set_xticks(xs)
        ax.set_xticklabels(["Homo", "Mixed"])
        ax.set_title(t_title)
        ax.grid(True, alpha=0.25)
        _disable_offset(ax)
        if t_key == "baseline_tt":
            _set_fixed_decimals(ax, decimals=3)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, "sf_trust_heterogeneity_summary.png"), dpi=300)
    plt.close(fig)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _resolve_outdir(fig_root: str, outdir: str) -> str:
    outdir = str(outdir)
    if os.path.isabs(outdir):
        return outdir
    fig_root = str(fig_root) if fig_root is not None else ""
    if fig_root.strip() in ("", "."):
        return outdir
    return os.path.join(fig_root, outdir)


def draw_controlled_framework(outpath: str) -> None:
    fig = plt.figure(figsize=(13, 7))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")

    def box(xy, w, h, text, fc="#F7F7F7"):
        from matplotlib.patches import FancyBboxPatch

        x, y = xy
        p = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.02",
            linewidth=1.2,
            edgecolor="#333333",
            facecolor=fc,
        )
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=10)

    def arrow(p0, p1, text=None):
        ax.annotate(
            "",
            xy=p1,
            xytext=p0,
            arrowprops=dict(arrowstyle="->", lw=1.4, color="#333333"),
        )
        if text:
            xm = (p0[0] + p1[0]) / 2
            ym = (p0[1] + p1[1]) / 2
            ax.text(xm, ym + 0.02, text, ha="center", va="bottom", fontsize=9)

    box((0.03, 0.66), 0.25, 0.26, "Inputs\n\nNetwork + OD set\nStatic BPR parameters\nSRD choice set (routes×windows)\nDisruption setting", fc="#E8F1FF")
    box((0.03, 0.36), 0.25, 0.26, "Information structure\n\nS0: experience only\nG: global broadcast\nN: neighbor social\nGN: Bayesian fusion\nR: rumor / sharing", fc="#FFF2E8")
    box((0.03, 0.06), 0.25, 0.26, "Behavioral parameters\n\nLogit sensitivity θ\nSchedule weights (β_E, β_L)\nUncertainty: σ0², σE²\nPenetration p\nSocial variance σ_N²", fc="#EAF8EE")

    box((0.35, 0.70), 0.30, 0.22, "Day d: pre-trip\nBayesian perception update\n(incorporate information signal)", fc="#FFFFFF")
    box((0.35, 0.44), 0.30, 0.22, "Day d: SRD choice\nLogit over perceived generalized cost\n(schedule delay + predicted TT)", fc="#FFFFFF")
    box((0.35, 0.18), 0.30, 0.22, "Day d: within-day loading\nStatic BPR (flows aggregated\nover departure windows)", fc="#FFFFFF")
    box((0.35, 0.02), 0.30, 0.12, "Day d: post-trip\nUpdate chosen alternative\nwith experienced TT", fc="#FFFFFF")

    box((0.72, 0.62), 0.25, 0.30, "Outputs (trajectories)\n\nAverage TT\nRelative gap\nSwitchings\nDeparture heatmap", fc="#F7F7F7")
    box((0.72, 0.22), 0.25, 0.30, "Outputs (metrics)\n\nPeak TT / Peak RG\nReturn-to-baseline day\nPost-shock TT variance\nDep-window mean", fc="#F7F7F7")

    arrow((0.29, 0.79), (0.35, 0.81))
    arrow((0.29, 0.49), (0.35, 0.55))
    arrow((0.29, 0.19), (0.35, 0.11))

    arrow((0.50, 0.70), (0.50, 0.66))
    arrow((0.50, 0.44), (0.50, 0.40))
    arrow((0.50, 0.18), (0.50, 0.14))

    arrow((0.65, 0.81), (0.72, 0.77), text="trajectories")
    arrow((0.65, 0.10), (0.72, 0.37), text="summary")

    ax.set_title("Controlled SRD experiments under static within-day congestion", fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def draw_social_topologies(outpath: str, n: int = 30, k: int = 6, seed: int = 7) -> None:
    rng = np.random.default_rng(seed)

    g_ring = nx.watts_strogatz_graph(n=n, k=(k if k % 2 == 0 else k - 1), p=0.0, seed=seed)
    p_er = min(1.0, float(k) / float(max(1, n - 1)))
    g_er = nx.erdos_renyi_graph(n=n, p=p_er, seed=seed)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    pos_ring = nx.circular_layout(g_ring)
    pos_er = nx.spring_layout(g_er, seed=seed)

    ego = int(rng.integers(0, n))
    ego_nb_ring = list(g_ring.neighbors(ego))
    ego_nb_er = list(g_er.neighbors(ego))

    def draw(ax, g, pos, title, ego_nb):
        ax.set_title(title)
        ax.axis("off")
        nx.draw_networkx_edges(g, pos, ax=ax, width=0.8, alpha=0.45, edge_color="#555555")
        node_colors = ["#4C78A8"] * n
        node_sizes = [50] * n
        node_colors[ego] = "#E45756"
        node_sizes[ego] = 140
        for j in ego_nb:
            node_colors[int(j)] = "#F58518"
            node_sizes[int(j)] = 90
        nx.draw_networkx_nodes(g, pos, ax=ax, node_color=node_colors, node_size=node_sizes, linewidths=0.0)
        ax.text(0.02, 0.02, "red: ego, orange: neighbors", transform=ax.transAxes, fontsize=9)

    draw(axes[0], g_ring, pos_ring, f"Ring lattice (avg degree k={k})", ego_nb_ring)
    draw(axes[1], g_er, pos_er, f"Erdos-Renyi (E[deg]≈{k})", ego_nb_er)

    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def draw_sioux_falls_network(outpath: str, highlight_edge_id: int = 19) -> None:
    G_multi, _, pos_xy, _ = load_graph(draw=False)

    edges_all: List[Tuple[int, int, int]] = []
    edges_hi: List[Tuple[int, int, int]] = []

    for u, v, k, data in G_multi.edges(keys=True, data=True):
        edges_all.append((u, v, k))
        if int(data.get("edge", -1)) == int(highlight_edge_id):
            edges_hi.append((u, v, k))

    fig = plt.figure(figsize=(7.2, 6.2))
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")

    nx.draw_networkx_edges(
        G_multi,
        pos_xy,
        edgelist=[(u, v) for (u, v, _) in edges_all],
        width=0.8,
        alpha=0.25,
        edge_color="#444444",
        arrows=False,
        ax=ax,
    )

    if edges_hi:
        nx.draw_networkx_edges(
            G_multi,
            pos_xy,
            edgelist=[(u, v) for (u, v, _) in edges_hi],
            width=3.0,
            alpha=0.95,
            edge_color="#E45756",
            arrows=False,
            ax=ax,
        )

        u0, v0, _ = edges_hi[0]
        x0, y0 = pos_xy[u0]
        x1, y1 = pos_xy[v0]
        ax.text((x0 + x1) / 2, (y0 + y1) / 2, f"edge={highlight_edge_id}", fontsize=9, color="#E45756")

    nx.draw_networkx_nodes(G_multi, pos_xy, node_size=30, node_color="#4C78A8", alpha=0.9, linewidths=0.0, ax=ax)

    ax.set_title("Sioux Falls network (highlight: disrupted edge)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def collect_sf_multiseed_metrics(
    scenario: str,
    seeds: List[int],
    shock_edge_id: int = 19,
    social_topology: str = "ring",
    social_k: int = 6,
    social_var: float = 1.0,
) -> List[Dict[str, float]]:
    G_multi, _, _, _ = load_graph(draw=False)

    metrics_list: List[Dict[str, float]] = []
    for seed in seeds:
        results = run_sioux_simulation(
            G_multi,
            scenario=scenario,
            days=60,
            od_pairs=[(14, 4), (15, 6)],
            demand_per_od=500,
            k_routes=5,
            num_dep_windows=20,
            dep_window_size=1.0,
            target_arrival=25.0,
            beta_early=0.8,
            beta_late=1.8,
            theta=3.0,
            init_var=4.0,
            info_var_base=0.2,
            exp_var=2.0,
            share_n=4.0,
            penetration=1.0,
            shock_day=30,
            shock_len=11,
            shock_edge_id=shock_edge_id,
            shock_factor=0.3,
            capacity_scale=0.1,
            seed=seed,
            route_only=False,
            no_sharing=False,
            use_pretrip=True,
            use_posttrip=True,
            update_rule="bayes",
            eta_info=0.3,
            eta_exp=0.3,
            s2_report_var=None,
            risk_eta=0.0,
            s1_lag_days=1,
            s1_meas_var=0.0,
            s2_info_var_mode="share_power",
            social_topology=social_topology,
            social_k=social_k,
            social_var=social_var,
        )
        m = compute_disruption_metrics(results, shock_day=30, shock_len=11)
        if m:
            out = dict(m)
            out["seed"] = float(seed)
            out["scenario"] = float(0.0)
            metrics_list.append(out)

    return metrics_list


def draw_multiseed_boxplot(outpath: str, data: Dict[str, List[float]], ylabel: str) -> None:
    labels = list(data.keys())
    series = [np.array(data[k], dtype=float) for k in labels]

    fig = plt.figure(figsize=(8.5, 4.8))
    ax = fig.add_subplot(1, 1, 1)

    ax.boxplot(series, labels=labels, showfliers=False)

    rng = np.random.default_rng(7)
    for i, arr in enumerate(series, start=1):
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        x = i + rng.normal(0.0, 0.04, size=arr.size)
        ax.scatter(x, arr, s=18, alpha=0.65, color="#4C78A8")

    ax.set_ylabel(ylabel)
    ax.set_title("Sioux Falls: across-seed distribution")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, default="figs_paper")
    parser.add_argument(
        "--fig_root",
        type=str,
        default=os.path.join("outputs", "paper"),
        help="If outdir is a plain folder name, it will be placed under this root (set '' or '.' to disable).",
    )
    parser.add_argument("--no_framework", action="store_true")
    parser.add_argument("--no_social", action="store_true")
    parser.add_argument("--no_sf_network", action="store_true")
    parser.add_argument("--no_multiseed", action="store_true")
    parser.add_argument("--main_table_figs", action="store_true")
    parser.add_argument("--seed0", type=int, default=7)
    parser.add_argument("--n_seeds", type=int, default=10)
    args = parser.parse_args()

    outdir = _resolve_outdir(args.fig_root, args.outdir)
    _ensure_dir(outdir)

    if not args.no_framework:
        draw_controlled_framework(os.path.join(outdir, "framework.png"))

    if not args.no_social:
        draw_social_topologies(os.path.join(outdir, "social_topology.png"), n=30, k=6, seed=7)

    if not args.no_sf_network:
        draw_sioux_falls_network(os.path.join(outdir, "sioux_falls_network_edge19.png"), highlight_edge_id=19)

    if not args.no_multiseed:
        n_seeds = max(1, int(args.n_seeds))
        seed0 = int(args.seed0)
        seeds = [seed0 + i for i in range(n_seeds)]

        data_post_var: Dict[str, List[float]] = {}
        for sc in ["G", "N", "GN", "R"]:
            m_list = collect_sf_multiseed_metrics(sc, seeds, shock_edge_id=19, social_topology="ring", social_k=6, social_var=1.0)
            vals = [float(m.get("post_shock_tt_var", float("nan"))) for m in m_list]
            data_post_var[sc] = [v for v in vals if np.isfinite(v)]

        draw_multiseed_boxplot(
            os.path.join(outdir, "sf_multiseed_post_shock_tt_var_box.png"),
            data_post_var,
            ylabel="post-shock TT variance",
        )

    if bool(args.main_table_figs):
        draw_main_table_figures(outdir)


if __name__ == "__main__":
    main()
