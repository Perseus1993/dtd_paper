#!/usr/bin/env python
"""
Run all experiments for the DTD paper.
This script generates all figures used in main.tex.
"""
from __future__ import annotations

import os
import argparse
import csv
import re
import sys
import subprocess
from typing import Dict, List, Optional, Tuple

# Configuration
BDI_TEST_DIR = os.path.join(os.path.dirname(__file__), "bdi_test")
OUTPUTS_ROOT = os.path.join(os.path.dirname(__file__), "outputs")
FIG_ROOT_PAPER = os.path.join(OUTPUTS_ROOT, "paper")
FIG_ROOT_TOY = os.path.join(OUTPUTS_ROOT, "toy")
FIG_ROOT_SIOUX = os.path.join(OUTPUTS_ROOT, "sioux")


def _sioux_data_path(filename: str) -> str:
    return os.path.join(os.path.dirname(__file__), "backup", "suffix", "data", filename)


def _load_sioux_flow_rows() -> List[Dict[str, object]]:
    path = _sioux_data_path("SiouxFalls_flow.tntp")
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            if len(parts) < 4:
                continue
            u = int(parts[0])
            v = int(parts[1])
            vol = float(parts[2])
            cost = float(parts[3])
            edge_id = len(rows) + 1
            rows.append({"edge": edge_id, "u": u, "v": v, "volume": vol, "cost": cost})
    return rows


def _load_sioux_net_rows() -> List[Dict[str, object]]:
    path = _sioux_data_path("SiouxFalls_net.tntp")
    rows: List[Dict[str, object]] = []
    with open(path, "r", encoding="utf-8") as f:
        edge_id = 0
        for line in f:
            s = line.strip()
            if not s:
                continue
            if s.startswith("<") or s.startswith("~"):
                continue
            parts = s.split()
            if len(parts) < 6:
                continue
            try:
                u = int(parts[0])
                v = int(parts[1])
                fft = float(parts[4])
            except (ValueError, IndexError):
                continue
            edge_id += 1
            rows.append({"edge": edge_id, "u": u, "v": v, "free_flow_time": fft})
    return rows


def _build_sioux_digraph():
    import networkx as nx

    rows = _load_sioux_net_rows()
    G = nx.DiGraph()
    for r in rows:
        u = int(r["u"])
        v = int(r["v"])
        fft = float(r["free_flow_time"])
        eid = int(r["edge"])
        if G.has_edge(u, v):
            if fft < float(G[u][v].get("free_flow_time", float("inf"))):
                G[u][v].update({"free_flow_time": fft, "edge": eid})
        else:
            G.add_edge(u, v, free_flow_time=fft, edge=eid)
    return G


def _candidate_edge_ids_sioux(od_pairs: List[Tuple[int, int]], k_routes: int) -> set:
    import networkx as nx

    G = _build_sioux_digraph()
    cand: set = set()
    for o, d in od_pairs:
        gen = nx.shortest_simple_paths(G, int(o), int(d), weight="free_flow_time")
        for _ in range(max(1, int(k_routes))):
            try:
                path = next(gen)
            except StopIteration:
                break
            for a, b in zip(path[:-1], path[1:]):
                cand.add(int(G[int(a)][int(b)]["edge"]))
    return cand


def _select_edges_e1(
    include_edge: int = 19,
    n_high: int = 2,
    n_low: int = 2,
    restrict_to_candidate: bool = False,
    od_pairs: Optional[List[Tuple[int, int]]] = None,
    k_routes: int = 5,
) -> List[int]:
    rows = _load_sioux_flow_rows()
    if not rows:
        return [int(include_edge)]

    include_edge = int(include_edge)
    candidate: Optional[set] = None
    if bool(restrict_to_candidate):
        if od_pairs is None:
            od_pairs = [(14, 4), (15, 6)]
        candidate = _candidate_edge_ids_sioux(list(od_pairs), int(k_routes))
        rows = [r for r in rows if int(r.get("edge", -1)) in candidate]
        if not rows:
            return [include_edge]

    by_high = sorted(rows, key=lambda r: float(r.get("volume", 0.0)), reverse=True)
    by_low = sorted(rows, key=lambda r: float(r.get("volume", 0.0)))

    selected: List[int] = []
    if include_edge > 0:
        if candidate is None or include_edge in candidate:
            selected.append(include_edge)
        else:
            selected.append(int(by_high[0]["edge"]))

    for r in by_high:
        if len(selected) >= 1 + int(n_high):
            break
        eid = int(r["edge"])
        if eid not in selected:
            selected.append(eid)

    for r in by_low:
        if len(selected) >= 1 + int(n_high) + int(n_low):
            break
        eid = int(r["edge"])
        if eid not in selected:
            selected.append(eid)

    return selected

def run_cmd(cmd: List[str], desc: str) -> bool:
    """Run a command and return success status."""
    print(f"\n{'='*60}")
    print(f"Running: {desc}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    result = subprocess.run(cmd, cwd=BDI_TEST_DIR)
    if result.returncode != 0:
        print(f"WARNING: {desc} failed with code {result.returncode}")
        return False
    return True


def run_cmd_capture(cmd: List[str], desc: str) -> Tuple[bool, str]:
    """Run a command and return (success, stdout)."""
    print(f"\n{'='*60}")
    print(f"Running: {desc}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    result = subprocess.run(cmd, cwd=BDI_TEST_DIR, capture_output=True, text=True)
    if result.stdout:
        print(result.stdout)
    if result.stderr:
        print(result.stderr)
    if result.returncode != 0:
        print(f"WARNING: {desc} failed with code {result.returncode}")
        return False, result.stdout or ""
    return True, result.stdout or ""


def run_cmd_capture_quiet(cmd: List[str], desc: str) -> Tuple[bool, str]:
    print(f"\n{'='*60}")
    print(f"Running: {desc}")
    print(f"Command: {' '.join(cmd)}")
    print('='*60)
    result = subprocess.run(cmd, cwd=BDI_TEST_DIR, capture_output=True, text=True)
    if result.returncode != 0:
        if result.stdout:
            print(result.stdout)
        if result.stderr:
            print(result.stderr)
        print(f"WARNING: {desc} failed with code {result.returncode}")
        return False, result.stdout or ""
    return True, result.stdout or ""


def _parse_multiseed_summary(stdout: str) -> Optional[Dict[str, Tuple[float, float]]]:
    out: Dict[str, Tuple[float, float]] = {}
    pat = re.compile(r"^(?P<k>\w+)\s+mean=(?P<mu>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan)\s+std=(?P<sd>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan)\s*$")
    for raw in stdout.splitlines():
        s = raw.strip()
        m = pat.match(s)
        if not m:
            continue
        k = str(m.group("k"))
        mu = float(m.group("mu"))
        sd = float(m.group("sd"))
        out[k] = (mu, sd)
    return out or None


def _parse_last_disruption_metrics(stdout: str) -> Optional[Dict[str, float]]:
    last = None
    for raw in stdout.splitlines():
        if "disruption_metrics" in raw:
            last = raw.strip()
    if not last:
        return None

    pat = re.compile(
        r"baseline_tt=(?P<baseline_tt>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan)\s+"
        r"shock_tt_peak=(?P<shock_tt_peak>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan)\s+"
        r"shock_rg_peak=(?P<shock_rg_peak>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan)\s+"
        r"recovery_day=(?P<recovery_day>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan|NA)\s+"
        r"post_shock_tt_var=(?P<post_shock_tt_var>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan)\s+"
        r"dep_mean\(base/shock/post\)=(?P<dep_mean_base>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan)\/"
        r"(?P<dep_mean_shock>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan)\/"
        r"(?P<dep_mean_post>[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?|nan)"
    )
    m = pat.search(last)
    if not m:
        return None

    def _to_float(s: str) -> float:
        s = str(s).strip()
        if s.upper() == "NA":
            return float("nan")
        return float(s)

    return {k: _to_float(m.group(k)) for k in pat.groupindex.keys()}


def run_toy_experiments():
    """Run Toy network experiments (S0/S1/S2) with shock."""
    print("\n" + "="*70)
    print("RUNNING TOY NETWORK EXPERIMENTS")
    print("="*70)

    # Default settings (figs_shock)
    for scenario in ["S0", "S1", "S2"]:
        run_cmd([
            sys.executable, "baseline1_srdt_bayes_toy.py",
            "--scenario", scenario,
            "--days", "60",
            "--demand", "300",
            "--theta", "1.0",
            "--info_var", "4.0",
            "--exp_var", "1.0",
            "--share_n", "2.0",
            "--shock_day", "30",
            "--shock_len", "11",
            "--shock_route", "P2",
            "--shock_factor", "0.3",
            "--outdir", "figs_shock",
            "--fig_root", FIG_ROOT_TOY,
            "--no_show",
            "--seed", "7",
        ], f"Toy {scenario} (default)")

    # Sensitivity settings (figs_shock2) - theta=3, info_var=0.2, exp_var=2, share_n=4
    for scenario in ["S0", "S1", "S2"]:
        run_cmd([
            sys.executable, "baseline1_srdt_bayes_toy.py",
            "--scenario", scenario,
            "--days", "60",
            "--demand", "300",
            "--theta", "3.0",
            "--info_var", "0.2",
            "--exp_var", "2.0",
            "--share_n", "4.0",
            "--shock_day", "30",
            "--shock_len", "11",
            "--shock_route", "P2",
            "--shock_factor", "0.3",
            "--outdir", "figs_shock2",
            "--fig_root", FIG_ROOT_TOY,
            "--no_show",
            "--seed", "7",
        ], f"Toy {scenario} (sensitivity)")


def run_toy_ablation_experiments():
    """Run Toy network ablation experiments."""
    print("\n" + "="*70)
    print("RUNNING TOY ABLATION EXPERIMENTS")
    print("="*70)

    base_args = [
        "--days", "60",
        "--demand", "300",
        "--theta", "3.0",
        "--info_var", "0.2",
        "--exp_var", "2.0",
        "--share_n", "4.0",
        "--shock_day", "30",
        "--shock_len", "11",
        "--shock_route", "P2",
        "--shock_factor", "0.3",
        "--fig_root", FIG_ROOT_TOY,
        "--no_show",
        "--seed", "7",
    ]

    # Ablation: updating mechanisms under S1
    ablations_s1 = [
        ("figs_ablation_toy_bayes2stage", []),  # Full Bayes
        ("figs_ablation_toy_exp_only", ["--no_pretrip"]),  # Experience only
        ("figs_ablation_toy_info_only", ["--no_posttrip"]),  # Info only
        ("figs_ablation_toy_smooth2stage", ["--update_rule", "exp_smooth"]),  # Exp smoothing
    ]

    for outdir, extra_args in ablations_s1:
        run_cmd([
            sys.executable, "baseline1_srdt_bayes_toy.py",
            "--scenario", "S1",
            "--outdir", outdir,
            *base_args,
            *extra_args,
        ], f"Toy S1 ablation: {outdir}")

    # Ablation: mechanism under S2 (sharing, route-only)
    ablations_s2 = [
        ("figs_mech_toy_s2", []),  # Full S2
        ("figs_mech_toy_s2_noshare", ["--no_sharing"]),  # No sharing
        ("figs_mech_toy_s2_routeonly", ["--route_only"]),  # Route only
        ("figs_mech_toy_s2_routeonly_noshare", ["--route_only", "--no_sharing"]),  # Both
    ]

    for outdir, extra_args in ablations_s2:
        run_cmd([
            sys.executable, "baseline1_srdt_bayes_toy.py",
            "--scenario", "S2",
            "--outdir", outdir,
            *base_args,
            *extra_args,
        ], f"Toy S2 mechanism: {outdir}")


def run_sioux_falls_experiments():
    """Run Sioux Falls network experiments."""
    print("\n" + "="*70)
    print("RUNNING SIOUX FALLS EXPERIMENTS")
    print("="*70)

    base_args = [
        "--days", "60",
        "--demand_per_od", "500",
        "--k_routes", "5",
        "--theta", "3.0",
        "--info_var", "0.2",
        "--exp_var", "2.0",
        "--share_n", "4.0",
        "--capacity_scale", "0.1",
        "--shock_day", "30",
        "--shock_len", "11",
        "--shock_edge", "19",
        "--shock_factor", "0.3",
        "--fig_root", FIG_ROOT_SIOUX,
        "--no_show",
    ]

    # Main S0/S1/S2 experiments
    for scenario in ["S0", "S1", "S2"]:
        run_cmd([
            sys.executable, "baseline1_srdt_bayes_sioux.py",
            "--scenario", scenario,
            "--outdir", "figs_sf2",
            *base_args,
        ], f"Sioux Falls {scenario}")

    # Multi-channel experiments (G/N/GN/R)
    for scenario in ["G", "N", "GN", "R"]:
        run_cmd([
            sys.executable, "baseline1_srdt_bayes_sioux.py",
            "--scenario", scenario,
            "--outdir", f"figs_sf_{scenario}",
            *base_args,
        ], f"Sioux Falls {scenario}")


def run_sioux_falls_ablation_experiments():
    """Run Sioux Falls ablation experiments."""
    print("\n" + "="*70)
    print("RUNNING SIOUX FALLS ABLATION EXPERIMENTS")
    print("="*70)

    base_args = [
        "--days", "60",
        "--demand_per_od", "500",
        "--k_routes", "5",
        "--theta", "3.0",
        "--info_var", "0.2",
        "--exp_var", "2.0",
        "--share_n", "4.0",
        "--capacity_scale", "0.1",
        "--shock_day", "30",
        "--shock_len", "11",
        "--shock_edge", "19",
        "--shock_factor", "0.3",
        "--fig_root", FIG_ROOT_SIOUX,
        "--no_show",
        "--seed", "7",
    ]

    # Ablation: updating mechanisms under S1/G
    ablations_s1 = [
        ("figs_ablation_sf_bayes2stage", []),
        ("figs_ablation_sf_exp_only", ["--no_pretrip"]),
        ("figs_ablation_sf_info_only", ["--no_posttrip"]),
        ("figs_ablation_sf_smooth2stage", ["--update_rule", "exp_smooth"]),
    ]

    for outdir, extra_args in ablations_s1:
        run_cmd([
            sys.executable, "baseline1_srdt_bayes_sioux.py",
            "--scenario", "S1",
            "--outdir", outdir,
            *base_args,
            *extra_args,
        ], f"Sioux Falls S1 ablation: {outdir}")

    # Ablation: mechanism under S2/R
    ablations_s2 = [
        ("figs_mech_sf_s2", []),
        ("figs_mech_sf_s2_noshare", ["--no_sharing"]),
        ("figs_mech_sf_s2_routeonly", ["--route_only"]),
        ("figs_mech_sf_s2_routeonly_noshare", ["--route_only", "--no_sharing"]),
    ]

    for outdir, extra_args in ablations_s2:
        run_cmd([
            sys.executable, "baseline1_srdt_bayes_sioux.py",
            "--scenario", "S2",
            "--outdir", outdir,
            *base_args,
            *extra_args,
        ], f"Sioux Falls S2 mechanism: {outdir}")


def run_sioux_falls_br_sensitivity(n_seeds: int = 1, seed0: int = 7):
    """Run Sioux Falls bounded rationality (indifference band) sensitivity."""
    print("\n" + "="*70)
    print("RUNNING SIOUX FALLS BOUNDED RATIONALITY (BR) SENSITIVITY")
    print("="*70)

    base_args = [
        "--days", "60",
        "--demand_per_od", "500",
        "--k_routes", "5",
        "--theta", "3.0",
        "--info_var", "0.2",
        "--exp_var", "2.0",
        "--share_n", "4.0",
        "--capacity_scale", "0.1",
        "--shock_day", "30",
        "--shock_len", "11",
        "--shock_edge", "19",
        "--shock_factor", "0.3",
        "--fig_root", FIG_ROOT_SIOUX,
        "--no_show",
    ]

    # Keep experiment scope minimal: fix one information scenario and sweep BR tolerance.
    scenario = "S1"
    br_deltas = ["0.0", "0.3", "0.6"]
    n_seeds = max(1, int(n_seeds))
    seed0 = int(seed0)

    csv_rows: List[Dict[str, object]] = []
    for br_delta in br_deltas:
        outdir = f"figs_sf_br_{scenario}_delta_{br_delta.replace('.', 'p')}"
        ok, stdout = run_cmd_capture([
            sys.executable, "baseline1_srdt_bayes_sioux.py",
            "--scenario", scenario,
            "--br_delta", br_delta,
            "--outdir", outdir,
            "--seed", str(seed0),
            "--n_seeds", str(n_seeds),
            *base_args,
        ], f"Sioux Falls BR sensitivity: {scenario}, br_delta={br_delta}, n_seeds={n_seeds}")

        if not ok:
            continue

        summary = _parse_multiseed_summary(stdout) if n_seeds > 1 else None
        single = _parse_last_disruption_metrics(stdout) if summary is None else None

        row: Dict[str, object] = {
            "scenario": scenario,
            "br_delta": float(br_delta),
            "n_seeds": n_seeds,
            "seed0": seed0,
            "seed_last": seed0 + n_seeds - 1,
        }

        keys = [
            "baseline_tt",
            "shock_tt_peak",
            "shock_rg_peak",
            "recovery_day",
            "post_shock_tt_var",
            "dep_mean_base",
            "dep_mean_shock",
            "dep_mean_post",
        ]

        if summary is not None:
            for k in keys:
                mu, sd = summary.get(k, (float("nan"), float("nan")))
                row[f"{k}_mean"] = mu
                row[f"{k}_std"] = sd
        elif single is not None:
            for k in keys:
                row[f"{k}_mean"] = float(single.get(k, float("nan")))
                row[f"{k}_std"] = 0.0
        else:
            for k in keys:
                row[f"{k}_mean"] = float("nan")
                row[f"{k}_std"] = float("nan")

        csv_rows.append(row)

    if csv_rows:
        outdir_csv = os.path.join(FIG_ROOT_SIOUX, "figs_sf_br_summary")
        os.makedirs(outdir_csv, exist_ok=True)
        csv_path = os.path.join(outdir_csv, f"sf_br_{scenario}_summary_seed{seed0}_n{n_seeds}.csv")

        fieldnames: List[str] = [
            "scenario",
            "br_delta",
            "n_seeds",
            "seed0",
            "seed_last",
        ]
        for k in [
            "baseline_tt",
            "shock_tt_peak",
            "shock_rg_peak",
            "recovery_day",
            "post_shock_tt_var",
            "dep_mean_base",
            "dep_mean_shock",
            "dep_mean_post",
        ]:
            fieldnames.append(f"{k}_mean")
            fieldnames.append(f"{k}_std")

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)

        print(f"saved_csv={csv_path}")


def run_sioux_falls_edge_sweep(n_seeds: int = 1, seed0: int = 7, mode: str = "all"):
    print("\n" + "=" * 70)
    print("RUNNING SIOUX FALLS SHOCK-EDGE SWEEP")
    print("=" * 70)

    rows = _load_sioux_flow_rows()
    meta = {int(r["edge"]): r for r in rows}

    scenario = "S1"
    n_seeds = max(1, int(n_seeds))
    seed0 = int(seed0)

    od_pairs = [(14, 4), (15, 6)]
    k_routes = 5

    mode = str(mode).strip().lower()
    if mode not in {"all", "candidate"}:
        raise ValueError("edge sweep mode must be 'all' or 'candidate'")

    if mode == "all":
        edge_ids = sorted(meta.keys())
    else:
        cand = _candidate_edge_ids_sioux(od_pairs, k_routes)
        edge_ids = sorted([eid for eid in meta.keys() if int(eid) in cand])

    base_args = [
        "--days",
        "60",
        "--demand_per_od",
        "500",
        "--k_routes",
        "5",
        "--theta",
        "3.0",
        "--info_var",
        "0.2",
        "--exp_var",
        "2.0",
        "--share_n",
        "4.0",
        "--capacity_scale",
        "0.1",
        "--shock_day",
        "30",
        "--shock_len",
        "11",
        "--shock_factor",
        "0.3",
        "--fig_root",
        FIG_ROOT_SIOUX,
        "--no_show",
        "--no_plot",
    ]

    keys = [
        "baseline_tt",
        "shock_tt_peak",
        "shock_rg_peak",
        "recovery_day",
        "post_shock_tt_var",
        "dep_mean_base",
        "dep_mean_shock",
        "dep_mean_post",
    ]

    outdir_csv = os.path.join(FIG_ROOT_SIOUX, "figs_sf_edge_sweep")
    os.makedirs(outdir_csv, exist_ok=True)
    csv_path = os.path.join(outdir_csv, f"sf_edge_{scenario}_SWEEP_{mode}_seed{seed0}_n{n_seeds}.csv")

    fieldnames: List[str] = [
        "scenario",
        "sweep_mode",
        "shock_edge",
        "u",
        "v",
        "base_volume",
        "base_cost",
        "n_seeds",
        "seed0",
        "seed_last",
    ]
    for k in keys:
        fieldnames.append(f"{k}_mean")
        fieldnames.append(f"{k}_std")

    done_edges: set[int] = set()
    if os.path.exists(csv_path):
        try:
            with open(csv_path, "r", newline="") as rf:
                reader = csv.DictReader(rf)
                for r in reader:
                    try:
                        done_edges.add(int(r.get("shock_edge", "")))
                    except (TypeError, ValueError):
                        continue
        except OSError:
            done_edges = set()

    file_exists = os.path.exists(csv_path)
    open_mode = "a" if file_exists else "w"
    with open(csv_path, open_mode, newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()

        for eid in edge_ids:
            eid = int(eid)
            if eid in done_edges:
                continue
            outdir = f"figs_sf_sweep_{scenario}_edge_{eid}"
            ok, stdout = run_cmd_capture_quiet(
                [
                    sys.executable,
                    "baseline1_srdt_bayes_sioux.py",
                    "--scenario",
                    scenario,
                    "--shock_edge",
                    str(eid),
                    "--outdir",
                    outdir,
                    "--seed",
                    str(seed0),
                    "--n_seeds",
                    str(n_seeds),
                    *base_args,
                ],
                f"Sioux Falls edge sweep: {scenario}, edge={eid}, n_seeds={n_seeds}",
            )
            if not ok:
                continue

            summary = _parse_multiseed_summary(stdout) if n_seeds > 1 else None
            single = _parse_last_disruption_metrics(stdout) if summary is None else None

            m = meta.get(eid, {})
            row: Dict[str, object] = {
                "scenario": scenario,
                "sweep_mode": mode,
                "shock_edge": eid,
                "u": int(m.get("u", -1)) if m else -1,
                "v": int(m.get("v", -1)) if m else -1,
                "base_volume": float(m.get("volume", float("nan"))) if m else float("nan"),
                "base_cost": float(m.get("cost", float("nan"))) if m else float("nan"),
                "n_seeds": n_seeds,
                "seed0": seed0,
                "seed_last": seed0 + n_seeds - 1,
            }

            if summary is not None:
                for k in keys:
                    mu, sd = summary.get(k, (float("nan"), float("nan")))
                    row[f"{k}_mean"] = mu
                    row[f"{k}_std"] = sd
            elif single is not None:
                for k in keys:
                    row[f"{k}_mean"] = float(single.get(k, float("nan")))
                    row[f"{k}_std"] = 0.0
            else:
                for k in keys:
                    row[f"{k}_mean"] = float("nan")
                    row[f"{k}_std"] = float("nan")

            writer.writerow(row)
            f.flush()

    print(f"saved_csv={csv_path}")


def run_sioux_falls_edge_sensitivity(n_seeds: int = 10, seed0: int = 7):
    print("\n" + "=" * 70)
    print("RUNNING SIOUX FALLS SHOCK-EDGE ROBUSTNESS (E1)")
    print("=" * 70)

    rows = _load_sioux_flow_rows()
    meta = {int(r["edge"]): r for r in rows}
    od_pairs = [(14, 4), (15, 6)]
    k_routes = 5
    edge_ids = _select_edges_e1(
        include_edge=19,
        n_high=2,
        n_low=2,
        restrict_to_candidate=True,
        od_pairs=od_pairs,
        k_routes=k_routes,
    )

    print("E1 selected shock_edge set (edge_id: u->v, base_volume, base_cost):")
    for eid in edge_ids:
        m = meta.get(int(eid), {})
        u = int(m.get("u", -1)) if m else -1
        v = int(m.get("v", -1)) if m else -1
        vol = float(m.get("volume", float("nan"))) if m else float("nan")
        cost = float(m.get("cost", float("nan"))) if m else float("nan")
        print(f"  edge={int(eid)}: {u}->{v}, vol={vol:.3f}, cost={cost:.3f}")

    scenario = "S1"
    n_seeds = max(1, int(n_seeds))
    seed0 = int(seed0)

    base_args = [
        "--days",
        "60",
        "--demand_per_od",
        "500",
        "--k_routes",
        "5",
        "--theta",
        "3.0",
        "--info_var",
        "0.2",
        "--exp_var",
        "2.0",
        "--share_n",
        "4.0",
        "--capacity_scale",
        "0.1",
        "--shock_day",
        "30",
        "--shock_len",
        "11",
        "--shock_factor",
        "0.3",
        "--fig_root",
        FIG_ROOT_SIOUX,
        "--no_show",
    ]

    keys = [
        "baseline_tt",
        "shock_tt_peak",
        "shock_rg_peak",
        "recovery_day",
        "post_shock_tt_var",
        "dep_mean_base",
        "dep_mean_shock",
        "dep_mean_post",
    ]

    out_rows: List[Dict[str, object]] = []
    for eid in edge_ids:
        eid = int(eid)
        outdir = f"figs_sf_edge_{scenario}_edge_{eid}"
        ok, stdout = run_cmd_capture(
            [
                sys.executable,
                "baseline1_srdt_bayes_sioux.py",
                "--scenario",
                scenario,
                "--shock_edge",
                str(eid),
                "--outdir",
                outdir,
                "--seed",
                str(seed0),
                "--n_seeds",
                str(n_seeds),
                *base_args,
            ],
            f"Sioux Falls edge robustness: {scenario}, edge={eid}, n_seeds={n_seeds}",
        )
        if not ok:
            continue

        summary = _parse_multiseed_summary(stdout) if n_seeds > 1 else None
        single = _parse_last_disruption_metrics(stdout) if summary is None else None

        m = meta.get(eid, {})
        row: Dict[str, object] = {
            "scenario": scenario,
            "shock_edge": eid,
            "u": int(m.get("u", -1)) if m else -1,
            "v": int(m.get("v", -1)) if m else -1,
            "base_volume": float(m.get("volume", float("nan"))) if m else float("nan"),
            "base_cost": float(m.get("cost", float("nan"))) if m else float("nan"),
            "n_seeds": n_seeds,
            "seed0": seed0,
            "seed_last": seed0 + n_seeds - 1,
        }

        if summary is not None:
            for k in keys:
                mu, sd = summary.get(k, (float("nan"), float("nan")))
                row[f"{k}_mean"] = mu
                row[f"{k}_std"] = sd
        elif single is not None:
            for k in keys:
                row[f"{k}_mean"] = float(single.get(k, float("nan")))
                row[f"{k}_std"] = 0.0
        else:
            for k in keys:
                row[f"{k}_mean"] = float("nan")
                row[f"{k}_std"] = float("nan")

        out_rows.append(row)

    if out_rows:
        outdir_csv = os.path.join(FIG_ROOT_SIOUX, "figs_sf_edge_summary")
        os.makedirs(outdir_csv, exist_ok=True)
        csv_path = os.path.join(outdir_csv, f"sf_edge_{scenario}_E1_seed{seed0}_n{n_seeds}.csv")

        fieldnames: List[str] = [
            "scenario",
            "shock_edge",
            "u",
            "v",
            "base_volume",
            "base_cost",
            "n_seeds",
            "seed0",
            "seed_last",
        ]
        for k in keys:
            fieldnames.append(f"{k}_mean")
            fieldnames.append(f"{k}_std")

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(out_rows)

        print(f"saved_csv={csv_path}")


def run_sioux_falls_heatmap():
    """Run Sioux Falls heatmap sweep."""
    print("\n" + "="*70)
    print("RUNNING SIOUX FALLS HEATMAP SWEEP")
    print("="*70)

    run_cmd([
        sys.executable, "sweep_sf_heatmap.py",
        "--outdir", "figs_sf_heatmap_demo",
        "--fig_root", FIG_ROOT_SIOUX,
        "--scenario", "S1",
    ], "Sioux Falls heatmap (S1)")


def run_paper_figures():
    """Generate paper framework and topology figures."""
    print("\n" + "="*70)
    print("GENERATING PAPER FIGURES")
    print("="*70)

    run_cmd([
        sys.executable, "paper_figs.py",
        "--outdir", "figs_paper",
        "--fig_root", FIG_ROOT_PAPER,
        "--n_seeds", "10",
        "--seed0", "7",
    ], "Paper figures (framework, topology, multiseed boxplot)")


def main():
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--n_seeds",
        type=int,
        default=1,
        help="Number of random seeds for the Sioux Falls BR sensitivity sweep (seed, seed+1, ..., seed+n_seeds-1).",
    )
    parser.add_argument("--seed", type=int, default=7, help="Seed0 for the Sioux Falls BR sensitivity sweep.")
    parser.add_argument(
        "--run_edge_sens",
        action="store_true",
        help="If set, run a small Sioux Falls shock-edge robustness check (E1) and save a CSV summary.",
    )
    parser.add_argument(
        "--run_edge_sweep",
        action="store_true",
        help="If set, sweep shock_edge over many edges and save a CSV summary (can take a long time).",
    )
    parser.add_argument(
        "--edge_sweep_mode",
        type=str,
        default="all",
        choices=["all", "candidate"],
        help="Edge sweep mode: 'all' sweeps all network edges; 'candidate' restricts to edges on the k-shortest-path choice set.",
    )
    parser.add_argument(
        "--only_edge",
        action="store_true",
        help="If set, only run Sioux Falls edge experiments (E1 and/or sweep) and skip other experiments.",
    )
    args, _ = parser.parse_known_args()

    print("="*70)
    print("DTD PAPER EXPERIMENT RUNNER")
    print("Running all experiments for main.tex")
    print("="*70)

    # Check dependencies
    try:
        import numpy
        import matplotlib
        import networkx
    except ImportError as e:
        print(f"ERROR: Missing dependency: {e}")
        print("Please install: pip install numpy matplotlib networkx")
        sys.exit(1)

    # Run all experiments
    if bool(args.only_edge):
        if bool(args.run_edge_sens):
            run_sioux_falls_edge_sensitivity(n_seeds=int(args.n_seeds), seed0=int(args.seed))
        if bool(args.run_edge_sweep):
            run_sioux_falls_edge_sweep(n_seeds=int(args.n_seeds), seed0=int(args.seed), mode=str(args.edge_sweep_mode))

        print("\n" + "="*70)
        print("ALL EXPERIMENTS COMPLETED")
        print("="*70)
        return

    run_toy_experiments()
    run_toy_ablation_experiments()
    run_sioux_falls_experiments()
    run_sioux_falls_ablation_experiments()
    run_sioux_falls_br_sensitivity(n_seeds=int(args.n_seeds), seed0=int(args.seed))
    if bool(args.run_edge_sens):
        run_sioux_falls_edge_sensitivity(n_seeds=int(args.n_seeds), seed0=int(args.seed))
    if bool(args.run_edge_sweep):
        run_sioux_falls_edge_sweep(n_seeds=int(args.n_seeds), seed0=int(args.seed), mode=str(args.edge_sweep_mode))
    run_sioux_falls_heatmap()
    run_paper_figures()

    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETED")
    print("="*70)
    print("\nGenerated figure folders:")
    print("  - figs_shock/          : Toy network default settings")
    print("  - figs_shock2/         : Toy network sensitivity settings")
    print("  - figs_ablation_toy_*/ : Toy ablation experiments")
    print("  - figs_mech_toy_*/     : Toy mechanism experiments")
    print("  - figs_sf2/            : Sioux Falls S0/S1/S2")
    print("  - figs_sf_*/           : Sioux Falls G/N/GN/R")
    print("  - figs_ablation_sf_*/  : Sioux Falls ablation")
    print("  - figs_mech_sf_*/      : Sioux Falls mechanism")
    print("  - figs_sf_br_*/        : Sioux Falls bounded rationality sensitivity")
    print("  - figs_sf_br_summary/  : Sioux Falls bounded rationality CSV summaries")
    print("  - figs_sf_edge_summary/: Sioux Falls shock-edge robustness CSV summaries")
    print("  - figs_sf_heatmap_demo/: Sioux Falls phase diagram")
    print("  - figs_paper/          : Paper framework figures")


if __name__ == "__main__":
    main()
