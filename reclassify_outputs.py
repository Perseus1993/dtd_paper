import argparse
import os
import shutil
from pathlib import Path
from typing import Dict, Iterable, Optional


def _iter_dirs(p: Path) -> Iterable[Path]:
    if not p.exists():
        return
    for child in p.iterdir():
        if child.is_dir():
            yield child


def _unique_dest(dest_dir: Path, name: str) -> Path:
    cand = dest_dir / name
    if not cand.exists():
        return cand
    i = 1
    while True:
        cand = dest_dir / f"{name}__dup{i}"
        if not cand.exists():
            return cand
        i += 1


def _classify(name: str) -> str:
    n = str(name)

    if n.startswith("tmp_"):
        return "tmp"

    if n == "figs_paper":
        return "paper"

    if (
        n.startswith("figs_shock")
        or n.startswith("figs_toy")
        or n.startswith("figs_mech_toy")
        or n.startswith("figs_ablation_toy")
    ):
        return "toy"

    if (
        n.startswith("figs_sf")
        or n.startswith("figs_mech_sf")
        or n.startswith("figs_ablation_sf")
    ):
        return "sioux"

    if n.startswith("figs_") and ("_spatial" in n) and (not n.startswith("figs_sf")):
        return "city"

    if n.startswith("figs_hiroshima"):
        return "city"

    return "legacy"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default="outputs")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--dest_map", type=str, default=None)
    args = parser.parse_args()

    repo_root = Path(".").resolve()
    root = Path(args.root)
    if not root.is_absolute():
        root = (repo_root / root).resolve()

    dest_map: Dict[str, str] = {
        "paper": "paper",
        "toy": "toy",
        "sioux": "sioux",
        "city": "city",
        "tmp": os.path.join("tmp"),
        "legacy": os.path.join("legacy"),
    }

    if args.dest_map:
        # format: category=path,category=path
        for item in str(args.dest_map).split(","):
            if not item.strip():
                continue
            k, v = item.split("=", 1)
            dest_map[k.strip()] = v.strip()

    moves = []
    kept = []

    for d in sorted(_iter_dirs(root), key=lambda p: p.name):
        if d.name in ("paper", "toy", "sioux", "city", "tmp", "legacy"):
            kept.append(d)
            continue
        if not (d.name.startswith("figs") or d.name.startswith("tmp_")):
            kept.append(d)
            continue

        cat = _classify(d.name)
        rel = dest_map.get(cat, "legacy")
        target_parent = root / rel
        target = _unique_dest(target_parent, d.name)
        moves.append((d, target, cat))

    print(f"outputs_root={root}")
    print("\nplanned_moves=")
    for src, dst, cat in moves:
        print(f"  - {src.name}  ->  {dst.relative_to(root)}  (category={cat})")

    if not args.apply:
        print("\n(dry-run) use --apply to actually move directories")
        return

    for src, dst, _cat in moves:
        dst.parent.mkdir(parents=True, exist_ok=True)
        print(f"moving {src} -> {dst}")
        shutil.move(str(src), str(dst))


if __name__ == "__main__":
    main()
