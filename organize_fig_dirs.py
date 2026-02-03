import argparse
import os
import re
import shutil
from pathlib import Path
from typing import Iterable, Set


def _find_tex_referenced_dirs(tex_path: Path) -> Set[str]:
    if not tex_path.exists():
        return set()

    text = tex_path.read_text(encoding="utf-8", errors="ignore")
    # capture folder name after ../figs* or ../outputs/figs*
    matches = []
    matches += re.findall(r"\.\./(figs[^/\s\}]+)", text)
    matches += re.findall(r"\.\./outputs/(figs[^/\s\}]+)", text)
    out = set()
    for m in matches:
        out.add(m.replace("\\_", "_"))
    return out


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


def _iter_candidate_dirs(root: Path) -> Iterable[Path]:
    for p in root.iterdir():
        if not p.is_dir():
            continue
        n = p.name
        if n.startswith("figs") or n.startswith("tmp_"):
            yield p


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, default=".")
    parser.add_argument("--dest", type=str, default="outputs")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--no_tex_scan", action="store_true")
    parser.add_argument("--keep_tex_referenced", action="store_true")
    parser.add_argument("--keep", action="append", default=None)
    args = parser.parse_args()

    root = Path(args.root).resolve()
    dest = (root / args.dest).resolve() if not os.path.isabs(args.dest) else Path(args.dest).resolve()

    keep_set = set(args.keep or [])

    referenced = set()
    if not args.no_tex_scan:
        referenced |= _find_tex_referenced_dirs(root / "Day2Day" / "main.tex")

    candidates = list(_iter_candidate_dirs(root))

    to_move = []
    kept = []
    for d in sorted(candidates, key=lambda x: x.name):
        if (args.keep_tex_referenced and d.name in referenced) or (d.name in keep_set):
            kept.append(d)
        else:
            to_move.append(d)

    print(f"root={root}")
    print(f"dest={dest}")
    if referenced:
        print("tex_referenced_dirs=" + ",".join(sorted(referenced)))

    print("\nkept_dirs=")
    for d in kept:
        print(f"  - {d.name}")

    print("\nmove_dirs=")
    for d in to_move:
        print(f"  - {d.name}")

    if not args.apply:
        print("\n(dry-run) use --apply to actually move directories")
        return

    dest.mkdir(parents=True, exist_ok=True)

    for d in to_move:
        target = _unique_dest(dest, d.name)
        print(f"moving {d} -> {target}")
        shutil.move(str(d), str(target))


if __name__ == "__main__":
    main()
