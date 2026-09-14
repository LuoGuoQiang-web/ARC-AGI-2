#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
exp_suite.py -- run a controlled comparison of solver configurations and print one table.

Why: the binding constraint is pool recall (on the 2026-09-14 eval probe the truth was
never generated -- `selection_headroom = 0.0`), and the only levers that can move it are the
DFS search-width knobs. A GPU-period is 6 h and a full submission costs 8.5 h, so every
experiment has to be small, comparable and self-summarising -- that is this tool.

Fair-comparison rules it enforces:
  * every config runs the SAME tasks: `--limit N` takes the first N of the *sorted* task
    ids, so identical N == identical subset;
  * identical time budget and augmentation counts, only the tested knob differs;
  * one kernel per config, all pinned to machine_shape=NvidiaTeslaT4 (see kpush.py).

Usage
-----
    # build the notebooks only, inspect them, spend nothing
    python tools/exp_suite.py --dry-run

    # run the default 2-config coverage probe (~0.9 h GPU) and print the table
    python tools/exp_suite.py

    # widen the sweep once the direction is known
    python tools/exp_suite.py --configs base,prob10,branch4,combined

    # re-print the table from reports already downloaded (no GPU, no network)
    python tools/exp_suite.py --collect-only
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARC_W1 = HERE.parent
sys.path.insert(0, str(HERE))
import kpush as K  # noqa: E402

OUT_ROOT = ARC_W1 / "kaggle_out"
BUILD_ROOT = ARC_W1 / ".kpush_build"

# Controlled comparison. BASE is shared by every config; each config adds only its knob.
BASE = "--split evaluation --limit {limit} --time-budget-seconds {budget} " \
       "--aug-train 2 --aug-infer 2 --seed 0"

PRESETS: dict[str, str] = {
    # shipped defaults: prune below p=0.2, <=3 children per beam, 6000 nodes per DFS
    "base": "",
    # the hypothesis: the correct token often ranks below 0.2 and is pruned forever
    "prob10": "--dfs-prob-threshold 0.1",
    # widen the beam instead of the threshold
    "branch4": "--dfs-max-branches 4 --dfs-max-nodes 12000",
    # both, to see whether they compose
    "combined": "--dfs-prob-threshold 0.1 --dfs-max-branches 4 --dfs-max-nodes 12000",
}


def build_and_push(name: str, argv: str, dry_run: bool,
                   extras: list[tuple[str, str]] | None = None) -> str:
    slug = f"arc26-exp-{name}"
    solver_text = (ARC_W1 / "arc26_solver.py").read_text(encoding="utf-8")
    extras = extras or []
    nb = K.build_notebook(solver_text, argv, extras)
    meta = K.build_metadata(slug, slug.replace("-", " "), True, True, False,
                            K.COMPETITION, K.MODEL_SOURCE, "NvidiaTeslaT4")
    build = BUILD_ROOT / slug
    build.mkdir(parents=True, exist_ok=True)
    (build / f"{slug}.ipynb").write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")
    (build / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"  built {slug}: {argv}")
    if extras:
        print(f"    + {len(extras)} extra file cell(s): {[r for r, _ in extras]}")
    if not dry_run:
        K.get_api().kernels_push(str(build))
        print(f"  pushed {K.OWNER}/{slug}")
    return slug


def wait_for_report(slug: str, timeout_s: float, poll_s: float = 60.0) -> Path | None:
    """Poll the kernel output until report.json lands (a successful run publishes no log)."""
    ref = f"{K.OWNER}/{slug}"
    dest = OUT_ROOT / slug
    dest.mkdir(parents=True, exist_ok=True)
    api = K.get_api()
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            api.kernels_output(ref, path=str(dest), force=True, quiet=True)
        except Exception as exc:
            print(f"    poll: {type(exc).__name__} (still running?)")
        rep = dest / "report.json"
        if rep.exists():
            print(f"    report after {time.time() - t0:.0f}s")
            return rep
        time.sleep(poll_s)
    print(f"    TIMEOUT after {timeout_s:.0f}s; no report.json")
    return None


def metrics(slug: str, rep_path: Path) -> dict:
    rep = json.loads(rep_path.read_text(encoding="utf-8"))
    pr = rep.get("pool_recall") or {}
    stages = {s.get("name"): s for s in (rep.get("stages") or [])}
    per_task = rep.get("per_task") or []
    ttt_steps = sum(int(t.get("ttt_steps") or 0) for t in per_task)
    dfs = rep.get("dfs") or {}
    return {
        "config": slug.replace("arc26-exp-", ""),
        "scored": rep.get("n_tasks_scored") or 0,
        "solved": rep.get("n_solved") or 0,
        "acc": rep.get("accuracy"),
        "recall": pr.get("pool_recall"),
        "headroom": pr.get("selection_headroom"),
        "in_pool": pr.get("n_in_pool"),
        "with_truth": pr.get("n_with_truth"),
        "A_s": round((stages.get("A_sweep") or {}).get("seconds_per_task") or 0.0, 1),
        "B_tasks": (stages.get("B_ttt") or {}).get("tasks"),
        "ttt_steps": ttt_steps,
        "dfs": f"{dfs.get('prob_threshold')}/{dfs.get('max_branches')}/{dfs.get('max_nodes')}",
        "elapsed_s": round(rep.get("elapsed_seconds") or 0.0, 0),
        "peak_gb": rep.get("peak_mem_gb"),
        "engine": (rep.get("engine") or {}).get("status"),
        "eng_valid": (rep.get("engine") or {}).get("n_tasks_validated"),
        "eng_prior": (rep.get("engine") or {}).get("n_tasks_prior_only"),
    }


def pct(x) -> str:
    return "-" if x is None else f"{100 * float(x):.1f}%"


def print_table(rows: list[dict]) -> None:
    cols = [("config", 10), ("dfs p/b/n", 12), ("scored", 7), ("solved", 7), ("acc", 7),
            ("recall", 8), ("in_pool", 8), ("headroom", 9), ("A s/task", 9),
            ("B tasks", 8), ("ttt steps", 10), ("engine", 14), ("elapsed", 8), ("peak GB", 8)]
    head = "".join(f"{c:<{w}}" for c, w in cols)
    print()
    print("=" * len(head))
    print(head)
    print("-" * len(head))
    for r in rows:
        eng = f"{r['engine']}/v{r['eng_valid']}" if r.get("eng_valid") is not None else str(r["engine"])
        print(f"{r['config']:<10}{r['dfs']:<12}{r['scored']:<7}{r['solved']:<7}"
              f"{pct(r['acc']):<7}{pct(r['recall']):<8}{str(r['in_pool'] if r['in_pool'] is not None else '-'):<8}"
              f"{pct(r['headroom']):<9}{r['A_s']:<9}{str(r['B_tasks']):<8}{r['ttt_steps']:<10}"
              f"{eng:<14}{r['elapsed_s']:<8}{str(r['peak_gb']):<8}")
    print("=" * len(head))
    print("read `recall` against `acc`: that gap IS the addressable loss. If recall stays 0,")
    print("no selection change can help and the lever is the DFS row (p/b/n).")
    print("`engine` shows status/validated-count: v0 means the DSL validated nothing here.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--configs", default="base,prob10",
                    help=f"comma-separated subset of: {','.join(PRESETS)}")
    ap.add_argument("--limit", type=int, default=8, help="tasks per config (same for all)")
    ap.add_argument("--budget", type=int, default=3600, help="time-budget-seconds per config")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--collect-only", action="store_true")
    ap.add_argument("--timeout", type=float, default=3600.0, help="max wait per config (s)")
    ap.add_argument("--engine", default="",
                    help="also ship this symbolic engine and pass --engine to every config "
                         "(constant across configs, so the comparison stays fair; the engine "
                         "costs ~32 ms/task so it rides along for free)")
    args = ap.parse_args()

    extras: list[tuple[str, str]] = []
    engine_arg = ""
    if args.engine:
        ep = Path(args.engine)
        if not ep.exists():
            print(f"--engine not found: {ep}")
            return 2
        remote = f"/kaggle/working/{ep.name}"
        extras.append((remote, ep.read_text(encoding="utf-8")))
        engine_arg = f"--engine {remote}"

    names = [n.strip() for n in args.configs.split(",") if n.strip()]
    unknown = [n for n in names if n not in PRESETS]
    if unknown:
        print(f"unknown config(s): {unknown}; known: {sorted(PRESETS)}")
        return 2

    print(f"suite: {names}  | limit={args.limit} tasks  budget={args.budget}s each")
    est = len(names) * (args.limit * 190 + 120) / 3600.0
    print(f"estimated GPU cost: ~{est:.2f} h  (runs would be serialised below)")

    rows = []
    for name in names:
        slug = f"arc26-exp-{name}"
        argv = (BASE.format(limit=args.limit, budget=args.budget)
                + (" " + PRESETS[name] if PRESETS[name] else "")
                + (" " + engine_arg if engine_arg else ""))
        if args.collect_only:
            rep = OUT_ROOT / slug / "report.json"
            if rep.exists():
                rows.append(metrics(slug, rep))
            else:
                print(f"  {slug}: no downloaded report")
            continue
        build_and_push(name, argv, args.dry_run, extras)
        if args.dry_run:
            continue
        rep = wait_for_report(slug, args.timeout)
        if rep:
            rows.append(metrics(slug, rep))

    if rows:
        print_table(rows)
    if args.dry_run:
        print(f"\n--dry-run: notebooks built under {BUILD_ROOT}; nothing pushed, no GPU spent.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
