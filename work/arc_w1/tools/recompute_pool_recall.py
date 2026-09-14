#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Recompute pool recall from the stored per-task rows, without a GPU.

Why this exists: the shipped `report["pool_recall"]` block used to be accumulated inside
`record_task`, which runs once per *stage* rather than once per task. A task solved in
Stage A and then revisited in Stages B and C had its test inputs counted three times, so
the published denominator disagreed with the `per_task` rows it summarises (measured on
`arc26-exp-base`: 16 reported, 11 real). The accumulator is now replaced by a recompute
from the unique per-task entries, but the reports already on disk still carry the inflated
numbers -- this tool re-derives the honest ones from the per-task rows, which were always
correct.

Run:  python tools/recompute_pool_recall.py [kaggle_out]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def summarise(path: Path) -> dict | None:
    try:
        rep = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    per = rep.get("per_task") or []
    if not per:
        return None
    # A report written before the pool-recall instrumentation simply lacks the field.
    # Reading that absence as "0 in pool" would manufacture a false confirmation of the
    # very hypothesis under test, so reports without the field are marked uninstrumented
    # and excluded from every recall claim.
    instrumented = [e for e in per if "truth_in_pool" in e]
    wt = sum(int(e.get("n_with_truth") or 0) for e in instrumented)
    ip = sum(int(e.get("truth_in_pool") or 0) for e in instrumented)
    nd = [int(e.get("n_distinct") or 0) for e in per]
    nc = [int(e.get("n_candidates") or 0) for e in per]
    ttt = [int(e.get("ttt_steps") or 0) for e in per]
    claimed = (rep.get("pool_recall") or {}).get("n_with_truth")
    return {
        "slug": path.parent.name,
        "tasks": len(per),
        "instrumented": len(instrumented),
        "with_truth": wt,
        "in_pool": ip,
        "recall": (ip / wt) if wt else None,
        "distinct": f"{min(nd)}-{max(nd)}",
        "cands": f"{min(nc)}-{max(nc)}",
        "ttt_tasks": sum(1 for s in ttt if s > 0),
        "claimed": claimed,
        "agree": (claimed == wt) if claimed is not None else None,
    }


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "kaggle_out"
    rows = [s for s in (summarise(p) for p in sorted(root.glob("*/report.json"))) if s]
    if not rows:
        print(f"no reports under {root}")
        return 1
    hdr = (f"{'run':<22}{'tasks':>6}{'instr':>6}{'inputs':>8}{'in_pool':>9}{'recall':>9}"
           f"{'distinct':>10}{'cands':>9}{'ttt>0':>7}{'claimed':>9}{'ok':>5}")
    print(hdr)
    print("-" * len(hdr))
    tot_wt = tot_ip = 0
    for r in rows:
        if r["instrumented"] == 0:
            rec, ip = "n/a", "-"
        else:
            rec = "-" if r["recall"] is None else f"{r['recall']:.4f}"
            ip = r["in_pool"]
            tot_wt += r["with_truth"]
            tot_ip += r["in_pool"]
        ok = "-" if r["agree"] is None else ("yes" if r["agree"] else "NO")
        print(f"{r['slug']:<22}{r['tasks']:>6}{r['instrumented']:>6}{r['with_truth']:>8}{str(ip):>9}"
              f"{rec:>9}{r['distinct']:>10}{r['cands']:>9}{r['ttt_tasks']:>7}"
              f"{str(r['claimed']):>9}{ok:>5}")
    if tot_wt:
        print(f"\nINSTRUMENTED TOTAL: {tot_ip}/{tot_wt} test inputs had the truth in the pool "
              f"(recall {tot_ip / tot_wt:.4f})")
    print("`inputs` = test inputs carrying ground truth (the honest denominator).")
    print("`n/a` = report predates the pool-recall instrumentation; excluded from the total.")
    print("`ok` = does the report's own aggregate match its per-task rows?")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
