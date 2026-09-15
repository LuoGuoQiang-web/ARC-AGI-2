#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Audit the corpus for demonstration pairs whose ground truth contradicts the task's own rule.

Why this exists: the competition forum reports that training task `17829a00` has a corrupted
first demonstration -- every column except one conserves its colour count exactly, and that one
column "gains two cells out of nowhere" (topic 698462). The thread's conclusion is that solvers
which filter candidates by "must reproduce all demonstration pairs" are *actively harmed* by it,
because the correct answer violates the demos.

That rule is one of this repository's stated conventions (AGENTS.md: "Every solver candidate must
reproduce all demonstration pairs before it may vote"), used by the symbolic engine in Stage C.
So the question is not academic: does the corruption reach the evaluation or test splits, where
it would mislead the model at inference rather than merely during training?

This checks every task in every split for the cheapest observable symptom of a bad demo -- a
demonstration whose output is *inconsistent with the others* under an exactly-repeating shape
rule -- and reports where the known-bad task actually lives.

Run:  python tools/audit_demo_integrity.py [comp_data]
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

KNOWN_BAD = "17829a00"


def load(d: Path, name: str):
    p = d / name
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def shape_pairs(task: dict):
    """(input shape, output shape) for each demonstration pair."""
    out = []
    for p in task.get("train") or []:
        try:
            out.append((len(p["input"]), len(p["input"][0]),
                        len(p["output"]), len(p["output"][0])))
        except Exception:
            out.append(None)
    return out


def main() -> int:
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "comp_data")
    splits = {
        "training": load(d, "arc-agi_training_challenges.json"),
        "evaluation": load(d, "arc-agi_evaluation_challenges.json"),
        "test(public)": load(d, "arc-agi_test_challenges.json"),
    }
    if not any(splits.values()):
        print(f"no data under {d}; see paper/README.md")
        return 1

    print("=" * 78)
    print(f"where does the known-corrupted task {KNOWN_BAD} appear?")
    print("=" * 78)
    for name, dd in splits.items():
        if KNOWN_BAD in dd:
            t = dd[KNOWN_BAD]
            print(f"  {name:14} PRESENT -- {len(t['train'])} demos, {len(t['test'])} test inputs")
        else:
            print(f"  {name:14} absent")
    print()
    print("  If it is absent from evaluation and test, the corruption cannot mislead the model at")
    print("  inference; it is a training-data defect only. If it is present, it does.")

    print()
    print("=" * 78)
    print("splits where the demonstration SHAPE rule is inconsistent within a task")
    print("=" * 78)
    print("  A task whose demos disagree on (in_shape -> out_shape) is either genuinely")
    print("  shape-dependent or contains a bad demo. Counted, not adjudicated.")
    for name, dd in splits.items():
        if not dd:
            continue
        inconsistent = []
        for tid, task in dd.items():
            sp = shape_pairs(task)
            if len(sp) < 2 or any(x is None for x in sp):
                continue
            if len(set(sp)) > 1:
                inconsistent.append(tid)
        frac = 100 * len(inconsistent) / max(1, len(dd))
        flag = "  <-- includes the known-bad task" if KNOWN_BAD in inconsistent else ""
        print(f"  {name:14} {len(inconsistent):4}/{len(dd):5} tasks ({frac:5.1f}%){flag}")
        if tid_show := (inconsistent[:3] if inconsistent else []):
            print(f"                 e.g. {tid_show}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
