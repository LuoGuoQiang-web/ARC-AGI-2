"""
Measure spurious program acceptance (leave-one-out).

The v1 acceptance rule is: a chain is accepted if it reproduces every demonstration.
On real tasks with only 3-4 demonstrations that rule is weak — a chain can pass by luck.

This probe quantifies it on real ARC-AGI-2 data:
  * fit the chain on all demonstrations EXCEPT one (leave-one-out)
  * a chain that still reproduces the fitted demonstrations is "accepted"
  * then check the held-out demonstration
  * acceptance precision = fraction of accepted chains that also get the held-out pair right

Low precision means the current acceptance rule mostly admits coincidence, which is both
the explanation for the 0% baseline and the empirical case for a leave-one-out guard.

Usage:  python arc_prize_v1/tests/loo_probe.py [evaluation|training]
"""

from __future__ import annotations

import collections
import importlib.util
import json
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
ROOT_WS = ROOT.parent
ENGINE = ROOT / "arc_prize_v1.py"

spec = importlib.util.spec_from_file_location("arc_prize_v1", ENGINE)
arc = importlib.util.module_from_spec(spec)
sys.modules["arc_prize_v1"] = arc
spec.loader.exec_module(arc)


def all_chains(bg: int):
    prims = arc.build_primitives(bg)
    by = dict(prims)
    out = [[(n, f)] for n, f in prims]
    names = [n for n in arc.D2_NAMES if n in by]
    for a in names:
        for b in names:
            out.append([(a, by[a]), (b, by[b])])
    return out


def loo_task(task, max_splits: int = 3):
    """Leave-one-out over the demonstrations. Returns (accepted, correct, splits)."""
    pairs = task["train"]
    if len(pairs) < 3:
        return 0, 0, 0
    accepted = correct = splits = 0
    # hold out each pair in turn (cap the work on tasks with many pairs)
    for hold_idx in range(min(len(pairs), max_splits)):
        fit = [p for i, p in enumerate(pairs) if i != hold_idx]
        hold = pairs[hold_idx]
        bg = arc.infer_bg(fit)
        tin = [p["input"] for p in fit]
        tout = [p["output"] for p in fit]
        h_in, h_out = hold["input"], hold["output"]
        splits += 1
        for chain in all_chains(bg):
            transformed = []
            ok = True
            for g in tin:
                tg = arc.run_chain(chain, g)
                if tg is None:
                    ok = False
                    break
                transformed.append(tg)
            if not ok:
                continue
            cmap = arc.fit_color_map(transformed, tout)
            if cmap is None:
                continue
            accepted += 1
            pred = arc.run_chain(chain, h_in)
            if pred is not None and arc.grids_equal(arc.apply_color_map(pred, cmap), h_out):
                correct += 1
    return accepted, correct, splits


def main(dataset: str = "evaluation") -> int:
    base = ROOT_WS / "ARC-AGI-2-main" / "data" / dataset
    if not base.is_dir():
        print(f"missing data dir: {base}")
        return 2
    files = sorted(base.glob("*.json"))
    print(f"dataset={dataset}  tasks={len(files)}")

    stats = collections.Counter()
    per_task_precision = []
    t0 = time.monotonic()
    for f in files:
        task = json.loads(f.read_text(encoding="utf-8"))
        accepted, correct, splits = loo_task(task)
        stats["splits"] += splits
        stats["accepted"] += accepted
        stats["correct"] += correct
        if accepted:
            stats["tasks_with_acceptance"] += 1
            per_task_precision.append(correct / accepted)
        if correct:
            stats["tasks_with_a_truly_correct_chain"] += 1

    dt = time.monotonic() - t0
    acc, cor = stats["accepted"], stats["correct"]
    print(f"\nleave-one-out over demonstrations ({stats['splits']} splits, {dt:.0f}s)")
    print(f"  accepted chains (reproduced the fitted demos) : {acc}")
    print(f"  of which also correct on the held-out demo    : {cor}")
    precision = cor / acc if acc else 0.0
    print(f"  >>> acceptance precision                      : {precision*100:.2f}%")
    print(f"  tasks with >=1 accepted chain                 : {stats['tasks_with_acceptance']}/{len(files)}")
    print(f"  tasks with >=1 truly correct chain            : {stats['tasks_with_a_truly_correct_chain']}/{len(files)}")
    if per_task_precision:
        per_task_precision.sort()
        mid = per_task_precision[len(per_task_precision) // 2]
        print(f"  per-task precision: median {mid*100:.1f}%  "
              f"mean {sum(per_task_precision)/len(per_task_precision)*100:.1f}%  "
              f"max {max(per_task_precision)*100:.1f}%")
    return 0


if __name__ == "__main__":
    ds = sys.argv[1] if len(sys.argv) > 1 else "evaluation"
    sys.exit(main(ds))
