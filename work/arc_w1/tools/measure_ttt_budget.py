#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Measure what the TTT context budget actually does to optimizer-step length.

Why this exists: the shipped TTT step packed the *entire* demonstration set into one
sequence, which for large grids builds an ~8192-token step. A backward pass over that on a
3.63B model does not fit a 14.56 GiB T4 alongside the search's residual caches, and the
failure is silent -- the task is recorded exactly like one whose adaptation ran and did not
help. Measured on the 240-task submission run: 102 tasks were routed to TTT and the run
performed 121 optimizer steps in total, a mean of 1.34 per task against the reference
recipe's 128.

This tool says how much shorter the steps get, per task, for a given budget. It runs on CPU
with the fake tokenizer, so it costs nothing and needs no GPU.

Run:  python tools/measure_ttt_budget.py [--budget 2048] [--aug 4] [comp_data]
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # work/arc_w1
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "tests"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=2048, help="context budget to evaluate")
    ap.add_argument("--aug", type=int, default=4, help="augmentations (sequences per task)")
    ap.add_argument("--data", default=str(ROOT / "comp_data"))
    ap.add_argument("--top", type=int, default=10, help="worst tasks to print")
    args = ap.parse_args()

    import importlib.util
    spec = importlib.util.spec_from_file_location("S", str(ROOT / "arc26_solver.py"))
    S = importlib.util.module_from_spec(spec)
    sys.modules["S"] = S
    spec.loader.exec_module(S)
    import test_model_free as T          # FakeTokenizer, built from the recovered vocabulary

    path = Path(args.data) / "arc-agi_evaluation_challenges.json"
    if not path.exists():
        print(f"missing {path}; run: kaggle competitions download -c "
              f"arc-prize-2026-arc-agi-2 ... see paper/README.md")
        return 1
    data = json.loads(path.read_text(encoding="utf-8"))
    tok = T.FakeTokenizer()

    rows = []
    for tid, task in sorted(data.items()):
        unc, _ = S.build_ttt_sequences(task["train"], tok, args.aug, 8192, random.Random(0))
        cap, _ = S.build_ttt_sequences(task["train"], tok, args.aug, 8192, random.Random(0),
                                       seq_token_budget=args.budget)
        if not unc or not cap:
            continue
        rows.append((max(len(s) for s in unc), max(len(s) for s in cap),
                     sum(len(s) for s in unc), sum(len(s) for s in cap),
                     tid, len(task["train"])))

    if not rows:
        print("no encodable tasks")
        return 1

    rows.sort(key=lambda r: -r[0])
    print(f"budget={args.budget}  aug={args.aug}  tasks={len(rows)}  (max sequence length "
          f"per optimizer step)\n")
    print(f"{'uncapped':>10}{'capped':>10}{'ratio':>8}   task       demos")
    print("-" * 46)
    for u, c, _su, _sc, tid, d in rows[:args.top]:
        print(f"{u:>10}{c:>10}{u / max(c, 1):>7.2f}x   {tid}   {d}")

    print(f"\nuncapped: median {statistics.median(r[0] for r in rows):.0f}, "
          f"max {max(r[0] for r in rows)}")
    print(f"capped  : median {statistics.median(r[1] for r in rows):.0f}, "
          f"max {max(r[1] for r in rows)}")
    bound = sum(1 for r in rows if r[1] < r[0])
    print(f"\nthe cap binds on {bound}/{len(rows)} tasks ({100 * bound / len(rows):.0f}%)")
    still = [r for r in rows if r[1] > args.budget]
    print(f"tasks still above the budget after dropping all context: {len(still)}"
          + (f" (largest {max(r[1] for r in still)})" if still else ""))
    print("\nsteps do NOT shrink to the budget when a single grid pair already exceeds it;")
    print("those tasks are exactly where the retry ladder in run_ttt has to earn its keep.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
