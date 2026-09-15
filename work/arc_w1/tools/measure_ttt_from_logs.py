#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Count TTT execution from the KERNEL LOGS, which cannot be overwritten.

`report.json` turned out to be an unreliable source for this: `record_task` runs once per stage
(A, then B, then C) and assigns `entry["ttt_steps"] = int(res.ttt_steps)`, so a Stage C re-record
of the same task clobbers whatever Stage B measured. On `arc26-exp-base` the log plainly shows two
tasks completing two optimizer steps each, while the report claims zero tasks stepped. That is the
same failure mode as the pool-recall denominator bug: a per-stage write destroying a per-task fact.

The log line `[<task>] TTT <n> steps in <t>s` is emitted once per TTT attempt by `run_ttt`'s
caller, so counting it gives attempts and total steps without depending on the report at all.

Run:  python tools/measure_ttt_execution.py --from-logs
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

ATTEMPT_RE = re.compile(r"TTT (\d+) steps in ([\d.]+)s")


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "kaggle_out"
    logs = sorted(root.glob("*/kernel_stdout.log"))
    if not logs:
        print(f"no kernel logs under {root}")
        return 1

    w = max(len(p.parent.name) for p in logs)
    print(f"{'run'.ljust(w)}  {'attempts':>8} {'ran':>5} {'zero':>5} {'steps':>6} "
          f"{'OOM':>4} {'breaker':>8}")
    print("-" * (w + 44))
    grand = {"att": 0, "ran": 0, "zero": 0, "steps": 0, "oom": 0, "brk": 0}
    for p in logs:
        text = p.read_text(encoding="utf-8", errors="replace")
        pairs = [(int(n), float(t)) for n, t in ATTEMPT_RE.findall(text)]
        if not pairs:
            continue
        ran = sum(1 for n, _ in pairs if n > 0)
        zero = sum(1 for n, _ in pairs if n == 0)
        steps = sum(n for n, _ in pairs)
        oom = len(re.findall(r"TTT step \d+ failed: CUDA out of memory", text))
        brk = bool(re.search(r"! TTT disabled for the rest of the run", text))
        grand["att"] += len(pairs)
        grand["ran"] += ran
        grand["zero"] += zero
        grand["steps"] += steps
        grand["oom"] += oom
        grand["brk"] += int(brk)
        print(f"{p.parent.name.ljust(w)}  {len(pairs):>8} {ran:>5} {zero:>5} {steps:>6} "
              f"{oom:>4} {str(brk):>8}")

    print()
    print(f"TTT attempts across all runs     : {grand['att']}")
    print(f"...that executed >= 1 step       : {grand['ran']}")
    print(f"...that executed ZERO steps      : {grand['zero']}"
          + (f"  ({100 * grand['zero'] / grand['att']:.0f}%)" if grand["att"] else ""))
    print(f"total optimizer steps            : {grand['steps']}")
    print(f"mean steps per executed attempt  : "
          f"{(grand['steps'] / grand['ran']):.2f}" if grand["ran"] else "")
    print(f"OOM events                       : {grand['oom']}")
    print()
    print("Read this as: of every attempt the pipeline made at adapting to a task, this fraction")
    print("never took a single optimizer step -- while still being recorded as a TTT result.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
