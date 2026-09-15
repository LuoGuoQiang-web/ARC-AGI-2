#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""How often did test-time training actually execute, across every run we have?

The paper's central claim rests on one 240-task run. That is a single observation, and a reviewer
is right to ask whether it was a one-off. This tool answers that from the artefacts already on
disk -- the committed per-task rows and the kernel logs -- so the answer costs no GPU and can be
re-derived by anyone.

It reports, per run: tasks processed, tasks routed to the TTT stage, how many of those executed at
least one optimizer step, the total optimizer steps, out-of-memory events, and whether the
run-wide circuit breaker fired.

Run:  python tools/measure_ttt_execution.py [kaggle_out]
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

OOM_RE = re.compile(r"TTT step \d+ failed: CUDA out of memory")
BREAKER_RE = re.compile(r"! TTT disabled for the rest of the run")
ROUTED_RE = re.compile(r"\[stage B[^\]]*\]")


def summarise(run_dir: Path) -> dict | None:
    rep_p = run_dir / "report.json"
    if not rep_p.exists():
        return None
    try:
        rep = json.loads(rep_p.read_text(encoding="utf-8"))
    except Exception:
        return None
    per = rep.get("per_task") or []
    if not per:
        return None
    steps = [int(e.get("ttt_steps") or 0) for e in per]
    routed = [e for e in per
              if any(str(s).startswith("B_ttt") for s in (e.get("stages") or []))]
    ttt_block = rep.get("ttt") or {}

    log_p = run_dir / "kernel_stdout.log"
    text = log_p.read_text(encoding="utf-8", errors="replace") if log_p.exists() else ""

    return {
        "run": run_dir.name,
        "tasks": len(per),
        "routed": ttt_block.get("tasks_routed", len(routed)),
        "executed": sum(1 for s in steps if s > 0),
        "steps": sum(steps),
        "oom": len(OOM_RE.findall(text)),
        "breaker": bool(BREAKER_RE.search(text)),
        "instrumented": all("ttt_steps" in e for e in per),
    }


def main() -> int:
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "kaggle_out"
    rows = [r for r in (summarise(d) for d in sorted(root.iterdir()) if d.is_dir()) if r]
    if not rows:
        print(f"no reports under {root}")
        return 1

    w = max(len(r["run"]) for r in rows)
    print(f"{'run'.ljust(w)}  {'tasks':>6} {'routed':>7} {'ran':>5} {'steps':>6} "
          f"{'OOM':>4} {'breaker':>8}")
    print("-" * (w + 44))
    for r in rows:
        print(f"{r['run'].ljust(w)}  {r['tasks']:>6} {r['routed']:>7} {r['executed']:>5} "
              f"{r['steps']:>6} {r['oom']:>4} {str(r['breaker']):>8}")

    tasks = sum(r["tasks"] for r in rows)
    routed = sum(r["routed"] for r in rows)
    executed = sum(r["executed"] for r in rows)
    steps = sum(r["steps"] for r in rows)
    oom = sum(r["oom"] for r in rows)
    broke = sum(1 for r in rows if r["breaker"])

    print()
    print(f"runs                     : {len(rows)}")
    print(f"tasks processed          : {tasks}")
    print(f"tasks routed to TTT      : {routed}")
    print(f"...that executed >=1 step: {executed}"
          + (f"  ({100 * executed / routed:.0f}% of routed)" if routed else ""))
    print(f"total optimizer steps    : {steps}")
    print(f"OOM events               : {oom}")
    print(f"runs where the breaker fired: {broke}/{len(rows)}")
    print()
    if executed == 0:
        print("VERDICT: in no run did any task execute a single optimizer step. Every 'TTT no gain'")
        print("         verdict in this repository describes a stage that never ran.")
    elif routed and executed / routed < 0.5:
        print(f"VERDICT: TTT executed on {executed}/{routed} routed tasks ({100 * executed / routed:.0f}%).")
        print("         The pattern is not a one-off: in run after run, most tasks recorded against")
        print("         the adaptation stage never took an optimizer step, so their 'no gain'")
        print("         verdicts carry no information about adaptation.")
    else:
        print("VERDICT: TTT executed on most routed tasks; the masked-failure pattern is NOT present.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
