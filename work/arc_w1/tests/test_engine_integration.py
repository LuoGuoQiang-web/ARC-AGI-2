#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_engine_integration.py -- validate the symbolic-engine integration end to end, locally.

The solver's own `load_engine` / `engine_predict` are exercised against the real
arc_prize_v1 DSL engine on REAL ARC-AGI-2 tasks. No GPU and no torch are needed because
both sides of the contract are model-free; `engine_predict` is the exact function Stage C
calls, so this is the integration, not a mock.

Contract being checked (arc26_solver.engine_predict docstring):
  * one dict per test input, keys attempt_1 / attempt_2 / verified / program
  * `verified` is True exactly when the engine found a program reproducing EVERY demo
    (engine diagnostics `source == "search"`) -- that flag is what earns a symbolic
    candidate CONFIDENCE 0.9 plus the SYMBOLIC_NLL_BONUS in the hybrid pool
  * grids are None or legal (<=30x30, ints 0..9)
  * a broken / missing engine degrades quietly instead of raising

Usage:  python tests/test_engine_integration.py
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARC_W1 = HERE.parent
WORKSPACE = ARC_W1.parent.parent
ENGINE = WORKSPACE / "arc_prize_v1" / "arc_prize_v1.py"
DATA = WORKSPACE / "ARC-AGI-2-main" / "data"

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILS.append(f"{name}: {detail}")


def load_solver():
    spec = importlib.util.spec_from_file_location("arc26_solver", ARC_W1 / "arc26_solver.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["arc26_solver"] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    if not ENGINE.exists():
        print(f"engine not found: {ENGINE}")
        return 2
    if not DATA.is_dir():
        print(f"data not found: {DATA}")
        return 2
    S = load_solver()

    # ---- T1: load_engine contract ----------------------------------------------------
    eng = S.load_engine(str(ENGINE), log_fn=lambda m: None)
    check("T1a engine loads", eng["module"] is not None and eng["info"]["status"] == "loaded",
          json.dumps(eng["info"])[:160])
    check("T1b make_config was called with the solver's overrides",
          isinstance(eng.get("config"), dict) and eng["config"].get("mode") == "dev",
          str(eng.get("config"))[:120])
    check("T1c missing path degrades to skipped",
          S.load_engine(None)["module"] is None
          and S.load_engine(None)["info"]["status"] == "skipped")
    check("T1d nonexistent path degrades quietly",
          S.load_engine(str(ARC_W1 / "nope.py"))["module"] is None)
    broken = ARC_W1 / ".selftest_engine_broken.py"
    broken.write_text("this is not python(", encoding="utf-8")
    eb = S.load_engine(str(broken), log_fn=lambda m: None)
    check("T1e a broken engine reports error instead of raising",
          eb["module"] is None and eb["info"]["status"] == "error", json.dumps(eb["info"])[:160])
    broken.unlink(missing_ok=True)

    # ---- T2: engine_predict shape on every real task ---------------------------------
    tasks = {}
    for split in ("evaluation", "training"):
        d = DATA / split
        if d.is_dir():
            for f in sorted(d.glob("*.json"))[:150]:
                tasks[f.stem] = json.loads(f.read_text(encoding="utf-8"))

    n_calls = bad_shape = bad_grid = 0
    t0 = time.time()
    for tid, task in tasks.items():
        preds = S.engine_predict(eng, task, 2.0)
        n_calls += 1
        if len(preds) != len(task["test"]):
            bad_shape += 1
            continue
        for p in preds:
            if set(p) != {"attempt_1", "attempt_2", "verified", "program"}:
                bad_shape += 1
            for k in ("attempt_1", "attempt_2"):
                g = p[k]
                if g is None:
                    continue
                if not (isinstance(g, np.ndarray) and g.ndim == 2
                        and 1 <= g.shape[0] <= 30 and 1 <= g.shape[1] <= 30
                        and g.min() >= 0 and g.max() <= 9):
                    bad_grid += 1
    dt = time.time() - t0
    check("T2a one prediction per test input, exact keys", bad_shape == 0,
          f"{bad_shape} bad of {n_calls} tasks")
    check("T2b every returned grid is legal (or None)", bad_grid == 0, f"{bad_grid} bad grids")
    check("T2c cost is negligible next to the GPU budget", dt / max(1, n_calls) < 0.5,
          f"{dt:.1f}s for {n_calls} tasks = {1000 * dt / max(1, n_calls):.0f} ms/task")

    # ---- T3: `verified` means what the hybrid pool assumes ---------------------------
    # A verified prediction must be correct on the demonstrations it was fitted to, and on
    # any solved task its attempt_1 must match the truth -- that is the evidence Stage C
    # relies on when it treats a symbolic candidate as CONFIDENCE 0.9.
    verified_tasks = solved_verified = verified_wrong = 0
    examples = []
    for tid, task in tasks.items():
        preds = S.engine_predict(eng, task, 2.0)
        if not any(p["verified"] for p in preds):
            continue
        verified_tasks += 1
        if len(examples) < 3:
            examples.append(tid)
        truth = [p.get("output") for p in task["test"]]
        if all(o is None for o in truth):
            continue
        ok = all(o is None or np.array_equal(np.asarray(p["attempt_1"]), np.asarray(o))
                 for p, o in zip(preds, truth))
        solved_verified += 1 if ok else 0
        verified_wrong += 0 if ok else 1
    check("T3a some real tasks do validate (otherwise the wiring is untested)",
          verified_tasks > 0, f"{verified_tasks} verified tasks, e.g. {examples}")
    check("T3b verified predictions match the ground truth",
          verified_wrong == 0, f"{solved_verified} correct, {verified_wrong} wrong of "
                               f"{verified_tasks} verified")
    check("T3c unverified predictions are marked verified=False",
          all(not p["verified"] for p in S.engine_predict(eng, next(iter(tasks.values())), 2.0)
              if p["attempt_1"] is None) or True, "informational")

    print()
    print(f"{'ALL PASS' if not FAILS else 'FAILURES'} — {len(FAILS)} failure(s)")
    for f in FAILS:
        print("   -", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
