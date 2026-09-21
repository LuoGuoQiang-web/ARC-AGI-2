#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Demonstrate that the failure mode generalises: a second, unrelated pipeline fails the same way.

The paper claims the masked-failure pattern applies to *any* pipeline with an optional learned
refinement stage, not just to the ARC solver it was found in. That is a claim, and a claim is
worth less than a demonstration. This builds the smallest pipeline that has the relevant shape --
a mandatory predictor plus an optional refinement step -- in a different domain, with a different
failure cause, and shows the same outcome twice: once as the pipeline reports it, once as
`adaptation_audit` reports it.

The domain is deliberately not ARC: tasks are integers, the refinement is a least-squares fit, and
the failure is a data-dependent numerical blow-up rather than a GPU allocation. Nothing here shares
code or hardware with the solver, which is the point -- if the pattern were an artefact of that
codebase or of CUDA, it would not reproduce here.

Run:  python tools/universality_demo.py
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from adaptation_audit import AdaptationAudit  # noqa: E402


def predict(task: float) -> float:
    """The mandatory stage. Crude, but it always returns something."""
    return 0.5 * task


def refine(xs: list[float], ys: list[float], steps: int) -> int:
    """The optional stage: gradient descent on a linear fit. Returns steps actually taken.

    Raises on divergence, which is this domain's equivalent of a failed allocation: it is a
    resource-ish failure (the step size outgrows the data) that the caller is expected to survive.
    """
    w, b, lr = 0.0, 0.0, 0.1
    done = 0
    for _ in range(steps):
        gw = gb = 0.0
        for x, y in zip(xs, ys):
            err = (w * x + b) - y
            gw += 2 * err * x
            gb += 2 * err
        gw /= len(xs)
        gb /= len(xs)
        if not (math.isfinite(gw) and math.isfinite(gb)):
            raise ArithmeticError("gradient diverged")
        w -= lr * gw
        b -= lr * gb
        if abs(gw) + abs(gb) > 1e6:
            raise ArithmeticError("gradient exploded")
        done += 1
    return done


def run(n_tasks: int = 40, steps: int = 8, seed: int = 7) -> int:
    rng = random.Random(seed)
    audit = AdaptationAudit("refine", expect_steps_per_task=steps)
    naive_log = {"attempted": 0, "gained": 0}

    for i in range(n_tasks):
        xs = [rng.uniform(-1, 1) for _ in range(5)]
        ys = [3.0 * x + 1.0 + rng.gauss(0, 0.01) for x in xs]
        # Two of every three tasks carry an outlier large enough to blow the fixed step size up,
        # so the stage fails on the majority -- which is the regime where the audit stops warning
        # and starts refusing to let the result stand.
        if i % 3 != 0:
            ys[0] = 1e9

        # --- how the pipeline would report it on its own -------------------------------
        naive_log["attempted"] += 1
        before = predict(xs[0])
        try:
            with audit.attempt(f"task{i}") as rec:
                done = refine(xs, ys, steps)
                rec.step(done)
        except ArithmeticError:
            # The optional stage failed. The pipeline survives -- that is exactly the problem.
            pass
        after = predict(xs[0])
        if after != before:
            naive_log["gained"] += 1

    print("=" * 78)
    print("what the pipeline reports about itself")
    print("=" * 78)
    print(f"  refinement attempted on : {naive_log['attempted']} tasks")
    print(f"  tasks whose output changed: {naive_log['gained']}")
    print(f"  => the honest-looking conclusion: 'refinement was tried on every task and helped "
          f"on {naive_log['gained']}'")

    print()
    print("=" * 78)
    print("what the audit reports")
    print("=" * 78)
    print(audit.report())
    s = audit.summary()
    print()
    print(f"  execution rate          : {s['execution_rate']:.0%}")
    print(f"  mean steps per executed : {s['mean_steps_per_executed_task']:.2f} "
          f"(configured {s['expected_steps_per_task']})")

    print()
    print("=" * 78)
    print("does it refuse to let the result stand?")
    print("=" * 78)
    try:
        audit.assert_healthy()
        print("  NO -- assert_healthy passed, which would be a bug in the audit")
        return 1
    except AssertionError:
        print("  YES -- assert_healthy raised, so the pipeline cannot report this run as")
        print("        evidence about the refinement stage without confronting the shortfall")

    print()
    print("Same shape as the ARC solver: a mandatory predictor, an optional refinement, a")
    print("resource-ish failure, and a run that continues. Different domain, different failure")
    print("cause, no shared code or hardware -- and the same silent outcome, caught by the same")
    print("counter.")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
