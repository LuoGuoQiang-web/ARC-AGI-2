#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""adaptation_audit -- count the optimizer steps that EXECUTED, not the ones you configured.

The problem this exists for
---------------------------
Any pipeline with an *optional learned refinement stage* -- test-time training, gradient
adaptation, self-critique, verifier-guided repair -- has a failure mode that is invisible in
its own output. The stage is wrapped in error handling because it is optional; the pipeline
must return a prediction regardless; so a resource failure inside the stage (an OOM, a
timeout, an empty batch) is caught, the run continues, and the final metric looks exactly
like a stage that ran correctly and did not help.

Measured, in the ARC-AGI-2 solver this module was extracted from: 102 tasks were routed to
the test-time-training stage, which consumed 7,266 seconds to perform 121 optimizer steps --
a mean of 1.34 per task against the reference recipe's 128. And the run's own counter
reported 21 steps across 16 tasks, under-counting by 6x and disagreeing with the kernel log
on 77 of 102 tasks, because a later pipeline stage re-recorded each task and overwrote the
count. The result was read as a statement about model capability rather than about an
adaptation stage running at one percent of its budget -- measured by an instrument that
could not see straight. That is the failure this module exists to make impossible.

The fix is one counter, and it generalises well beyond ARC: record how many steps ran, per
task, and refuse to draw conclusions from a stage that mostly did not execute.

Usage
-----
    from adaptation_audit import AdaptationAudit

    audit = AdaptationAudit("ttt", expect_steps_per_task=8)

    for task in tasks:
        with audit.attempt(task.id) as rec:
            for batch in batches(task):
                ...                      # forward, backward
                optimizer.step()
                rec.step()               # <-- the one line that matters
        # anything that raised inside the block is counted as a failed attempt,
        # not silently as a task that "did not benefit"

    print(audit.report())
    audit.assert_healthy()               # raises if the stage mostly never ran

Stdlib only, no dependencies, no GPU, no framework assumptions.
"""

from __future__ import annotations

import contextlib
import math
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional


@dataclass
class _TaskRecord:
    task_id: str
    steps: int = 0
    error: str = ""
    attempted: bool = True


@dataclass
class AdaptationAudit:
    """Per-task executed-step counter for an optional refinement stage.

    ``expect_steps_per_task`` is what the configuration *intends*; it is used only to phrase
    the warning, never to substitute for the measurement. A stage that reports intended
    rather than executed steps is the bug this class exists to prevent.
    """

    name: str = "adaptation"
    expect_steps_per_task: int = 0
    records: Dict[str, _TaskRecord] = field(default_factory=dict)
    _order: List[str] = field(default_factory=list)

    # -- recording ---------------------------------------------------------------------
    @contextlib.contextmanager
    def attempt(self, task_id: str) -> Iterator["_StepRecorder"]:
        """Register one attempt at adapting ``task_id``.

        An exception escaping the block is recorded as a *failed attempt* -- explicitly not
        as a task that was adapted and did not improve. Failures are re-raised; this class
        observes, it does not swallow.
        """
        rec = self.records.get(task_id)
        if rec is None:
            rec = _TaskRecord(task_id=task_id)
            self.records[task_id] = rec
            self._order.append(task_id)
        recorder = _StepRecorder(rec)
        try:
            yield recorder
        except BaseException as exc:                    # noqa: BLE001 - record, then re-raise
            rec.error = f"{type(exc).__name__}: {exc}"[:300]
            raise

    # -- reporting ---------------------------------------------------------------------
    @property
    def attempted(self) -> int:
        return len(self.records)

    @property
    def executed(self) -> int:
        return sum(1 for r in self.records.values() if r.steps > 0)

    @property
    def never_executed(self) -> int:
        return self.attempted - self.executed

    @property
    def total_steps(self) -> int:
        return sum(r.steps for r in self.records.values())

    @property
    def failed(self) -> int:
        return sum(1 for r in self.records.values() if r.error)

    def summary(self) -> dict:
        att, ex = self.attempted, self.executed
        steps = [r.steps for r in self.records.values()]
        return {
            "name": self.name,
            "tasks_attempted": att,
            "tasks_executed": ex,
            "tasks_never_executed": self.never_executed,
            "tasks_failed": self.failed,
            "execution_rate": (ex / att) if att else None,
            "total_optimizer_steps": self.total_steps,
            "mean_steps_per_executed_task": (self.total_steps / ex) if ex else None,
            "max_steps_on_any_task": max(steps) if steps else 0,
            "expected_steps_per_task": self.expect_steps_per_task or None,
        }

    def report(self) -> str:
        s = self.summary()
        if not s["tasks_attempted"]:
            return f"[{self.name}] no tasks were routed to this stage"
        rate = 100 * s["execution_rate"]
        lines = [
            f"[{self.name}] routed={s['tasks_attempted']} executed={s['tasks_executed']} "
            f"never_executed={s['tasks_never_executed']} failed={s['tasks_failed']}",
            f"[{self.name}] total_optimizer_steps={s['total_optimizer_steps']} "
            f"max_on_any_task={s['max_steps_on_any_task']}"
            + (f" (configured {s['expected_steps_per_task']})"
               if s["expected_steps_per_task"] else ""),
        ]
        if s["tasks_executed"] == 0:
            lines.append(
                f"! WARNING: '{self.name}' was routed {s['tasks_attempted']} task(s) and "
                f"executed ZERO optimizer steps. Any downstream result from this run describes "
                f"the UNADAPTED pipeline and must not be reported as a result of '{self.name}'.")
        elif self.never_executed:
            lines.append(
                f"! WARNING: {self.never_executed}/{s['tasks_attempted']} task(s) never executed "
                f"a single step ({rate:.0f}% executed); their 'no gain' verdicts carry no "
                f"information about '{self.name}'.")
        return "\n".join(lines)

    def assert_healthy(self, min_execution_rate: float = 0.5) -> None:
        """Raise if the stage mostly did not run.

        Call this BEFORE writing up any result that depends on the stage having run. The
        whole point is to fail loudly here rather than quietly in a paper.
        """
        if not self.attempted:
            return
        rate = self.executed / self.attempted
        if rate < min_execution_rate:
            raise AssertionError(
                f"'{self.name}' executed on only {self.executed}/{self.attempted} tasks "
                f"({rate:.0%} < {min_execution_rate:.0%}); results from this run do not "
                f"describe the adapted pipeline.\n{self.report()}")


class _StepRecorder:
    """Handed to the caller inside ``with audit.attempt(...)``; call ``.step()`` per step."""

    __slots__ = ("_rec",)

    def __init__(self, rec: _TaskRecord) -> None:
        self._rec = rec

    def step(self, n: int = 1) -> None:
        """Record that ``n`` optimizer step(s) actually completed."""
        self._rec.steps += int(n)

    @property
    def steps(self) -> int:
        return self._rec.steps


# ---------------------------------------------------------------------------------------
# Self-test: `python adaptation_audit.py`
# ---------------------------------------------------------------------------------------
def _selftest() -> int:
    checks = []

    def ck(name, cond, detail=""):
        checks.append((name, bool(cond), detail))

    # 1. A healthy stage.
    a = AdaptationAudit("ttt", expect_steps_per_task=4)
    for t in range(4):
        with a.attempt(f"t{t}") as rec:
            for _ in range(4):
                rec.step()
    ck("healthy stage reports full execution", a.executed == 4 and a.total_steps == 16, a.report())
    a.assert_healthy()
    ck("healthy stage passes assert_healthy", True)

    # 2. The measured failure: most tasks never step.
    b = AdaptationAudit("ttt")
    for t in range(10):
        with b.attempt(f"t{t}") as rec:
            if t < 2:
                rec.step()
    ck("silent no-op stage is counted, not assumed",
       b.attempted == 10 and b.executed == 2 and b.never_executed == 8, b.report())
    ck("zero-adaptation rate is visible", abs(b.summary()["execution_rate"] - 0.2) < 1e-9)
    try:
        b.assert_healthy()
        ck("assert_healthy rejects a mostly-not-run stage", False, "did not raise")
    except AssertionError:
        ck("assert_healthy rejects a mostly-not-run stage", True)

    # 3. A stage that never runs at all must warn explicitly.
    c = AdaptationAudit("ttt")
    for t in range(3):
        with c.attempt(f"t{t}"):
            pass
    ck("total no-op warns in the report", "ZERO optimizer steps" in c.report(), c.report())

    # 4. Exceptions are recorded as failures and re-raised, never swallowed.
    d = AdaptationAudit("ttt")
    raised = False
    try:
        with d.attempt("t0"):
            raise RuntimeError("CUDA out of memory")
    except RuntimeError:
        raised = True
    ck("failures propagate", raised and d.failed == 1 and d.executed == 0,
       str(d.summary()))

    # 5. Re-attempting a task accumulates rather than double-counting the task.
    e = AdaptationAudit("ttt")
    with e.attempt("t0") as rec:
        rec.step()
    with e.attempt("t0") as rec:
        rec.step(2)
    ck("repeat attempts accumulate steps on one task",
       e.attempted == 1 and e.total_steps == 3 and e.executed == 1, str(e.summary()))

    # 6. Nothing routed is not an error.
    f = AdaptationAudit("ttt")
    ck("no tasks is a clean state", f.summary()["execution_rate"] is None
       and "no tasks" in f.report())
    f.assert_healthy()

    # 7. The expected count is reported but never substituted for the measurement.
    g = AdaptationAudit("ttt", expect_steps_per_task=128)
    with g.attempt("t0"):
        pass
    ck("configured steps never masquerade as executed",
       g.total_steps == 0 and g.summary()["expected_steps_per_task"] == 128, g.report())
    ck("the shortfall against the configured budget is visible",
       math.isclose(g.summary()["max_steps_on_any_task"] / 128, 0.0))

    bad = [c for c in checks if not c[1]]
    for name, ok, detail in checks:
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if not ok else ""))
    print(f"\n{len(checks) - len(bad)}/{len(checks)} checks passed")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(_selftest())
