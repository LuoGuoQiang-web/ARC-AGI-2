#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_merge_shards.py -- prove tools/merge_shards.py is correct on the real task set.

Builds synthetic shards from the REAL ARC-AGI-2 evaluation split (interleaved exactly the
way `arc26_solver.py --num-shards` does), merges them, and cross-checks the result against
the engine's own validator (`arc_prize_v1.validate_submission`) so the two independent
implementations must agree.

Usage:  python tests/test_merge_shards.py
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARC_W1 = HERE.parent
WORKSPACE = ARC_W1.parent.parent
MERGE = ARC_W1 / "tools" / "merge_shards.py"
ENGINE = WORKSPACE / "arc_prize_v1" / "arc_prize_v1.py"
DATA = WORKSPACE / "ARC-AGI-2-main" / "data"
TMP = ARC_W1 / ".selftest_merge"

FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILS.append(f"{name}: {detail}")


def run_merge(args: list[str]) -> tuple[int, str]:
    r = subprocess.run([sys.executable, str(MERGE), *args], capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr


def main() -> int:
    if not DATA.is_dir():
        print(f"missing data: {DATA}")
        return 2
    shutil.rmtree(TMP, ignore_errors=True)
    TMP.mkdir(parents=True)

    ev = DATA / "evaluation"
    tasks = {}
    for f in sorted(ev.glob("*.json")):
        body = json.loads(f.read_text(encoding="utf-8"))
        tasks[f.stem] = [p.get("output") for p in body["test"]]

    # challenges file, as the competition ships it (test entries carry the answers here,
    # which does not matter -- the merge only reads the *shape* from it)
    challenges = TMP / "arc-agi_evaluation_challenges.json"
    challenges.write_text(json.dumps({
        tid: {"train": [{"input": [[0]], "output": [[1]]}],
              "test": [{"input": [[0]]} for _ in outs]}
        for tid, outs in tasks.items()
    }), encoding="utf-8")

    ids = sorted(tasks)

    def shard_payload(shard: int, n_shards: int, drop: set[str] = frozenset(),
                      corrupt: set[str] = frozenset()) -> dict:
        out = {}
        for i, tid in enumerate(ids):
            if i % n_shards != shard or tid in drop:
                continue
            entries = []
            for out_grid in tasks[tid]:
                g = out_grid if out_grid is not None else [[3]]
                entries.append({"attempt_1": g, "attempt_2": [[0]]})
            if tid in corrupt:
                entries = entries[:-1]  # wrong entry count -> must be rejected
            out[tid] = entries
        return out

    s0 = TMP / "shard0_submission.json"
    s1 = TMP / "shard1_submission.json"
    s0.write_text(json.dumps(shard_payload(0, 2)), encoding="utf-8")
    s1.write_text(json.dumps(shard_payload(1, 2)), encoding="utf-8")

    # ---- T1: a clean 2-way merge covers everything and is schema-valid ---------------
    out = TMP / "merged.json"
    rc, log = run_merge(["--expected", str(challenges), "--shards", str(s0), str(s1),
                         "--out", str(out), "--report", str(TMP / "report.json")])
    check("T1a merge exits 0", rc == 0, log[-300:])
    check("T1b output exists", out.exists())
    payload = json.loads(out.read_text(encoding="utf-8"))
    check("T1c all tasks covered", set(payload) == set(ids),
          f"{len(payload)} vs {len(ids)}")
    check("T1d entry counts preserved",
          all(len(payload[t]) == len(tasks[t]) for t in ids))
    check("T1e reports full coverage", "tasks covered by real shard output : 120/120" in log,
          [l for l in log.splitlines() if "covered" in l][:1])

    # ---- T2: cross-check against the engine's own validator -------------------------
    spec = importlib.util.spec_from_file_location("arc_prize_v1", ENGINE)
    arc = importlib.util.module_from_spec(spec)
    sys.modules["arc_prize_v1"] = arc
    spec.loader.exec_module(arc)
    ok, problems = arc.validate_submission(out, [(t, len(tasks[t])) for t in ids])
    check("T2 engine validator agrees it is VALID", ok is True, str(problems[:3]))

    # ---- T3: gaps are filled, conflicts reported, junk rejected ---------------------
    # Only ids[0] is a genuine gap: the interleaved split puts ids[1] (i=1) in shard 1.
    gap = ids[0]
    drop = {ids[0]}
    p0b = shard_payload(0, 2, drop=drop)
    p1 = shard_payload(1, 2)
    # Force a REAL conflict: ids[3] (i=3) belongs to shard 1, so also carry it in shard 0
    # with different content -- the merge must keep the first and report the clash.
    conflict_tid = ids[3]
    p0b[conflict_tid] = [{"attempt_1": [[7, 7], [7, 7]], "attempt_2": [[6]]}
                         for _ in p1[conflict_tid]]
    s0b = TMP / "shard0b_submission.json"
    s0b.write_text(json.dumps(p0b), encoding="utf-8")
    s1b = TMP / "shard1b_submission.json"
    s1b.write_text(json.dumps(p1), encoding="utf-8")
    # a third file that is garbage
    junk = TMP / "junk_submission.json"
    junk.write_text(json.dumps({"not_a_task": [{"attempt_1": [[1]]}],
                                ids[5]: "nonsense"}), encoding="utf-8")

    out2 = TMP / "merged2.json"
    rc2, log2 = run_merge(["--expected", str(challenges), "--shards", str(s0b), str(s1b),
                           str(junk), "--out", str(out2), "--report", str(TMP / "r2.json")])
    pay2 = json.loads(out2.read_text(encoding="utf-8"))
    check("T3a gaps filled so the file stays valid", rc2 == 0, log2[-250:])
    check("T3b gap entry is the legal floor",
          pay2[gap] == [{"attempt_1": [[0]], "attempt_2": [[0]]} for _ in tasks[gap]],
          json.dumps(pay2[gap])[:120])
    check("T3c conflict detected and first kept",
          "differ; kept the first" in log2
          and pay2[conflict_tid][0]["attempt_1"] == [[7, 7], [7, 7]],
          [l for l in log2.splitlines() if "differ" in l][:1])
    check("T3d junk task id ignored", "not_a_task" not in pay2)
    check("T3e malformed entry rejected", "malformed or wrong entry count" in log2,
          [l for l in log2.splitlines() if "malformed" in l][:1])
    ok3, _ = arc.validate_submission(out2, [(t, len(tasks[t])) for t in ids])
    check("T3f still valid after fill", ok3 is True)

    # ---- T4: --no-fill surfaces the gap as INVALID (never silently incomplete) -------
    out3 = TMP / "merged3.json"
    rc3, log3 = run_merge(["--expected", str(challenges), "--shards", str(s0b), str(s1b),
                           "--out", str(out3), "--no-fill"])
    check("T4a --no-fill reports INVALID", "INVALID" in log3 and rc3 == 1, log3[-200:])
    check("T4b --no-fill leaves the gap visible", "missing task" in log3)

    # ---- T5: sample_submission.json is accepted as the expected file ----------------
    sample = TMP / "sample_submission.json"
    sample.write_text(json.dumps({t: [{"attempt_1": [[0]], "attempt_2": [[0]]}
                                      for _ in tasks[t]] for t in ids}), encoding="utf-8")
    out4 = TMP / "merged4.json"
    rc4, log4 = run_merge(["--expected", str(sample), "--shards", str(s0), str(s1),
                           "--out", str(out4)])
    check("T5 sample_submission accepted as expected", rc4 == 0 and "120/120" in log4,
          log4[-200:])

    print()
    print(f"{'ALL PASS' if not FAILS else 'FAILURES'} — {len(FAILS)} failure(s)")
    for f in FAILS:
        print("   -", f)
    if "--keep" not in sys.argv:
        shutil.rmtree(TMP, ignore_errors=True)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
