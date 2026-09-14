#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Independently verify the two structural claims about the ARC-AGI-2 2026 data.

Claim 1 (leak): the public `arc-agi_test_challenges.json` is not a holdout -- its 240 tasks
    are byte-identical to training tasks and their answers ship in training_solutions.
Claim 2 (format): `{task_id: [{"attempt_1": g, "attempt_2": g}, ...]}` with the list length
    equal to the number of test inputs, and the public test file under-represents
    multi-input tasks relative to the evaluation split.

We check both against the files themselves rather than trusting a third-party notebook.
Run:  python tools/verify_leak.py comp_data
"""

from __future__ import annotations

import collections
import hashlib
import json
import sys
from pathlib import Path


def load(d: Path, name: str):
    return json.loads((d / name).read_text(encoding="utf-8"))


def task_hash(t) -> str:
    return hashlib.md5(json.dumps(t, sort_keys=True).encode()).hexdigest()


def main() -> int:
    d = Path(sys.argv[1] if len(sys.argv) > 1 else "comp_data")
    test = load(d, "arc-agi_test_challenges.json")
    train = load(d, "arc-agi_training_challenges.json")
    tsol = load(d, "arc-agi_training_solutions.json")
    evalc = load(d, "arc-agi_evaluation_challenges.json")
    samp = load(d, "sample_submission.json")

    print(f"counts: test={len(test)} train={len(train)} eval={len(evalc)} sample={len(samp)}")

    print("\n--- claim 1: the public test file is leaked training data ---")
    shared = set(test) & set(train)
    identical = sum(1 for k in shared if task_hash(test[k]) == task_hash(train[k]))
    answers = sum(1 for k in test if k in tsol)
    overlap = len({task_hash(v) for v in test.values()} & {task_hash(v) for v in evalc.values()})
    print(f"  ids shared with training      : {len(shared)}/{len(test)}")
    print(f"  byte-identical to training    : {identical}/{len(test)}")
    print(f"  answers in training_solutions : {answers}/{len(test)}")
    print(f"  content overlap with eval     : {overlap}")
    leak = identical == len(test) == answers and overlap == 0
    print(f"  => claim 1 {'CONFIRMED' if leak else 'REFUTED'}")

    print("\n--- claim 2: the submission format ---")
    print(f"  sample keys == test keys      : {set(samp) == set(test)}")
    first = sorted(samp)[0]
    print(f"  entry keys                    : {sorted(samp[first][0])}")
    print(f"  list length == #test inputs   : "
          f"{all(len(samp[k]) == len(test[k]['test']) for k in samp)}")

    multi = lambda dd: sum(1 for t in dd.values() if len(t["test"]) > 1)
    mt, me = 100 * multi(test) / len(test), 100 * multi(evalc) / len(evalc)
    print(f"  multi-input tasks: public test {mt:.1f}%  vs  evaluation {me:.1f}%")
    print(f"  test dist {dict(sorted(collections.Counter(len(t['test']) for t in test.values()).items()))}")
    print(f"  eval dist {dict(sorted(collections.Counter(len(t['test']) for t in evalc.values()).items()))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
