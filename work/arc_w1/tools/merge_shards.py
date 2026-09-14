#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
merge_shards.py -- merge per-shard submission.json files into one complete submission.

Why this exists (P0, 2026-09-14): a full 240-task run costs ~8.5 h of GPU while the
account's allowance is 21,600 s (6 h) per period, so the submission has to be produced as
interleaved shards across periods. `arc26_solver.py --num-shards/--shard-index` makes each
run emit only its own tasks, and Kaggle scores the output of ONE notebook version -- so
something has to union them. That is this tool.

Operational shape on Kaggle (the merge must happen inside a notebook):

    1. run shard 0 and shard 1 as separate kernels (they write submission.json)
    2. download both and upload them as a Kaggle Dataset (or attach the kernels as
       kernel_sources), e.g. /kaggle/input/arc26-shards/shard0_submission.json
    3. run ONE notebook whose only job is:
           python merge_shards.py --expected /kaggle/input/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json \
                                  --shards '/kaggle/input/arc26-shards/*.json' \
                                  --out /kaggle/working/submission.json
    4. submit that notebook version

Self-contained on purpose: it must run in a minimal notebook, so it does not import the
2600-line solver. Its own validator is cross-checked against
``arc_prize_v1.validate_submission`` by ``tests/test_merge_shards.py``.

Usage:
    python tools/merge_shards.py --expected <challenges.json|sample_submission.json|dir>
                                 --shards <glob> [<glob> ...]
                                 --out submission.json
                                 [--no-fill] [--report report.json]
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

MAX_SIDE = 30


# --------------------------------------------------------------------------------------
# schema helpers (kept in sync with arc_prize_v1 by tests/test_merge_shards.py)
# --------------------------------------------------------------------------------------
def sanitize_grid(grid: Any) -> List[List[int]]:
    """Return a legal submission grid: non-empty, rectangular, ints 0..9, side <= 30."""
    if not isinstance(grid, list) or not grid or not all(isinstance(r, list) and r for r in grid):
        return [[0]]
    width = max(len(r) for r in grid)
    out = []
    for row in grid:
        row = list(row) + [0] * (width - len(row))
        vals = []
        for v in row:
            try:
                iv = int(v)
            except Exception:
                iv = 0
            vals.append(max(0, min(9, iv)))
        out.append(vals)
    if not (1 <= len(out) <= MAX_SIDE) or not (1 <= len(out[0]) <= MAX_SIDE):
        return [[0]]
    return out


def normalise_task(entries: Any, n_expected: Optional[int]) -> Optional[List[Dict[str, Any]]]:
    """Coerce one task's entry list into [{'attempt_1': grid, 'attempt_2': grid}, ...]."""
    if not isinstance(entries, list):
        return None
    out: List[Dict[str, Any]] = []
    for entry in entries:
        if isinstance(entry, dict) and ("attempt_1" in entry or "attempt_2" in entry):
            out.append({"attempt_1": sanitize_grid(entry.get("attempt_1")),
                        "attempt_2": sanitize_grid(entry.get("attempt_2"))})
        elif isinstance(entry, list):
            # some people write a bare grid; accept it as attempt_1
            out.append({"attempt_1": sanitize_grid(entry), "attempt_2": sanitize_grid(entry)})
        else:
            return None
    if n_expected is not None and len(out) != n_expected:
        return None
    return out


def validate(payload: Any, expected: Sequence[Tuple[str, int]]) -> List[str]:
    """Return a list of problems (empty == valid). Mirrors the official contract."""
    problems: List[str] = []
    if not isinstance(payload, dict):
        return ["submission root is not a JSON object"]
    for tid, n in expected:
        if tid not in payload:
            problems.append(f"missing task {tid}")
            continue
        entries = payload[tid]
        if not isinstance(entries, list) or len(entries) != n:
            got = len(entries) if isinstance(entries, list) else type(entries).__name__
            problems.append(f"task {tid}: expected {n} entries, got {got}")
            continue
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict) or set(entry) != {"attempt_1", "attempt_2"}:
                problems.append(f"task {tid}[{i}]: keys must be exactly attempt_1/attempt_2")
                continue
            for k in ("attempt_1", "attempt_2"):
                g = entry[k]
                if not isinstance(g, list) or not g or not all(isinstance(r, list) and r for r in g):
                    problems.append(f"task {tid}[{i}].{k}: not a non-empty 2-D array")
                    continue
                w = len(g[0])
                if any(len(r) != w for r in g):
                    problems.append(f"task {tid}[{i}].{k}: ragged rows")
                if any((not isinstance(v, int)) or v < 0 or v > 9 for row in g for v in row):
                    problems.append(f"task {tid}[{i}].{k}: values must be ints 0..9")
    extra = set(payload) - {t for t, _ in expected}
    if extra:
        problems.append(f"{len(extra)} unexpected task ids (e.g. {sorted(extra)[:3]})")
    return problems


# --------------------------------------------------------------------------------------
# expected structure
# --------------------------------------------------------------------------------------
def expected_from_expected_file(path: Path) -> List[Tuple[str, int]]:
    """Accept challenges ({tid: {train, test}}), sample_submission ({tid: [entry,...]})
    or a directory of one-task-per-file JSONs."""
    if path.is_dir():
        out = []
        for f in sorted(path.glob("*.json")):
            body = json.loads(f.read_text(encoding="utf-8"))
            if isinstance(body, dict) and "test" in body:
                out.append((f.stem, len(body["test"])))
        return out
    body = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(body, dict):
        raise ValueError(f"{path}: expected a JSON object")
    sample = next(iter(body.values()))
    if isinstance(sample, dict) and "test" in sample:      # challenges file
        return [(tid, len(v["test"])) for tid, v in sorted(body.items())]
    if isinstance(sample, list):                            # sample submission
        return [(tid, len(v)) for tid, v in sorted(body.items())]
    raise ValueError(f"{path}: unrecognised structure")


def expand_shard_args(patterns: Sequence[str]) -> List[Path]:
    found: List[Path] = []
    for pat in patterns:
        hits = [Path(p) for p in glob.glob(pat, recursive=True)]
        if not hits and Path(pat).exists():
            hits = [Path(pat)]
        for h in sorted(hits):
            if h.is_dir():
                hits_sub = sorted(h.rglob("*.json"))
                found.extend(x for x in hits_sub if "submission" in x.name)
            else:
                if "log" not in h.name.lower():
                    found.append(h)
    # dedupe, keep order
    seen, uniq = set(), []
    for p in found:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


# --------------------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description="merge shard submissions into one valid submission")
    ap.add_argument("--expected", required=True,
                    help="challenges JSON (preferred), sample_submission.json, or a task dir")
    ap.add_argument("--shards", nargs="+", required=True, help="shard submission globs/paths")
    ap.add_argument("--out", default="/kaggle/working/submission.json")
    ap.add_argument("--no-fill", action="store_true",
                    help="do not pad tasks no shard covered (default: pad so the file stays valid)")
    ap.add_argument("--report", default="", help="also write a JSON report here")
    args = ap.parse_args()

    expected = expected_from_expected_file(Path(args.expected))
    exp_map = dict(expected)
    print(f"expected: {len(expected)} tasks, {sum(n for _, n in expected)} test inputs "
          f"(from {args.expected})")

    shard_paths = expand_shard_args(args.shards)
    print(f"shards  : {len(shard_paths)} file(s)")
    for p in shard_paths:
        print(f"   {p}")

    merged: Dict[str, Any] = {}
    origin: Dict[str, str] = {}
    conflicts: List[str] = []
    rejected: List[str] = []
    contributed: Dict[str, int] = {}

    for path in shard_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            rejected.append(f"{path}: unreadable ({exc})")
            continue
        if not isinstance(payload, dict):
            rejected.append(f"{path}: root is not an object")
            continue
        kept = 0
        for tid, entries in payload.items():
            if tid not in exp_map:
                continue                      # ignored: not part of this competition split
            norm = normalise_task(entries, exp_map[tid])
            if norm is None:
                rejected.append(f"{path}:{tid}: malformed or wrong entry count "
                                f"(expected {exp_map[tid]})")
                continue
            if tid in merged:
                if merged[tid] != norm:
                    conflicts.append(f"{tid}: {origin[tid]} vs {path.name} differ; kept the first")
                continue
            merged[tid] = norm
            origin[tid] = path.name
            kept += 1
        contributed[path.name] = kept
        print(f"   {path.name}: contributed {kept} task(s)")

    missing = [tid for tid, _ in expected if tid not in merged]
    filled = []
    if missing and not args.no_fill:
        for tid in missing:
            merged[tid] = [{"attempt_1": [[0]], "attempt_2": [[0]]} for _ in range(exp_map[tid])]
            origin[tid] = "FILL"
            filled.append(tid)

    # emit in the canonical order, with every task normalised once more
    out_payload = {tid: merged[tid] for tid, _ in expected if tid in merged}
    problems = validate(out_payload, expected)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_text(json.dumps(out_payload), encoding="utf-8")
    os.replace(tmp, out_path)

    covered = len(expected) - len(missing)
    print()
    print("=" * 74)
    print(f"tasks covered by real shard output : {covered}/{len(expected)}")
    print(f"filled with the all-zero floor      : {len(filled)}"
          + (f"  e.g. {filled[:5]}" if filled else ""))
    print(f"conflicts                           : {len(conflicts)}")
    for c in conflicts[:10]:
        print(f"   ! {c}")
    print(f"rejected shard records              : {len(rejected)}")
    for r in rejected[:10]:
        print(f"   ! {r}")
    print(f"schema                              : {'VALID' if not problems else 'INVALID'}")
    for p in problems[:10]:
        print(f"   ! {p}")
    print(f"written                             : {out_path} "
          f"({out_path.stat().st_size} bytes)")
    print("=" * 74)

    if args.report:
        Path(args.report).write_text(json.dumps({
            "expected_tasks": len(expected),
            "expected_inputs": sum(n for _, n in expected),
            "shards": [str(p) for p in shard_paths],
            "contributed": contributed,
            "covered": covered,
            "filled": filled,
            "conflicts": conflicts,
            "rejected": rejected,
            "problems": problems,
            "valid": not problems,
            "out": str(out_path),
        }, indent=2), encoding="utf-8")
        print(f"report  : {args.report}")

    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
