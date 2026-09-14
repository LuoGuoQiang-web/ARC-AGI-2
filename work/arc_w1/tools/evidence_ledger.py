#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Dump every number the paper will cite, straight from the artifacts.

The point is provenance: the paper must not contain a figure that cannot be traced to a
file in this repo. This prints one line per fact, each tagged with its source path, so the
ledger can be pasted into the paper's evidence appendix and re-derived by a reviewer.

Run:  python tools/evidence_ledger.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                      # work/arc_w1
REPO = ROOT.parent.parent               # repo root
OUT = ROOT / "kaggle_out"

rows: list[tuple[str, str, str]] = []


def fact(value, source: str, note: str = "") -> None:
    rows.append((str(value), source, note))


def load_safe(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------- corpus facts
def corpus() -> None:
    d = ROOT / "comp_data"
    if not (d / "arc-agi_test_challenges.json").exists():
        fact("UNAVAILABLE", "comp_data/", "run tools/verify_leak.py after downloading")
        return
    L = lambda n: json.loads((d / n).read_text(encoding="utf-8"))
    test, train = L("arc-agi_test_challenges.json"), L("arc-agi_training_challenges.json")
    evalc, samp = L("arc-agi_evaluation_challenges.json"), L("sample_submission.json")
    ti = lambda dd: sum(len(t["test"]) for t in dd.values())
    fact(f"{len(train)} / {ti(train)}", "arc-agi_training_challenges.json", "tasks / test inputs")
    fact(f"{len(evalc)} / {ti(evalc)}", "arc-agi_evaluation_challenges.json", "tasks / test inputs")
    fact(f"{len(test)} / {ti(test)}", "arc-agi_test_challenges.json", "tasks / test inputs")
    fact(f"{sum(1 for t in test.values() if len(t['test'])>1)}",
         "arc-agi_test_challenges.json", "multi-test-input tasks")
    fact(f"{sum(1 for t in evalc.values() if len(t['test'])>1)}",
         "arc-agi_evaluation_challenges.json", "multi-test-input tasks")
    fact(f"{len(set(samp) & set(evalc))} / {len(samp)}",
         "sample_submission.json", "eval ids present in the public sample (expect 0)")


# ---------------------------------------------------------------- run reports
def runs() -> None:
    for p in sorted(OUT.glob("*/report.json")):
        r = load_safe(p)
        if not isinstance(r, dict):
            continue
        slug = p.parent.name
        pr = r.get("pool_recall") or {}
        st = {s.get("name"): s for s in (r.get("stages") or [])}
        df = r.get("dfs") or {}
        fact(f"{r.get('split')} n={r.get('n_tasks_total')} scored={r.get('n_tasks_scored')} "
             f"solved={r.get('n_solved')} acc={r.get('accuracy')}",
             f"kaggle_out/{slug}/report.json", "run outcome")
        if pr:
            fact(f"pool_recall={pr.get('pool_recall')} in_pool={pr.get('n_in_pool')}"
                 f"/{pr.get('n_with_truth')} headroom={pr.get('selection_headroom')}",
                 f"kaggle_out/{slug}/report.json", "oracle recall")
        fact(f"dfs p/b/n={df.get('prob_threshold')}/{df.get('max_branches')}/{df.get('max_nodes')}",
             f"kaggle_out/{slug}/report.json", "search width")
        fact(f"elapsed={r.get('elapsed_seconds')}s peak={r.get('peak_mem_gb')}GiB "
             f"errors={len(r.get('errors') or [])}",
             f"kaggle_out/{slug}/report.json", "cost")
        for name, s in st.items():
            fact(f"{name}: tasks={s.get('tasks')} sec={s.get('seconds')} "
                 f"s/task={s.get('seconds_per_task')}",
                 f"kaggle_out/{slug}/report.json", "stage")
        eng = r.get("engine") or {}
        fact(f"engine={eng.get('status')} validated={eng.get('n_tasks_validated')} "
             f"prior_only={eng.get('n_tasks_prior_only')}",
             f"kaggle_out/{slug}/report.json", "symbolic engine")


# ---------------------------------------------------------------- TTT telemetry
def ttt() -> None:
    for p in sorted(OUT.glob("*/kernel_stdout.log")):
        txt = p.read_text(encoding="utf-8", errors="replace")
        slug = p.parent.name
        steps = re.findall(r"TTT (\d+) steps in ([\d.]+)s", txt)
        oom = len(re.findall(r"TTT step (\d+) failed: CUDA out of memory", txt))
        dis = re.findall(r"! TTT disabled for the rest of the run[^\n]*", txt)
        if steps or oom or dis:
            fact(f"ttt step records={steps}", f"kaggle_out/{slug}/kernel_stdout.log", "TTT ran")
            fact(f"ttt OOM events={oom}", f"kaggle_out/{slug}/kernel_stdout.log", "TTT died")
            fact(f"breaker fired={len(dis)}", f"kaggle_out/{slug}/kernel_stdout.log", "global disable")


def main() -> int:
    corpus()
    runs()
    ttt()
    w = [max(len(a) for a, _, _ in rows), max(len(b) for _, b, _ in rows)]
    print(f"{'VALUE'.ljust(w[0])}  {'SOURCE'.ljust(w[1])}  NOTE")
    print("-" * (w[0] + w[1] + 30))
    for a, b, c in rows:
        print(f"{a.ljust(w[0])}  {b.ljust(w[1])}  {c}")
    print(f"\n{len(rows)} facts. Repo root: {REPO}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
