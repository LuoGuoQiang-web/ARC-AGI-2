#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kpush.py -- push a local solver script to Kaggle as a single-cell notebook.

RECONSTRUCTED on 2026-09-14 from the artefacts the original tool left behind:
the `# --- injected by kpush.py ... ---` prelude embedded in every pushed notebook
(arc26-submit-full / -shard0 / eval-validate / solver-smoke). The prelude below is
reproduced faithfully, including the two hard-won details it documents:

  * sys.argv must be pinned BEFORE any `if __name__ == "__main__"` guard, because a
    Kaggle notebook runs the cell with its own argv;
  * stdout/stderr must be teed into /kaggle/working/kernel_stdout.log, because Kaggle
    publishes NO execution log for a *successful* run -- that file is the only channel;
  * the tee must implement __getattr__ (isatty and friends), otherwise transformers/tqdm
    crash while loading the model.

Why a single cell: the whole solver is shipped verbatim as one code cell, so what runs on
Kaggle is byte-identical to the local file (no hidden state, no cell-order surprises).

Usage
-----
    # build only, inspect the generated notebook (no network)
    python tools/kpush.py --dry-run

    # build + push a test-split submission in two shards
    python tools/kpush.py --slug arc26-submit-shard0 --argv "--split test --num-shards 2 --shard-index 0 --time-budget-seconds 39000 --calibrate-tasks 4 --aug-train 2 --aug-infer 1"
    python tools/kpush.py --slug arc26-submit-shard1 --argv "--split test --num-shards 2 --shard-index 1 --time-budget-seconds 39000 --calibrate-tasks 4 --aug-train 2 --aug-infer 1"

    # while it runs, then afterwards
    python tools/kpush.py --status --slug arc26-submit-shard0
    python tools/kpush.py --log    --slug arc26-submit-shard0     # -> kernel_stdout.log
    python tools/kpush.py --output --slug arc26-submit-shard0     # -> report.json / submission.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARC_W1 = HERE.parent
DEFAULT_SOLVER = ARC_W1 / "arc26_solver.py"
BUILD_DIR = ARC_W1 / ".kpush_build"
OWNER = "luoguoqiang"
COMPETITION = "arc-prize-2026-arc-agi-2"
MODEL_SOURCE = "sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1"

# ---------------------------------------------------------------------------------------
# The injected prelude. Reproduced from the pushed notebooks; only the argv line varies.
# ---------------------------------------------------------------------------------------
PRELUDE = '''# --- injected by kpush.py: pin sys.argv BEFORE any __main__ guard ---
import sys as _sys, os as _os
_sys.argv = ['arc26_solver.py'{argv}]
# A successful Kaggle run publishes NO execution log; only failed runs do.
# The only reliable channel for reading a successful run is /kaggle/working,
# so tee every stream into a file there.
# __getattr__ delegates everything we do not define (isatty, fileno, encoding,
# ...) to the original stream. This is load-bearing: transformers/tqdm call
# sys.stdout.isatty(), and a tee missing it makes model loading fail with
# AttributeError: '_Tee' object has no attribute 'isatty'.
class _Tee:
    def __init__(self, primary, secondary):
        self._primary = primary
        self._secondary = secondary
    def write(self, s):
        for _st in (self._primary, self._secondary):
            try:
                _st.write(s)
            except Exception:
                pass
        return len(s)
    def flush(self):
        for _st in (self._primary, self._secondary):
            try:
                _st.flush()
            except Exception:
                pass
    def __getattr__(self, name):
        if name in ('_primary', '_secondary'):
            raise AttributeError(name)
        return getattr(self._primary, name)
try:
    _os.makedirs('/kaggle/working', exist_ok=True)
    _fh = open('/kaggle/working/kernel_stdout.log', 'w', buffering=1, encoding='utf-8')
    _sys.stdout = _Tee(_sys.stdout, _fh)
    _sys.stderr = _Tee(_sys.stderr, _fh)
    print('kpush: teeing stdout to /kaggle/working/kernel_stdout.log',
          '| isatty =', _sys.stdout.isatty())
except Exception as _e:
    print('kpush: tee setup failed:', _e)
# --- end kpush prelude ---
'''


def build_notebook(solver_text: str, argv: str) -> dict:
    """One code cell: prelude + the solver verbatim."""
    argv_tokens = ""
    if argv.strip():
        argv_tokens = ", " + ", ".join(json.dumps(t) for t in argv.split())
    source = PRELUDE.format(argv=argv_tokens) + solver_text
    return {
        "cells": [{
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": source.splitlines(keepends=True),
        }],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def build_metadata(slug: str, title: str, gpu: bool, private: bool, internet: bool,
                   competition: str | None, model: str | None,
                   machine_shape: str = "NvidiaTeslaT4") -> dict:
    # Kaggle derives the *actual* kernel slug from the title, so a title that does not
    # slugify back to `slug` silently creates a different ref (observed: title
    # "arc26 diag smoke (pool recall)" produced slug "arc26-diag-smoke-pool-recall",
    # and every later lookup by the intended slug failed). Keep them consistent by
    # deriving the title from the slug unless the caller gives an exact match.
    if not title or _slugify(title) != slug:
        title = slug.replace("-", " ")
    return {
        "id": f"{OWNER}/{slug}",
        "title": title,
        "code_file": f"{slug}.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": bool(private),
        "enable_gpu": bool(gpu),
        "enable_tpu": False,
        # Load-bearing: with only enable_gpu=true Kaggle assigns the generic "Gpu" shape
        # and may hand out a Tesla P100 (sm_60), which torch 2.10+cu128 cannot execute at
        # all -- every task then fails, is caught per-task, and the run silently emits an
        # all-fallback submission with an empty `errors` list. The author's working kernels
        # all pinned NvidiaTeslaT4; do the same by default.
        "machine_shape": machine_shape,
        "enable_internet": bool(internet),
        "dataset_sources": [],
        "competition_sources": [competition] if competition else [],
        "kernel_sources": [],
        "model_sources": [model] if model else [],
    }


def _slugify(text: str) -> str:
    """Approximate Kaggle's slug rules: lowercase, non-alphanumerics -> single dash."""
    out = []
    for ch in text.lower():
        if ch.isalnum():
            out.append(ch)
        elif out and out[-1] != "-":
            out.append("-")
    return "".join(out).strip("-")


def get_api():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except Exception as exc:  # pragma: no cover
        print(f"kaggle package not usable: {exc!r}\n  install with: python -m pip install kaggle")
        raise SystemExit(2)
    api = KaggleApi()
    api.authenticate()
    return api


def main() -> int:
    ap = argparse.ArgumentParser(description="push a local solver script to Kaggle as a notebook")
    ap.add_argument("--solver", default=str(DEFAULT_SOLVER), help="local solver .py to ship")
    ap.add_argument("--slug", default="arc26-solver-dev", help="Kaggle kernel slug (without owner)")
    ap.add_argument("--title", default="", help="kernel title (defaults to the slug)")
    ap.add_argument("--argv", default="--split evaluation --limit 3 --time-budget-seconds 1800",
                    help="arguments to pin inside the notebook")
    ap.add_argument("--no-gpu", action="store_true", help="disable the GPU accelerator")
    ap.add_argument("--machine-shape", default="NvidiaTeslaT4",
                    help="accelerator shape (default NvidiaTeslaT4; never leave it generic)")
    ap.add_argument("--public", action="store_true", help="make the kernel public (default: private)")
    ap.add_argument("--internet", action="store_true", help="enable internet (must stay OFF for scored runs)")
    ap.add_argument("--no-competition", action="store_true", help="do not attach the competition data")
    ap.add_argument("--no-model", action="store_true", help="do not attach the SFT model")
    ap.add_argument("--dry-run", action="store_true", help="build the notebook locally, do not push")
    ap.add_argument("--status", action="store_true", help="print the kernel's last-run status and exit")
    ap.add_argument("--log", action="store_true", help="download the run log (kernel_stdout.log) and exit")
    ap.add_argument("--output", action="store_true", help="download the run output files and exit")
    args = ap.parse_args()

    # ---- read-only subcommands -------------------------------------------------------
    if args.status:
        api = get_api()
        st = api.kernels_status(f"{OWNER}/{args.slug}")
        print(json.dumps(st, indent=2, default=str) if not isinstance(st, str) else st)
        return 0
    if args.log or args.output:
        api = get_api()
        out = ARC_W1 / "kaggle_out" / args.slug
        out.mkdir(parents=True, exist_ok=True)
        api.kernels_output(f"{OWNER}/{args.slug}", path=str(out), force=True, quiet=False)
        files = sorted(p.name for p in out.rglob("*") if p.is_file())
        print(f"downloaded into {out}: {files}")
        logf = out / "kernel_stdout.log"
        if logf.exists():
            lines = logf.read_text(encoding="utf-8", errors="replace").splitlines()
            print(f"\n--- kernel_stdout.log tail ({len(lines)} lines) ---")
            for l in lines[-25:]:
                print("   ", l[:160])
        reps = list(out.rglob("report.json"))
        for r in reps:
            try:
                rep = json.loads(r.read_text(encoding="utf-8"))
                print(f"\n--- {r.name} ---")
                for k in ("split", "n_tasks_total", "n_with_solutions", "solved", "score", "shard"):
                    if k in rep:
                        print(f"   {k}: {rep[k]}")
            except Exception as exc:
                print(f"   report parse failed: {exc!r}")
        return 0

    # ---- build ------------------------------------------------------------------------
    solver_path = Path(args.solver)
    if not solver_path.exists():
        print(f"solver not found: {solver_path}")
        return 2
    solver_text = solver_path.read_text(encoding="utf-8")
    if "kpush prelude" in solver_text:
        print("refusing to ship a file that already contains the prelude")
        return 2

    nb = build_notebook(solver_text, args.argv)
    meta = build_metadata(args.slug, args.title or args.slug, not args.no_gpu, not args.public,
                          args.internet, None if args.no_competition else COMPETITION,
                          None if args.no_model else MODEL_SOURCE, args.machine_shape)

    build = BUILD_DIR / args.slug
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)
    (build / f"{args.slug}.ipynb").write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")
    (build / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    cell_source = "".join(nb["cells"][0]["source"])
    print(f"built  : {build}")
    print(f"  solver : {solver_path} ({len(solver_text)} chars, {solver_text.count(chr(10))+1} lines)")
    print(f"  cell   : {len(cell_source)} chars (prelude {len(cell_source)-len(solver_text)} + solver)")
    print(f"  argv   : {args.argv or '(none)'}")
    print(f"  gpu={meta['enable_gpu']} private={meta['is_private']} internet={meta['enable_internet']} "
          f"competition={meta['competition_sources']} model={meta['model_sources']}")
    if args.dry_run:
        print("\n--dry-run: not pushing. Verify the cell matches the solver:")
        print(f"  cell ends with solver's last line: "
              f"{cell_source.endswith(solver_text.rstrip()) or cell_source.rstrip().endswith(solver_text.rstrip())}")
        return 0

    api = get_api()
    api.kernels_push(str(build))
    print(f"\npushed : {OWNER}/{args.slug}")
    print(f"  watch : python {Path(__file__).name} --status --slug {args.slug}")
    print(f"  log   : python {Path(__file__).name} --log --slug {args.slug}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
