#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
kpush.py -- push a local solver script to Kaggle as a single-cell notebook.

RECONSTRUCTED on 2026-09-14 from the artefacts the original tool left behind:
the `# --- injected by kpush.py ... ---` prelude embedded in every pushed notebook
(arc26-submit-full / -shard0 / eval-validate / solver-smoke). The prelude below is
reproduced faithfully **except for two deliberate additions**, both marked in the text:

  * `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` -- fights the fragmentation that
    made LoRA TTT OOM at 14.09/14.56 GiB (see arc26_solver.assert_gpu_compatible notes);
  * an `# --- end kpush prelude ---` marker, which `--solver` validation uses to refuse
    a file that already contains the prelude.

The original two hard-won details are preserved verbatim:

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
import re
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
# Fragmentation, not capacity, is what killed LoRA TTT: peak memory reached 14.09 of
# 14.56 GiB and the 1.77 GiB activation allocation then failed with 724 MiB free.
# expandable_segments lets the allocator reuse the freed DFS cache instead of splitting
# a new segment. Must be set before torch is imported (it is imported lazily, so here is
# still early enough).
_os.environ.setdefault('PYTORCH_CUDA_ALLOC_CONF', 'expandable_segments:True')
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


def solver_version(solver_text: str) -> str:
    """Pull ``SOLVER_VERSION`` out of the solver source.

    The notebook must be able to say which code it contains. That matters here more than usual:
    a participant in this competition reported a leaderboard score appearing "from a previous
    version of the notebook" after a crash and could not tell which code had produced which
    number, and with one submission per day that ambiguity costs a day.
    """
    m = re.search(r'^SOLVER_VERSION\s*=\s*"([^"]+)"', solver_text, re.M)
    return m.group(1) if m else "unknown"


def build_header(solver_text: str, argv: str, machine_shape: str, slug: str) -> dict:
    """A markdown cell that makes the notebook self-describing.

    Records the version, the exact pinned argv, the accelerator, and the constants that actually
    determine behaviour, so a saved Kaggle version is traceable without reading the code cell.
    """
    def const(name, default="?"):
        m = re.search(rf"^{name}\s*=\s*([^\n#]+)", solver_text, re.M)
        return m.group(1).strip() if m else default

    version = solver_version(solver_text)
    lines = [
        f"# arc26 solver — version `{version}`",
        "",
        f"`{slug}` · accelerator **`{machine_shape}`** · internet **off** · competition data attached",
        "",
        "This notebook is generated from `work/arc_w1/arc26_solver.py` by `tools/kpush.py`.",
        f"The version string is also written into `report.json` as `solver_version`, so a",
        "leaderboard result can be traced back to the exact code that produced it.",
        "",
        "## Pinned arguments",
        "",
        "```",
        f"{argv.strip() or '(none)'}",
        "```",
        "",
        "## Behaviour-determining constants at this version",
        "",
        "| constant | value |",
        "|---|---|",
        f"| `SOLVER_VERSION` | `{version}` |",
        f"| `LORA_R` | `{const('LORA_R')}` |",
        f"| `LORA_ALPHA` | `{const('LORA_ALPHA')}` |",
        f"| `TTT_LR` | `{const('TTT_LR')}` |",
        f"| `TTT_WARMUP_RATIO` | `{const('TTT_WARMUP_RATIO')}` |",
        f"| `TTT_SEQ_TOKEN_BUDGET` | `{const('TTT_SEQ_TOKEN_BUDGET')}` |",
        f"| `TTT_MAX_SEQ_LENGTH` | `{const('TTT_MAX_SEQ_LENGTH')}` |",
        f"| `DFS_TOKEN_PROB_THRESHOLD` | `{const('DFS_TOKEN_PROB_THRESHOLD')}` |",
        f"| `DFS_MAX_BRANCHES_PER_BEAM` | `{const('DFS_MAX_BRANCHES_PER_BEAM')}` |",
        f"| `DFS_MAX_NODES` | `{const('DFS_MAX_NODES')}` |",
        "",
        "## Changelog",
        "",
    ]
    block = re.search(r"SOLVER_CHANGELOG[^=]*=\s*\[(.*?)\n\]", solver_text, re.S)
    body = block.group(1) if block else ""
    # Each entry is a tuple whose note is several implicitly-concatenated string literals spread
    # over multiple lines, so match the whole tuple and then join the literals inside it.
    for entry in re.finditer(r'\(\s*"([^"]+)"\s*,(.*?)\)\s*,?\s*(?=\n\s*\(|\Z)', body, re.S):
        ver = entry.group(1)
        note = " ".join(re.findall(r'"((?:[^"\\]|\\.)*)"', entry.group(2)))
        flat = " ".join(note.replace('\\"', '"').split())
        marker = " ← **this version**" if ver == version else ""
        lines.append(f"- **`{ver}`**{marker} — {flat}")
    if not body:
        lines.append("- (changelog not found in the shipped source)")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": ("\n".join(lines) + "\n").splitlines(keepends=True),
    }


def build_notebook(solver_text: str, argv: str, extras: Sequence[tuple[str, str]] = (),
                   machine_shape: str = "NvidiaL4", slug: str = "arc26-solver") -> dict:
    """Cells: a self-describing header, one `%%writefile` per extra file, then prelude + solver.

    Extras are shipped as `%%writefile` cells rather than as a Kaggle Dataset so the whole
    deliverable stays ONE notebook file with no manual upload step (fixed by R4 in
    REVIEW_2026-09-14.md: `--engine` was unusable because there was no way to get the
    engine file onto the Kaggle side).
    """
    cells = [build_header(solver_text, argv, machine_shape, slug)]
    for remote, text in extras:
        body = f"%%writefile {remote}\n{text}"
        cells.append({
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": body.splitlines(keepends=True),
        })
    argv_tokens = ""
    if argv.strip():
        argv_tokens = ", " + ", ".join(json.dumps(t) for t in argv.split())
    source = PRELUDE.format(argv=argv_tokens) + solver_text
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    })
    return {
        "cells": cells,
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
    ap.add_argument("--machine-shape", default="NvidiaL4",
                    help="accelerator shape (default NvidiaL4 = the 4xL4 machine, 4 x 22.03 GiB. "
                         "NEVER use the generic 'Gpu' (falls back to a P100, sm_60, on which "
                         "every CUDA op fails), and note the shape string is 'NvidiaL4' -- NOT "
                         "'NvidiaL4x4', which is silently downgraded to a P100 as well)")
    ap.add_argument("--public", action="store_true", help="make the kernel public (default: private)")
    ap.add_argument("--internet", action="store_true", help="enable internet (must stay OFF for scored runs)")
    ap.add_argument("--no-competition", action="store_true", help="do not attach the competition data")
    ap.add_argument("--no-model", action="store_true", help="do not attach the SFT model")
    ap.add_argument("--dry-run", action="store_true", help="build the notebook locally, do not push")
    ap.add_argument("--extra", action="append", default=[],
                    metavar="PATH[:REMOTE]",
                    help="ship another local file too, as a %%writefile cell "
                         "(default remote: /kaggle/working/<basename>)")
    ap.add_argument("--engine", default="",
                    help="shortcut: ship this symbolic engine and add "
                         "--engine /kaggle/working/<basename> to the argv")
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

    # ---- extras (e.g. the symbolic engine for --engine) --------------------------------
    extras: list[tuple[str, str]] = []
    argv = args.argv
    for spec in list(args.extra):
        local, _, remote = spec.partition(":")
        lp = Path(local)
        if not lp.exists():
            print(f"--extra file not found: {lp}")
            return 2
        remote = remote or f"/kaggle/working/{lp.name}"
        extras.append((remote, lp.read_text(encoding="utf-8")))
        print(f"  extra: {lp} -> {remote} ({lp.stat().st_size} bytes)")
    if args.engine:
        ep = Path(args.engine)
        if not ep.exists():
            print(f"--engine file not found: {ep}")
            return 2
        remote = f"/kaggle/working/{ep.name}"
        extras.append((remote, ep.read_text(encoding="utf-8")))
        if "--engine" not in argv:
            argv = (argv + f" --engine {remote}").strip()
        print(f"  engine: {ep} -> {remote} ({ep.stat().st_size} bytes)")

    nb = build_notebook(solver_text, argv, extras,
                        machine_shape=args.machine_shape, slug=args.slug)
    meta = build_metadata(args.slug, args.title or args.slug, not args.no_gpu, not args.public,
                          args.internet, None if args.no_competition else COMPETITION,
                          None if args.no_model else MODEL_SOURCE, args.machine_shape)

    build = BUILD_DIR / args.slug
    if build.exists():
        shutil.rmtree(build)
    build.mkdir(parents=True)
    (build / f"{args.slug}.ipynb").write_text(json.dumps(nb, ensure_ascii=False), encoding="utf-8")
    (build / "kernel-metadata.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    cell_source = "".join(nb["cells"][-1]["source"])   # the solver cell is last
    print(f"built  : {build}")
    print(f"  cells  : {len(nb['cells'])} ({len(extras)} extra + 1 solver)")
    print(f"  solver : {solver_path} ({len(solver_text)} chars, {solver_text.count(chr(10))+1} lines)")
    print(f"  cell   : {len(cell_source)} chars (prelude {len(cell_source)-len(solver_text)} + solver)")
    print(f"  argv   : {argv or '(none)'}")
    print(f"  gpu={meta['enable_gpu']} private={meta['is_private']} internet={meta['enable_internet']} "
          f"shape={meta['machine_shape']} competition={meta['competition_sources']} "
          f"model={meta['model_sources']}")
    if args.dry_run:
        print("\n--dry-run: not pushing. Verify the cells:")
        ok_solver = cell_source.endswith(solver_text.rstrip()) or \
            cell_source.rstrip().endswith(solver_text.rstrip())
        print(f"  solver cell ends with the solver's last line : {ok_solver}")
        # Cell 0 is the self-describing markdown header; extras follow it, the solver is last.
        header = "".join(nb["cells"][0]["source"])
        print(f"  header cell names the version               : "
              f"{solver_version(solver_text) in header}")
        print(f"  header cell lists the changelog             : "
              f"{header.count(chr(10) + '- **`') >= 2}")
        for i, (remote, text) in enumerate(extras):
            body = "".join(nb["cells"][i + 1]["source"])
            head, _, written = body.partition("\n")
            print(f"  extra cell {i}: {head!r} -> content byte-identical: {written == text}")
        print(f"  cell order: [header] + {len(extras)} extra + [solver]")
        return 0

    api = get_api()
    api.kernels_push(str(build))
    print(f"\npushed : {OWNER}/{args.slug}")
    print(f"  watch : python {Path(__file__).name} --status --slug {args.slug}")
    print(f"  log   : python {Path(__file__).name} --log --slug {args.slug}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
