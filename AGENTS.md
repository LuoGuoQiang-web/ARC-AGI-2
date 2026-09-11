# AGENTS.md

Guidance for coding agents working in this repository.

## Project

ARC Prize 2026 campaign. The deliverable code is an offline, CPU-only ARC-AGI-2 solver
(`arc_prize_v1/`) plus a ready-to-import Kaggle notebook. The campaign plan lives in
`ARC_PRIZE_2026_MASTER_BRIEF.md`; the engine design, test evidence and limitations live in
`arc_prize_v1/README.md`.

Verified competition facts (check `ARC_PRIZE_2026_MASTER_BRIEF.md` §2.6 before assuming):
submissions must be a Kaggle notebook, no internet during evaluation, exactly 2 predicted
outputs per test input, `/kaggle/working/submission.json`.

## Agent skills

### Issue tracker

Issues live as GitHub issues in `LuoGuoQiang-web/ARC-AGI-2`, driven by the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical roles keep their default names: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` at the repo root plus `docs/adr/`. See `docs/agents/domain.md`.

## Commands

```bash
python arc_prize_v1/tests/selftest.py           # engine end-to-end, expects 36/36
python arc_prize_v1/tests/run_notebook_dryrun.py # executes the generated .ipynb end to end
python arc_prize_v1/tests/eval_local.py          # measure on a real ARC-AGI-2 eval set
python arc_prize_v1/tools/build_notebook.py      # regenerate the notebook after editing the engine
python arc_prize_v1/tools/build_notebook.py --check  # fails if the notebook is stale
```

## Conventions

- The engine is **standard library only, offline, CPU only**. Don't add heavy dependencies.
- `arc_prize_v1/arc_prize_v1.py` is the source of truth; `ARC_PRIZE_2026_v1.ipynb` is generated
  from it. After editing the engine, always regenerate and let `--check` pass.
- Never commit credentials. `kaggle.json` is git-ignored — keep it that way.
- Every solver candidate must reproduce **all** demonstration pairs before it may vote.
