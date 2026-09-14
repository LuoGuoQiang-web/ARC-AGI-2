# Reproduction guide

This directory holds the Paper Track submission and everything needed to check it.

- `ARC_PRIZE_2026_PAPER.md` — the paper.

Every number in the paper is reproducible from this repository. Nothing depends on the
authors' machines, credentials, or luck. The commands below are grouped by what they cost:
**most of the paper is reproducible with no GPU at all**, which is the point of the
diagnostic it argues for.

---

## 0. Setup

```bash
pip install numpy                     # the only hard requirement for the CPU work
pip install kaggle                    # only needed to re-download competition data

# competition data (excluded from the repo: not ours to redistribute)
kaggle competitions download -c arc-prize-2026-arc-agi-2 -p work/arc_w1/comp_data
cd work/arc_w1/comp_data && unzip -o arc-prize-2026-arc-agi-2.zip
```

The solver itself imports `torch` / `transformers` lazily inside functions, so it can be
imported, and its logic tested, on a machine with no GPU and no deep-learning stack.

---

## 1. The data-leak and format claims (§4.4) — no GPU

```bash
cd work/arc_w1
python tools/verify_leak.py comp_data
```

Prints, straight from the competition files: the 240/240 byte-identity with the training
split, the 240/240 answer availability, the zero content overlap with the evaluation split,
and the multi-answer share of each split (7.1% public test vs 40.8% evaluation).

## 2. The pool-recall claims (§4.1) — no GPU

The raw run reports are committed, so the headline measurement does not require re-running
the model:

```bash
python tools/recompute_pool_recall.py
```

This recomputes pool recall from the stored per-task rows and prints, per run, the number of
test inputs carrying ground truth, how many had it in the pool, the pool's distinct-candidate
range, and whether the report's own aggregate agrees with its rows.

## 3. Every cited number, with its source path (§Appendix A) — no GPU

```bash
python tools/evidence_ledger.py
```

Prints each value the paper cites alongside the file it came from. If a number is not in
this output, it should not be in the paper.

## 4. The correctness claims (§3.3) — no GPU

```bash
python tests/test_model_free.py          # 38 checks
python tests/test_merge_shards.py        # 20 checks
python tests/test_engine_integration.py  # 11 checks
```

69 checks total. `T9g`/`T9g2` are the regression checks for the pool-recall denominator bug
described in §3.3: they assert that recording a task once per stage does not count its test
inputs more than once, and that the aggregate equals the per-task rows it summarises. `T9g`
fails against the pre-fix code.

## 5. Re-running the solver (GPU, hours) — optional

Not required to check the paper, but here is the exact harness behind §4.2:

```bash
python tools/exp_suite.py --dry-run            # build the notebooks, spend nothing
python tools/exp_suite.py --configs base,prob10,branch4 --limit 8 --budget 3600
```

`--limit 8` takes the first eight *sorted* task ids, so every configuration sees an identical
task subset and the comparison is paired. That is what makes §4.2 a controlled result rather
than three anecdotes. The three runs cost 4,313 s, 4,219 s and 4,212 s of Tesla T4 time.

---

## What is deliberately *not* claimed

The paper reports a negative result about one system, and this guide is written to make its limits
checkable rather than to make the result look larger than it is:

- **Pool recall is not our contribution.** It is published practice — the ARChitects' "coverage"
  curve (arXiv:2505.07859, Fig. 4), Li et al.'s "Sample+Oracle" (arXiv:2411.02272, Fig. 8), and
  Moghe & Chin's generation-bound analysis (arXiv:2607.06764). §2.4 cites all three. The paper uses
  the diagnostic; it does not invent it.
- **The split-mismatch number is a reproduction.** A Habr analysis reports 6.9% of train and 40.8%
  of eval tasks requiring two or more test inputs, against our 7.1% and 40.8% (cited in §2.5).
- **The byte-identity / zero-overlap property is claimed as inconclusive, not as priority.** No prior
  publication was found, but the search was not exhaustive, so the paper says so.
- **The 35 figure is not 35 independent trials.** The three configurations share the same eight
  tasks, so the union is 8 tasks / 11 unique test inputs. `recompute_pool_recall.py` makes that
  visible. The 95% upper bound on pool recall is **23.8%**, not 0.
- **The one claim made for novelty is the masked adaptation failure** — an OOM on step 0 that the
  pipeline records as attempted and reports as "no gain" (§4.3). Memory-bound TTT-for-ARC is well
  documented; a *silently masked* failure was not found in prior work.
- **The remedy in §5.3 is identified, not validated.** We did not get to test it within the compute
  budget, and the paper says so rather than presenting an untested fix as a result.
- **Reported accuracy is evaluation-split only.** Training-set numbers are never reported; the
  competition rules forbid it, and the public test file is training data, so a "test score" from it
  would be meaningless twice over.
