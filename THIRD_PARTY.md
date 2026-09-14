# Licensing notes

The `LICENSE` file at the repository root contains the licence for this project's own code:
**MIT No Attribution (MIT-0)**. It is kept as the bare licence text so that licence detectors
(GitHub's included) identify it unambiguously — an earlier revision appended the notes below to
that file and GitHub reported `NOASSERTION` as a result.

## Why MIT-0 rather than MIT

ARC Prize 2026 requires that "all code and methods authored by the submitter must be made open
source under a **permissive public domain license (eg. CC0 or MIT-0)**". Plain MIT is permissive
but still *conditions* reuse on preserving the copyright notice; MIT-0 removes that condition and
is the licence the rules name explicitly. Third-party components are unaffected and stay under
their own licences.

## Third-party components

This project's own code is MIT-0. It additionally *uses* the following, which remain under their
own licences:

- **Qwen3-4B** (base model) — Apache License 2.0.
- **The fine-tuned ARC grid checkpoint `sorokin/qwen3_4b_grids15_sft139`** — ⚠️ **declares no
  licence.** Checked against the Kaggle API on 2026-09-14: the model record exposes no
  `licenseName` field, or any other licence metadata. The checkpoint is publicly downloadable and
  is a derivative of an Apache-2.0 base model, but the uploader has not stated terms.

  **This is an unresolved eligibility risk** under the ARC Prize rule that third-party methods
  "must be available under, at least, an open source license which allows public sharing". Note
  that the checkpoint is *mounted* from Kaggle at scoring time and is not redistributed here. An
  earlier revision of this repository asserted Apache-2.0 for this checkpoint; that assertion was
  not verified and has been corrected rather than left standing. Anyone reusing this work for a
  prize submission should confirm the position with the model author or the competition hosts.
- **PyTorch**, **Transformers**, **PEFT**, **NumPy** — BSD-3-Clause / Apache-2.0.

## Competition data

The ARC-AGI-2 competition data (`arc-agi_*.json`, `sample_submission.json`) is **not**
redistributed here. It is downloaded on demand via the Kaggle API and excluded by `.gitignore`.
See <https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2>.
