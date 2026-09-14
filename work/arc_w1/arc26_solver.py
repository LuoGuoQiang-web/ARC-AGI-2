#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""arc26_solver.py -- ARC-AGI-2 test-time-training (TTT) solver.

Clean-room implementation of ``work/arc_w1/ARC26_SOLVER_SPEC.md`` (Option C).
Everything here is written from that spec; no third-party solver code is copied.

Two scheduling strategies (``--schedule``)
------------------------------------------
``cascade`` (default) -- spend the scarce GPU budget where it buys coverage:

* **Stage A / cheap sweep**: prompt the already-SFT'd model for ALL tasks and run
  constrained DFS generation only (``do_ttt=False``, no LoRA, no gradients).
  Each task gets a confidence signal (augmentation agreement + best rescore NLL).
* **Stage B / refinement**: rank the tasks by Stage A confidence and run the full
  LoRA TTT pipeline (``do_ttt=True``) on the weakest ones until the budget is gone.
* **Stage C / backfill**: for any task without a confident answer, merge the
  optional ``--engine`` symbolic candidates plus the heuristic floor into the
  candidate pool, then re-select the two attempts.  Never leaves a slot blank.

``uniform`` -- the spec's original schedule: every task gets TTT immediately, its
time slice driven by the calibration phase (measured seconds/task).

Pipeline per task (spec section 2)
----------------------------------
1.  Augment the demonstration pairs (geometry + colour permutation).
2.  Optional: LoRA TTT on the augmented demonstrations (bf16 + gradient
    checkpointing -- mandatory to fit 3.63B params in 14.6 GiB).
3.  Per test input: constrained batched beam DFS over the 12 ARC tokens under
    several inference augmentations, inverse-transformed back to grid space.
4.  Merge neural + symbolic + heuristic candidates into ONE pool, rescore with
    teacher-forced NLL, and emit the two best *distinct* grids.
5.  Time is allocated dynamically, ``submission.json`` is flushed incrementally,
    and the whole run is wrapped in try/finally.

Only ``numpy`` + ``torch`` + ``transformers`` are used.  ``torch`` and
``transformers`` are imported lazily so the pure format helpers stay importable and
unit-testable without a GPU.  ``peft`` is deliberately avoided: the Kaggle image ships
``torchao 0.10.0`` which makes ``peft 0.19.1`` raise on import, so LoRA lives in this
file (:class:`LoRALinear`).

Notebook safety: a Jupyter cell has ``__name__ == "__main__"``, so the file ends
with ``if __name__ == "__main__": main()`` -- never ``raise SystemExit(...)``.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import random
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:  # torch exists both locally and on Kaggle; keep the import non-fatal anyway.
    import torch  # noqa: F401
except Exception:  # pragma: no cover - only on machines without torch
    torch = None  # type: ignore[assignment]


# ======================================================================================
# 0. Constants (spec section 1)
# ======================================================================================

# --- the 16-token alphabet -----------------------------------------------------------
# 0-9 : the ten ARC colours, one grid cell == one token
# 10  : "\n"  (row separator)
# 11  : "<|im_start|>user\n"      (single token, user turn marker)
# 12  : "<|im_start|>assistant\n" (single token, assistant turn marker)
# 13  : "<|endoftext|>"           (PAD)
# 14  : "<|im_start|>"            (unused)
# 15  : "<|im_end|>"              (EOS)
PAD_ID = 13
EOS_ID = 15
USER_TOKEN_ID = 11
ASSISTANT_TOKEN_ID = 12
NEWLINE_TOKEN_ID = 10

#: the 12 tokens allowed during constrained decoding: 10 colours + newline + EOS
ARC_TOKEN_IDS: Tuple[int, ...] = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15)

#: literal text for the two chat markers (the tokenizer maps each to ONE token)
USER = "<|im_start|>user\n"
ASSISTANT = "<|im_start|>assistant\n"
ENDOFTEXT = "<|im_end|>"

MAX_GRID_SIDE = 30
MAX_CELLS_PER_GRID = MAX_GRID_SIDE * MAX_GRID_SIDE
MAX_NEW_TOKENS_30x30 = MAX_CELLS_PER_GRID + (MAX_GRID_SIDE - 1) + 1  # 900 + 29 + EOS = 930

DEFAULT_MODEL_DIR = (
    "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1"
)
DEFAULT_DATA_DIR = "/kaggle/input/arc-prize-2026-arc-agi-2"

#: Where the competition files can actually be found, in order of preference.
#:
#: MEASURED on Kaggle (2026-09-13): a CLI-pushed kernel with
#: ``competition_sources: ["arc-prize-2026-arc-agi-2"]`` mounts the data under
#: ``/kaggle/input/competitions/<slug>/`` -- note the extra ``competitions/`` level --
#: NOT under ``/kaggle/input/<slug>/``. Model sources do mount at
#: ``/kaggle/input/models/...`` with no extra level. Rather than hard-code one layout,
#: resolve at runtime and fall back to a recursive search from ``/kaggle/input``.
DATA_DIR_CANDIDATES: Tuple[str, ...] = (
    "/kaggle/input/competitions/arc-prize-2026-arc-agi-2",
    "/kaggle/input/arc-prize-2026-arc-agi-2",
    "/kaggle/input/competitions/arc-prize-2026-agi-2",
    "/kaggle/input/arc-prize-2026-agi-2",
    "/kaggle/input",
)

DEFAULT_OUT = "/kaggle/working/submission.json"
DEFAULT_REPORT = "/kaggle/working/report.json"

# TTT hyper-parameters.
#
# These now follow the *verified* configuration of the strongest published instance of this
# recipe, rather than our earlier guesses. The NVARC 2025-winning submission notebook (pulled
# from the Kaggle API on 2026-09-14) uses:
#
#     r=256, lora_alpha=32, use_rslora=True, lora_dropout=0.0, bias="none"
#     target_modules = q,k,v,o,gate,up,down  PLUS  embed_tokens and lm_head
#     learning_rate=5e-5, lr_scheduler_type="cosine", warmup_ratio=0.1
#     optim="adamw_torch", max_grad_norm=1.0, bf16, 1 epoch, per-device batch 1
#     max_seq_length=8192, 16 augmentations per puzzle, adapters RESET per puzzle
#
# We had r=16 -- sixteen times smaller -- and no scheduler. Since the pool-recall measurement
# says our generator is the bottleneck, and adaptation is the only mechanism that can change the
# generator, under-parameterising it by 16x was not a defensible default.
#
# ⚠️ Cost: r=256 over all seven projections of a 36-layer 4B model is ~0.53B trainable
# parameters, i.e. ~2.1 GiB of fp32 adapters plus ~4.2 GiB of AdamW state on top of the ~7.3 GiB
# bf16 base. That does NOT fit a 14.56 GiB T4 and is the reason the accelerator had to be fixed
# first (see assert_gpu_compatible). Use --lora-r to dial it back on smaller devices.
LORA_R = 256
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj",
)
# Adapters on the token embedding and the output head. With a 16-token vocabulary these are
# tiny in parameter count but they let adaptation change *which* grid tokens are preferred at
# all, rather than only how the transformer mixes features.
LORA_EMBED_MODULES = ("embed_tokens", "lm_head")
TTT_LR = 5e-5
TTT_WARMUP_RATIO = 0.1     # cosine schedule with a linear warmup, as in the reference config
TTT_EPOCHS = 1
TTT_MAX_SEQ_LENGTH = 8192
# Cap on the *context* of one TTT optimizer step. Measured failure it prevents: with the whole
# demonstration set in one sequence, a 30x30 task builds an ~8192-token step, the backward
# allocation (1.77 GiB) failed on a 14.56 GiB T4, and 86 of 102 tasks routed to TTT executed zero
# optimizer steps while being reported as "no gain". On an L4 (22 GiB) the 8192-token step fits,
# so this stays a knob rather than a hard limit; 0 disables the cap.
TTT_SEQ_TOKEN_BUDGET = 0
# How many tasks in a row must fail to allocate before TTT is abandoned for the whole run. One
# task's OOM must not unadapt the remaining 200: degrade that task, keep the mechanism.
TTT_MAX_CONSECUTIVE_OOM = 3
TTT_MAX_GRAD_NORM = 1.0

# Constrained DFS (spec section 2.3)
#: Per-token log-probability floor: keep a token when its probability exceeds 0.2.
#: The spec writes this quantity as ``-log(0.2)`` (a *magnitude*); the DFS compares raw
#: log-probabilities, which are all <= 0, so the comparable value is ``log(0.2)``
#: (-1.609).  Using the positive magnitude would reject every token and silently
#: degrade the search to a single forced greedy path.  (Caught by test_neural_stub.py.)
DFS_TOKEN_LOGPROB_THRESHOLD = math.log(0.2)
DFS_TOKEN_PROB_THRESHOLD = 0.2                  # the same rule, stated as a probability
DFS_MAX_BRANCHES_PER_BEAM = 3                   # child cap (memory guard, see turbo_dfs)
DFS_MAX_NODES = 6000                            # global node budget per DFS call
DFS_DEFAULT_CACHE_BUDGET_GB = 3.0               # KV-cache budget for live beams

#: per-token, per-beam KV cache cost for the Qwen3-4B config:
#: 36 layers * 2 (K,V) * 8 kv heads * 128 head_dim * 2 bytes (bf16) = 147456 B
KV_BYTES_PER_TOKEN_PER_LAYER = 2 * 8 * 128 * 2
KV_BYTES_PER_TOKEN_PER_BEAM = 36 * KV_BYTES_PER_TOKEN_PER_LAYER

#: a symbolic candidate is a program validated against every demonstration pair
#: (spec: 96.3-100% precise *when one validates*), so it gets a small NLL bonus when
#: ranking against the neural candidates, and a default score when rescoring is not
#: possible at all.  An *unvalidated* engine prior (no program reproduced the demos)
#: gets no bonus and a deliberately poor default score: it only outranks the floor.
SYMBOLIC_NLL_BONUS = 0.5
UNSCORED_SYMBOLIC_NLL = 0.0
UNSCORED_PRIOR_NLL = 3.0
UNSCORED_NEURAL_NLL = 2.0

SOURCE_PRIORITY = {"neural": 0, "symbolic": 1, "prior": 2, "fallback": 3}

# --- cascade scheduling ---------------------------------------------------------------
CONFIDENCE_BAR = 0.6            # >= this => Stage C leaves the task alone
SYMBOLIC_CONFIDENCE = 0.9       # a validated symbolic program is strong evidence

# Set when LoRA TTT hits an unrecoverable OOM; TTT then stays off for the rest of the run.
# Why: each doomed attempt still costs ~40 s of the task's slice AND leaves the GPU too full
# for the Stage B inference that follows (observed 2026-09-14 on arc26-diag-smoke2:
# "TTT 0 steps in 38.7s", then "prompt forward failed (CUDA out of memory ... 146.81 MiB
# free)" while peak memory was already 14.09 of 14.56 GiB). The budget is far better spent
# on Stage A coverage -- that is where the loss actually is (pool_recall was 0.0).
TTT_DISABLED_REASON: Optional[str] = None
# Consecutive tasks whose whole retry ladder failed to allocate. TTT is abandoned run-wide only
# once this reaches --ttt-max-consecutive-oom. Measured cost of getting this wrong: abandoning
# on the *first* OOM left 86 of 102 "TTT" tasks unadapted (FINDINGS_2026-09-14.md section 5).
TTT_CONSECUTIVE_OOM: int = 0
STAGE_A_DEFAULT_SLICE = 45.0    # seconds per task during the cheap sweep (before measurement)
STAGE_A_CALIBRATION_SLICE = 90.0
STAGE_A_MIN_SLICE = 8.0
STAGE_B_MIN_SLICE = 45.0
STAGE_C_RESERVE_MIN = 300.0     # always leave room for the symbolic backfill

SPLIT_FILES: Dict[str, Tuple[str, Optional[str]]] = {
    "test": ("arc-agi_test_challenges.json", None),
    "training": ("arc-agi_training_challenges.json", "arc-agi_training_solutions.json"),
    "evaluation": (
        "arc-agi_evaluation_challenges.json",
        "arc-agi_evaluation_solutions.json",
    ),
}
#: alternative spellings that other mirrors of the dataset use
SPLIT_FILE_ALIASES: Dict[str, Tuple[str, ...]] = {
    "arc-agi_test_challenges.json": ("test_challenges.json",),
    "arc-agi_training_challenges.json": ("training_challenges.json",),
    "arc-agi_training_solutions.json": ("training_solutions.json",),
    "arc-agi_evaluation_challenges.json": ("evaluation_challenges.json",),
    "arc-agi_evaluation_solutions.json": ("evaluation_solutions.json",),
}


# ======================================================================================
# 1. Format helpers -- PURE (no torch / no tokenizer needed), unit-testable
# ======================================================================================

def convert_grid_to_string(grid: Any) -> str:
    """Serialise a grid exactly as the model was trained to read/write it.

    One digit character per cell, one ``\\n`` per row, trailing newline stripped
    (spec section 1.3 -- must stay byte-for-byte identical).
    """
    text = ""
    for row in grid:
        for cell in row:
            text += str(int(cell))
        text += "\n"
    return text.rstrip("\n")


def validate_grid(grid: Any) -> Optional[np.ndarray]:
    """Return ``grid`` as a rectangular int array of shape 1..30 x 1..30, else ``None``.

    Rejects: None, empty, ragged rows, non-integer cells, colours outside 0..9 and
    any side longer than 30.  Final gate before a grid reaches ``submission.json``.
    """
    if grid is None:
        return None
    if isinstance(grid, np.ndarray):
        arr = grid
    elif isinstance(grid, (list, tuple)):
        rows = []
        for row in grid:
            if isinstance(row, np.ndarray):
                row = row.tolist()
            if not isinstance(row, (list, tuple)) or len(row) == 0:
                return None
            rows.append(list(row))
        if not rows:
            return None
        width = len(rows[0])
        for row in rows:  # ragged input -> reject
            if len(row) != width:
                return None
        try:
            arr = np.array(rows)
        except Exception:
            return None
    else:
        return None
    if arr.ndim != 2 or arr.size == 0:
        return None
    if not (1 <= arr.shape[0] <= MAX_GRID_SIDE):
        return None
    if not (1 <= arr.shape[1] <= MAX_GRID_SIDE):
        return None
    try:
        out = arr.astype(int)
    except Exception:
        return None
    if out.min() < 0 or out.max() > 9:
        return None
    return np.ascontiguousarray(out)


def parse_grid_string(text: Any, limit_rows: int = MAX_GRID_SIDE,
                      recover_truncated: bool = False) -> Optional[np.ndarray]:
    """Parse decoded model output back into a grid (spec section 1.5).

    Blank lines are dropped, at most ``limit_rows`` rows are kept and non-digit
    characters are ignored (faithful to the reference implementation).  A ragged /
    short row makes the whole parse fail (``None``) -- that is the validator the
    smoke test exercises.
    """
    if not isinstance(text, str):
        return None
    lines = text.strip().split("\n")
    rows = [[int(ch) for ch in line if ch.isdigit()] for line in lines]
    rows = [r for r in rows if r]
    if not rows:
        return None
    if limit_rows is not None and limit_rows > 0 and len(rows) > limit_rows:
        rows = rows[:limit_rows]
    width = len(rows[0])
    if width < 1 or width > MAX_GRID_SIDE:
        return None
    if recover_truncated:
        # A beam cut off by the time budget ends mid-row, leaving a short final row.
        # Truncating at the first incomplete row salvages a usable grid instead of
        # throwing the candidate away: an unparseable answer scores zero anyway, so a
        # slightly short grid is strictly better -- and it keeps the pool non-empty,
        # which is what the cascade depends on. Strict parsing stays the default so the
        # validator keeps rejecting genuinely malformed output.
        good = 0
        for row in rows:
            if len(row) != width:
                break
            good += 1
        rows = rows[:good]
        if not rows:
            return None
    else:
        for row in rows:
            if len(row) != width:
                return None  # ragged / short row -> invalid output
    try:
        arr = np.array(rows, dtype=int)
    except Exception:
        return None
    if arr.ndim != 2 or arr.size == 0:
        return None
    if not all(1 <= s <= MAX_GRID_SIDE for s in arr.shape):
        return None
    if arr.min() < 0 or arr.max() > 9:
        return None
    return np.ascontiguousarray(arr)


def decode_arc_tokens(tokens: Sequence[int]) -> str:
    """Pure decoder for the 16-token alphabet (no tokenizer object required).

    Equivalent to ``tokenizer.decode(tokens)`` for well-formed ARC token runs,
    which is what makes :func:`tokens_to_array` testable without transformers.
    """
    out: List[str] = []
    for t in tokens:
        t = int(t)
        if 0 <= t <= 9:
            out.append(str(t))
        elif t == NEWLINE_TOKEN_ID:
            out.append("\n")
        # 11/12/13/14/15 carry no grid content
    return "".join(out)


def tokens_to_array(tokens: Sequence[int], limit_rows: int = MAX_GRID_SIDE,
                    tokenizer: Any = None,
                    recover_truncated: bool = False) -> Optional[np.ndarray]:
    """Token ids -> grid array, or ``None`` when the output is malformed.

    ``tokens`` are the continuation ids produced by the constrained DFS.  The
    trailing EOS is dropped before decoding (spec section 1.5); if the beam was cut
    off by ``max_new_tokens`` instead, the final token is a real digit and is kept
    -- otherwise a complete 30x30 grid would lose its last cell.

    Pass ``tokenizer`` to decode with the real tokenizer; without it the pure
    :func:`decode_arc_tokens` fallback is used.
    """
    if tokens is None:
        return None
    if isinstance(tokens, str):
        return parse_grid_string(tokens, limit_rows=limit_rows,
                                 recover_truncated=recover_truncated)
    toks = [int(t) for t in tokens]
    if toks and toks[-1] == EOS_ID:
        toks = toks[:-1]
    if tokenizer is not None:
        try:
            text = tokenizer.decode(toks)
        except Exception:
            text = decode_arc_tokens(toks)
    else:
        text = decode_arc_tokens(toks)
    return parse_grid_string(text, limit_rows=limit_rows,
                             recover_truncated=recover_truncated)


def fmt_query(test_input_grid: Any) -> str:
    """User turn carrying the test input + the assistant prompt (spec 1.4)."""
    return USER + convert_grid_to_string(test_input_grid) + ENDOFTEXT + ASSISTANT


def fmt_reply(output_grid: Any) -> str:
    """Assistant answer text: grid + EOS (spec 1.4)."""
    return convert_grid_to_string(output_grid) + ENDOFTEXT


def fmt_train(demos: Sequence[dict], test_input_grid: Any) -> str:
    """Full prompt: every demonstration round followed by the query round (1.4)."""
    text = ""
    for d in demos:
        text += USER + convert_grid_to_string(d["input"]) + ENDOFTEXT
        text += ASSISTANT + convert_grid_to_string(d["output"]) + ENDOFTEXT
    text += fmt_query(test_input_grid)
    return text


def max_new_tokens_for(tokenizer: Any = None) -> int:
    """Maximum generation length: a full 30x30 reply + EOS (spec 1.5)."""
    if tokenizer is None:
        return MAX_NEW_TOKENS_30x30
    try:
        text = fmt_reply(np.zeros([MAX_GRID_SIDE, MAX_GRID_SIDE], dtype=int))
        return int(len(_encode_text(tokenizer, text)) + 1)
    except Exception:
        return MAX_NEW_TOKENS_30x30


def _encode_text(tokenizer: Any, text: str) -> List[int]:
    """Tokenize without adding special tokens (the turn markers are in-band)."""
    enc = tokenizer(text, add_special_tokens=False)
    if isinstance(enc, dict):
        return list(enc["input_ids"])
    return list(getattr(enc, "input_ids", enc))


def encode_text(tokenizer: Any, text: str) -> List[int]:
    """Public wrapper around :func:`_encode_text`."""
    return _encode_text(tokenizer, text)


def grid_to_json(grid: Any) -> List[List[int]]:
    """Grid -> plain nested lists of ints (submission/JSON friendly)."""
    arr = validate_grid(grid)
    if arr is None:
        return [[0]]
    return [[int(v) for v in row] for row in arr.tolist()]


# ======================================================================================
# 2. Augmentation (spec section 2.5) -- geometric + colour, with exact inverses
# ======================================================================================

@dataclass(frozen=True)
class AugKey:
    """One composed augmentation: rot90 -> horizontal flip -> transpose -> colour perm.

    ``perm`` maps old colour -> new colour (``perm[c]``); ``None`` leaves colours
    untouched.  The composition order is fixed so the inverse is exact.
    """

    rot: int = 0
    flip: bool = False
    transpose: bool = False
    perm: Optional[Tuple[int, ...]] = None

    def to_json(self) -> List[Any]:
        return [int(self.rot), bool(self.flip), bool(self.transpose),
                list(self.perm) if self.perm is not None else None]

    @staticmethod
    def identity() -> "AugKey":
        return AugKey(0, False, False, None)

    @property
    def is_identity(self) -> bool:
        return self.rot == 0 and not self.flip and not self.transpose and self.perm is None


def apply_augment(grid: Any, key: AugKey) -> np.ndarray:
    """Apply ``key`` to a grid (forward direction)."""
    g = np.asarray(grid, dtype=int)
    if key.rot:
        g = np.rot90(g, k=int(key.rot) % 4)
    if key.flip:
        g = np.fliplr(g)
    if key.transpose:
        g = g.T
    if key.perm is not None:
        lut = np.asarray(key.perm, dtype=int)
        g = lut[g]
    return np.ascontiguousarray(g)


def invert_augment(grid: Any, key: AugKey) -> np.ndarray:
    """Undo :func:`apply_augment` (inverse ops, reverse order)."""
    g = np.asarray(grid, dtype=int)
    if key.perm is not None:
        inv = np.empty(10, dtype=int)
        inv[np.asarray(key.perm, dtype=int)] = np.arange(10, dtype=int)
        g = inv[g]
    if key.transpose:
        g = g.T
    if key.flip:
        g = np.fliplr(g)
    if key.rot:
        g = np.rot90(g, k=(-int(key.rot)) % 4)
    return np.ascontiguousarray(g)


def random_augment_key(rng: random.Random, allow_color: bool = True) -> AugKey:
    """Sample a random composed augmentation (identity is possible but rare)."""
    rot = rng.randrange(4)
    flip = rng.random() < 0.5
    trans = rng.random() < 0.5
    perm: Optional[Tuple[int, ...]] = None
    if allow_color and rng.random() < 0.75:
        p = list(range(10))
        rng.shuffle(p)
        perm = tuple(p)
    return AugKey(rot=rot, flip=flip, transpose=trans, perm=perm)


def augment_demos(demos: Sequence[dict], key: AugKey) -> List[dict]:
    """Apply the same key to every input/output of every demonstration (2.5).

    Consistency across the whole task is what makes the learned mapping invariant.
    """
    return [
        {"input": apply_augment(d["input"], key), "output": apply_augment(d["output"], key)}
        for d in demos
    ]


# ======================================================================================
# 3. Data loading (spec section 2.1 / section 4)
# ======================================================================================

def utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json_atomic(path: Any, obj: Any, indent: Optional[int] = None) -> None:
    """Write JSON via a temp file + replace, so a kill never truncates the output."""
    p = Path(path)
    if p.parent and str(p.parent):
        p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, p)


def _find_file(root: Path, name: str, max_depth: int = 3) -> Optional[Path]:
    """Find ``name`` (or one of its aliases) under ``root``."""
    if not root.exists():
        return None
    candidates = [name] + list(SPLIT_FILE_ALIASES.get(name, ()))
    direct = root / name
    if direct.is_file():
        return direct
    for cand in candidates:
        for path in root.rglob(cand):
            if path.is_file():
                return path
    # bounded manual walk (rglob can be slow on huge dataset mounts)
    stack: List[Tuple[Path, int]] = [(root, 0)]
    while stack:
        cur, depth = stack.pop()
        if depth > max_depth:
            continue
        try:
            entries = list(cur.iterdir())
        except Exception:
            continue
        for entry in entries:
            if entry.is_file() and entry.name in candidates:
                return entry
            if entry.is_dir():
                stack.append((entry, depth + 1))
    return None


def load_split(data_dir: Any, split: str
               ) -> Tuple[Dict[str, dict], Optional[Dict[str, list]], Dict[str, Any]]:
    """Load challenges (+ solutions when the split has them).

    Returns ``(tasks, solutions_or_None, info)``.  ``tasks[task_id]`` has the keys
    ``train`` (list of ``{"input": grid, "output": grid}``) and ``test`` (list of
    ``{"input": grid}``); ``solutions[task_id][i]`` is the expected output for
    ``tasks[task_id]["test"][i]["input"]`` (attached by index, spec section 4).
    """
    if split not in SPLIT_FILES:
        raise ValueError(f"unknown --split {split!r}; expected one of {sorted(SPLIT_FILES)}")
    root = Path(data_dir)
    challenge_name, solution_name = SPLIT_FILES[split]
    ch_path = _find_file(root, challenge_name) if root.exists() else None
    if ch_path is None:
        # ``--data-dir`` is missing or does not contain this split; auto-resolve
        # against the known Kaggle mount layouts before giving up.
        for candidate in DATA_DIR_CANDIDATES:
            cand_root = Path(candidate)
            if not cand_root.exists():
                continue
            found = _find_file(cand_root, challenge_name)
            if found is not None:
                print(f"[data] {data_dir!r} did not yield {challenge_name}; "
                      f"auto-resolved to {cand_root}", flush=True)
                root, ch_path = cand_root, found
                break
    if ch_path is None:
        raise FileNotFoundError(
            f"{challenge_name} not found under {data_dir!r} nor any of "
            f"{list(DATA_DIR_CANDIDATES)}")
    with open(ch_path, "r", encoding="utf-8") as fh:
        tasks = json.load(fh)
    solutions: Optional[Dict[str, list]] = None
    sol_path: Optional[Path] = None
    if solution_name is not None:
        sol_path = _find_file(root, solution_name)
        if sol_path is not None:
            with open(sol_path, "r", encoding="utf-8") as fh:
                solutions = json.load(fh)
    info = {
        "data_dir": str(root),
        "split": split,
        "challenges_path": str(ch_path),
        "solutions_path": str(sol_path) if sol_path is not None else None,
        "n_tasks": len(tasks),
    }
    return tasks, solutions, info


# ======================================================================================
# 4. Candidate containers
# ======================================================================================

@dataclass
class Candidate:
    """One grid proposed for one test input."""

    grid: np.ndarray
    source: str = "neural"           # neural | symbolic | fallback
    nll: float = float("inf")        # teacher-forced mean NLL (lower is better)
    augment: str = "identity"
    note: str = ""

    @property
    def sort_key(self) -> float:
        """Ranking key: NLL minus a bonus for engine-validated programs."""
        bonus = SYMBOLIC_NLL_BONUS if self.source == "symbolic" else 0.0
        return float(self.nll) - bonus

    @staticmethod
    def make(grid: Any, source: str, **kw: Any) -> Optional["Candidate"]:
        arr = validate_grid(grid)
        if arr is None:
            return None
        return Candidate(grid=arr, source=source, **kw)


@dataclass
class TaskResult:
    """Everything the driver loop needs to know about one finished task."""

    attempts: List[Dict[str, np.ndarray]] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    n_candidates: int = 0
    seconds: float = 0.0
    error: str = ""
    ttt_steps: int = 0
    timed_out: bool = False
    n_correct_1: int = 0
    n_correct_2: int = 0
    n_with_truth: int = 0
    # --- confidence bookkeeping (cascade scheduling) --------------------------------
    confidence: float = 0.0
    agree_max: int = 0
    n_distinct: int = 0
    n_aug_used: int = 0
    best_nll: float = float("inf")
    best_key: float = float("inf")
    has_symbolic: bool = False
    # --- pool-recall diagnostics (are we losing to generation or to selection?) ------
    n_truth_in_pool: int = 0
    n_truth_top1: int = 0
    truth_ranks: List[int] = field(default_factory=list)
    # --- confidence components, recorded so the formula can be tuned from data ------
    conf_agree: float = 0.0
    conf_coverage: float = 0.0
    stage: str = ""
    engine_source: str = ""      # "search" = a program validated, "prior" = none did
    engine_program: str = ""
    pools: List[List[Candidate]] = field(default_factory=list)
    fallbacks: List[Tuple[np.ndarray, np.ndarray]] = field(default_factory=list)

    @property
    def confident(self) -> bool:
        return self.confidence >= CONFIDENCE_BAR


# ======================================================================================
# 5. Model loading + LoRA TTT (spec section 2.2)
# ======================================================================================

def describe_gpu() -> str:
    if torch is None:
        return "torch-unavailable"
    try:
        if not torch.cuda.is_available():
            return "cpu"
        names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
        return "; ".join(names) if names else "cuda-unknown"
    except Exception as exc:  # pragma: no cover
        return f"error: {exc}"


def assert_gpu_compatible() -> str:
    """Abort loudly when the assigned GPU cannot execute this PyTorch build.

    Kaggle assigns a machine shape per run. With the generic ``Gpu`` shape it handed us a
    Tesla P100 (compute capability 6.0 / sm_60) while the shipped torch 2.10+cu128 only
    carries sm_70..sm_120. Every CUDA op then fails with ``cudaErrorNoKernelImageForDevice``;
    because each task is wrapped in a per-task handler, the run still "succeeds" and
    silently emits an all-fallback submission with an **empty** ``errors`` list -- a wasted
    submission slot that is invisible in the report. Observed 2026-09-14 on
    ``arc26-diag-smoke`` (0/2, every attempt a fallback).

    Request ``machine_shape = NvidiaTeslaT4`` (``tools/kpush.py`` does this by default) and
    keep this guard as the seatbelt. Returns the device description for logging.

    ⚠️ The original implementation gated on set membership in ``torch.cuda.get_arch_list()``.
    That was wrong and expensive: ``get_arch_list()`` lists the cubins compiled into the wheel
    and this image's list omits ``sm_89``, so the guard rejected the **NVIDIA L4** -- the 4xL4
    machine the competition offers, 4 x 22.03 GiB -- and reported the best hardware available
    as unusable. Measured 2026-09-14 with the guard bypassed: allocation, fp32 matmul, autograd,
    bf16 autocast and an 8192-token forward all succeed on the L4 (PyTorch's own gate is a
    *range* check, and sm_89 is in range, so it JITs from the PTX the wheel ships). The guard
    now runs a real op instead of reading a manifest. Request ``NvidiaL4``.
    """
    if torch is None:
        raise RuntimeError("torch is not importable in this environment")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available: this solver needs a GPU "
                           "(a 3.63B bf16 model has no CPU fallback inside the budget)")
    described = []
    for i in range(torch.cuda.device_count()):
        major, minor = torch.cuda.get_device_capability(i)
        name = torch.cuda.get_device_name(i)
        sm = f"sm_{major}{minor}"
        try:
            arch_list = list(torch.cuda.get_arch_list())
        except Exception:
            arch_list = []
        if arch_list and sm not in arch_list:
            # Informational only. The list is the set of cubins compiled into the wheel; it is
            # NOT the set of devices that work, because a device inside torch's supported range
            # runs by JIT-ing the PTX that ships alongside. Testing this list as a hard gate is
            # what made this function reject the NVIDIA L4 (sm_89 is absent from the list) --
            # i.e. reject the 4xL4 machine with 88 GiB that this competition actually offers,
            # while we ran on a single 14.56 GiB T4. The smoke test below decides instead.
            print(f"[gpu] note: {sm} absent from compiled cubins {arch_list}; "
                  f"relying on PTX JIT -- verifying with a real op", flush=True)
        try:
            dev = torch.device(f"cuda:{i}")
            a = torch.ones(256, 256, device=dev)
            _ = float((a @ a).sum())
            x = torch.randn(64, 64, device=dev, requires_grad=True)
            (x @ x).pow(2).mean().backward()
            torch.cuda.synchronize(i)
            del a, x
        except Exception as exc:
            raise RuntimeError(
                f"GPU {i} = {name} ({sm}) cannot execute this PyTorch build: "
                f"{type(exc).__name__}: {str(exc)[:200]}. Every CUDA op would fail and the "
                f"run would silently emit an all-fallback submission. Request a supported "
                f"machine_shape (tools/kpush.py defaults to NvidiaL4)."
            ) from exc
        described.append(f"{name} ({sm})")
    return "; ".join(described)


def _import_ml():
    """Lazy import of ``transformers`` so the pure helpers stay testable.

    ``peft`` is deliberately NOT imported. The Kaggle image ships ``torchao 0.10.0``,
    which makes ``peft 0.19.1`` raise the moment it is imported::

        ImportError: Found an incompatible version of torchao. Found version 0.10.0,
        but only versions above 0.16.0 are supported

    That is not something we can fix (competition notebooks have no internet), so LoRA
    is implemented in this module instead -- see :class:`LoRALinear`. It is ~40 lines,
    has no dependency, and keeps the whole thing licence-clean.
    """
    import transformers  # noqa: WPS433
    return transformers


def load_model_and_tokenizer(model_dir: str, attn_impl: Optional[str] = None):
    """Load the 3.63B bf16 Qwen3ForCausalLM on GPU 0 plus its 16-token tokenizer."""
    if torch is None:
        raise RuntimeError("torch is not importable in this environment")
    # Fail fast on an incompatible accelerator *before* spending a model load on it.
    assert_gpu_compatible()
    _import_ml()
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: WPS433

    if not Path(model_dir).exists():
        raise FileNotFoundError(f"model dir not found: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    impl = attn_impl or os.environ.get("ARC26_ATTN_IMPL", "sdpa")
    common: Dict[str, Any] = dict(device_map="cuda:0", low_cpu_mem_usage=True)
    if impl:
        common["attn_implementation"] = impl
    try:  # transformers >= 5 spells it `dtype`
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16, **common)
    except TypeError:  # pragma: no cover - older signature
        model = AutoModelForCausalLM.from_pretrained(
            model_dir, torch_dtype=torch.bfloat16, **common
        )
    model.eval()
    try:
        model.config.use_cache = True
    except Exception:
        pass
    return model, tokenizer


def _target_module_names(model: Any) -> List[str]:
    """Keep only LoRA target names that actually exist in this model."""
    present = {name.split(".")[-1] for name, _ in model.named_modules()}
    return [m for m in LORA_TARGET_MODULES if m in present]


if torch is not None:
    class LoRALinear(torch.nn.Module):
        """A LoRA adapter wrapped around a frozen ``nn.Linear``.

        ``y = W x + scale * B (A x)`` with ``A`` kaiming-initialised and ``B`` zero, so
        the adapter starts as an exact no-op and the SFT base behaviour is preserved
        until TTT actually moves it.

        ``scale`` follows rsLoRA (``alpha / sqrt(r)``) to match the configuration the
        reference run used. Both factors are kept in float32 for optimizer stability;
        the matmuls run in bf16 under the autocast context that ``run_ttt`` opens.
        """

        def __init__(self, base: Any, r: int, alpha: float, rs_lora: bool = True):
            super().__init__()
            self.base = base
            for param in self.base.parameters():
                param.requires_grad_(False)
            self.r = int(r)
            self.scale = float(alpha) / (math.sqrt(self.r) if rs_lora else self.r)
            # The adapters must be created ON the base weight's device. The base model is
            # loaded with device_map="cuda:0", which only relocates the parameters that
            # exist at load time -- anything allocated afterwards stays on CPU and every
            # TTT step then dies with:
            #   Expected all tensors to be on the same device, but got mat2 is on cpu
            device = base.weight.device
            self.lora_A = torch.nn.Parameter(
                torch.zeros(self.r, base.in_features, dtype=torch.float32, device=device)
            )
            self.lora_B = torch.nn.Parameter(
                torch.zeros(base.out_features, self.r, dtype=torch.float32, device=device)
            )
            torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        def forward(self, x: Any) -> Any:
            out = self.base(x)
            delta = (x @ self.lora_A.t()) @ self.lora_B.t()
            return out + self.scale * delta.to(out.dtype)

    class LoRAEmbedding(torch.nn.Module):
        """A LoRA adapter wrapped around a frozen ``nn.Embedding``.

        The reference configuration puts adapters on ``embed_tokens`` and ``lm_head`` as well as
        on the attention and MLP projections. With this model's 16-token vocabulary those two
        are small in parameter count (16 x r + r x hidden) but qualitatively different from the
        rest: adapting a projection re-mixes features, whereas adapting the embedding/head
        changes which *grid tokens* the model prefers at all. For a task whose answer is a grid,
        that is the more direct lever, so it is worth the few million parameters.

        ``y = E[x] + scale * B (A[x])`` -- a per-token lookup of the same low-rank delta.
        """

        def __init__(self, base: Any, r: int, alpha: float, rs_lora: bool = True):
            super().__init__()
            self.base = base
            for param in self.base.parameters():
                param.requires_grad_(False)
            self.r = int(r)
            self.scale = float(alpha) / (math.sqrt(self.r) if rs_lora else self.r)
            num_embeddings, dim = base.weight.shape
            device = base.weight.device
            self.lora_A = torch.nn.Parameter(
                torch.zeros(num_embeddings, self.r, dtype=torch.float32, device=device)
            )
            self.lora_B = torch.nn.Parameter(
                torch.zeros(self.r, dim, dtype=torch.float32, device=device)
            )
            torch.nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

        def forward(self, x: Any) -> Any:
            out = self.base(x)
            delta = (self.lora_A[x] @ self.lora_B)
            return out + self.scale * delta.to(out.dtype)


def _wrap_lora_modules(model: Any, targets: Sequence[str]) -> List[str]:
    """Replace target ``nn.Linear``/``nn.Embedding`` submodules with LoRA wrappers, in place.

    ``embed_tokens`` and ``lm_head`` are listed separately in ``LORA_EMBED_MODULES`` because
    they are Embedding/Linear rather than the attention and MLP projections, and because they
    are optional: a model may tie or omit them.
    """
    target_set = set(targets) | set(LORA_EMBED_MODULES)
    to_wrap: List[str] = []
    for name, module in model.named_modules():
        if name.split(".")[-1] not in target_set:
            continue
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
            to_wrap.append(name)
    for name in to_wrap:
        parent_path, _, leaf = name.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        base = getattr(parent, leaf)
        wrapper = (LoRAEmbedding if isinstance(base, torch.nn.Embedding) else LoRALinear)
        setattr(parent, leaf, wrapper(base, LORA_R, LORA_ALPHA, True))
    return to_wrap


def attach_lora(model: Any) -> Any:
    """Freeze the base weights and inject fresh LoRA adapters, in place.

    Returns the same model object, so every other code path (``run_ttt``, generation)
    keeps working unchanged. Gradient checkpointing is enabled because it is mandatory
    to fit seq length 8192 of a 3.63B model in 14.6 GiB.
    """
    if torch is None:
        raise RuntimeError("torch is not importable in this environment")
    for param in model.parameters():
        param.requires_grad_(False)

    targets = _target_module_names(model) or list(LORA_TARGET_MODULES)
    wrapped = _wrap_lora_modules(model, targets)
    if not wrapped:
        raise RuntimeError(f"no LoRA target modules found among {targets}")

    gc_ok = False
    for kwargs in ({"gradient_checkpointing_kwargs": {"use_reentrant": False}}, {}):
        try:
            model.gradient_checkpointing_enable(**kwargs)
            gc_ok = True
            break
        except Exception:
            continue
    if not gc_ok:
        # Not fatal, but at seq length 8192 a 3.63B model will not fit 14.6 GiB without it.
        print("[lora] WARNING: could not enable gradient checkpointing; "
              "memory may not fit at seq 8192", flush=True)
    try:
        # With a frozen embedding, checkpointed layers need an input that requires grad
        # or no gradient reaches the adapters at all.
        model.enable_input_require_grads()
    except Exception:
        pass
    try:
        model.config.use_cache = False
    except Exception:
        pass

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    devices = sorted({str(p.device) for p in model.parameters() if p.requires_grad})
    # Adapters are fp32 and AdamW keeps two fp32 moments per parameter, so the optimiser alone
    # costs 12 bytes per trainable parameter on top of the bf16 base model. At rank 256 over all
    # seven projections that is ~0.53B trainable -> ~6.4 GiB, which is why this recipe needs an
    # L4 and not a 14.56 GiB T4. Report it before the first step so an over-large rank is a
    # visible number rather than a silent OOM three minutes later.
    opt_gib = ttt_optimiser_gib(trainable)
    print(f"[lora] wrapped {len(wrapped)} modules, trainable params {trainable / 1e6:.2f}M "
          f"(r={LORA_R}, scale={LORA_ALPHA / math.sqrt(LORA_R):.2f}, "
          f"device={','.join(devices)})", flush=True)
    print(f"[lora] optimiser+adapters will need ~{opt_gib:.2f} GiB on top of the base model",
          flush=True)
    if torch is not None and torch.cuda.is_available():
        try:
            free_b, total_b = torch.cuda.mem_get_info()
            free_gib = free_b / 2 ** 30
            print(f"[lora] device free {free_gib:.2f} GiB of {total_b / 2 ** 30:.2f} GiB",
                  flush=True)
            if opt_gib > free_gib:
                raise RuntimeError(
                    f"LoRA rank {LORA_R} needs ~{opt_gib:.2f} GiB for adapters and optimiser "
                    f"state but only {free_gib:.2f} GiB is free on this device. This is the "
                    f"configuration error that silently wasted earlier runs: the step would "
                    f"OOM, the task would be recorded as 'TTT attempted, no gain', and the "
                    f"result would be reported as a statement about model capability. Either "
                    f"lower --lora-r or use a bigger accelerator (machine_shape=NvidiaL4)."
                )
        except RuntimeError:
            raise
        except Exception:
            pass
    return model


def detach_lora(model: Any) -> None:
    """Remove the LoRA adapters, restoring the original frozen ``nn.Linear`` modules."""
    if torch is not None and model is not None:
        to_restore: List[Tuple[str, Any]] = [
            (name, module.base)
            for name, module in model.named_modules()
            if isinstance(module, (LoRALinear, LoRAEmbedding))
        ]
        for name, base in to_restore:
            parent_path, _, leaf = name.rpartition(".")
            parent = model.get_submodule(parent_path) if parent_path else model
            setattr(parent, leaf, base)
        try:
            model.gradient_checkpointing_disable()
        except Exception:
            pass
        try:
            model.config.use_cache = True
        except Exception:
            pass
    gc.collect()
    if torch is not None:
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def build_ttt_sequences(train_demos: Sequence[dict], tokenizer: Any, n_aug: int,
                        max_seq_length: int, rng: random.Random,
                        seq_token_budget: int = 0
                        ) -> Tuple[List[List[int]], List[AugKey]]:
    """Augmented TTT sequences (spec 2.1 step 2).

    For augmentation ``k`` the whole demonstration set is transformed with one key
    and the ``k % len(demos)``-th demonstration becomes the *target*: the sequence
    is exactly ``fmt_train(other_demos, target_input) + fmt_reply(target_output)``,
    i.e. the same shape as an inference prompt plus its answer.  Rotating which demo
    is the target means every demonstration gets predicted, under a different
    augmentation each time.

    ``seq_token_budget`` (0 = disabled) caps the *context* so that one optimizer step fits in
    memory.  This matters more than it looks: with the whole demonstration set packed into a
    single sequence, a task whose grids are 30x30 produces one ~8192-token sequence, so a
    single step is a full forward+backward over a 3.63B model -- measured, that allocation
    fails on a 14.56 GiB T4 and adaptation silently never happens (`FINDINGS_2026-09-14.md`
    section 5).  Context demonstrations are dropped from the *front* (furthest from the target)
    until the sequence fits, so the target pair -- which is what actually carries the loss --
    always survives intact.  Truncating the token stream instead would cut a grid in half and
    leave the model fitting half a rectangle, so it is only the last resort.
    """
    seqs: List[List[int]] = []
    keys: List[AugKey] = []
    demos = list(train_demos)
    if not demos or n_aug <= 0:
        return seqs, keys
    m = len(demos)
    budget = int(seq_token_budget) if seq_token_budget else 0
    for k in range(int(n_aug)):
        key = AugKey.identity() if k == 0 else random_augment_key(rng, allow_color=True)
        aug = augment_demos(demos, key)
        target = k % m
        others = [aug[i] for i in range(m) if i != target]
        ids: List[int] = []
        # Drop context demos from the front until the step fits the budget.
        lo = 0
        while True:
            text = (fmt_train(others[lo:], aug[target]["input"])
                    + fmt_reply(aug[target]["output"]))
            ids = _encode_text(tokenizer, text)
            if not ids:
                break
            over_budget = budget and len(ids) > budget
            if not over_budget or lo >= len(others):
                break
            lo += 1
        if not ids:
            continue
        if len(ids) > max_seq_length:
            ids = ids[-max_seq_length:]
        seqs.append(ids)
        keys.append(key)
    return seqs, keys


def ttt_optimiser_gib(trainable_params: int) -> float:
    """Memory the TTT adapters and their AdamW state need, in GiB.

    Adapters are fp32 (4 bytes/param) and AdamW keeps two fp32 moments (8 bytes/param), so the
    cost is 12 bytes per trainable parameter. Pure arithmetic, so the decision "does this rank
    fit here" is testable without a GPU -- which matters because getting it wrong is invisible:
    the step OOMs, the task is recorded as 'TTT attempted, no gain', and the run reports a
    capability result where a configuration error occurred.
    """
    return max(0, int(trainable_params)) * 12 / 2 ** 30


def ttt_lr_at(step: int, total: int, base_lr: float = TTT_LR,
              warmup_ratio: float = TTT_WARMUP_RATIO) -> float:
    """Learning rate for TTT step ``step`` (0-based) out of ``total``: warmup then cosine decay.

    Matches the reference configuration (``lr_scheduler_type="cosine"``, ``warmup_ratio=0.1``).
    Pulled out as a pure function so the schedule is testable without a GPU -- the shape is the
    part that is easy to get subtly wrong (off-by-one at the warmup boundary, or a decay that
    never actually reaches zero).
    """
    total = max(1, int(total))
    warmup = max(0, int(warmup_ratio * total))
    if warmup and step < warmup:
        return base_lr * (step + 1) / warmup
    # `- 1` so that the FINAL step (index total-1) lands exactly on prog = 1 and the schedule
    # reaches zero. Dividing by `total - warmup` instead never reaches the end of the cosine and
    # leaves the last step running at ~1% of peak -- harmless in a long run, but with only a
    # handful of steps it is a meaningful fraction of the adaptation. Caught by T15h.
    denom = max(1, total - 1 - warmup)
    prog = min(1.0, max(0.0, (step - warmup) / denom))
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * prog))


def run_ttt(peft_model: Any, sequences: Sequence[Sequence[int]], deadline: float,
            device: Any, log_fn: Callable[[str], None] = lambda _m: None
            ) -> Tuple[int, float, bool]:
    """One epoch of LoRA TTT: bf16, batch 1, lr 5e-5, grad-norm 1.0 (spec 2.2).

    Stops early -- between optimizer steps -- once ``deadline`` passes and reports how many
    steps actually ran, how long it took, and whether an OOM occurred (so the caller can
    stop wasting slices on a task that cannot fit).
    """
    if not sequences:
        return 0, 0.0, False
    if torch is None:
        raise RuntimeError("torch is not importable in this environment")
    params = [p for p in peft_model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=TTT_LR)
    # Cosine decay with a linear warmup, matching the reference configuration
    # (lr_scheduler_type="cosine", warmup_ratio=0.1). Without it the last steps of a short run
    # still move at the full 5e-5 and can undo earlier progress; with only a handful of steps
    # the schedule is most of what keeps the update well-behaved.
    steps_total = max(1, len(sequences))
    peft_model.train()
    t0 = time.time()
    steps = 0
    oom = False
    for seq in sequences:
        if time.time() > deadline:
            break
        lr = ttt_lr_at(steps, steps_total)
        for group in optimizer.param_groups:
            group["lr"] = lr
        ids = torch.tensor([list(seq)], dtype=torch.long, device=device)
        labels = ids.clone()
        labels[labels == PAD_ID] = -100
        optimizer.zero_grad(set_to_none=True)
        loss = None
        out = None
        try:
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = peft_model(input_ids=ids, attention_mask=torch.ones_like(ids),
                                 labels=labels, use_cache=False)
                loss = out.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, TTT_MAX_GRAD_NORM)
            optimizer.step()
        except RuntimeError as exc:  # OOM or dtype problem: skip this step
            if "out of memory" in str(exc).lower():
                oom = True
            log_fn(f"    TTT step {steps} failed: {exc}")
            try:
                optimizer.zero_grad(set_to_none=True)
            except Exception:
                pass
            del ids, labels
            if out is not None:
                del out
            gc.collect()
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            continue
        steps += 1
        del ids, labels, out, loss
    peft_model.eval()
    try:
        peft_model.gradient_checkpointing_disable()
    except Exception:
        pass
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return steps, time.time() - t0, oom


# ======================================================================================
# 6. Constrained batched DFS (spec section 2.3)
# ======================================================================================

def arc_logprobs(logits: Any, arc_ids: Any) -> np.ndarray:
    """Full-vocabulary log_softmax on the GPU, only the 12 ARC values to the CPU.

    Normalising over the FULL vocabulary (not re-normalising over the 12 allowed
    tokens) is what makes the ``-log(0.2)`` threshold meaningful, and moving just 12
    floats per beam is what stops CPU<->GPU copies from dominating.
    """
    lp_full = torch.log_softmax(logits.float(), dim=-1)
    return lp_full.index_select(-1, arc_ids).to("cpu").numpy()


def _cache_set_layer_kv(layer: Any, keys: Any, values: Any) -> bool:
    for kname, vname in (("keys", "values"), ("_keys", "_values")):
        if hasattr(layer, kname):
            try:
                setattr(layer, kname, keys)
                setattr(layer, vname, values)
                return True
            except Exception:
                continue
    return False


def _cache_select(cache: Any, indices: Sequence[int], device: Any) -> Any:
    """Index the batch dimension of a KV cache (version tolerant).

    Handles the modern ``DynamicCache.layers[i].keys/values`` layout, the legacy
    ``key_cache``/``value_cache`` lists, and a ``batch_select_indices`` that either
    mutates in place or returns a new cache.  Raises when the layout is unknown --
    the caller then falls back to cached greedy generation, so a cache API change
    can never crash the run.
    """
    if cache is None:
        return None
    idx = torch.as_tensor(list(indices), dtype=torch.long, device=device)
    selector = getattr(cache, "batch_select_indices", None)
    if callable(selector):
        try:
            new_cache = selector(idx)
            return new_cache if new_cache is not None else cache
        except Exception:
            pass
    layers = getattr(cache, "layers", None)
    if layers:
        for layer in layers:
            keys = getattr(layer, "keys", None)
            values = getattr(layer, "values", None)
            if keys is None or values is None:
                continue
            k2 = keys.index_select(0, idx.to(keys.device))
            v2 = values.index_select(0, idx.to(values.device))
            if not _cache_set_layer_kv(layer, k2, v2):
                raise RuntimeError("unsupported DynamicLayer layout")
        return cache
    key_cache = getattr(cache, "key_cache", None)
    if isinstance(key_cache, list) and key_cache:
        cache.key_cache = [k.index_select(0, idx.to(k.device)) for k in key_cache]
        cache.value_cache = [v.index_select(0, idx.to(v.device))
                             for v in cache.value_cache]
        return cache
    raise RuntimeError("unsupported KV cache layout")


def _cache_length(cache: Any) -> int:
    for getter in ("get_seq_length",):
        fn = getattr(cache, getter, None)
        if callable(fn):
            try:
                return int(fn())
            except Exception:
                pass
    try:
        return int(cache.layers[0].keys.shape[-2])
    except Exception:
        pass
    try:
        return int(cache.key_cache[0].shape[-2])
    except Exception:
        pass
    return -1


def _forward_step(model: Any, next_ids: Any, cache: Any, attn_mask: Any,
                  cache_position: Any) -> Any:
    """One cached forward step for a batch of live beams (with an API fallback)."""
    try:
        return model(input_ids=next_ids, attention_mask=attn_mask,
                     past_key_values=cache, cache_position=cache_position,
                     use_cache=True, return_dict=True)
    except TypeError:
        return model(input_ids=next_ids, past_key_values=cache,
                     use_cache=True, return_dict=True)


def max_live_beams_for(prompt_len: int, max_new_tokens: int, n_layers: int,
                       cache_budget_bytes: float, per_token_bytes: Optional[float] = None
                       ) -> int:
    """How many beams may be alive at once without risking a 14.6 GiB OOM.

    KV cost is ``n_layers * 2 * n_kv_heads * head_dim * 2 B`` per token per beam
    (147456 B for the Qwen3-4B config).  With 7.3 GB of bf16 weights there is about
    3 GB left for caches + activations by default, so the beam count is capped
    accordingly (always at least 1).
    """
    per_token = per_token_bytes or (n_layers * KV_BYTES_PER_TOKEN_PER_LAYER)
    per_beam = per_token * max(1, prompt_len + max_new_tokens)
    if per_beam <= 0:
        return 1
    return max(1, min(8, int(cache_budget_bytes // per_beam)))


def turbo_dfs(model: Any, logits: Any, max_new_tokens: int, max_score: float,
              scores: Sequence[float], beams: Sequence[Sequence[int]], caches: Any,
              cache_len: int, end_time: float, ctx: Dict[str, Any],
              results: List[Tuple[float, List[int]]]) -> None:
    """Batched beam DFS over the 12 ARC tokens (spec section 2.3).

    ``logits``    : [B, V] next-token logits, one row per live beam.
    ``scores``    : cumulative log-probability per live beam.
    ``beams``     : continuation token ids already emitted per live beam.
    ``caches``    : batched KV cache, batch dim == number of live beams.
    ``cache_len`` : number of tokens currently cached per row (prompt + beams).
    ``max_score`` : per-token log-probability floor, ``log(0.2)`` (i.e. p > 0.2).
    ``results``   : accumulates ``(score, tokens)`` completed candidates.

    A token becomes a child when its per-token log-probability exceeds
    ``max_score`` (spec: keep tokens with p > 0.2); the cumulative parent score is
    carried along for ranking.  Dead branches are dropped instead of being padded
    with PAD / score-1000 sentinels -- the same search semantics, but no wasted
    KV-cache rows, which is what keeps this safe at 8192 sequence length.  If every
    child of a beam is below the threshold the arg-max token is kept anyway, so the
    search always terminates with at least one complete candidate (improvement 1:
    never come back empty).
    """
    if not beams:
        return
    if time.time() > end_time or ctx["nodes"] >= ctx["max_nodes"]:
        # Flush the in-flight beams instead of dropping them. Discarding them made hard
        # tasks -- the ones that need more than the time slice to finish a 930-token grid
        # -- come back with an EMPTY candidate pool, which is the worst possible outcome
        # because coverage is what scores. A truncated grid is still worth ranking: the
        # parser accepts any 1..30 x 1..30 rectangle, and a partial candidate that scores
        # badly still beats no candidate at all.
        if time.time() > end_time:
            ctx["timed_out"] = True
        ctx["flushed_beams"] = ctx.get("flushed_beams", 0) + len(beams)
        for score, beam in zip(scores, beams):
            results.append((float(score), list(beam)))
        return
    ctx["nodes"] += len(beams)

    arc_ids = ctx["arc_ids"]
    lp = arc_logprobs(logits, arc_ids)  # [B, 12] float32, on the CPU

    children: List[Tuple[float, int, int]] = []
    for i in range(len(beams)):
        row = lp[i]
        parent_score = float(scores[i])
        cand = [(parent_score + float(row[j]), int(ctx["arc_id_list"][j]))
                for j in range(12) if float(row[j]) > max_score]
        if not cand:
            j = int(np.argmax(row))
            cand = [(parent_score + float(row[j]), int(ctx["arc_id_list"][j]))]
            ctx["forced_branches"] += 1
        cand.sort(key=lambda t: -t[0])
        children.extend((s, tok, i) for s, tok in cand[: ctx["max_branches"]])

    live: List[Tuple[float, int, int]] = []
    for score, tok, parent in children:
        if tok == EOS_ID:
            results.append((score, list(beams[parent]) + [tok]))
        else:
            live.append((score, tok, parent))

    if max_new_tokens - 1 <= 0:
        for score, tok, parent in live:
            results.append((score, list(beams[parent]) + [tok]))
        return
    if not live:
        return

    live.sort(key=lambda t: -t[0])
    if len(live) > ctx["max_live"]:
        ctx["pruned_beams"] += len(live) - ctx["max_live"]
        live = live[: ctx["max_live"]]

    parents = [parent for _s, _t, parent in live]
    next_ids = torch.tensor([[tok] for _s, tok, _p in live], dtype=torch.long,
                            device=ctx["device"])
    selected = _cache_select(caches, parents, ctx["device"])
    new_len = cache_len + 1
    attn = torch.ones(len(live), new_len, dtype=torch.long, device=ctx["device"])
    cache_position = torch.arange(cache_len, cache_len + 1, device=ctx["device"])
    out = _forward_step(model, next_ids, selected, attn, cache_position)
    new_beams = [list(beams[parent]) + [tok] for _s, tok, parent in live]
    new_scores = [score for score, _t, _p in live]
    turbo_dfs(model, out.logits[:, -1, :], max_new_tokens - 1, max_score, new_scores,
              new_beams, out.past_key_values, new_len, end_time, ctx, results)


def greedy_cached_fallback(model: Any, prompt_ids: Sequence[int], max_new_tokens: int,
                           end_time: float, ctx: Dict[str, Any]) -> List[int]:
    """Single-beam cached greedy generation -- the safety net if beam DFS fails.

    Used when the KV-cache API is incompatible with :func:`_cache_select` or any
    other unexpected error occurs; it still produces one constrained candidate so a
    task is never left without a neural attempt.
    """
    device = ctx["device"]
    ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    attn = torch.ones_like(ids)
    with torch.inference_mode():
        out = model(input_ids=ids, attention_mask=attn, use_cache=True, return_dict=True)
    cache = out.past_key_values
    logits = out.logits[:, -1, :]
    cache_len = _cache_length(cache)
    if cache_len < 0:
        cache_len = len(prompt_ids)
    tokens: List[int] = []
    for _step in range(int(max_new_tokens)):
        if time.time() > end_time:
            ctx["timed_out"] = True
            break
        row = arc_logprobs(logits, ctx["arc_ids"])[0]
        order = np.argsort(-row)
        pick = int(order[0])
        for j in order:  # honour the same 0.2 threshold when possible
            if float(row[int(j)]) > DFS_TOKEN_LOGPROB_THRESHOLD:
                pick = int(j)
                break
        tok = int(ctx["arc_id_list"][pick])
        tokens.append(tok)
        if tok == EOS_ID:
            break
        next_ids = torch.tensor([[tok]], dtype=torch.long, device=device)
        attn = torch.ones(1, cache_len + 1, dtype=torch.long, device=device)
        cache_position = torch.arange(cache_len, cache_len + 1, device=device)
        with torch.inference_mode():
            out = _forward_step(model, next_ids, cache, attn, cache_position)
        cache = out.past_key_values
        logits = out.logits[:, -1, :]
        cache_len += 1
    return tokens


def constrained_generate(model: Any, prompt_ids: Sequence[int], max_new_tokens: int,
                         end_time: float, cache_budget_bytes: float,
                         log_fn: Callable[[str], None] = lambda _m: None
                         ) -> Tuple[List[List[int]], Dict[str, Any]]:
    """Constrained search for one prompt; returns (candidate token lists, stats).

    Primary path: :func:`turbo_dfs`.  On any exception the cached greedy fallback is
    used, so the caller always gets at least one continuation.
    """
    device = next(model.parameters()).device
    arc_ids = torch.tensor(list(ARC_TOKEN_IDS), dtype=torch.long, device=device)
    try:
        n_layers = int(model.config.num_hidden_layers)
        n_kv = int(getattr(model.config, "num_key_value_heads", 8) or 8)
        head_dim = int(getattr(model.config, "head_dim", 128) or 128)
        per_token = float(n_layers * 2 * n_kv * head_dim * 2)
    except Exception:
        n_layers, per_token = 36, float(KV_BYTES_PER_TOKEN_PER_BEAM)
    max_live = max_live_beams_for(len(prompt_ids), max_new_tokens, n_layers,
                                  cache_budget_bytes, per_token)
    ctx: Dict[str, Any] = {
        "device": device,
        "arc_ids": arc_ids,
        "arc_id_list": list(ARC_TOKEN_IDS),
        "max_branches": DFS_MAX_BRANCHES_PER_BEAM,
        "max_live": max_live,
        "max_nodes": DFS_MAX_NODES,
        "nodes": 0,
        "forced_branches": 0,
        "pruned_beams": 0,
        "timed_out": False,
        "per_token_cache_bytes": per_token,
    }
    results: List[Tuple[float, List[int]]] = []
    ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    attn = torch.ones_like(ids)
    out = None
    try:
        with torch.inference_mode():
            out = model(input_ids=ids, attention_mask=attn, use_cache=True,
                        return_dict=True)
    except Exception as exc:
        log_fn(f"    prompt forward failed ({exc}); no neural candidate")
        ctx["error"] = str(exc)
        return [], ctx

    cache_len = _cache_length(out.past_key_values)
    if cache_len < 0:
        cache_len = len(prompt_ids)
    ctx["cache_len"] = cache_len
    ctx["path"] = "turbo_dfs"
    try:
        with torch.inference_mode():
            turbo_dfs(model, out.logits[:, -1, :], max_new_tokens,
                      DFS_TOKEN_LOGPROB_THRESHOLD, [0.0], [[]],
                      out.past_key_values, cache_len, end_time, ctx, results)
    except Exception as exc:  # cache API mismatch, OOM, anything -> safety net
        log_fn(f"    turbo_dfs failed ({exc}); using greedy fallback")
        ctx["path"] = "greedy_fallback"
        ctx["error"] = f"{type(exc).__name__}: {exc}"
        try:
            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass
        try:
            with torch.inference_mode():
                tokens = greedy_cached_fallback(model, prompt_ids, max_new_tokens,
                                                end_time, ctx)
            if tokens:
                results.append((0.0, tokens))
        except Exception as exc2:
            ctx["error"] += f" | greedy failed: {type(exc2).__name__}: {exc2}"

    results.sort(key=lambda t: -t[0])
    seen: set = set()
    out_tokens: List[List[int]] = []
    for _score, tokens in results:
        key = tuple(tokens)
        if key in seen:
            continue
        seen.add(key)
        out_tokens.append(list(tokens))
    del out
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return out_tokens, ctx


# ======================================================================================
# 7. Teacher-forced NLL rescoring (spec section 2.4)
# ======================================================================================

def _nll_batches(lengths: Sequence[int], max_batch_tokens: int = 8192) -> List[List[int]]:
    """Greedily group sequence lengths so that batch_size * max_len stays bounded."""
    order = sorted(range(len(lengths)), key=lambda i: lengths[i])
    batches: List[List[int]] = []
    cur: List[int] = []
    cur_max = 0
    for i in order:
        nxt = max(cur_max, lengths[i])
        if cur and nxt * (len(cur) + 1) > max_batch_tokens:
            batches.append(cur)
            cur, cur_max = [], 0
            nxt = lengths[i]
        cur.append(i)
        cur_max = nxt
    if cur:
        batches.append(cur)
    return batches


def calc_scores(queries: Sequence[str], answers: Sequence[str], tokenizer: Any,
                model: Any, pad_id: int = PAD_ID,
                log_fn: Callable[[str], None] = lambda _m: None) -> List[float]:
    """Teacher-forced mean NLL per (query, answer) pair -- lower is better (2.4).

    ``queries``/``answers`` are already-formatted strings (``fmt_train`` /
    ``fmt_reply``).  The tokenised query and answer are concatenated and only the
    answer positions are scored (``logits[p-1]`` predicts ``full[p]``); the
    normaliser is the FULL-vocabulary ``logsumexp``.  Right padding with ``pad_id``
    is used as the spec describes, but the micro-batch is chosen so that
    ``batch_size * max_len <= 8192`` to stay memory safe on a 14.6 GiB card.
    """
    if not queries:
        return []
    device = next(model.parameters()).device
    enc_q = [_encode_text(tokenizer, q) for q in queries]
    enc_a = [_encode_text(tokenizer, a) for a in answers]
    full_ids = [q + a for q, a in zip(enc_q, enc_a)]
    q_lens = [len(q) for q in enc_q]
    a_lens = [len(a) for a in enc_a]
    scores: List[float] = [float("inf")] * len(queries)
    batches = _nll_batches([len(f) for f in full_ids])
    model.eval()
    for batch in batches:
        width = max(len(full_ids[i]) for i in batch)
        ids = torch.full((len(batch), width), pad_id, dtype=torch.long, device=device)
        mask = torch.zeros((len(batch), width), dtype=torch.long, device=device)
        for row, i in enumerate(batch):
            ids[row, : len(full_ids[i])] = torch.tensor(full_ids[i], dtype=torch.long,
                                                        device=device)
            mask[row, : len(full_ids[i])] = 1
        out = None
        logits = None
        try:
            with torch.inference_mode():
                out = model(input_ids=ids, attention_mask=mask, use_cache=False,
                            return_dict=True)
                logits = out.logits.float()
                log_norm = torch.logsumexp(logits, dim=-1)  # [B, L]
                for row, i in enumerate(batch):
                    q_len = q_lens[i]
                    n_ans = a_lens[i]
                    if n_ans <= 0:
                        scores[i] = float("inf")
                        continue
                    # logits[p] predicts the token at p+1, so the predicting positions
                    # for the answer tokens are q_len-1 .. q_len+n_ans-2 and the targets
                    # are full_ids[q_len:].  Index BOTH axes: the position and the vocab.
                    pred_pos = torch.arange(q_len - 1, q_len - 1 + n_ans, device=device)
                    targets = torch.tensor(full_ids[i][q_len:], dtype=torch.long,
                                           device=device)
                    picked = logits[row][pred_pos, targets]      # logit of the target
                    norm = log_norm[row][pred_pos]               # full-vocab normaliser
                    scores[i] = float((-(picked - norm)).mean().item())
        except Exception as exc:  # keep going: unscored candidates rank last
            log_fn(f"    rescoring batch failed: {exc}")
        finally:
            del ids, mask, out, logits
    gc.collect()
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass
    return scores


# ======================================================================================
# 8. Symbolic engine (optional; spec section 3 improvements 1 and 3)
# ======================================================================================

def load_engine(engine_path: Optional[str],
                log_fn: Callable[[str], None] = lambda _m: None) -> Dict[str, Any]:
    """Load the optional symbolic engine; never fatal when missing or broken."""
    info: Dict[str, Any] = {"path": engine_path, "status": "skipped", "reason": ""}
    if not engine_path:
        info["reason"] = "no --engine path given"
        return {"module": None, "config": None, "info": info}
    path = Path(engine_path)
    if not path.is_file():
        info["reason"] = "path does not exist"
        return {"module": None, "config": None, "info": info}
    try:
        spec = importlib.util.spec_from_file_location("arc26_symbolic_engine", str(path))
        if spec is None or spec.loader is None:
            raise ImportError("could not build import spec")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        config = None
        if hasattr(module, "make_config"):
            overrides = {
                "mode": "dev",
                "project_dir": str(Path(os.environ.get("TEMP", "/tmp")) / "arc26_engine"),
                "resume": False,
                "snapshot_on_finish": False,
            }
            try:
                config = module.make_config(overrides)
            except Exception:
                config = module.make_config({})
        info["status"] = "loaded"
        return {"module": module, "config": config, "info": info}
    except Exception as exc:
        info["status"] = "error"
        info["reason"] = f"{type(exc).__name__}: {exc}"
        log_fn(f"  engine load failed: {info['reason']}")
        return {"module": None, "config": None, "info": info}


def engine_predict(engine: Dict[str, Any], task: dict, timeout_seconds: float
                   ) -> List[Dict[str, Any]]:
    """Run the symbolic engine on one task.

    Returns one dict per test input: ``{"attempt_1": grid|None, "attempt_2": grid|None,
    "verified": bool, "program": str}``.

    ``verified`` comes from the engine's own diagnostics: ``source == "search"`` means
    a program that reproduces **every** demonstration pair was found, whereas
    ``source == "prior"`` means only shape/palette priors survived (no program
    validated).  That distinction matters for the hybrid pool: a validated program is
    treated as strong evidence, an unvalidated prior is ranked *below* every scored
    neural candidate and only beats the heuristic floor.

    The engine compares ``task_deadline`` against ``time.monotonic()``, so the deadline
    is built in that clock -- passing an epoch timestamp would silently disable its
    internal timeout.
    """
    module = engine.get("module")
    if module is None or not hasattr(module, "solve_task"):
        return []
    deadline = time.monotonic() + max(1.0, float(timeout_seconds))
    cfg = engine.get("config") or {}
    try:
        res = module.solve_task(task, cfg, task_deadline=deadline)
    except TypeError:  # engine without a deadline parameter
        try:
            res = module.solve_task(task, cfg)
        except Exception:
            return []
    except Exception:
        return []
    res = res or {}
    diag = res.get("diagnostics") or {}
    verified = str(diag.get("source", "")) == "search"
    program = str(diag.get("top_program") or "")
    out: List[Dict[str, Any]] = []
    for entry in res.get("attempts") or []:
        if not isinstance(entry, dict):
            out.append({"attempt_1": None, "attempt_2": None,
                        "verified": False, "program": program})
            continue
        out.append({
            "attempt_1": validate_grid(entry.get("attempt_1")),
            "attempt_2": validate_grid(entry.get("attempt_2")),
            "verified": verified,
            "program": program,
        })
    return out


# ======================================================================================
# 9. Heuristic floor (spec section 3 improvement 1)
# ======================================================================================

def mode_colour(task: dict) -> int:
    """Most common colour across the demonstration outputs (inputs as fallback)."""
    counts = np.zeros(10, dtype=int)
    found = False
    for d in task.get("train", []):
        for key in ("output", "input"):
            arr = validate_grid(d.get(key))
            if arr is not None:
                counts += np.bincount(arr.ravel(), minlength=10)[:10]
                found = True
    if not found:
        return 0
    return int(np.argmax(counts))


def fallback_pair(task: dict, test_index: int) -> Tuple[np.ndarray, np.ndarray]:
    """Two legal, always-available, and *mutually distinct* grids.

    The ordering is deliberate. The identity grid is the first bet: for a task the
    model failed on, staying close to the input is a much better bet than a
    constant-colour fill, which is essentially never the answer. The second bet is the
    foreground/background swap of the input -- community reports flag this as a common
    trivial transformation on sparse tasks.

    Why this replaced the old implementation: it returned a mode-colour fill plus an
    all-zeros grid. ``mode_colour`` pools demonstration inputs *and* outputs, so the
    background (usually 0) dominates; both attempts therefore came out as the SAME
    all-zeros grid. That wasted the second-attempt slot entirely and made a constant
    fill the primary guess.
    """
    tests = task.get("test", [])
    arr = None
    if 0 <= test_index < len(tests):
        arr = validate_grid(tests[test_index].get("input"))
    if arr is None:
        return np.zeros((1, 1), dtype=int), np.ones((1, 1), dtype=int)

    identity = arr

    counts = np.bincount(arr.ravel(), minlength=10)[:10]
    bg = int(np.argmax(counts))
    others = [c for c in range(10) if c != bg and counts[c] > 0]
    swap = None
    if others:
        fg = max(others, key=lambda c: counts[c])
        swap = arr.copy()
        swap[arr == bg] = fg       # `arr` is untouched, so the second mask is correct
        swap[arr == fg] = bg
    if swap is not None and not np.array_equal(swap, identity):
        return identity, swap

    # Degenerate input (uniform, or one colour only): fall back to distinct constants.
    colour = mode_colour(task)
    constant = np.full(arr.shape, colour, dtype=int)
    if not np.array_equal(constant, identity):
        return identity, constant
    return identity, np.full(arr.shape, (colour + 1) % 10, dtype=int)


def fallback_attempts(task: dict) -> List[Dict[str, np.ndarray]]:
    """The full per-test fallback list for one task."""
    out: List[Dict[str, np.ndarray]] = []
    for i in range(max(1, len(task.get("test", [])))):
        a1, a2 = fallback_pair(task, i)
        out.append({"attempt_1": a1, "attempt_2": a2})
    return out


# ======================================================================================
# 10. Candidate pool + selection (spec section 2.4 / section 3 improvement 3)
# ======================================================================================

def dedupe_pool(cands: Iterable[Candidate], limit: int = 32) -> List[Candidate]:
    """Drop exact duplicate grids, keeping the best-ranked instance of each."""
    best: Dict[bytes, Candidate] = {}
    for cand in cands:
        key = cand.grid.tobytes() + bytes(cand.grid.shape)
        cur = best.get(key)
        if cur is None or cand.sort_key < cur.sort_key:
            best[key] = cand
    pool = list(best.values())
    if limit and len(pool) > limit:
        pool.sort(key=lambda c: c.sort_key)
        pool = pool[:limit]
    return pool


def pool_stats(pools: Sequence[Sequence[Candidate]]) -> Tuple[int, int, float, float]:
    """(n_candidates, n_distinct, best_sort_key, max agreement) over all test pools.

    Agreement = how many candidates share the single most duplicated grid, which is
    the cheapest available confidence signal for the cascade scheduler.
    """
    n_total = 0
    counts: Dict[bytes, int] = {}
    best_key = float("inf")
    for pool in pools:
        n_total += len(pool)
        for cand in pool:
            key = cand.grid.tobytes() + bytes(cand.grid.shape)
            counts[key] = counts.get(key, 0) + 1
            best_key = min(best_key, cand.sort_key)
    agree = max(counts.values()) if counts else 0
    return n_total, len(counts), best_key, agree


def truth_rank_in_pool(pool: Sequence[Candidate], want: np.ndarray) -> Optional[int]:
    """0-based rank of the truth inside one test input's pool, or ``None`` if absent.

    This separates the two failure modes that look identical from the outside:
    truth present but not selected  -> the loss is in *selection* (a better reranker helps);
    truth absent from the pool      -> the loss is in *generation* and no selector can help.
    """
    ranked = sorted(pool, key=lambda c: (c.sort_key, SOURCE_PRIORITY.get(c.source, 9)))
    for rank, cand in enumerate(ranked):
        if np.array_equal(cand.grid, want):
            return rank
    return None


def score_pool_recall(pools: Sequence[Sequence[Candidate]],
                      solutions: Optional[Dict[str, list]],
                      task_id: str) -> Dict[str, Any]:
    """Pool-level recall: how often the ground truth is in the candidate pool at all.

    ``n_in_pool`` is the ceiling that any reranking/selection change could reach;
    ``n_top1`` is what the current ranking already gets right; the gap between
    ``n_in_pool`` and the selected attempts is the *addressable* selection loss.
    """
    out: Dict[str, Any] = {"n_with_truth": 0, "n_in_pool": 0, "n_top1": 0, "ranks": []}
    if not solutions or task_id not in solutions:
        return out
    truth = solutions[task_id]
    for i, pool in enumerate(pools):
        if i >= len(truth):
            break
        want = validate_grid(truth[i])
        if want is None:
            continue
        out["n_with_truth"] += 1
        rank = truth_rank_in_pool(pool, want)
        if rank is not None:
            out["n_in_pool"] += 1
            out["ranks"].append(int(rank))
            if rank == 0:
                out["n_top1"] += 1
    return out


def select_attempts(pool: Sequence[Candidate], fallback: Tuple[np.ndarray, np.ndarray]
                    ) -> Tuple[np.ndarray, np.ndarray, str, str]:
    """Best two *distinct* grids from the unified pool (rank ascending).

    ``attempt_2`` is the best candidate whose grid differs from ``attempt_1``; if the
    pool holds only one distinct grid, the fallback floor supplies the second attempt
    so both slots stay legal.
    """
    ordered = sorted(pool, key=lambda c: (c.sort_key, SOURCE_PRIORITY.get(c.source, 9)))
    if ordered:
        a1, s1 = ordered[0].grid, ordered[0].source
    else:
        a1, s1 = fallback[0], "fallback"
    a2: Optional[np.ndarray] = None
    s2 = "fallback"
    for cand in ordered:
        if not np.array_equal(cand.grid, a1):
            a2, s2 = cand.grid, cand.source
            break
    if a2 is None:
        alt = fallback[1]
        if np.array_equal(alt, a1):
            alt = fallback[0] if not np.array_equal(fallback[0], a1) else a1
        a2, s2 = alt, "fallback"
    return a1, a2, s1, s2


# ======================================================================================
# 11. Dynamic scheduler (spec section 3 improvement 2 + cascade stages)
# ======================================================================================

class BudgetScheduler:
    """Dynamic per-task time budget: calibrate first, then follow the measured pace.

    ``uniform`` schedule: the first ``calibrate_tasks`` tasks run with a generous
    slice and their measured wall time defines ``seconds_per_task``; every later
    slice is ``remaining_budget / remaining_tasks`` (clamped, never more than ~2.5x
    the calibrated pace).

    ``cascade`` schedule: the same calibration measures the *cheap sweep* cost
    (:meth:`observe_cheap`), which then sizes the rest of Stage A, while Stage B
    slices come from the fair share of the budget that Stage B may still spend.
    """

    def __init__(self, time_budget_seconds: float, n_tasks: int, calibrate_tasks: int,
                 reserve_seconds: float = 300.0, started_at: Optional[float] = None,
                 log_fn: Callable[[str], None] = lambda _m: None):
        self.started_at = started_at if started_at is not None else time.time()
        self.time_budget_seconds = float(time_budget_seconds)
        self.n_tasks = int(n_tasks)
        self.calibrate_tasks = max(0, int(calibrate_tasks))
        self.reserve_seconds = float(reserve_seconds)
        self.times: List[float] = []          # calibration-phase task times
        self.ttt_step_seconds: Optional[float] = None
        self.cheap_seconds: Optional[float] = None
        self.ttt_task_seconds: Optional[float] = None
        self.log_fn = log_fn

    # -- clocks ------------------------------------------------------------------------
    @property
    def hard_deadline(self) -> float:
        return self.started_at + self.time_budget_seconds

    def remaining(self) -> float:
        return self.hard_deadline - time.time()

    @property
    def expired(self) -> bool:
        return self.remaining() <= self.reserve_seconds

    @property
    def seconds_per_task(self) -> float:
        """Calibrated per-task cost from the calibration phase (0 if none yet)."""
        if not self.times:
            return 0.0
        return float(sum(self.times) / len(self.times))

    # -- measurement -------------------------------------------------------------------
    def record(self, seconds: float, index: int) -> None:
        """Record a calibration-phase measurement (per-task cost estimate)."""
        if index < max(1, self.calibrate_tasks):
            self.times.append(float(seconds))

    def observe_cheap(self, seconds: float) -> None:
        """Exponential moving average of the Stage A (no-TTT) per-task cost."""
        self.cheap_seconds = (float(seconds) if self.cheap_seconds is None
                              else 0.7 * self.cheap_seconds + 0.3 * float(seconds))

    def observe_ttt_task(self, seconds: float) -> None:
        """Exponential moving average of the Stage B (with-TTT) per-task cost."""
        self.ttt_task_seconds = (float(seconds) if self.ttt_task_seconds is None
                                 else 0.7 * self.ttt_task_seconds + 0.3 * float(seconds))

    # -- slices ------------------------------------------------------------------------
    def cheap_slice(self, index: int) -> float:
        """Stage A slice: small, and sized from the measured cheap-sweep cost."""
        if index < self.calibrate_tasks:
            return STAGE_A_CALIBRATION_SLICE
        if self.cheap_seconds:
            return float(min(max(4.0 * self.cheap_seconds, STAGE_A_MIN_SLICE), 150.0))
        return STAGE_A_DEFAULT_SLICE

    def uniform_slice(self, index: int, remaining_tasks: int) -> float:
        """Slice for the ``uniform`` schedule (spec improvement 2)."""
        remaining_tasks = max(1, int(remaining_tasks))
        usable = max(0.0, self.remaining() - self.reserve_seconds)
        fair = usable / remaining_tasks
        if index < self.calibrate_tasks:
            return float(min(max(fair, 30.0), 900.0))
        pace = self.seconds_per_task or fair
        return float(min(max(fair, 15.0), max(30.0, 2.5 * pace)))

    def stage_b_slice(self, remaining_refinable: int, stage_c_reserve: float) -> float:
        """Slice for one Stage B (TTT) task: fair share of what Stage B may spend."""
        spendable = max(0.0, self.remaining() - self.reserve_seconds - stage_c_reserve)
        fair = spendable / max(1, int(remaining_refinable))
        return float(min(max(fair, STAGE_B_MIN_SLICE), 1200.0))

    def stage_c_reserve(self, n_tasks: int) -> float:
        """Seconds held back for the symbolic/heuristic backfill stage."""
        return max(STAGE_C_RESERVE_MIN, min(1200.0, 3.0 * float(n_tasks)))

    def augmentation_budget(self, slice_seconds: float, aug_train: int, aug_infer: int
                            ) -> Tuple[int, int]:
        """Shrink TTT / inference augmentation counts when the slice is tight.

        A TTT step is ~8-10 s on a T4 at 8192 tokens, so an adaptive count keeps a
        task inside its slice instead of overrunning the whole session.
        """
        step = self.ttt_step_seconds or 8.0
        ttt_budget = max(0.0, slice_seconds * 0.55)
        n_train = int(ttt_budget // max(1e-6, step))
        n_train = int(min(max(1, n_train), max(1, aug_train)))
        if slice_seconds >= 120.0:
            n_infer = max(1, aug_infer)
        elif slice_seconds >= 60.0:
            n_infer = max(1, min(2, aug_infer))
        else:
            n_infer = 1
        return n_train, n_infer


# ======================================================================================
# 12. Per-task solve (spec section 2.1) -- driven by the two schedules
# ======================================================================================

def solve_task(task: dict, task_id: str, model: Any, tokenizer: Any,
               engine: Dict[str, Any], args: argparse.Namespace, time_slice: float,
               rng: random.Random, do_ttt: bool = True, use_engine: bool = True,
               aug_override: Optional[int] = None, pool_prev: Optional[Sequence[Sequence[Candidate]]] = None,
               scheduler: Optional[BudgetScheduler] = None,
               log_fn: Callable[[str], None] = print) -> TaskResult:
    """Full pipeline for one task: [TTT] -> DFS(+augs) -> pool -> rescore -> 2 attempts.

    ``do_ttt=False`` skips LoRA entirely (the Stage A cheap sweep); the model is then
    only prompted, which is what makes the full 240-task sweep cost seconds per task.
    ``use_engine`` decides whether the symbolic engine contributes to the pool (Stage
    C / the uniform schedule) or not (Stages A and B, which are pure neural).
    ``pool_prev`` seeds the candidate pool so a later stage can add evidence instead
    of starting over.
    """
    result = TaskResult(stage="B" if do_ttt else "A")
    t0 = time.time()
    tests = task.get("test", [])
    n_test = len(tests)
    if n_test == 0:
        result.seconds = time.time() - t0
        return result

    deadline = time.time() + max(1.0, time_slice)
    if scheduler is not None:
        deadline = min(deadline, scheduler.hard_deadline - scheduler.reserve_seconds * 0.5)
    max_new_tokens = max_new_tokens_for(tokenizer)

    fallbacks: List[Tuple[np.ndarray, np.ndarray]] = [fallback_pair(task, i)
                                                      for i in range(n_test)]
    pools: List[List[Candidate]] = [
        list(pool_prev[i]) if pool_prev is not None and i < len(pool_prev) else []
        for i in range(n_test)
    ]
    result.fallbacks = fallbacks

    # ---- (a) symbolic engine candidates ------------------------------------------
    # A validated program ("symbolic") is treated as strong evidence; an unvalidated
    # prior ("prior") ranks below every scored neural candidate but above the heuristic
    # floor, and a degenerate 1x1 zero grid from the engine's own last-resort path is
    # dropped entirely -- our floor is strictly better than [[0]].
    if use_engine and engine.get("module") is not None:
        eng_timeout = max(5.0, time_slice * 0.2)
        for i, pred in enumerate(engine_predict(engine, task, eng_timeout)):
            if i >= n_test:
                break
            source = "symbolic" if pred["verified"] else "prior"
            result.engine_source = "search" if pred["verified"] else "prior"
            result.engine_program = pred["program"]
            default_nll = UNSCORED_SYMBOLIC_NLL if pred["verified"] else UNSCORED_PRIOR_NLL
            for grid in (pred["attempt_1"], pred["attempt_2"]):
                if grid is None:
                    continue
                if not pred["verified"] and grid.shape == (1, 1) and int(grid[0, 0]) == 0:
                    continue  # engine fell through to fallback_zeros
                cand = Candidate.make(grid, source, augment="identity",
                                      note=f"engine:{pred['program']}", nll=default_nll)
                if cand is not None:
                    pools[i].append(cand)
        log_fn(f"  [{task_id}] engine pool sizes {[len(p) for p in pools]}")

    # ---- (b) TTT on the augmented demonstrations ---------------------------------
    global TTT_DISABLED_REASON, TTT_CONSECUTIVE_OOM
    if do_ttt and model is not None and not TTT_DISABLED_REASON:
        if scheduler is not None:
            n_train_aug, n_infer_aug = scheduler.augmentation_budget(
                time_slice, args.aug_train, args.aug_infer)
        else:
            n_train_aug, n_infer_aug = max(1, args.aug_train), max(1, args.aug_infer)
        if aug_override:
            n_infer_aug = max(1, int(aug_override))
        if time.time() < deadline - 10.0:
            # Retry ladder: full context first, then progressively shorter ones. A task that
            # cannot fit a 2048-token step may still fit a 1024-token one, and adapting it
            # partially beats not adapting it at all.
            budget = int(args.ttt_seq_tokens)
            ladder = [budget] if budget <= 0 else [budget, budget // 2, budget // 4]
            ladder = [b for b in ladder if b > 0] or [0]
            steps = ttt_seconds = 0
            oom = False
            peft_model = None
            for attempt, ctx_budget in enumerate(ladder):
                sequences, _keys = build_ttt_sequences(
                    task.get("train", []), tokenizer, n_train_aug, args.max_seq_length, rng,
                    seq_token_budget=ctx_budget)
                if not sequences:
                    break
                try:
                    # Hand every cached byte back to the driver *before* asking for the
                    # gradient-checkpointed activations a step needs. Stage A's DFS leaves its
                    # KV caches behind: observed peak 14.09 of 14.56 GiB, 724 MiB free, and the
                    # allocation then failing.
                    gc.collect()
                    try:
                        torch.cuda.empty_cache()
                        free_b, total_b = torch.cuda.mem_get_info()
                        log_fn(f"  [{task_id}] pre-TTT free {free_b / 2**30:.2f} GiB "
                               f"of {total_b / 2**30:.2f} GiB (ctx<= {ctx_budget or 'uncapped'})")
                    except Exception:
                        pass
                    peft_model = attach_lora(model)
                    steps, ttt_seconds, oom = run_ttt(peft_model, sequences, deadline,
                                                      next(model.parameters()).device,
                                                      log_fn=log_fn)
                finally:
                    if peft_model is not None:
                        detach_lora(peft_model)
                        peft_model = None
                if steps > 0 or not oom:
                    break
                if attempt + 1 < len(ladder):
                    log_fn(f"  [{task_id}] TTT OOM at ctx<={ctx_budget}; retrying at "
                           f"ctx<={ladder[attempt + 1]}")
            result.ttt_steps = steps
            if steps > 0:
                per_step = ttt_seconds / steps
                if scheduler is not None:
                    if scheduler.ttt_step_seconds is None:
                        scheduler.ttt_step_seconds = per_step
                    else:
                        scheduler.ttt_step_seconds = (
                            0.7 * scheduler.ttt_step_seconds + 0.3 * per_step
                        )
            log_fn(f"  [{task_id}] TTT {steps} steps in {ttt_seconds:.1f}s "
                   f"(aug_train={n_train_aug}, aug_infer={n_infer_aug})")
            if oom and steps == 0:
                # Doomed *for this task*, not for the run. Disabling TTT globally after one
                # failure is how a single 1.77 GiB allocation turned a 240-task run into a beam
                # search over an unadapted model: 86 of 102 "TTT" tasks executed zero steps.
                # Degrade the mechanism only after the ladder has failed repeatedly.
                self_consec = TTT_CONSECUTIVE_OOM + 1
                TTT_CONSECUTIVE_OOM = self_consec
                if self_consec >= max(1, int(args.ttt_max_consecutive_oom)):
                    TTT_DISABLED_REASON = (
                        f"LoRA TTT OOM on {self_consec} consecutive tasks even at the shortest "
                        f"context (<= {ladder[-1]} tokens); budget moved to Stage A coverage")
                    log_fn(f"  ! TTT disabled for the rest of the run: {TTT_DISABLED_REASON}")
            else:
                TTT_CONSECUTIVE_OOM = 0
    else:
        n_infer_aug = max(1, int(aug_override or args.aug_infer))

    # ---- (c) neural candidates: DFS under each inference augmentation ------------
    n_aug_used = 0
    if model is not None and time.time() < deadline - 3.0:
        train_demos = task.get("train", [])
        n_aug = max(1, n_infer_aug)
        for k in range(n_aug):
            if time.time() >= deadline - 3.0:
                result.timed_out = True
                break
            key = AugKey.identity() if k == 0 else random_augment_key(rng, allow_color=True)
            aug_demos = augment_demos(train_demos, key)
            per_input = max(2.0, (deadline - time.time()) / max(1, n_aug - k) / max(1, n_test))
            for i in range(n_test):
                if time.time() >= deadline - 2.0:
                    result.timed_out = True
                    break
                input_ids = _encode_text(
                    tokenizer, fmt_train(aug_demos, apply_augment(tests[i]["input"], key))
                )
                if len(input_ids) > args.max_seq_length:
                    input_ids = input_ids[-args.max_seq_length:]
                tokens_list, stats = constrained_generate(
                    model, input_ids, max_new_tokens,
                    min(deadline, time.time() + per_input),
                    args.cache_budget_gb * (1024 ** 3), log_fn=log_fn,
                )
                for tokens in tokens_list:
                    # recover_truncated=True: a beam stopped by the time budget ends
                    # mid-row, and salvaging the completed rows is what keeps this pool
                    # from coming back empty on hard tasks.
                    grid_aug = tokens_to_array(tokens, tokenizer=tokenizer,
                                               recover_truncated=True)
                    if grid_aug is None:
                        continue
                    grid = invert_augment(grid_aug, key)
                    cand = Candidate.make(grid, "neural", augment=str(key.to_json()),
                                          note=f"dfs:{stats.get('path')}")
                    if cand is not None:
                        pools[i].append(cand)
            n_aug_used += 1
            log_fn(f"  [{task_id}] aug {k + 1}/{n_aug} {key.to_json()} -> "
                   f"pool sizes {[len(p) for p in pools]}")
    result.n_aug_used = max(1, n_aug_used)

    # ---- (d) unified rescoring in the IDENTITY space (improvement 3) -------------
    if model is not None:
        identity_query = [fmt_train(task.get("train", []), tests[i]["input"])
                          for i in range(n_test)]
        for i in range(n_test):
            pool = dedupe_pool(pools[i])[:32]
            if not pool:
                continue
            if scheduler is not None and \
                    time.time() >= scheduler.hard_deadline - scheduler.reserve_seconds * 0.5:
                result.timed_out = True
            try:
                scores = calc_scores(
                    [identity_query[i]] * len(pool),
                    [fmt_reply(c.grid) for c in pool],
                    tokenizer, model, log_fn=log_fn,
                )
                for cand, score in zip(pool, scores):
                    cand.nll = float(score)
            except Exception as exc:
                log_fn(f"  [{task_id}] rescoring failed for test {i}: {exc}")
            pools[i] = pool

    # ---- (e) pick two distinct grids; always fill both slots ---------------------
    result.attempts = []
    result.sources = []
    for i in range(n_test):
        pool = dedupe_pool(pools[i])
        pools[i] = pool
        a1, a2, s1, s2 = select_attempts(pool, fallbacks[i])
        v1 = validate_grid(a1)
        v2 = validate_grid(a2)
        if v1 is None:
            v1, s1 = fallbacks[i][0], "fallback"
        if v2 is None:
            v2, s2 = fallbacks[i][1], "fallback"
        result.attempts.append({"attempt_1": v1, "attempt_2": v2})
        result.sources.extend([s1, s2])

    # ---- (f) confidence signal for the cascade scheduler -------------------------
    n_total, n_distinct, best_key, agree = pool_stats(pools)
    result.n_candidates = n_total
    result.n_distinct = n_distinct
    result.best_key = best_key
    result.best_nll = best_key
    result.agree_max = agree
    result.has_symbolic = any(c.source == "symbolic" for p in pools for c in p)
    base = 0.0
    agree_frac = 0.0
    coverage = 0.0
    if n_total > 0:
        # Agreement is measured *across augmentations*: a single augmentation cannot
        # corroborate itself. The shipped cascade runs used --aug-infer 1, which made this
        # term a silent constant 0, leaving `confidence` = 0.5 * coverage -- a candidate-count
        # proxy that ties every task with >= 4 candidates at exactly 0.50. That made Stage B's
        # "run TTT on the weakest tasks" ordering arbitrary (observed: B_ttt_no_gain on 84 of
        # 102 tasks). With a single augmentation, fall back to within-pool beam convergence,
        # which is a real signal at any augmentation count; the components are recorded in the
        # report so the formula can be tuned from data instead of guesswork.
        if result.n_aug_used >= 2:
            agree_frac = max(0, agree - 1) / max(1, result.n_aug_used - 1)
        else:
            agree_frac = max(0, agree - 1) / max(1, n_total - 1)
        coverage = min(1.0, n_total / 4.0)
        base = 0.5 * agree_frac + 0.5 * coverage
    if result.has_symbolic:
        base = max(base, SYMBOLIC_CONFIDENCE)
    result.conf_agree = float(agree_frac)
    result.conf_coverage = float(coverage)
    result.confidence = float(min(1.0, base))
    result.pools = pools
    result.seconds = time.time() - t0
    return result


# ======================================================================================
# 13. Reporting helpers
# ======================================================================================

def score_attempts(task_id: str, attempts: Sequence[Dict[str, np.ndarray]],
                   solutions: Optional[Dict[str, list]]) -> Tuple[int, int, int]:
    """Compare attempts with ground truth. Returns (ok1, ok2, n_with_truth).

    ``solutions[task_id][i]`` is the truth for ``test[i]`` -- the solutions file
    attaches outputs by index within the task's ``test`` list (spec section 4).
    """
    if not solutions or task_id not in solutions:
        return 0, 0, 0
    truth = solutions[task_id]
    ok1 = ok2 = 0
    n = 0
    for i, entry in enumerate(attempts):
        if i >= len(truth):
            break
        want = validate_grid(truth[i])
        if want is None:
            continue
        n += 1
        if np.array_equal(entry["attempt_1"], want):
            ok1 += 1
        if np.array_equal(entry["attempt_2"], want):
            ok2 += 1
    return ok1, ok2, n


def build_submission_payload(task_ids: Sequence[str],
                             submission: Dict[str, List[Dict[str, np.ndarray]]]
                             ) -> Dict[str, List[Dict[str, List[List[int]]]]]:
    """JSON-ready submission: ``{task_id: [{"attempt_1": grid, "attempt_2": grid}, ...]}``."""
    payload: Dict[str, List[Dict[str, List[List[int]]]]] = {}
    for tid in task_ids:
        entries = submission.get(tid) or []
        payload[tid] = [
            {"attempt_1": grid_to_json(e["attempt_1"]),
             "attempt_2": grid_to_json(e["attempt_2"])}
            for e in entries
        ]
    return payload


def flush_submission(out_path: str, task_ids: Sequence[str],
                     submission: Dict[str, List[Dict[str, np.ndarray]]],
                     log_fn: Callable[[str], None] = print) -> None:
    """Incremental write of submission.json (every 10 tasks + at the very end)."""
    try:
        write_json_atomic(out_path, build_submission_payload(task_ids, submission))
    except Exception as exc:
        log_fn(f"  ! submission flush failed: {exc}")


# ======================================================================================
# 14. CLI
# ======================================================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="arc26_solver.py",
        description="ARC-AGI-2 TTT solver (LoRA test-time training + constrained DFS + "
                    "cascade scheduling).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--limit", type=int, default=0,
                        help="only run the first N tasks (0 = all)")
    parser.add_argument("--num-shards", type=int, default=1,
                        help="split the task list into this many interleaved shards "
                             "(2 concurrent 12 h GPU sessions is Kaggle's ceiling)")
    parser.add_argument("--shard-index", type=int, default=0,
                        help="which shard this session runs, in [0, --num-shards)")
    parser.add_argument("--split", choices=sorted(SPLIT_FILES), default="test",
                        help="data split: test | training | evaluation")
    parser.add_argument("--schedule", choices=("cascade", "uniform"), default="cascade",
                        help="cascade = cheap sweep -> TTT on weak tasks -> symbolic "
                             "backfill; uniform = TTT on every task")
    parser.add_argument("--time-budget-seconds", type=float, default=39000.0,
                        help="total wall-clock budget in seconds")
    parser.add_argument("--calibrate-tasks", type=int, default=5,
                        help="number of tasks in the calibration phase")
    parser.add_argument("--aug-train", type=int, default=16,
                        help="number of TTT augmentations")
    parser.add_argument("--aug-infer", type=int, default=4,
                        help="number of inference augmentations (voting)")
    parser.add_argument("--out", default=DEFAULT_OUT, help="submission.json path")
    parser.add_argument("--report", default=DEFAULT_REPORT, help="report.json path")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR,
                        help="competition data directory")
    parser.add_argument("--engine", default=None,
                        help="optional symbolic engine path (skipped when missing)")
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR,
                        help="model weights directory")
    parser.add_argument("--max-seq-length", type=int, default=TTT_MAX_SEQ_LENGTH,
                        help="TTT / prompt truncation length")
    # The binding constraint on TTT is memory, not the learning rate: one optimizer step over
    # an 8192-token sequence on a 3.63B model does not fit a 14.56 GiB T4 alongside the search's
    # residual caches, and the failure is silent (the stage is recorded as attempted and
    # reported as "no gain"). Capping the context keeps the target pair -- which carries the
    # loss -- while making the step fit.
    parser.add_argument("--ttt-seq-tokens", type=int, default=TTT_SEQ_TOKEN_BUDGET,
                        help="cap TTT context length per optimizer step by dropping context "
                             "demonstrations, keeping the target pair (0 = no cap; default "
                             f"{TTT_SEQ_TOKEN_BUDGET}). This is what makes adaptation fit.")
    parser.add_argument("--ttt-max-consecutive-oom", type=int, default=TTT_MAX_CONSECUTIVE_OOM,
                        help="give up on TTT for the whole run only after this many tasks in a "
                             "row fail to allocate (default "
                             f"{TTT_MAX_CONSECUTIVE_OOM}); each task is retried at a shorter "
                             "context first, so one task's OOM does not unadapt the run")
    parser.add_argument("--lora-r", type=int, default=LORA_R,
                        help=f"LoRA rank for TTT (default {LORA_R}, matching the strongest "
                             "published instance of this recipe). Rank is the capacity of the "
                             "only mechanism that can change the generator, so lowering it "
                             "trades memory for score. ~0.53B trainable params at 256, which "
                             "does NOT fit a 14.56 GiB T4 -- use an L4 (22 GiB).")
    parser.add_argument("--cache-budget-gb", type=float, default=DFS_DEFAULT_CACHE_BUDGET_GB,
                        help="KV-cache budget for live DFS beams, in GiB")
    # Search width. These are the only levers that can raise pool recall, which is the
    # binding constraint: on the 2026-09-14 eval probe `selection_headroom` was 0.0, i.e.
    # the truth was never generated, so no reranking change could have helped.
    parser.add_argument("--dfs-prob-threshold", type=float, default=DFS_TOKEN_PROB_THRESHOLD,
                        help="prune DFS branches whose token probability is below this "
                             "(default 0.2; lower = wider search = more coverage, slower)")
    parser.add_argument("--dfs-max-branches", type=int, default=DFS_MAX_BRANCHES_PER_BEAM,
                        help="child cap per beam (default 3; memory guard)")
    parser.add_argument("--dfs-max-nodes", type=int, default=DFS_MAX_NODES,
                        help="global node budget per DFS call (default 6000)")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    return parser.parse_args(argv)


# ======================================================================================
# 15. Run context + the two schedules
# ======================================================================================

@dataclass
class RunContext:
    """Mutable state shared by the schedule drivers and the reporters."""

    args: argparse.Namespace
    tasks: Dict[str, dict]
    task_ids: List[str]
    solutions: Optional[Dict[str, list]]
    submission: Dict[str, List[Dict[str, np.ndarray]]]
    report: Dict[str, Any]
    scheduler: BudgetScheduler
    model: Any = None
    tokenizer: Any = None
    engine: Dict[str, Any] = field(default_factory=dict)
    rng: random.Random = field(default_factory=random.Random)
    log: Callable[[str], None] = print
    per_task_index: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    state: Dict[str, TaskResult] = field(default_factory=dict)
    n_processed: int = 0

    # -- bookkeeping -------------------------------------------------------------------
    def record_task(self, tid: str, res: TaskResult, stage: str,
                    replace: bool = False) -> Dict[str, Any]:
        """Store the result and keep exactly one report entry per task."""
        if replace or tid not in self.per_task_index:
            entry = {
                "task_id": tid,
                "seconds": round(res.seconds, 2),
                "n_candidates": int(res.n_candidates),
                "attempts": list(res.sources),
                "source": (res.sources[0] if res.sources else "fallback"),
                "n_test": len(res.attempts),
                "ttt_steps": int(res.ttt_steps),
                "timed_out": bool(res.timed_out),
                "error": res.error,
                "stages": [stage],
                "confidence": round(res.confidence, 3),
                "agree_max": int(res.agree_max),
                "n_distinct": int(res.n_distinct),
                "best_key": (None if res.best_key == float("inf")
                             else round(res.best_key, 4)),
            }
            self.per_task_index[tid] = entry
            if replace:
                for i, old in enumerate(self.report["per_task"]):
                    if old.get("task_id") == tid:
                        self.report["per_task"][i] = entry
                        break
                else:
                    self.report["per_task"].append(entry)
            else:
                self.report["per_task"].append(entry)
        else:
            entry = self.per_task_index[tid]
            entry["seconds"] = round(entry["seconds"] + res.seconds, 2)
            entry["n_candidates"] = int(res.n_candidates)
            entry["attempts"] = list(res.sources)
            entry["source"] = (res.sources[0] if res.sources else "fallback")
            entry["ttt_steps"] = int(res.ttt_steps)
            entry["timed_out"] = bool(res.timed_out)
            entry["confidence"] = round(res.confidence, 3)
            entry["agree_max"] = int(res.agree_max)
            entry["n_distinct"] = int(res.n_distinct)
            entry["best_key"] = (None if res.best_key == float("inf")
                                 else round(res.best_key, 4))
            entry.setdefault("stages", []).append(stage)
            if res.error:
                entry["error"] = res.error
        if res.engine_source:
            # whether a symbolic program actually validated on this task
            entry["engine"] = {"source": res.engine_source,
                               "program": res.engine_program}
        self.state[tid] = res
        self.n_processed += 1
        self.report["n_tasks_done"] = len(self.per_task_index)

        ok1, ok2, n_truth = score_attempts(tid, res.attempts, self.solutions)
        res.n_correct_1, res.n_correct_2, res.n_with_truth = ok1, ok2, n_truth
        entry["conf_agree"] = round(res.conf_agree, 4)
        entry["conf_coverage"] = round(res.conf_coverage, 4)
        if n_truth:
            entry["solved"] = bool(ok1 or ok2)
            entry["attempt_1_correct"] = ok1
            entry["attempt_2_correct"] = ok2
            entry["n_with_truth"] = n_truth
            # Pool recall is the ceiling any selection change could ever reach, and the
            # per-input solved count is where we actually are. The gap between them is the
            # addressable *selection* loss; whatever recall leaves on the table is
            # *generation* loss. Without this split the two are indistinguishable from a run.
            rec = score_pool_recall(res.pools, self.solutions, tid)
            res.n_truth_in_pool = int(rec["n_in_pool"])
            res.n_truth_top1 = int(rec["n_top1"])
            res.truth_ranks = list(rec["ranks"])
            entry["truth_in_pool"] = res.n_truth_in_pool
            entry["truth_top1"] = res.n_truth_top1
            entry["truth_ranks"] = res.truth_ranks
            n_sel = 0
            truth = self.solutions[tid]
            for i, att in enumerate(res.attempts):
                if i >= len(truth):
                    break
                want = validate_grid(truth[i])
                if want is None:
                    continue
                if (np.array_equal(att["attempt_1"], want)
                        or np.array_equal(att["attempt_2"], want)):
                    n_sel += 1
            entry["n_solved_inputs"] = n_sel
            # Recompute from the *unique per-task entries*; never accumulate into the
            # aggregate here. record_task runs once per stage (A, then B, then C), so an
            # incremental accumulator counts the same test input several times: measured on
            # arc26-exp-base it reported n_with_truth = 16 where only 11 test inputs exist,
            # silently deflating pool_recall's denominator. The aggregate must always equal
            # the sum over the per_task rows it claims to summarise.
            agg = {"n_with_truth": 0, "n_in_pool": 0, "n_top1": 0, "n_solved_inputs": 0}
            for e in self.per_task_index.values():
                agg["n_with_truth"] += int(e.get("n_with_truth") or 0)
                agg["n_in_pool"] += int(e.get("truth_in_pool") or 0)
                agg["n_top1"] += int(e.get("truth_top1") or 0)
                agg["n_solved_inputs"] += int(e.get("n_solved_inputs") or 0)
            if agg["n_with_truth"]:
                agg["pool_recall"] = round(agg["n_in_pool"] / agg["n_with_truth"], 4)
                agg["top1_rate"] = round(agg["n_top1"] / agg["n_with_truth"], 4)
                agg["selected_rate"] = round(agg["n_solved_inputs"] / agg["n_with_truth"], 4)
                agg["selection_headroom"] = round(
                    (agg["n_in_pool"] - agg["n_solved_inputs"]) / agg["n_with_truth"], 4)
            self.report["pool_recall"] = agg
        return entry

    def update_submission(self, tid: str, res: TaskResult) -> None:
        self.submission[tid] = res.attempts

    def note_no_gain(self, tid: str, res: TaskResult, stage: str) -> None:
        """Record an extra stage that spent time but did not improve the answer.

        Only the elapsed time and the stage name are touched: the attempts, the
        solved flags and the confidence of the stored (better) result stand.
        """
        entry = self.per_task_index.get(tid)
        if entry is None:
            return
        entry["seconds"] = round(entry["seconds"] + res.seconds, 2)
        entry.setdefault("stages", []).append(stage)

    def log_task(self, tid: str, res: TaskResult, entry: Dict[str, Any]) -> None:
        verdict = ""
        if res.n_with_truth:
            verdict = (f" truth: a1={'OK' if res.n_correct_1 else 'x'}"
                       f"({res.n_correct_1}/{res.n_with_truth})"
                       f" a2={'OK' if res.n_correct_2 else 'x'}")
        self.log(f"  -> {res.seconds:.1f}s candidates={res.n_candidates} "
                 f"conf={res.confidence:.2f} sources={res.sources}"
                 f" task_ok={entry.get('solved')}{verdict}")


def run_uniform(ctx: RunContext) -> None:
    """Spec's original schedule: TTT for every task, dynamic slice per task."""
    args = ctx.args
    scheduler = ctx.scheduler
    stage_t0 = time.time()
    for index, tid in enumerate(ctx.task_ids):
        if scheduler.expired:
            ctx.log(f"[budget] exhausted before task {index + 1}/{len(ctx.task_ids)}; "
                    f"remaining tasks keep their fallback grids")
            break
        remaining_tasks = len(ctx.task_ids) - index
        slice_seconds = scheduler.uniform_slice(index, remaining_tasks)
        ctx.log(f"[task {index + 1}/{len(ctx.task_ids)}] {tid} "
                f"slice={slice_seconds:.0f}s remaining={scheduler.remaining():.0f}s")
        t_task = time.time()
        try:
            res = solve_task(ctx.tasks[tid], tid, ctx.model, ctx.tokenizer, ctx.engine,
                             args, slice_seconds, ctx.rng, do_ttt=True, use_engine=True,
                             scheduler=scheduler, log_fn=ctx.log)
        except Exception as exc:
            res = _task_error_result(ctx, tid, exc)
        ctx.update_submission(tid, res)
        scheduler.record(res.seconds, index)
        ctx.report["calibration"]["seconds_per_task"] = round(scheduler.seconds_per_task, 3)
        entry = ctx.record_task(tid, res, "uniform")
        ctx.log_task(tid, res, entry)
        if ctx.n_processed % 10 == 0:
            flush_submission(args.out, ctx.task_ids, ctx.submission, ctx.log)
            ctx.log(f"  [flush] submission.json written ({ctx.n_processed} tasks done)")
    ctx.report["stages"].append({
        "name": "uniform", "tasks": ctx.n_processed,
        "seconds": round(time.time() - stage_t0, 1),
    })


def _task_error_result(ctx: RunContext, tid: str, exc: BaseException) -> TaskResult:
    """Turn a task-level exception into a fallback result (never fatal)."""
    tb = traceback.format_exc(limit=4)
    ctx.log(f"  ! task {tid} failed: {type(exc).__name__}: {exc}")
    ctx.report["errors"].append({"task_id": tid, "error": str(exc), "trace": tb})
    n_test = max(1, len(ctx.tasks[tid].get("test", [])))
    return TaskResult(
        attempts=fallback_attempts(ctx.tasks[tid]),
        sources=["fallback"] * (2 * n_test),
        error=f"{type(exc).__name__}: {exc}",
    )


def run_cascade(ctx: RunContext) -> None:
    """Cheap sweep over everything, then TTT where it is needed, then backfill.

    Stage A costs seconds per task (single prompt + constrained DFS, no gradients), so
    the whole split is covered before any expensive work starts.  Stage B spends the
    remaining GPU budget on the tasks with the weakest Stage A confidence.  Stage C
    guarantees that every task ends up with engine / heuristic evidence.
    """
    args = ctx.args
    scheduler = ctx.scheduler
    stage_c_reserve = scheduler.stage_c_reserve(len(ctx.task_ids))

    # ---------------- Stage A: cheap full sweep (do_ttt=False) --------------------
    stage_t0 = time.time()
    aug_a = max(1, min(args.aug_infer, 2))  # 2 augmentations give an agreement signal
    n_stage_a = 0
    for index, tid in enumerate(ctx.task_ids):
        if scheduler.expired:
            ctx.log(f"[stage A] budget exhausted at {index + 1}/{len(ctx.task_ids)}")
            break
        remaining_after_reserve = max(
            STAGE_A_MIN_SLICE,
            (scheduler.remaining() - scheduler.reserve_seconds - stage_c_reserve)
            / max(1, len(ctx.task_ids) - index),
        )
        slice_a = max(STAGE_A_MIN_SLICE,
                      min(scheduler.cheap_slice(index), remaining_after_reserve))
        ctx.log(f"[stage A {index + 1}/{len(ctx.task_ids)}] {tid} "
                f"slice={slice_a:.0f}s remaining={scheduler.remaining():.0f}s")
        try:
            res = solve_task(ctx.tasks[tid], tid, ctx.model, ctx.tokenizer, ctx.engine,
                             args, slice_a, ctx.rng, do_ttt=False, use_engine=False,
                             aug_override=aug_a, scheduler=scheduler, log_fn=ctx.log)
        except Exception as exc:
            res = _task_error_result(ctx, tid, exc)
        ctx.update_submission(tid, res)
        n_stage_a += 1
        if index < scheduler.calibrate_tasks:
            scheduler.record(res.seconds, index)
            scheduler.observe_cheap(res.seconds)
            ctx.report["calibration"]["seconds_per_task"] = round(
                scheduler.seconds_per_task, 3)
        entry = ctx.record_task(tid, res, "A_sweep")
        ctx.log_task(tid, res, entry)
        if n_stage_a % 10 == 0:
            flush_submission(args.out, ctx.task_ids, ctx.submission, ctx.log)
            ctx.log(f"  [flush] submission.json written ({n_stage_a} stage-A tasks)")
    ctx.report["stages"].append({
        "name": "A_sweep", "tasks": n_stage_a,
        "seconds": round(time.time() - stage_t0, 1),
        "seconds_per_task": round(scheduler.cheap_seconds or 0.0, 3),
        "aug_infer": aug_a,
    })

    # ---------------- Stage B: TTT refinement, weakest tasks first ----------------
    stage_t0 = time.time()
    ranked = sorted(
        ctx.task_ids,
        key=lambda t: (ctx.state[t].confidence if t in ctx.state else 0.0,
                       ctx.state[t].best_key if t in ctx.state else float("inf")),
    )
    n_stage_b = 0
    if ctx.model is None:
        ctx.log("[stage B] skipped: no model is loaded (nothing to train)")
    for pos, tid in enumerate(ranked if ctx.model is not None else []):
        remaining_refinable = len(ranked) - pos
        if scheduler.remaining() <= scheduler.reserve_seconds + stage_c_reserve + 30.0:
            ctx.log(f"[stage B] stopping: reserve reached "
                    f"(remaining={scheduler.remaining():.0f}s)")
            break
        slice_b = scheduler.stage_b_slice(remaining_refinable, stage_c_reserve)
        prev = ctx.state.get(tid)
        prev_pools = prev.pools if prev is not None else None
        ctx.log(f"[stage B {pos + 1}/{len(ranked)}] {tid} "
                f"conf={0.0 if prev is None else prev.confidence:.2f} "
                f"slice={slice_b:.0f}s remaining={scheduler.remaining():.0f}s")
        try:
            res = solve_task(ctx.tasks[tid], tid, ctx.model, ctx.tokenizer, ctx.engine,
                             args, slice_b, ctx.rng, do_ttt=True, use_engine=False,
                             pool_prev=prev_pools, scheduler=scheduler, log_fn=ctx.log)
        except Exception as exc:
            ctx.log(f"  ! stage B task {tid} failed: {type(exc).__name__}: {exc}")
            ctx.report["errors"].append({"task_id": tid,
                                         "error": f"stage B: {exc}",
                                         "trace": traceback.format_exc(limit=4)})
            continue
        scheduler.observe_ttt_task(res.seconds)
        improved = (prev is None) or (res.best_key < prev.best_key) or \
                   (res.confidence > prev.confidence + 1e-9)
        if improved:
            ctx.update_submission(tid, res)
            entry = ctx.record_task(tid, res, "B_ttt", replace=True)
            ctx.log_task(tid, res, entry)
        else:
            ctx.log(f"  -> {res.seconds:.1f}s conf={res.confidence:.2f} "
                    f"(no improvement over stage A; attempts kept)")
            ctx.note_no_gain(tid, res, "B_ttt_no_gain")
            ctx.state[tid] = prev if prev is not None else res
            ctx.update_submission(tid, ctx.state[tid])
        n_stage_b += 1
        if n_stage_b % 10 == 0:
            flush_submission(args.out, ctx.task_ids, ctx.submission, ctx.log)
            ctx.log(f"  [flush] submission.json written ({n_stage_b} stage-B tasks)")
    ctx.report["stages"].append({
        "name": "B_ttt", "tasks": n_stage_b,
        "seconds": round(time.time() - stage_t0, 1),
        "seconds_per_task": round(scheduler.ttt_task_seconds or 0.0, 3),
    })

    # ---------------- Stage C: symbolic / heuristic backfill ----------------------
    stage_t0 = time.time()
    n_stage_c = 0
    weak = [tid for tid in ctx.task_ids
            if tid not in ctx.state or not ctx.state[tid].confident]
    if ctx.engine.get("module") is None:
        # No engine: the heuristic floor is already in every slot (seeded at startup
        # and re-selected at the end of each stage), so a third neural pass would only
        # burn the reserve.  Say so instead of pretending to work.
        ctx.log(f"[stage C] skipped: no symbolic engine loaded "
                f"({len(weak)} task(s) kept their stage A/B + fallback answers)")
    else:
        ctx.log(f"[stage C] backfilling {len(weak)} task(s) below confidence "
                f"{CONFIDENCE_BAR}")
    for tid in (weak if ctx.engine.get("module") is not None else []):
        prev = ctx.state.get(tid)
        prev_pools = prev.pools if prev is not None else None
        slice_c = min(60.0, max(5.0, scheduler.remaining() - 10.0))
        try:
            res = solve_task(ctx.tasks[tid], tid, ctx.model, ctx.tokenizer, ctx.engine,
                             args, slice_c, ctx.rng, do_ttt=False, use_engine=True,
                             aug_override=1, pool_prev=prev_pools,
                             scheduler=scheduler, log_fn=ctx.log)
        except Exception as exc:
            ctx.log(f"  ! stage C task {tid} failed: {type(exc).__name__}: {exc}")
            continue
        improved = (prev is None) or (res.best_key < prev.best_key) or \
                   (res.confidence > prev.confidence + 1e-9)
        if improved:
            ctx.update_submission(tid, res)
            entry = ctx.record_task(tid, res, "C_backfill", replace=True)
            ctx.log_task(tid, res, entry)
        else:
            ctx.note_no_gain(tid, res, "C_backfill_no_gain")
        n_stage_c += 1
    ctx.report["stages"].append({
        "name": "C_backfill", "tasks": n_stage_c,
        "seconds": round(time.time() - stage_t0, 1),
    })
    flush_submission(args.out, ctx.task_ids, ctx.submission, ctx.log)


# ======================================================================================
# 16. Entry point
# ======================================================================================

def submission_is_worthless(model: Any, engine: Optional[Dict[str, Any]]) -> bool:
    """True when the run can only emit the heuristic floor for every task.

    A model-less run is legitimate *only* with the symbolic engine (``--engine``). With
    neither, every task falls back to ``fallback_pair`` -- and because the model-load
    failure is caught and the run continues, the notebook still exits 0 and leaves a
    schema-valid, 100%-fallback ``submission.json``. That is strictly worse than failing:
    it wastes a submission slot and looks like success. Observed 2026-09-14 when Kaggle
    handed out a Tesla P100 the installed torch cannot execute.
    """
    return model is None and not (engine or {}).get("module")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the solver. Returns an exit code; never raises out of the notebook cell."""
    args = parse_args(argv)
    # `turbo_dfs` / `constrained_generate` read these as module globals at call time, so
    # assigning here is what actually changes the search width for this run.
    global DFS_TOKEN_PROB_THRESHOLD, DFS_TOKEN_LOGPROB_THRESHOLD
    global DFS_MAX_BRANCHES_PER_BEAM, DFS_MAX_NODES
    global LORA_R
    DFS_TOKEN_PROB_THRESHOLD = float(args.dfs_prob_threshold)
    DFS_TOKEN_LOGPROB_THRESHOLD = math.log(max(1e-6, min(1.0, DFS_TOKEN_PROB_THRESHOLD)))
    DFS_MAX_BRANCHES_PER_BEAM = max(1, int(args.dfs_max_branches))
    DFS_MAX_NODES = max(100, int(args.dfs_max_nodes))
    # `_wrap_lora_modules` reads LORA_R at call time, same pattern as the DFS globals above.
    LORA_R = max(1, int(args.lora_r))
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    started_at = time.time()
    report: Dict[str, Any] = {
        "started_at": utc_stamp(),
        "finished_at": None,
        "elapsed_seconds": None,
        "split": args.split,
        "schedule": args.schedule,
        "n_tasks_total": 0,
        "n_tasks_done": 0,
        "n_tasks_processed": 0,
        "calibration": {"tasks": args.calibrate_tasks, "seconds_per_task": 0.0,
                        "seconds_per_task_ttt": None},
        "per_task": [],
        "stages": [],
        "gpu": describe_gpu(),
        "torch": (getattr(torch, "__version__", "unavailable")
                  if torch is not None else "unavailable"),
        "peak_mem_gb": 0.0,
        "args": vars(args),
        "dfs": {"prob_threshold": DFS_TOKEN_PROB_THRESHOLD,
                "max_branches": DFS_MAX_BRANCHES_PER_BEAM,
                "max_nodes": DFS_MAX_NODES},
        "model_dir": args.model_dir,
        "model_load_seconds": None,
        "model_error": None,
        "engine": {"path": args.engine, "status": "skipped", "reason": ""},
        "data": {},
        "n_solved": 0,
        "n_tasks_scored": 0,
        "accuracy": None,
        "n_tasks_with_real_candidate": 0,
        "errors": [],
    }
    task_ids: List[str] = []
    submission: Dict[str, List[Dict[str, np.ndarray]]] = {}

    def log(message: str) -> None:
        print(message, flush=True)

    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.reset_peak_memory_stats()
        except Exception:
            pass

    try:
        # ---- data -----------------------------------------------------------------
        tasks, solutions, data_info = load_split(args.data_dir, args.split)
        report["data"] = data_info
        task_ids = sorted(tasks.keys())
        # Sharding: the 30 h GPU budget is spent as two concurrent 12 h sessions (Kaggle
        # allows at most 2 GPU sessions at once), so the 240 tasks have to be split
        # across them. Interleaved rather than blocked, so any task-difficulty ordering
        # in the key space is spread evenly instead of landing in one shard.
        if args.num_shards and args.num_shards > 1:
            if not (0 <= args.shard_index < args.num_shards):
                raise ValueError(f"--shard-index must be in [0, {args.num_shards})")
            task_ids = [t for i, t in enumerate(task_ids) if i % args.num_shards == args.shard_index]
            log(f"[shard] {args.shard_index}/{args.num_shards} -> {len(task_ids)} tasks")
        if args.limit and args.limit > 0:
            task_ids = task_ids[: args.limit]
        report["n_tasks_total"] = len(task_ids)
        report["shard"] = {"index": args.shard_index, "num_shards": args.num_shards}
        report["n_with_solutions"] = len(solutions) if solutions else 0
        log(f"[data] split={args.split} n_tasks={len(task_ids)} "
            f"challenges={data_info['challenges_path']} "
            f"solutions={data_info['solutions_path']}")

        # ---- never leave a task blank: seed every slot with the heuristic floor ---
        for tid in task_ids:
            submission[tid] = fallback_attempts(tasks[tid])
        flush_submission(args.out, task_ids, submission, log)

        # ---- engine ---------------------------------------------------------------
        engine = load_engine(args.engine, log)
        report["engine"] = engine["info"]
        log(f"[engine] status={engine['info']['status']} {engine['info'].get('reason', '')}")

        # Reserve for the wrap-up (final scoring, flush, report write).  Scaled to the
        # budget so that a short smoke test is not expired before it starts: with the
        # spec's 1800 s smoke run this is 180 s, with the real 39000 s budget it is the
        # full 300 s.
        reserve = float(min(max(30.0, 0.1 * args.time_budget_seconds),
                            0.5 * args.time_budget_seconds, 300.0))
        scheduler = BudgetScheduler(
            args.time_budget_seconds, len(task_ids), args.calibrate_tasks,
            started_at=started_at, reserve_seconds=reserve, log_fn=log,
        )
        report["reserve_seconds"] = round(reserve, 1)
        ctx = RunContext(args=args, tasks=tasks, task_ids=task_ids,
                         solutions=solutions, submission=submission, report=report,
                         scheduler=scheduler, engine=engine, rng=rng, log=log)

        # ---- model ----------------------------------------------------------------
        if torch is None:
            report["model_error"] = "torch is not importable in this environment"
            log(f"[model] {report['model_error']}")
        else:
            t_load = time.time()
            try:
                ctx.model, ctx.tokenizer = load_model_and_tokenizer(args.model_dir)
                report["model_load_seconds"] = round(time.time() - t_load, 2)
                log(f"[model] loaded {args.model_dir} in {report['model_load_seconds']}s")
            except Exception as exc:
                report["model_error"] = f"{type(exc).__name__}: {exc}"
                ctx.model = ctx.tokenizer = None
                log(f"[model] FAILED to load: {report['model_error']}")

        if submission_is_worthless(ctx.model, engine):
            # Abort loudly instead of emitting an all-fallback submission. SystemExit is a
            # BaseException, so the `except Exception` below will not swallow it; the
            # finally block sees report["aborted"] and removes the submission file.
            report["aborted"] = "model_unavailable_and_no_engine"
            report["errors"].append({"task_id": None, "error": report["model_error"],
                                     "trace": "aborted before any task ran"})
            log("[abort] no model and no symbolic engine: every task would be the heuristic "
                "floor, so this run must not be scored")
            raise SystemExit(2)

        # ---- schedule -------------------------------------------------------------
        log(f"[schedule] {args.schedule} budget={args.time_budget_seconds:.0f}s "
            f"calibrate={args.calibrate_tasks} aug_train={args.aug_train} "
            f"aug_infer={args.aug_infer}")
        if args.schedule == "cascade":
            run_cascade(ctx)
        else:
            run_uniform(ctx)

        # ---- final scoring --------------------------------------------------------
        report["n_tasks_processed"] = ctx.n_processed
        report["n_tasks_done"] = len(ctx.per_task_index)
        report["n_tasks_with_real_candidate"] = sum(
            1 for tid in task_ids
            if ctx.per_task_index.get(tid, {}).get("source") in ("neural", "symbolic")
        )
        source_counts: Dict[str, int] = {}
        for entry in report["per_task"]:
            key = str(entry.get("source", "fallback"))
            source_counts[key] = source_counts.get(key, 0) + 1
        report["source_counts"] = source_counts
        report["engine"]["n_tasks_validated"] = sum(
            1 for e in report["per_task"]
            if (e.get("engine") or {}).get("source") == "search"
        )
        report["engine"]["n_tasks_prior_only"] = sum(
            1 for e in report["per_task"]
            if (e.get("engine") or {}).get("source") == "prior"
        )
        report["calibration"]["seconds_per_task_ttt"] = (
            round(scheduler.ttt_task_seconds, 3)
            if scheduler.ttt_task_seconds else None
        )
        if solutions:
            tried = [e for e in report["per_task"] if e.get("n_with_truth")]
            solved = [e for e in tried if e.get("solved")]
            report["n_tasks_scored"] = len(tried)
            report["n_solved"] = len(solved)
            report["accuracy"] = (len(solved) / len(tried)) if tried else None
            log(f"[score] solved {len(solved)}/{len(tried)} tasks with ground truth")
        return 0
    except Exception as exc:  # data errors etc.: report, but still write outputs
        report["errors"].append({"task_id": None, "error": str(exc),
                                 "trace": traceback.format_exc(limit=6)})
        log(f"! fatal: {type(exc).__name__}: {exc}")
        return 1
    finally:
        try:
            if torch is not None and torch.cuda.is_available():
                report["peak_mem_gb"] = round(
                    torch.cuda.max_memory_allocated() / (1024 ** 3), 3
                )
        except Exception:
            pass
        try:
            if report.get("aborted"):
                # Nothing here is worth scoring; leaving submission.json in place would
                # hand the platform a schema-valid, all-fallback artifact.
                if os.path.exists(args.out):
                    os.remove(args.out)
                    log(f"[abort] removed {args.out} so this run cannot be scored")
            else:
                flush_submission(args.out, task_ids or list(submission.keys()), submission, log)
        except Exception:
            pass
        report["finished_at"] = utc_stamp()
        report["elapsed_seconds"] = round(time.time() - started_at, 2)
        if task_ids:
            report["n_tasks_total"] = len(task_ids)

        # --- adaptation telemetry: intended steps are not executed steps --------------
        # The failure this guards against is silent by construction: a task whose TTT stage
        # cannot allocate is recorded exactly like a task whose TTT stage ran and did not
        # help. Measured on arc26-submit-full, that let 86 of 102 tasks be reported as
        # "B_ttt_no_gain" after executing zero optimizer steps, and the run's accuracy was
        # read as a statement about model capability rather than about a failed allocation.
        # A run-level counter is the cheapest thing that makes the difference visible.
        per_task = report.get("per_task") or []
        routed = [e for e in per_task
                  if any(str(s).startswith("B_ttt") for s in (e.get("stages") or []))]
        stepped = [e for e in routed if int(e.get("ttt_steps") or 0) > 0]
        total_steps = sum(int(e.get("ttt_steps") or 0) for e in per_task)
        stage_b = next((s for s in (report.get("stages") or [])
                        if s.get("name") == "B_ttt"), {})
        report["ttt"] = {
            "tasks_routed": len(routed),
            "tasks_executed": len(stepped),
            "tasks_routed_but_never_stepped": len(routed) - len(stepped),
            "total_optimizer_steps": total_steps,
            "stage_b_seconds": stage_b.get("seconds"),
            "seconds_per_step": (round(stage_b["seconds"] / total_steps, 1)
                                 if total_steps and stage_b.get("seconds") else None),
            "seq_token_budget": args.ttt_seq_tokens,
            "disabled_reason": TTT_DISABLED_REASON,
        }
        if routed:
            log(f"[ttt] routed={len(routed)} executed={len(stepped)} "
                f"never_stepped={len(routed) - len(stepped)} steps={total_steps}"
                + (f" disabled={TTT_DISABLED_REASON}" if TTT_DISABLED_REASON else ""))
            if not stepped:
                log("! WARNING: TTT was routed tasks but executed ZERO optimizer steps. This run "
                    "is a beam search over an UNADAPTED model; do not report it as a TTT result.")
            elif len(stepped) < len(routed):
                log(f"! WARNING: {len(routed) - len(stepped)}/{len(routed)} TTT tasks never "
                    "stepped; their 'no gain' verdicts say nothing about adaptation.")

        # --- submission integrity: the placeholder is the default in the reference pipeline ---
        # In the strongest published implementation of this recipe, a puzzle whose TTT times out
        # writes no output at all and the submission builder then defaults it to `[[0]]`, which
        # is worth exactly zero. That is the documented mechanism behind the largest local-to-LB
        # gap in the competition's forum (11.67% LB against ~36% local), and it is silent: a
        # placeholder looks identical to a wrong answer. We seed every task with a real fallback
        # before solving begins, so this is a check that the seed survived -- not a hope.
        sub = report.get("submission_health") or {}
        placeholder = 0
        missing_slots = 0
        for tid in (task_ids or []):
            entries = submission.get(tid) or []
            want = len((tasks.get(tid) or {}).get("test") or [])
            if want and len(entries) != want:
                missing_slots += 1
            for e in entries:
                for key in ("attempt_1", "attempt_2"):
                    g = e.get(key)
                    if g is None:
                        continue
                    a = np.asarray(g)
                    if a.size == 1 and int(a.ravel()[0]) == 0:
                        placeholder += 1
        sub.update({"placeholder_attempts": placeholder,
                    "tasks_with_wrong_entry_count": missing_slots,
                    "tasks_expected": len(task_ids or [])})
        report["submission_health"] = sub
        if placeholder:
            log(f"! WARNING: {placeholder} attempt(s) are the [[0]] placeholder, which scores "
                "exactly zero. Every task should carry a real fallback before solving starts.")
        if missing_slots:
            log(f"! WARNING: {missing_slots} task(s) have the wrong number of entries for their "
                "test inputs; the submission may be rejected as malformed.")
        if not placeholder and not missing_slots and task_ids:
            log(f"[submission] healthy: {len(task_ids)} tasks, no placeholders, entry counts ok")

        try:
            write_json_atomic(args.report, report)
            log(f"[done] submission -> {args.out}  report -> {args.report}  "
                f"elapsed={report['elapsed_seconds']}s")
        except Exception as exc:
            print(f"! could not write report.json: {exc}", flush=True)


if __name__ == "__main__":
    # Propagate the exit code: bare `main()` discarded it, so "return 1" had no effect on
    # whether Kaggle considered the run failed.
    raise SystemExit(main())
