#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test_model_free.py -- validate every part of arc26_solver that does NOT need the model,
against the REAL ARC-AGI-2 data, on a machine with no GPU and no torch.

Why this exists: the solver only imports stdlib + numpy at module level (torch and
transformers are lazily imported inside functions), so ~everything except the forward
pass is testable locally. Those parts are exactly where a bug costs accuracy *silently*:
a lossy serialiser, a non-invertible augmentation, a truncated 30x30 reply, two identical
attempts in a task, or a candidate selector that drops the best answer.

The tokenizer is faked from the authoritative vocabulary recovered from the
`arc26-stage1-alphabet-and-cost` run log:

    model.vocab = {"0":0 ... "9":9, "Ċ":10, "user":11, "assistant":12}
    added       = 13:"<|endoftext|>"  14:"<|im_start|>"  15:"<|im_end|>"
    ('Ċ' is ByteLevel's encoding of "\\n", so NEWLINE_TOKEN_ID = 10)

That makes the *entire* text->token->text path testable without transformers, including
the exact strings fmt_train/fmt_query/fmt_reply produce.

Usage:  python tests/test_model_free.py [--data-root PATH]
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ARC_W1 = HERE.parent
sys.path.insert(0, str(ARC_W1))
import arc26_solver as S  # noqa: E402

FAILS: list[str] = []
COUNTS: dict[str, tuple[int, int]] = {}


def check(name: str, ok: bool, total: int, first_bad: list[str] | None = None) -> None:
    good, all_ = COUNTS.get(name, (0, 0))
    COUNTS[name] = (good + (1 if ok else 0), all_ + 1)
    if not ok:
        FAILS.append(f"{name}: {(first_bad or [''])[0][:200]}")


def report(name: str) -> None:
    good, all_ = COUNTS.get(name, (0, 0))
    mark = "PASS" if good == all_ else "FAIL"
    print(f"  [{mark}] {name:44s} {good}/{all_}")


# --------------------------------------------------------------------------------------
# Fake tokenizer built from the recovered vocabulary
# --------------------------------------------------------------------------------------
VOCAB_STR = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "Ċ", "user", "assistant"]
ADDED = {13: "<|endoftext|>", 14: "<|im_start|>", 15: "<|im_end|>"}
TOK2ID = {s: i for i, s in enumerate(VOCAB_STR)}
TOK2ID.update({s: i for i, s in ADDED.items()})
ID2TOK = {i: s for s, i in TOK2ID.items()}
_BY_LEN = sorted(TOK2ID, key=len, reverse=True)


class FakeTokenizer:
    """Greedy longest-match encoder over the real 16-token vocabulary.

    Mirrors ByteLevel: the pre-tokenizer maps "\\n" to "Ċ" before matching, and the
    decoder maps "Ċ" back to "\\n". Special tokens (13/14/15) match literally.
    """

    def __call__(self, text, add_special_tokens=False):
        text = text.replace("\n", "Ċ")
        ids, i = [], 0
        while i < len(text):
            for tok in _BY_LEN:
                if text.startswith(tok, i):
                    ids.append(TOK2ID[tok])
                    i += len(tok)
                    break
            else:
                raise ValueError(f"cannot encode {text[i:i+8]!r} with the 16-token vocabulary")
        return {"input_ids": ids}

    def decode(self, ids):
        return "".join(ID2TOK[int(t)] for t in ids).replace("Ċ", "\n")


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------
def load_tasks(data_root: Path):
    out = {}
    for split in ("evaluation", "training"):
        d = data_root / split
        if not d.is_dir():
            continue
        tasks = {}
        for f in sorted(d.glob("*.json")):
            tasks[f.stem] = json.loads(f.read_text(encoding="utf-8"))
        out[split] = tasks
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", default=str(ARC_W1.parent.parent / "ARC-AGI-2-main" / "data"))
    args = ap.parse_args()
    data_root = Path(args.data_root)
    print(f"data root: {data_root}")
    splits = load_tasks(data_root)
    if not splits:
        print("no data found; pass --data-root")
        return 2
    for s, t in splits.items():
        print(f"  {s}: {len(t)} tasks")
    tok = FakeTokenizer()

    # ---- T1: grid -> string -> grid round-trip -------------------------------------
    for split, tasks in splits.items():
        for tid, task in tasks.items():
            for pi, pair in enumerate(task["train"] + task["test"]):
                for which in ("input", "output"):
                    g = pair.get(which)
                    if g is None:
                        continue
                    back = S.parse_grid_string(S.convert_grid_to_string(g))
                    ok = back is not None and np.array_equal(back, np.array(g, dtype=int))
                    check(f"T1 string round-trip ({split})", ok, 1,
                          [f"{tid}[{pi}].{which} shape={np.shape(g)}"])

    # ---- T2: full generation path: grid -> fmt_reply -> tokens -> grid --------------
    for split, tasks in splits.items():
        for tid, task in tasks.items():
            for pi, pair in enumerate(task["train"]):
                g = pair["output"]
                text = S.fmt_reply(g)
                try:
                    ids = tok(text)["input_ids"]
                except ValueError as e:
                    check(f"T2 reply encode ({split})", False, 1, [f"{tid}[{pi}] {e}"])
                    continue
                back = S.tokens_to_array(ids, tokenizer=tok)
                ok = back is not None and np.array_equal(back, np.array(g, dtype=int))
                check(f"T2 reply encode->decode->grid ({split})", ok, 1,
                      [f"{tid}[{pi}] shape={np.shape(g)} back={None if back is None else back.shape}"])

    # ---- T2b: prompt is encodable and ends with the assistant turn ------------------
    for split, tasks in splits.items():
        for tid, task in list(tasks.items())[:400]:
            ok, tail = False, ""
            try:
                text = S.fmt_train(task["train"], task["test"][0]["input"])
                ids = tok(text)["input_ids"]
                tail = str(ids[-3:])
                ok = len(ids) > 0 and ids[-3:] == [14, 12, 10]
            except Exception as exc:
                tail = repr(exc)[:90]
            check(f"T2b prompt encodable ({split})", ok, 1, [f"{tid} tail={tail}"])

    # ---- T3: augmentation invertibility (exhaustive over the geometric group) -------
    rng = random.Random(0)
    sample_grids = []
    for split, tasks in splits.items():
        for tid, task in list(tasks.items())[:60]:
            p = task["train"][0]
            sample_grids.append((tid, p["input"]))
            sample_grids.append((tid, p["output"]))
    perms = [None, tuple(range(10)), tuple(reversed(range(10))), (1, 0, 2, 3, 4, 5, 6, 7, 8, 9)]
    for rot, flip, trans, perm in itertools.product(range(4), (False, True), (False, True), perms):
        key = S.AugKey(rot=rot, flip=flip, transpose=trans, perm=perm)
        for tid, g in sample_grids:
            a = S.apply_augment(g, key)
            b = S.invert_augment(a, key)
            ok = np.array_equal(b, np.array(g, dtype=int))
            check("T3 augment invertible", ok, 1,
                  [f"{tid} key={key.to_json()} {np.shape(g)}->{a.shape}"])

    # ---- T4: augment_demos keeps input/output consistent ---------------------------
    for split, tasks in splits.items():
        for tid, task in list(tasks.items())[:200]:
            key = S.random_augment_key(rng, allow_color=True)
            # invert-augmented outputs must be exactly the original outputs
            aug = S.augment_demos(task["train"], key)
            ok = all(np.array_equal(S.invert_augment(a["output"], key),
                                    np.array(d["output"], dtype=int))
                     for a, d in zip(aug, task["train"]))
            check("T4 augment_demos invertible", ok, 1, [f"{tid} key={key.to_json()}"])

    # ---- T5: max_new_tokens must fit a full 30x30 reply ----------------------------
    full = np.zeros((S.MAX_GRID_SIDE, S.MAX_GRID_SIDE), dtype=int)
    need = len(tok(S.fmt_reply(full)))
    limit = S.max_new_tokens_for(tok)
    check("T5 max_new_tokens covers 30x30", limit >= need, 1,
          [f"limit={limit} need={need} (would truncate the last cells)"])

    # ---- T6: fallback_pair invariants on every real task ---------------------------
    for split, tasks in splits.items():
        for tid, task in tasks.items():
            for ti, tp in enumerate(task["test"]):
                a, b = S.fallback_pair(task, ti)
                va, vb = S.validate_grid(a), S.validate_grid(b)
                tshape = np.shape(np.array(tp["input"]))
                ok = (va is not None and vb is not None
                      and va.shape == tshape and vb.shape == tshape)
                check(f"T6a fallback legal+shaped ({split})", ok, 1,
                      [f"{tid}[{ti}] tshape={tshape} a={None if va is None else va.shape}"])
                check(f"T6b fallback attempts differ ({split})",
                      not np.array_equal(va, vb) if (va is not None and vb is not None) else False,
                      1, [f"{tid}[{ti}] both attempts are identical -> second slot wasted"])

    # ---- T7: dedupe_pool / select_attempts invariants ------------------------------
    for split, tasks in splits.items():
        for tid, task in list(tasks.items())[:200]:
            fb = S.fallback_pair(task, 0)
            g1 = fb[0]
            pool = [S.Candidate(grid=g1, source="neural", nll=1.0),
                    S.Candidate(grid=g1.copy(), source="neural", nll=0.5),   # duplicate
                    S.Candidate(grid=fb[1], source="symbolic", nll=0.2)]
            ded = S.dedupe_pool(pool)
            check("T7a dedupe removes duplicates", len(ded) == 2, 1,
                  [f"{tid} {len(pool)} -> {len(ded)}"])
            a1, a2, s1, s2 = S.select_attempts(pool, fb)
            check("T7b select returns two attempts", a1 is not None and a2 is not None, 1, [tid])
            check("T7c select returns distinct attempts",
                  not np.array_equal(np.asarray(a1), np.asarray(a2)), 1,
                  [f"{tid} sources={s1},{s2}"])
            e1, e2, es1, es2 = S.select_attempts([], fb)
            check("T7d empty pool falls back", e1 is not None and e2 is not None, 1, [tid])

    # ---- T8: validate_grid edge cases ----------------------------------------------
    cases = [
        (None, False), ([], False), ([[1, 2], [3]], False), ([[0]], True),
        ([[1, 2], [3, 4]], True), ([[10]], False), ([[-1]], False),
        (np.zeros((30, 30), dtype=int), True), (np.zeros((31, 1), dtype=int), False),
        ([[1.5]], True),  # cast to int, allowed
    ]
    for g, expect in cases:
        got = S.validate_grid(g) is not None
        check("T8 validate_grid", got == expect, 1, [f"{g!r:.40} expected {expect} got {got}"])

    # ---- T9: pool-recall diagnostic separates generation loss from selection loss ----
    def cand(g, nll, source="neural"):
        return S.Candidate(grid=np.asarray(g, dtype=int), source=source, nll=nll)

    truth = np.array([[1, 1], [1, 1]])
    other = np.array([[2, 2], [2, 2]])
    third = np.array([[3, 3], [3, 3]])
    # truth is the best candidate -> rank 0
    check("T9a truth at rank 0", S.truth_rank_in_pool([cand(truth, 0.1), cand(other, 0.9)], truth) == 0, 1)
    # truth is present but ranked third -> the loss here is pure selection
    pool = [cand(other, 0.1), cand(third, 0.2), cand(truth, 0.9)]
    check("T9b truth ranked 3rd is found", S.truth_rank_in_pool(pool, truth) == 2, 1,
          [f"got {S.truth_rank_in_pool(pool, truth)}"])
    # truth absent -> generation loss, no selector can fix it
    check("T9c absent truth returns None",
          S.truth_rank_in_pool([cand(other, 0.1), cand(third, 0.2)], truth) is None, 1)
    # source priority must be part of the ranking used by the diagnostic
    sym = cand(truth, 0.9, source="symbolic")
    check("T9d ranking honours the symbolic bonus",
          S.truth_rank_in_pool([cand(other, 1.0), sym], truth) == 0, 1)
    # aggregate over a 3-input task: 2 present (ranks 0 and 1), 1 absent
    pools = [[cand(truth, 0.1)], [cand(other, 0.1), cand(truth, 0.5)], [cand(other, 0.1)]]
    sols = {"t": [[[1, 1], [1, 1]], [[1, 1], [1, 1]], [[1, 1], [1, 1]]]}
    rec = S.score_pool_recall(pools, sols, "t")
    check("T9e aggregate recall counts", rec["n_with_truth"] == 3 and rec["n_in_pool"] == 2
          and rec["n_top1"] == 1 and rec["ranks"] == [0, 1], 1, [str(rec)])
    check("T9f no solutions -> empty recall",
          S.score_pool_recall(pools, None, "t")["n_with_truth"] == 0, 1)

    # ---- T10: on real evals the diagnostic runs and is self-consistent ---------------
    real_sols = {}
    ev = splits.get("evaluation") or {}
    for tid, task in ev.items():
        outs = [p.get("output") for p in task["test"]]
        if all(o is not None for o in outs):
            real_sols[tid] = outs
    if real_sols:
        tid = sorted(real_sols)[0]
        task = ev[tid]
        fb = S.fallback_pair(task, 0)
        pools = [[S.Candidate(grid=fb[0], source="neural", nll=0.5),
                  S.Candidate(grid=fb[1], source="neural", nll=0.9)]]
        rec = S.score_pool_recall(pools, real_sols, tid)
        check("T10a real-task recall callable", rec["n_with_truth"] == 1, 1, [str(rec)])
        # put the true grid in the pool: recall must become 1
        want = np.asarray(real_sols[tid][0], dtype=int)
        pools2 = [[S.Candidate(grid=want, source="neural", nll=0.5)]]
        rec2 = S.score_pool_recall(pools2, real_sols, tid)
        check("T10b truth injected is detected", rec2["n_in_pool"] == 1 and rec2["n_top1"] == 1, 1,
              [str(rec2)])

    # ---- report ---------------------------------------------------------------------
    print()
    for name in sorted(COUNTS):
        report(name)
    print()
    print(f"{'ALL PASS' if not FAILS else 'FAILURES'} — {len(FAILS)} failing check group(s)")
    for f in FAILS[:25]:
        print("   -", f)
    return 1 if FAILS else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        traceback.print_exc()
        sys.exit(2)
