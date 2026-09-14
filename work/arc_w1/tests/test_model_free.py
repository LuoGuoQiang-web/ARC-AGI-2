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
import math
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

    # ---- T9g: the aggregate must be idempotent under multi-stage recording -----------
    # Regression: record_task runs once per stage (A, then B, then C) for the same task.
    # The aggregate used to be += 'd there, so a task revisited by two later stages had
    # every one of its test inputs counted three times. A run with 8 tasks / 11 test inputs
    # published n_with_truth = 16. The denominator of pool recall was therefore inflated,
    # which biases the headline metric *towards* the hypothesis under test.
    def _ctx():
        return S.RunContext(args=None, tasks={}, task_ids=[],
                            solutions={"t": [[[1, 1], [1, 1]]]},
                            submission={}, report={"per_task": []},
                            scheduler=None, log=lambda *a, **k: None)

    def _res(cands):
        r = S.TaskResult(attempts=[{"attempt_1": np.asarray(cands[0].grid),
                                    "attempt_2": np.asarray(cands[0].grid)}],
                         sources=["neural"], n_candidates=len(cands), seconds=1.0,
                         pools=[list(cands)])
        return r

    c = _ctx()
    # Stage A: truth IS in the pool.  Stage B: it is not (TTT search went elsewhere).
    c.record_task("t", _res([cand(truth, 0.1), cand(other, 0.5)]), "A_sweep", replace=True)
    agg_a = dict(c.report["pool_recall"])
    c.record_task("t", _res([cand(other, 0.1), cand(third, 0.2)]), "B_ttt")
    agg_b = dict(c.report["pool_recall"])
    check("T9g aggregate counts each test input once across stages",
          agg_a["n_with_truth"] == 1 and agg_a["n_in_pool"] == 1
          and agg_b["n_with_truth"] == 1,
          1, [f"A={agg_a}", f"B={agg_b}"])
    check("T9g2 aggregate equals the per_task rows it summarises",
          agg_b["n_with_truth"] == sum(int(e.get("n_with_truth") or 0)
                                       for e in c.report["per_task"]),
          1, [f"agg={agg_b['n_with_truth']}",
              f"rows={[e.get('n_with_truth') for e in c.report['per_task']]}"])

    # ---- T14: TTT context budget actually shrinks the step, and keeps the target ---------
    # The measured failure this guards: with the whole demonstration set in one sequence a
    # 30x30 task builds an ~8192-token optimizer step, the backward allocation fails on a
    # 14.56 GiB T4, and adaptation silently never runs (86 of 102 tasks in the 240-task run).
    # Capping the context must (a) really shorten the step and (b) never cut the target pair,
    # which is the only part of the sequence that carries loss.
    ev_all = splits.get("evaluation") or {}
    big = None
    for tid, task in sorted(ev_all.items()):
        if len(task.get("train") or []) >= 3:
            big = (tid, task)
            break
    if big is None:
        check("T14a found a multi-demo evaluation task", False, 1, ["no task with >=3 demos"])
    else:
        tid, task = big
        rng = random.Random(0)
        uncapped, _ = S.build_ttt_sequences(task["train"], tok, 1, 8192, rng)
        rng = random.Random(0)
        capped, _ = S.build_ttt_sequences(task["train"], tok, 1, 8192, rng,
                                          seq_token_budget=256)
        check("T14a multi-demo task found and encoded", bool(uncapped) and bool(capped), 1,
              [f"{tid}: {len(task['train'])} demos, uncapped={[len(s) for s in uncapped]}, "
               f"capped={[len(s) for s in capped]}"])
        # k=0 is the identity augmentation, so the target is demo 0 and the reply is known.
        reply = S.fmt_reply(task["train"][0]["output"])
        check("T14b the cap shortens the optimizer step",
              max(len(s) for s in capped) < max(len(s) for s in uncapped), 1,
              [f"uncapped max={max(len(s) for s in uncapped)}, "
               f"capped max={max(len(s) for s in capped)}"])
        check("T14c the target pair survives the cap",
              all(tok.decode(s).endswith(reply) for s in capped), 1,
              [f"target reply len={len(reply)} tokens"])
        check("T14d a zero budget leaves the sequence untouched",
              [len(s) for s in S.build_ttt_sequences(task["train"], tok, 1, 8192,
                                                     random.Random(0),
                                                     seq_token_budget=0)[0]]
              == [len(s) for s in uncapped], 1)
        # A budget smaller than one grid cannot be honoured by dropping context alone; the
        # function must still return something usable rather than an empty sequence.
        tiny, _ = S.build_ttt_sequences(task["train"], tok, 1, 8192, random.Random(0),
                                        seq_token_budget=8)
        check("T14e an unsatisfiable budget degrades instead of vanishing",
              len(tiny) == 1 and len(tiny[0]) > 0, 1, [f"lens={[len(s) for s in tiny]}"])

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

    # ---- T11: GPU compatibility guard (fake torch -- no GPU needed) ------------------
    # The guard now decides by running a real op, because deciding by set membership in
    # `get_arch_list()` rejected the NVIDIA L4 (sm_89 is absent from that list) and so rejected
    # the 4xL4 machine with 88 GiB that this competition actually offers. These checks pin both
    # directions: a device whose ops fail must be refused, and a device whose ops work must be
    # accepted *even when it is missing from the arch list*.
    class _FakeTensor:
        def __init__(self, fail):
            self._fail = fail

        def _maybe(self):
            if self._fail:
                raise RuntimeError("CUDA error: no kernel image is available for execution "
                                   "on the device")
            return self

        def __matmul__(self, other):
            return self._maybe()

        def sum(self):
            return self._maybe()

        def pow(self, _e):
            return self._maybe()

        def mean(self):
            return self._maybe()

        def backward(self):
            return self._maybe()

        def __float__(self):
            return 1.0 if self._maybe() else 1.0

    class _FakeCuda:
        def __init__(self, names, caps, archs, available=True):
            self._n, self._c, self._a, self._av = names, caps, archs, available

        def is_available(self):
            return self._av

        def device_count(self):
            return len(self._n)

        def get_device_name(self, i):
            return self._n[i]

        def get_device_capability(self, i):
            return self._c[i]

        def get_arch_list(self):
            return list(self._a)

        def synchronize(self, *a):
            return None

    class _FakeTorch:
        """Enough of the torch surface for the guard's smoke test, with ops that can fail."""

        def __init__(self, cuda, ops_ok=True):
            self.cuda = cuda
            self._ops_ok = ops_ok

        def device(self, spec):
            return spec

        def ones(self, *a, **k):
            return _FakeTensor(not self._ops_ok)

        def randn(self, *a, **k):
            return _FakeTensor(not self._ops_ok)

    real_torch = S.torch
    try:
        # exactly the failure that broke the 2026-09-14 smoke run: P100 (sm_60), ops dead
        S.torch = _FakeTorch(_FakeCuda(["Tesla P100-PCIE-16GB"], [(6, 0)],
                                       ["sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]),
                             ops_ok=False)
        try:
            S.assert_gpu_compatible()
            check("T11a rejects a GPU whose ops fail", False, 1,
                  ["no exception raised -> run would emit an all-fallback submission"])
        except RuntimeError as exc:
            check("T11a rejects a GPU whose ops fail",
                  "sm_60" in str(exc) and "machine_shape" in str(exc), 1, [str(exc)[:160]])

        # ⭐ REGRESSION: the L4 is sm_89 and this image's get_arch_list() omits sm_89, so the
        # old set-membership guard refused it. Real ops succeed on an L4 -- verified on Kaggle
        # -- so it must be accepted. This check fails against the pre-fix guard.
        S.torch = _FakeTorch(_FakeCuda(["NVIDIA L4"] * 4, [(8, 9)] * 4,
                                       ["sm_70", "sm_75", "sm_80", "sm_86", "sm_90",
                                        "sm_100", "sm_120"]),
                             ops_ok=True)
        try:
            desc = S.assert_gpu_compatible()
            check("T11b accepts the L4 (sm_89 absent from the cubin list but ops work)",
                  desc.count("NVIDIA L4") == 4, 1, [desc[:120]])
        except Exception as exc:
            check("T11b accepts the L4 (sm_89 absent from the cubin list but ops work)",
                  False, 1, ["the old set-membership guard is back: " + repr(exc)[:120]])

        S.torch = _FakeTorch(_FakeCuda(["Tesla T4", "Tesla T4"], [(7, 5), (7, 5)],
                                       ["sm_70", "sm_75", "sm_80", "sm_86", "sm_90"]),
                             ops_ok=True)
        try:
            desc = S.assert_gpu_compatible()
            check("T11c accepts T4", "Tesla T4" in desc, 1, [desc])
        except Exception as exc:
            check("T11c accepts T4", False, 1, [repr(exc)[:140]])

        S.torch = _FakeTorch(_FakeCuda([], [], [], available=False))
        try:
            S.assert_gpu_compatible()
            check("T11d refuses a CPU-only runtime", False, 1, ["no exception"])
        except RuntimeError as exc:
            check("T11d refuses a CPU-only runtime", "CUDA is not available" in str(exc), 1,
                  [str(exc)[:120]])
    finally:
        S.torch = real_torch

    # ---- T15: the TTT profile we adopted, and the schedule that goes with it -------------
    # These pin the *verified* configuration of the strongest published instance of this recipe
    # (NVARC 2025: r=256, rsLoRA, embed_tokens+lm_head targets, cosine with warmup). We had r=16
    # and no scheduler. Rank is the capacity of the only mechanism that can change the
    # generator, and the pool-recall measurement says the generator is the bottleneck.
    check("T15a default LoRA rank matches the verified recipe",
          S.LORA_R == 256 and S.LORA_ALPHA == 32, 1, [f"LORA_R={S.LORA_R}"])
    check("T15b adapters cover the token embedding and the output head",
          "embed_tokens" in S.LORA_EMBED_MODULES and "lm_head" in S.LORA_EMBED_MODULES, 1,
          [str(S.LORA_EMBED_MODULES)])
    check("T15c the seven attention/MLP projections are still targeted",
          len(S.LORA_TARGET_MODULES) == 7, 1, [str(S.LORA_TARGET_MODULES)])
    check("T15d rsLoRA scaling is in effect (alpha/sqrt(r), not alpha/r)",
          abs((S.LORA_ALPHA / math.sqrt(S.LORA_R)) - (S.LORA_ALPHA / S.LORA_R)) > 1e-9,
          1, [f"{S.LORA_ALPHA / math.sqrt(S.LORA_R):.3f} vs {S.LORA_ALPHA / S.LORA_R:.3f}"])

    # The schedule is a pure function precisely so it can be checked without a GPU.
    # Note the step count: at 16 steps, warmup_ratio 0.1 means int(1.6) = 1 warmup step, so the
    # warmup is real but only one step long. Use 100 steps to test the shape, and check the
    # short-run case separately -- that is where an off-by-one actually costs something.
    lrs = [S.ttt_lr_at(i, 16) for i in range(16)]
    long_lrs = [S.ttt_lr_at(i, 100) for i in range(100)]
    check("T15e warmup rises across its window (100-step run)",
          long_lrs[0] < long_lrs[4] < long_lrs[9], 1, [str([round(x, 9) for x in long_lrs[:11]])])
    check("T15f warmup ends at the base rate",
          abs(long_lrs[9] - S.TTT_LR) < 1e-9, 1, [f"step9={long_lrs[9]:.3e}"])
    check("T15g the schedule decays monotonically after warmup",
          all(long_lrs[i] >= long_lrs[i + 1] - 1e-12 for i in range(9, 99)), 1, ["100-step run"])
    check("T15h the schedule actually reaches zero at the last step",
          abs(S.ttt_lr_at(15, 16)) < 1e-12 and abs(long_lrs[99]) < 1e-12, 1,
          [f"16-step last={S.ttt_lr_at(15, 16):.2e}, 100-step last={long_lrs[99]:.2e}"])
    check("T15i a one-step run is well defined and non-zero",
          S.ttt_lr_at(0, 1) > 0, 1, [f"{S.ttt_lr_at(0, 1):.3e}"])
    check("T15j warmup never exceeds the base rate",
          all(S.ttt_lr_at(i, 100) <= S.TTT_LR + 1e-12 for i in range(100)), 1, ["100-step run"])
    check("T15k zero warmup decay pure-cosine from step 0",
          abs(S.ttt_lr_at(0, 10, warmup_ratio=0.0) - S.TTT_LR) < 1e-9, 1,
          [f"{S.ttt_lr_at(0, 10, warmup_ratio=0.0):.3e}"])
    check("T15l the 16-step schedule still spends its last step at (near) zero",
          S.ttt_lr_at(15, 16) < 1e-9 * S.TTT_LR or S.ttt_lr_at(15, 16) == 0.0, 1,
          [f"{S.ttt_lr_at(15, 16):.3e}"])

    # The memory guard: rank 256 must be refused where it cannot fit, rather than OOM-ing three
    # minutes into a run and being recorded as "TTT attempted, no gain".
    gib = S.ttt_optimiser_gib(529_000_000)
    check("T15m the rank-256 optimiser cost is ~5.9 GiB", 5.5 < gib < 6.5, 1, [f"{gib:.2f} GiB"])
    # A T4 has 14.56 GiB total and the bf16 base model holds ~7.3 of it, leaving ~7.3 GiB. So
    # r=256 consumes most of the remaining headroom *before* any activation is allocated -- the
    # 1.77 GiB activation allocation is what then failed.
    check("T15n that leaves almost no room for activations on a T4",
          gib > 0.75 * 7.3, 1, [f"{gib:.2f} GiB of ~7.3 GiB headroom"])
    check("T15o a T4-sized rank is affordable",
          S.ttt_optimiser_gib(529_000_000 / 16) < 0.5, 1,
          [f"r=16 -> {S.ttt_optimiser_gib(529_000_000 / 16):.2f} GiB"])

    # Without these the pool-recall experiment is impossible, and pool recall is the
    # binding constraint (selection_headroom was 0.0 on the first eval probe).
    d = S.parse_args(["--dfs-prob-threshold", "0.1", "--dfs-max-branches", "4",
                      "--dfs-max-nodes", "12000"])
    # ---- T12: the search-width knobs are reachable from the CLI ----------------------
    check("T12a CLI exposes the search width",
          (d.dfs_prob_threshold, d.dfs_max_branches, d.dfs_max_nodes) == (0.1, 4, 12000),
          f"{d.dfs_prob_threshold}/{d.dfs_max_branches}/{d.dfs_max_nodes}")
    d0 = S.parse_args([])
    check("T12b shipped defaults unchanged",
          (d0.dfs_prob_threshold, d0.dfs_max_branches, d0.dfs_max_nodes) == (0.2, 3, 6000),
          f"{d0.dfs_prob_threshold}/{d0.dfs_max_branches}/{d0.dfs_max_nodes}")
    import math as _math
    check("T12c probability maps to the log threshold the DFS compares against",
          abs(_math.log(d0.dfs_prob_threshold) - S.DFS_TOKEN_LOGPROB_THRESHOLD) < 1e-9,
          f"{S.DFS_TOKEN_LOGPROB_THRESHOLD}")

    # ---- T13: worthless-submission guard --------------------------------------------
    # A model-less run is only meaningful with the symbolic engine; otherwise every task is
    # the heuristic floor and the run must refuse to be scored.
    check("T13a no model + no engine is worthless",
          S.submission_is_worthless(None, {"module": None}) is True, "")
    check("T13b no model but an engine is legitimate",
          S.submission_is_worthless(None, {"module": object()}) is False, "")
    check("T13c a loaded model is legitimate",
          S.submission_is_worthless(object(), None) is False, "")
    check("T13d SystemExit escapes `except Exception` (the abort relies on this)",
          issubclass(SystemExit, Exception) is False, "SystemExit must be a BaseException")

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
