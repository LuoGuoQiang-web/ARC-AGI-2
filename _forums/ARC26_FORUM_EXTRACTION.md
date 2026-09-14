# ARC Prize 2026 / ARC-AGI-2 — Discussion-Forum Mining Report

**Corpus:** `C:\Users\LRDC07\Desktop\Kaggle\_forums\arc-prize-2026-arc-agi-2\*.json` — 68 topic files, 234,681 bytes,
snapshot taken 2026-09-14; latest message in corpus 2026-09-13. Every one of the 68 topics is accounted for below.
Plus `_forums\arc-prize-2026-arc-agi-2_topics.json` (topic index with votes).

**Verification work done beyond the corpus** (linked sources the posts merely summarise):
- Live leaderboard + per-submission runtime history: `https://tonghuikang--arc3-leaderboard-monitor-get-history.modal.run?comp=arc2` (the data behind topic 709522's `arc3.huikang.dev` page).
- NVARC official repo `github.com/1ytic/NVARC` (README, `ARChitects/sft_mg.yaml`, `ARChitects/run_sft_4b.sh`, `TRM/README.md`, `SDG/README.md`).
- **The actual NVARC submission notebook source** via `kaggle.com/api/v1/kernels/pull/sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue` — this is the single highest-value artifact found; it contains the real TTT configuration.
- TRM submission notebook `cpmpml/arc2-trm-v31`, nex-n2-mini harness `dominic789654/arc-probe-nex-n2-mini-w4a16`, synthesis harness `dominic789654/arc-agi-2-program-synthesis-harness`.
- Official rules/dates: `arcprize.org/competitions/2026` and `.../2026/arc-agi-2`.

---

## 0. Provenance limitation you must know before reading anything below

**Every `authorName` field in the corpus is empty** — 0 non-empty authors across all 68 topics and all messages
(verified programmatically). The scraper dropped them.

Consequences:
- All forum attribution below is by **topic id + message id + date**, which is exact.
- Where I name a person/handle, it is because **the post self-identifies** (e.g. "Best, Victor" in 695556) or the
  handle appears in reply text (`@cpmpml`, `@madarshbb`, `@lauranoarcanio`, `@gregkamradt`, `@macruzbar`).
- `cpmpml` is Jean-Francois Puget, co-author of NVARC (`1ytic/NVARC` README confirms NVARC = Ivan Sorokin +
  Jean-Francois Puget, both of NVIDIA KGMoN). Several corpus posts are plainly his (he answers as the 2025 winner).
  I mark those "cpmpml (NVARC, 2025 winner)" only where the content makes it unambiguous, and I say so when it is inference.
- **Nothing here can be attributed to a named individual with certainty unless the text itself does it.**

Labels used throughout: **[MEASURED]** = a number someone reports measuring; **[CLAIMED]** = asserted without
supporting detail; **[SPECULATION]** = clearly hypothetical/unverified; **[DISPUTED]** = contradicted in-thread.

---

## 1. Status board as of the corpus snapshot

Corpus-as-of (late July 2026), from topic 724647 / 729509 / 740422:
- rabbithole 1st (67.50 on 2026-07-21; a poster asks about "76" on 2026-09-13).
- nvbanana 2nd — "made a big jump over 70%", reported at 70.42 on 2026-08-23; 2nd team was at 44% on 2026-07-30
  [724647 msgid 3505590, 3516212, 3516290].
- cpmpml (NVARC) states explicitly: *"I have no clue about what they do"* about nvbanana [724647, msgid 3516236,
  2026-08-23]; and *"We won't share more till the deadline."* [724647 msgid 3507814, 2026-08-05].

**Live leaderboard I independently pulled (2026-09-13, the latest row in the monitor):**

| Rank | Team | Best score | Submissions tracked |
|---|---|---|---|
| 1 | rabbithole | **76.94** | 79 |
| 2 | nvbanana | **72.08** | 74 |
| 3 | Yi-Chia Chen | 48.89 | 9 |
| 4 | Tufa Labs | 48.61 | 23 |
| 5 | Junhua Yang | 37.22 | 105 |
| 6 | Kha Vo | 36.39 | 80 |
| 9 | Hiểu Vy | 34.44 | 106 |
| 10 | Team BlackBox | 33.89 | 75 |

So the "76" the 2026-09-09 poster asked about [740422 msgid 3522709] is confirmed: **rabbithole = 76.94**. Note
this is the **public (semi-private, 120-task) score**. The private 120 tasks are unhidden only at the end
[cpmpml, 740422 msgid 3524002, 2026-09-13].

**Official dates (not in corpus, verified at arcprize.org/competitions/2026):** competition started 2026-03-25;
**submissions due 2026-11-02**; papers due 2026-11-08; results 2026-12-04. (The "October 26 deadline" in
topic 740808 is that poster's own claim and is not the official submission date.)

---

## 2. Named systems and their scores

### 2.1 NVARC / ARChitects / KGMoN (the 2025 winning lineage)

| Claim | Value | Provenance | Status |
|---|---|---|---|
| 2025 winning solution on ARC-AGI-2 eval | 31% | 684857 msgid 3431634, 2026-03-30 (cpmpml) | [MEASURED] |
| Same solution on ARC-AGI-1 eval | >95% | 684857 msgid 3431634; 694698 OP, 2026-04-26 | [MEASURED] |
| 2025 competition: public LB / private LB | 27% / 24% | 684857 msgid 3431636, 2026-03-30 | [MEASURED] |
| Post-deadline rerun | 29% | 684857 msgid 3431636; 694698 msgid 3448736 | [MEASURED] |
| 2025 pass@2 vs pass@10 on public eval | 30.5% vs ~40% | 697859 msgid 3455218, 2026-05-08 | [MEASURED, self-reported, hedged "if I remember correctly"] |
| Scorer algo cost them | "Our scorer algorithm missed 10%." | 697859 msgid 3455218 | [CLAIMED — but it is the winner's own number] |
| 2×T4 score | "around 10%" | 684857 msgid 3433210, 2026-04-01 | [MEASURED] |
| Public 2×T4 notebook w/ 2B model | "close to 20%" | 684857 msgid 3437236, 2026-04-07 | [MEASURED, notebook made public] |
| H100 vs L4 | "similar score but at a much lower cost" | 684857 msgid 3432280 | [CLAIMED] |

**The gap explanation, in the winner's own words** [684857 msgid 3433210, 2026-04-01]:
> "Compute limit is the main issue. You can see it here. With 2 T4 the score people get is around 10%, when we
> got almost 30% with 4 L4. We spent time optimizing inference to produce as many candidates as possible given
> the time and compute limits."

⚠ **This is the single most decision-relevant claim in the corpus and it is about the 2025 environment.** In 2026
the competition was upgraded to L4×4 (see §5), so the constraint that produced that 10%→30% spread no longer applies.

### 2.2 The actual NVARC/ARChitects architecture (from the repo + live notebook, not from forum prose)

From `1ytic/NVARC` README, the 2025 solution has three components:
1. multi-stage **synthetic data generation pipeline** (LLM-generated puzzle logic → executed grids);
2. an improved **ARChitects** (2024 winner) solution — Qwen3-4B fine-tune with a **16-token grid vocabulary**;
3. an improved **Tiny Recursive Model (TRM)**.

SFT recipe, verified from repo files:
- Base: `Qwen3-4B-Thinking-2507-16t` (16 embeddings; grid digits 0–9 + 5 control tokens) — `ARChitects/run_sft_4b.sh`.
- Checkpoint name `qwen3_4b_grids15_sft139`; a 2B sibling `qwen3_2b_grids15_sft141` also exists [735058 msgid 3516747].
- **lr = 1e-4**, min_lr 1e-7, warmup_init 1e-6, warmup 200 iters, linear decay over 12,716 iters,
  **global batch 256**, micro-batch 1, clip_grad 0.5, 1 epoch, seed 24, 4 nodes × 8 GPUs,
  tensor-parallel 8, sequence packing 256,000 mb-tokens.
- Data mixture (`sft_mg.yaml`): `arc2_training, mini, concept, rearc, nvarc_training, nvarc_full`;
  validation on `arc2_evaluation6`. Augmented corpora published: **103k synthetic puzzles** and
  **3.2M augmented puzzles** [NVARC README + 685142 msgid 3436730].

### 2.3 TOPAS / Bitterbot ("Victor")

The only system in the corpus with a *named architecture + local score + LB score + a diagnosed cause of the gap*.

- **Architecture:** "architectural blend of HRM's hierarchical structure (but instead of having the prefrontal
  cortex design with slow and fast modules, they have a logic and canvas core) and TRM's single latent space"
  [madarshbb, 729743 OP, 2026-07-26].
- **Params / hardware:** **~100M params**, trained on a **single consumer RTX 4090**, 14 days into training at
  time of submission [697859 OP, 2026-05-07]. Prior year: 24M-param model trained ~4 weeks on a single Colab A100
  [697859 msgid 3455894, 2026-05-10].
- **Local:** ~36% on the public eval split [697859 OP]. Later **44% on the public 120 puzzles** [695556 msgid
  3457610, 2026-05-14].
- **LB:** 11.67% (title), then 19.03% (see gotcha below), then ~16.5% [697859 msgid 3456249, 2026-05-11].
- **The stated cause of the gap — and it is a scheduling failure, not a modelling failure:**
  > "Because TOPAS utilizes deep recursion, our TTT becomes computationally heavy. During our initial sandbox
  > tests, we were struggling to stay within the strict time limits and kept timing out. While tinkering to
  > ensure the submission would actually complete, we overcompensated on our time-limit thresholds. This
  > resulted in the model bailing out early to save time, ultimately failing to output a prediction on a large
  > chunk of the puzzles. Moving forward ... we need to fine-tune our time-management so we convert [[0]]
  > placeholders into real beam outputs." [697859 OP, 2026-05-07]
- Corroboration: *"It's clear from the logs that the algo bailed on 56% of the second attempts"* [697859 msgid
  3456280, 2026-05-11]. And an independent observer notes the model "output the correct answer multiple times in
  the beam pool only to have the scorer algo confidently pick the wrong one" [697859 msgid 3455165, 2026-05-08].
- The published claim about TOPAS's convergence — "after 50,000 epochs, the base 24M model converges with a score
  of 24% (on local, not on LB)" [madarshbb, 729743 OP, 2026-07-26] — is explicitly labelled by its reporter as
  *"The claim is..."*, and madarshbb abandoned it: *"the model had severe issues with debugging, the training was
  extremely unstable and loss frequently became nan"* and *"I didnt sub ToPaS. I dont have the resources to train it."*
  → **[CLAIMED / not reproduced by the reporter]**. Treat TOPAS convergence claims as unverified.

### 2.4 TRM (Tiny Recursive Model)

- **Verified architecture (repo):** `arch=trm`, `L_layers=2`; pretraining `H_cycles=3, L_cycles=4`;
  test-time eval/finetune `H_cycles=4, L_cycles=4, halt_max_steps=10`, `freeze_weights=False` [TRM/README.md].
- **Pretraining:** 4,073 puzzles (competition + ~3k SDG-generated), **256 augmentations → 1,041,207 puzzles**,
  lr 3e-4, global batch 3,072, 10,000 epochs, 8×H100 [TRM/README.md].
- **Test-time:** eval dataset built with **128 augmentations per puzzle "to match what we use in Kaggle notebooks"**;
  run `--nproc-per-node 4`, `global_batch_size=128`, **lr=1e-4**, `lr_warmup_steps=200`, `epochs=4000`,
  `eval_interval=4000`, checkpoint `step_220708` [TRM/README.md and live `cpmpml/arc2-trm-v31` notebook].
- **Official measured results** (TRM/README.md, from `eval-arc-k-10.py`): `ARC/pass@1` 0.0764, `pass@2` 0.1014,
  `pass@5` 0.1014, `pass@10` 0.1014, `pass@100` 0.1375, `pass@1000` 0.1375. **"When submitted this scored 10.0 on
  Kaggle public leaderboard."**
- Independent corroboration: *"With TRM I got about 22% on the public evaluation, and only 10.83% when
  submitting."* [697859 msgid 3456171, 2026-05-11] — the reporter's own 22%→10.83% is **[MEASURED, self-reported]**;
  their diagnosis ("maybe that their model overfits") is **[SPECULATION]**.
- **Decision-relevant limitation on TRM ensembling**, from madarshbb [729743 OP, 2026-07-26]: adding TRM
  candidates to the NVARC ranker pool "didnt work meaningfully well"; using TRM as a cross-reference filter also
  failed, and the stated reason is blunt — **"TRM scores 10 on LB, its not good enough for this strategy."**

### 2.5 nvbanana (2nd, 72.08)

Almost nothing is disclosed. What exists:
- They will only expand the team if (a) they risk not winning or (b) you raise their score; "A score of 30 is
  likely to be using our last year code, hence does not meet condition 2" [694676 msgid 3491533, 2026-07-06].
- "Different weights." [729509 msgid 3503572, 2026-07-26] — in answer to "What led to their breakthrough?",
  where the poll options were "same NVARC recipe, different model" / "different recipe and different model" /
  "something else".
- **First-party confirmation that the name is a Nano-Banana reference** (nothing more): *"It is true that the name
  nvbanana was inspired by nanobana. I won't comment on the rest of the post"* [724647 msgid 3509467, 2026-08-06].
  The surrounding speculation in that post (that they use image generation, that they're NVIDIA) is
  **[SPECULATION by a third party]**, and this is the only part of it the team confirmed.
- Their jump: 44% (2026-07-30) → 70.42% (2026-08-23) → 72.08 [724647, msgids 3505590 / 3516212 / live LB].

### 2.6 rabbithole (1st, 76.94)

**No architectural information whatsoever appears in the corpus.** The only substantive data points are:
- cpmpml: "2nd team has clearly moved beyond what could be done with knob fitting on our last year solution"
  [724647 OP, 2026-07-12].
- The **runtime forensics I ran myself** (§4.2) — rabbithole's 79 tracked submissions are overwhelmingly
  clustered in a narrow ~713–744 s band. That is a real, measurable behavioural fingerprint, and it is the best
  available evidence about what they do.

### 2.7 Tufa Labs (4th, 48.61)

23 tracked submissions, 21 of them in **705–746 s**. First submissions scored **0** (2026-08-22, 5 s; 2026-08-23,
738 s), then 32.5, 32.92 … 48.61. No architectural disclosure in the corpus. Their runtime signature is
*indistinguishable* from rabbithole's — which is informative in itself (§4.2).

### 2.8 Yi-Chia Chen (3rd, 48.89) and Junhua Yang (5th, 37.22)

No architectural disclosure in the corpus. Runtime medians ≈ 11.4 min and ≈ 11.9 min respectively (§4.2).
Yi-Chia Chen's score progression is steep at the end: 30.97 → 39.03 (09-06) → 39.58 → 40.69 → 48.89 (09-09).

### 2.9 nex-N2-mini vs Aquila-mini (topic 732723 — a genuine controlled comparison)

[dominic789654, 732723 OP, 2026-08-04]. Both are `Qwen3_5MoeForConditionalGeneration` (35B total, 256 experts,
~3B active, 2 KV heads, hybrid linear/full attention). Same engine (vLLM 0.26, fp8_per_tensor, Triton MoE),
same harness, same hardware (4×L4). 64 grids each, evaluation split:

| Metric | Nex-N2-mini | Aquila-mini |
|---|---|---|
| Solved | **0/64** | **0/64** |
| Wrote an answer | 96.9% | 93.8% |
| Grid correct size | 65.6% | 18.8% |
| Truncated | 3 | 4 |
| Tokens/sec | 438 | 452 |

- Control that proves the harness works: on the **training** split, Nex W4A16 scores **8/64 (12.5%)**.
- **"Reasoning effort does nothing"** — same output at low/medium/high effort for this architecture.
- **W4A16 int4 (compressed-tensors, experts only, 122k of 124k tensors): 22.8 GiB vs 65.4 GiB bf16, same score,
  +20% throughput.**
- They explicitly never got to TTT: *"The obvious next step is test-time LoRA training (NVARC-style, r=16 per
  puzzle)... I just ran out of Kaggle's 30-hour weekly GPU quota."*

### 2.10 Program synthesis (topic 733501) — see §7 and §9 for why this is weaker than it looks

[dominic789654, 733501 OP, 2026-08-07]: GPT-5.6-Sol + program synthesis **2/12 (16.7%)** vs plain reasoning
**0/12**; DeepSeek-V4-Flash synthesis 0/12. All at `reasoning_effort="none"`. The author is careful and honest
about this being a **lower bound** (msgid 3510025).

### 2.11 EGO-ARC, Haumea, "TruthGuard", Bicameral/TOPAS-presentation (novelty section, §7)

---

## 3. Test-time training (TTT) — the complete evidence, with the real numbers

### 3.1 The authoritative NVARC TTT configuration

**Not from the forum** — from the live source of `sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue`
(currentVersionNumber 10, `machineShapeNullable: NvidiaL4`, `enableInternetNullable: false`), which the corpus
points at twice [691081 msgid 3465577; 685142/691081]. Extracted verbatim:

```python
global_end_time = time.time() + 12 * 3600 - 600      # 12 h wall minus a 600 s finalisation reserve

peft_params = dict(
    r=256,
    target_modules=["q_proj","k_proj","v_proj","o_proj",
                    "gate_proj","up_proj","down_proj",
                    "embed_tokens","lm_head"],        # LoRA on embeddings AND lm_head, not just attention
    lora_alpha=32, lora_dropout=0.0, bias="none",
    use_gradient_checkpointing=False,
    random_state=42, use_rslora=True, loftq_config=None,
)
train_args = dict(
    per_device_eval_batch_size=1, per_device_train_batch_size=1,
    gradient_accumulation_steps=1, num_train_epochs=1,
    warmup_steps=0, warmup_ratio=0.1, max_grad_norm=1.0,
    learning_rate=5e-5, optim="adamw_torch", weight_decay=0.0,
    lr_scheduler_type="cosine", seed=42, bf16=True, ...
)
max_seq_length = 8192
max_score      = -np.log(0.2)                        # ≈ 1.609, the DFS beam-pruning NLL threshold
```

Per-puzzle TTT loop:
- **Reset LoRA to the pretrained default state at the start of every puzzle** (`set_peft_model_state_dict(...,
  default_weights.copy())`) — i.e. adapters are per-task, not accumulated.
- Training set = **`puzzle_ds.augment(n=16, shfl_keys=True, seed=1)`**, then `cut_to_len(max_len=8192)`.
- Trained with `UnslothFixedTrainer` / `UnslothTrainingArguments`; `for_training` → `train()` → `for_inference`.
- `gc.collect(); torch.cuda.empty_cache()` between puzzles.

Decoding (this is where the TTT budget actually gets spent):
- Eval set = `puzzle_ds_multi.augment(n=2, seed=2)` over the D4 group → 16 subkeys per puzzle, re-batched in
  groups of **4** (`offsets [0,4]`, `[2,6]`, `[8,12]`, `[10,14]`).
- `turbo_dfs` is a **best-first / beam DFS over the 16-token vocabulary** with an explicit time guard.

### 3.2 The three nested time limits — and the exact code that enforces them

This is the concrete answer to "how do they fit TTT in the budget":

| Scope | Constant | Enforcement point |
|---|---|---|
| Whole run | `12*3600 - 600` = **42,600 s ≈ 11.83 h** | `starter.py --end-time`; checked in the worker loop (`if time.time() > end_time: break`) and inside `turbo_dfs` |
| Single DFS decode | `while time.time() - start_time < 540` | inside `turbo_dfs`, i.e. **540 s per decode call** |
| Single puzzle | `if spend_time > 1200 or time.time() > end_time` | worker loop, checked *between* decode batches; prints `timeout after {spend_time:.1f}s for puzzle {key}` |

**What happens when a puzzle times out — this is the critical detail:**
```python
if spend_time > 1200 or time.time() > end_time:
    print(f"[Rank {rank}] timeout after {spend_time:.1f}s for puzzle {key}")
    break        # breaks the batch loop, falls through, writes NOTHING for this puzzle
```
It **breaks out of the decode loop and writes no output file for that puzzle**. The submission is then built from
`ArcDataset.get_submission()`, whose default is:

```python
submission = {k: [{f'attempt_{i+1}': [[0]] for i in range(2)} for _ in ...]}
```

So a timed-out puzzle silently degrades to the **`[[0]]` placeholder** — exactly the failure Bitterbot described
in §2.3. TTT timing out is therefore **not** an exception, not a retry, and not logged as an error in the
submission: it is a silent score-zeroing path that you only detect in the notebook log line
`[Rank N] timeout after ...s for puzzle <key>`.

Also note: **`max_score = -np.log(0.2)`** prunes the DFS. If TTT has not converged, the model's NLL on every
continuation exceeds this threshold, `num_alive_beams` hits 0, and the loop breaks with an empty beam set — a
*second*, quieter way to get no prediction. The author of the corpus's most detailed TTT post reports exactly
this class of variability (§3.4).

### 3.3 How the work is distributed — NVARC uses a dynamic queue, not static sharding

```python
queue = mp.Manager().Queue()
for key in sorted(data.keys()):  queue.put(key)
for _ in range(4):               queue.put(None)      # sentinels
mp.spawn(local_worker, args=(queue, args.end_time), nprocs=4)
```

`nprocs=4` with `CUDA_VISIBLE_DEVICES = rank` = one worker per L4. Because every worker pulls the next puzzle
from a **shared queue**, this is **dynamic work-stealing by construction**: a rank that draws easy puzzles simply
consumes more of the queue. There is **no static partitioning**, and the one poster in the corpus who *did* use
static partitioning lost a whole run to it (§4.3). Puzzle order is `sorted(data.keys())` — alphabetical, not
difficulty-sorted.

### 3.4 Independent TTT findings from someone who rebuilt the pipeline (madarshbb, topic 729743)

This is the richest *narrative* TTT evidence in the corpus [729743 OP, 2026-07-26]. It is
**[MEASURED by the poster, on their own CV]** and the poster is candid about where LB disagreed with CV:

1. **Batch-invariance was necessary to make experiments meaningful at all.** They used
   `github.com/thinking-machines-lab/batch_invariant_ops` to build a batch-invariant version of the NVARC
   notebook, "since their notebook is stochastic". Their stated purpose: *"This will help you understand whether
   your move in CV is due to experiments or variance due to LLM stochasticity."*
2. **Adapter caching:** the notebook "allows you to cache the per-puzzle adapters and use the same ones in the
   future runs. Because note while batch-invariant ops is deterministic, ttft is not. This cached adapters +
   batch invariant ops is what makes the entire pipeline deterministic. But as @cpmpml mentioned in the last
   writeup, **this is slow**."
3. **Early stopping during TTT** ("TTFT") on loss convergence: *"I have tried early stopping during TTFT if the
   puzzle loss approaches convergence quickly, which seemed to be the case for many puzzles. Seemed to help in CV,
   but its hard to tell in LB because ultimately, during submission, we still use the batched version as its
   faster and solves more puzzles."* → **Note this carefully: the batched version is faster AND solves more
   puzzles. Early exit helped CV but did not help LB.**
4. **NLL threshold tuning did nothing:** *"We played around a lot with the nll threshold, changed them, made them
   dynamic, but there was no statistically significant move in score."*
5. **The speed win that did work:** *"we load-balanced the puzzle queue such that largest puzzles are solved
   first. Again improved speed."* — i.e. **longest-processing-time-first scheduling** (LPT), the classic
   makespan heuristic.
6. **They ran out of things to do with surplus time:** *"we reached a point where we had a lot of inference time
   remaining, but offloading this time to the core NVARC solver did not help at all. Per-puzzle loss function
   plateaued so no point in training longer."* Plus: alternative selection strategies, and "a second decoding
   round with different seeds now that we had time, but it didnt move score."
   → **This is a hard ceiling result: more TTT compute stops paying.**
7. **TRM ensembling failed** for two different strategies (§2.4), and the models they'd want to ensemble
   ("mindsai and the architect's LLaDa solution") "arent fast enough to ensemble with the current approach".
8. Their conclusion: *"It does seem like the only significant way to boost score is to create more synthetic data
   and further train the nvarc qwen 3 model."*

### 3.5 The only fully-specified, reproducible TTT timing measurements in the corpus

[696718 msgid 3453039, Boladi, 2026-05-04] — a non-NVIDIA (Intel Arc A770 16 GB) but numerically explicit run:

- Config: **Qwen3-4B + per-task LoRA TTT `n_steps=8, augs_per_step=2` + AIRV `n_augs=4` +
  `chkpt_threshold=1500` + `MAX_TTT_TOKENS=3000` skip-TTT cap.**
- **38–43 min for 6 tasks → ~6.5–7 min/task average.**
- Extrapolated to the full set: **"~12-15 hours for the full 120 eval set."** — i.e. it does **not** fit.
- Per-task breakdown:
  - short prompts (~600 tok): TTT **30–60 s** + inference ~6 min
  - medium (~1500–2200 tok): TTT **2–4 min** + inference 2–10 min
  - **long prompts (>3000 tok): TTT SKIPPED, base AIRV only, 3–7 min**
- **The skip rule:** a hard `MAX_TTT_TOKENS=3000` cap. Prompts above it get **no TTT at all** and fall back to
  plain inference. This is a *deliberate, policy-level* TTT skip, and it is the clearest example in the corpus of
  "what to do when TTT does not fit": **skip TTT for long prompts rather than time out.**
- Two measured memory laws on that hardware:
  - training VRAM **~1.61 MB/prompt-token** for Qwen3-4B fp16 → `max_prompt_tokens = (VRAM_free_MB − margin)/1.61`
  - inference KV cache **~0.144 MB/token**
  - Consequence, stated as a design rule: *"separate MAX_TTT_TOKENS (training-step, logit-bound) from
    MAX_INFERENCE_TOKENS (KV-cache-bound) in your skip-too-long-prompt logic; using one threshold for both either
    wastes inference headroom or OOMs at training time."*
- Operational: `model.eval()` after training + `gc.collect()` + `torch.xpu.empty_cache()` between tasks —
  "Memory creep accumulates without it." (Same pattern as the NVARC notebook's `empty_cache()`.)
- **WDDM spill (a detection technique worth stealing):** on Arc/Windows, allocations above 16 GiB silently spill
  to shared system RAM instead of OOMing, costing **5–10×** wallclock over PCIe. Detection: *"watch for
  `vram_peak > total_memory_mb` — if you see that, you've spilled"*, corroborated by the 1 Hz
  `\GPU Adapter Memory(*)\Shared Usage` perf counter. Their T4-calibrated `chkpt_threshold=1500` projected
  **23 hours** before they found this; dropping it to ~800–1000 fixed it.

### 3.6 TTT in the TRM lineage — a different meaning of "TTT"

The TRM notebook does not use LoRA at all; its "test-time training" is **continued full-weight training**
(`freeze_weights=False`) over **128 pre-augmented copies of each puzzle**, at **lr 1e-4**, batch 128, 4 GPUs,
scored at `eval_interval=4000` [TRM/README.md; live `cpmpml/arc2-trm-v31`]. TRM's own README says the eval-time
`H_cycles`/`halt_max_steps` were *changed* from pretraining "because they yield better results during the test
time fine tuning step" — i.e. the recursion depth is a TTT-time hyperparameter, not just a training one.

### 3.7 The single most important TTT fact for your decision

**Every team at the top of the live leaderboard finishes a submission in ~10–12 minutes, not 12 hours.**
See §4.2. Whatever TTT they are doing, it is not consuming anything like the nominal budget. The NVARC-style
"TTT on 120 puzzles with a 12-hour wall" configuration is the *2025 baseline*, and the top of the 2026 board
does not look like it.

---

## 4. Compute-budget and scheduling techniques

### 4.1 The budget rules, as stated in-corpus

- **12-hour submission wall** — asserted repeatedly: "The time constraint comes from the 12h evaluation on
  Kaggle" [690497 msgid 3441510, 2026-04-14]; "the 12 hour submission run time" [685621 OP, 2026-03-28];
  "the model already runs for 12 h in submission" [691081 msgid 3441442]; "don't blow the 12-hour Kaggle wall"
  [732723 OP]. The official 2026 overview page does **not** publish a runtime limit to me; these are participant
  reports. NVARC's code budgets against `12*3600`. Treat 12 h as the operative assumption.
- **Weekly GPU quota = 30 h** [694745 OP, 2026-04-26], and crucially: **"gpu use when submitting is not subject
  to user quota. The quota is for testing code."** [gregkamradt, 694745 msgid 3449170, 2026-04-27]. So submission
  runs are *free* of quota, but your local experiments are not.
- **Quota is charged per accelerator, not per session:** "when I run a notebook using the T4 x4 GPU for, say,
  10 minutes, my GPU quota shows 20 minutes consumed" [696364 OP, 2026-05-02] — this is because x4 counts as
  2× the accelerator. Practical reading: **4×L4 burns quota at 2× the wallclock rate.**
- **1 submission per day** [704079, 694745]. Requests to raise it were declined/not actioned
  [694745 msgid 3449141; 704079 msgid 3465987].
- **Prizes/quota:** 30 h/week is "more than used up in a single 12-hour test of a solution (12 hours x 4 = 48
  hours)" [694745 OP]. An unanswered request asks for the full 30 h on L4×4 and for H100s at 2.5× quota rate
  [689054 msgid 3518958, 2026-08-31; 689054 msgid 3450251, 2026-04-29].

### 4.2 Runtime forensics — what the leaders actually do (my own measurement)

From the live per-submission runtime history behind topic 709522:

| Team | Submissions | Runtime range | Distribution |
|---|---|---|---|
| **rabbithole** | 79 | 13 s – 1708 s | **70 of 79 fall in 713–744 s** (median ≈ 12.1 min) |
| **nvbanana** | 74 | 13 s – 1479 s | 24 in 600–736 s; median ≈ 9.4 min; wider spread |
| **Tufa Labs** | 23 | 5 s – 746 s | **21 of 23 in 705–746 s** |
| **Junhua Yang** | 105 | 2.5 – 52.8 min | median ≈ 11.9 min |
| **Yi-Chia Chen** | 9 | 84 s – 65,635 s | median ≈ 11.4 min; one 18.2 h outlier |

**Reading this honestly:**
- The 713–744 s band shared by rabbithole and Tufa Labs is a **~30-second-wide window across 70 and 21
  independent runs respectively**. That is not a natural completion time; it is a **budget being hit**. Two
  unrelated teams converging on the same ~730 s wall strongly suggests a **shared, self-imposed per-run time
  budget of roughly 12 minutes** (or a natural per-puzzle budget × 120 that lands there).
- **This is inference from timing data, not a disclosed fact.** No corpus post states a 12-minute budget.
  I am labelling it **[MEASURED pattern, INFERRED cause]**.
- The practical implication is large and cuts against the "fill the 12 hours" instinct: **the highest-scoring
  system on the board is spending ~1.7% of the available wall-clock.** Whatever it does, it does *per-puzzle and
  cheaply*, and it is not grinding 120 puzzles through per-task LoRA TTT for 6 minutes each.
- **Sub-60 s runs are a real failure mode**, and they occur even in top-10 teams: rabbithole 3, nvbanana 1,
  Tufa Labs 1, Junhua Yang 0 but 3 sub-60 s, Team BlackBox 9, Seok Jeongeum 20. A ~5–13 s run is a crash or a
  trivial early exit that still consumes the daily submission.

### 4.3 The static-partitioning disaster — the corpus's clearest scheduling lesson

[697859 msgid 3455909, 2026-05-10] — same thread, likely Bitterbot (the OP):
> "I am feeling pretty stupid for using static partitioning, I thought I had enough headroom to get away with it.
> Instead I watched 12 hours of compute evaporate because one GPU drew the short straw and got a bunch of hard
> puzzles while the other three finished in under 11 hours and sat there twiddling their silicon thumbs."

Numbers: 4 GPUs; 3 finished in **<11 h**; 1 GPU's shard contained the hard puzzles; the run produced a **dead
result**. This is the canonical argument for **dynamic work-stealing over static sharding** — which is exactly
what NVARC's `mp.Manager().Queue()` does (§3.3), and what madarshbb's largest-first queue ordering refines
(§3.4 item 5).

### 4.4 Other concrete scheduling techniques observed

1. **Global deadline propagated into every worker** (`--end-time` passed to `mp.spawn`), checked at two
   granularities (worker loop, inner DFS). Never rely on the platform to kill you gracefully.
2. **Reserve a finalisation margin:** NVARC subtracts **600 s** from the wall (`12*3600 - 600`) so that
   submission assembly and `json.dump` still happen.
3. **Nested per-unit timeouts:** 540 s per decode, 1200 s per puzzle, 11.83 h total. Three independent
   guards with different scopes.
4. **Largest-puzzle-first queue ordering** [729743 OP] — LPT scheduling to minimise makespan.
5. **Skip policy over timeout policy:** hard token cap (`MAX_TTT_TOKENS=3000`) → skip TTT, fall back to base
   inference [696718]. Better to submit a base-model answer than a placeholder.
6. **`[[0]]` avoidance:** the placeholder is the *default* in `get_submission()`; the only robust defence is to
   ensure every puzzle writes a real (even if weak) beam output. Bitterbot's diagnosis of their own 11.67% is
   precisely "convert `[[0]]` placeholders into real beam outputs" [697859 OP]. Note the asymmetry: a placeholder
   is **guaranteed 0** for that task, whereas a low-confidence guess has positive expected value under the
   "either of 2 attempts matches" metric.
7. **Daily-limit economics:** with 1 submission/day, every failed experiment costs a day. This is why
   several posters run **"Save & Run All" as a dry run and only then submit**
   — "I always save first, I don't submit directly. It catches this mistake" [697859 msgid 3454844, 2026-05-08].
8. **Log interpretation trap:** "you get that the logs are not the logs of the submission, right? They are the
   logs of the notebook running on the evaluation data." [697859 msgid 3456774, 2026-05-12] — you cannot debug a
   submission from its logs the way you debug a local run; the rerun logs are what you get.

### 4.5 What people did with surplus time — and why it failed

Directly from madarshbb [729743 OP]: with inference time left over they tried (a) longer TTT — "Per-puzzle loss
function plateaued so no point in training longer"; (b) alternative selection strategies beyond `score_kgmon` —
"while some helped in CV, it couldnt translate to LB"; (c) a second decoding round with different seeds —
"it didnt move score"; (d) TRM ensembling, two ways — both failed. This is a **strong negative result about the
returns to extra compute at the NVARC-Qwen operating point.**

---

## 5. What hardware people actually get and use

| Fact | Detail | Provenance |
|---|---|---|
| Launch hardware | Started on **P100 / T4×2** | 694745 msgid 3448863, 2026-04-26 |
| Upgrade | **2026-04-07: L4×4, 96 GB total GPU memory**, "enabling submissions with much larger models" | 689054 OP (María), 2026-04-07 |
| Official capacity note | "any of our advanced accelerators are heavily supply constrained ... we can face variable supply constraints" | 689054 msgid 3437555, 2026-04-07 |
| H100 | **Not for ARC-AGI-2.** "Our plan is to implement the H100s for the ARC-AGI-3 competition." | 689054 msgid 3437555; confirmed 689054 msgid 3437561 |
| 2025→2026 continuity | 2025 was 4×L4; when the 2026 comp launched on 2×T4 the 2025 solution *could not be resubmitted*; the L4×4 upgrade restored parity | 685142 OP + msgs 3429713/3431691/3438315 |
| Effect on the winning recipe | "this year environment is exactly the same as last year environment. It is why people could easily submit our code and build on it." | 691081 msgid 3445969, 2026-04-20 |
| Quota accounting | T4×4 charges 2× wallclock to quota | 696364 OP |
| Quota exhaustion | "I just ran out of Kaggle's 30-hour weekly GPU quota" — TTT never attempted | 732723 OP, 2026-08-04 |
| L4 selection gotcha | The L4 accelerator only appears once the competition dataset is attached and the notebook is created from the competition's Code panel; internet must be off | 735044 msgs 3512641/3512604 |
| Upgrade timing | "when will we get gpu L4x4" (2026-04-02) → "now. It just happened." (2026-04-07) | 687095 |
| Raw throughput reference | 438–452 tok/s for a 35B MoE at fp8 on 4×L4 (vLLM) | 732723 OP |
| Memory envelope | 4×L4 = 96 GB, so a 35B bf16 (65.4 GiB) *just* fits; W4A16 (22.8 GiB) is the comfortable option | 732723 OP |
| Compute partner | "we have connected many people with our partners in the past"; a 2026 participant asks how to apply and reports no reply | 694745 msgid 3449141; 689054 msgid 3462714, 2026-05-24 |
| Non-NVIDIA | Intel Arc A770 16 GB works for Qwen3-4B + LoRA TTT via torch 2.11+xpu, but Unsloth, FlashAttention-2 and bitsandbytes are all CUDA-only → **no QLoRA on Arc** | 696718 OP |

**Design constraint this imposes:** you get 4×L4 (96 GB) with no guaranteed availability and a 30 h/week *testing*
quota; submission runs are quota-exempt. That asymmetry means **you should be willing to submit long runs you
cannot afford to test locally** — and it makes local, cheap proxies for TTT behaviour very valuable.

---

## 6. Data and validation methodology

### 6.1 The placeholder discovery (multiple independent confirmations)

- **The public test file is a placeholder — it is training data.** [724073 msgid 3494909, 2026-07-11]:
  *"The publicly accessible test challenges file is only a placeholder. Scoring is done on different and hidden
  tasks."*
- A participant proves it concretely: the first "competition task" in the public test file is identical to
  training task **13713586** (2×2 → 8×8 tiling); the correct answer produces **no score** [724073 OP, 2026-07-09].
- **The winner says the same thing, earlier and more bluntly** [694698 OP, 2026-04-26]: ARC-AGI-1 eval scores of
  >95% are inflated because *"Most of ARC AGI1 evaluation is included in ARC AGI2 training data"* — and the
  corollary he draws is that *"All frontier models use ARC AGI2 training data, hence their scores on ARC AGI1
  are meaningless."*
- [684857 msgid 3431634, 2026-03-30]: *"Part of arc agi1 eval and test is included in arc agi2 train."*

→ **Anyone scoring themselves on the public test file is measuring training data.** The *evaluation* split
(120 tasks, with solutions) is the only honest local proxy. This is stated independently three times in the corpus
and is consistent with the "public test file is training data" note in this repo's own brief.

### 6.2 The public-eval → leaderboard drop: every number in the corpus

| Source | Local/eval | LB | Drop | Status |
|---|---|---|---|---|
| 697859 msgid 3456171 (TRM, 2026-05-11) | **22%** public eval | **10.83%** | ~51% | [MEASURED, self-reported] — diagnosis "maybe overfits" = [SPECULATION] |
| 697859 msgid 3456171 (same, about others) | — | — | *"Every top team last year reported a significant drop from public evaluation to public leaderboard. **A 50% drop is not uncommon.**"* | [CLAIMED, unsourced generalisation] |
| 697859 msgid 3458771 (cpmpml, 2026-05-16) | **30%** eval | **27%** public LB | 10% | [MEASURED] — **the Qwen-based pipeline's drop is far smaller than TRM's** |
| 697859 msgid 3455218 (cpmpml, 2026-05-08) | 30.5% pass@2 / ~40% pass@10 | 27% public / 24% private | — | [MEASURED] |
| 732723 OP (2026-08-04) | training split 12.5% (8/64) | eval split **0/64** | absolute | [MEASURED] |
| 684857 msgid 3431636 | 27% public / 24% private (2025) | post-deadline 29% | — | [MEASURED] |
| 694676 msgid 3514745 | "reload score of 35.33333" | 28.09 public | 21% | [CLAIMED, anonymous poster, unverifiable] |
| 685142 msgs 3440071/3440407/3442399 | same notebook, repeated runs | 28.47 every time for one forker, "30+" for others; "Same code submitted twice after deadline yield two different scores" | — | [MEASURED] — **the drop is partly nondeterminism** |

**Interpretation, stated with appropriate caution:**
- The "50% drop" figure is a **generalisation from a post whose own measured number is 22→10.83**, and the *same
  thread* contains a counter-example from the 2025 winner of only **30→27**. The corpus does **not** support a
  universal 50% drop. What it supports is that **drops are real, variable by method, and larger for TRM than for
  the Qwen pipeline.**
- **Nondeterminism is a first-class confound.** cpmpml: *"Test data is the same as last year... The reason scores
  change is that people modify a bit the code. **Also, our code is not deterministic.**"* [685142 msgid 3441234,
  2026-04-13]. And *"we saw that last year. Same code submitted twice after deadline yield two different scores"*
  [685142 msgid 3443042, 2026-04-16]. One forker got 28.47 three times in a row while others got 30+ with the same
  notebook [685142 msgid 3440071].
- → **Do not tune hyperparameters on a single LB submission.** This is exactly why madarshbb built the
  batch-invariant variant (§3.4 item 1).

### 6.3 Structural analysis of the eval/train gap (two solo posts, no replies)

[734018 OP, 2026-08-09] — structural pass over all 1,120 tasks (1,000 train + 120 eval), computed *without
solving anything*:
- 68.0% of training tasks have an identity shape rule; 18.6% are crops → ~86% have a rule-derivable output shape.
- 69.6% never introduce a colour not already in the input.
- Only **0.9%** are "pure recolouring" — and **0.0% of evaluation** → not worth a special case.
- Train vs eval: **median max grid area 144 vs 400 (~2.8×)**; median distinct colours **5 vs 7**;
  input has symmetry **21.3% vs 9.2%**; shape rule inconsistent across a task's own demos **4.9% vs 11.7%**.
- Riposte to the obvious objection: permutation test on the median-area gap, **20,000 resamples, p < 0.0001**,
  and excluding every 30×30-capped grid the medians are still **142 vs 400**.
- **An actual measured consequence:** a dumb solver (10 primitives composed to depth 2 + fitted colour map,
  requiring consistency across every demo pair) scores **training 20/1000 = 2.0%, evaluation 0/120 = 0.0%**.
- [734068 OP, 2026-08-09] extends this to ARC-AGI-1 (400/400): v1's "shape rule inconsistent" share is 5.0%
  train vs 4.8% eval (essentially identical) whereas v2 is 4.9% vs 11.7%. **v1's eval was a bigger version of its
  train set; v2's is structurally different.** The author flags the comparability caveat honestly (1000/120 vs
  400/400, so the v2 eval estimates are noisy) and says only the area row was permutation-tested.
- Practical upshot, quoted: *"if you calibrate search budgets, grid-size assumptions or symmetry priors on the
  training split, you're calibrating against a distribution whose typical task is ~2.8x smaller in area, has two
  fewer colours, and is more than twice as likely to hand you an exploitable symmetry. **Local validation will
  flatter you.**"*

### 6.4 The scoring metric and its denominator

- **Official metric** (arcprize.org): predict exactly **2 outputs per test input**; if *either* matches
  ground truth exactly you score 1 for that task; final score = mean over all task test outputs.
- **The denominators are 120 and 120:** "Public leaderboard is based on 120 semi private tasks, the private
  leaderboard is 120 private tasks" [699858 OP, 2026-05-15], and the *code* is run against **240 hidden puzzles**
  [cpmpml, 740422 msgid 3524002, 2026-09-13]. At the end you select two submissions; their stored hidden scores
  are retrieved — **nothing is rerun**.
- A long thread tries to reverse-engineer the "true number of correct predictions" from the displayed score
  [699858 msgs 3458826/3458857/3459375] and concludes scores land on `N/240`-like fractions with
  partial credit per puzzle (`k/N` of a puzzle's test inputs). **This analysis is the posters' own arithmetic and
  is internally contested in-thread** (one poster asserts "51 answers yield 42.5%", the other "50 correct answers
  would yield 41.67%"). Treat the fractions as **[SPECULATION]**; the 120/120 split is **[CONFIRMED]** by the host.
- **The 2-attempt metric materially changes strategy:** with 2 guesses/task, a *diverse* guess has positive
  expected value where a `[[0]]` placeholder has exactly zero. This is why the "convert placeholders to beam
  outputs" fix in §2.3 is expected to be worth a lot.

### 6.5 Data quality — the training set is not perfect

- **Task `17829a00`, training example 1 is wrong.** "Looks like ground truth is wrong in the first training
  sample." [698462 msgid 3455687]; agreed by cpmpml [698462 msgid 3459521: *"An obvious error in train #1...
  Occasional errors are to be expected."*]. Detailed: *"Every column except column 15 conserves its 8-count
  exactly. Column 15 gains two cells out of nowhere. **With augmented training data this error is compounded.**"*
  [698462 msgid 3458044, 2026-05-14]. The thread's conclusion: solvers that filter candidates by "must reproduce
  all demo pairs" are **actively harmed by this** [698462 msgid 3461312].
- **Task `963c33f8` also appears to contain errors** [731040 OP, 2026-07-30: two missing grey cells in train
  example 2; a maroon block misplaced in train 3].
- **Task `16b78196`** was misidentified by one poster and retracted [697058].
- **Augmentation can *amplify* tainted examples** (see above) — an argument for validating your synthetic
  pipeline's consistency.
- **Synthetic data generation, by the winner's own recipe** [693607 msgid 3446928, cpmpml, 2026-04-22]:
  rotations/flips/colour perms have "been used for years already, both at training times and at inference time",
  but *"arc-agi-2 has more complex puzzles and these hard puzzles should be augmented in a different manner than
  just rotation/colour permutation"* — and the winner already trained on "a huge dataset of synthetic data".
  A separate proposal in the same thread lists concrete label-preserving augmentations
  [693607 msgid 3446489]: pad edges with random values, split a colour into two, shuffle input colours, trim
  background edges, rotate/flip, swap grid regions with separator lines, upscale with/without separators —
  with the constraint **"Make sure the grid stays smaller than 30x30."**

---

## 7. Explicit gotchas and failure modes that cost submissions

Ranked by how much they cost people. Every one is someone's real loss.

1. **Accelerator silently off → daily submission burned.** *"Tweaked the config, patiently waited for the daily
   submission to become available… forgot to turn on the accelerator… promptly crashed… burnt the daily
   submission"* [697859 msgid 3454839, 2026-05-08]. Mitigation, from the same thread: *"I always save first, I
   don't submit directly. It catches this mistake (and other mistakes). A mistake I keep doing, because
   accelerators are turned off automatically in this comp."* [697859 msgid 3454844].
2. **Score appears from a *previous* notebook version.** *"During our last submission attempt last night the
   notebook crashed within 250s and I assumed we burnt the attempt for the day. In the morning we had a new
   leaderboard score of 19.03%, but from a previous version of the notebook. Makes no sense. Now I don't know
   which variables to adjust for the next run."* [697859 msgid 3458738, 2026-05-16] — **you may not know which
   code produced your score.**
3. **Same submission, two scores: 0.00 in 15 minutes, then 46% the next day in ~10 hours.** [704408 OP,
   2026-06-04; characterised by a third party in msgid 3468082]. Finished in ~15 min with **0.00 and no error
   message**, from a copy of a notebook that previously ran for many hours; smoke tests passed. 40+ prior
   submissions without this behaviour. **Root cause never established.** A related report: same code, different
   weights, zero LB score, traced to a Kaggle API dataset-version upload *changing the model path*
   [704408 msgid 3471764].
4. **Infrastructure timeout that is not your code.** Topic 738313, 2026-08-31: *"Same notebook that ran in less
   than 10 hours, repeatedly, timed out last night. We never timed out so far, we monitor time (as everyone else
   I guess)."* The reporter checked the public runtime history: *"all our subs run in 10 hours on average, with a
   max of 10h30 since our best score. That all of a sudden we use 2 hours more and that our time gating after
   each puzzle did not work is extremely suspicious."* Kaggle refunded the daily submission; the resubmission ran
   in **9h10** [738313 msgs 3518548 / 3518923 / 3519891 / 3521260]. **Lesson: keep your own external runtime
   record — it is the only evidence you will have.**
5. **Static partitioning wastes the entire run** (§4.3) [697859 msgid 3455909].
6. **TTT silently not running / bailing out** — the `[[0]]` placeholder path (§3.2) and the `>MAX_TTT_TOKENS`
   skip (§3.5). Bitterbot bailed on **56%** of second attempts [697859 msgid 3456280].
7. **Private notebook / missing dataset on fork.** Datasets are **not carried over in a fork by default**
   [724231 msgid 3496586]; created a weeks-long support thread [724231, 19 messages, 2026-07-10 → 2026-07-24].
   Related: the L4 accelerator and submit button only appear if the notebook is created from the competition Code
   panel with the competition dataset attached [735044].
8. **`/kaggle/inference_outputs` does not exist** — the 2025 NVARC notebook writes decoded results there; a
   forker hit `FileNotFoundError` [693074 OP, 2026-04-19]. The *current* notebook version does `os.makedirs(...)`,
   so this is fixed in the official copy but will bite anyone reviving an older fork.
9. **Wheel/artifact bloat breaks teardown.** *"Don't install wheels under `/kaggle/working`. Kaggle will try to
   commit 7 GiB of unpacked wheels as notebook output, and a run that finished all its work will be marked ERROR
   during teardown."* [732723 OP]; corroborated in the harness source, which routes bulky files to a scratch dir
   because *"a run whose work had finished — all 64 grids scored and checkpointed — was still [marked ERROR]"*.
10. **Environment/Docker regressions.** Status code 56 on utility scripts, broken notebook image for ~2 days;
    workarounds: pin to the original environment, "Always use latest environment" on each utility script, or
    duplicate an old notebook and paste cells in [699675 msgid 3458711, 2026-05-16; 729885].
11. **CLI submission 403.** `kaggle` 2.1.2 returned 403 on submit while "Save and run all" worked; never resolved
    in-thread [697831 OP, 2026-05-07].
12. **"Save & Run All" may hang and not exit**; the displayed "running" state can be stale — refresh before
    cancelling [738313 msgs 3520730 / 3520862 / 3521166].
13. **Non-determinism across identical submissions** [685142 msgs 3440071 / 3440402 / 3442399]. Same code, same
    data, different scores.
14. **Rules traps:** internet is off during scoring, so no hosted LLM at runtime [690497 msgid 3441230;
    701528 — host confirms coding assistants are fine *during development* if the final notebook is fully offline];
    all code must be open-sourced to be prize-eligible (arcprize.org); **model *license* is not the same as
    Kaggle's "fine-tunable" UI toggle** — the NVARC Qwen3 model card said "not fine-tunable", which is a Kaggle
    UI limitation, not a legal one, and the author then enabled it [735058 msgs 3512548 / 3512605 / 3512632].
    A still-unanswered rules question asks how §2.5's OSAID base-model requirement interacts with the
    incompatible-license carve-out for exactly this Qwen3 case [740268 OP, 2026-09-08].
15. **Misc:** GPU quota double-charging [696364]; "Corrupted datasets" — irrelevant JSONs appearing in the data
    section [704871]; a leaderboard stuck at 33.87 for weeks while everyone built on the 2025 solution [693263].

---

## 8. Genuinely novel (non-NVARC / non-ARChitects-rehash) approaches — with honest ratings

| Approach | What it is | Evidence | Rating |
|---|---|---|---|
| **TOPAS / Bitterbot** (§2.3) | HRM-style logic/canvas cores + TRM single latent space, ~100M params, deep recursion; trained on one RTX 4090 | ~36% local → 44% local; **11.67% / 19.03% LB** | **Real and different.** The only non-Qwen architecture with a *published LB number*. But it is ~100× below the leaders and its own author attributes the gap to scheduling, not architecture. The deeper recursion made TTT heavier — a structural, not incidental, cost. |
| **TRM as a first-class component** | Tiny recursive model, 7M-ish class, per-puzzle continued full-weight TTT | Official: pass@1 7.6%, pass@2 10.1% eval → **10.0 LB** | **Real but capped.** Independently reproduced at 10.83% LB by a second person. madarshbb's verdict that it is "not good enough for this strategy" is the operative finding. |
| **Program synthesis + offline function library** (§2.10) | Model writes `transform(grid)`; sandbox verifies against demo pairs; 67 pre-generated functions shipped as a 69 KB JSON so Kaggle only *executes* | **2/12 = 16.7%** at `reasoning_effort=none`, vs 0/12 for plain reasoning | **Novel and clever, but the headline number is far weaker than it reads** — see §9. The *architectural* idea (offline strong model, online sandbox execution, zero API at runtime) is the most transferable idea in the corpus for working around the no-internet rule. |
| **EGO-ARC (embryonic growth)** [694618 OP, 2026-04-25] | Neuro-symbolic: scene compiler → object selector → rule executor → fitness evaluator; deterministic; **no rule may depend on task IDs**; every candidate validated on all demo pairs | **9/50 → 17/50 on a local ARC-AGI-1 first-50 sandbox only.** Author states plainly: *"This is not an official hidden-test result and should not be read as a Kaggle leaderboard claim."* | **Honest, small, and self-limiting.** Its most useful content is the negative result: Wake-Sleep, VSA, MCTS, SMT/Z3, cellular automata, loop unrolling and LLM-suggested DSL all "mostly plateaued until the representation itself improved" → *"ARC progress here is currently representation-limited, not search-limited."* No LB score. |
| **Rubik's-cube subgroup decomposition** [712218, 2026-06-22/23] | Treat each task as a group; find the subgroup, solve layer by layer | No score. Thread's own summary of why it fails: *"The subgroup idea is right, it's just that the hard part quietly moved from 'solve the cube' to 'find the cube.'"* One participant reports "subgroup plus greedy nails the train pairs every time" but fails on size/coefficient generalisation | [SPECULATION / partial]. **But it contains a valuable warning** from a 2024 veteran: *"IN 2024 I developed a solver that was solving almost all arc agi eval. And when i submitted it I got like 6%."* [712218 msgid 3478835] |
| **Haumea ARC-1 training solver** [694653 OP, 2026-04-26] | Modular "laws" library solving **400/400 ARC-1 training** tasks; open source + Zenodo DOI | 400/400 **training**, not evaluation | [MEASURED but off-target]. ARC-1 *training* is the easiest possible target, and much of ARC-1 eval is in ARC-2 train (§6.1). A reply notes the smoking gun: *"that's 459 algorithms for 400 tasks. No composition, single algorithm has to fully solve single task"* — i.e. near per-task lookup, not a general solver. |
| **Intel Arc A770 / non-CUDA path** [696718, Boladi, 2026-05-03] | Qwen3-4B + LoRA TTT + AIRV entirely on a 16 GB Arc card, Windows | "Boston-class submission (Qwen3-4B + AIRV n=4) scored **5.83 public**" — the poster's own descriptor, not an official class | [MEASURED, low score] but **methodologically the most rigorous TTT write-up in the corpus** (§3.5). |
| **DDL / DSL abstraction-level argument** [702981, 2026-05-27] | "Primitives are the wrong abstraction level… search a smaller space that generates rotate/flip/transpose as special cases" | No implementation, no score; cpmpml replies *"I never believed in DSL approach for arc agi2"* and another poster claims "I think I've cracked it" without evidence | [SPECULATION / unproven] |
| **Nex/Aquila 35B MoE harness** [732723] | 35B MoE, fp8 vLLM, two-stage generation forcing an answer after reasoning | **0/64 on eval**; 8/64 on train | [MEASURED, negative]. Valuable as a **negative result + engineering notes** (§2.9), not as an approach. |

---

## 9. Claims that look unreliable and why

Ordered by how likely they are to mislead a build decision.

1. **"Bicameral Latent Space / Cortical Sheet / dopamine reward" (topic 695556 + 697859, "Victor"/Bitterbot) —
   the poster's own thread contradicts the marketing.** The Apr-29 OP claims "a true Bicameral Latent Space",
   a "Cortical Sheet" with "Phase 2 event-driven predictive coding", "simulated 'dopamine' reward (Time-To-Live)",
   "post-optimizer Hebbian plasticity", and ~36% at "only mid-way through the grokking phase" with "a significant
   phase change in performance as the model crosses the generalization threshold".
   **Why it is unreliable:**
   - The community reaction was immediate and sceptical: *"any new repo is assumed to be AI slop until proven
     otherwise. Terms like 'cortical sheet' and 'dopamine reward' are not helping."* (msgid 3452290, +4)
   - **It published no leaderboard score** until 8 days later, and that score was **11.67%** — the OP's own
     framing: *"We used this submission primarily as a pipeline test ... using an early, not fully-trained
     checkpoint."*
   - The same author later reports the model **bailed out on 56% of second attempts** and that a run **crashed
     within 250 s**, i.e. the system was not stable.
   - "Grokking phase change" is an **unfalsifiable prediction** with no date and no evidence.
   → **Verdict: treat the neuroscience vocabulary as decoration. The only solid content in that thread is the
   engineering post-mortem (TTT timeouts → `[[0]]` placeholders → 11.67% LB), which is genuinely useful.
   There is no evidence that the Cortical Sheet or dopamine mechanism contributes anything.**

2. **"A 50% drop is not uncommon" — over-generalised from a single measurement.** The measured fact in the same
   post is **22% → 10.83%** for TRM [697859 msgid 3456171]. In the *same thread* the 2025 winner reports
   **30% → 27%** for the Qwen pipeline [697859 msgid 3458771]. The "50%" is a generalisation about "every top
   team last year", offered without data, and the poster immediately concedes they have no way to analyse it:
   *"I don't see how to answer your question as we have no signal from semi private data other than the score."*
   → **Use ~10% as the Qwen-pipeline expectation and ~50% as the TRM expectation. Do not plan around "50%".**

3. **"Program synthesis beats reasoning: 16.7% vs 0%" — the headline is misleading.** [733501]
   - n = **12 grids**. 2 vs 0 solves. No confidence interval; the difference is not statistically meaningful.
   - All runs were at **`reasoning_effort="none"`** — the author states this only in a *reply*, not the OP
     (msgid 3510025), and concedes: *"it is a lower bound, not a miss."*
   - The comparison model is `GPT-5.6-Sol`, which **cannot run in the competition at all** (no internet at
     evaluation). The author's own workaround — a 67-function library pre-generated offline — is a **per-task
     lookup table**, not a generalising solver, and it is only viable because the tasks were known in advance.
   - The title was corrected by a commenter: *"Your post title should be: program synthesis beats no reasoning."*
     (msgid 3510409) — which is the accurate summary.
   - The function library applies "if a function passes all training pairs", but NVARC's `1232`-style
     train-consistency filter is known to be fooled by the **minimality/consistency problem** and by the
     **`17829a00` corrupted demo** (§6.5), so "passes all training pairs" is not a correctness guarantee.
   → **The transferable idea is the offline/online split. The 16.7% number should not drive any decision.**

4. **The `42.64 → 51.17 correct answers` reverse-engineering (topic 699858) is numerology.**
   Two posters derive contradictory arithmetic from the same score ("50 correct answers would be 42.5%" vs
   "50 correct answers would yield 41.67%") and both are contradicted by cpmpml in-thread. The 120/120 split is
   confirmed by the host, but the fractional-credit reconstruction is **[SPECULATION]**. **Do not use it to
   estimate your task count.**

5. **"nvbanana == nano-banana image generation" — speculation the team explicitly declined to endorse.**
   The only confirmed fact is the name inspiration [724647 msgid 3509467]: *"It is true that the name nvbanana
   was inspired by nanobana. I won't comment on the rest of the post."* The rest of msgid 3509366 (that they are
   NVIDIA, that they use Nemotron, that they use a 14B, "no test-time-train now", "sub 10hrs") is
   **[SPECULATION]** by a third party, much of it hedged with "guessing" and "thinking they go for".
   **Note the internal contradiction**: it claims "no test-time-train" while the team's own runtime signature
   (median ~9.4 min, §4.2) is *consistent* with that but proves nothing.

6. **"This competition has upgraded the GPUs to 4xL4, so we can now submit last year's solution"** — true, but
   the same post's headline claim ("I suspect it would achieve Public LB 27 and Private LB 24") was **retracted
   in the same post** once the hardware mismatch was noticed [685142 OP, 2026-03-27]. Later, identical forks of
   the published notebook scored anywhere from 28.47 to 30+, attributed by the author to *"people modify a bit the
   code. Also, our code is not deterministic."* **The "reproduce last year = 27%" framing is not reliable.**

7. **TRM's `pass@10 = pass@5 = pass@2 = 0.1014` is a red flag, not a result.** In the official TRM README the
   top-k curve is flat from k=2 to k=10 and only reaches 0.1375 at k=1000. Combined with `exact_accuracy 0.0436`
   and `accuracy 0.7745` (token-level), this says the model is **almost always wrong in the same way**, so extra
   samples do not help. Any plan that relies on "more TRM samples will fix it" is contradicted by the authors'
   own numbers.

8. **"More synthetic data + more SFT is the only significant lever"** [madarshbb, 729743] is **[OPINION]** from
   someone who did not submit a qualifying run in this competition ("I didnt sub ToPaS") and who repeatedly hit
   GPU limits. It is a well-informed opinion but it is a single practitioner's inference from a set of negative
   ensemble experiments, not a measured result.

9. **The "50M model completes twice as fast" question** [697859 msgid 3458394] is **[SPECULATION]** offered as a
   question, with no measurement. Do not treat it as a scaling law.

10. **`694704` ("How to Improve Code ARC Prize Score from 0.9 to 30.22")** is **[NOISE]** — a help request
    with a title that reads like a claim. The one substantive number in it ("to the first 8 questions from this
    year's test_challenges, basically accuracy is 3/8, and sometimes 5/8") is measured on the **placeholder test
    file** (§6.1) and is therefore **meaningless**. The post was heavily downvoted (-4).

11. **`684761` "Truth guard"** (votes **-8**) — "no hallucinations, pure reasoning", no notebook, no score,
    no evidence. **[NOISE / unproven].** Community response: *"Your model seems really powerful… it can predict
    winning, just not actually do it."*

---

## 10. Coverage: the noise, explicitly

So you know nothing was skipped. Of the 68 topics, the following contain **no actionable method, technique,
configuration or measured result** and are listed here only for completeness:

**Team formation / recruiting (7):** 739494, 740808, 693826, 694676, 689254 (partly), 689054 (partly),
696718 (partly — but it is the best TTT post, see §3.5).

**Rules & admin questions (9):** 740268 (OSAID licence carve-out — genuinely important for prize eligibility in
§7.14), 737679, 701528, 685621, 685660, 697831, 704079, 694745, 684441.

**Prizes / leaderboard meta (3):** 684662 (prize discrepancy; resolved — Kaggle page was correct: $75k/$50k/$40k/
$35k/$25k/$20k/$15k/$15k + $275k Grand + $150k bonus), 693263, 709522 (but its *data* was the most valuable
artifact I used, §4.2).

**Submission/infra bugs (7):** 704408, 738313, 689423, 729885, 735044, 693074, 704871.

**Individual-task questions (6):** 698462 (contains the corrupted-demo finding), 731040, 735545, 697058, 724073
(contains the placeholder finding), 712218 (contains the Rubik's negative result).

**Off-topic / low quality (6):** 733217 (LLM-vs-synthesis opinion, no data, -1), 687701 (philosophy),
691202 (beginner "is 0.00 normal"), 684880 (GPU vs TPU, generic), 684761 ("Truth guard", -8),
694825 (task taxonomy question).

**Short/thin (5):** 687095, 733930 (but supplies the `r=256+rslora / DFS p≥0.2 / 1200 s/puzzle` reproduction
details, §11), 740422 (supplies the 120/120 + "nothing is rerun" confirmation), 703343 (ARC-AGI-3, wrong forum),
720019 (visual EDA notebook, no results).

**Duplicated/low-content (2):** 684857 and 694698 overlap heavily (both are cpmpml on ARC-1 vs ARC-2 difficulty);
both are retained for §6.1.

---

## 11. Federated configuration — the one number you cannot find in the official repo

Topic **733930 msgid 3510624 (2026-08-09)** is the only place in the corpus that states NVARC's *effective*
inference/TTT settings as a participant reproduced them, and it is corroborated by the live notebook:

| Setting | Forum claim (733930) | Live notebook (§3.1) | Agrees? |
|---|---|---|---|
| LoRA rank | `r=256` | `r=256` | ✅ |
| LoRA variant | `+rslora` | `use_rslora=True` | ✅ |
| LoRA targets | `incl. embed/lm_head` | `embed_tokens`, `lm_head` included | ✅ |
| DFS threshold | `DFS p≥0.2` | `max_score = -np.log(0.2)` | ✅ |
| Per-puzzle budget | `1200s/puzzle` | `spend_time > 1200` | ✅ |
| Selection algo | `kgmon` | `score_kgmon` is the default | ✅ |
| Reported score | "consistently score 28–29" vs public notebook's 33.89 | — | [MEASURED, self-reported] |

**This is a valuable cross-check: an independent reproducer matched every headline hyperparameter and still
landed 5 points below the published 33.89.** Their own question — *"which implementation details did you find
score-critical beyond the headline hyperparameters — decode batching/view pairing, KV-cache handling, collator
masking, anything else?"* — went **unanswered**. A reply in the same thread is blunt and worth weighing:
*"NVARC notebook is near-optimized. There are almost zero low-hanging fruits in that notebook. You need to get
really lucky with seeds to get a 33 just with NVARC."* [733930 msgid 3511243, 2026-08-10].

→ **Reproducing the public NVARC notebook is not a path to a competitive score.** It is a path to 28–33.

---

## 12. The three highest-leverage, actually-verified techniques in this corpus, ranked

### #1 — Dynamic, deadline-aware work distribution with nested per-unit time guards
**Evidence:**
- NVARC's shipped code uses a **shared `mp.Manager().Queue()` across 4 L4 workers** with a **global
  `end_time = start + 12*3600 − 600`** propagated into every worker, checked at two granularities
  (per-puzzle `spend_time > 1200`, per-decode `time.time() - start < 540`) — verified in the live notebook source,
  and independently corroborated by a reproducer who cites `1200s/puzzle` [733930].
- The counter-example is **directly measured and catastrophic**: static partitioning with 4 GPUs caused a
  **12-hour run to be wasted** because one GPU drew the hard puzzles while three idled after <11 h
  [697859 msgid 3455909].
- madarshbb independently confirms the queue design and improves it with **largest-puzzles-first** ordering
  [729743 OP], and confirms the guard matters: *"while some helped in CV, it couldnt translate to LB"* is the
  pattern for everything *except* speed work, which did translate.
**Why it is highest-leverage:** it is the only technique in the corpus whose *failure* is publicly documented to
have destroyed a full attempt, whose *success* is visible in shipped winning-lineage code, and which costs
nothing in accuracy. The single most-quoted score gap in the corpus (11.67% LB vs 36% local) is attributed by its
own author to exactly this class of failure.
**Caveat:** the 600 s reserve, 1200 s/puzzle and 540 s/decode constants are tuned for a 12-hour run. If you
adopt the ~12-minute regime the leaders appear to use (§4.2), these constants must be rescaled.

### #2 — Per-puzzle LoRA TTT that is verified to run, with the levers that are measured *not* to matter
**Evidence:**
- The complete, verified configuration (§3.1): **r=256, alpha=32, rslora, dropout 0, targets = all 7 linear
  projections + embed_tokens + lm_head, lr 5e-5 cosine with 0.1 warmup ratio, AdamW, bf16, max_grad_norm 1.0,
  max_seq_length 8192, 1 epoch, batch 1, 16 augmentations per puzzle, LoRA reset per puzzle.** Confirmed against
  an independent reproducer on six of six headline parameters [733930].
- **The negative results are as valuable as the positive ones and they are measured:** NLL-threshold tuning
  ("no statistically significant move in score"); extra TTT steps once loss plateaus ("no point in training
  longer"); alternative selection strategies ("some helped in CV, it couldnt translate to LB"); a second decoding
  round with different seeds ("didnt move score") [729743 OP].
- **The skip policy is measured:** `MAX_TTT_TOKENS=3000` → long prompts get **no TTT** and fall back to base
  inference; per-task TTT cost is **30–60 s (short) / 2–4 min (medium)** and the whole 120-task run extrapolates
  to **12–15 h**, i.e. it does not fit [696718].
- The **`[[0]]` placeholder is the default** in `get_submission()`, and a TTT timeout writes nothing — so a
  timeout is indistinguishable from a wrong answer in the submission JSON. This is the mechanism behind the
  corpus's largest documented local→LB gap.
**Why it is second and not first:** it is verified as *working*, but the measured returns are exhausted at the
NVARC operating point, and the current leaders appear not to be spending their budget this way at all (§4.2).
Adopt it as a baseline, not as a differentiator.

### #3 — Build a deterministic, batch-invariant test harness *before* tuning anything
**Evidence:**
- The winner states flatly: **"our code is not deterministic"** [685142 msgid 3441234]; and **"Same code submitted
  twice after deadline yield two different scores, both better than our winning score"** [685142 msgid 3443042].
  A forker got **28.47 three times** while others got 30+ from the same notebook [685142 msgid 3440071].
- madarshbb's response is the technique: use `github.com/thinking-machines-lab/batch_invariant_ops` to build a
  batch-invariant fork, **cache per-puzzle adapters** so TTT is not re-run, and thereby make the whole pipeline
  deterministic — *"This will help you understand whether your move in CV is due to experiments or variance due
  to LLM stochasticity"* [729743 OP]. He also flags the cost honestly ("this is slow") and reports the tension
  directly: **during submission the batched version is used because "its faster and solves more puzzles."**
- Corroborating measurement noise: TRM's flat pass@2…pass@10 curve (§9.7); the public-eval→LB drop ranging from
  10% to 51% across methods (§6.2); the 0.00-in-15-min then 46%-in-10-h incident (§7.3).
**Why it is third:** it produces no score by itself, but with **1 submission per day** it is the difference
between measuring and guessing. Every "measured" claim in this report that is actually a coin flip traces back to
this problem.

---

## 13. The four things I would act on, stated as decisions

1. **Do not budget 12 hours.** The two teams at the top of the live board run **~713–744 s** and **~9.4 min**
   per submission (70/79 and 21/23 runs in a 30-second band). Treat ~12 minutes as the operating point to beat
   and 12 hours as the ceiling you never approach. Rounding this off: **rabbithole spends 1.7% of the available
   wall-clock for 76.94%.**
2. **Never emit `[[0]]`.** It is the default in the reference pipeline's `get_submission()`, it is what a TTT
   timeout silently produces, and it is worth exactly zero. A diverse low-confidence guess dominates it under a
   best-of-2 metric. Make "every puzzle has a non-placeholder output" an invariant with a test.
3. **Dynamic queue + nested deadlines + a finalisation reserve.** Copy the shape of NVARC's scheduler; skip the
   static sharding. Use a skip policy (long prompt → no TTT) rather than a timeout policy.
4. **Do not expect to win by reproducing NVARC.** An independent reproducer matched r=256/rslora/embed+lm_head/
   p≥0.2/1200 s/kgmon exactly and got **28–29** against a published 33.89, and the community consensus in-thread
   is that the notebook is "near-optimized" with "almost zero low-hanging fruits". The corpus's own strongest
   advice from the 2025 winner is also the least helpful: *"simple tweaks to our last year winning solution won't
   be enough to win this time."* [691081 msgid 3447956]
