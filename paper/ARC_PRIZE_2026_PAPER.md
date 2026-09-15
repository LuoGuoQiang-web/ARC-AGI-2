> # ⚠️ SUPERSEDED — do not attach this file
>
> **The submission artefact is `ARC_PRIZE_2026_WRITEUP.md`** (1,477 words, within Kaggle's
> 1,500-word cap). This file is the longer working draft it was condensed from, kept only as a
> record of the reasoning.
>
> **It contains three claims now known to be wrong.** Attaching it would contradict the Writeup:
>
> 1. The title's "Memory-Constrained" framing, and the body's claim that LoRA TTT is
>    memory-infeasible "on the competition's 14.56 GiB T4". **False as stated.** It was infeasible
>    on the accelerator *we* had selected by default; the competition offers
>    `machine_shape=NvidiaL4` — four L4s, 88 GiB. See Writeup §4.4.
> 2. `assert_gpu_compatible` is described as a working safeguard. It was in fact the thing that
>    *rejected* the L4, because it tested set membership in `torch.cuda.get_arch_list()`, which
>    omits `sm_89`. Fixed in solver v0.5.0.
> 3. **The headline number in §4.3 is false.** It reports "16 of 102 tasks executed a single
>    optimizer step", "21 optimizer steps" and "84 recorded `B_ttt_no_gain` with zero steps",
>    taken from `report.json`. Counting from the kernel log instead: **93 of 102 tasks stepped and
>    the run performed 121 optimizer steps**, disagreeing with the report on **77 of 102 tasks**.
>    The report was wrong because `record_task` runs once per *stage* and a later stage re-records
>    the task with a fresh result whose step count is zero, erasing what adaptation had measured.
>    Fixed in solver v0.5.0 (accumulate instead of overwrite) with regression checks T16a/T16b.
>
> **Point 3 matters most, because it changes the finding itself.** The claim is no longer "the
> adaptation never ran" but "the adaptation ran, on 93 of 102 tasks, at **1.34 optimizer steps**
> against the reference recipe's **128** — and the run's own counter misreported even that, by a
> factor of six". The novelty claim therefore moves from *a masked no-op* to *an adaptation stage
> whose execution was misreported by its own instrumentation*. The Writeup carries the corrected
> version; §4.3 below does not.

# The Adaptation That Wasn't: Silent Test-Time-Training Failure in a Memory-Constrained ARC-AGI-2 Solver

**ARC Prize 2026 — Paper Track submission**

---

## Abstract

The winning entries of ARC Prize 2024 and 2025 share a recipe: a small open LLM, LoRA
test-time training (TTT) on each task's demonstration pairs, and a thresholded search over the
model's outputs. NVARC's 2025-winning system uses a Qwen3-4B backbone with LoRA — the same
configuration we study here. That recipe assumes adaptation happens. We report a case where it
did not, and where nothing in the reported metrics said so.

> ⚠️ **CORRECTED (see the banner at the top).** The paragraph below is wrong. The true numbers,
> counted from the kernel log rather than from `report.json`: **93 of 102 tasks stepped, and the
> run performed 121 optimizer steps** (the report claimed 16 and 21; the two disagree on 77 of 102
> tasks). So the adaptation *did* run — at **1.34 optimizer steps per task** against the reference
> recipe's **128** — and the finding is that its execution was misreported by the run's own
> instrumentation, not that it never executed. The corrected text is in the Writeup, §4.3.

In a 240-task ARC-AGI-2 submission run, **102 tasks were routed to the test-time-training stage.
Only 16 of them executed a single optimizer step.** Across the entire run the stage consumed
7,266 seconds and performed **21 optimizer steps in total**. The remaining 86 tasks were recorded
by the system as `B_ttt_no_gain` — that is, the pipeline attributed a *negative result* to an
adaptation mechanism that had never executed. The cause is an out-of-memory failure on the first
optimizer step: LoRA TTT must share a 14.56 GiB T4 with the search's residual activations, and
when the allocation fails the run continues, because a refinement stage that fails is
indistinguishable, in the metrics, from a refinement stage that does not help.

We consider this failure mode structural rather than incidental. Adaptation is an *optional
refinement* inside a pipeline that must produce a submission regardless; a resource failure
inside it therefore degrades gracefully into "unadapted model", and the accuracy number is
identical in both cases. Anyone reproducing this recipe on hardware smaller than the L4×4 and
P100 configurations used by the prize winners — including Kaggle's free single T4 — is exposed to
it, and an accuracy-only report cannot detect it.

We reach this through an established diagnostic: reporting whether the ground truth appears
anywhere in the solver's candidate pool, which upper-bounds what any selection change could
achieve. This is not our invention — the ARChitects' "coverage" curve, Li et al.'s
"Sample+Oracle", and Moghe & Chin's generation-bound/selection-bound analysis all publish it. We
use it as prior work does, to locate the loss, and it tells us we are generation-bound with zero
selection headroom. We then falsify the obvious remedy with three paired configurations:
halving the search probability threshold, and doubling both beam breadth and node budget, all
leave pool recall at zero. Widening the search is not the lever. Making the adaptation actually
execute is the lever, and we say plainly that we identified it rather than validated it.

Our contribution is a negative result with a concrete, cheap safeguard: **report executed
optimizer steps per task, not intended ones.** One counter would have caught this on day one.

---

## 1. Introduction

ARC-AGI-2 is a benchmark of novel visual reasoning puzzles. Each task supplies a handful of
demonstration input/output grid pairs and asks for the output grid(s) for one or more unseen test
inputs; grids are rectangular arrays of integers 0–9. Tasks are constructed so that each requires
inferring a *new* rule from a few examples, which is what makes the benchmark resistant to
retrieval and to brute force.

Because each task *is* a tiny training set — a few pairs that jointly specify a rule — test-time
training is a natural fit, and it has become the dominant approach: the ARC Prize 2024 technical
report states that all top LLM-based transduction entries use TTT, and that no static-inference
transduction solution scores above 11% [2024 TR]. Our system follows that lineage: a Qwen3-4B
backbone, LoRA adaptation per task, and a thresholded depth-first beam search over the model's
next-token distribution. NVARC's 2025-winning entry uses the same Qwen3-4B + LoRA configuration
[NVARC], so this is a scaled-down instance of an established recipe rather than a new solver. We
make no claim otherwise.

The premise of the recipe is that the model is adapted to the task before it predicts. This paper
is about the gap between that premise and what a run actually does.

**The measurement problem.** Accuracy is a single number, and it is the product of two
capabilities: *generating* the correct output among the candidates, and *selecting* it. The ARC
Prize technical reports and the Kaggle leaderboards report only the product. When it is low, the
natural inference is "the model is not good enough", and the natural remedies are a wider search
or a larger model — both expensive, and at least one of them sometimes pointless.

**What we found.** We applied the standard generation/selection decomposition to our own solver
and it said, correctly, that we are generation-bound: across every configuration and input we
instrumented, the ground truth was never in the candidate pool, so selection contributed exactly
nothing. We then showed that widening the search changes none of this. Only when we asked *why*
the generator was so bad did we look at the adaptation stage's resource telemetry — and found
that it had barely run at all: 21 optimizer steps in 7,266 seconds, with 86 of 102 "TTT" tasks
executing zero steps while being reported as evidence that TTT does not help.

That is the finding. The rest of the paper establishes it, bounds it honestly, and states what we
did and did not verify.

---

## 2. Prior work

### 2.1 ARC and the ARC Prize

ARC-AGI-1 was introduced by Chollet (2019) with a formal definition of intelligence as
skill-acquisition efficiency over an explicit set of Core Knowledge priors
([arXiv:1911.01547](https://arxiv.org/abs/1911.01547)). A task counts as solved only if every
test input is correct, with two attempts per input
([ARC Prize 2025 Technical Report](https://arxiv.org/abs/2601.10904);
[ARC-AGI-2 repo readme](https://raw.githubusercontent.com/arcprize/ARC-AGI-2/main/readme.md)).

ARC-AGI-2 keeps the format but is deliberately harder and less brute-forcible: calibrated 120-task
public/semi-private/private eval sets, removal of tasks susceptible to brute-force search,
controlled human testing with 400+ participants (every task solved by ≥2 humans within 2 attempts),
and new task families targeting symbolic interpretation, compositional reasoning and contextual
rule application
([arXiv:2505.11831](https://arxiv.org/abs/2505.11831);
[benchmark page](https://arcprize.org/arc-agi/2/);
[launch post](https://arcprize.org/blog/arc-agi-2-technical-report)). At publication no leading
model exceeded 5% on ARC-AGI-2, against 20–50% on ARC-AGI-1. ARC Prize 2025's top Kaggle score was
24.03% ([2025 TR](https://arxiv.org/abs/2601.10904)). ARC Prize now reports cost alongside
accuracy, on the stated principle that exhaustive brute force would not demonstrate intelligence.

### 2.2 Solver families

*DSL / program search.* The 2020 winner searched a hand-built DSL by brute force, scoring 20% on
the private set ([ARC Prize 2024 TR](https://arxiv.org/abs/2412.04604)). Hodel's `arc-dsl`
improved program search and released Re-ARC for procedural data generation
([arc-dsl](https://raw.githubusercontent.com/michaelhodel/arc-dsl/main/README.md)).

*Neuro-symbolic induction.* Li et al.'s induction+transduction ensemble reached 56.75% on public
eval, trained on ~400k synthetic variations of Python programs
([arXiv:2411.02272](https://arxiv.org/abs/2411.02272)). Note that "BARC" is ambiguous in this
literature: the ARChitects use it for Li et al.'s system
([arXiv:2505.07859](https://arxiv.org/html/2505.07859v2)) while the 2024 technical report uses it
for a separate "Bootstrapping ARC" repository
([xu3kev/BARC](https://github.com/xu3kev/BARC)).

*Test-time training.* ARChitects won 2024 (53.5% private) with an 8B model, D4 and colour
augmentations, thresholded DFS, and product-of-experts scoring
([2024 TR](https://arxiv.org/abs/2412.04604);
[arXiv:2505.07859](https://arxiv.org/html/2505.07859v2)). NVARC won 2025 (24.03%) by combining an
ARChitects-style TTT model with TRM; their ARChitects component is explicitly **Qwen3-4B with
LoRA** ([NVARC readme](https://raw.githubusercontent.com/1ytic/NVARC/main/README.md);
[2025 TR](https://arxiv.org/abs/2601.10904)). TTT itself originates with Sun et al.
([arXiv:1909.13231](https://arxiv.org/abs/1909.13231)) and was brought to ARC by Cole & Osman and
Akyürek et al. ([arXiv:2506.14276](https://arxiv.org/abs/2506.14276);
[arXiv:2411.07279](https://arxiv.org/abs/2411.07279)).

**Our position.** Architecturally we are a smaller-scale instance of the ARChitects/NVARC recipe.
We do not claim a new solver.

### 2.3 Memory-constrained TTT is known — but *silent* failure is not

That TTT-for-ARC is memory-bound is thoroughly documented. Cole & Osman report two hours on a
single 16 GB P100 ([arXiv:2506.14276](https://arxiv.org/abs/2506.14276)). ARC Prize 2025 doubled
compute to L4×4
([competition archive](https://arcprize.org/competitions/2025/archive)). The ARChitects used
separate models per sub-task "because of the limited compute budget on the Kaggle servers, in
terms of both memory and speed", with rank-32 LoRA, batch size 1, and 128 test-time steps
([ARChitects 2025 report](https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/)).
McGovern used a global batch of 384 instead of 768 "owing to lower total VRAM available on
4×L4s", and found LoRA-only adaptation near zero while full fine-tuning worked best
([arXiv:2511.02886](https://arxiv.org/abs/2511.02886)).

The field knows adaptation is tight. **We could not find any prior report of an adaptation failure
that is masked — where the run continues, the stage is recorded as attempted, and the resulting
accuracy is reported as if adaptation had occurred.** That is the gap this paper addresses, and it
is the one claim we make for novelty.

### 2.4 The generation/selection decomposition is *not* ours

Reporting whether the ground truth appears anywhere in the candidate pool, as an upper bound on
selection, is established practice and we cite it rather than claim it:

- The **ARChitects** plot "coverage": "the fraction of tasks where the correct solution was among
  the sampled candidates, and thereby **provides an upper bound for the performance of the
  selection algorithms**", alongside the fraction of present-correct candidates that selection
  actually recovers ([arXiv:2505.07859, Fig. 4](https://arxiv.org/html/2505.07859v2)).
- **Li et al.** report "Sample+Oracle", which "**upper-bounds** the accuracy of randomly selecting
  one program consistent with the training examples"
  ([arXiv:2411.02272, Fig. 8](https://arxiv.org/html/2411.02272v4)).
- **Moghe & Chin** make the split their headline: pass@k analysis showing a pipeline "is
  generation-bound, not selection-bound"
  ([arXiv:2607.06764](https://arxiv.org/html/2607.06764v1)).

pass@k is likewise standard in program synthesis. So the diagnostic we use in §4.1 is borrowed,
and the conclusion it reaches about our system is a replication of a known pattern, not a
discovery. We use it for what it is good at: telling us which half of the system to fix.

### 2.5 Data-validity findings

The ARC Prize 2024 technical report itself documents private-eval overfitting risk and warns that
"the different evaluation datasets are not drawn from a consistent human difficulty distribution"
([arXiv:2412.04604](https://arxiv.org/abs/2412.04604)). The 2025 report documents knowledge
contamination of frontier models with ARC data, concluding the phenomenon "is now occurring with
ARC-AGI-1 and ARC-AGI-2 – accidentally or intentionally" ([arXiv:2601.10904](https://arxiv.org/html/2601.10904v1)).

Distribution mismatch between splits has been quantified: a Habr analysis reports that "two or
more test inputs are required by **6.9% of train tasks and 40.8% of eval tasks**" (χ² = 127.3),
with further divergence in input area, output area and colour counts
([Habr](https://habr.com/ru/articles/1071730/)). That is essentially our §4.4 measurement, and we
therefore treat ours as a **reproduction, not a priority claim**. Related work includes McGovern
§4.7 ([arXiv:2511.02886](https://arxiv.org/html/2511.02886v1)) and, for ARC-AGI-1, H-ARC's
measurement that eval output grids are significantly larger than train
([arXiv:2409.01374](https://arxiv.org/abs/2409.01374)).

---

## 3. Approach

### 3.1 System under study

**Grid-native language model.** A public supervised-fine-tuned Qwen3-4B checkpoint (3.634B
parameters, bfloat16) whose tokenizer has been reduced to a 16-token vocabulary: the digits `0`–`9`,
a newline marker, `user`/`assistant` role markers, a padding token and the chat delimiters. A grid
is emitted as a token sequence, and every emitted grid is checked for rectangularity and for values
in 0–9 before it can become a candidate — which removes malformed-grid failures by construction.

**LoRA test-time training.** Per task, adapt on that task's demonstration pairs with LoRA (rank 16,
α = 32, learning rate 5e-5, one epoch, maximum sequence length 8192). Augmentations are geometric
plus colour permutations, applied consistently to inputs and outputs.

**Constrained beam search.** A depth-first beam search over the next-token distribution, pruned by a
per-token probability threshold, a maximum number of children per beam, and a global node budget.
This is what turns the model into a candidate *generator* rather than a single-shot predictor.

**Cascade scheduling.** A cheap full-coverage sweep (Stage A) runs first; TTT is then spent on the
tasks that look weakest by a confidence score (Stage B); a symbolic engine backfills remaining slots
(Stage C).

### 3.2 Instrumentation

Two counters matter, and only one of them is conventional.

**Executed optimizer steps per task.** `ttt_steps` is incremented inside the optimizer loop, so it
counts steps that actually ran. This is the counter that produced this paper's result. It is not a
metric anyone would think to add, because the *intended* number of steps is known from the
configuration — and, as §4.3 shows, the intended and executed numbers can differ by an order of
magnitude with nothing in the output revealing it.

**Pool recall.** For each test input we check whether the ground-truth grid appears anywhere in the
candidate pool, then compare that ceiling against the accuracy achieved (§2.4 establishes this as
borrowed practice). Because a task may carry several test inputs, each needing its own predictions,
the aggregate must be taken over *test inputs*; §3.3 explains how we got that wrong first.

### 3.3 Two ways a diagnostic lies in favour of its author

Both of these bugs produced results that supported the hypothesis we were testing. We report them
because a measurement whose failure mode is "confirms what you hoped" is worse than no measurement.

**Counting stages instead of tasks.** The pool-recall aggregate was accumulated inside the
per-stage recording function, which runs once per *stage* rather than once per task. A task
revisited by Stages B and C therefore had its test inputs counted again. One run published a
denominator of 16 where only 11 test inputs existed, inflating the denominator of the very metric
under test. The aggregate is now recomputed from the unique per-task rows, with a regression check
(`T9g`) that fails against the old code.

**Reading a missing field as zero.** Reports produced before the instrumentation existed have no
pool-recall field. Reading that absence as "0 in pool" manufactures false confirmation of exactly
the claim under test. Our tooling now marks such runs uninstrumented and excludes them from every
recall claim.

---

## 4. Results

### 4.1 We are generation-bound, with zero selection headroom

Across four instrumented runs on the evaluation split, pool recall was zero in every one:
**the ground truth was in the candidate pool for 0 of the 35 (configuration, test-input)
measurements.**

Those 35 measurements are **not 35 independent samples**. The three controlled configurations run
the *same* eight evaluation tasks, and the smoke run's two tasks are a subset of them, so the union
is **8 unique tasks carrying 11 unique test inputs**. Treating it as n = 35 would overstate our
confidence considerably, so we do not.

| Run | Tasks | Unique test inputs w/ truth | In pool | Pool recall | Distinct candidates per task |
|---|---|---|---|---|---|
| smoke (tasks ⊆ those below) | 2 | 2 (⊆ 11) | 0 | 0.0000 | 4–9 |
| `base` (shipped defaults) | 8 | 11 | 0 | 0.0000 | 4–18 |
| `prob10` (threshold 0.2 → 0.1) | 8 | 11 (same inputs) | 0 | 0.0000 | 5–29 |
| `branch4` (beam 3→4, nodes 6000→12000) | 8 | 11 (same inputs) | 0 | 0.0000 | 4–21 |

**8 unique tasks · 11 unique test inputs · 33 (config, input) measurements · 0 hits.**

Against the correct denominator, a zero count over 11 independent inputs bounds the true rate at
**≤ 23.8% (95% confidence)**. We report the bound rather than the point estimate because the point
estimate is the more flattering number. What the data supports without qualification is the
within-sample statement: in every configuration and on every input we measured, the truth was
absent and selection headroom was therefore exactly zero.

The control that matters is the last column. With 4–29 *distinct* grids per task, the pools are not
degenerate, so "the pool was just too small" is ruled out. `prob10` is the strongest single
refutation: its pools reach 29 distinct candidates, more than twice the shipped configuration's,
and still contain the truth zero times.

### 4.2 Search width is not the lever

Because a small pool would trivially give zero recall, the experiment was designed as a paired
comparison: identical tasks, time budget, augmentation counts and seed, with only the search-width
knobs changed.

| Configuration | Probability threshold | Max children/beam | Node budget | Solved | Pool recall |
|---|---|---|---|---|---|
| `base` | 0.2 | 3 | 6,000 | 0/8 | 0.0% |
| `prob10` | **0.1** | 3 | 6,000 | 0/8 | 0.0% |
| `branch4` | 0.2 | **4** | **12,000** | 0/8 | 0.0% |

Halving the probability threshold and doubling both beam breadth and node budget changed nothing:
not accuracy, not pool recall, not the selected rate. The wider search cost 30% more time per task
(203 s → 263 s mean Stage A) and bought zero. The three configurations were specified as the
hypothesis test before being run; the hypothesis failed.

### 4.3 The result: the adaptation stage barely executed

> ⚠️ **CORRECTED — the table and the two paragraphs below are wrong.** They were taken from
> `report.json`. Counted from the kernel log, the same run shows **93 of 102 tasks executing at
> least one optimizer step and 121 optimizer steps in total**; the report and the log disagree on
> **77 of 102 tasks**. The report was wrong because `record_task` runs once per *stage* and a later
> stage re-records the task with a fresh result whose step count is zero, erasing what adaptation
> had measured. The corrected finding — adaptation ran, at **1.34 steps per task** against the
> reference recipe's **128**, and its execution was misreported by the run's own instrumentation —
> is in the Writeup, §4.3. The prose further down that describes the TTT failure as *silent and
> total* should be read with that correction applied.

Pool recall says the generator is bad. It does not say why. The answer is in the resource telemetry,
and it is the central finding of this paper.

**Full 240-task submission run:**

| Quantity | Value |
|---|---|
| Tasks routed to the TTT stage (Stage B) | **102** |
| Tasks that executed ≥ 1 optimizer step | **16** |
| Tasks recorded as `B_ttt_no_gain` with **zero** optimizer steps | **84** |
| Total optimizer steps in the entire run | **21** |
| Wall-clock consumed by Stage B | **7,266 s** |

**102 tasks entered test-time training. 16 of them trained.** The stage spent 7,266 seconds to
perform 21 optimizer steps — roughly 346 seconds per step. The 84 tasks that never stepped were
recorded by the pipeline as `B_ttt_no_gain`, i.e. as evidence that adaptation was attempted and did
not help.

**How far short that is.** This is not a marginal shortfall against the recipes we are imitating,
and the comparison is worth stating in the same units. The ARChitects — the 2024 winners — report
adapting with rank-32 LoRA at batch size 1 for **128 test-time steps**
([their 2025 report](https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/)). Our
240-task run executed **21 optimizer steps in total, across all 240 tasks**, and never more than
**2 steps on any single task**. A task in our system therefore received on the order of **1/64th**
of the adaptation the published recipe uses, and 86 of 102 tasks that reached the stage received
none at all. Whatever one believes about how many steps are needed to install a task-specific
algorithm from a handful of demonstrations, two is not a serious attempt at it — and, crucially,
nothing in our reported output distinguished "we tried two steps and it did not help" from "we
executed zero steps and called it a negative result".

The same comparison explains why the memory wall is the binding constraint rather than a nuisance.
Adaptation budget in this family is bought in steps, and steps are bought in memory; running 128
steps at batch size 1 requires the activations for those steps to coexist with the model and the
search's caches. We had 14.56 GiB, of which the base model occupies roughly half.

The mechanism, from the kernel log of the eight-task controlled run (abridged; the elided text is a
memory-accounting sentence, the lines shown are verbatim):

```
[0934a4d8] pre-TTT free 7.57 GiB of 14.56 GiB
  TTT step 0 failed: CUDA out of memory. Tried to allocate 1.77 GiB.
    GPU 0 has a total capacity of 14.56 GiB of which 1.13 GiB is free.
[0934a4d8] TTT 0 steps in 44.1s
! TTT disabled for the rest of the run: LoRA TTT OOM on the first step
```

In that run, Stage B was entered by five tasks. TTT was attempted on three: two logged **two
optimizer steps each**, and the third logged **zero** after the allocation above failed — which is
what fired the circuit breaker. It then disabled TTT for the *entire remaining run*, so the last two
Stage B tasks never attempted adaptation at all.

Note the two distinct masking behaviours, because both hide the failure:

- **Without a breaker** (the 240-task run): each task fails independently, the pipeline records
  `B_ttt_no_gain`, and the run continues. 16 out-of-memory events were logged; 86 tasks were
  silently unadapted.
- **With a breaker** (the controlled run): one failure disables the mechanism globally, and the
  remaining tasks are unadapted by construction. This is the *reasonable* engineering choice — do
  not burn the budget retrying a doomed allocation — and it converts a per-task failure into a
  run-wide one.

Neither is visible in the accuracy number. And the failure is structural, not incidental: adaptation
is an optional refinement inside a pipeline that must emit a submission regardless, so a resource
failure inside it degrades gracefully into "unadapted model". Graceful degradation is exactly what
makes it silent. In this system the competition is direct — LoRA TTT must share a 14.56 GiB T4 with
the search's residual activations, and the mechanism the design depends on is the one that cannot
fit.

**This reframes §4.1.** Zero pool recall is not evidence that a 4B model cannot do ARC. It is
evidence that *this* generator was never adapted: the candidate pools were produced by the base
model, and widening a beam over an unadapted model samples more wrong grids rather than better ones
— precisely what §4.2 observed.

### 4.4 Why the local development loop could not have caught it

The competition's public test-challenges file is not a holdout. Measured against the files
themselves (`tools/verify_leak.py`):

| Property | Measured |
|---|---|
| Task ids shared with the training split | 240 / 240 |
| Byte-identical to the training task of the same id | **240 / 240** |
| Answers present in the training solutions file | 240 / 240 |
| Content overlap with the evaluation split | **0** |
| Multi-answer share: public test vs evaluation | **7.1% vs 40.8%** |

Every one of the 240 public test tasks is a training task with its answer shipped alongside it, so
any accuracy measured against that file is a training-set number — which the competition rules also
forbid reporting. (We did not find prior publication of the byte-identity and zero-overlap
properties specifically, but our search was inconclusive rather than exhaustive, so we claim no
priority. The 7.1%/40.8% structural mismatch is a reproduction of the Habr analysis cited in §2.5.)

The structural half has a concrete cost. Because the public file under-represents multi-answer
tasks, a submission builder that emits one entry per task validates cleanly against it for 93% of
tasks and produces a *malformed* submission on the hidden set. Our own earlier submission was
rejected with *"Your notebook generated a submission file with incorrect format"* and scored
nothing, and we could not reproduce it locally because the local file does not exercise it.

---

## 5. Discussion

### 5.1 What is new here, and what is not

We state this explicitly, because the honest accounting is narrower than the results might suggest.

| Claim | Status |
|---|---|
| Pool recall / coverage as an upper bound on selection | **Not ours.** ARChitects' "coverage", Li et al.'s "Sample+Oracle", Moghe & Chin's generation-bound analysis (§2.4) |
| Generation-bound, not selection-bound, for this family | **Replication** of a known pattern (Moghe & Chin) |
| Widening the search does not fix a generation bound | **Weak / confirmatory**, shown here for three paired configurations on our system |
| Split distribution mismatch (multi-answer share) | **Reproduction** of the Habr analysis (§2.5) |
| Public test file is byte-identical to training | **Possibly original, claimed as inconclusive** — no prior publication found, search not exhaustive |
| **A masked adaptation failure: OOM on step 0, recorded as attempted, reported as "no gain"** | **Novel as far as we can determine.** Memory-bound TTT-for-ARC is well documented (§2.3); a *silently masked* failure is not |

### 5.2 The safeguard

The finding generalises past our codebase. Any pipeline with an optional learned refinement stage —
test-time training, gradient adaptation, self-critique, verifier-guided repair — has the property
that a resource failure inside the stage is indistinguishable, from the output, from the stage
working correctly and not helping. In both cases the pipeline returns a prediction and a score.

We therefore recommend one change, which costs nothing:

> **Report executed optimizer steps per task, not intended steps.** One counter, incremented inside
> the training loop, distinguishes "adaptation did not help" from "adaptation did not happen". Had
> we reported it from the start, this paper would not have needed to be written.

Two corollaries from our own experience:

1. **Make the degradation per-task, not global.** A circuit breaker that disables adaptation for the
   remainder of a run converts one task's OOM into a systematically unadapted submission. Degrade
   the *sequence length* and retry; do not disable the mechanism.
2. **Do not report a stage's null result when the stage did not run.** Recording 84 zero-step tasks
   as `B_ttt_no_gain` was the single most misleading thing our own reporting did.

To make this adoptable rather than merely advisable, we release the counter as a standalone,
dependency-free module (`tools/adaptation_audit.py`, stdlib only, no framework assumptions, with
its own 11-check self-test). It wraps any refinement stage in a context manager, counts the steps
that actually complete, records an exception inside the block as a *failed attempt* rather than as
a task that did not benefit, and raises before you write up a result that depends on a stage which
mostly did not run. It is deliberately general: nothing in it is specific to ARC, to PyTorch, or to
test-time training.

### 5.3 What would actually raise the score

Our evidence points away from search and towards the generator. The intervention that matters is
making adaptation fit in memory — batching or sharding the training sequences rather than
materialising 8192-token activations at once — and treating an OOM as a per-task degradation. Both
are engineering fixes to the mechanism whose failure we measured. **We identified this lever; we did
not validate it within this competition's compute budget.** We flag it as the next experiment rather
than as a result, and we would rather say so than present an untested remedy as a finding.

---

## 6. Conclusion

> ⚠️ **CORRECTED.** As above: the true figures are **93 of 102 tasks stepping** and **121 optimizer
> steps**, not 16 and 21. The conclusion is therefore not that adaptation never ran, but that it
> ran at about one percent of its intended budget and that the run's own counter concealed it.

We set out to improve an ARC-AGI-2 solver and found that the part of it we believed in most was not
running. The system follows an established and successful recipe — a 4B open model, LoRA test-time
training per task, a thresholded beam search, as used by the ARC Prize 2024 and 2025 winners. In a
240-task run, 102 tasks entered the adaptation stage and 16 executed a single optimizer step; the
stage consumed 7,266 seconds to perform 21 steps, and 84 unadapted tasks were reported as evidence
that adaptation did not help.

The diagnostic we used to locate the loss is not ours — pool recall, or "coverage", is published
practice, and it correctly told us we were generation-bound with zero selection headroom. Following
it led us to falsify the obvious remedy (wider search, three paired configurations, no change) and
then to ask why the generator was so weak, which is where the adaptation telemetry answered.

The contribution we would ask readers to take is not the negative result about one solver but the
failure mode and its safeguard. Optional learned refinement stages fail silently by construction,
and the fix is one counter: report the optimizer steps that executed, not the ones you configured.
Applied to our own system, that counter turned a plausible story about model capability into a much
simpler and more actionable one about a 1.77 GiB allocation.

---

## Appendix A — Reproducibility and artefact

The solver, the diagnostic tooling, the raw run reports quoted above (`report.json` per-task rows),
and the kernel logs containing the TTT out-of-memory and circuit-breaker messages are public under
MIT-0. Every number in this paper is reproduced by `tools/evidence_ledger.py`, which prints each
cited value next to the file it came from.

| Component | Purpose |
|---|---|
| `work/arc_w1/arc26_solver.py` | The solver under study |
| `tools/adaptation_audit.py` | The safeguard of §5.2 as a standalone, dependency-free module |
| `tools/evidence_ledger.py` | Prints every cited number with its source path |
| `tools/recompute_pool_recall.py` | Recomputes pool recall from stored reports, without a GPU |
| `tools/measure_ttt_budget.py` | Re-derives the optimizer-step length table on CPU |
| `tools/verify_leak.py` | Re-derives the §4.4 table from the competition files |
| `tools/exp_suite.py` | The paired-configuration harness behind §4.2 |
| `tests/` | 74 checks requiring no GPU, including regression checks for both bugs in §3.3 |

**Cost.** The full 240-task run took 37,963 s of wall clock on a single Tesla T4 at 14.12 GiB peak
memory. The three controlled configurations cost 4,313 s, 4,219 s and 4,212 s. The §4.3 result and
all of §4.4 cost no additional GPU time — they are counters and file comparisons.

**Limitations, stated plainly.**

- The instrumented pool-recall sample is **8 tasks / 11 test inputs** from the 120-task public
  evaluation split, re-measured under three configurations. The 95% upper bound on pool recall is
  **23.8%**, not 0. We deliberately do not inflate the sample by counting repeated measurements of
  the same inputs as independent trials.
- The masked-failure finding rests on **one system** at one scale on one device class. The mechanism
  is well-evidenced in our logs and the counter is cheap to replicate, but we have not shown how
  often it occurs elsewhere. Establishing that is a matter of adding the counter, which is the point.
- Reported accuracy is evaluation-split only. Training-set numbers are never reported; the
  competition forbids it, and the public test file is training data, so a "test score" from it would
  be meaningless twice over.
- We make no claim that small models cannot do ARC. The claim is narrower: in this system the
  adaptation did not execute, so the search was sampling from an unadapted distribution.

---

## Appendix B — Submission linkage

This paper describes a real entry in the ARC Prize 2026 ARC-AGI-2 Kaggle competition. The linked
notebook produces `/kaggle/working/submission.json` in the required format —
`{task_id: [{"attempt_1": grid, "attempt_2": grid}, ...]}`, one entry per test input, in input
order, grids as rectangular 0–9 integer arrays. It runs with internet disabled and writes a
structurally complete submission before any solving begins, so a time-limited or interrupted run
still emits a well-formed file.

Leaderboard score: `{{LEADERBOARD_SCORE}}`
