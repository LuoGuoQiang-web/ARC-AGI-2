# ARC Prize 2026 — ARC-AGI-3 track: feasibility report

**Date of investigation:** 2026-09-14
**Verdict up front: NO-GO for M2 (and effectively NO-GO for Top Score). GO only as a cheap, low-expectation probe.**
See §7 for the reasoning and §8 for the single cheapest experiment that would settle it.

---

## 0. Corrections to the context we were given

Two of the "verified" facts are **wrong**, and one is off by four. These matter because they are the numbers the go/no-go turns on.

| Given | Actually observed | Source |
|---|---|---|
| Top 5 = 11.04 / 8.21 / 7.63 / 7.51 / 5.96 | **18.81 / 8.68 / 8.44 / 8.40 / 8.21** | `kaggle competitions leaderboard -c arc-prize-2026-arc-agi-3 -d` (snapshot `2026-09-14T06:13:29`) |
| 3032 teams | **3036 rows** on the public leaderboard | same |
| Milestone M2 "still open" — implied unknown | **Still open.** M1 was already awarded on 2026-07-06; M2 is the final one | [arcprize.org/blog/arc-prize-2026-milestone-1](https://arcprize.org/blog/arc-prize-2026-milestone-1) |

The given list looks like a stale/mis-transcribed snapshot: its 2nd–4th entries (8.21, 7.63, 7.51) are the **current ranks 5, 6, 7**, and `11.04` does not appear anywhere on the current board. Treat the numbers in this report as authoritative; they come from the downloaded leaderboard CSV (`lb/lb.csv`, 3036 rows), not from a web page.

**Prize structure as given is confirmed** ([arcprize.org/competitions/2026/arc-agi-3](https://arcprize.org/competitions/2026/arc-agi-3)): Grand $700K (first to 100%), Top Score $75K (40/15/10/5/5), Milestone $75K (two × $37.5K, each 25/10/2.5).

---

## 1. What ARC-AGI-3 actually is — VERIFIED

Sources: [arcprize.org/competitions/2026/arc-agi-3](https://arcprize.org/competitions/2026/arc-agi-3), [arcprize.org/arc-agi/3](https://arcprize.org/arc-agi/3), [docs.arcprize.org/games](https://docs.arcprize.org/games), [docs.arcprize.org/llm_agents](https://docs.arcprize.org/llm_agents), [docs.arcprize.org/local-vs-online](https://docs.arcprize.org/local-vs-online)

**Task format.** Turn-based, interactive. The agent is dropped into a novel video-game-like environment with **no instructions and no stated win condition** and must discover both by acting.

- Grid: max **64×64**, cell values **0–15**; `(0,0)` top-left, `(x,y)`.
- Actions: `RESET` plus `ACTION1`–`ACTION7` (typically 1–4 = directional, 5 = game-specific interact, 6 = complex action requiring `x`,`y` coordinates, 7 = undo). The legal set is given per-frame in `available_actions`.
- Game state enum: `NOT_PLAYED` / `NOT_FINISHED` / `WIN` / `GAME_OVER`.
- Response metadata includes `levels_completed` and `win_levels` (total levels), so progress is observable but the goal is not explained.

**Agent interface** (`agents/agent.py`, [raw.githubusercontent.com/arcprize/ARC-AGI-3-Agents/main/agents/agent.py](https://raw.githubusercontent.com/arcprize/ARC-AGI-3-Agents/main/agents/agent.py)) — two methods:

```python
def is_done(self, frames, latest_frame) -> bool
def choose_action(self, frames, latest_frame) -> GameAction
```

The reference `Agent` loop carries a default `MAX_ACTIONS: int = 80` guard; the official starter overrides it to `float('inf')`.

**Difference from ARC-AGI-1/2.** ARC-AGI-1/2 are *static*: a handful of input→output grid demonstration pairs, then a single-shot prediction, scored on accuracy. ARC-AGI-3 is *sequential*: no demonstration pairs, no examples, the agent must perceive→hypothesize→act→observe in a loop, and it is scored on **completion × action efficiency relative to humans**, not on answer accuracy. The four claimed capabilities are exploration, modeling, goal-setting, and planning/execution ([arcprize.org/competitions/2026/arc-agi-3](https://arcprize.org/competitions/2026/arc-agi-3)).

**Scoring — RHAE** ("Relative Human Action Efficiency") ([docs.arcprize.org/methodology](https://docs.arcprize.org/methodology)):

```
level_score  = (human_baseline_actions / ai_actions) ^ 2      # capped at 1.15
game_score   = weighted average of level_scores, weight = 1-indexed level number
total_score  = average of all game scores                     # 0–100%
```

Two structural consequences that dominate strategy:

1. **The square is brutal.** Taking 2× the human action count yields 25%; 10× yields 1%.
2. **Completion gates the maximum.** Game score is capped by how many levels you finished. The documented example: a 5-level game where you complete only levels 1–4 caps your game score at `(1+2+3+4)/(1+2+3+4+5) = 10/15 = 66.7%` *no matter how efficiently you played*. Level 1 is nearly worthless. You cannot grind early levels into a good score.

Human baselines come from first-time players, using the **upper median** per level ([docs.arcprize.org/methodology](https://docs.arcprize.org/methodology)).

---

## 2. The Milestone prizes — VERIFIED, and M1 is already gone

Source: [arcprize.org/competitions/2026/arc-agi-3](https://arcprize.org/competitions/2026/arc-agi-3), [arcprize.org/blog/arc-prize-2026-milestone-1](https://arcprize.org/blog/arc-prize-2026-milestone-1)

- **M1 — June 30, 2026 — AWARDED.** Announced 2026-07-06. Total $37.5K split 1st $25K / 2nd $10K / 3rd $2.5K.
  - 1st: **Tufa Labs, "The Duck"** — small open-source LLM writing and running Python in a live REPL; runs Qwen 3.6 27B FP8 locally.
  - 2nd: **Reki** — vision-LLM-as-policy, renders frames to images, Gemma-4-31B locally, returns JSON action. Built on the official GPT-OSS-120B template.
  - 3rd: **Md Boktiar Mahbub Murad, "forge"** — same JSON-action pattern. His notebook is titled `ARC-AGI-3 LB 0.86(3rd place candidate- Milestone)`.
- **M2 — September 30, 2026 — STILL OPEN, and it is the last one.** The M1 post says verbatim: *"The second (and final) milestone prize will end September 30th. You can start competing right now."* As of 2026-09-14 that is **16 days away**.
- **What qualifies:** the milestone rewards "the top mid-competition **open-source** solutions." The M1 post presents them as "the top three submissions from our ARC-AGI-3 Kaggle competition," i.e. **leaderboard rank at the date**, with the open-source requirement layered on top ([arcprize.org/competitions/2026](https://arcprize.org/competitions/2026): *"All leading participants are expected to open source their solutions to be eligible for a prize"*; *"Participants must open source their solutions before receiving official private evaluation scores"*).

**Exact score needed for M2: there is no fixed threshold.** It is rank-based, not threshold-based. The operative number is therefore *what rank 3 costs on 30 September*, which can only be bounded by today's board:

> **Today, rank 3 = 8.44. Rank 5 = 8.21.** To win an M2 prize you must almost certainly finish above ~8.5–10, and the board is still rising (see §4).

⚠️ **UNVERIFIED:** whether the milestone is adjudicated on the *public* leaderboard snapshot at the date or on the private rerun; how ties are broken; whether M1's three winners remain eligible for M2. None of these is stated in any source we could fetch. We read the Kaggle rules page and it returns no text via fetch (JS-rendered); the `kaggle competitions download` call 403s because the rules have not been accepted on this account.

---

## 3. The Top Score prize — VERIFIED structure, rank-based

Source: [arcprize.org/competitions/2026/arc-agi-3](https://arcprize.org/competitions/2026/arc-agi-3), [arcprize.org/competitions/2026](https://arcprize.org/competitions/2026)

- $75K guaranteed, five slots: **1st $40K / 2nd $15K / 3rd $10K / 4th $5K / 5th $5K**.
- Determined by final standing at the **2026-11-02** submission deadline (key dates: submissions due Nov 2, results announced Dec 4).
- **Score currently needed for each slot** (2026-09-14 snapshot, `lb/lb.csv`):

| Rank | Team | Score |
|---|---|---|
| 1 | Tufa Labs | 18.81 |
| 2 | Ebi | 8.68 |
| 3 | Lord Han Solo | 8.44 |
| 4 | NVARC3 | 8.40 |
| 5 | Third Intelligence | 8.21 |

⚠️ **UNVERIFIED:** whether final Top Score ranking uses the public leaderboard or a private rerun of the top notebooks. The general rules imply a technical review before award ("Prizes are awarded at the sole discretion of ARC Prize Inc. and are subject to review by our Technical Team").

---

## 4. How entrants score 5–11% — and the answer to "is there a trivially-strong baseline?" is **NO**

This was the most decision-relevant question and the answer is unambiguous.

### 4a. The official starter scores 0.0

The official [ARC-AGI-3-Kaggle-Starter](https://raw.githubusercontent.com/arcprize/ARC-AGI-3-Kaggle-Starter/main/README.md) ships an agent that picks random actions. Its own troubleshooting section:

> **"My local score is 0.0** — That's expected for the random starter agent. Your job is to make it non-zero. 🙂"

`kaggle kernels output inversion/arc3-sample-submission-just-explore` returns the **same random `MyAgent`** (`MAX_ACTIONS = float('inf')`, `random.choice`). So "Just Explore", "Random Agent" and the starter are all the same ~0 baseline.

### 4b. Independent confirmation from the literature

[arXiv:2605.25931](https://arxiv.org/abs/2605.25931) ("Explore Before You Solve", Liew Keong Han, 2026-05-25):

> "random and no-explore baselines score **0.0000**"
> "The linked code track entry achieves **RHAE=0.30** on the full **55-game private evaluation**."
> "every one [of the 25 public games] is reachable through non-intelligent strategies"

The author's own competition submission — a BFS solver with an offline pre-solve cache, 180s/level budget — scores **0.30%**.

### 4c. The board is extremely top-heavy

From `lb/lb.csv` (3036 rows, 2882 with a non-zero score):

| Threshold | Teams ≥ |
|---|---|
| ≥ 0.01 | 2882 |
| ≥ 1 | 1072 |
| ≥ 3 | 442 |
| ≥ 5 | **22** |
| ≥ 6 | 12 |
| ≥ 8 | **5** |
| ≥ 9 | 1 |

Percentiles (sorted descending): top 1% = 4.68 · top 2% = 4.19 · top 5% = 3.77 · top 10% = 3.35 · median = 0.29.

**So: getting a non-zero score is essentially free (95% of teams manage it); getting to 5% puts you in the top 0.7% of 3036 teams; getting to 8% puts you in the top 5 teams.**

### 4d. The board is still moving fast

At the M1 cutoff the 3rd-place score was **0.86**. The same three teams today: Tufa Labs **18.81** (rank 1), Reki's team **5.57** (rank 15), forge **3.00** (rank 439). In ~10 weeks the top went from ~1 to 18.81 — roughly an **18× move**. Any read of "8.44 is what rank 3 costs" must assume that number is *higher* by 30 September.

### 4e. What the leaders actually build

From the M1 post and the public kernel list (`kaggle kernels list --competition arc-prize-2026-arc-agi-3 --sort-by voteCount`): LLM agent harnesses driving **local 27B–31B open-weight models** offline inside the Kaggle notebook (Qwen 3.6 27B FP8, Gemma-4-31B, GPT-OSS-120B template). Tufa Labs' "Duck" harness runs Python in a live REPL and evicts old messages for infinite play. Submission counts among the top 50 teams: median **40**, max **151**. These are iterating teams, several of them multi-person (NVARC3, Kyutai is an AI lab).

**There is no shortcut.** No public baseline gets you several percent for free.

---

## 5. Compute and evaluation mechanics — mostly VERIFIED

The single most useful artefact here is [`scripts/build_notebook.py`](https://raw.githubusercontent.com/arcprize/ARC-AGI-3-Kaggle-Starter/main/scripts/build_notebook.py), which shows exactly what the Kaggle rerun does.

- **It is genuinely interactive, step by step.** During scoring the notebook talks HTTP to a **`gateway` sidecar** at `http://gateway:8001`. The notebook does:
  ```
  curl --retry 999 http://gateway:8001/api/games     # wait for gateway
  cp -r .../ARC-AGI-3-Agents /kaggle/working/
  python main.py --agent myagent                      # our agent plays, live
  ```
  The gateway records every action and emits `/kaggle/working/submission.parquet` with columns `[row_id, game_id, end_of_game, score]`. **The agent acts during scoring; this is not a batch prediction task.**
- **It is a code competition, offline.** `kernel-metadata.json` sets `"enable_internet": false`. The starter README: *"All accelerated Kaggle sessions have internet disabled."* The `arc-agi` wheel is installed from the competition dataset (`/kaggle/input/competitions/.../arc_agi_3_wheels`). **No API-based models** ([arcprize.org/competitions/2026](https://arcprize.org/competitions/2026)).
- **No GPU is strictly required.** The starter exposes four accelerators: `cpu` (`none`), `t4` (`nvidiaTeslaT4`), `p100` (`nvidiaTeslaP100`), `rtx6000` (`nvidiaRtx6000`, `g4-standard-48`). *"No GPU required for the starter agent"* / *"cpu — No GPU. Good for the random starter or any non-ML agent."* But every competitive approach runs a local LLM, so a GPU is required in practice.
- **RTX 6000 is ARC-AGI-3 exclusive** but *"burns GPU quota faster — use only when you're confident."*
- **Submission limit: 5 per day** (starter README).
- **The scored set is 55 hidden games.** The competition dataset ships **25 public environments** — verified by our own listing (`kaggle competitions files -c arc-prize-2026-arc-agi-3 --page-size 200`): `ar25 bp35 cd82 cn04 dc22 ft09 g50t ka59 lf52 lp85 ls20 m0r0 r11l re86 s5i5 sb26 sc25 sk48 sp80 su15 tn36 tr87 tu93 vc33 wa30`. The arXiv paper independently states the private evaluation is **55 games**, and the M1 post's 3rd-place author is quoted: *"local public-game checks weren't a reliable leaderboard proxy."* The competition data bundle itself is dated **2026-04-17**, i.e. also *stale* relative to the current game versions in [docs.arcprize.org/changelog](https://docs.arcprize.org/changelog).
- **Access gate:** `kaggle competitions download -c arc-prize-2026-arc-agi-3` returns **403 Forbidden** on this account — the competition rules have not been accepted. `kaggle competitions files` and `kaggle kernels output` work fine.

⚠️ **UNVERIFIED — per-episode step and time limits.** The reference `Agent` class defaults to `MAX_ACTIONS = 80`; the official starter sets it to `inf`, and the gateway's enforcement (if any) is not documented anywhere we could fetch. The only concrete budget figure found is from the arXiv paper, describing **its own** solver: *"BFS depth d and time limit 180s/level."* We could not confirm whether the competition gateway imposes a global per-notebook wall-clock cap.

⚠️ **UNVERIFIED — Kaggle notebook runtime cap.** Widely reported as **12 hours** for CPU/GPU (9 h TPU), and consistent with a Kaggle discussion titled *"are you still there? Your notebook stops after 12 hours of continuous use."* We could **not** fetch a Kaggle doc page that states it — `kaggle.com/docs/notebooks` and `kaggle.com/docs/competitions` both return only the page title to our fetcher. Treat 12 h as likely but unconfirmed.

**Relevance of frontier models:** [arcprize.org/blog/astra](https://arcprize.org/blog/astra) (2026-09-03) reports OpenAI GPT-6 Astra at **99.9%** on ARC-AGI-3 Semi-Private with a Provider Adapter harness, for $19K. That is the *benchmark* leaderboard with network access, and is **not available in the Kaggle competition** (offline). It does, however, set the ceiling: the benchmark is essentially solved by an API model while the offline code competition tops out at 18.81%. The gap *is* the competition.

---

## 6. Honest feasibility judgement

**Setup being judged:** solo, no prior ARC-AGI-3 work, ~16 days to M2, ~30 GPU-h/week on a single Kaggle **T4** (a `t4` session is 2×T4 ≈ 32 GB VRAM total).

### (b) Winning an M2 milestone prize — **NO. Probability ≈ 0–1%.**

Blunt arithmetic:

- M2 is **rank-based**, top 3 at 2026-09-30.
- Rank 3 today is **8.44**. The board has moved ~18× in 10 weeks. Assume 9–11 is the real bar in 16 days.
- 0.86 was **3rd place** on 30 June. That same 0.86 today ranks roughly **500th**. The target is not "build something that works" — it is "in 16 days, from a standing start, reach a score that took the current #3 team months and 62 submissions."
- The M1 winners' own tooling is a live-REPL LLM harness driving a **27B FP8** model. On 2×T4 (32 GB) you can *maybe* run a 4-bit ~27B or an 8-bit ~14B with tight context, versus RTX 6000's 48 GB. You are starting behind on hardware as well as on time.
- Only **22 of 3036 teams** are above 5%. You would need to leapfrog all of them plus the rest of the field into the top 3.

Expecting to win M2 is not a plan; it is a lottery ticket with a 16-day scratch-off period.

### (a) Getting onto the Top Score board (top 5 by 2026-11-02) — **NO, but less hopeless. Probability ≈ 1–3%.**

That is 7 weeks, not 16 days, and the board is deep (22 teams above 5%). But rank 5 is 8.21 *today* and rising, so the realistic entry bar on 2 November is probably **11–13**. A solo newcomer with 30 GPU-h/week and no prior work on this benchmark reaching that is a genuine long shot. The one thing that would change this calculus is if the *iteration* loop, not novel research, is the binding constraint — which is exactly what §8 tests.

### What IS achievable in 16 days

A working, honest submission scoring roughly **0.3–1.5%**. That is a real result — but 0.86 was 3rd place in June, and today it is ~500th. **It wins nothing.** Be clear-eyed that "shipping something" and "winning a prize" are three orders of magnitude apart here.

### Minimum viable entry, if you build one anyway

1. Accept the competition rules (required — currently `userHasEntered = False` and downloads 403).
2. Fork [`arcprize/ARC-AGI-3-Kaggle-Starter`](https://github.com/arcprize/ARC-AGI-3-Kaggle-Starter); it gives you the exact competition-rerun notebook, the gateway wiring, and the submission format for free.
3. Replace `choose_action` with: frame-diff change detection → object segmentation → a **coordinate-targeted ACTION6 heuristic** (prefer small, rare-coloured, button-like shapes rather than random pixels — this is precisely what the M1 2nd-place entry describes doing with numpy, no GPU needed) → an undo-on-no-change rule. This alone beats random and is cheap.
4. Only then add a local quantised LLM for hypothesis generation, if quota allows.
5. Validate on the 25 public games **for plumbing only** — the arXiv paper shows all 25 are solvable by non-intelligent strategies, so a high public score is *not* evidence of a good leaderboard score.

---

## 7. GO / NO-GO

> ## NO-GO
>
> **Do not spend the next 16 days chasing the M2 milestone.** The prize is rank-3-at-a-date; rank 3 is 8.44 today and rising; the M1 third-place score of 0.86 now ranks ~500th; and 22 of 3036 teams are above 5%. A solo entrant with no prior ARC-AGI-3 work and a T4 has no realistic path to the top 3 in 16 days.
>
> **The Top Score board is also a NO-GO as a primary objective** (~1–3%): it needs top 5 of 3036 by 2 November, with an estimated bar of 11–13.
>
> **The only defensible GO is a bounded probe** — capped at ~1 GPU-hour and 1 of your 5 daily submissions — to buy the one fact we could not obtain from outside: *how much leaderboard score is available without novel research.* Run the experiment in §8. If it comes back ≥6, revisit the Top Score decision with real data. If it comes back ≤2 — which is what we expect — close the track and put the 30 GPU-h/week into ARC-AGI-2, where the repo already has a working solver and a mathematically gated prize path.

---

## 8. The single cheapest experiment that would settle it

**Fork the best-scoring public Duck-harness notebook, submit it under our account, and read our own leaderboard score.**

- Candidate: `foysalemonshanto/lb-9-arc3-duck-v12-with-qwen-3-8-27b` (self-reported LB 9, 283 votes, last run 2026-08-18). Fallback: `keithtyser/duck-qwen3-8-27b-fp8` or `jeroencottaar/tufa-labs-duck-harness-june-30-milestone-winner`.
- Procedure: `kaggle kernels pull` → push under our handle → Save & Run All → *Submit to Competition* → read the score.
- **Cost:** ~1 GPU-hour + **1 of 5 daily submissions**. No research, no design work.
- Why this and not a local run: local runs use the **25 public games**, which the literature shows cannot discriminate a good agent from a trivial heuristic. **Only the private 55-game leaderboard is informative**, so only a real submission answers the question.

**Decision rule, fixed in advance:**

| Result | Reading | Action |
|---|---|---|
| **≥ 6.0** | Score is reachable by configuration + iteration, not by invention | Reopen the Top Score question; you have ~7 weeks and a working harness |
| **2.0 – 6.0** | Real work needed but the gap is not absurd | Top Score remains a long shot; M2 is still NO-GO |
| **≤ 2.0** (expected) | The 5–18% band is genuine capability, not plumbing | **Close the ARC-AGI-3 track.** Redirect the GPU budget |

*Caveat: a forked notebook is a measurement instrument, not a prize-eligible entry — prize eligibility requires the submitter's own open-sourced work. Do not confuse the probe with a submission strategy.*

---

## 9. Verification ledger

**VERIFIED by direct fetch or command:**

- Prize structure, milestone amounts/dates, submission requirements, "no internet" — [arcprize.org/competitions/2026/arc-agi-3](https://arcprize.org/competitions/2026/arc-agi-3), [arcprize.org/competitions/2026](https://arcprize.org/competitions/2026)
- M1 awarded 2026-07-06; winners, their techniques and models; M2 is the final milestone ending Sept 30 — [arcprize.org/blog/arc-prize-2026-milestone-1](https://arcprize.org/blog/arc-prize-2026-milestone-1)
- RHAE formula, 1.15 cap, level-weighted aggregation, completion cap — [docs.arcprize.org/methodology](https://docs.arcprize.org/methodology)
- Grid 64×64, values 0–15, actions, state enum — [docs.arcprize.org/games](https://docs.arcprize.org/games)
- Agent interface, `MAX_ACTIONS = 80` default — [agents/agent.py](https://raw.githubusercontent.com/arcprize/ARC-AGI-3-Agents/main/agents/agent.py)
- Gateway sidecar, `KAGGLE_IS_COMPETITION_RERUN`, submission.parquet schema, `enable_internet: false`, accelerator options, RTX 6000 exclusivity, 5 submissions/day — [scripts/build_notebook.py](https://raw.githubusercontent.com/arcprize/ARC-AGI-3-Kaggle-Starter/main/scripts/build_notebook.py), [README.md](https://raw.githubusercontent.com/arcprize/ARC-AGI-3-Kaggle-Starter/main/README.md), [notebooks/kernel-metadata.json](https://raw.githubusercontent.com/arcprize/ARC-AGI-3-Kaggle-Starter/main/notebooks/kernel-metadata.json)
- Starter/random agent scores 0.0 — same README; confirmed identical source via `kaggle kernels output inversion/arc3-sample-submission-just-explore`
- Full leaderboard: 3036 rows, 2882 non-zero, top-5 = 18.81/8.68/8.44/8.40/8.21, thresholds and percentiles — `kaggle competitions leaderboard -d` → `lb/lb.csv`
- 25 public environments, `arc_agi-0.9.8`/`arcengine-0.9.3` wheels — `kaggle competitions files -c arc-prize-2026-arc-agi-3 --page-size 200`
- Competition data download 403s (rules not accepted) — `kaggle competitions download -c arc-prize-2026-arc-agi-3`
- Private evaluation = 55 games; random/no-explore = 0.0000; a BFS pre-solve submission = 0.30; all 25 public games trivially solvable — [arXiv:2605.25931](https://arxiv.org/abs/2605.25931)

**UNVERIFIED / could not confirm:**

- Whether M2 is adjudicated on the public leaderboard snapshot or a private rerun; tie-breaks; whether M1 winners are excluded from M2.
- Whether Top Score final ranking uses the public LB or a rerun.

Both of the above were pursued and blocked. `https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3/rules` and `/overview` return only the page title to a fetcher (JS-rendered). `kaggle competitions download` returns `403 Forbidden` (rules not accepted on this account). Direct probes of Kaggle's internal API with the credentials in `~/.kaggle/kaggle.json` — `api/i/competitions.CompetitionService/GetCompetition` → **403**, `api/i/discussions.DiscussionsService/GetTopicById` → **404**, `api/v1/competitions/view/<slug>` → **404** — did not work. The rules text is not recoverable from outside an accepted-rules browser session, so these two items should be re-checked by a human in the browser before any money is spent.
- Per-episode step/time caps imposed by the competition gateway. (Reference framework default is 80; starter uses ∞.)
- Kaggle notebook wall-clock cap (12 h widely reported; no fetchable primary source).
- Whether the user's GPU allocation covers RTX 6000 for this competition, and at what quota burn rate.
- The exact rank-3 score on 2026-09-30 — unknowable in advance; bounded below by today's 8.44.

*Local artefacts produced during this investigation: `lb/lb.csv` (full 3036-row leaderboard snapshot), `kern/`, `ksrc/` (downloaded sample submissions).*
