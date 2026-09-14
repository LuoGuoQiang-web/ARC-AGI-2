# AGENTS.md

Guidance for coding agents working in this repository.

## 最高准则（用户指令，2026-09-14 确立）

> **该文件夹中一切任务修改和变动都是为了向 100% 获得奖金去设计。**

这条指令优先于本文档其余所有内容，也优先于任何"技术上更有趣"或"工程上更干净"的选择。
任何改动在动手前都要能回答一个问题：**它是否提高了拿到奖金的概率？**
答不上来、或答案是否定的，就不做——哪怕它本身是好的工程。

### 由这条准则推出的硬约束（算术事实，不是判断）

奖金路径是 **ARC-AGI-2 Grand Prize $275k** 与 **Paper Prize $450k**，两者共用同一套
6 维评分表（Accuracy / Universality / Progress / Theory / Completeness / Novelty），
每项 0–5 分、**取算术平均**；Outstanding Papers Pool 要求 **> 4.5**。

$$\text{平均} \le \frac{\text{Accuracy} + 25}{6}$$

⇒ **Accuracy 必须 ≥ 3/5**，否则无论论文写得多好，数学上都不可能过线
（Accuracy = 1 时上限 4.33；Accuracy = 2 时恰好 4.50，而要求是**严格大于**）。

**⇒ 提高求解器精度是拿到任何奖金的前置条件，不是"六分之一的小加分"。**
推导与出处见 `ARC_PRIZE_2026_MASTER_BRIEF.md` §2.3.1.1。

### ⚠️ 配套纪律：不得为了凑分而夸大结果

这条不是与上一条并列的目标，而是实现它的手段。本仓库已经**两次**修正过"恰好支持自己
假设"的测量错误（pool_recall 的分母在多个 stage 间重复计数；把缺失字段读成 0），
并在 `paper/README.md` 与 brief §2.3.2 中显式写下了新颖性边界。

评委看得到提交的代码与日志。主张一旦被发现注水，**Completeness 与 Theory 两项会直接崩掉**，
而那正是真正降低获奖概率的做法。诚实的负面结果得分，永远高于被戳穿的正面结果。

## Project

ARC Prize 2026 campaign. Two deliverables, both aimed at the same prizes:

- **`work/arc_w1/arc26_solver.py`** — the main line: Qwen3-4B (16-token grid vocabulary) +
  LoRA test-time training + constrained beam search, run on Kaggle GPU. **This is where the
  score comes from**, and the score gates every prize (see the constraint above).
- **`arc_prize_v1/`** — the earlier offline, CPU-only symbolic DSL engine. Now serves as the
  `--engine` symbolic backfill (Stage C) rather than as the main solver.
- **`paper/`** — the Paper Track submission, its reproduction guide, and the novelty accounting.

The campaign plan lives in `ARC_PRIZE_2026_MASTER_BRIEF.md`; the solver design, test evidence
and limitations live in `work/arc_w1/README.md` and `work/arc_w1/FINDINGS_2026-09-14.md`.

Verified competition facts (check `ARC_PRIZE_2026_MASTER_BRIEF.md` §2.6 before assuming):
submissions must be a Kaggle notebook, no internet during evaluation, exactly 2 predicted
outputs per test input, `/kaggle/working/submission.json`. Note also §2.6.1 (a code
competition needs `-k` **and** `-f` **and** `-v`), §2.6.2 (the public test file is training
data, so any score measured on it is meaningless), and §2.4 (GPU quota is **30 h/period**).

## Agent skills

### Issue tracker

Issues live as GitHub issues in `LuoGuoQiang-web/ARC-AGI-2`, driven by the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

The five canonical roles keep their default names: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` at the repo root plus `docs/adr/`. See `docs/agents/domain.md`.

## Commands

Main line (`work/arc_w1/`) — **all of these run without a GPU**, so they are always safe to run:

```bash
cd work/arc_w1
python tests/test_model_free.py           # 38 checks: serialisation, augmentation, DFS, diagnostics
python tests/test_merge_shards.py         # 20 checks: cross-period shard merge
python tests/test_engine_integration.py   # 11 checks: symbolic engine wiring
python tools/verify_leak.py comp_data     # re-derives the data-validity table (needs comp_data/)
python tools/recompute_pool_recall.py     # honest pool recall from stored runs, no GPU
python tools/evidence_ledger.py           # every cited number + the file it came from
python tools/exp_suite.py --dry-run       # build the experiment notebooks, spend no quota
```

Spending GPU — always check `kaggle quota` first, and remember `kpush.py` defaults to
`machine_shape=NvidiaTeslaT4` (a generic `Gpu` gets a P100, on which every CUDA op fails):

```bash
python tools/kpush.py --slug <slug> --argv "<solver argv>"   # push a run
python tools/kpush.py --status --slug <slug>                 # poll
python tools/kpush.py --output --slug <slug>                 # collect report.json
```

Do **not** wrap long orchestration in a background job: a previous attempt was killed after
pushing only the first of two kernels. Push each kernel as its own short foreground command.

Legacy symbolic engine (`arc_prize_v1/`):

```bash
python arc_prize_v1/tests/selftest.py            # engine end-to-end, expects 36/36
python arc_prize_v1/tests/run_notebook_dryrun.py # executes the generated .ipynb end to end
python arc_prize_v1/tests/eval_local.py          # measure on a real ARC-AGI-2 eval set
python arc_prize_v1/tools/build_notebook.py      # regenerate the notebook after editing the engine
python arc_prize_v1/tools/build_notebook.py --check  # fails if the notebook is stale
```

## Conventions

- **Never report a training-set score.** The competition forbids it, and the public test file
  *is* training data (§2.6.2), so a "test score" measured locally is meaningless twice over.
  Validate on the evaluation split only.
- **Any new metric must be checked against its own denominator.** Two of our metrics have
  already been caught inflating the denominator in the direction that supported the
  hypothesis under test. Add a check that the aggregate equals the rows it summarises.
- Never commit credentials. `kaggle.json` is git-ignored — keep it that way.
- `arc_prize_v1/` is **standard library only, offline, CPU only**. Don't add heavy dependencies
  to it. `arc26_solver.py` deliberately imports `torch`/`transformers` lazily inside functions
  so that its logic stays testable on a machine with no GPU — preserve that property.
- `arc_prize_v1/arc_prize_v1.py` is the source of truth for the legacy engine;
  `ARC_PRIZE_2026_v1.ipynb` is generated from it. After editing, regenerate and let `--check` pass.
- Every solver candidate must reproduce **all** demonstration pairs before it may vote.
