# ARC Prize 2026 — v1 引擎（ARC-AGI-2，离线符号搜索）

本目录是手册 [`../ARC_PRIZE_2026_MASTER_BRIEF.md`](../ARC_PRIZE_2026_MASTER_BRIEF.md) §4「一个项目，三次兑现」的**代码本体第一版**：
一个离线可复现的 ARC-AGI-2 求解器 + 完整的运行/备份/续跑/自评工程外壳。

---

## 1. 已核实事实（核实时间 2026-09-11，来源见链接）

| 事实 | 内容 | 来源 |
|---|---|---|
| 竞赛名 | **ARC Prize 2026**，3 条赛道 | [总览](https://arcprize.org/competitions/2026) |
| 主攻赛道 | **ARC-AGI-2**（$700k，Progress $275k / Grand $275k / Bonus $150k） | [ARC-AGI-2 赛道](https://arcprize.org/competitions/2026/arc-agi-2) |
| 提交形式 | **必须是 Kaggle Notebook**，评测期**无网络** | 同上 |
| 计分 | 每个 test 输入**恰好预测 2 个输出**，任一完全匹配则该题得 1 分，取平均 | 同上 |
| 提交文件 | `/kaggle/working/submission.json` = `{task_id: [{"attempt_1": grid, "attempt_2": grid}, ...]}`，list 长度 = 该题 test 输入个数，grid 为 0–9 矩形整数 | 官方 `sample_submission.json`（三方交叉验证） |
| 开奖条件 | 获奖必须**开源**（CC0/MIT-0 等宽松许可） | [Paper Prize](https://arcprize.org/competitions/2026/paper) |
| Paper Track | 论文必须挂一份 ARC-AGI-2 / ARC-AGI-3 的 Kaggle 代码提交，**代码分数不必高** | 同上 |
| 手册 §0 的日期 | 手册写「2026-09-09」，**实际为 2026-09-11** | 本机 `Get-Date` |

> NOTE.md 内容是 Paper Track 的数据页原文「This is a Hackathon with no provided dataset.」——
> 证实 Paper Track 无数据集，代码必须落在 ARC-AGI-2 赛道上，与本目录设计一致。

---

## 2. 文件清单

| 文件 | 作用 |
|---|---|
| `arc_prize_v1.py` | **引擎本体**，纯标准库、无网络、无 GPU 依赖 |
| `ARC_PRIZE_2026_v1.ipynb` | **开箱即用的 Kaggle Notebook**（引擎已内嵌，上传即用，见下方方法 A） |
| `KAGGLE_CELLS.md` | Kaggle 操作手册：上传法（推荐）+ 手动粘贴法 + 速查 + 故障排查 |
| `tools/build_notebook.py` | 从 `arc_prize_v1.py` 生成 `.ipynb`，并在生成时校验内嵌引擎与源文件逐字节一致 |
| `tests/make_fixture.py` | 合成 ARC 风格 fixture（12 题：10 题可解、2 题不可解），变换独立实现，避免自证 |
| `tests/selftest.py` | 引擎端到端自测：32 项断言（提交格式、续跑、重建、备份、回滚、负例） |
| `tests/run_notebook_dryrun.py` | **把生成的 `.ipynb` 整份真跑一遍**（本地无 Jupyter，故自带最小执行器） |
| `tests/perf_probe.py` | 真实规模性能探针（30×30、2–8 组示范、1–3 个 test 输入） |

---

## 3. 怎么用

### 最快路径：上传现成 Notebook（零复制）

1. 下载 `ARC_PRIZE_2026_v1.ipynb`；
2. <https://www.kaggle.com/code> → **+ New Notebook** → **File → Import Notebook → Upload** 拖进去；
3. 右侧 **Add Input → Competitions → `ARC Prize 2026 - ARC-AGI-2`**；
4. Session options：**Internet Off**、**Accelerator None**；
5. **Run All** → 看到 `✅ 成功：提交文件已生成`；
6. **Save Version → Save & Run All** → 到竞赛页 Submit 那个版本。

详细步骤、手动粘贴备选方案、速查表与故障排查都在 `KAGGLE_CELLS.md`。

### 本地（开发/诊断）

```bash
cd arc_prize_v1
python tests/selftest.py              # 引擎 32/32 通过
python tests/run_notebook_dryrun.py    # Notebook 整份执行通过
python tests/perf_probe.py             # 实测每题材耗时
python tools/build_notebook.py         # 改完引擎后重新生成 .ipynb
```

---

## 4. 求解器设计（v1）

```
任务 -> 推断背景色 -> 生成候选程序 -> 逐条用全部示范验证 -> 按复杂度排序 -> 去重取前 2 个预测
                                     ↑ 验证不通过一律淘汰（执行反馈）
```

**DSL 原语 29 个**：8 个二面体变换、裁剪/去边框、4 向重力、整数放大缩小、1×2/2×1/2×2 平铺、
4 轴镜像补全、去均匀行列、最大/最小连通块裁剪。

**组合与着色**：
- 深度 0：单原语（29 个）
- 深度 1：17 个几何/形状原语的两两组合（289 个）
- 每条链尾自动**拟合颜色映射**（逐格拟合、不一致即淘汰）——这一项让「几何 + 换色」类任务一次覆盖
- 合计约 318 条候选链 + 常数输出程序

**分层排序（tier）**，这是 v1 最关键的设计决策：

| tier | 内容 | 说明 |
|---|---|---|
| 0 | 已验证的 DSL 链 / 常数程序 | 排序键 `(tier, 组合深度, 换色数量, 原语注册序)` —— 越简单越靠前，**不是**字母序 |
| 1 | 通过示范验证的结构先验（逐格多数、众数输出、固定画布、平铺到目标形状…） | 即使通过验证也**永远排在 tier 0 之后**，因为它没有解释结构 |
| —— | 未验证先验 | 仅当 tier 0/1 全空时兜底 |

**两输出策略**：`attempt_1` 取最简有效程序；`attempt_2` 取**预测结果不同**的次优有效程序；若只有一种预测，则用下一个先验补位。
（官方规则允许任一命中，因此两路必须是**不同**假设才有收益。）

---

## 5. 工程外壳（「备份、更新」等正常功能）

| 功能 | 实现 | 触发方式 |
|---|---|---|
| 断点续跑 | 追加式 `checkpoint.jsonl`（每题一行，后写覆盖先写，天然幂等） | 同配置再跑一次即可，自动复用 run |
| 强制重跑 | 新 run 目录 + 忽略 checkpoint（run id 保证不撞车） | `force=True` |
| 提交重建 | 只用 checkpoint 重算 `submission.json`，不重跑求解 | `rebuild_submission()` |
| 备份快照 | zip 打包 manifest/checkpoint/submission/diagnostics/state/log，按标签轮转保留 | 结束时自动 + `snapshot(label, keep)` |
| 回滚 | 备份恢复到任意目录后再人工确认覆盖 | `Run.restore_backup(zip, dest)` |
| 运行清单 | `manifest.json`：引擎哈希、配置哈希、数据指纹（大小+首 1MB 哈希）、环境、Kaggle 变量 | 自动 |
| 状态文件 | `state/PROJECT_STATE.md`（对标手册 §0 的字段） | 每次运行结束自动 |
| 旧 run 清理 | 保留最近 N 个 run，避免 `/kaggle/working` 膨胀 | `runs_keep=5` |
| 崩溃保命 | 开跑即写全 0 合法提交；每 25 题刷盘；全局超时自动收尾 | 自动 |
| 找不数据的保命路径 | 若 challenges 缺失但存在 `sample_submission.json`，按官方样例结构产出合法提交 | 自动 |
| 自评与失败分类 | dev 模式对公开评估集打分，输出 `diagnostics.csv`（失败类别 / 像素准确率 / 形状对比） | `mode="dev"` |

---

## 6. 测试证据（本地实测，非推测）

`tests/selftest.py` — **32/32 通过**，覆盖：

- 12 题 fixture 端到端跑通，10 题可解题**全部**被已验证搜索解出（`search=10`），2 题不可解按预期落到先验（`prior_only=2`），准确率 13/15 = 86.67%
- 每题 test 输入个数为 1/2/3 的任务都能生成**长度正确**的 attempts 列表
- 提交结构校验通过；负例（缺题、锯齿行）被校验器正确拒绝
- 续跑：同配置第二次运行 `search=0`（一题不重算），分数完全一致
- 重建：仅凭 checkpoint 重建提交，12/12 题；并且 `bind_run_id()` 会把**所有**派生路径指向被指定的 run（防回归）
- 强制重跑：新 run id，重新求解
- 备份：快照生成、轮转上限、恢复出 `submission.json`
- 保命路径：只有 `sample_submission.json` 时仍产出结构合法提交

`tests/run_notebook_dryrun.py` — **生成的 `.ipynb` 整份执行通过（0 失败）**，覆盖：

- 7 个代码格按顺序执行无异常，`%%writefile` 写出的引擎与 `arc_prize_v1.py` **逐字节一致**（57327 bytes）
- 求解格输出 `✅ 成功`，产出 3534 字节的 `submission.json` 并通过结构校验
- 校验格、备份/续跑/重建格、自评格全部正常；自评 86.67%、失败分类 `{'solved': 13, 'shape_mismatch': 2}`
- 快照非空（4946 bytes）；**没有留下任何空 run 目录**

`tests/perf_probe.py` — 真实规模实测：

| 场景 | 耗时 | 选中程序 |
|---|---|---|
| 30×30 rot90，4 组示范，3 个 test | 0.16 s | `rot90` |
| 20×25 换色，6 组示范，2 个 test | 0.14 s | `id+cmap9` |
| 30×30 平铺 2×2，3 组示范，1 个 test | 0.11 s | `tile2x2` |
| 30×30 裁剪→rot90（深度 1），4 组示范，2 个 test | 0.14 s | `rot90` |

→ 均值 **0.14 s/题**，240 题约 **34 秒**、120 题评估集约 **17 秒**。默认预算 90 分钟，留有 100 倍以上余量。

---

## 7. 诚实的期望与局限（不要自我安慰）

- **本 v1 在真实 ARC-AGI-2 评估集上预期只有个位数百分比的正确率**，很可能在 0–5%。它不是冲榜方案，而是：
  1. 一条**打通的端到端提交通路**（奖牌与后续一切的前提）；
  2. 一台**失败诊断机**（`diagnostics.csv` 的失败分类就是论文「先诊断、再对症下药」的原料）。
- 深度只到 2、无对象级推理、无「同色计数/排序/图结构」类原语、无测试时训练 —— 这些正是 W2–W4 的活。
- 演示集很小（3–4 组）时，链 + 颜色映射可能**偶然**通过全部示范，这是所有 DSL 方法的固有风险；tier 排序与
  「换色数量」惩罚只能缓解、不能消除。W2 要用「留一法验证」（留一组示范做伪验证）来量化这个风险。

---

## 8. 与三条兑现路径的关系（手册 §4）

| 路径 | 本 v1 的贡献 |
|---|---|
| ARC-AGI-2 榜单 / Featured 奖牌 | 提交通路已打通（保住参赛资格与奖牌前提）；分数待 W2+ 提升 |
| ARC-AGI-2 榜单 / Progress Prizes（8 个奖位） | 本版不构成竞争力，属长线目标 |
| ARC-AGI-2 Grand Prize（Solution Writeup $275k） | 工程外壳 + 失败分类 = Writeup 的「Completeness / Progress」素材 |
| Paper Track（$375k 池 + $75k Top） | `diagnostics.csv` 是论文「诊断 → 归因 → 方法 → 证据」结构的第一段证据 |

---

## 9. 对手册 §11（W1 检查清单）的推进状态

| W1 项 | 状态 |
|---|---|
| 1. 规则核实 | ✅ 提交形式（Notebook-only / 无网络）、计分（2 输出任一命中）、提交文件结构、开源要求均已核实；**算力/时限上限官方标注「随开赛公布」，仍未核实** |
| 2. 额度实测 | ⬜ 待你在 Notebook 里记录（v1 不需要 GPU，但提交是否计入额度仍需实测） |
| 3. 基线调研（3–5 个开源方案） | ⬜ 未做（v1 是自研浅层 DSL，不是复用基线；W2 起按手册标准打分挑选） |
| 4. 最简提交 | 🟡 通路已就绪，等你实际推一次版本确认榜单资格 |
| 5. W1 报告 | ⬜ 待第 2/3 项完成后产出 |

---

## 10. v2 候选方向（按预期收益排序，待 W1 报告后定稿）

1. **对象级原语**：连通块提取 → 按大小/颜色/接触关系排序 → 重排、拉伸、嵌入、计数。
2. **留一法验证**：用「留一组示范」检测偶然通过，直接提升真实集正确率。
3. **深度 3 + 束搜索**，并以「换色数量 + 原语序」做 MDL 修剪。
4. **对称/周期类任务**：周期检测、按周期补全、边框/分隔带解析。
5. **测试时自适应（TTT）** 接入：先用现有符号搜索产出伪标签，再小模型打分剪枝（这一层才需要 GPU）。

---

## 11. 许可

引擎以 **MIT-0** 意图发布（No Attribution），满足 ARC Prize「获奖必须开源且许可宽松」的硬要求。
正式对外发布前，在仓库根目录补 `LICENSE` 文件与署名（手册 §12：开源发布由你执行）。
