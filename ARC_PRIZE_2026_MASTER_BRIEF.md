# ARC Prize 2026 作战手册（Agent 自解释版）

> 本文档的唯一目的：**让任何一个没有本次对话上下文的 agent（包括未来的我们）读完就能独立接手执行**。
> 所有事实均标注来源与核实时间；所有未核实项都显式标为 `TODO-W1`。
> 本文件取代 `arc_prize_2026_8week_plan.md`（后者保留作历史版本）。

---

## 0. 当前状态（每次会话开始/结束时更新）

| 字段 | 值 |
|---|---|
| 当前阶段 | **两条路线并存，正在合并**：① 本机 v1 符号 DSL（评估集 0.00%，见 `arc_prize_v1/BASELINE_V1.md`）；② **另一台电脑的 GPU 神经路线**（TTT + 约束 DFS，评估集子集 **1/24**）—— 已抢救到 `work/arc_w1/` |
| 当前日期 | **2026-09-14**（实测；本表此前写 09-11，已过期 3 天） |
| 硬截止 | 代码提交 2026-11-02；论文 2026-11-08～09（距今约 **7 周**）。⚠️ **私有仓库必须在收到官方私有评测分数之前转为公开**（官方开源要求） |
| 当前主力 | **`work/arc_w1/arc26_solver.py`**（2,536 行）：Qwen3-4B（vocab=16 网格字母表，3.634B）SFT + **LoRA TTT** + 16-token 约束束搜索 DFS；`cascade` 调度分 Stage A 廉价扫描 / B 精修 / C 回填。详见 `work/arc_w1/ARC26_SOLVER_SPEC.md`（**重建版**，原件缺失） |
| 真实分数 | 神经路线：评估集 24 题子集 **1/24**；smoke 5 题 4/5。符号 v1：评估集 **0/167**、训练集 34/1076。⚠️ **v1 现在是 Stage C 的候选引擎（`--engine`），但从未成功加载过** |
| 运行约束 | Kaggle **最多 2 个并发 GPU 会话**、周配额约 30 h → 240 题拆 2 个交错分片各约 12 h；单次 `--time-budget-seconds 39000`（实测跑满 37,976 s）；T4 ×2（各 14.6 GiB），3.634B 必须 bf16 + 梯度检查点；**成功运行不发布日志**，靠 tee 到 `/kaggle/working/kernel_stdout.log` |
| 下一个动作 | ① 用 `work/arc_w1/tools/kpush.py` 建立本机「编辑→推送→读回」闭环（已重建并通过保真度校验）；② 补 `--engine` 符号引擎（Stage C 目前空转）；③ 补 13 个网格符号↔颜色对应表（`arc26-stage1` 因 torchao 不兼容未完成） |
| 阻塞项 | ❌ **无硬件/账号阻塞**：手机验证已过（GPU 可开、240 题竞赛数据可挂载）。剩余都是技术性缺口，见上「下一个动作」 |
| 已解决 | ✅ 「公开仓库」（github.com/LuoGuoQiang-web/ARC-AGI-2 私有，`kaggle.json` 不在远端）；✅ **手机验证**（GPU 与竞赛数据均可用）；✅ MattSkills 初始化；✅ 本机 git 2.55 / gh 2.100 / kaggle 包；✅ 官方数据克隆；✅ **从 Kaggle 抢救出另一台电脑的求解器 + 重建推送工具** |
| 待用户确认 | 另一台电脑的代码**是否已推到 GitHub**？本机 `git fetch` 时网络失败，无法判断；`work/arc_w1/` 目前是唯一本地副本 |
| 已花预算 | 0 / 300 RMB |
| 工作目录 | `C:\Users\LRDC07\Desktop\Kaggle` |

---

## 1. 任务与目标

**目标**：在 Kaggle 上赢得**现金 + 奖牌**。
**唯一选定赛道**：ARC Prize 2026（同一份工作三次兑现，见 §4）。
**资源**：本机 RTX 5060 Ti **8 GB**；单人；每周 10–15 小时（8 周约 100 小时）；预算 ≤300 RMB（目标实际 ≤150）；接受开源（CC0/MIT-0）。
**约束**：不生成代码直到用户明确说「开始」；一切以官方规则为准，不臆测。

---

## 2. 已核实事实（2026-09-09，来源：Kaggle API + ARC 官网）

### 2.1 开放竞赛全景

| 竞赛 slug | 类别 | 队伍数 | 奖池 | 截止（UTC） | 结论 |
|---|---|---|---|---|---|
| `arc-prize-2026-paper-track` | Featured | **176** | $450k | 2026-11-09T23:59 | **主攻** |
| `arc-prize-2026-arc-agi-2` | Featured | 1,889 | $700k | 2026-11-02T23:59 | **主攻（写作奖 + 奖牌）** |
| `arc-prize-2026-arc-agi-3` | Featured | 2,897 | $850k | 2026-11-02T23:59 | 放弃 |
| `rsna-knee-abnormality-detection` | Research | 3,377 | $77k | 2026-10-22T23:59 | 放弃 |
| `biohub-cell-tracking-during-development` | Research | 3,278 | $60k | 2026-09-29T23:59 | 放弃 |
| `pokemon-tcg-ai-battle-challenge-strategy` | Featured | 698 | $240k | 2026-09-13T23:59 | 放弃（只剩 4 天） |
| `kaggriculture` | Featured | 8,275 | $50k | 2026-09-30T23:59 | 放弃 |
| `playground-series-s6e9` | Playground | 1,284 | 仅 swag | 2026-09-30T23:59 | **奖牌保险（封顶 10h）** |
| `secom-defect-detection-challenge` | Community | 2 | 无 | 2027-03-06 | 放弃（无价值） |
| `molecular-taste-classification-2026` | Community | 10 | 无 | 2027-04-29 | 放弃 |

材料/铝合金类竞赛（Aluminium challenge、Metallurgica 2025、Steel Quality Challenge、ML for Band gap prediction、NOMAD2018、CHAMPS、Open Polymer 2025 等）**全部已关闭**，不参与本轮。

### 2.2 榜单实况（决定难度判断）

| 竞赛 | 前 5 名分数 |
|---|---|
| ARC-AGI-2 | 76.94 / 72.08 / 40.83 / 39.58 / 37.22 |
| ARC-AGI-3 | 11.04 / 8.21 / 7.63 / 7.51 / 5.96 |
| Biohub | 0.970 / 0.966 / 0.964 / 0.964 / 0.963（极度饱和） |
| Kaggriculture | 2957.3 / 2937.0 / 2924.1 / 2897.0 / 2885.1 |
| RSNA 主榜 | 榜首 0.954；**效率榜共 3,278 条**，前 10 名 0.939–0.952（**没有冷门通道**） |

### 2.3 ARC Prize 2026 奖池与规则（来源：arcprize.org/competitions/2026 及各赛道页）

- **总奖池 $2M，3 条赛道**。
- **Paper Prize $450k**：Top Paper $75k 保证（1st $50k / 2nd $20k / 3rd $5k）+ **$375k Outstanding Papers Pool**（论文评分 **>4.5/5** 即可分，**可多组获奖**）。
- **ARC-AGI-2 $700k**：
  - Progress Prizes $275k，**8 个奖位**：75 / 50 / 40 / 35 / 25 / 20 / 15 / 15（千美元）
  - **Grand Prize $275k**：按 **Solution Writeup** 评分（6 项各 0–5 取平均）
  - Bonus $150k：首个 private eval ≥85% 的方案
- **ARC-AGI-3 $850k**：Grand $700k（首个 100%）+ Top Score $75k（40/15/10/5/5）+ Milestone $75k（M1 2026-06-30、M2 2026-09-30，各 25/10/2.5）。
- **论文评分维度**：Accuracy / Universality / Progress / Theory / Completeness / Novelty。
- **关键规则（原文要点）**：
  1. 论文必须关联一份 ARC-AGI-2 或 ARC-AGI-3 的 Kaggle 代码提交；**"The code submission need not achieve a high score for the corresponding paper to be eligible."**
  2. Grand Prize 的 artifacts 需在提交截止后 **7 天内**开源并挂到官方 Solution Writeup。
  3. **评估期无网络**（不能用 GPT/Claude 等 API 推理）。
  4. **获奖必须开源**：自己写的代码用 CC0/MIT-0 等宽松许可；第三方需 Apache-2.0/GPLv3 等允许公开分享的许可。
  5. 奖项由 ARC Prize Inc. 全权裁量，可能多发或少发。
- **ARC-AGI-2 计分**：每个测试输入需预测**恰好 2 个输出**，任一完全匹配即该题得 1 分，最终取平均。
- **时间线**：2026-03-25 开赛 → **11-02 代码提交** → **11-08 论文（Kaggle 页面显示 11-09T23:59Z）** → 12-04 公布结果。

### 2.4 Kaggle 免费 GPU 事实

- **手机验证是开启 GPU/TPU/Internet 的前置条件**（`kaggle.com/settings` → Phone Verification）。
- 加速器：**GPU T4 x2**（2×16 GB，**仅 fp16，不支持 bf16**）；**P100 于 2026-09-15 下线**；TPU VM v3-8。
- 额度：GPU ≈ **30 小时/周**、TPU ≈ 20 小时/周，按周重置；**单会话上限 12 小时**；空闲约 20 分钟断开。
- **额度无法通过 API 读取**（我们实测 `KernelsService` 下 12 个候选端点全部 404），只能在 Notebook 的 **Session options → Accelerator** 面板查看。
- 目录：`/kaggle/input` 只读；`/kaggle/working` 可写、随版本保存（约 20 GB）；`/kaggle/temp` 临时。
- 离线：竞赛环境无网 → 模型权重/数据需上传为 Kaggle Dataset 后挂载；无网装不了 pip 包，只能用镜像预装的或挂载 wheel。
- 用户账号现状：`luoguoqiang`，已有 3 个 Notebook，**全部 `enableGpu=False`、`enableInternet=False`**（从未启用过加速器）。

### 2.5 可用的 Kaggle API 调用配方（供未来 agent 复用，避免重复踩坑）

鉴权：从 `%USERPROFILE%\.kaggle\kaggle.json` 取 `username`/`key`，做 `base64("user:key")` 放进 `Authorization: Basic ...`。

| 用途 | 调用 | 说明 |
|---|---|---|
| 按类别列竞赛 | `GET /api/v1/competitions/list?category=research` | 返回 teamCount / reward / deadline / evaluationMetric / id |
| 竞赛详情 | `POST /api/i/competitions.CompetitionService/GetCompetition`，body `{"competitionName":"<slug>"}` | 可拿 `reward.clarification`（奖位数）、totalTeams |
| 榜单 | `POST /api/i/competitions.LeaderboardService/GetLeaderboard`，body `{"competitionId":<id>,"page":1,"pageSize":20}` | 返回 publicLeaderboard |
| 我的 Notebook | `GET /api/v1/kernels/list?user=<user>` | 含 enableGpu / enableInternet |
| 数据集检索 | `GET /api/v1/datasets/list?search=<q>` | 可用 |
| 数据集/竞赛搜索 | `GET /api/v1/competitions/list?search=<q>` | **不可靠**（多数返回 0） |
| 额度查询 | — | **不存在**（12 个端点全 404） |
| 网页抓取 | `www.kaggle.com/*` | **被 reCAPTCHA 拦截**，不要尝试 |

### 2.6 提交契约（2026-09-11 复核，取代任何旧笔记）

来源：arcprize.org 两条赛道页 + 官方 `sample_submission.json`（真实 task id，含 2 个与 3 个 test 输入的任务）。

- 官方原文：**"Submissions must be made through the Kaggle competition as a Kaggle notebook."** → 只能推 Notebook 版本。
- 产物路径：`/kaggle/working/submission.json`。
- 结构：`{"<task_id>": [{"attempt_1": grid, "attempt_2": grid}, ...]}`；**list 长度 = 该题 test 输入个数**（1–3）；grid 为 0–9 的矩形整数数组。
- 计分：每个 test 输入必须给**恰好 2 个**输出；任一完全匹配则该题得 1 分；最终取所有 test 输出的平均。
- 评测期**无网络**；获奖必须**开源**（宽松许可）。
- **仍未核实**：「Hardware and compute limits will be announced with the competition launch」→ 这是 §11 W1 唯一残留的规则项，不得臆测。

---

## 3. 为什么是这个赛道（决策依据，勿重复推翻）

1. **奖位/队伍比最好**：Paper Track 176 队，$375k 池可多组获奖；ARC-AGI-2 另有 8 个进度奖位 + $275k 写作奖。
2. **不拼算力**：论文与写作奖**明确不要求高分**，8 GB 本机 + 免费 T4 足以支撑。
3. **与我们的强项重合**：评审给 Theory / Progress / Completeness / Novelty 的分，靠文献梳理、受控实验、清晰写作拿到，而不是靠堆 GPU。
4. 其余现金赛道的奖位/队伍比都差一个数量级以上（见 §2.1）。

---

## 4. 核心策略：一个项目，三次兑现

**项目本体**：构建一个**离线可复现的 ARC-AGI-2 求解器**，并以「先诊断、再对症下药」的结构写成 ≤6 页论文。

**兑现路径**：

| # | 路径 | 奖 | 门槛 |
|---|---|---|---|
| 1 | Paper Track | $375k 池（≥4.5/5）+ 前 3 保证 $75k | 论文质量 |
| 2 | ARC-AGI-2 Solution Writeup | $275k Grand Prize | 写作质量 |
| 3 | ARC-AGI-2 榜单 | 8 个进度奖位 + Featured 奖牌 | 基线分数越高越好 |
| 保险 | Playground S6E9（09-30） | 奖牌 + swag | 封顶 10 小时 |

**执行方法论（因 8 GB + 单干 + 100 h 而定）**：

1. **复用**：W1 选一个许可宽松（MIT/Apache-2.0/CC0）、能在 Kaggle 限制内离线跑通的开源 ARC-AGI-2 方案作基线（ARC Prize 强制获奖者开源，2024/2025 方案公开可查）。
2. **诊断**：在基线上做受控实验，定位其最主要的一类失败（感知 / 抽象 / 搜索 / 组合），标注 ≥100 个失败案例。
3. **窄贡献**：只针对该类失败做一个有理论解释的改进，用消融证明有效。
4. **论文**：「诊断 → 归因 → 方法 → 证据」结构，≤6 页。

---

## 5. 技术边界（8 GB 本机 + T4 x2 云端）

| 能力 | 可行 | 不可行 |
|---|---|---|
| 本地推理 | ≤7B 4-bit（Qwen2.5-Coder-7B、R1-Distill-Qwen-7B，约 4.7 GB，8k 上下文） | 14B+ 本地推理 |
| 本地训练 | ≤3B LoRA；≤1B 全量 | 7B 全量微调 |
| 云端训练 | 单张 T4 16 GB：7B 4-bit 推理、≤3B LoRA、fp16 + GradScaler | bf16 训练（T4 不支持） |
| 主力算力 | **符号搜索 / 执行反馈 / 测试时算力调度** | 大规模神经训练 |
| 方法形态 | 小神经先验（打分/剪枝）+ 大量符号搜索与执行验证，离线可复现、算力可调 | 依赖在线 API 的方案 |

---

## 6. 8 周排期（小时预算合计 ≈100 h）

| 周 | 日期 | 小时 | 任务 | 交付 | 门槛 |
|---|---|---|---|---|---|
| W1 | 09-09→09-16 | 12 | §11 检查清单 | 基线选定 + 额度结论 | — |
| W2 | 09-17→09-23 | 14 | 基线在 Kaggle 复现；建立失败分类；标注 ≥100 案例 | 失败分类表 | 复现 ≥公开分数 90%，否则 3 天内换基线 |
| W3 | 09-24→09-30 | 14 | 8h Playground S6E9 终版；6h 方向定稿 | 奖牌兜底 + 方向定稿 | — |
| W4 | 10-01→10-07 | 14 | 窄贡献 v1 + 消融框架；对齐 Kaggle 效率限制 | 方法 v1 + 首轮消融 | 相对基线提升 ≥5%（公开集），否则转纯诊断论文 |
| W5 | 10-08→10-14 | 14 | 消融补全 + 失败模式量化 + 理论框架 | Theory 素材 | — |
| W6 | 10-15→10-21 | 12 | 论文 v1（≤6 页）+ 开源仓库 + 复现脚本 | 论文初稿 | 骨架未完成 → 砍掉额外实验 |
| W7 | 10-22→10-28 | 12 | **冻结方法**；最终提交；论文 v2；Writeup 初稿 | 提交 + 论文 v2 | — |
| W8 | 10-29→11-08 | 8 | 缓冲；11-02 代码提交；11-08～09 论文 + Writeup；仓库开源 | 交付验收 | — |

**砍单顺序（落后时按此顺序砍）**：Playground 保险 → 额外消融 → 论文图表数量 → 基线升级尝试。

---

## 7. 明确不做（Do-Not 清单）

- 不做 ARC-AGI-3、RSNA、Biohub、Kaggriculture、Pokémon。
- 不做任何 >3B 的本地训练；不做 7B 全量微调。
- 不做复杂多模型集成工程；不追榜单前十。
- 不写超过 6 页的论文；不写与主线无关的教程型 Notebook。
- 不使用在线 API 做最终推理（竞赛环境无网，且违规）。
- 不使用许可不明或非宽松许可的第三方代码。
- 不在未核实规则前假设「提交运行不耗额度」。

---

## 8. 预算与记账

| 项目 | 上限 | 已花 | 说明 |
|---|---|---|---|
| LLM API（合成数据 / 候选程序 / 润色） | 150 | 0 | 优先本机开源模型 |
| 云 GPU 应急（约 2 元/小时） | 100 | 0 | 仅当 Kaggle 额度不足 |
| 预留 | 50 | 0 | — |

记账规则：每笔支出当天记入本表；单笔 >50 元需用户确认；总支出触达 200 元即暂停并复盘。

---

## 9. 风险登记

| 风险 | 概率 | 影响 | 对策 |
|---|---|---|---|
| 8 GB 撑不起预期方法 | 中 | 进度 | 方法形态已按 8 GB 设计（搜索为主） |
| 提交运行计入额度，算力不够 | 中 | 实验次数 | W1 实测；实验「≤1 h/次、≤4 次/轮」 |
| 论文 <4.5 分 | 中 | 现金 | 诊断型结构 + 两次评审机会 |
| 窄贡献提升不足 | 中高 | 论文素材 | W4 门槛 → 转纯诊断论文 |
| 单干时间不够 | 中 | 交付 | 100 h 预算 + 砍单顺序 + W8 缓冲 |
| 忘记开源 / 许可不合规 | 低 | **致命（失去资格）** | 仓库从 W1 建；W8 复查许可清单 |
| 手机验证未完成 | 中 | 无法用 GPU | W1 第一件事 |
| 日期误算 | 低 | 错过截止 | 每次会话开头核对 §0 与当前日期 |

---

## 10. 预期结果（主观判断，非承诺）

| 结果 | 估计 |
|---|---|
| 论文进 $375k 池（≥4.5/5） | 15–30% |
| 论文进前 3（保证 $75k） | 5–15% |
| ARC-AGI-2 进度奖（前 8） | <3% |
| ARC-AGI-2 榜单奖牌 | 20–40% |
| Playground S6E9 奖牌 | 50–70% |
| 至少拿到现金或奖牌 | 明显高于任何单赛道 |

---

## 11. W1 检查清单（下一个 agent 从这里开始）

1. **规则核实**（读 ARC-AGI-2 与 Paper Track 的 Kaggle 页面 + arcprize.org）：
   - 提交环境的算力/运行时限、是否可用预训练模型、是否允许外部数据、每日提交次数
   - Solution Writeup 的提交入口与要求
   - 记录为 `TODO-W1` 项，逐条确认，不得跳过
2. **额度实测**：记录提交前额度 → 跑一次最简提交 → 记录提交后额度 → 记录单次提交墙钟时间。
3. **基线调研**：从 ARC Prize 2025 结果博客、Kaggle Solution Writeups、GitHub 检索 3–5 个开源 ARC-AGI-2 方案，按以下标准打分：
   - 许可是否宽松（MIT/Apache-2.0/CC0）
   - 能否在 Kaggle 限制内离线跑通
   - 公开分数
   - 代码复杂度（单人可维护）
   - 显存需求（≤16 GB 单卡）
   - 运行时长（≤12 h）
   - **不臆造方案名称，必须实际检索并核实**
4. **最简提交**：跑通一条端到端提交，确认榜单资格（奖牌前提）。
5. **输出 W1 报告**：基线选择 + 额度结论 + 未决问题 + W2 实验计划（≤1 页）。

---

## 12. 人机分工

| 事项 | Agent | 用户 |
|---|---|---|
| 文献/方案/实验设计 | ✅ | 拍板 |
| 代码实现（用户说「开始」后） | ✅ | 机器可用性 |
| 论文撰写与润色 | ✅ | 终审 |
| Kaggle 手机验证、开 GPU、提交、看额度 | 提供步骤 | ✅ |
| 开源仓库发布与署名 | 准备内容 | ✅ 发布 |
| 付款 | 计划与监控 | ✅ |

**用户必须提供的信息**：① Session options 里剩余 GPU 小时数；② 是否可建公开 GitHub 仓库；③ 是否已完成手机验证。

---

## 13. 文件清单

| 文件 | 作用 |
|---|---|
| `ARC_PRIZE_2026_MASTER_BRIEF.md` | **本文件**，唯一权威来源 |
| `arc_prize_v1/arc_prize_v1.py` | **v1 引擎**（ARC-AGI-2 求解器 + 运行/备份/续跑/自评外壳，纯标准库、离线、CPU） |
| `arc_prize_v1/ARC_PRIZE_2026_v1.ipynb` | **开箱即用的 Kaggle Notebook**（引擎内嵌，上传即用；由 `tools/build_notebook.py` 生成） |
| `arc_prize_v1/BASELINE_V1.md` | **真实基线报告**（评估集 0.00%、训练集 3.16% + 失败诊断） |
| `arc_prize_v1/LOO_PROBE.md` | 留一法探针：采纳精度 64.19%，证明瓶颈是表达力而非验证 |
| `arc_prize_v1/KAGGLE_CELLS.md` | Kaggle 操作手册（上传法 + 手动粘贴法 + 速查 + 故障排查） |
| `arc_prize_v1/README.md` | v1 设计说明、测试证据、局限与 v2 方向 |
| `arc_prize_v1/tests/` | `selftest`(36 项) / `eval_local`(真实测量) / `loo_probe`(留一法) / `make_fixture` / `perf_probe` / `run_notebook_dryrun` |
| `arc_prize_v1/tools/build_notebook.py` | 从引擎生成 Notebook，并校验内嵌源码逐字节一致（`--check` 检测过期） |
| `arc_prize_v1/evidence/` | **结论依据**：评估集 167 行 / 训练集 1076 行逐题诊断 CSV + 列说明 |
| `AGENTS.md` | 仓库级 agent 指引 + MattSkills 的 `## Agent skills` 配置块 |
| `docs/agents/` | MattSkills 初始化产物：`issue-tracker.md` / `triage-labels.md` / `domain.md` |
| `.gitignore` | 排除凭据、第三方数据、运行产物（**`kaggle.json` 在其中，绝不可入库**） |
| `NOTE.md` | Paper Track 数据页原文（证实「无数据集」，代码须落在 ARC-AGI-2） |
| `arc_prize_2026_8week_plan.md` | 历史版本（v2），已被本文件取代 |
| `kaggle_winning_odds_2026-09.md` | 全赛道「拿下把握」评估 |
| `kaggle_materials_competitions.md` | 材料/铝合金竞赛调研（背景资料） |
| `rsna_efficiency_lb.csv` | RSNA 效率榜原始数据（3,278 行，佐证判断） |
| `kaggle.json` | Kaggle 凭据（**不得外泄、不得提交到仓库**；本机存在但被 git 忽略） |
| `ARC-AGI-2-main/` | 官方数据克隆（第三方，被 git 忽略；换机器用 `git clone --depth 1` 重建） |

**远端**：`https://github.com/LuoGuoQiang-web/ARC-AGI-2`（私有）｜仓库 28 个文件｜换机器流程见 §16

---

## 14. 术语表

- **ARC-AGI-2**：静态抽象推理基准，每题给若干输入输出对，要求预测测试网格；每题需提交 2 个候选输出。
- **Solution Writeup**：Kaggle 竞赛页面上的「方案写作」入口，ARC-AGI-2 的 $275k Grand Prize 依此评分。
- **DSL**：领域特定语言，用于把网格变换表示成可搜索的程序空间。
- **测试时训练/自适应（TTT）**：在推理阶段用测试样本做少量自适应更新。
- **效率限制（efficiency limits）**：ARC-AGI-2 提交运行的算力/时长上限，**具体数值待 W1 核实**。
- **奖牌（medal）**：Kaggle Featured/Research/Playground 等类别按排名百分位授予的金/银/铜牌，计入账号进度。
- **Community 竞赛**：社区自办，通常不记积分/奖牌（页面上常见 "does not award ranking points"）。

---

## 15. 会话恢复流程（新 agent 必读）

1. 读 §0 当前状态，核对当前日期与 §2.3 的硬截止。
2. 读 §2 全部事实——**不要重新联网查证已核实项**，只在发现矛盾时复查。
3. 确认阻塞项（§0）是否解除；未解除则只推进不依赖 GPU 的工作（规则验证、文献调研、方案设计、本地测量）。
4. 按 §11 执行当前周任务，完成后更新 §0 与对应周的状态。
5. 每次会话结束前更新：当前阶段、已花预算、下一个动作、阻塞项。

---

## 16. 换机器 / 新环境启动流程（2026-09-11 实测）

代码 100% 可移植：仓库已跟踪文件里**零处** `dsh-tools` 依赖，引擎路径全是环境探测
（`/kaggle/working` 存在则用它，否则相对路径 `./arcprize_runtime`；数据根为相对 `./data`、`.`）。

**跟着 Git 走**：引擎、Notebook、6 个测试、工具、全部文档、2 份证据 CSV、`AGENTS.md`、MattSkills 配置。
**跟着账号走（云端）**：GitHub 仓库、Kaggle Notebook 及其已保存版本。
**必须在新机器重建**：

```powershell
npm install -g @deepseek-ai/dsh                       # 1. harness 本体
#    2. 装 git + GitHub CLI，然后 gh auth login
#       （winget 的 github.com/releases 源可能被墙；本次实测用清华镜像装 git、
#        用 api.github.com 资产通道装 gh，见 §0 已解决栏）
git clone https://github.com/LuoGuoQiang-web/ARC-AGI-2.git Kaggle   # 3. 代码
cd Kaggle
git clone --depth 1 https://github.com/arcprize/ARC-AGI-2.git ARC-AGI-2-main  # 4. 官方数据（被 git 忽略）
python arc_prize_v1/tests/selftest.py                 # 5. 应输出 36/36
python arc_prize_v1/tests/eval_local.py               # 6. 应复现 0.00%（评估集）
python arc_prize_v1/tests/eval_local.py --dataset training   #    应复现 3.16%（训练集）
```

**不需要迁移**：本机 `kaggle.json`（凭据，靠 Kaggle 账号重下）、运行历史 `arcprize_runtime/`
（全量重跑评估集 6 秒、训练集 31 秒）、本次对话上下文（§0 就是为此写的）。
DSH 历史会话存在 `C:\Users\LRDC07\.dsh`，理论上可整目录拷贝到新机同路径，**跨机恢复未实测**，不要依赖。

### 16.1 ⚠️ 本机网络：`github.com:443` 会被阻断 —— 用 API 推送

2026-09-14 实测：**只有 `github.com` 这一台主机不通**（connection reset / timeout），
而 `api.github.com`、`codeload.github.com`、`raw.githubusercontent.com`、`registry.npmjs.org`
全部 200。因此：

- `git push` / `git fetch` / `git clone https://github.com/...` **会失败**（时通时断，同一天内两种都出现过）；
- **替代方案**：用仓库自带的 `tools/gh_api_push.py` 通过 GitHub REST API 推送：
  ```powershell
  $env:GH_TOKEN = (gh auth token)
  python tools/gh_api_push.py --repo LuoGuoQiang-web/ARC-AGI-2 --branch main
  ```
  它会上传本地 HEAD 的**全部文件**构建 tree（不依赖远端 tree），并**强制校验
  `GitHub tree == 本地 tree`**，不一致就拒绝写入 —— 所以内容零损失。
- **副作用**：API 产生的提交 SHA 与本地不同（内容相同、元数据不同）。
  网络恢复后用一条命令调和：
  ```bash
  git fetch origin && git reset --hard origin/main
  ```

### 16.2 安全规则（2026-09-14 事件后确立）

探测脚本**禁止** `print(os.environ)` —— 曾导致 4 个 kernel 的日志泄露
`KAGGLE_DATA_PROXY_TOKEN` / `KAGGLE_USER_SECRETS_TOKEN`（该 4 个 kernel 已删除）。
只打印键名与值长度。推送前确认 `is_private=True`（`tools/kpush.py` 默认私有）。
细节见 `work/arc_w1/README.md` §7。
