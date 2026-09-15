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
| 真实分数 | 神经路线：评估集 24 题子集 **1/24**；smoke 5 题 4/5。符号 v1：评估集 **0/167**、训练集 34/1076。**线上**：首次提交（`arc26-submit-full`，2026-09-14 04:01）状态 **PENDING**，结果待观察；上一次 v1 提交因**格式错误**得 0（见 §2.6.3） |
| 运行约束 | Kaggle **最多 2 个并发 GPU 会话**；**周配额 30 h**（§2.4 已更正）、单次 `--time-budget-seconds 39000`（实测跑满 37,963 s）；T4 ×2（各 14.6 GiB），3.634B 必须 bf16 + 梯度检查点；**成功运行不发布日志**，靠 tee 到 `/kaggle/working/kernel_stdout.log`。**每日提交次数有限**，比 GPU 额度更稀缺 |
| 下一个动作 | ① 等 `arc26-submit-full` 的线上分数（决定提交管线是否真的通了）；② 等 `arc26-exp-prob10` / `arc26-exp-branch4` 两个搜索宽度实验（各 8 题，与 base 配对）；③ **写论文**（真正的奖金来源，0 GPU 成本） |
| 阻塞项 | ❌ **无硬件/账号阻塞**。真正的瓶颈已实测锁定为**生成**而非选择：`pool_recall = 0.0`、`selection_headroom = 0.0`（两次独立测量：24 题与 8 题评估子集）→ **任何重排序/选择改进都不可能提分** |
| 已解决 | ✅ 「公开仓库」（github.com/LuoGuoQiang-web/ARC-AGI-2 私有，`kaggle.json` 不在远端）；✅ **手机验证**；✅ MattSkills 初始化；✅ 本机 git / gh / kaggle 包；✅ 官方数据克隆；✅ **从 Kaggle 抢救出另一台电脑的求解器 + 重建推送工具**；✅ **符号引擎接入 Stage C**（`--engine`，11/11 本地测试通过）；✅ **分片合并 + 实验脚手架**；✅ **首次跑通竞赛提交动作**（§2.6.1） |
| 待用户确认 | 额度事实已更正（30 h/周期，非 6 h）——**是否需要重排全局计划**？当前仅剩 5.2 h，09-19 刷新后有 30 h |
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

⚠️ **2026-09-14 重新实测并更正。** 本节此前两行**都是过期快照**，而它们是难度判断的依据，
所以两个结论都曾被错误的事实支撑。教训：**榜单数字必须现取，不能沿用笔记**（用
`kaggle competitions leaderboard -c <slug> -s`）。

| 竞赛 | 榜首 | 前 5 名分数 | 备注 |
|---|---|---|---|
| **ARC-AGI-2** | 76.94 | 76.94 / 72.08 / 48.89 / 48.61 / 37.22 | **第 8 名 = 35.42**（Progress Prize 最后一个奖位）。此前误记为 `40.83 / 39.58` |
| **ARC-AGI-3** | 18.81 | 18.81 / 8.68 / 8.44 / 8.40 / 8.21 | **3036 队**。此前误记为 `11.04 / 8.21 / …`，而 `11.04` 在榜上根本不存在 |
| Biohub | 0.970 | 0.970 / 0.966 / 0.964 / 0.964 / 0.963 | 极度饱和 |
| Kaggriculture | 2957.3 | 2957.3 / 2937.0 / 2924.1 / 2897.0 / 2885.1 | — |
| RSNA 主榜 | 0.954 | 效率榜共 3,278 条，前 10 名 0.939–0.952 | **没有冷门通道** |

#### 2.2.1 ARC-AGI-2 Progress Prize 不可达（确认）

8 个奖位的门槛是 **第 8 名 = 35.42%**。我们实测约 0–4%，且公开同规模方案的上限也远低于此
（2025 冠军 NVARC 用 Qwen3-4B + LoRA + TRM 也只到 24.03%）。
⇒ **Progress Prize 与 $150k Bonus（需 85%）均判定为不可达**，不再分配任何额度给"冲榜"。

#### 2.2.2 ARC-AGI-3 判定为 **NO-GO**（2026-09-14 调研，含证据）

完整报告：`arc_agi_3_feasibility_2026-09-14.md`；原始榜单 3036 行：`lb/lb.csv`。

- **交互式基准**：64×64 网格、取值 0–15、`RESET` + `ACTION1`–`ACTION7`，按
  **完成度 × 动作效率**计分，不是静态 grid→grid。逐回合实时评测，notebook 离线驱动本地模型。
- **没有"白捡"的基线**（这条决定了结论）：官方 starter **就是随机智能体，自评 0.0**；
  3036 队里**只有 22 队 ≥5、5 队 ≥8**；中位数 0.29。独立文献
  （`arXiv:2605.25931`）报告随机基线 **0.0000**，其作者自己的 BFS 参赛方案 **0.30**。
- **M1 已于 2026-07-06 颁出**（$37.5k）；**M2 是最后一次，2026-09-30 截止，且是排名制
  （前三名）而非阈值制** → 真正的门槛是 09-30 当天第 3 名的分数，今天第 3 名 = **8.44**。
  榜首十周内从 ~1 涨到 18.81；**六月第 3 名的 0.86 现在排到约第 500 名**。
- **对我们的判定**：M2 16 天内 **~0–1%**；Top Score（11-02 前五）**~1–3%**。
  16 天能做出的大概是 0.3–1.5% 的真实提交，**不获奖**。
- **另有一条关键事实**：计分集是 **55 个隐藏游戏**，而竞赛只给 **25 个公开**环境
  → **本地分数不是榜单的代理指标**，唯一有信息量的测量是真提交。
- **⚠️ 未核实（花钱前需人工在两分钟内确认）**：M2/Top Score 是按公开榜还是私有重跑裁定；
  单回合步数/时间上限。Kaggle 规则页是 JS 渲染，内部 API 探测返回 403/404。
- **若将来要重开这条路**，预先定好判定规则的实验是（~1 GPU 小时 + 1 次提交）：fork 公开的
  Duck harness notebook 跑一次自己的分。**≥6 才重开问题；≤2（预期）则彻底关闭**，
  把每周 30 GPU 小时全部投回 ARC-AGI-2。

### 2.3 ARC Prize 2026 奖池与规则（来源：arcprize.org/competitions/2026 及各赛道页）

- **总奖池 $2M，3 条赛道**。
- **Paper Prize $450k**：Top Paper $75k 保证（1st $50k / 2nd $20k / 3rd $5k）+ **$375k Outstanding Papers Pool**（论文评分 **>4.5/5** 即可分，**可多组获奖**）。
- **ARC-AGI-2 $700k**：
  - Progress Prizes $275k，**8 个奖位**：75 / 50 / 40 / 35 / 25 / 20 / 15 / 15（千美元）
  - **Grand Prize $275k**：按 **Solution Writeup** 评分（6 项各 0–5 取平均）
  - Bonus $150k：首个 private eval ≥85% 的方案
- **ARC-AGI-3 $850k**：Grand $700k（首个 100%）+ Top Score $75k（40/15/10/5/5）+ Milestone $75k（M1 2026-06-30、M2 2026-09-30，各 25/10/2.5）。
- **论文评分维度**：Accuracy / Universality / Progress / Theory / Completeness / Novelty。

#### 2.3.1 ⭐ $275k Grand Prize **不挂钩排行榜名次**——但**并非不看成绩**（2026-09-14 官网原文核实）

来源：<https://arcprize.org/competitions/2026/arc-agi-2>，原文逐字：

> **ARC-AGI-2 Grand Prize: $275,000** — "The Grand Prize will be awarded to the
> **highest scoring Solution Writeup** based on the below criteria. All artifacts should be
> open sourced and attached to an official competition Solution Writeup within seven days
> of the competition's submission deadline to be considered eligible.
> Submissions for the Grand Prize are evaluated equally across the following six criteria.
> Each criterion is scored on a scale from 0 (lowest) to 5 (highest), with the final score
> calculated as the average of all six."

**这不是"给榜首的附加奖"，而是独立的写作奖。** 加上 Paper Track 的 $450k，两项共用同一套 6 维评分表。

#### 2.3.1.1 🔒 **硬约束：Accuracy 必须 ≥3/5，否则算术上不可能过 4.5 线**

> ⚠️ **本节修正了本文档先前的错误结论。** 此前写的是"$725k 完全由写作质量决定"。
> **这是错的。** 正确的区别是：**Grand Prize 不挂钩"名次"，但成绩仍以 Accuracy 一项参与评分。**
> "名次不设门槛" ≠ "成绩不参与评分"。

评分 = 6 项各 0–5 分，**取算术平均**，Outstanding Papers Pool 要求 **>4.5**。设 Accuracy 为 $a$，
其余五项最高各 5 分：

$$\text{平均} \le \frac{a + 25}{6}$$

| Accuracy | 其余五项全满分时 | 能否 >4.5 |
|---|---|---|
| 1 | 26/6 = **4.33** | ❌ |
| 2 | 27/6 = **4.50** | ❌（要求**严格大于**） |
| 3 | 28/6 = **4.67** | ✅ |

**⇒ Accuracy ≥3/5 是拿到 Outstanding Papers Pool 的*必要条件*。**
官方明文：*"The submission's score will be used in the rubric's 'accuracy' category."*

**对我们的直接后果**（2026-09-14 评估）：求解器评估集约 0–4% → Accuracy 0–1 分
→ **即便其余五项全部满分，论文上限 4.33，够不到 4.5。**

**⇒ 战略结论（取代此前的"钱在写作上"）**：

- 提分**不是**"锦上添花的 1/6"，而是**拿到任何奖金的必要前提**；
- 只有 Top Paper 前三（$50k/$20k/$5k）是**排名制**，不受上述算术约束——但要在强队中进前三；
- **唯一可行的路径**：修好 TTT（见 §2.3.2 末），让 Accuracy 有机会够到 2–3。
  这同时把论文从"发现静默失效"升级为"发现 + 修复 + 验证"，
  Progress / Theory / Novelty 三项会一起上移。

#### 2.3.1.2 ⭐ 战略决定（用户指令，2026-09-15）：**只走论文路，目标锁定 Top Paper**

用户决定：**其余路线不再考虑，全部精力投入论文。**

**但「专注论文」内部必须分清两个目标，可行性差一个数量级：**

| 目标 | 门槛 | 专注论文能否解锁 |
|---|---|---|
| Outstanding Papers Pool **$375k**（瓜分） | 评分 **>4.5** | ❌ **不能。** 上面那道算术与投入多少写作精力无关：其余五项全满分、Accuracy=1 时上限仍是 **4.33** |
| **Top Paper $75k**（50/20/5） | **跨赛道排名前三** | ✅ **真正的目标。** 排名制**没有算术地板**，低分不构成取消资格 |

**⇒ 不要为 Pool 做任何规划；它只在 Accuracy 提升后才重新打开。**

**同时明确放弃（记录理由，勿反复推翻）：**

- **Progress Prizes（$275k，8 席）**：门槛第 8 名 **35.42%**，而**完美复现 2025 夺冠配方的落点是
  28–33%**（独立复现者对齐六个头部超参仍低 5 分；作者自称该 notebook "near-optimized，几乎没有
  低垂果实"）。起点 0.42%，7 周内不可能跨越。
- **Innovation Prize（$275k）**：单一赢家，而 76.94% 的队伍同样会交 writeup。**不作为目标**；
  但它与 Paper Track **共用同一份内容、零边际成本**，属于顺手投递而非需要投入的路线。
- **ARC-AGI-3**：已按 §2.2.2 判定 NO-GO。

**期望值诚实估计**：Top Paper 前三 **≈2–3%**，其余全部 ≈0。
**成本不是约束**（预算 300 RMB 基本未花、GPU 免费）；真正约束是**时间与单人产能**：
约 7 周 × 每周 10–15 小时。

#### 2.3.1.3 论文路线下**仍然要做**的两件事（按杠杆排序，勿视为跑题）

Accuracy 仍占评分表 **1/6**，且 Top Paper 是排名制——**每一项都算分**。因此：

1. **验证修复（最高杠杆，1 次运行）**——把论文从"我们发现静默失效"升级为
   "**我们发现、修复并测量了它**"。这同时抬升当前最弱且可改的 **Progress**，并带动
   **Theory** / **Completeness** / **Accuracy**。
   判据：`report.ttt.tasks_executed` 是否从 **16/102** 变为接近全中，pool recall 是否离开 0。
2. **把 Universality 从"断言"变成"实测"（零 GPU）**——论文声称该失效模式适用于*任何带可选
   学习式精修阶段的管线*，但这目前是**主张**而非证据。用 `tools/adaptation_audit.py` 在一个
   最小的第二套管线上复现同一失效，即可把 Universality 从 2–3 抬到 4–5。

#### 2.3.2 新颖性边界（2026-09-14 文献核查后确立，**不得再重复主张**）

我们曾打算把 **oracle pool recall** 当作论文的核心贡献。**这个主张是错的，已放弃。**
核查（每条 URL 均实际抓取）结论：

| 我们想主张的 | 真实状态 | 必须引用 |
|---|---|---|
| 「真值是否在候选池中」= 选择上限 | **不是我们的**：ARChitects 的 "coverage" 曲线原文就写着 *"provides an upper bound for the performance of the selection algorithms"* | ARChitects, `arXiv:2505.07859` Fig.4 |
| Sample+Oracle 式上限 | 已有 | Li et al., `arXiv:2411.02272` Fig.8 |
| 「生成受限而非选择受限」 | **已有**：Moghe & Chin 把它做成了论文主贡献 | `arXiv:2607.06764` |
| 公开 test 结构性不具代表性（多输入占比） | **近似复现**：Habr 分析给出 6.9% vs 40.8%，我们是 7.1% vs 40.8% | Habr `habr.com/ru/articles/1071730/` |
| 公开 test 与训练集逐字节相同 / 与 eval 零重合 | **可能原创，但只能声明"检索未穷尽"**，不得主张优先权 | （未找到先例） |
| **TTT 静默失效被掩蔽** | ✅ **唯一可主张的新颖点** | 内存受限的 TTT-for-ARC 有大量先例（P100 16GB、L4×4、rank-32 LoRA batch=1），但**"被掩蔽的失效"无人报告** |

**定位**：NVARC（2025 冠军）用的就是 **Qwen3-4B + LoRA**，ARChitects 用 8B + D4 增强 + 阈值 DFS。
**我们的系统是既有配方的小规模实例，不是新求解器**——论文必须明说，不能暗示新架构。

**因此论文的主轴已改为**：*"一种被静默掩蔽的适配失效"*。实测数据：

- 240 题提交运行中，**Stage B 收到 102 题，只有 16 题真正执行了优化步**；
- 整轮 **共 21 个优化步**，而 Stage B 烧掉 **7,266 秒**（≈346 秒/步）；
- **84 道一步没跑的题被系统记成 `B_ttt_no_gain`** ——把自己的空转当成了"试过但没用"的证据。
- 两种掩蔽路径都要写：(a) 无熔断 → 每题独立静默失败（16 次 OOM，86 题未适配）；(b) 有熔断
  → 一次失败全局禁用（这是"合理"的工程决定，却把单题失败升级成整轮失败）。

**给未来的 agent**：这条发现来自 `ttt_steps` 这个计数器。**任何人做 TTT 都应该报告
"实际执行的优化步数"，而不是配置里打算跑的步数。** 这是论文 §5.2 的 safeguard。

**下一步的杠杆也随之确定**：把 TTT 做成内存可行（分批/分片喂入，而不是一次materialize
8192-token 激活），并把 OOM 降级为**每题独立**的重试。⚠️ **论文里这条写的是"已识别、未验证"**
——不要在没有实验前把它改写成结论。

**根因已在代码中定位（2026-09-14）**：`build_ttt_sequences()` 把**整套演示对打包成一条序列**
（对第 k 个增强：`fmt_train(其余全部演示, 目标输入) + fmt_reply(目标输出)`，从左侧截断到
`max_seq_length=8192`）。于是**一个优化步 = 对 3.63B 模型做一次 8192 token 的前向+反向**，
还要和 Stage A 的残存 KV 缓存抢 14.56 GiB 的 T4。而 `--aug-train 2` 只产生**2 条**这样的序列
——**整个适配预算就是 2 步**。修法：把整包序列**切块**（每步只回传一小段，注意力显存随长度
平方下降），并把 OOM 降级为**每题独立**。这同时治两个病：装得下，且步数从 2 涨到几十。
- **关键规则（原文要点）**：
  1. 论文必须关联一份 ARC-AGI-2 或 ARC-AGI-3 的 Kaggle 代码提交；**"The code submission need not achieve a high score for the corresponding paper to be eligible."**
  2. Grand Prize 的 artifacts 需在提交截止后 **7 天内**开源并挂到官方 Solution Writeup。
  3. **评估期无网络**（不能用 GPT/Claude 等 API 推理）。
  4. **获奖必须开源**：自己写的代码用 CC0/MIT-0 等宽松许可；第三方需 Apache-2.0/GPLv3 等允许公开分享的许可。
  5. 奖项由 ARC Prize Inc. 全权裁量，可能多发或少发。
- **ARC-AGI-2 计分**：每个测试输入需预测**恰好 2 个输出**，任一完全匹配即该题得 1 分，最终取平均。
- **时间线**：2026-03-25 开赛 → **11-02 代码提交** → **11-08 论文（Kaggle 页面显示 11-09T23:59Z）** → 12-04 公布结果。

### 2.4 Kaggle 免费 GPU 事实（**2026-09-14 全面修订**）

- **手机验证是开启 GPU/TPU/Internet 的前置条件**（`kaggle.com/settings` → Phone Verification）。✅ 本账号已通过。
- 加速器：**必须显式指定 `machine_shape`，且正确值是 `NvidiaL4`**（2026-09-14 实测更正，
  此前写 `NvidiaTeslaT4` 是**错的**）。
  - **`NvidiaL4` = 4× NVIDIA L4，每张 22.03 GiB，共 88 GiB，compute capability 8.9。**
    竞赛官方页面《Upgraded Accelerators》(2026-04-07) 原文：*"This competition has access to
    Kaggle's pool of powerful new L4x4 machines! These machines offer 96GB of GPU memory."*
    额度按 **2 倍**速率扣（不是 4 倍），所以每周 30 h 额度 ≈ 15 h 墙钟。
  - ⚠️ **踩坑 1**：字符串是 **`NvidiaL4`**，**不是 `NvidiaL4x4`**。写 `NvidiaL4x4` 会被
    **静默降级成 Tesla P100**（sm_60），而 P100 上 torch 2.10+cu128 每个 CUDA 算子都失败。
  - ⚠️ **踩坑 2**：只设 `enable_gpu=true` 得到通用 `Gpu` 形状，同样落到 P100。
  - ⚠️ **踩坑 3**：`assert_gpu_compatible()` 原来用 `torch.cuda.get_arch_list()` 做**集合判定**，
    而该列表**不含 `sm_89`** → **它把 L4 判成不可用**，等于亲手拒绝了本竞赛能拿到的最好硬件。
    L4 实测可用（分配、fp32 matmul、autograd、bf16 autocast、8192-token 前向全部通过）——
    torch 自己的闸门是**区间**判定（7.0–12.0），sm_89 在区间内，走 PTX JIT。
    **已改为跑一个真实算子来判定，而不是读清单。**
  - **代价**：整场战役此前一直跑在 **1× Tesla T4（14.56 GiB）** 上，而 88 GiB 一直可用。
    论文里"TTT 在 14.56 GiB 上装不下"的结论必须改写成**配置失误**，不是任务固有上限。
- 额度（**2026-09-14T03:47Z 用 `subprocess kaggle quota` + 类型化字段二次核对**）：
  - **GPU `totalTimeAllowed = 108,000s` = 30.0 小时/周期**（`minimumTimeAllowed` 同为 108,000s，
    `isPayToScaleEnabled=false`）
  - 已用 `timeUsed = 88,918.69s` = **24.70 小时（82%）；本周期仅剩 5.30 小时**
  - TPU `totalTimeAllowed = 72,000s` = 20 小时，**完全未使用**
  - 刷新时间 **2026-09-19T00:00Z**（约每周），记账精确到秒且实时
  - ⚠️ **踩坑记录**：此前版本的本节曾写成 "`totalTimeAllowed = 21,600s` = 6 小时、已超额 3.9 倍"，
    这是**错的**，并已据此错误地重排过额度计划。错误来源是直接 `repr()` 打印 SDK 返回的
    protobuf 对象（Duration 字段被拼成 `"2490.200591.0s"` 这类畸形串）。**正确读法**：
    `q.gpu_quota.total_time_allowed.total_seconds()`，或直接跑 CLI `kaggle quota`。
    教训：跨进程边界读数值，一律走 CLI 或类型化字段，不要读 `repr()`。
- **额度可以通过 API 读取**（此前记为"读不到"，已推翻）：`KaggleApi().quota_view()`，亦见 CLI `kaggle quota`。
- 实测成本：**完整 240 题提交 = 37,963s（10.5h）**，其中 Stage A 30,600s、Stage B 7,266s；
  模型加载 84–140s/run。→ **单周期 30h 容得下一次完整提交**；但若该周期已消耗大半，
  仍必须分片 + 合并（工具见 `work/arc_w1/tools/merge_shards.py`）。
  分配方案与真实成本表见 `work/arc_w1/QUOTA_PLAN.md`。
- 单会话上限 12 小时；空闲约 20 分钟断开。
- 目录：`/kaggle/input` 只读（竞赛数据实际挂在 `/kaggle/input/competitions/<slug>/`，引擎会自动定位）；
  `/kaggle/working` 可写、随版本保存；`/kaggle/temp` 临时。
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

### 2.6 提交契约（2026-09-11 复核，**2026-09-14 增补实战事实**）

来源：arcprize.org 两条赛道页 + 官方 `sample_submission.json`（真实 task id，含 2 个与 3 个 test 输入的任务）。

- 官方原文：**"Submissions must be made through the Kaggle competition as a Kaggle notebook."** → 只能推 Notebook 版本。
- 产物路径：`/kaggle/working/submission.json`。
- 结构：`{"<task_id>": [{"attempt_1": grid, "attempt_2": grid}, ...]}`；**list 长度 = 该题 test 输入个数**（1–3）；grid 为 0–9 的矩形整数数组。
- 计分：每个 test 输入必须给**恰好 2 个**输出；任一完全匹配则该题得 1 分；最终取所有 test 输出的平均。
- 评测期**无网络**；获奖必须**开源**（宽松许可）。
- **仍未核实**：「Hardware and compute limits will be announced with the competition launch」→ 这是 §11 W1 唯一残留的规则项，不得臆测。

#### 2.6.1 提交的**实际操作方式**（2026-09-14 首次跑通）

这是 **code competition**：不能只上传一个 `submission.json` 就完事。可用的 CLI 形式是
**三者必须同时给**：

```bash
kaggle competitions submit -c arc-prize-2026-arc-agi-2 \
    -k <owner>/<kernel-slug> -f submission.json -v <version>
```

- 只给 `-k` 会报错：`Code competition submissions require both the output file name and
  the version number`。
- `-f` 这里填的是**内核产出的输出文件名**（`submission.json`），不是本地路径。
- 提交后 Kaggle 会**在隐藏测试集上重跑该 notebook**，此时才真正评分；状态先 `PENDING`。
- 重跑**不占用我们的 GPU 额度**（提交后额度计数纹丝不动，已核对）。
- **每日提交次数有限**：用掉一次后显示 `0 submissions remaining today`。→
  提交机会比 GPU 额度更稀缺，**不要拿它做格式试探**。

#### 2.6.2 公开测试集是**训练集**（泄漏，与本地自评分直接相关）

来源：Kaggle 公开 notebook《ARC-AGI-2: A Submission Starter and a Trap》（逐条 md5 验证过）。

- Data 页说公开的 `arc-agi_test_challenges.json` 是"取自 evaluation challenges"的占位文件。
  **实测相反**：其 240 个 task **全部**与训练集同 id 任务**逐字节相同**，答案全在
  `arc-agi_training_solutions.json` 里；与 evaluation split 的**内容重合度为 0**。
- ⇒ **任何在公开 test 文件上量出来的分数都是训练集分数，没有意义。** 本地验证必须用
  **evaluation split**（有答案、无重合、且更难）。
- 公开 test 文件只适合**验证提交管线的形状**（能不能产出合法 JSON）。
- 另一个致命形状差异：**公开 test 文件只有 7.1% 的题需要 >1 个 test 输入；evaluation split 是
  40.8%**。→ 写"一题一个输出"的构造器**本地全过、线上一定畸形**。这正是 §2.6.3 的失败原因。
  必须 `for ti in task["test"]` 逐个产出条目，顺序与输入一致。

#### 2.6.3 已发生的真实失败（必须记住，别再犯）

| 提交 | 时间 | 结果 |
|---|---|---|
| `arc-prize-2026-arc-agi-2-v1-diagnostic-engine` | 2026-09-13 05:17 | **COMPLETE 但 errorDescription = "Your notebook generated a submission file with incorrect format."**，分数栏空白、`userRank = 0` |
| `arc26-submit-full` | 2026-09-14 04:01 | **PENDING**（首次跑通提交动作；结果待观察） |

- 排行榜是**公开可见**的（2026-09-14 榜首 76.94，参赛 2004 队），所以"没有分数"就是**0 分**，
  不是"不公开"。
- 教训：**提交前必须用结构校验器自检**（`validate_submission`：task id 集合、每题条目数 ==
  test 输入数、grid 非空/非锯齿/≤30×30/值域 0–9）。本地校验 0 成本，线上失败 1 天 + 1 次提交机会。

#### 2.6.4 顺带发现：ARC-AGI-3 赛道

`arc-prize-2026-arc-agi-3`，奖金 **850,000 USD**（比 ARC-AGI-2 的 700k 更多），已有 **3032 队**，
我们**尚未参赛**（`userHasEntered=False`）。交互式/agentic 基准，与现有投入不兼容，
**记录在案但当前不切换**。

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
