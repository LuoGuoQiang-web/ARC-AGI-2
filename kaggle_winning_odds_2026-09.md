# Kaggle 开放竞赛「拿下把握」评估（2026-09-09）

数据来源：Kaggle API（`api/v1/competitions/list`、`CompetitionService/GetCompetition`、`LeaderboardService/GetLeaderboard`），效率榜来自官方 notebook `ryanholbrook/rsna-knee-abnormalities-efficiency-lb` 的输出 CSV。
「把握」为主观判断，用于排序，不是承诺。

## 一、所有当前开放竞赛的硬数据

| 竞赛 | 类别 | 队伍数 | 奖池 / 奖位 | 截止（北京） | 榜首 | 我们的把握 |
|---|---|---|---|---|---|---|
| [ARC Prize 2026 - Paper Track](https://www.kaggle.com/competitions/arc-prize-2026-paper-track) | Featured | 176 | $450k：保证 $75k（1st $50k / 2nd $20k / 3rd $5k）+ **$375k 池**（论文 ≥4.5/5 即可分，可多组获奖） | 2026-11-09 | 评审制，无数值榜 | ★★★★ **最高** |
| [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) | Research | 3,377 | $77k：10 个榜单奖 + 3 个效率奖 | 2026-10-22 | 0.954 AUC；效率榜 3,278 条，前 10 名 0.939–0.952 | ★★ |
| [Pokémon PTCG AI Battle Challenge - Strategy](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle-challenge-strategy) | Featured | 698 | $240k | 2026-09-14 07:59（**仅约 4 天**） | 评审制 | ★★ 但时间不够 |
| [Biohub - Cell Tracking During Development](https://www.kaggle.com/competitions/biohub-cell-tracking-during-development) | Research | 3,278 | $60k | 2026-09-29 | 0.970（0.963–0.970 极度拥挤） | ★ |
| [Kaggriculture](https://www.kaggle.com/competitions/kaggriculture) | Featured | 8,275 | $50k | 2026-09-30 | 2957.3 | ★ |
| [ARC Prize 2026 - ARC-AGI-2](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2) | Featured | 1,889 | $700k（$550k 保证） | 2026-11-02 | 76.94%（#2 72.08%，#3 骤降至 40.83%） | 极低 |
| [ARC Prize 2026 - ARC-AGI-3](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-3) | Featured | 2,897 | $850k | 2026-11-02 | 11.04% | 极低 |
| [Playground S6E9 - EV 购车预测](https://www.kaggle.com/competitions/playground-series-s6e9) | Playground | 1,284 | 仅 swag（无现金） | 2026-09-30 | — | 奖牌有戏，钱没有 |
| [SECOM Defect Detection Challenge](https://www.kaggle.com/competitions/secom-defect-detection-challenge) | Community | **2** | 无奖金 | 2027-03-06 | — | 拿第 1 几乎必成，但无价值 |
| [TasteBench: Molecular Taste Classification](https://www.kaggle.com/competitions/molecular-taste-classification-2026) | Community | **10** | 无奖金 | 2027-04-29 | — | 同上 |

> 上一轮找到的材料类竞赛（Aluminium challenge、Metallurgica 2025、Steel Quality Challenge、ML for Band gap prediction 等）**全部已关闭**，不参与本轮"拿下"评估。

## 二、首选：ARC Prize 2026 Paper Track

规则原文要点（[arcprize.org](https://arcprize.org/competitions/2026/paper)）：

1. **赛道仅 176 支队伍**，且 $375k 池是"论文评分 >4.5/5 即有机会，可多组获奖"，不是只有第 1 名。
2. **"The code submission need not achieve a high score for the corresponding paper to be eligible."** —— 不需要打赢 76.94% 的榜首，只要有一份真实能跑的 Kaggle 提交。
3. 评分维度：Accuracy / Universality / Progress / **Theory** / Completeness / **Novelty** —— 与我们的强项（文献梳理、对照实验、把"为什么有效"讲清楚）高度重合。
4. 需要交付：ARC-AGI-2 或 ARC-AGI-3 的一份代码提交 + 一篇结构清晰的论文；截止 2026-11-09，剩约 8 周。

判断（主观）：认真投入 8 周，进前 3（保证 $75k）估计 5–15%；论文 ≥4.5 分进池子估计 15–30%。**这是所有现金赛道里最高的。**

## 三、次选：RSNA Knee Abnormality Detection

- 13 个奖位看着多，但**效率榜有 3,278 条记录**，前 10 名分数 0.939–0.952 —— 没有"冷门赛道"可捡，效率奖同样是强队之争。
- 需要 GPU、多模态医学影像工程能力、6 周；判断：夺冠 1–3%，进前 25 更现实。
- 只有在有 GPU + 愿意组 5 人队时才值得打。

## 四、基本别碰

- **Biohub Cell Tracking**：榜单饱和在 0.963–0.970，20 天，奖位少。
- **Kaggriculture**：8,275 队，3 周。
- **ARC-AGI-2 / AGI-3 榜单**：榜首 76.94% / 11.04%，属于前沿研究级投入，投入产出比最差。

## 五、如果"拿下"只要求第一名

SECOM（2 队）、TasteBench（10 队）这类社区赛几乎可以稳拿第 1，但按 Kaggle 现行规则，社区竞赛通常不记积分/奖牌（页面上常见 "This competition does not award ranking points" 标注），除履历一句话外没有实际收益。

## 六、下一步需要确认的输入

1. 目标：现金 / 奖牌 / 履历？
2. 算力：本机有无 GPU？能否使用 Kaggle Notebook 的 GPU 配额（ARC 与 RSNA 都是 kernels-only 提交）？
3. 时间：能投入多少小时/周？
4. 若选论文赛道：我们可以在本周给出 3 个候选研究角度 + 一个可跑的 baseline，2 周内出第一版实验。
