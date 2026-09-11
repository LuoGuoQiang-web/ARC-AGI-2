# Kaggle 材料领域竞赛调研报告

调研时间：2026-09-09（北京时间）
调研方式：Kaggle 官方 API（`api/v1/competitions/list`、内部 `CompetitionService` 接口）+ 网络检索，逐条核实标题、任务描述、截止时间、参赛队伍数。
（注：Kaggle 网页端对本机 IP 返回 reCAPTCHA，因此改走 API 核实，数据来自 Kaggle 自身。）

---

## 一、结论速览

1. **目前 Kaggle 上没有正在开放的铝合金竞赛**（搜索 `aluminium` / `aluminum` / `alloy` / `material` / `metallurgy` / `steel` 在开放竞赛中均为 0 命中）。
2. **历史上确实有铝合金竞赛，但都已结束**（仍可下载数据、看方案、当练习赛）：
   - `Aluminium challenge`（铝合金结构能量预测）
   - `Metallurgica Hackathon 2025`（冶金方向回归赛）
3. 与材料相关且**仍可提交**的开放竞赛只有 2 个：半导体制造缺陷检测（SECOM）、分子味觉/性质预测（SMILES）。
4. 铝合金方向更现实的做法：**用 Kaggle Datasets 里的铝合金数据集 + 自建基线**，而不是等竞赛。

---

## 二、铝合金直接相关（已结束，数据/榜单仍可用）

| 竞赛 | 链接 | 任务 | 评价指标 | 队伍数 | 截止 | 主办 |
|---|---|---|---|---|---|---|
| Aluminium challenge | https://www.kaggle.com/competitions/al-challenge | 机器学习预测**铝结构的能量**（energy of aluminium structures） | MAE | 1 | 2020-01-01 | EfimMazhnik |
| Metallurgica Hackathon 2025 | https://www.kaggle.com/competitions/metallurgica2025 | 冶金 hackathon（表格回归，ID 列 + MAE） | MAE | 17 | 2025-04-05 | AKSHAT |

> 两个都是社区竞赛（非官方大奖赛），已锁定提交（`locked=true`），但数据、Notebook、讨论区仍可访问。
> `Aluminium challenge` 数据量约 2000 行解集，属于典型的小样本回归，适合做铝合金成分-性能建模的基线参考。

---

## 三、金属 / 材料 / 化学相关竞赛（均已结束，可作方案参考）

| 竞赛 | 链接 | 方向 | 队伍数 | 截止 |
|---|---|---|---|---|
| Machine Learning for Band gap prediction | https://www.kaggle.com/competitions/machine-learning-band-gap-prediction | **材料科学**：带隙预测（MetSA，IIT Madras 冶金与材料工程协会 Amalgam 2024） | 80 | 2024-02-27 |
| Steel Quality Challenge | https://www.kaggle.com/competitions/steel-quality-challenge | 钢坯质量预测（工业过程数据） | 33 | 2025-05-11 |
| Severstal: Steel Defect Detection | https://www.kaggle.com/competitions/severstal-steel-defect-detection | 钢材表面缺陷分割 | 2427 | 2019-10-24 |
| NOMAD2018 Predicting Transparent Conductors | https://www.kaggle.com/competitions/nomad2018-predict-transparent-conductors | 透明导电材料带隙/形成能预测（材料信息学经典赛） | 878 | 2018-02-15 |
| Predicting Molecular Properties (CHAMPS) | https://www.kaggle.com/competitions/champs-scalar-coupling | 分子磁相互作用（量子化学） | 2737 | 2019-08-28 |
| NeurIPS - Open Polymer Prediction 2025 | https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025 | 从 SMILES 预测聚合物性质（可持续材料） | 2240 | 2025-09-15 |
| Geochemical Dating | https://www.kaggle.com/competitions/geochemical-dating | 地球化学数据预测超基性熔体年龄 | 29 | 2026-05-22 |
| Vesuvius Challenge - Ink Detection | https://www.kaggle.com/competitions/vesuvius-challenge-ink-detection | X 射线 CT 扫描古卷识墨（材料+成像） | 1249 | 2023-06-14 |
| Predicting Optimal Fertilizers | https://www.kaggle.com/competitions/playground-series-s5e6 | 肥料配方优化（化学组分） | 2648 | 2025-06-30 |

---

## 四、当前仍开放的、与材料/物理化学沾边的竞赛

| 竞赛 | 链接 | 任务 | 队伍数 | 截止 |
|---|---|---|---|---|
| SECOM Defect Detection Challenge | https://www.kaggle.com/competitions/secom-defect-detection-challenge | 用高维传感器数据预测**半导体制造缺陷**（SECOM 数据集，制造过程质量建模） | 2 | 2027-03-06 |
| TasteBench: Molecular Taste Classification | https://www.kaggle.com/competitions/molecular-taste-classification-2026 | 从 **SMILES 分子式**预测味道类别（分子性质预测） | 10 | 2027-04-29 |
| AI4S Open Innovation: AI for Life Science | https://www.kaggle.com/competitions/ai-4-s-open-innovation-artificial-intelligence-for-life-scien | AI + 器官芯片（偏生物） | 1 | 2026-10-10 |

> SECOM 是"制造业工艺数据 → 缺陷预测"的范式，与铝合金铸锭/挤压缺陷预测的建模思路高度同构，且目前参赛队伍极少，适合练手刷榜。

---

## 五、Kaggle 上的铝合金 / 合金数据集（可直接建模）

| 数据集 | 链接 | 点赞 | 大小 |
|---|---|---|---|
| TIG Aluminium 5083（TIG 焊铝板图像） | https://www.kaggle.com/datasets/danielbacioiu/tig-aluminium-5083 | 48 | ~12 GB |
| Defects in aluminium profiles（铝型材表面缺陷图像） | https://www.kaggle.com/datasets/tngw0702/defects-in-aluminium-profiles | 2 | ~2.3 GB |
| Aluminum profile surface defects data set | https://www.kaggle.com/datasets/weihaoreal/aluminum-profile-surface-defects-data-set | 7 | ~1.9 GB |
| Modified Al-Si Alloys: Solidification & Mechanical（铝硅合金凝固与力学性能） | https://www.kaggle.com/datasets/utkarshx27/thermal-profiles-of-the-alloys | 8 | 455 KB |
| 6xxx Series Aluminum Alloys（6xxx 系散热器数据集） | https://www.kaggle.com/datasets/yusufuzunoglu/6xxx-series-heat-sink-data-set | — | — |
| Aluminium Alloy Wheel Dataset | https://www.kaggle.com/datasets/rajsharg/aluminium-alloy-wheel-dataset | 0 | 919 KB |
| Data for weld of 5052 Aluminium Alloy | https://www.kaggle.com/datasets/nutanjain/data-for-weld-of-5052-aluminium-alloy | 1 | 8.7 KB |
| Anodic Aluminum Oxide for ML | https://www.kaggle.com/datasets/muhammadwaseemashraf/anodic-aluminum-oxide-for-ml | 5 | 63 KB |
| Al_dataset（铝相关） | https://www.kaggle.com/datasets/anjalisingh065/al-dataset | 0 | 24 KB |
| Yield strength of alloys dataset（合金屈服强度） | https://www.kaggle.com/datasets/arnabk123/yield-strength-of-alloys-dataset | 13 | 13.7 KB |
| High Entropy Alloys - Properties（高熵合金性能） | https://www.kaggle.com/datasets/sethpointaverage/high-entropy-alloys-properties | 21 | 21 KB |
| Superalloys（高温合金） | https://www.kaggle.com/datasets/edgedislocation/superalloys | 7 | 951 B |
| Metallic Glass Forming（金属玻璃形成能力） | https://www.kaggle.com/datasets/saurabhshahane/metallic-glass-forming | 11 | 23.9 KB |
| Fatigue striations marked on SEM photos（SEM 疲劳条纹） | https://www.kaggle.com/datasets/roishik/fatigue-striations-marked-on-sem-photos | 24 | ~2 GB |
| Aluminum Good And Defective Sample (B&W) | https://www.kaggle.com/datasets/ahibafnan/aluminum-good-and-defective-sample-b-and-w | — | — |

---

## 六、Kaggle 之外、更对口的机会

- **1st Machine Learning meets Materials Science Hackathon**：https://indico.global/event/17097/
- **PEPR Diadem — AI for the discovery of new materials 黑客松（法国）**：https://www.pepr-diadem.fr/2025/11/13/launch-of-the-first-hackathon-ai-for-the-discovery-of-new-materials/
- **MetSA / Amalgam（IIT Madras 冶金与材料工程协会）** 系列赛事，历史上在 Kaggle 办过带隙预测赛，可关注其后续届次。

---

## 七、建议路径

1. **想立刻打榜**：SECOM Defect Detection Challenge（开放、队伍少、制造缺陷范式与铝合金缺陷预测同构）。
2. **想复现铝合金建模**：`Aluminium challenge` + `Metallurgica Hackathon 2025` 的公开数据/Notebook 作为基线，配合上表中的 Al-Si、6xxx、屈服强度、高熵合金数据集扩充特征。
3. **想等对口竞赛**：定期看
   - 开放竞赛列表：https://www.kaggle.com/competitions?listOption=active
   - 竞赛搜索：https://www.kaggle.com/competitions?searchQuery=material
   - 材料相关关键词：`materials`、`alloy`、`polymer`、`crystal`、`battery`、`semiconductor`
