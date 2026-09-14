# 符号引擎接入（`--engine`）—— 2026-09-14

第 2 项完成记录。目的：填上 Stage C 的空转（日志里长期是
`[stage C] skipped: no symbolic engine loaded`）。

---

## 1. 接口契约：**逐字兼容**，无需适配器

审查时逐项比对求解器与 `arc_prize_v1.py`，结论是两边本来就对得上：

| 契约 | 求解器要求 | v1 引擎 | 判定 |
|---|---|---|---|
| 配置构造 | 可选 `make_config(overrides)`，传入 `mode=dev` / `project_dir` / `resume=False` / `snapshot_on_finish=False` | 同名函数，且这些键都在合法范围内 | ✅ |
| 解题入口 | `solve_task(task, cfg, task_deadline=)`（带 TypeError 回落到 `solve_task(task, cfg)`） | 签名逐字匹配 | ✅ |
| 返回值 | `{"attempts": [...], "diagnostics": {...}}` | 同 | ✅ |
| 已验证语义 | `diagnostics.source == "search"` → `verified=True` | `"search"` = 有程序复现全部示范；否则 `"prior"` | ✅ 语义相同 |
| 时钟 | `time.monotonic()` | 同 | ✅ |
| 失败模式 | `load_engine` 永不致命；`engine_predict` 全异常返回 `[]` | — | ✅ |

## 2. 交付通道（原 R4 阻塞项，已修）

`kpush.py` 新增两种方式，**都不需要上传 Dataset** —— 整个交付物仍是**一个 notebook 文件**：

```bash
# 方式 A（推荐）：一条命令把引擎一并送上去，并自动补 --engine 参数
python tools/kpush.py --slug arc26-eng --engine ../../../arc_prize_v1/arc_prize_v1.py \
    --argv "--split evaluation --limit 8 --time-budget-seconds 3600 --aug-train 2 --aug-infer 2"

# 方式 B：任意附带文件（可重复）
python tools/kpush.py --slug x --extra ../../../arc_prize_v1/arc_prize_v1.py:/kaggle/working/eng.py ...
```

机制：每个附带文件生成一个 `%%writefile <remote>` 单元格，放在求解器单元格**之前**，按顺序执行。
`--dry-run` 会逐格校验附带内容**逐字节一致**（实测 59,388 bytes 完全一致），并打印最终 argv。

## 3. 本地验证结果（无 GPU、无 torch）

`tests/test_engine_integration.py` 直接调用求解器**真实的** `load_engine` / `engine_predict`
（即 Stage C 走的那条代码路径），**11 项检查全过**：

| 检查 | 结果 |
|---|---|
| 引擎加载 | `status=loaded`，且 `make_config` 收到求解器的 overrides |
| 失败模式 | 缺失 → `skipped`；损坏的 Python → `error`，**都不抛异常** |
| 270 个真实任务 | 每个 test 输入一个预测、键恰为 `attempt_1/attempt_2/verified/program`、网格全部合法（≤30×30，0–9） |
| **开销** | **32 ms/题**（GPU 侧每题 79–180 s，可忽略） |
| **`verified` 语义** | **4 个真实任务被验证，4/4 预测与真值完全一致（0 错）** |

最后一行是核心：它证明 `verified` 标志确实等价于"存在复现全部示范的程序"，
而这正是混合池里符号候选能拿到 `SYMBOLIC_CONFIDENCE = 0.9` 与 `SYMBOLIC_NLL_BONUS = 0.5` 的依据。

## 4. 诚实的期望管理

v1 引擎的实测能力（`arc_prize_v1/BASELINE_V1.md`）：**评估集 0/167、训练集 34/1076**。
本地抽样的 270 个任务里只有 **4 个**能让引擎验证出程序。

因此：

- **在 ARC-AGI-2 评估/测试集上，接入引擎的精度提升预计接近 0** —— 因为瓶颈是"生成不到"
  （`pool_recall = 0`），而 v1 DSL 在评估集上同样一个程序都验证不出来；
- 它的价值在于：① 架构补齐（Stage C 不再空转）；② 在**较易的题**上提供高置信候选
  （符号候选的 NLL 奖励让它在同分时优先于神经候选）；③ 为论文提供**可量化的混合证据**。

**不要指望它把分数抬起来** —— 抬分数要靠搜索宽度（`--dfs-prob-threshold` 等）与 TTT。

## 5. 怎么读它到底贡献了多少

报告 `report.json` 里：

| 字段 | 含义 |
|---|---|
| `engine.status` | `loaded` / `skipped` / `error` —— **必须不是 `skipped`**，否则路径没传对 |
| `engine.n_tasks_validated` | 有多少题找到了**复现全部示范**的程序（真正的符号贡献） |
| `engine.n_tasks_prior_only` | 只有形状/颜色先验存活（几乎无价值） |
| `per_task[].engine.source` | `search`（已验证）或 `prior` |

判据：若 `n_tasks_validated` 为 0 而 `status=loaded`，说明引擎跑通了但一道题都没帮上 ——
这时不该在引擎上再投入，应回到搜索宽度与 TTT。

## 6. Kaggle 上的运行命令

```bash
python tools/kpush.py --slug arc26-eng-probe \
  --engine ../../../arc_prize_v1/arc_prize_v1.py \
  --argv "--split evaluation --limit 8 --time-budget-seconds 3600 --aug-train 2 --aug-infer 2"
python tools/kpush.py --log --slug arc26-eng-probe
```
预计消耗 ~0.45 h GPU。**按 `QUOTA_PLAN.md`，这笔应记在 P3（搭车验证），不要单独为它开一个周期。**
