# 基线证据（逐题明细）

这两份 CSV 是 v1.0.0 引擎在**官方 ARC-AGI-2 数据**上的逐题诊断原始记录，是
`BASELINE_V1.md` 与 `LOO_PROBE.md` 里所有汇总数字的来源。

| 文件 | 来源数据集 | 行数（= test 输入数） | 产生的 run |
|---|---|---|---|
| `eval_v1_diagnostics.csv` | 公开评估集 `data/evaluation`（120 题） | 167 | `20260911-153209-d5d666b4` |
| `train_v1_diagnostics.csv` | 训练集 `data/training`（1000 题） | 1076 | `20260911-153234-679d2166` |

## 列说明

| 列 | 含义 |
|---|---|
| `task_id` | ARC 任务 id |
| `test_index` | 该题第几个 test 输入 |
| `solved` | 官方口径：两个 attempt 中任一完全匹配真值 = 1 |
| `top1_solved` | 只看 `attempt_1` 是否匹配 |
| `failure_class` | `solved` / `shape_mismatch` / `palette_mismatch` / `layout_mismatch_minor` / `layout_mismatch_major` |
| `pixel_acc` | 同形状时的逐像素匹配率（不同形状记 0） |
| `pred_shape` / `true_shape` | 预测与真值的网格形状，形如 `30x30` |

## 复现方式

```bash
git clone --depth 1 https://github.com/arcprize/ARC-AGI-2.git ARC-AGI-2-main
python arc_prize_v1/tests/eval_local.py                      # 评估集
python arc_prize_v1/tests/eval_local.py --dataset training   # 训练集
python arc_prize_v1/tests/loo_probe.py evaluation            # 留一法探针
python arc_prize_v1/tests/loo_probe.py training
```

运行产物（checkpoint / 备份 / submission）写在 `arc_prize_v1/arcprize_runtime/`，该目录被
`.gitignore` 排除——只有上面这两份**作为结论依据**的明细被固化进仓库。
