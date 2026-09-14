# ARC26_SOLVER_SPEC —— **重建版**

> ⚠️ **这不是原始规格文档。** 代码 docstring 引用的是 `work/arc_w1/ARC26_SOLVER_SPEC.md`（Option C），
> 但该文件**不在任何 Kaggle notebook 里**，本机也没有。以下内容是从
> `arc26_solver.py`（v_full_1122）的 docstring、常量与函数结构**反向重建**的，
> 目的是让续写者不再丢失设计意图。**凡是从代码里读出来的，都标注了出处；凡是不确定的，标 `TODO-ver1`。**
>
> 重建时间：2026-09-14 ｜ 依据：`work/arc_w1/versions/v_full_1122.py`（2,536 行）

---

## 1. 总体思路

**测试时训练（TTT）+ 约束解码**：对一个已在网格上做过 SFT 的小模型，在**推理时**用当前任务的
示范对做 LoRA 微调，然后在 16 个 ARC token 的空间里做**约束束搜索 DFS**，最后对候选做 NLL 重排。

（出处：模块 docstring 开头 —— "ARC-AGI-2 test-time-training (TTT) solver … Clean-room
implementation of work/arc_w1/ARC26_SOLVER_SPEC.md (Option C). Everything here is written from
that spec; no third-party solver code is copied."）

## 2. 模型与词表

| 项 | 值 | 出处 |
|---|---|---|
| 模型 | `sorokin/qwen3_4b_grids15_sft139`（Qwen3ForCausalLM，bfloat16） | `DEFAULT_MODEL_DIR` |
| 参数量 / 结构 | 3.634B / hidden 2560 / 36 层 | `arc26-stage1` 日志 |
| 词表 | **vocab_size = 16** | `VOCAB` / stage1 日志 |
| token 布局 | id 0–12 = 13 个网格符号；`\n`=10，user=11，assistant=12，pad=13，eos=15 | 常量 `NEWLINE_TOKEN_ID` / `USER_TOKEN_ID` / `ASSISTANT_TOKEN_ID` / `PAD_ID` / `EOS_ID` |
| 网格上限 | 30×30（`MAX_GRID_SIDE`，最多 900 cell） | `MAX_CELLS_PER_GRID` |
| 对话模板 | `<\|im_start\|>user\n … <\|im_start\|>assistant\n … <\|im_end\|>` | `USER` / `ASSISTANT` / `ENDOFTEXT` |

> `TODO-ver1`：13 个符号与 0–9 颜色的完整对应表未在代码常量里显式列出（`arc26-stage1` 的意图正是
> 把它 dump 出来，但该次运行卡在 torchao 不兼容）。**这是续写的第一个待补项** —— `convert_grid_to_string`
> / `decode_arc_tokens` / `validate_grid` 三个函数是唯一权威来源。

## 3. 数据与序列化

- 数据目录默认 `/kaggle/input/arc-prize-2026-arc-agi-2`；`load_split()` 用 `_find_file()` 在 3 层内
  按文件名找 `arc-agi_{training,evaluation,test}_{challenges,solutions}.json`（`SPLIT_FILES`）。
- 序列化：`convert_grid_to_string`（网格 → 字符串）、`parse_grid_string` / `tokens_to_array`
  （字符串/token → `np.ndarray`，受 `limit_rows` 约束）、`fmt_train` / `fmt_query` / `fmt_reply`
  拼出完整的 prompt 与答案格式。

## 4. 增强（augmentation）

- `AugKey`：几何变换 + 可选颜色置换的组合键；
- `apply_augment` / `invert_augment` 成对出现 —— **所有增强都必须可逆**，因为模型输出要逆变换回原网格空间；
- `augment_demos(demos, key)` 作用于示范对；
- CLI：`--aug-train N`（TTT 用的增强份数）、`--aug-infer M`（推理时的增强份数）。
  实测配置 `--aug-train 2 --aug-infer 1`。

## 5. 调度：`cascade`（默认）与 `uniform`

（出处：模块 docstring 的 "Two scheduling strategies" 一节）

**`cascade`** —— 把稀缺 GPU 预算花在能换来覆盖面的地方：

| 阶段 | 做什么 | 关键点 |
|---|---|---|
| **A 廉价扫描** | 对**所有**任务 prompt 已 SFT 的模型，只做约束 DFS 生成（`do_ttt=False`，无 LoRA、无梯度） | 每题产出一个**置信度信号** = 增强一致性 + 最佳重排 NLL |
| **B 精修** | 按 A 的置信度排序，对**最弱**的题跑完整 LoRA TTT（`do_ttt=True`），直到预算耗尽 | 相关常量：`STAGE_A_DEFAULT_SLICE=45.0`、`STAGE_A_CALIBRATION_SLICE=90.0`、`STAGE_A_MIN_SLICE=8.0`、`STAGE_B_MIN_SLICE=45.0` |
| **C 回填** | 对仍无高置信答案的题，合并可选 `--engine` 符号候选 + 启发式兜底，重新选出两个 attempt | `CONFIDENCE_BAR=0.6`（≥此值则在 C 阶段不动它）、`SYMBOLIC_CONFIDENCE=0.9` |

**`uniform`** —— 规格里的原始调度：每题立刻做 TTT，时间片由标定阶段测得的「秒/题」驱动
（CLI：`--calibrate-tasks N`）。实测 `--calibrate-tasks 4`。

**硬约束**：`--time-budget-seconds 39000`（≈10.8 h）。日志实测 `arc26-submit-full` 跑了 37,976 s
后打印 `[stage B] stopping: reserve reached (remaining=1037s)`。

## 6. LoRA TTT

| 项 | 值 |
|---|---|
| r / alpha / dropout | **16 / 32 / 0.0** |
| 学习率 / epoch | **5e-5 / 1** |
| 最大序列长度 | **8192** |
| 梯度裁剪 | 1.0 |
| 目标模块 | `LORA_TARGET_MODULES`（元组常量） |
| 显存前提 | **bf16 + 梯度检查点**（docstring 明写：3.63B 参数塞进 14.6 GiB 的必需条件） |

相关函数：`attach_lora` / `detach_lora`（自定义包装，未用 peft 默认注入路径）、
`_target_module_names`、`_wrap_lora_modules`、`build_ttt_sequences`、`run_ttt`。

## 7. 约束束搜索（DFS）

| 常量 | 值 | 含义 |
|---|---|---|
| `DFS_TOKEN_LOGPROB_THRESHOLD` | `log(0.2)` | token 对数概率下限 |
| `DFS_TOKEN_PROB_THRESHOLD` | `0.2` | 同一规则的另一种表述 |
| `DFS_MAX_BRANCHES_PER_BEAM` | 3 | 子节点上限（显存护栏） |
| `DFS_MAX_NODES` | 6000 | 单次 DFS 全局节点预算 |
| `DFS_DEFAULT_CACHE_BUDGET_GB` | 3.0 | 活跃 beam 的 KV 缓存预算 |
| `KV_BYTES_PER_TOKEN_PER_LAYER` | `2 * 8 * 128 * 2` | KV 头数 × head_dim × dtype 字节 |
| `KV_BYTES_PER_TOKEN_PER_BEAM` | `36 * 上者` | 36 层 |

相关函数：`turbo_dfs`（手写 KV 缓存操作：`_cache_set_layer_kv` / `_cache_select` / `_cache_length`
/ `_forward_step`）、`greedy_cached_fallback`、`constrained_generate`、
`max_live_beams_for`（按 prompt 长度与 KV 预算算最多活几个 beam）、`arc_logprobs`（只在 13 个
ARC token 上取 logits）。

## 8. 候选打分与选择

| 常量 | 值 | 作用 |
|---|---|---|
| `SYMBOLIC_NLL_BONUS` | 0.5 | 符号候选的 NLL 奖励 |
| `UNSCORED_SYMBOLIC_NLL` | 0.0 | 未打分的符号候选 |
| `UNSCORED_NEURAL_NLL` | 2.0 | 未打分的神经候选 |
| `UNSCORED_PRIOR_NLL` | 3.0 | 未打分的先验候选（最低优先） |
| `SOURCE_PRIORITY` | `neural:0, symbolic:1, prior:2, fallback:3` | 同分时的来源优先序 |

流程：`calc_scores`（批量算 NLL）→ `dedupe_pool`（去重，上限 32）→ `pool_stats` →
`select_attempts`（选出两个 attempt，调用 `fallback` 保证不留空）→ `score_attempts`（有真值时自评）。

## 9. 兜底答案（最新一次改动，务必保留）

`fallback_pair(task, test_index)` 返回**两个合法且互不相同**的网格：

1. **原样输入**（第一押注 —— 模型答不出的题上，贴近输入远好于常数填充）；
2. **前后景颜色互换**（第二押注 —— 稀疏题上的常见平凡变换）；
3. 退化输入（均匀/单色）退回「原样 + 相异常数」。

`mode_colour` 会同时统计示范的输入与输出，因此**不能**用它构造两个 attempt（旧实现的 bug：
两个 attempt 会变成同一个全 0 网格，第二个提交位被浪费）。

## 10. 分片

`--num-shards N` / `--shard-index K`：任务 id 先 `sorted()`，再按 `i % N == K` **交错**取。
理由（原注释）：Kaggle 最多 2 个并发 GPU 会话，30 h 周配额 → 240 题拆两个约 12 h 的分片；
交错可避免难度排序在 key 空间聚集。

## 11. CLI 参考

```
--split {training,evaluation,test}     默认 test
--data-dir PATH                        默认 /kaggle/input/arc-prize-2026-arc-agi-2
--out PATH                             默认 /kaggle/working/submission.json
--report PATH                          默认 /kaggle/working/report.json
--time-budget-seconds N
--schedule {cascade,uniform}
--calibrate-tasks N
--aug-train N / --aug-infer M
--engine PATH                          可选符号引擎（**从未成功加载过**）
--num-shards N / --shard-index K
--limit N                              只跑前 N 题（0 = 全部）
```
（完整清单以 `parse_args()` 为准）

## 12. 产物

| 文件 | 内容 |
|---|---|
| `/kaggle/working/submission.json` | 正式提交（每题两个 attempt） |
| `/kaggle/working/report.json` | 运行报告（`data` / `n_tasks_total` / `n_with_solutions` / `shard` …） |
| `/kaggle/working/kernel_stdout.log` | **成功运行唯一可读的日志通道**（由 kpush prelude tee 出来） |

## 13. 待补（续写清单）

1. 13 个网格符号与颜色的对应表（`arc26-stage1` 的目标，因 torchao 不兼容未完成）；
2. `--engine` 符号引擎的实现或接入（Stage C 目前空转）；
3. `arc26-stage0` 的 CUDA kernel 错误定位；
4. 本规格文档与原始 `ARC26_SOLVER_SPEC.md` 的差异核对（原件到手后）。
