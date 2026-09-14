# work/arc_w1 — ARC-AGI-2 测试时训练（TTT）求解器工作区

> 本目录是 **2026-09-14 从 Kaggle notebook 里抢救出来的**：另一台电脑上的求解器代码只存在于
> Kaggle kernel 与那台机器上，本机（编辑+推送端）此前完全没有。抢救过程与校验见文末。

## 1. 目录内容

| 路径 | 说明 |
|---|---|
| `arc26_solver.py` | **当前 canonical 求解器**（2,536 行 / 111,223 字符），取自 `arc26-submit-full`（09-13 11:22，最新） |
| `versions/v_full_1122.py` | 同上（留档副本） |
| `versions/v_shard_1029.py` | 09-13 10:29 版本（`submit-shard0` / `eval-validate` 用的是它） |
| `versions/v_smoke_0941.py` | 09-13 09:41 版本（`solver-smoke`） |
| `tools/kpush.py` | **重建的推送工具**：本地 .py → Kaggle 单格 notebook（含 argv 固化 + stdout tee） |
| `tools/kpush_prelude` | 见 `kpush.py` 内嵌常量；prelude 已与原始推送版本逐行比对一致 |
| `kaggle_probes/` | 9 个探测 notebook 源码（环境/字母表/加速器/挂载） |
| `ARC26_SOLVER_SPEC.md` | **重建**的规格文档（原文件不在任何 notebook 里） |
| `.kpush_build/` | 构建产物（已被 `.gitignore` 排除） |

原始 notebook 与运行日志另存于 `C:\Users\LRDC07\kaggle_refs\`（**未入库**，体积大且含 token 泄露）。

## 2. 三个版本的差异（**续写时别改回去**）

`def/class` 集合三版**完全一致**，差异只在函数体：

| 变化 | 函数 | 内容 |
|---|---|---|
| smoke(09:41) → full(11:22) | `fallback_pair` (+35/-7) | 重写（见下） |
| 同上 | `main` (+11/-1)、`parse_args` (+5/-0) | 新增 `--num-shards` / `--shard-index` 分片 |
| shard(10:29) → full(11:22) | `fallback_pair` | 同上 |

**`fallback_pair` 为什么必须重写**（原注释已说明）：旧实现返回「众数色填充 + 全 0」，
但 `mode_colour` 把示范输入**和输出**一起统计，背景 0 占绝对多数 → **两个 attempt 输出同一个全 0 网格，
第二个提交位被完全浪费，且主押注变成常数填充**。新实现：

1. 第一押注 = **原样输入**（模型答不出的题，贴近输入远好于常数填充）；
2. 第二押注 = **前后景颜色互换**（社区反馈：稀疏题上的常见平凡变换）；
3. 退化输入（单色/均匀）再退回「原样 + 相异常数」。

**分片为什么是交错而非分块**（原注释）：Kaggle **最多 2 个并发 GPU 会话**，周配额约 30 h，
240 题必须拆成两个 12 h 会话；交错切分可避免任务难度排序在 key 空间聚集而落进同一分片。

## 3. 这台电脑能做什么 / 不能做什么

| | 状态 |
|---|---|
| 编辑求解器、构建 notebook、推送到 Kaggle、读回日志与 report | ✅ 可以（`kaggle` 官方包已装） |
| **本地运行求解器** | ❌ 不行：缺 `torch` / `transformers` / `peft` / `tokenizers`（本机只有 `numpy`），且无 GPU |

**定位**：这台机器是**编辑 + 推送端**；真正的 GPU 运行在 Kaggle 上（另一台电脑负责发起与观察，
本机现在也具备同样能力）。

## 4. 工作循环

```bash
cd work/arc_w1
python tools/kpush.py --dry-run                 # 本地构建并校验（不联网）
python tools/kpush.py --slug arc26-solver-dev \
    --argv "--split evaluation --limit 3 --time-budget-seconds 1800"
python tools/kpush.py --status --slug arc26-solver-dev     # 运行状态
python tools/kpush.py --log    --slug arc26-solver-dev     # 取 kernel_stdout.log
python tools/kpush.py --output --slug arc26-solver-dev     # 取 report.json / submission.json
```

## 5. 运行约束（从代码注释与日志里读出来的，不是猜的）

- **并发上限**：Kaggle 最多 2 个并发 GPU 会话 → 240 题拆 2 个分片，各约 12 h；
- **时间预算**：`--time-budget-seconds 39000`（约 10.8 h，`arc26-submit-full` 实测跑了 37,976 s）；
- **成功的运行不发布日志**，因此必须 tee 到 `/kaggle/working/kernel_stdout.log` —— 这是唯一读取通道；
- **显存**：T4 ×2（各 14.6 GiB）；3.634B 参数必须 **bf16 + 梯度检查点** 才能塞进单卡；
- **词表**：vocab_size = 16，id 0–12 为网格符号，10=`\n`、11=user、12=assistant、13=pad、15=eos；
- **DFS 约束**：token 概率阈值 0.2、每 beam 最多 3 分支、单次 DFS 最多 6000 节点、KV 缓存预算 3 GB。

## 6. 已知缺口（续写的入手点）

| 缺口 | 证据 |
|---|---|
| `ARC26_SOLVER_SPEC.md` **原件缺失** | 代码 docstring 引用 `work/arc_w1/ARC26_SOLVER_SPEC.md`，但不在任何 notebook 里 → 见本目录重建版 |
| **符号引擎从未加载** | 日志：`[stage C] skipped: no symbolic engine loaded` → Stage C 一直是空转，回填只靠启发式 |
| `arc26-stage1` 依赖不兼容 | `ImportError: Found an incompatible version of torchao. Found version 0.10.0` |
| `arc26-stage0` CUDA 错误 | 日志出现 `CUDA kernel errors might be asynchronously reported` |
| **当前分数** | 评估集 24 题子集 **solved 1/24**；smoke 5 题 4/5（简单题） |

## 7. 安全事件：token 泄露（**已处理 2026-09-14**）

**事件**：4 个探测型 kernel 把环境变量表打印进了运行日志，其中含 `KAGGLE_DATA_PROXY_TOKEN` 与
`KAGGLE_USER_SECRETS_TOKEN` 的值：

| kernel | token 命中 | 状态 |
|---|---|---|
| `arc26-mount-probe` | 4 处 | ✅ 已删除 |
| `accel-probe-nvidial4x4` | 2 处 | ✅ 已删除 |
| `accel-probe-nvidial4` | 2 处 | ✅ 已删除 |
| `accel-probe-nvidiateslat4` | 2 处 | ✅ 已删除 |

（扫描了全部 14 个 ARC kernel 的日志；`accelerator-probe-l4x4-check`、`nvarc-env-probe-shape-err`
以及 4 个求解器/stage kernel 均为 0 命中。四个被删 kernel 的**源码已留档**在本目录
`kaggle_probes/`，删除无信息损失。）

**风险更正**：初次判定"公开可见"来自 `kernels/list` 的 `isPrivate` 字段，但该字段在 list 投影里
**未填充**（同批返回的 `hasIsPrivate` 为 false）。用 `kernels/pull` 元数据核对存活 kernel，
全部为 `isPrivate: True` + `hasIsPrivate: True`（真私有）→ 被删的 4 个大概率也是私有，
**实际暴露面小于初判**。删除仍然正确：把"不确定"变成"确定"。

**本地**：`kaggle_refs/arc26-mount-probe.log` 已删除；仓库与拉取目录全量复扫 **0 命中**
（token 从未进入 git 历史）。

**预防规则（以后必须遵守）**：
1. 探测脚本**禁止** `print(os.environ)`，只打印白名单键名，不打印值；
2. 需要检查环境时用键名列表 + 值长度，不输出值；
3. 运行含输出的 notebook 前先确认 `is_private=True`（`kpush.py` 默认就是私有）。

## 8. 抢救过程与校验

1. 用 Kaggle API 列出 17 个 kernel，识别 14 个 ARC 相关；
2. `kernels/pull` 用 `userName`+`kernelSlug`（`user`/`kernel` 已废弃，返回 400）拉取源码；
3. 从单格 notebook 中剥离 `kpush` prelude，还原出 `arc26_solver.py`；
4. **三个版本全部 `py_compile` 通过**；
5. 重建的 prelude 与原版**逐行比对一致**（仅多一行结束标记）；
6. 构建出的 cell 中求解器部分与 `arc26_solver.py` **逐字节相同**（1866 + 111223 = 113089）。
