# ZMWiki Stage 8/9 历史评测数据归档审计

> 审计范围：Stage 8、Stage 9、Stage 9.3 历史结果的数据集来源、SHA、版本、运行配置和 Git 归档可追溯性。
> 审计方式：只读检查；未修改历史结果、数据集或 Git 历史；未调用模型 API、未运行全量测试。

## 结论与验收状态

- **Stage 8、Stage 9、Stage 9.3 均不能在当前工作区中精确复现。** 三组结果记录了数据集版本和哈希信息，但在当前数据集目录、Git 历史及已检查工件中均未找到与其哈希匹配的历史数据集快照。
- 三组结果应标记为**仅保留历史结果**，不能作为当前 v1.4 数据集的可复现基线，也不应据此宣称实验可重跑。
- Stage 8 的来源链较弱：主结果仅保存 16 位 SHA，未发现对应 `run_info`；Stage 9 和 Stage 9.3 保存了完整 SHA、运行信息和配置，但仍缺少匹配的数据集内容。
- 未发现数据被覆盖或 Git 冲突的证据；主要问题是历史数据集快照缺失及结果之间的 SHA 不一致，具体变更原因无法由现有证据确定。

## 检查证据

| 阶段 | 结果与配套工件 | 版本 | 结果中 SHA | 工件中的完整 SHA | Git SHA | 精确复现判断 |
|---|---|---:|---|---|---|---|
| Stage 8 | `benchmark/results/stage8_smoke5_20261031_172818.json`、checkpoint、metrics、log | 1.4 | `e92d4d1c060aa050` | 未记录 | `1cdd718481c03772867cd924b9f0ebfeb30d71e1` | 不可精确复现 |
| Stage 9 | `benchmark/results/stage9_full50_20261031_185801.json`、checkpoint、metrics、`run_info`、log | 1.4 | `5f1ea74f4aeb149d` | `5f1ea74f4aeb149d57bbb81debab24c41d9c6f7ce8679d27e3a0a555c3279a91` | `1cdd718481c03772867cd924b9f0ebfeb30d71e1` | 不可精确复现 |
| Stage 9.3 | `benchmark/results/stage9_3_full50_20261031_194331.json`、checkpoint、metrics、`run_info`、log | 1.4 | `4d77bae1519941b6` | `4d77bae1519941b63ee551123b12d2f49fcf6b4544aaa9b6822ac698ae6ca821` | `1cdd718481c03772867cd924b9f0ebfeb30d71e1` | 不可精确复现 |

三组运行配置均记录了：`DeterministicRAGPipeline`、`HybridRetriever(dense + bm25)`、`chunk_size=120`、`overlap=20`、`top_k=5`、离线确定性生成器；同时保留了 WikiAgent 的历史运行配置（`deepseek-flash`、`max_steps=8`、`search_top_k=5`）。Stage 9 和 Stage 9.3 的 `run_info` 标记为 `completed`，并记录 `api_key_present: True`；这只是历史运行元数据，不包含密钥材料。

当前工作区 `benchmark/datasets/multihop_wiki.json` 的完整 SHA 为：

```text
f12127421c762ce191252f6f82559dc6b60397f7b603101bc1b034ee3299cce8
```

该 SHA 与 Stage 8、Stage 9、Stage 9.3 均不一致。

## Git 历史、数据集目录与工件搜索

1. Git 历史中可见 `benchmark/datasets/multihop_wiki.json` 的历史提交（包括 `5c0589b`、`c66bfa8` 等），但没有发现与 Stage 8/9/9.3 三个 SHA 对应的独立数据集快照。
2. 当前 `benchmark/datasets` 目录只有工作区数据集及其说明、检索数据集，没有发现 v1.3/v1.4 的归档副本或按 SHA 命名的历史快照。
3. `git ls-tree` 检查显示 Stage 8/9/9.3 结果文件不在当前 `HEAD` 的已跟踪树中，而是在工作区工件目录中；结果文件本身包含哈希、路径和配置元数据，但不包含可恢复的数据集内容。
4. Stage 9/9.3 的 metrics/run_info 使用绝对路径 `D:\desk\ZMWiki\benchmark\datasets\multihop_wiki.json`，会降低跨机器复现能力。

## 归档分类

### 可精确复现

- 当前 v1.4 数据集及与其 SHA 一致的后续评测工件（例如 Stage 10–12 范围内已核对的结果）具备较完整的数据集—脚本—配置链路。
- 这不改变 Stage 8/9/9.3 的历史结果判定；它们仍不能使用当前数据集精确重跑。

### 仅保留历史结果

- Stage 8：主结果、checkpoint、metrics、log。可用于追溯当时的记录和指标，但缺少完整 SHA 与 `run_info`，证据链较弱。
- Stage 9：结果、checkpoint、metrics、`run_info`；log 可作为补充证据。完整 SHA 和运行配置可追溯，但数据集内容缺失。
- Stage 9.3：结果、checkpoint、metrics、`run_info`；log 可作为补充证据。完整 SHA 和运行配置可追溯，但数据集内容缺失。

### 建议排除

- `tatus --short` 根目录临时文件、调试日志和无法证明与结果一一对应的重复工件，不纳入稳定发布提交。
- Stage 8/9/9.3 不应被放入“可复现基准”目录；若保留，应在归档说明中明确“历史结果、不可精确复现”。本轮不删除任何文件。

## 问题、影响与最小处理方案

| 问题 | 影响 | 是否必须修复 | 最小处理 |
|---|---|---|---|
| 三个历史 SHA 与当前数据集不一致，且未找到匹配快照 | 无法证明可精确重跑，跨阶段比较存在口径风险 | 是（归档标注层面） | 将三组结果标为 historical-only，不改结果、不伪造快照 |
| Stage 8 缺完整 SHA/`run_info` | 来源链不完整 | 建议 | 在报告中保留缺口说明；不回填未知信息 |
| Stage 9/9.3 含绝对数据集路径 | 跨机器不可直接复现 | 建议 | 归档时在说明中记录路径不可移植；不修改历史 JSON |
| `api_key_present: True` | 说明历史运行曾配置外部 API，不能称为纯离线复现 | 必须在表述中澄清 | 仅作为元数据保留；未发现 API Key/token 明文 |

## 最小 Git 归档建议

本轮不执行 `git add`、提交或任何历史改写。若后续确认需要归档，建议按以下顺序分组：

1. **历史结果说明组（可选）**：Stage 8/9/9.3 的结果、metrics、`run_info`、checkpoint，以及本报告；提交信息建议 `docs(benchmark): document stage8-9 historical-only provenance`。提交说明必须写明缺少匹配数据集快照，不能标记为 reproducible baseline。
2. **稳定可复现基线组（优先）**：仅纳入与当前数据集 SHA 一致、且具备脚本和配置对应关系的评测工件；不要混入 Stage 8/9/9.3。
3. **排除组**：`tatus --short`、未关联日志、临时调试输出和无法追溯来源的大型重复工件，不提交稳定版本。

如果项目策略要求“所有历史实验均可重跑”，当前证据不足，应先由外部归档流程提供三份匹配数据集快照；在快照缺失前，不应自行补造、替换或改写历史数据。

## 下一步

唯一优先事项是确定归档策略：接受 Stage 8/9/9.3 为“历史结果、不可精确复现”，或由项目维护者提供匹配快照后再考虑可复现归档。除此之外，本轮没有必要修改代码、评分器、数据集或实验结果。

## 已执行检查与未执行项目

- 已执行：`git log --all`、`git ls-tree`、数据集与结果目录只读清单、Python 元数据解析、SHA 比对、文本检索。
- 未执行：全量 pytest、模型/API 调用、评测重跑、Git 写操作。
- 本报告是本轮唯一新增文件；未修改其他文件。
