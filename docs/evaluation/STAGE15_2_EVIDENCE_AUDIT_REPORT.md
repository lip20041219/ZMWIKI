# Stage 15.2 — EvidenceAudit v0.2 离线校准与验收

日期：2026-11-04。状态：离线评测验收通过；不启用运行时门禁。

## 目标与输入

在保存的回答上检查“引用页面是否支持答案中的可提取 claim”，不改变回答、评分器或历史实验。脚本为 [`benchmark/run_evidence_audit_v02.py`](../../benchmark/run_evidence_audit_v02.py)，结果为 [`benchmark/results/stage15_2_evidence_audit_v02.json`](../../benchmark/results/stage15_2_evidence_audit_v02.json)。以下 v0.2 数字均从该 JSON 读取。

- 语料：`benchmark/corpus/wiki/` 的 9 篇文本；数据集 `benchmark/datasets/multihop_wiki.json` v1.4，共 50 题（43 可答、7 不可答），SHA-256 `f12127421c762ce191252f6f82559dc6b60397f7b603101bc1b034ee3299cce8`。
- 回答：`benchmark/results/stage12_multiround_20261101_120442_r1.json`、`..._r2.json`、`..._r3.json`。每轮两组各 50 条；文件内数据集指纹与当前 v1.4 一致。
- 人工裁定：`benchmark/reviews/stage13_0_human_review_20261101.json`、`stage13_1_batch1_verdicts_20261101.json`、`stage13_3_batch2_verdicts_20261101.json`，选取 35 条 `human_verdict=correct`。
- 历史错误：脚本中的三段固定历史错误答案及假设的合法引用页面；它们不是 Stage 12 原记录中的空引用答案。Fixture：`benchmark/datasets/evidence_audit_v0_cases.json` 的 10 条独立案例。

## 环境与复现

在 Windows / Python 项目现有 `.venv` 下执行；脚本只读取上述本地文件，调用规则式 `benchmark/evidence_audit.py`，未调用模型或网络接口。使用当前仓库源码重新生成；Stage 12 结果记录的 runner/Agent 源码指纹与当前源码不同，因此本报告不重算或跨版本比较旧的答案准确率。

```powershell
.venv/Scripts/python.exe benchmark/run_evidence_audit_v02.py --output benchmark/results/stage15_2_evidence_audit_v02.json
.venv/Scripts/python.exe -m pytest -q tests/wiki/test_evidence_audit.py tests/wiki/test_evidence_audit_v02.py
```

脚本无 `--output` 时将 JSON 打印到标准输出；给出该参数才写指定文件。结果 JSON 含 `full_50_r3`、`all_rounds`、`known_correct`、`historical`、`fixture`、`ablation` 六个顶层字段。测试会在临时路径重新运行脚本，并逐字段比对已保存 JSON。

## R3 与三轮结果

| 范围 | records | claims | supported | contradicted | unverifiable | case coverage | claim coverage |
|---|---:|---:|---:|---:|---:|---:|---:|
| R3 全部 | 100 | 257 | 153 | 0 | 104 | 65/100 = 65.00% | 153/257 = 59.53% |
| R3 可答 | 86 | 240 | 146 | 0 | 94 | 61/86 = 70.93% | 146/240 = 60.83% |
| R3 不可答 | 14 | 17 | 7 | 0 | 10 | 4/14 = 28.57% | 7/17 = 41.18% |
| 三轮全部 | 300 | 746 | 435 | 0 | 311 | 188/300 = 62.67% | 435/746 = 58.31% |
| 三轮可答 | 258 | 702 | 413 | 0 | 289 | 177/258 = 68.60% | 413/702 = 58.83% |
| 三轮不可答 | 42 | 44 | 22 | 0 | 22 | 11/42 = 26.19% | 22/44 = 50.00% |

`case_coverage` 的分子是至少有一个 supported 或 contradicted claim 的**记录**数，分母是记录数；`claim_coverage` 是 `(supported + contradicted) / claims`。这里“case”是脚本字段名称，R3 同一题有两组记录，三轮再重复三次，不能把这些记录当独立的 300 道题。三轮比 R3 的记录覆盖率低 2.33 个百分点；这只是已保存回答上的敏感性描述，不是新模型推理实验。

## 人工正确样本、历史错误与 Fixture

35 条已知正确记录涉及 17 个唯一 case：85 claims，52 supported、0 contradicted、33 unverifiable；记录覆盖率 19/35 = 54.29%，claim 覆盖率 52/85 = 61.18%。误判为 contradicted 的记录为 **0/35**，按 claim 为 **0/85**。这是 Stage 13 风险富集并经人工裁定的子集，不代表全量假阳性率。

| 历史错误 | claims | supported | contradicted | unverifiable | 判定 |
|---|---:|---:|---:|---:|---|
| `mh38_responsibility_synthesis` | 4 | 2 | 2 | 0 | contradicted |
| `mh10_direct_reports` | 4 | 2 | 1 | 1 | contradicted |
| `mh49_orion_upward_chain_ceo` | 3 | 2 | 1 | 0 | contradicted |

因此指定引用前提下检出 **3/3**。Fixture 的 case 判定与逐 claim 预期均匹配 **10/10**。历史错误的检测依赖脚本中指定的合法引用；对原始 Stage 12 空引用记录不能套用这个结论。

## 特征消融

各配置的 claim 拆分数量不同，claim coverage 不宜直接当作固定分母的准确率。下表列出相同记录集合上的可判定记录覆盖率与 claim 分布（`supported/contradicted/unverifiable`）。

| 配置 | 正确样本 35 条：S/C/U | 正确样本覆盖率 | R3 可答 86 条：S/C/U | R3 可答覆盖率 |
|---|---|---:|---|---:|
| `core_v0.1` | 25/1/34 | 14/35 = 40.00% | 71/1/98 | 44/86 = 51.16% |
| `paraphrase` | 50/1/29 | 17/35 = 48.57% | 148/1/77 | 62/86 = 72.09% |
| `clause_split` | 26/0/39 | 15/35 = 42.86% | 69/0/119 | 41/86 = 47.67% |
| `dimensions` | 25/1/34 | 14/35 = 40.00% | 71/1/98 | 44/86 = 51.16% |
| `paraphrase+clause` | 52/0/33 | 19/35 = 54.29% | 146/0/94 | 61/86 = 70.93% |
| `all_v0.2` | 52/0/33 | 19/35 = 54.29% | 146/0/94 | 61/86 = 70.93% |

`dimensions` 单独与 core 相同；全量与 `paraphrase+clause` 在这两个集合上相同。单用 paraphrase 提高可判定记录覆盖率，但在正确样本中出现一条 contradicted，组合 clause split 后消失；这说明单项覆盖率提高不保证误判降低。R3 全量 0 contradicted 也不能证明全部回答正确。

## 与 Stage 15.1 比较及边界

[`STAGE15_1_EVIDENCE_AUDIT_REPORT.md`](STAGE15_1_EVIDENCE_AUDIT_REPORT.md) 记录 v0 在 35 条正确样本上为 62 claims、25 supported、0 contradicted、37 unverifiable，记录覆盖率 14/35 = 40%。v0.2 对同一 35 条记录为 85/52/0/33，覆盖率 19/35 = 54.29%，增加 5 条可判定记录。claim 提取和拆分规则已改变，不能将 25/62 与 52/85 解释为模型答案质量改善。两版在指定历史错误上均为 3/3，fixture 均为 10/10。

Coverage 不等于 Accuracy；unverifiable 表示规则或引用不足以作出判断，不表示答案错误。`citation_claim_precision=1.0` 只以可判定 claim 为分母；未判定的 104 条 R3 claim 不在其中。没有 contradicted 不表示其余答案正确。此规则式审计针对 9 页合成语料和显式关系，尚不能作为生产级事实核查系统或 WikiAgent Runtime Gate；低覆盖的不可答记录、间接关系、聚合与代词链仍需谨慎处理。

## 验收

- 离线结果 JSON 可解析，六个字段齐全；R3 100、三轮 300、人工正确 35、历史错误 3、fixture 10，聚合数量与明细一致。
- 已知正确记录 contradicted 0/35，历史错误检出 3/3，fixture 匹配 10/10；符合本阶段离线校准目标。
- 定向命令结果：**28 passed in 11.22s**；无外部模型调用。未执行全量 pytest 或新的模型推理评测。
- 后续最小行动：若要评估运行时使用，先取得更广的人工标注与低覆盖类型专项样本，再另行定义门禁误判与漏检准则。本阶段不进入 Stage 16。
