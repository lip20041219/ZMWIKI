# Stage 13 Final — 人工复核汇总报告

- **生成日期**：2026-11-01
- **数据批次**：`stage12_multiround_20261101_120442`（Stage 12 三轮）
- **数据集**：`benchmark/datasets/multihop_wiki.json` v1.4
- **评分器**：`stage9_4_frozen`（`benchmark/run_multihop_evaluation.py`）
- **人工裁定输入**：Stage 13.1 batch1（9 条）+ Stage 13.3 batch2（29 条）
- **本报告性质**：对既有自动评测与人工裁定的汇总分析；**不修改**原始评测结果、裁定文件、Agent、评分器或数据集。

---

## 0. 结论摘要

| 指标 | 数值 |
|---|---:|
| 实际纳入的唯一 review_id | **38**（不是 41，见 §1.2） |
| 人工答案裁定 | correct 35 / incorrect 3 / pending 0 |
| 人工证据裁定 | supported 35 / unsupported 3 / pending 0 |
| 自动 `auto_correct` | True 9 / False 29 |
| 自动 `verified_correct` | True 6 / False 32 |
| scorer false negative | **29**（占 38 条的 76.3%） |
| Agent 真实错误 | **3**（占 38 条的 7.9%；占 32 条风险记录的 9.4%） |
| 自动/人工一致率 | `auto_correct` 15.8%；`verified_correct` 23.7% |

**核心判断**：在这批**风险富集样本**上，自动评分器的主要问题不是“漏掉错误”，而是**大量把正确回答判错**（29 条），同时又**把 3 条无来源的编造回答判对**（`auto_correct=True`，但 `verified_correct=False`）。因此：

- 29 条 scorer false negative **不能计为 Agent 答案错误**。
- 3 条 `auto_correct=True` 但 `verified_correct=False` 的记录才是人工确认的 Agent 真实错误。
- 人工复核后的 35/38（92.1%）**只是这 38 条风险富集子集的结果**，不能冒充 50-case 全量准确率，也不能替代原始自动评测结果。

---

## 1. 数据范围与统计口径

### 1.1 输入文件

| 文件 | 内容 | 记录数 |
|---|---|---:|
| `benchmark/reviews/stage13_0_human_review_20261101.json` | Stage 13.0 风险记录 + 自动正确抽查样本 | 32 risk + 24 sample = 56 |
| `benchmark/reviews/stage13_1_batch1_verdicts_20261101.json` | 第一批人工裁定 | 9（3 risk + 6 sample） |
| `benchmark/reviews/stage13_3_batch2_verdicts_20261101.json` | 第二批人工裁定 | 29（全部 risk） |

原始评测结果按记录中的 `original_result_path` 分别对应：

- `benchmark/results/stage12_multiround_20261101_120442_r1.json`
- `benchmark/results/stage12_multiround_20261101_120442_r2.json`
- `benchmark/results/stage12_multiround_20261101_120442_r3.json`

### 1.2 41 条口径核对：实际为 38 条

任务描述为“应汇总 41 条”。按 `review_id` 去重核对后，**实际唯一记录为 38 条**：

- Stage 13.1：9 条 = 3 条 risk + 6 条 auto-correct sample；
- Stage 13.3：29 条 = 剩余 risk；
- 并集：32 条 risk（3 + 29）+ 6 条 sample = **38 条**。

**41 的来源应为重复计算**：若把 Stage 13.0 的 32 条 risk 与 Stage 13.1 的 9 条直接相加，会得到 41，但 Stage 13.1 中的 3 条 risk 已经包含在 32 条 risk 内，被计算了两次。

本报告以 **38 条唯一记录**为权威口径；未虚构另外 3 条记录，也未把重复计数计入统计。

### 1.3 统计口径（分别保留自动与人工）

| 字段 | 定义 | 来源 |
|---|---|---|
| `auto_correct` | 关键词组匹配的答案正确性，不含来源/阅读要求 | Stage 12 原始评分 `answer_correct` |
| `verified_correct` | `auto_correct` + `status=answered` + 至少 1 个 source（`grounded_correct`） | Stage 12 原始评分 |
| `human_verdict` | 人工答案裁定：correct / incorrect | Stage 13.1、13.3 |
| `human_evidence_verdict` | 人工证据裁定：supported / unsupported | Stage 13.1、13.3 |
| `scorer_false_negative` | `verified_correct=False` 且 `human_verdict=correct` | 本报告推导 |
| Agent 真实错误 | `human_verdict=incorrect` | 人工裁定 |
| 一致率（auto） | `auto_correct` 与“人工答案是否正确”一致的比例 | 本报告推导 |
| 一致率（verified） | `verified_correct` 与“人工答案是否正确”一致的比例 | 本报告推导 |

**重要口径边界**：

1. 29 条 scorer false negative 是“评分器判错”，不是 Agent 错误。
2. 3 条真实错误是“`auto_correct=True` 但无来源/未读页面”，`verified_correct` 已判 False；它们是评分器的**假阳性**，不是 false negative。
3. 本报告不把人工复核后的 92.1% 写成原始自动评测准确率。
4. 本样本是风险富集样本，不是 50-case 随机样本；所有比例不能外推为总体准确率。

---

## 2. 38 条人工复核汇总

### 2.1 按轮次

| 轮次 | 记录数 | correct | incorrect |
|---|---:|---:|---:|
| R1 | 12 | 11 | 1 |
| R2 | 17 | 15 | 2 |
| R3 | 9 | 9 | 0 |
| **合计** | **38** | **35** | **3** |

### 2.2 按组别

| 组别 | 记录数 | correct | incorrect |
|---|---:|---:|---:|
| traditional_rag | 27 | 27 | 0 |
| wiki_agent | 11 | 8 | 3 |

3 条真实错误全部来自 `wiki_agent`；但 wiki_agent 样本仅 11 条，不能据此推断总体组间差异。

### 2.3 按记录类型

| 类型 | 记录数 | correct | incorrect |
|---|---:|---:|---:|
| risk | 32 | 29 | 3 |
| auto_correct_sample | 6 | 6 | 0 |

### 2.4 按风险类型

| 风险类型 | 记录数 | correct | incorrect |
|---|---:|---:|---:|
| 拒答/否定识别 | 11 | 11 | 0 |
| 连续短语/数字格式 | 8 | 8 | 0 |
| 关系方向/顺序/代词 | 7 | 7 | 0 |
| 跨句比较/实体绑定 | 3 | 3 | 0 |
| 无来源关键词假阳性 | 3 | 0 | 3 |
| 自动正确抽查（none_suspected） | 6 | 6 | 0 |
| **合计** | **38** | **35** | **3** |

### 2.5 重复案例的轮次一致性

- 38 条记录覆盖 **20 个唯一 case**。
- 同一 case 的不同轮次**没有出现 correct/incorrect 混合**：同一 case 要么全对，要么全错。
- 因此未发现“同一问题在不同轮次结论冲突”。

### 2.6 人工证据裁定

| 证据裁定 | 记录数 | 对应答案裁定 |
|---|---:|---|
| supported | 35 | 35 条 correct |
| unsupported | 3 | 3 条 incorrect |
| pending | 0 | — |

---

## 3. 自动评分与人工裁定对照

### 3.1 自动评分结果分布（38 条）

| 指标 | True | False |
|---|---:|---:|
| `auto_correct` | 9（23.7%） | 29（76.3%） |
| `verified_correct` | 6（15.8%） | 32（84.2%） |

按记录类型拆分：

| 类型 | auto True | auto False | verified True | verified False |
|---|---:|---:|---:|---:|
| risk（32） | 3 | 29 | 0 | 32 |
| sample（6） | 6 | 0 | 6 | 0 |

### 3.2 混淆矩阵：`auto_correct` vs 人工

| | 人工 correct | 人工 incorrect |
|---|---:|---:|
| auto True | 6（TP） | 3（FP） |
| auto False | 29（FN） | 0（TN） |

- 准确率：6/38 = **15.8%**
- 精确率：6/9 = **66.7%**
- 召回率：6/35 = **17.1%**
- `auto_correct` 从未正确拒绝一条错误回答（TN=0），却把 3 条无来源回答判对。

### 3.3 混淆矩阵：`verified_correct` vs 人工

| | 人工 correct | 人工 incorrect |
|---|---:|---:|
| verified True | 6（TP） | 0（FP） |
| verified False | 29（FN） | 3（TN） |

- 准确率：9/38 = **23.7%**
- 精确率：6/6 = **100%**
- 召回率：6/35 = **17.1%**
- `verified_correct` 抓住了 3 条真实错误（全部因无 source/未回答而判 False），但把 29 条正确回答判错。

### 3.4 scorer false negative

**定义**：`verified_correct=False` 且 `human_verdict=correct`。

| 口径 | 数值 |
|---|---:|
| false negative 数 | **29** |
| 占 38 条已复核记录 | 29/38 = **76.3%** |
| 占 29 条自动负例 | 29/29 = **100%** |
| 占 32 条 risk 记录 | 29/32 = **90.6%** |
| 占 35 条人工 correct | 29/35 = **82.9%** |

29 条 false negative 全部来自 Stage 13.3 batch2；Stage 13.1 的 6 条 auto-correct sample 均为自动正确且人工 correct，不构成 false negative。

### 3.5 评分器与人工裁定一致率

| 评分器 | 一致数 | 一致率 | 说明 |
|---|---:|---:|---|
| `auto_correct` | 6/38 | **15.8%** | 仅 6 条 TP；3 条错误被误判为正确 |
| `verified_correct` | 9/38 | **23.7%** | 6 条 TP + 3 条 TN；29 条正确被误判为错误 |

> 这两个一致率**只适用于本批 38 条风险富集样本**。Stage 13.0 的抽样刻意优先选择 `only_rag_correct`、`unanswerable`、`only_agent_correct` 等风险层，不能将其与 50-case 全量自动指标直接比较。

---

## 4. 评分器误判分析

### 4.1 第一类：答案语义匹配问题（29 条 false negative）

Stage 13.2 审计把这 29 条分为：

| 子类 | 数量 | 对应风险类型 |
|---|---:|---|
| 语义正确但关键词组未命中 | 18 | 连续短语/数字 8；跨句比较/绑定 3；关系方向/顺序 7 |
| 正确拒答但拒答词未识别 | 11 | 拒答/否定 11 |
| **合计** | **29** | — |

**证据侧不是问题**：这 29 条人工均判 correct/supported，且 35 条 correct 的平均 `read_evidence_recall=0.9611`、平均 `source_precision=0.6648`；它们有实际阅读页面和来源，问题出在答案与参考词的匹配，而不是证据缺失。

#### 4.1.1 连续短语/数字格式（8 条）

- 评分器是 `keyword_group_match`：每个关键词组必须至少命中一个关键词；标点/大小写归一化后仍要求连续短语或数字形态匹配。
- 已知例子：mh02 的 `$500,000` 与参考词组要求 `['500','five'] + ['thousand']` 不一致；Stage 13.0 metadata 的评分测试备注明确记录了该策略：“`$500,000` is False for a group requiring `['500','five'] + ['thousand']`”。
- 同类问题包括金额、百分比、日期格式、`Q2 2027` 与 “second quarter 2027” 等表达差异。

#### 4.1.2 跨句比较/实体绑定（3 条）

- 答案语义正确，但关键词组覆盖不全或实体绑定顺序与参考不同。
- 评分器的 `answer_bindings` 只在部分 case 启用；这些风险记录仍主要依赖关键词组，未真正做实体绑定校验。

#### 4.1.3 关系方向/顺序（7 条）

- 答案给出了正确的 Alice→Bob→Eve / Eve→Bob→Alice 链条，并正确标注 direct/indirect；但关键词组或关系链检查对符号、箭头、顺序格式敏感，导致 `auto_correct=False`。
- 这类误判不是“答案错”，而是“参考文本与模型表述的形式差异”。

#### 4.1.4 正确拒答（11 条）

- 模型使用 “I do not know”“does not give an exact completion date”“could not confirm” 等表达；拒答判定词表未覆盖这些变体，导致 `auto_correct=False`。
- Stage 13.0 metadata 已记录：“Refusal tests cover only a small keyword set; the observed 'I do not know' / 'does not give' / 'could not confirm' gaps are not covered by existing tests.”
- 这是**拒答词表覆盖问题**，不是 Agent 拒答错误。

### 4.2 第二类：证据判定问题（3 条 false positive）

3 条真实错误的共同特征：

| 字段 | 值 |
|---|---|
| `status` | `insufficient_evidence` |
| `error` | `answered without reading any pages` |
| `read_page_ids` | `[]` |
| `source_ids` | `[]` |
| `auto_correct` | `True` |
| `verified_correct` | `False` |
| `human_verdict` | `incorrect` |
| `human_evidence_verdict` | `unsupported` |

**机制**：`auto_correct` 只做关键词组匹配，不检查 `status`、`read_page_ids`、`source_ids`。模型在未读任何页面的情况下，凭关键词生成了一段“看起来包含参考答案关键词”的内容，因此 `auto_correct=True`。`verified_correct` 要求 `status=answered` 且至少 1 个 source，才把这三条判为 False。

**结论**：

- 这 3 条是**评分器的证据判定缺口**，不是 false negative。
- `verified_correct` 能抓住“完全无来源”的情况，但它只检查来源是否存在，不检查来源是否真的支持答案；若未来模型引用无关页面，仍可能漏过。
- 不应把 `auto_correct` 当作最终答案准确率；至少应以 `verified_correct` 为底线，并增加“来源-答案蕴含”检查。

### 4.3 误判模式总结

| 模式 | 数量 | 性质 | 根因 |
|---|---:|---|---|
| 正确回答被关键词组判错 | 18 | 答案语义匹配 | 连续短语、数字格式、实体顺序、关系符号过严 |
| 正确拒答被词表判错 | 11 | 答案语义匹配 | 拒答词表覆盖不足 |
| 无来源编造被关键词判对 | 3 | 证据判定 | `auto_correct` 不看 status/source；`verified_correct` 只看 source 存在性 |

---

## 5. Agent 真实错误清单

**真实错误 = 人工 `incorrect` + 证据 `unsupported`，共 3 条。**

### 5.1 汇总

| # | case_id | round / group | 人工裁定 | 自动结果 | 证据状态 |
|---|---|---|---|---|---|
| 1 | `mh38_responsibility_synthesis` | R1 / wiki_agent | incorrect / unsupported | auto True, verified False | 未读页面，无 source |
| 2 | `mh10_direct_reports` | R2 / wiki_agent | incorrect / unsupported | auto True, verified False | 未读页面，无 source |
| 3 | `mh49_orion_upward_chain_ceo` | R2 / wiki_agent | incorrect / unsupported | auto True, verified False | 未读页面，无 source |

共同点：

- 全部为 `wiki_agent`；
- `status=insufficient_evidence`；
- `error=answered without reading any pages`；
- `read_page_ids=[]`、`source_ids=[]`；
- `verified_correct=False`，但 `auto_correct=True`；
- 均为**无证据生成/责任错配**，不是评分器漏判。

### 5.2 mh38_responsibility_synthesis

**问题**：哪位 manager 负责 technical roadmap planning 与 quarterly performance reviews，哪位 manager 拥有 annual engineering budget？

**参考答案**：Bob Jones 负责 roadmap 与 quarterly reviews；Eve Martinez 负责 annual engineering budget。

**Agent 答案**：把 technical roadmap 归给 Eve，把 annual engineering budget 归给 Bob；并加入 architecture review board、quarterly engineering objectives、budget forecasting、headcount planning、vendor spend 等语料未出现内容。

**证据核对**：

- `bob` 页面：Bob 负责 technical roadmap planning 和 quarterly performance reviews；
- `eve` 页面：Eve 负责 annual engineering budget；
- 答案把两项责任对调，并加入无来源细节；
- `read_page_ids=[]`，无任何页面可支撑。

**判定**：incorrect / unsupported。

### 5.3 mh10_direct_reports

**问题**：列出所有向 Bob Jones 汇报的人。

**参考答案**：Alice Smith、Carol Chen。

**Agent 答案**：Bob Jones 有 5 个 direct reports：Alice Smith、Carol Chen、David Kim、Frank Osei、Grace Liu。

**证据核对**：

- `bob` 页面：manages a team of twelve engineers, including Alice Smith and Carol Chen；
- `alice` / `carol` 页面支持两人向 Bob 汇报；
- `david` 页面明确 David reports to Eve；
- Frank Osei、Grace Liu 在 9 个 Wiki 页面中不存在；
- 答案把“team of twelve”误写成“five direct reports”，并虚构两人。

**判定**：incorrect / unsupported。

### 5.4 mh49_orion_upward_chain_ceo

**问题**：从 Project Orion 向上追踪 technical lead 及其 manager，语料中最高的人是谁、向谁汇报？

**参考答案**：David Kim → Eve Martinez（VP）；Eve 是语料中最高的人，直接向 CEO 汇报，CEO 姓名未提供。

**Agent 答案**：Orion → David Kim → Eve Martinez → Chief Technology Officer (CTO) → CEO。

**证据核对**：

- `project_orion` / `david` 支持 Orion→David→Eve；
- `eve` 支持 Eve 是 VP 并 reports directly to the CEO；
- 9 个页面中没有 CTO/Chief Technology Officer；
- 答案在 Eve 与 CEO 之间插入了不存在的 CTO 层级。

**判定**：incorrect / unsupported。

### 5.5 错误率口径

| 口径 | 数值 |
|---|---:|
| 3 / 38 已复核记录 | 7.9% |
| 3 / 32 risk 记录 | 9.4% |
| 3 / 11 wiki_agent 记录 | 27.3% |
| 0 / 27 traditional_rag 记录 | 0.0% |

> 27.3% 仅基于 11 条 wiki_agent 风险记录，样本很小；不能据此断言 wiki_agent 的总体错误率。但它提示：**无阅读即回答**是当前 agent 的明确失效模式，应优先修复。

---

## 6. 证据支持情况

| 指标 | 全体 38 | correct 35 | incorrect 3 |
|---|---:|---:|---:|
| 人工证据 supported | 35（92.1%） | 35 | 0 |
| 人工证据 unsupported | 3（7.9%） | 0 | 3 |
| `read_page_ids` 为空 | 3（7.9%） | 0 | 3 |
| `source_ids` 为空 | 3（7.9%） | 0 | 3 |
| 平均 `read_evidence_recall` | 0.8543 | 0.9611 | 0.0000 |
| 平均 `source_precision` | 0.6123 | 0.6648 | 0.0000 |

**关键观察**：

- 3 条真实错误的 `evidence_recall`（旧指标，计搜索/访问候选）为 1.0，但 `read_evidence_recall=0.0`、`source_precision=0.0`。这说明旧 `evidence_recall` 不能证明“实际读过并使用了证据”；评估证据支持必须看 `read_evidence_recall` / `read_page_ids` / `source_ids`。
- 35 条 correct 全部有非空 `read_page_ids` 与 `source_ids`，平均 read evidence recall 0.9611。
- 3 条 incorrect 全部无阅读、无来源，证据支持为 0。

---

## 7. 局限性与未解决问题

1. **41 vs 38 口径不一致**：输入实际唯一 review_id 为 38；41 会把 3 条 batch1 risk 重复计算。本报告未虚构缺失记录，已按 38 统计。
2. **样本非随机**：38 条是风险富集样本（32 risk + 6 自动正确抽查），不能外推为 50-case 总体准确率。
3. **18 条 auto-correct sample 未复核**：Stage 13.0 共 24 条样本，本次只复核了 batch1 的 6 条；剩余 18 条状态仍为未复核。
4. **单评审人**：人工裁定来自单一来源，未计算评审人一致性（inter-annotator agreement）。
5. **评分器是词法匹配**：没有语义 judge；29 条 false negative 说明关键词/拒答词表对 paraphrases 与数字格式覆盖不足。
6. **`verified_correct` 只检查来源存在性**：不检查来源是否支持答案；本次 3 条因无 source 被抓住，但“有 source 但不支持”的情况仍可能漏过。
7. **人工裁定与 Stage 13.0 原文件分离**：13.0 的 `human_verdict` 字段仍为空；本报告通过 review_id 关联 13.1/13.3，未回写 13.0（遵守“不修改原始文件”）。
8. **3 条错误全部在 wiki_agent**：traditional_rag 在本批 27 条中 0 错误；但组间样本不平衡，不能直接比较组间能力。
9. **未跑全量测试**：本报告只做数据完整性检查，未重新执行评测或全量测试。
10. **未解决的口径问题**：是否把 29 条 false negative 修正后重算 50-case 指标、是否将 18 条样本纳入复核、是否将 `auto_correct` 改为 `verified_correct` 作为主指标，均需后续决策。

---

## 8. Stage 13 最终结论

1. **Stage 13 人工复核完成度**：38 条唯一记录全部有裁定；32 条 risk 全覆盖；6/24 条 auto-correct sample 已复核；0 条 pending。
2. **人工结果**：correct 35/38（92.1%），incorrect 3/38（7.9%）；证据 supported 35/38，unsupported 3/38。
3. **自动结果不能直接作为最终准确率**：
   - `auto_correct` 在 38 条上仅 9 条 True，且其中 3 条是错误回答；
   - `verified_correct` 仅 6 条 True，29 条正确回答被判 False；
   - 两者与人工的一致率分别为 15.8% / 23.7%，不适合作为这批风险样本的最终结论。
4. **评分器问题分两类**：
   - 答案语义匹配问题：29 条 false negative（18 条关键词/数字/关系表达 + 11 条拒答词表）；
   - 证据判定问题：3 条 false positive（无来源编造被 `auto_correct` 判对）。
5. **Agent 真实错误**：3 条，全部是 wiki_agent 在未读页面的情况下生成无来源答案：
   - mh38 责任错配；
   - mh10 虚构 direct reports；
   - mh49 插入不存在的 CTO 层级。
6. **最终建议**：
   - 不要把 92.1% 当作 50-case 全量准确率，也不要用 `auto_correct` 替代人工裁定；
   - 优先修复 agent 的“无阅读即回答”路径（无阅读应拒答，或强制先读页面）；
   - 修复评分器的拒答词表、数字/短语匹配和来源-答案蕴含检查；
   - 复核剩余 18 条 auto-correct sample 后，再决定是否重算 50-case 指标；
   - 在 README/评测文档中分别呈现“原始自动结果”和“风险子集人工复核结果”，不得混写。

---

## 附录：数据完整性检查

| 检查项 | 结果 |
|---|---|
| Stage 13.1 + 13.3 唯一 review_id | 38 |
| Stage 13.1 与 13.3 交集 | 0 |
| 未匹配到 Stage 13.0 的 review_id | 0 |
| Stage 13.0 risk 覆盖 | 32 / 32 |
| Stage 13.0 sample 覆盖 | 6 / 24（18 条未复核） |
| pending 裁定 | 0 |
| 同一 review_id 冲突 | 0 |
| 同一 case 跨轮次混合裁定 | 0（20 个 case） |
| 38 条与 Stage 12 原始结果逐字段比对 | 38 / 38 一致（answer/status/sources/read_page_ids/auto/verified） |
| JSON 可解析 | 是 |
| 41 条口径 | 不成立；实际 38 条（重复计算 3 条 batch1 risk） |
