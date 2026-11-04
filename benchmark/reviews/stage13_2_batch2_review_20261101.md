# Stage 13.2 第二批人工复核卡片

- 生成日期：2026-11-01
- 来源：`benchmark/reviews/stage13_0_human_review_20261101.json`
- 排除：`benchmark/reviews/stage13_1_batch1_verdicts_20261101.json` 中的 9 个 review_id
- 范围：Stage 13.0 剩余 risk_records 共 **29** 条；不包含 auto-correct sample
- Wiki 页面：`benchmark/corpus/wiki/*.txt`（实际读取，不只看 reference_answer）
- 本报告不填写 `human_verdict`、`human_evidence_verdict`、`review_notes`；这些字段留空待人工裁定。

## 选择方法

- 从 Stage 13.0 的 32 条 `risk_records` 中排除第一批已裁定的 9 个 review_id。
- 第一批的 9 个 review_id 中，3 个属于 risk_records（mh38 R1、mh10 R2、mh49 R2），6 个属于 auto-correct samples；因此剩余 risk_records 为 **29** 条。
- 本批不包含 auto-correct sample，防止与第一批重复。
- 排序优先级：连续短语/数字格式 → 跨句比较/实体绑定 → 拒答/否定 → 关系方向/顺序/代词。
- 同一 case 的不同轮次分别保留，round/group/review_id 均从 Stage 13.0 读取，不自行推断。

## 概览

| 维度 | 数量/分布 |
|---|---|
| 记录总数 | 29 |
| 风险类型 | 连续短语/数字 8；跨句/绑定 3；拒答/否定 11；关系方向/顺序 7 |
| 轮次 | 10 / 11 / 8 |
| 组别 | traditional_rag 23；wiki_agent 6 |
| 语义正确但评分器漏判 | 18 |
| 正确拒答 | 11 |
| 语义错误 | 0 |
| 答案正确但证据不足 | 0 |
| 错误拒答 | 0 |

> 分类是审计观察，不是人工裁定。人工仍可在每张卡片的空白字段中给出不同结论。

## 核对约定

- `auto_correct` 是 Stage 12 的原自动评分；`verified_correct` 是原有 `grounded_correct`（answer_correct + status=answered + 至少一个 source），不是人工判定。
- `source_ids` 是模型 citation；`read_page_ids` 是实际读取页面；`accessed_page_ids` 中的未读页面只是搜索/访问候选，不能当读取证据。
- `read_evidence_recall=None` 只用于 unanswerable 题；不是失败。
- `source_precision=0.0` 对 unanswerable 题是“无 required pages”的计分口径，不代表答案质量。
- 证据片段均为 `benchmark/corpus/wiki/*.txt` 的原文子串；`model_answer` 为 Stage 12 原始完整字符串。
- “有页面支持”“页面未提及”“与页面文本不一致”是审计观察，不是人工结论。

## 总览

| # | review_id | case_id | round | group | 分类 | auto | verified |
|---:|---|---|---:|---|---|---:|---:|
| 1 | `s13_risk_r2_traditional_rag_mh02_project_budget` | `mh02_project_budget` | 2 | traditional_rag | 语义正确但被评分器漏判 | False | False |
| 2 | `s13_risk_r1_traditional_rag_mh18_orion_contributor_manager_office` | `mh18_orion_contributor_manager_office` | 1 | traditional_rag | 语义正确但被评分器漏判 | False | False |
| 3 | `s13_risk_r2_traditional_rag_mh20_phoenix_deadline_office` | `mh20_phoenix_deadline_office` | 2 | traditional_rag | 语义正确但被评分器漏判 | False | False |
| 4 | `s13_risk_r3_traditional_rag_mh25_project_launch_order` | `mh25_project_launch_order` | 3 | traditional_rag | 语义正确但被评分器漏判 | False | False |
| 5 | `s13_risk_r1_traditional_rag_mh40_walkway_data_team_synthesis` | `mh40_walkway_data_team_synthesis` | 1 | traditional_rag | 语义正确但被评分器漏判 | False | False |
| 6 | `s13_risk_r2_traditional_rag_mh40_walkway_data_team_synthesis` | `mh40_walkway_data_team_synthesis` | 2 | traditional_rag | 语义正确但被评分器漏判 | False | False |
| 7 | `s13_risk_r3_traditional_rag_mh40_walkway_data_team_synthesis` | `mh40_walkway_data_team_synthesis` | 3 | traditional_rag | 语义正确但被评分器漏判 | False | False |
| 8 | `s13_risk_r3_traditional_rag_mh54_alice_carol_role_office_disambiguation` | `mh54_alice_carol_role_office_disambiguation` | 3 | traditional_rag | 语义正确但被评分器漏判（borderline：答案省略 core team member，但问题只问 leader/floors） | False | False |
| 9 | `s13_risk_r1_traditional_rag_mh24_engineering_budget_vs_projects` | `mh24_engineering_budget_vs_projects` | 1 | traditional_rag | 语义正确但被评分器漏判 | False | False |
| 10 | `s13_risk_r2_traditional_rag_mh24_engineering_budget_vs_projects` | `mh24_engineering_budget_vs_projects` | 2 | traditional_rag | 语义正确但被评分器漏判 | False | False |
| 11 | `s13_risk_r3_wiki_agent_mh24_engineering_budget_vs_projects` | `mh24_engineering_budget_vs_projects` | 3 | wiki_agent | 语义正确但被评分器漏判 | False | False |
| 12 | `s13_risk_r1_traditional_rag_mh08_ceo_unanswerable` | `mh08_ceo_unanswerable` | 1 | traditional_rag | 正确拒答（评分器漏判） | False | False |
| 13 | `s13_risk_r3_wiki_agent_mh08_ceo_unanswerable` | `mh08_ceo_unanswerable` | 3 | wiki_agent | 正确拒答（评分器漏判） | False | False |
| 14 | `s13_risk_r2_traditional_rag_mh29_alice_salary` | `mh29_alice_salary` | 2 | traditional_rag | 正确拒答（评分器漏判） | False | False |
| 15 | `s13_risk_r1_traditional_rag_mh30_david_university` | `mh30_david_university` | 1 | traditional_rag | 正确拒答（评分器漏判） | False | False |
| 16 | `s13_risk_r2_traditional_rag_mh30_david_university` | `mh30_david_university` | 2 | traditional_rag | 正确拒答（评分器漏判） | False | False |
| 17 | `s13_risk_r1_traditional_rag_mh50_orion_exact_completion_date` | `mh50_orion_exact_completion_date` | 1 | traditional_rag | 正确拒答（评分器漏判） | False | False |
| 18 | `s13_risk_r2_traditional_rag_mh50_orion_exact_completion_date` | `mh50_orion_exact_completion_date` | 2 | traditional_rag | 正确拒答（评分器漏判） | False | False |
| 19 | `s13_risk_r3_traditional_rag_mh50_orion_exact_completion_date` | `mh50_orion_exact_completion_date` | 3 | traditional_rag | 正确拒答（评分器漏判） | False | False |
| 20 | `s13_risk_r1_traditional_rag_mh51_total_employee_count` | `mh51_total_employee_count` | 1 | traditional_rag | 正确拒答（评分器漏判） | False | False |
| 21 | `s13_risk_r2_traditional_rag_mh51_total_employee_count` | `mh51_total_employee_count` | 2 | traditional_rag | 正确拒答（评分器漏判） | False | False |
| 22 | `s13_risk_r3_traditional_rag_mh51_total_employee_count` | `mh51_total_employee_count` | 3 | traditional_rag | 正确拒答（评分器漏判） | False | False |
| 23 | `s13_risk_r1_wiki_agent_mh46_phoenix_reporting_chain` | `mh46_phoenix_reporting_chain` | 1 | wiki_agent | 语义正确但被评分器漏判 | False | False |
| 24 | `s13_risk_r2_wiki_agent_mh46_phoenix_reporting_chain` | `mh46_phoenix_reporting_chain` | 2 | wiki_agent | 语义正确但被评分器漏判 | False | False |
| 25 | `s13_risk_r1_traditional_rag_mh47_exec_level_to_phoenix_chain` | `mh47_exec_level_to_phoenix_chain` | 1 | traditional_rag | 语义正确但被评分器漏判 | False | False |
| 26 | `s13_risk_r1_wiki_agent_mh47_exec_level_to_phoenix_chain` | `mh47_exec_level_to_phoenix_chain` | 1 | wiki_agent | 语义正确但被评分器漏判 | False | False |
| 27 | `s13_risk_r2_traditional_rag_mh47_exec_level_to_phoenix_chain` | `mh47_exec_level_to_phoenix_chain` | 2 | traditional_rag | 语义正确但被评分器漏判 | False | False |
| 28 | `s13_risk_r2_wiki_agent_mh47_exec_level_to_phoenix_chain` | `mh47_exec_level_to_phoenix_chain` | 2 | wiki_agent | 语义正确但被评分器漏判 | False | False |
| 29 | `s13_risk_r3_traditional_rag_mh47_exec_level_to_phoenix_chain` | `mh47_exec_level_to_phoenix_chain` | 3 | traditional_rag | 语义正确但被评分器漏判 | False | False |

## 1. mh02_project_budget · R2 · traditional_rag

- review_id: `s13_risk_r2_traditional_rag_mh02_project_budget`
- case_id: `mh02_project_budget`
- round: `2`
- group: `traditional_rag`
- 风险类型: `continuous_phrase_or_numeric_format`
- 审计分类（非人工裁定）: 语义正确但被评分器漏判
- 原始结果: `benchmark/results/stage12_multiround_20261101_120442_r2.json`
- checkpoint: `benchmark/results/stage12_multiround_20261101_120442_r2.checkpoint.jsonl`

### 2) 原始问题

> What is the budget of the project led by Alice Smith?

### 3) 参考答案

> Project Phoenix has a budget of five hundred thousand dollars ($500,000).

### 4) 模型完整原始答案

```text
The project led by Alice Smith is Project Phoenix, and its allocated budget is $500,000.
```

### 5) 自动评分及具体评分细节

- auto_correct: `False`
- verified_correct: `False`（原有 grounded 指标，不是人工判定）
- status: `answered`
- keyword_groups: `0:true, 1:false`
- structured_checks: `n/a`
- relation_chain_pairs: `n/a`
- evidence_recall: `1.0`；read_evidence_recall: `1.0`；source_precision: `1.0`
- usage: requests=1, total_tokens=1069, latency=1.53s
- 漏判机制: keyword group 1 False: '$500,000' does not contain 'thousand' or 'hundred'

### 6) 模型提供的 source_ids

`alice, project_phoenix`

### 7) 实际 read_page_ids

`alice, project_phoenix, building_a, project_orion, bob`

- source_ids 是否都在 read_page_ids 中：是

### 8) 对应 Wiki 页面中的原始证据片段

**alice** (`benchmark/corpus/wiki/alice_smith.txt`)

> She joined the company in 2021 and currently leads Project Phoenix, an internal platform migration initiative.

**project_phoenix** (`benchmark/corpus/wiki/project_phoenix.txt`)

> It has an allocated budget of five hundred thousand dollars and a target completion date of December 31, 2026.

### 9) 逐项事实核对（审计观察，不是人工判定）

| # | 答案中的关键事实 | 页面证据 | 审计观察 |
|---|---|---|---|
| 1 | Alice Smith 领导 Project Phoenix | `alice`：…currently leads Project Phoenix… | 有页面支持 |
| 2 | Project Phoenix 预算为 $500,000 | `project_phoenix`：…allocated budget of five hundred thousand dollars… | 有页面支持 |

**审计备注（不替代人工裁定）**：答案用 “$500,000”；keyword group 1 需要 “thousand”/“hundred”，因此漏判。测试文件 `tests/wiki/test_multihop_evaluation.py` 明确把裸 `$500,000` 判 False，这是 lexical policy，不是事实错误。

### 10) 待人工填写的两个字段（留空）

- human_verdict: `________________`
- human_evidence_verdict: `________________`

### 11) review_notes（留空）

- review_notes: `________________`

---

## 2. mh18_orion_contributor_manager_office · R1 · traditional_rag

- review_id: `s13_risk_r1_traditional_rag_mh18_orion_contributor_manager_office`
- case_id: `mh18_orion_contributor_manager_office`
- round: `1`
- group: `traditional_rag`
- 风险类型: `continuous_phrase_or_numeric_format`
- 审计分类（非人工裁定）: 语义正确但被评分器漏判
- 原始结果: `benchmark/results/stage12_multiround_20261101_120442_r1.json`
- checkpoint: `benchmark/results/stage12_multiround_20261101_120442_r1.checkpoint.jsonl`

### 2) 原始问题

> Carol Chen contributes to Project Orion. Who does she report to, on which floor of Building A is that manager's office, and how many floors does Building A have?

### 3) 参考答案

> Carol Chen reports to Bob Jones. Bob Jones's office is on the third floor of Building A, and Building A has three floors.

### 4) 模型完整原始答案

```text
Carol Chen reports to Bob Jones, the Director of Engineering. His office is on the third floor of Building A, which has three floors in total.
```

### 5) 自动评分及具体评分细节

- auto_correct: `False`
- verified_correct: `False`（原有 grounded 指标，不是人工判定）
- status: `answered`
- keyword_groups: `0:true, 1:true, 2:false`
- structured_checks: `n/a`
- relation_chain_pairs: `n/a`
- evidence_recall: `0.6666666666666666`；read_evidence_recall: `0.6666666666666666`；source_precision: `1.0`
- usage: requests=1, total_tokens=1159, latency=1.78s
- 漏判机制: keyword group 2 False: 'which has three floors in total' does not contain 'building a has three floors'

- 备注: read_evidence_recall=0.6666666666666666；已读页面仍覆盖答案事实，详见逐项核对。

### 6) 模型提供的 source_ids

`carol, building_a`

### 7) 实际 read_page_ids

`carol, project_orion, building_a, building_b, project_phoenix`

- source_ids 是否都在 read_page_ids 中：是

### 8) 对应 Wiki 页面中的原始证据片段

**carol** (`benchmark/corpus/wiki/carol_chen.txt`)

> Carol reports to Bob Jones, the Director of Engineering.

**building_a** (`benchmark/corpus/wiki/building_a.txt`)

> Building A is one of two office buildings occupied by Acme Corp. It houses the platform engineering group and has three floors.

**building_a** (`benchmark/corpus/wiki/building_a.txt`)

> The third floor houses the Director of Engineering's office, currently occupied by Bob Jones, plus two conference rooms.

### 9) 逐项事实核对（审计观察，不是人工判定）

| # | 答案中的关键事实 | 页面证据 | 审计观察 |
|---|---|---|---|
| 1 | Carol Chen 向 Bob Jones 汇报 | `carol`：Carol reports to Bob Jones | 有页面支持 |
| 2 | Bob Jones 的办公室在 Building A 三楼 | `building_a`：The third floor houses the Director of Engineering’s office, currently occupied by Bob Jones | 有页面支持 |
| 3 | Building A 有三层 | `building_a`：…has three floors | 有页面支持 |

**审计备注（不替代人工裁定）**：答案 “which has three floors in total” 与 `building_a` 的 “has three floors” 同义；keyword group 2 只收连续短语。read_evidence_recall=0.667 是因为 bob 页面未读，但 `building_a` 同时支持 Bob 办公室和楼层数。

### 10) 待人工填写的两个字段（留空）

- human_verdict: `________________`
- human_evidence_verdict: `________________`

### 11) review_notes（留空）

- review_notes: `________________`

---

## 3. mh20_phoenix_deadline_office · R2 · traditional_rag

- review_id: `s13_risk_r2_traditional_rag_mh20_phoenix_deadline_office`
- case_id: `mh20_phoenix_deadline_office`
- round: `2`
- group: `traditional_rag`
- 风险类型: `continuous_phrase_or_numeric_format`
- 审计分类（非人工裁定）: 语义正确但被评分器漏判
- 原始结果: `benchmark/results/stage12_multiround_20261101_120442_r2.json`
- checkpoint: `benchmark/results/stage12_multiround_20261101_120442_r2.checkpoint.jsonl`

### 2) 原始问题

> What is the target completion date of Project Phoenix, who leads the project, and on which floor of which building is that leader's office?

### 3) 参考答案

> Project Phoenix has a target completion date of December 31, 2026. Project Phoenix is led by Alice Smith. Alice Smith's office is on the second floor of Building A.

### 4) 模型完整原始答案

```text
Project Phoenix has a target completion date of December 31, 2026, and is led by Alice Smith. Alice Smith's office is on the second floor of Building A.
```

### 5) 自动评分及具体评分细节

- auto_correct: `False`
- verified_correct: `False`（原有 grounded 指标，不是人工判定）
- status: `answered`
- keyword_groups: `0:true, 1:false, 2:true`
- structured_checks: `n/a`
- relation_chain_pairs: `n/a`
- evidence_recall: `1.0`；read_evidence_recall: `1.0`；source_precision: `1.0`
- usage: requests=1, total_tokens=1137, latency=2.08s
- 漏判机制: keyword group 1 False: 'and is led by Alice Smith' is not one of the accepted leader phrases

### 6) 模型提供的 source_ids

`project_phoenix, building_a`

### 7) 实际 read_page_ids

`project_phoenix, building_b, building_a, eve, project_orion`

- source_ids 是否都在 read_page_ids 中：是

### 8) 对应 Wiki 页面中的原始证据片段

**project_phoenix** (`benchmark/corpus/wiki/project_phoenix.txt`)

> It has an allocated budget of five hundred thousand dollars and a target completion date of December 31, 2026.

**project_phoenix** (`benchmark/corpus/wiki/project_phoenix.txt`)

> The project is led by Alice Smith, with Carol Chen as a core team member.

**alice** (`benchmark/corpus/wiki/alice_smith.txt`)

> Her office is located on the second floor of Building A.

**building_a** (`benchmark/corpus/wiki/building_a.txt`)

> The second floor has additional offices and a small kitchenette; Alice Smith's office is on this floor.

### 9) 逐项事实核对（审计观察，不是人工判定）

| # | 答案中的关键事实 | 页面证据 | 审计观察 |
|---|---|---|---|
| 1 | Project Phoenix 的目标完成日期是 December 31, 2026 | `project_phoenix`：target completion date of December 31, 2026 | 有页面支持 |
| 2 | Project Phoenix 由 Alice Smith 领导 | `project_phoenix`：The project is led by Alice Smith | 有页面支持 |
| 3 | Alice Smith 的办公室在 Building A 二楼 | `alice`/`building_a`：second floor of Building A | 有页面支持 |

**审计备注（不替代人工裁定）**：答案用 “and is led by Alice Smith”；keyword group 1 的连续短语要求 “project phoenix is led by…”/“it is led by…”，因此漏判。

### 10) 待人工填写的两个字段（留空）

- human_verdict: `________________`
- human_evidence_verdict: `________________`

### 11) review_notes（留空）

- review_notes: `________________`

---

## 4. mh25_project_launch_order · R3 · traditional_rag

- review_id: `s13_risk_r3_traditional_rag_mh25_project_launch_order`
- case_id: `mh25_project_launch_order`
- round: `3`
- group: `traditional_rag`
- 风险类型: `continuous_phrase_or_numeric_format`
- 审计分类（非人工裁定）: 语义正确但被评分器漏判
- 原始结果: `benchmark/results/stage12_multiround_20261101_120442_r3.json`
- checkpoint: `benchmark/results/stage12_multiround_20261101_120442_r3.checkpoint.jsonl`

### 2) 原始问题

> Which project was launched or initiated first, Project Phoenix or Project Orion?

### 3) 参考答案

> Project Phoenix was launched first, in early 2025. Project Orion was initiated later, in mid-2025.

### 4) 模型完整原始答案

```text
Project Phoenix was launched or initiated first: Project Phoenix launched in early 2025, while Project Orion was initiated in mid-2025.
```

### 5) 自动评分及具体评分细节

- auto_correct: `False`
- verified_correct: `False`（原有 grounded 指标，不是人工判定）
- status: `answered`
- keyword_groups: `0:false, 1:true, 2:true`
- structured_checks: `n/a`
- relation_chain_pairs: `n/a`
- evidence_recall: `1.0`；read_evidence_recall: `1.0`；source_precision: `1.0`
- usage: requests=1, total_tokens=1053, latency=1.17s
- 漏判机制: keyword group 0 False: 'launched or initiated first' is not one of the accepted phrases

### 6) 模型提供的 source_ids

`project_phoenix, project_orion`

### 7) 实际 read_page_ids

`project_orion, project_phoenix, carol, bob, alice`

- source_ids 是否都在 read_page_ids 中：是

### 8) 对应 Wiki 页面中的原始证据片段

**project_phoenix** (`benchmark/corpus/wiki/project_phoenix.txt`)

> Project Phoenix is Acme Corp's internal platform migration initiative, launched in early 2025.

**project_orion** (`benchmark/corpus/wiki/project_orion.txt`)

> Project Orion is Acme Corp's data analytics platform project, initiated in mid-2025.

### 9) 逐项事实核对（审计观察，不是人工判定）

| # | 答案中的关键事实 | 页面证据 | 审计观察 |
|---|---|---|---|
| 1 | Project Phoenix 于 early 2025 启动 | `project_phoenix`：launched in early 2025 | 有页面支持 |
| 2 | Project Orion 于 mid-2025 启动 | `project_orion`：initiated in mid-2025 | 有页面支持 |
| 3 | Project Phoenix 先启动 | early 2025 早于 mid-2025；两页分别给出时间 | 有页面支持（时间比较） |

**审计备注（不替代人工裁定）**：答案 “launched or initiated first” 不在 keyword group 0 的连续短语中；语义上 early 2025 早于 mid-2025。

### 10) 待人工填写的两个字段（留空）

- human_verdict: `________________`
- human_evidence_verdict: `________________`

### 11) review_notes（留空）

- review_notes: `________________`

---

## 5. mh40_walkway_data_team_synthesis · R1 · traditional_rag

- review_id: `s13_risk_r1_traditional_rag_mh40_walkway_data_team_synthesis`
- case_id: `mh40_walkway_data_team_synthesis`
- round: `1`
- group: `traditional_rag`
- 风险类型: `continuous_phrase_or_numeric_format`
- 审计分类（非人工裁定）: 语义正确但被评分器漏判
- 原始结果: `benchmark/results/stage12_multiround_20261101_120442_r1.json`
- checkpoint: `benchmark/results/stage12_multiround_20261101_120442_r1.checkpoint.jsonl`

### 2) 原始问题

> Alice Smith walks from her office in Building A through the covered walkway to Building B. Which team's workspace does she reach?

### 3) 参考答案

> Alice Smith reaches the data engineering team's workspace on the second floor of Building B.

### 4) 模型完整原始答案

```text
She reaches the data engineering team's workspace, which occupies the second and third floors of Building B.
```

### 5) 自动评分及具体评分细节

- auto_correct: `False`
- verified_correct: `False`（原有 grounded 指标，不是人工判定）
- status: `answered`
- keyword_groups: `0:true, 1:false`
- structured_checks: `n/a`
- relation_chain_pairs: `n/a`
- evidence_recall: `1.0`；read_evidence_recall: `1.0`；source_precision: `1.0`
- usage: requests=1, total_tokens=1130, latency=1.91s
- 漏判机制: keyword group 1 False: 'second and third floors' does not contain 'second floor'

### 6) 模型提供的 source_ids

`building_b, building_a`

### 7) 实际 read_page_ids

`building_b, alice, building_a, carol, project_phoenix`

- source_ids 是否都在 read_page_ids 中：是

### 8) 对应 Wiki 页面中的原始证据片段

**building_a** (`benchmark/corpus/wiki/building_a.txt`)

> The second floor has additional offices and a small kitchenette; Alice Smith's office is on this floor.

**building_b** (`benchmark/corpus/wiki/building_b.txt`)

> Building B is connected to Building A by a covered walkway on the second floor.

**building_b** (`benchmark/corpus/wiki/building_b.txt`)

> The second and third floors contain workspace for the data engineering team.

### 9) 逐项事实核对（审计观察，不是人工判定）

| # | 答案中的关键事实 | 页面证据 | 审计观察 |
|---|---|---|---|
| 1 | Alice 的办公室在 Building A 二楼 | `building_a`：second floor … Alice Smith’s office is on this floor | 有页面支持 |
| 2 | Building B 与 Building A 由二楼 covered walkway 连接 | `building_b`：…covered walkway on the second floor | 有页面支持 |
| 3 | Building B 二、三楼是 data engineering team 的 workspace | `building_b`：The second and third floors contain workspace for the data engineering team | 有页面支持 |

**审计备注（不替代人工裁定）**：答案 “second and third floors” 与 `building_b` 原文一致；评分器 keyword group 只收 “second floor”，因此漏判。

### 10) 待人工填写的两个字段（留空）

- human_verdict: `________________`
- human_evidence_verdict: `________________`

### 11) review_notes（留空）

- review_notes: `________________`

---

## 6. mh40_walkway_data_team_synthesis · R2 · traditional_rag

- review_id: `s13_risk_r2_traditional_rag_mh40_walkway_data_team_synthesis`
- case_id: `mh40_walkway_data_team_synthesis`
- round: `2`
- group: `traditional_rag`
- 风险类型: `continuous_phrase_or_numeric_format`
- 审计分类（非人工裁定）: 语义正确但被评分器漏判
- 原始结果: `benchmark/results/stage12_multiround_20261101_120442_r2.json`
- checkpoint: `benchmark/results/stage12_multiround_20261101_120442_r2.checkpoint.jsonl`

### 2) 原始问题

> Alice Smith walks from her office in Building A through the covered walkway to Building B. Which team's workspace does she reach?

### 3) 参考答案

> Alice Smith reaches the data engineering team's workspace on the second floor of Building B.

### 4) 模型完整原始答案

```text
She reaches the data engineering team's workspace, which occupies the second and third floors of Building B.
```

### 5) 自动评分及具体评分细节

- auto_correct: `False`
- verified_correct: `False`（原有 grounded 指标，不是人工判定）
- status: `answered`
- keyword_groups: `0:true, 1:false`
- structured_checks: `n/a`
- relation_chain_pairs: `n/a`
- evidence_recall: `1.0`；read_evidence_recall: `1.0`；source_precision: `1.0`
- usage: requests=1, total_tokens=1126, latency=1.54s
- 漏判机制: keyword group 1 False: 'second and third floors' does not contain 'second floor'

### 6) 模型提供的 source_ids

`building_b, alice, building_a`

### 7) 实际 read_page_ids

`building_b, alice, building_a, carol, project_phoenix`

- source_ids 是否都在 read_page_ids 中：是

### 8) 对应 Wiki 页面中的原始证据片段

**building_a** (`benchmark/corpus/wiki/building_a.txt`)

> The second floor has additional offices and a small kitchenette; Alice Smith's office is on this floor.

**building_b** (`benchmark/corpus/wiki/building_b.txt`)

> Building B is connected to Building A by a covered walkway on the second floor.

**building_b** (`benchmark/corpus/wiki/building_b.txt`)

> The second and third floors contain workspace for the data engineering team.

### 9) 逐项事实核对（审计观察，不是人工判定）

| # | 答案中的关键事实 | 页面证据 | 审计观察 |
|---|---|---|---|
| 1 | Alice 的办公室在 Building A 二楼 | `building_a`：second floor … Alice Smith’s office is on this floor | 有页面支持 |
| 2 | Building B 与 Building A 由二楼 covered walkway 连接 | `building_b`：…covered walkway on the second floor | 有页面支持 |
| 3 | Building B 二、三楼是 data engineering team 的 workspace | `building_b`：The second and third floors contain workspace for the data engineering team | 有页面支持 |

**审计备注（不替代人工裁定）**：答案 “second and third floors” 与 `building_b` 原文一致；评分器 keyword group 只收 “second floor”，因此漏判。

### 10) 待人工填写的两个字段（留空）

- human_verdict: `________________`
- human_evidence_verdict: `________________`

### 11) review_notes（留空）

- review_notes: `________________`

---

## 7. mh40_walkway_data_team_synthesis · R3 · traditional_rag

- review_id: `s13_risk_r3_traditional_rag_mh40_walkway_data_team_synthesis`
- case_id: `mh40_walkway_data_team_synthesis`
- round: `3`
- group: `traditional_rag`
- 风险类型: `continuous_phrase_or_numeric_format`
- 审计分类（非人工裁定）: 语义正确但被评分器漏判
- 原始结果: `benchmark/results/stage12_multiround_20261101_120442_r3.json`
- checkpoint: `benchmark/results/stage12_multiround_20261101_120442_r3.checkpoint.jsonl`

### 2) 原始问题

> Alice Smith walks from her office in Building A through the covered walkway to Building B. Which team's workspace does she reach?

### 3) 参考答案

> Alice Smith reaches the data engineering team's workspace on the second floor of Building B.

### 4) 模型完整原始答案

```text
She reaches the data engineering team's workspace, which occupies the second and third floors of Building B.
```

### 5) 自动评分及具体评分细节

- auto_correct: `False`
- verified_correct: `False`（原有 grounded 指标，不是人工判定）
- status: `answered`
- keyword_groups: `0:true, 1:false`
- structured_checks: `n/a`
- relation_chain_pairs: `n/a`
- evidence_recall: `1.0`；read_evidence_recall: `1.0`；source_precision: `1.0`
- usage: requests=1, total_tokens=1174, latency=1.98s
- 漏判机制: keyword group 1 False: 'second and third floors' does not contain 'second floor'

### 6) 模型提供的 source_ids

`building_a, building_b`

### 7) 实际 read_page_ids

`building_b, alice, building_a, carol, project_phoenix`

- source_ids 是否都在 read_page_ids 中：是

### 8) 对应 Wiki 页面中的原始证据片段

**building_a** (`benchmark/corpus/wiki/building_a.txt`)

> The second floor has additional offices and a small kitchenette; Alice Smith's office is on this floor.

**building_b** (`benchmark/corpus/wiki/building_b.txt`)

> Building B is connected to Building A by a covered walkway on the second floor.

**building_b** (`benchmark/corpus/wiki/building_b.txt`)

> The second and third floors contain workspace for the data engineering team.

### 9) 逐项事实核对（审计观察，不是人工判定）

| # | 答案中的关键事实 | 页面证据 | 审计观察 |
|---|---|---|---|
| 1 | Alice 的办公室在 Building A 二楼 | `building_a`：second floor … Alice Smith’s office is on this floor | 有页面支持 |
| 2 | Building B 与 Building A 由二楼 covered walkway 连接 | `building_b`：…covered walkway on the second floor | 有页面支持 |
| 3 | Building B 二、三楼是 data engineering team 的 workspace | `building_b`：The second and third floors contain workspace for the data engineering team | 有页面支持 |

**审计备注（不替代人工裁定）**：答案 “second and third floors” 与 `building_b` 原文一致；评分器 keyword group 只收 “second floor”，因此漏判。

### 10) 待人工填写的两个字段（留空）

- human_verdict: `________________`
- human_evidence_verdict: `________________`

### 11) review_notes（留空）

- review_notes: `________________`

---

## 8. mh54_alice_carol_role_office_disambiguation · R3 · traditional_rag

- review_id: `s13_risk_r3_traditional_rag_mh54_alice_carol_role_office_disambiguation`
- case_id: `mh54_alice_carol_role_office_disambiguation`
- round: `3`
- group: `traditional_rag`
- 风险类型: `continuous_phrase_or_numeric_format`
- 审计分类（非人工裁定）: 语义正确但被评分器漏判（borderline：答案省略 core team member，但问题只问 leader/floors）
- 原始结果: `benchmark/results/stage12_multiround_20261101_120442_r3.json`
- checkpoint: `benchmark/results/stage12_multiround_20261101_120442_r3.checkpoint.jsonl`

### 2) 原始问题

> Alice Smith and Carol Chen both work in Building A and both have roles in Project Phoenix. One leads the project and has an office on the second floor; the other is a core team member with a desk on the first floor. Which engineer leads Project Phoenix, and which floor is each engineer on?

### 3) 参考答案

> Alice Smith leads Project Phoenix and has an office on the second floor of Building A. Carol Chen is a core team member of Project Phoenix and has a desk on the first floor of Building A.

### 4) 模型完整原始答案

```text
Alice Smith leads Project Phoenix. Alice’s office is on the second floor of Building A, while Carol Chen’s desk is on the first floor of Building A.
```

### 5) 自动评分及具体评分细节

- auto_correct: `False`
- verified_correct: `False`（原有 grounded 指标，不是人工判定）
- status: `answered`
- keyword_groups: `0:true, 1:true, 2:true, 3:true, 4:false, 5:true`
- structured_checks: `answer_bindings_check=false`
- relation_chain_pairs: `n/a`
- evidence_recall: `1.0`；read_evidence_recall: `1.0`；source_precision: `1.0`
- usage: requests=1, total_tokens=1133, latency=1.27s
- 漏判机制: BORDERLINE: keyword group 4 False because the answer omits 'core team member'; leader and floors are otherwise correct

### 6) 模型提供的 source_ids

`alice, carol, building_a, project_phoenix`

### 7) 实际 read_page_ids

`alice, carol, building_a, project_phoenix, building_b`

- source_ids 是否都在 read_page_ids 中：是

### 8) 对应 Wiki 页面中的原始证据片段

**project_phoenix** (`benchmark/corpus/wiki/project_phoenix.txt`)

> The project is led by Alice Smith, with Carol Chen as a core team member.

**alice** (`benchmark/corpus/wiki/alice_smith.txt`)

> Her office is located on the second floor of Building A.

**carol** (`benchmark/corpus/wiki/carol_chen.txt`)

> Her office is on the first floor of Building A.

**building_a** (`benchmark/corpus/wiki/building_a.txt`)

> The first floor contains open workspace for staff and senior engineers, including Carol Chen's desk.

**building_a** (`benchmark/corpus/wiki/building_a.txt`)

> The second floor has additional offices and a small kitchenette; Alice Smith's office is on this floor.

### 9) 逐项事实核对（审计观察，不是人工判定）

| # | 答案中的关键事实 | 页面证据 | 审计观察 |
|---|---|---|---|
| 1 | Alice Smith 领导 Project Phoenix | `project_phoenix`：The project is led by Alice Smith | 有页面支持 |
| 2 | Alice 的办公室在 Building A 二楼 | `alice`/`building_a`：second floor of Building A | 有页面支持 |
| 3 | Carol 的 desk 在 Building A 一楼 | `building_a`：…including Carol Chen’s desk；`carol` 写 office first floor | 有页面支持 |
| 4 | Carol Chen 是 Project Phoenix 的 core team member | `project_phoenix` 有此事实；答案未复述 | 页面有支持，但答案未陈述；问题只问 leader/floors，是否影响正确性待人工裁定 |

**审计备注（不替代人工裁定）**：答案回答了 leader 和 floors；省略 “core team member”，因此 keyword group 4 False。属于 borderline：问题本身已把 Carol 描述为 core team member，答案未复述是否影响正确性由人工裁定。

### 10) 待人工填写的两个字段（留空）

- human_verdict: `________________`
- human_evidence_verdict: `________________`

### 11) review_notes（留空）

- review_notes: `________________`

---

## 9. mh24_engineering_budget_vs_projects · R1 · traditional_rag

- review_id: `s13_risk_r1_traditional_rag_mh24_engineering_budget_vs_projects`
- case_id: `mh24_engineering_budget_vs_projects`
- round: `1`
- group: `traditional_rag`
- 风险类型: `cross_sentence_relation_or_entity_binding`
- 审计分类（非人工裁定）: 语义正确但被评分器漏判
- 原始结果: `benchmark/results/stage12_multiround_20261101_120442_r1.json`
- checkpoint: `benchmark/results/stage12_multiround_20261101_120442_r1.checkpoint.jsonl`

### 2) 原始问题

> Is Acme's annual engineering budget greater than the combined budgets of Project Phoenix and Project Orion?

### 3) 参考答案

> Yes. Acme's annual engineering budget is four million dollars ($4,000,000), which is greater than the combined budgets of Project Phoenix ($500,000) and Project Orion ($300,000), totaling eight hundred thousand dollars ($800,000).

### 4) 模型完整原始答案

```text
Yes. Acme's annual engineering budget is $4,000,000, while Project Phoenix has a budget of $500,000 and Project Orion has a budget of $300,000, for a combined project budget of $800,000. The annual engineering budget is therefore greater.
```

### 5) 自动评分及具体评分细节

- auto_correct: `False`
- verified_correct: `False`（原有 grounded 指标，不是人工判定）
- status: `answered`
- keyword_groups: `0:true, 1:true`
- structured_checks: `comparison_check=false`
- relation_chain_pairs: `n/a`
- evidence_recall: `1.0`；read_evidence_recall: `1.0`；source_precision: `1.0`
- usage: requests=1, total_tokens=1130, latency=1.77s
- 漏判机制: comparison_check=False: numbers are in one sentence and 'therefore greater' is in another; no single-sentence left/right/marker match

### 6) 模型提供的 source_ids

`eve, project_phoenix, project_orion`

### 7) 实际 read_page_ids

`project_orion, project_phoenix, eve, david, bob`

- source_ids 是否都在 read_page_ids 中：是

### 8) 对应 Wiki 页面中的原始证据片段

**eve** (`benchmark/corpus/wiki/eve_martinez.txt`)

> Eve is responsible for the company's annual engineering budget of four million dollars and for setting the long-term technical vision.

**project_phoenix** (`benchmark/corpus/wiki/project_phoenix.txt`)

> It has an allocated budget of five hundred thousand dollars and a target completion date of December 31, 2026.

**project_orion** (`benchmark/corpus/wiki/project_orion.txt`)

> Project Orion has a budget of three hundred thousand dollars and is expected to launch in Q2 2027.

### 9) 逐项事实核对（审计观察，不是人工判定）

| # | 答案中的关键事实 | 页面证据 | 审计观察 |
|---|---|---|---|
| 1 | Acme annual engineering budget = $4,000,000 | `eve`：annual engineering budget of four million dollars | 有页面支持 |
| 2 | Project Phoenix budget = $500,000 | `project_phoenix`：allocated budget of five hundred thousand dollars | 有页面支持 |
| 3 | Project Orion budget = $300,000 | `project_orion`：budget of three hundred thousand dollars | 有页面支持 |
| 4 | Combined budget = $800,000 | 500,000 + 300,000 = 800,000；两个项目页分别给出数字 | 有页面支持（算术核对） |
| 5 | $4,000,000 > $800,000 | 4,000,000 > 800,000；答案的比较方向正确 | 有页面支持（算术核对） |

**审计备注（不替代人工裁定）**：数字和比较方向都有页面支持；comparison_check 因实体与标记跨句，或 Agent 答案使用 “combined amount” 别名，导致漏判。

### 10) 待人工填写的两个字段（留空）

- human_verdict: `________________`
- human_evidence_verdict: `________________`

### 11) review_notes（留空）

- review_notes: `________________`

---

## 10. mh24_engineering_budget_vs_projects · R2 · traditional_rag

- review_id: `s13_risk_r2_traditional_rag_mh24_engineering_budget_vs_projects`
- case_id: `mh24_engineering_budget_vs_projects`
- round: `2`
- group: `traditional_rag`
- 风险类型: `cross_sentence_relation_or_entity_binding`
- 审计分类（非人工裁定）: 语义正确但被评分器漏判
- 原始结果: `benchmark/results/stage12_multiround_20261101_120442_r2.json`
- checkpoint: `benchmark/results/stage12_multiround_20261101_120442_r2.checkpoint.jsonl`

### 2) 原始问题

> Is Acme's annual engineering budget greater than the combined budgets of Project Phoenix and Project Orion?

### 3) 参考答案

> Yes. Acme's annual engineering budget is four million dollars ($4,000,000), which is greater than the combined budgets of Project Phoenix ($500,000) and Project Orion ($300,000), totaling eight hundred thousand dollars ($800,000).

### 4) 模型完整原始答案

```text
Yes. Acme's annual engineering budget is $4,000,000, while the combined budgets of Project Phoenix ($500,000) and Project Orion ($300,000) total $800,000. Therefore, the annual engineering budget is greater by $3,200,000.
```

### 5) 自动评分及具体评分细节

- auto_correct: `False`
- verified_correct: `False`（原有 grounded 指标，不是人工判定）
- status: `answered`
- keyword_groups: `0:true, 1:true`
- structured_checks: `comparison_check=false`
- relation_chain_pairs: `n/a`
- evidence_recall: `1.0`；read_evidence_recall: `1.0`；source_precision: `1.0`
- usage: requests=1, total_tokens=1137, latency=1.55s
- 漏判机制: comparison_check=False: left/right entities and the comparison marker are split across sentences

### 6) 模型提供的 source_ids

`eve, project_phoenix, project_orion`

### 7) 实际 read_page_ids

`project_orion, project_phoenix, eve, david, bob`

- source_ids 是否都在 read_page_ids 中：是

### 8) 对应 Wiki 页面中的原始证据片段

**eve** (`benchmark/corpus/wiki/eve_martinez.txt`)

> Eve is responsible for the company's annual engineering budget of four million dollars and for setting the long-term technical vision.

**project_phoenix** (`benchmark/corpus/wiki/project_phoenix.txt`)

> It has an allocated budget of five hundred thousand dollars and a target completion date of December 31, 2026.

**project_orion** (`benchmark/corpus/wiki/project_orion.txt`)

> Project Orion has a budget of three hundred thousand dollars and is expected to launch in Q2 2027.

### 9) 逐项事实核对（审计观察，不是人工判定）

| # | 答案中的关键事实 | 页面证据 | 审计观察 |
|---|---|---|---|
| 1 | Acme annual engineering budget = $4,000,000 | `eve`：annual engineering budget of four million dollars | 有页面支持 |
| 2 | Project Phoenix budget = $500,000 | `project_phoenix`：allocated budget of five hundred thousand dollars | 有页面支持 |
| 3 | Project Orion budget = $300,000 | `project_orion`：budget of three hundred thousand dollars | 有页面支持 |
| 4 | Combined budget = $800,000 | 500,000 + 300,000 = 800,000；两个项目页分别给出数字 | 有页面支持（算术核对） |
| 5 | $4,000,000 > $800,000 | 4,000,000 > 800,000；答案的比较方向正确 | 有页面支持（算术核对） |

**审计备注（不替代人工裁定）**：数字和比较方向都有页面支持；comparison_check 因实体与标记跨句，或 Agent 答案使用 “combined amount” 别名，导致漏判。

### 10) 待人工填写的两个字段（留空）

- human_verdict: `________________`
- human_evidence_verdict: `________________`

### 11) review_notes（留空）

- review_notes: `________________`

---

## 11. mh24_engineering_budget_vs_projects · R3 · wiki_agent

- review_id: `s13_risk_r3_wiki_agent_mh24_engineering_budget_vs_projects`
- case_id: `mh24_engineering_budget_vs_projects`
- round: `3`
- group: `wiki_agent`
- 风险类型: `cross_sentence_relation_or_entity_binding`
- 审计分类（非人工裁定）: 语义正确但被评分器漏判
- 原始结果: `benchmark/results/stage12_multiround_20261101_120442_r3.json`
- checkpoint: `benchmark/results/stage12_multiround_20261101_120442_r3.checkpoint.jsonl`

### 2) 原始问题

> Is Acme's annual engineering budget greater than the combined budgets of Project Phoenix and Project Orion?

### 3) 参考答案

> Yes. Acme's annual engineering budget is four million dollars ($4,000,000), which is greater than the combined budgets of Project Phoenix ($500,000) and Project Orion ($300,000), totaling eight hundred thousand dollars ($800,000).

### 4) 模型完整原始答案

```text
Yes. Acme's annual engineering budget is four million dollars (managed by VP of Engineering Eve Martinez). Project Phoenix has a budget of five hundred thousand dollars, and Project Orion has a budget of three hundred thousand dollars, for a combined total of eight hundred thousand dollars. Four million dollars is well above that combined amount.
```

### 5) 自动评分及具体评分细节

- auto_correct: `False`
- verified_correct: `False`（原有 grounded 指标，不是人工判定）
- status: `answered`
- keyword_groups: `0:true, 1:true`
- structured_checks: `comparison_check=false`
- relation_chain_pairs: `n/a`
- evidence_recall: `1.0`；read_evidence_recall: `1.0`；source_precision: `1.0`
- usage: requests=3, total_tokens=2249, latency=3.59s
- 漏判机制: comparison_check=False: 'well above that combined amount' lacks an accepted right alias and spans a sentence boundary

### 6) 模型提供的 source_ids

`eve, project_phoenix, project_orion`

### 7) 实际 read_page_ids

`eve, project_orion, project_phoenix`

- source_ids 是否都在 read_page_ids 中：是
- accessed_page_ids 中未被读取的页面（搜索/访问候选，不计为读取证据）：`alice, bob, building_a, building_b, carol, david`

### 8) 对应 Wiki 页面中的原始证据片段

**eve** (`benchmark/corpus/wiki/eve_martinez.txt`)

> Eve is responsible for the company's annual engineering budget of four million dollars and for setting the long-term technical vision.

**project_phoenix** (`benchmark/corpus/wiki/project_phoenix.txt`)

> It has an allocated budget of five hundred thousand dollars and a target completion date of December 31, 2026.

**project_orion** (`benchmark/corpus/wiki/project_orion.txt`)

> Project Orion has a budget of three hundred thousand dollars and is expected to launch in Q2 2027.

### 9) 逐项事实核对（审计观察，不是人工判定）

| # | 答案中的关键事实 | 页面证据 | 审计观察 |
|---|---|---|---|
| 1 | Acme annual engineering budget = $4,000,000 | `eve`：annual engineering budget of four million dollars | 有页面支持 |
| 2 | Project Phoenix budget = $500,000 | `project_phoenix`：allocated budget of five hundred thousand dollars | 有页面支持 |
| 3 | Project Orion budget = $300,000 | `project_orion`：budget of three hundred thousand dollars | 有页面支持 |
| 4 | Combined budget = $800,000 | 500,000 + 300,000 = 800,000；两个项目页分别给出数字 | 有页面支持（算术核对） |
| 5 | $4,000,000 > $800,000 | 4,000,000 > 800,000；答案的比较方向正确 | 有页面支持（算术核对） |

**审计备注（不替代人工裁定）**：数字和比较方向都有页面支持；comparison_check 因实体与标记跨句，或 Agent 答案使用 “combined amount” 别名，导致漏判。

### 10) 待人工填写的两个字段（留空）

- human_verdict: `________________`
- human_evidence_verdict: `________________`

### 11) review_notes（留空）

- review_notes: `________________`

---

## 12. mh08_ceo_unanswerable · R1 · traditional_rag

- review_id: `s13_risk_r1_traditional_rag_mh08_ceo_unanswerable`
- case_id: `mh08_ceo_unanswerable`
- round: `1`
- group: `traditional_rag`
- 风险类型: `negation_or_refusal_recognition`
- 审计分类（非人工裁定）: 正确拒答（评分器漏判）
- 原始结果: `benchmark/results/stage12_multiround_20261101_120442_r1.json`
- checkpoint: `benchmark/results/stage12_multiround_20261101_120442_r1.checkpoint.jsonl`

### 2) 原始问题

> Who is the CEO of Acme Corp?

### 3) 参考答案

> The CEO is not named in any page. Eve Martinez reports to the CEO, but the CEO's identity is not provided.

### 4) 模型完整原始答案

```text
I do not know. The pages I read mention that the CEO has an office on the fifth floor of Building B, but none of them state the CEO's name.
```

### 5) 自动评分及具体评分细节

- auto_correct: `False`
- verified_correct: `False`（原有 grounded 指标，不是人工判定）
- status: `answered`
- keyword_groups: `无`
- structured_checks: `refusal_keyword_match=false; fabrication_keyword_match=false; fabrication_pattern_match=false; fabrication_pattern_cs_match=false; fabrication_positive_pattern_match=false`
- relation_chain_pairs: `n/a`
- evidence_recall: `None`；read_evidence_recall: `None`；source_precision: `0.0`
- usage: requests=1, total_tokens=1076, latency=1.44s
- 漏判机制: refusal_keyword_match=False: 'I do not know' and 'none ... state' are not in refusal_keywords

- 备注: 该题 unanswerable；`read_evidence_recall=None` 不是失败，`source_precision=0.0` 是“无 required pages”的口径。

### 6) 模型提供的 source_ids

`eve, building_b, bob, building_a, david`

### 7) 实际 read_page_ids

`eve, building_b, bob, building_a, david`

- source_ids 是否都在 read_page_ids 中：是

### 8) 对应 Wiki 页面中的原始证据片段

**building_b** (`benchmark/corpus/wiki/building_b.txt`)

> The fifth floor is the executive level, containing the VP of Engineering's office (currently Eve Martinez), the CEO's office, and a boardroom.

**eve** (`benchmark/corpus/wiki/eve_martinez.txt`)

> She has been in this role since 2018 and reports directly to the CEO.

**证据不足/未出现的项**

- 9 个 Wiki 页面中没有任何页面给出 CEO 的姓名。

### 9) 逐项事实核对（审计观察，不是人工判定）

| # | 答案中的关键事实 | 页面证据 | 审计观察 |
|---|---|---|---|
| 1 | Acme Corp 有 CEO 办公室 | `building_b`：…the CEO’s office… | 有页面支持 |
| 2 | Eve Martinez 向 CEO 汇报 | `eve`：…reports directly to the CEO | 有页面支持 |
| 3 | CEO 姓名未提供 | 9 个页面均未出现 CEO 姓名 | 页面未提及（答案正确拒答） |

**审计备注（不替代人工裁定）**：参考答案说 CEO 姓名未提供，答案同样拒绝给出姓名。RAG R1 因 refusal keyword 缺 “I do not know”/“none … state” 漏判；Agent R3 因 “CEO’s name is not stated” 命中 fabrication_keyword `ceo’s name is` 漏判。属于正确拒答。

### 10) 待人工填写的两个字段（留空）

- human_verdict: `________________`
- human_evidence_verdict: `________________`

### 11) review_notes（留空）

- review_notes: `________________`

---

## 13. mh08_ceo_unanswerable · R3 · wiki_agent

- review_id: `s13_risk_r3_wiki_agent_mh08_ceo_unanswerable`
- case_id: `mh08_ceo_unanswerable`
- round: `3`
- group: `wiki_agent`
- 风险类型: `negation_or_refusal_recognition`
- 审计分类（非人工裁定）: 正确拒答（评分器漏判）
- 原始结果: `benchmark/results/stage12_multiround_20261101_120442_r3.json`
- checkpoint: `benchmark/results/stage12_multiround_20261101_120442_r3.checkpoint.jsonl`

### 2) 原始问题

> Who is the CEO of Acme Corp?

### 3) 参考答案

> The CEO is not named in any page. Eve Martinez reports to the CEO, but the CEO's identity is not provided.

### 4) 模型完整原始答案

```text
The CEO's name is not stated in any of the pages I read. The pages confirm that Acme Corp has a CEO — Building B's fifth floor is the executive level containing the VP of Engineering's office (Eve Martinez), the CEO's office, and a boardroom, and Eve Martinez reports directly to the CEO — but no page identifies who the CEO is.
```

### 5) 自动评分及具体评分细节

- auto_correct: `False`
- verified_correct: `False`（原有 grounded 指标，不是人工判定）
- status: `answered`
- keyword_groups: `无`
- structured_checks: `refusal_keyword_match=true; fabrication_keyword_match=true; fabrication_pattern_match=false; fabrication_pattern_cs_match=false; fabrication_positive_pattern_match=false`
- relation_chain_pairs: `n/a`
- evidence_recall: `None`；read_evidence_recall: `None`；source_precision: `0.0`
- usage: requests=8, total_tokens=11191, latency=10.52s
- 漏判机制: refusal=True but fabrication_keyword_match=True: 'ceo's name is' is a substring of 'The CEO's name is not stated'

- 备注: 该题 unanswerable；`read_evidence_recall=None` 不是失败，`source_precision=0.0` 是“无 required pages”的口径。

### 6) 模型提供的 source_ids

`building_b, eve, bob, david, alice, carol, building_a`

### 7) 实际 read_page_ids

`alice, bob, building_a, building_b, carol, david, eve`

- source_ids 是否都在 read_page_ids 中：是
- accessed_page_ids 中未被读取的页面（搜索/访问候选，不计为读取证据）：`project_orion, project_phoenix`

### 8) 对应 Wiki 页面中的原始证据片段

**building_b** (`benchmark/corpus/wiki/building_b.txt`)

> The fifth floor is the executive level, containing the VP of Engineering's office (currently Eve Martinez), the CEO's office, and a boardroom.

**eve** (`benchmark/corpus/wiki/eve_martinez.txt`)

> She has been in this role since 2018 and reports directly to the CEO.

**证据不足/未出现的项**

- 9 个 Wiki 页面中没有任何页面给出 CEO 的姓名。

### 9) 逐项事实核对（审计观察，不是人工判定）

| # | 答案中的关键事实 | 页面证据 | 审计观察 |
|---|---|---|---|
| 1 | Acme Corp 有 CEO 办公室 | `building_b`：…the CEO’s office… | 有页面支持 |
| 2 | Eve Martinez 向 CEO 汇报 | `eve`：…reports directly to the CEO | 有页面支持 |
| 3 | CEO 姓名未提供 | 9 个页面均未出现 CEO 姓名 | 页面未提及（答案正确拒答） |

**审计备注（不替代人工裁定）**：参考答案说 CEO 姓名未提供，答案同样拒绝给出姓名。RAG R1 因 refusal keyword 缺 “I do not know”/“none … state” 漏判；Agent R3 因 “CEO’s name is not stated” 命中 fabrication_keyword `ceo’s name is` 漏判。属于正确拒答。

### 10) 待人工填写的两个字段（留空）

- human_verdict: `________________`
- human_evidence_verdict: `________________`

### 11) review_notes（留空）

- review_notes: `________________`

---

## 14. mh29_alice_salary · R2 · traditional_rag

- review_id: `s13_risk_r2_traditional_rag_mh29_alice_salary`
- case_id: `mh29_alice_salary`
- round: `2`
- group: `traditional_rag`
- 风险类型: `negation_or_refusal_recognition`
- 审计分类（非人工裁定）: 正确拒答（评分器漏判）
- 原始结果: `benchmark/results/stage12_multiround_20261101_120442_r2.json`
- checkpoint: `benchmark/results/stage12_multiround_20261101_120442_r2.checkpoint.jsonl`

### 2) 原始问题

> What is Alice Smith's salary?

### 3) 参考答案

> The corpus does not provide Alice Smith's salary; none of the 9 wiki pages states an amount.

### 4) 模型完整原始答案

```text
I do not know. None of the pages I read mention Alice Smith's salary. The pages describe her role as a senior engineer at Acme Corp, her leadership of Project Phoenix, her office location in Building A, and her education and expertise, but no compensation information is provided.
```

### 5) 自动评分及具体评分细节

- auto_correct: `False`
- verified_correct: `False`（原有 grounded 指标，不是人工判定）
- status: `answered`
- keyword_groups: `无`
- structured_checks: `refusal_keyword_match=false; fabrication_keyword_match=false; fabrication_pattern_match=false; fabrication_pattern_cs_match=false; fabrication_positive_pattern_match=false`
- relation_chain_pairs: `n/a`
- evidence_recall: `None`；read_evidence_recall: `None`；source_precision: `0.0`
- usage: requests=1, total_tokens=1080, latency=1.60s
- 漏判机制: refusal_keyword_match=False: 'None ... mention' and 'no compensation information is provided' are not in refusal_keywords

- 备注: 该题 unanswerable；`read_evidence_recall=None` 不是失败，`source_precision=0.0` 是“无 required pages”的口径。

### 6) 模型提供的 source_ids

`alice, project_phoenix, project_orion, building_a, building_b`

### 7) 实际 read_page_ids

`alice, project_phoenix, project_orion, building_a, building_b`

- source_ids 是否都在 read_page_ids 中：是

### 8) 对应 Wiki 页面中的原始证据片段

**alice** (`benchmark/corpus/wiki/alice_smith.txt`)

> Alice Smith is a senior engineer at Acme Corp. She joined the company in 2021 and currently leads Project Phoenix, an internal platform migration initiative.

**alice** (`benchmark/corpus/wiki/alice_smith.txt`)

> Alice holds a master's degree in computer science and previously worked at a startup focused on distributed systems.

**证据不足/未出现的项**

- 9 个 Wiki 页面中没有任何 salary、compensation、pay 或具体金额。

### 9) 逐项事实核对（审计观察，不是人工判定）

| # | 答案中的关键事实 | 页面证据 | 审计观察 |
|---|---|---|---|
| 1 | Alice 页面没有 salary | `alice` 页面只写角色、项目、办公室、教育和经历 | 页面未提及 salary |
| 2 | 9 个页面都没有 Alice salary | 全库检索无 salary/compensation | 页面未提及（答案正确拒答） |

**审计备注（不替代人工裁定）**：答案明确说 “pages I read … no salary”，与语料一致；refusal keyword 缺 “I do not know”/“none … mention”。属于正确拒答。

### 10) 待人工填写的两个字段（留空）

- human_verdict: `________________`
- human_evidence_verdict: `________________`

### 11) review_notes（留空）

- review_notes: `________________`

---

## 15. mh30_david_university · R1 · traditional_rag

- review_id: `s13_risk_r1_traditional_rag_mh30_david_university`
- case_id: `mh30_david_university`
- round: `1`
- group: `traditional_rag`
- 风险类型: `negation_or_refusal_recognition`
- 审计分类（非人工裁定）: 正确拒答（评分器漏判）
- 原始结果: `benchmark/results/stage12_multiround_20261101_120442_r1.json`
- checkpoint: `benchmark/results/stage12_multiround_20261101_120442_r1.checkpoint.jsonl`

### 2) 原始问题

> Which university did David Kim attend?

### 3) 参考答案

> The corpus does not provide the name of the university David Kim attended; his page only states that he holds a PhD in distributed systems, with no school name.

### 4) 模型完整原始答案

```text
I do not know. The pages mention that David Kim holds a PhD in distributed systems and has published papers on stream processing, but they do not state which university he attended.
```

### 5) 自动评分及具体评分细节

- auto_correct: `False`
- verified_correct: `False`（原有 grounded 指标，不是人工判定）
- status: `answered`
- keyword_groups: `无`
- structured_checks: `refusal_keyword_match=false; fabrication_keyword_match=false; fabrication_pattern_match=false; fabrication_pattern_cs_match=false; fabrication_positive_pattern_match=false`
- relation_chain_pairs: `n/a`
- evidence_recall: `None`；read_evidence_recall: `None`；source_precision: `0.0`
- usage: requests=1, total_tokens=958, latency=1.46s
- 漏判机制: refusal_keyword_match=False: 'do not state' is not in refusal_keywords

- 备注: 该题 unanswerable；`read_evidence_recall=None` 不是失败，`source_precision=0.0` 是“无 required pages”的口径。

### 6) 模型提供的 source_ids

`david, project_orion, eve, building_b`

### 7) 实际 read_page_ids

`david, project_orion, eve, building_b`

- source_ids 是否都在 read_page_ids 中：是

### 8) 对应 Wiki 页面中的原始证据片段

**david** (`benchmark/corpus/wiki/david_kim.txt`)

> David holds a PhD in distributed systems and has published several papers on stream processing.

**证据不足/未出现的项**

- 9 个 Wiki 页面中没有任何 university/school 名称。

### 9) 逐项事实核对（审计观察，不是人工判定）

| # | 答案中的关键事实 | 页面证据 | 审计观察 |
|---|---|---|---|
| 1 | David Kim 有 PhD in distributed systems | `david`：David holds a PhD in distributed systems | 有页面支持 |
| 2 | David 的大学名称未提供 | 9 个页面均未出现 university/school 名称 | 页面未提及（答案正确拒答） |

**审计备注（不替代人工裁定）**：答案 “do not state which university” 与 `david` 页面一致；refusal keyword 收 “does not state” 但不收 “do not state”。属于正确拒答。

### 10) 待人工填写的两个字段（留空）

- human_verdict: `________________`
- human_evidence_verdict: `________________`

### 11) review_notes（留空）

- review_notes: `________________`

---

## 16. mh30_david_university · R2 · traditional_rag

- review_id: `s13_risk_r2_traditional_rag_mh30_david_university`
- case_id: `mh30_david_university`
- round: `2`
- group: `traditional_rag`
- 风险类型: `negation_or_refusal_recognition`
- 审计分类（非人工裁定）: 正确拒答（评分器漏判）
- 原始结果: `benchmark/results/stage12_multiround_20261101_120442_r2.json`
- checkpoint: `benchmark/results/stage12_multiround_20261101_120442_r2.checkpoint.jsonl`

### 2) 原始问题

> Which university did David Kim attend?

### 3) 参考答案

> The corpus does not provide the name of the university David Kim attended; his page only states that he holds a PhD in distributed systems, with no school name.

### 4) 模型完整原始答案

```text
I do not know. The provided pages mention that David Kim holds a PhD in distributed systems, but they do not state which university he attended.
```

### 5) 自动评分及具体评分细节

- auto_correct: `False`
- verified_correct: `False`（原有 grounded 指标，不是人工判定）
- status: `answered`
- keyword_groups: `无`
- structured_checks: `refusal_keyword_match=false; fabrication_keyword_match=false; fabrication_pattern_match=false; fabrication_pattern_cs_match=false; fabrication_positive_pattern_match=false`
- relation_chain_pairs: `n/a`
- evidence_recall: `None`；read_evidence_recall: `None`；source_precision: `0.0`
- usage: requests=1, total_tokens=941, latency=1.61s
- 漏判机制: refusal_keyword_match=False: 'do not state' is not in refusal_keywords

- 备注: 该题 unanswerable；`read_evidence_recall=None` 不是失败，`source_precision=0.0` 是“无 required pages”的口径。

### 6) 模型提供的 source_ids

`david, project_orion, eve, building_b`

### 7) 实际 read_page_ids

`david, project_orion, eve, building_b`

- source_ids 是否都在 read_page_ids 中：是

### 8) 对应 Wiki 页面中的原始证据片段

**david** (`benchmark/corpus/wiki/david_kim.txt`)

> David holds a PhD in distributed systems and has published several papers on stream processing.

**证据不足/未出现的项**

- 9 个 Wiki 页面中没有任何 university/school 名称。

### 9) 逐项事实核对（审计观察，不是人工判定）

| # | 答案中的关键事实 | 页面证据 | 审计观察 |
|---|---|---|---|
| 1 | David Kim 有 PhD in distributed systems | `david`：David holds a PhD in distributed systems | 有页面支持 |
| 2 | David 的大学名称未提供 | 9 个页面均未出现 university/school 名称 | 页面未提及（答案正确拒答） |

**审计备注（不替代人工裁定）**：答案 “do not state which university” 与 `david` 页面一致；refusal keyword 收 “does not state” 但不收 “do not state”。属于正确拒答。

### 10) 待人工填写的两个字段（留空）

- human_verdict: `________________`
- human_evidence_verdict: `________________`

### 11) review_notes（留空）

- review_notes: `________________`

---

## 17. mh50_orion_exact_completion_date · R1 · traditional_rag

- review_id: `s13_risk_r1_traditional_rag_mh50_orion_exact_completion_date`
- case_id: `mh50_orion_exact_completion_date`
- round: `1`
- group: `traditional_rag`
- 风险类型: `negation_or_refusal_recognition`
- 审计分类（非人工裁定）: 正确拒答（评分器漏判）
- 原始结果: `benchmark/results/stage12_multiround_20261101_120442_r1.json`
- checkpoint: `benchmark/results/stage12_multiround_20261101_120442_r1.checkpoint.jsonl`

### 2) 原始问题

> What is the exact completion date of Project Orion?

### 3) 参考答案

> The corpus does not provide an exact completion date for Project Orion. It only states that Project Orion is expected to launch in Q2 2027; no specific day or month is given.

### 4) 模型完整原始答案

```text
The context does not give an exact completion date for Project Orion. The Project Orion page only states that it is expected to launch in Q2 2027.
```

### 5) 自动评分及具体评分细节

- auto_correct: `False`
- verified_correct: `False`（原有 grounded 指标，不是人工判定）
- status: `answered`
- keyword_groups: `无`
- structured_checks: `refusal_keyword_match=false; fabrication_keyword_match=false; fabrication_pattern_match=false; fabrication_pattern_cs_match=false; fabrication_positive_pattern_match=false`
- relation_chain_pairs: `n/a`
- evidence_recall: `None`；read_evidence_recall: `None`；source_precision: `0.0`
- usage: requests=1, total_tokens=1062, latency=1.47s
- 漏判机制: refusal_keyword_match=False: 'does not give' is not in refusal_keywords

- 备注: 该题 unanswerable；`read_evidence_recall=None` 不是失败，`source_precision=0.0` 是“无 required pages”的口径。

### 6) 模型提供的 source_ids

`project_orion`

### 7) 实际 read_page_ids

`project_orion, project_phoenix, eve, david, bob`

- source_ids 是否都在 read_page_ids 中：是

### 8) 对应 Wiki 页面中的原始证据片段

**project_orion** (`benchmark/corpus/wiki/project_orion.txt`)

> Project Orion has a budget of three hundred thousand dollars and is expected to launch in Q2 2027.

**证据不足/未出现的项**

- 9 个 Wiki 页面中没有 exact completion date、具体 day/month 或日期字符串。

### 9) 逐项事实核对（审计观察，不是人工判定）

| # | 答案中的关键事实 | 页面证据 | 审计观察 |
|---|---|---|---|
| 1 | Project Orion 只给出 Q2 2027 目标 | `project_orion`：expected to launch in Q2 2027 | 有页面支持 |
| 2 | 没有 exact completion date | 9 个页面均无精确日期 | 页面未提及（答案正确拒答） |

**审计备注（不替代人工裁定）**：答案 “does not give”/“could not confirm” 不在 refusal_keywords；但语义上是正确拒答。

### 10) 待人工填写的两个字段（留空）

- human_verdict: `________________`
- human_evidence_verdict: `________________`

### 11) review_notes（留空）

- review_notes: `________________`

---

## 18. mh50_orion_exact_completion_date · R2 · traditional_rag

- review_id: `s13_risk_r2_traditional_rag_mh50_orion_exact_completion_date`
- case_id: `mh50_orion_exact_completion_date`
- round: `2`
- group: `traditional_rag`
- 风险类型: `negation_or_refusal_recognition`
- 审计分类（非人工裁定）: 正确拒答（评分器漏判）
- 原始结果: `benchmark/results/stage12_multiround_20261101_120442_r2.json`
- checkpoint: `benchmark/results/stage12_multiround_20261101_120442_r2.checkpoint.jsonl`

### 2) 原始问题

> What is the exact completion date of Project Orion?

### 3) 参考答案

> The corpus does not provide an exact completion date for Project Orion. It only states that Project Orion is expected to launch in Q2 2027; no specific day or month is given.

### 4) 模型完整原始答案

```text
I could not confirm an exact completion date for Project Orion. The project page states only that it is expected to launch in Q2 2027, which is not a precise date.
```

### 5) 自动评分及具体评分细节

- auto_correct: `False`
- verified_correct: `False`（原有 grounded 指标，不是人工判定）
- status: `answered`
- keyword_groups: `无`
- structured_checks: `refusal_keyword_match=false; fabrication_keyword_match=false; fabrication_pattern_match=false; fabrication_pattern_cs_match=false; fabrication_positive_pattern_match=false`
- relation_chain_pairs: `n/a`
- evidence_recall: `None`；read_evidence_recall: `None`；source_precision: `0.0`
- usage: requests=1, total_tokens=1057, latency=1.12s
- 漏判机制: refusal_keyword_match=False: 'could not confirm' is not in refusal_keywords

- 备注: 该题 unanswerable；`read_evidence_recall=None` 不是失败，`source_precision=0.0` 是“无 required pages”的口径。

### 6) 模型提供的 source_ids

`project_orion`

### 7) 实际 read_page_ids

`project_orion, project_phoenix, eve, david, bob`

- source_ids 是否都在 read_page_ids 中：是

### 8) 对应 Wiki 页面中的原始证据片段

**project_orion** (`benchmark/corpus/wiki/project_orion.txt`)

> Project Orion has a budget of three hundred thousand dollars and is expected to launch in Q2 2027.

**证据不足/未出现的项**

- 9 个 Wiki 页面中没有 exact completion date、具体 day/month 或日期字符串。

### 9) 逐项事实核对（审计观察，不是人工判定）

| # | 答案中的关键事实 | 页面证据 | 审计观察 |
|---|---|---|---|
| 1 | Project Orion 只给出 Q2 2027 目标 | `project_orion`：expected to launch in Q2 2027 | 有页面支持 |
| 2 | 没有 exact completion date | 9 个页面均无精确日期 | 页面未提及（答案正确拒答） |

**审计备注（不替代人工裁定）**：答案 “does not give”/“could not confirm” 不在 refusal_keywords；但语义上是正确拒答。

### 10) 待人工填写的两个字段（留空）

- human_verdict: `________________`
- human_evidence_verdict: `________________`

### 11) review_notes（留空）

- review_notes: `________________`

---

## 19. mh50_orion_exact_completion_date · R3 · traditional_rag

- review_id: `s13_risk_r3_traditional_rag_mh50_orion_exact_completion_date`
- case_id: `mh50_orion_exact_completion_date`
- round: `3`
- group: `traditional_rag`
- 风险类型: `negation_or_refusal_recognition`
- 审计分类（非人工裁定）: 正确拒答（评分器漏判）
- 原始结果: `benchmark/results/stage12_multiround_20261101_120442_r3.json`
- checkpoint: `benchmark/results/stage12_multiround_20261101_120442_r3.checkpoint.jsonl`

### 2) 原始问题

> What is the exact completion date of Project Orion?

### 3) 参考答案

> The corpus does not provide an exact completion date for Project Orion. It only states that Project Orion is expected to launch in Q2 2027; no specific day or month is given.

### 4) 模型完整原始答案

```text
I could not confirm an exact completion date for Project Orion. The available page only states that Project Orion is expected to launch in Q2 2027, which is not an exact date.
```

### 5) 自动评分及具体评分细节

- auto_correct: `False`
- verified_correct: `False`（原有 grounded 指标，不是人工判定）
- status: `answered`
- keyword_groups: `无`
- structured_checks: `refusal_keyword_match=false; fabrication_keyword_match=false; fabrication_pattern_match=false; fabrication_pattern_cs_match=false; fabrication_positive_pattern_match=false`
- relation_chain_pairs: `n/a`
- evidence_recall: `None`；read_evidence_recall: `None`；source_precision: `0.0`
- usage: requests=1, total_tokens=1052, latency=1.45s
- 漏判机制: refusal_keyword_match=False: 'could not confirm' is not in refusal_keywords

- 备注: 该题 unanswerable；`read_evidence_recall=None` 不是失败，`source_precision=0.0` 是“无 required pages”的口径。

### 6) 模型提供的 source_ids

`project_orion`

### 7) 实际 read_page_ids

`project_orion, project_phoenix, eve, david, bob`

- source_ids 是否都在 read_page_ids 中：是

### 8) 对应 Wiki 页面中的原始证据片段

**project_orion** (`benchmark/corpus/wiki/project_orion.txt`)

> Project Orion has a budget of three hundred thousand dollars and is expected to launch in Q2 2027.

**证据不足/未出现的项**

- 9 个 Wiki 页面中没有 exact completion date、具体 day/month 或日期字符串。

### 9) 逐项事实核对（审计观察，不是人工判定）

| # | 答案中的关键事实 | 页面证据 | 审计观察 |
|---|---|---|---|
| 1 | Project Orion 只给出 Q2 2027 目标 | `project_orion`：expected to launch in Q2 2027 | 有页面支持 |
| 2 | 没有 exact completion date | 9 个页面均无精确日期 | 页面未提及（答案正确拒答） |

**审计备注（不替代人工裁定）**：答案 “does not give”/“could not confirm” 不在 refusal_keywords；但语义上是正确拒答。

### 10) 待人工填写的两个字段（留空）

- human_verdict: `________________`
- human_evidence_verdict: `________________`

### 11) review_notes（留空）

- review_notes: `________________`

---

## 20. mh51_total_employee_count · R1 · traditional_rag

- review_id: `s13_risk_r1_traditional_rag_mh51_total_employee_count`
- case_id: `mh51_total_employee_count`
- round: `1`
- group: `traditional_rag`
- 风险类型: `negation_or_refusal_recognition`
- 审计分类（非人工裁定）: 正确拒答（评分器漏判）
- 原始结果: `benchmark/results/stage12_multiround_20261101_120442_r1.json`
- checkpoint: `benchmark/results/stage12_multiround_20261101_120442_r1.checkpoint.jsonl`

### 2) 原始问题

> How many employees work at Acme Corp in total?

### 3) 参考答案

> The corpus does not state Acme Corp's total employee count. It only says that Bob Jones manages a team of twelve engineers, which is not the same as the company total.

### 4) 模型完整原始答案

```text
I do not know. The provided pages describe several individual employees (Carol Chen, Alice Smith, David Kim) and their roles, offices, and projects, but none of them state the total number of employees at Acme Corp.
```

### 5) 自动评分及具体评分细节

- auto_correct: `False`
- verified_correct: `False`（原有 grounded 指标，不是人工判定）
- status: `answered`
- keyword_groups: `无`
- structured_checks: `refusal_keyword_match=false; fabrication_keyword_match=false; fabrication_pattern_match=false; fabrication_pattern_cs_match=false; fabrication_positive_pattern_match=false`
- relation_chain_pairs: `n/a`
- evidence_recall: `None`；read_evidence_recall: `None`；source_precision: `0.0`
- usage: requests=1, total_tokens=1085, latency=1.24s
- 漏判机制: refusal_keyword_match=False: 'I do not know' and 'none ... state' are not in refusal_keywords

- 备注: 该题 unanswerable；`read_evidence_recall=None` 不是失败，`source_precision=0.0` 是“无 required pages”的口径。

### 6) 模型提供的 source_ids

`carol, project_orion, building_b, alice, david`

### 7) 实际 read_page_ids

`carol, project_orion, building_b, alice, david`

- source_ids 是否都在 read_page_ids 中：是

### 8) 对应 Wiki 页面中的原始证据片段

**alice** (`benchmark/corpus/wiki/alice_smith.txt`)

> Alice Smith is a senior engineer at Acme Corp. She joined the company in 2021 and currently leads Project Phoenix, an internal platform migration initiative.

**carol** (`benchmark/corpus/wiki/carol_chen.txt`)

> Carol Chen is a staff engineer at Acme Corp, specializing in backend services and API design.

**david** (`benchmark/corpus/wiki/david_kim.txt`)

> David Kim is a principal engineer at Acme Corp and the technical lead for Project Orion.

**证据不足/未出现的项**

- 9 个 Wiki 页面中没有任何 company-wide total employee count/headcount。
- 模型 read_page_ids 不含 bob；bob 页面只写 “a team of twelve engineers”，也没有 company-wide total。

### 9) 逐项事实核对（审计观察，不是人工判定）

| # | 答案中的关键事实 | 页面证据 | 审计观察 |
|---|---|---|---|
| 1 | 模型已读页面描述了 Carol/Alice/David 等员工 | `alice`/`carol`/`david` 页面分别给出三名员工的角色；三页都在 read_page_ids 中 | 有页面支持 |
| 2 | 公司总员工数未提供 | 9 个页面均无 company-wide total；bob 页面（模型未读）也只有 team of twelve | 页面未提及（答案正确拒答） |

**审计备注（不替代人工裁定）**：答案 “I do not know … none … state” 正确；refusal keyword 缺这些短语。模型 read_page_ids 不含 bob；bob 页面也没有 company-wide total，因此不影响拒答的正确性。注意 mh51 是 manual_review case，本卡片只做事实/证据核对。

### 10) 待人工填写的两个字段（留空）

- human_verdict: `________________`
- human_evidence_verdict: `________________`

### 11) review_notes（留空）

- review_notes: `________________`

---

## 21. mh51_total_employee_count · R2 · traditional_rag

- review_id: `s13_risk_r2_traditional_rag_mh51_total_employee_count`
- case_id: `mh51_total_employee_count`
- round: `2`
- group: `traditional_rag`
- 风险类型: `negation_or_refusal_recognition`
- 审计分类（非人工裁定）: 正确拒答（评分器漏判）
- 原始结果: `benchmark/results/stage12_multiround_20261101_120442_r2.json`
- checkpoint: `benchmark/results/stage12_multiround_20261101_120442_r2.checkpoint.jsonl`

### 2) 原始问题

> How many employees work at Acme Corp in total?

### 3) 参考答案

> The corpus does not state Acme Corp's total employee count. It only says that Bob Jones manages a team of twelve engineers, which is not the same as the company total.

### 4) 模型完整原始答案

```text
I do not know. The pages I read describe individual employees (Carol Chen, Alice Smith, David Kim) and facilities/projects, but none states the total number of Acme Corp employees.
```

### 5) 自动评分及具体评分细节

- auto_correct: `False`
- verified_correct: `False`（原有 grounded 指标，不是人工判定）
- status: `answered`
- keyword_groups: `无`
- structured_checks: `refusal_keyword_match=false; fabrication_keyword_match=false; fabrication_pattern_match=false; fabrication_pattern_cs_match=false; fabrication_positive_pattern_match=false`
- relation_chain_pairs: `n/a`
- evidence_recall: `None`；read_evidence_recall: `None`；source_precision: `0.0`
- usage: requests=1, total_tokens=1079, latency=1.26s
- 漏判机制: refusal_keyword_match=False: 'I do not know'/'none states' are not in refusal_keywords

- 备注: 该题 unanswerable；`read_evidence_recall=None` 不是失败，`source_precision=0.0` 是“无 required pages”的口径。

### 6) 模型提供的 source_ids

`carol, project_orion, building_b, alice, david`

### 7) 实际 read_page_ids

`carol, project_orion, building_b, alice, david`

- source_ids 是否都在 read_page_ids 中：是

### 8) 对应 Wiki 页面中的原始证据片段

**alice** (`benchmark/corpus/wiki/alice_smith.txt`)

> Alice Smith is a senior engineer at Acme Corp. She joined the company in 2021 and currently leads Project Phoenix, an internal platform migration initiative.

**carol** (`benchmark/corpus/wiki/carol_chen.txt`)

> Carol Chen is a staff engineer at Acme Corp, specializing in backend services and API design.

**david** (`benchmark/corpus/wiki/david_kim.txt`)

> David Kim is a principal engineer at Acme Corp and the technical lead for Project Orion.

**证据不足/未出现的项**

- 9 个 Wiki 页面中没有任何 company-wide total employee count/headcount。
- 模型 read_page_ids 不含 bob；bob 页面只写 “a team of twelve engineers”，也没有 company-wide total。

### 9) 逐项事实核对（审计观察，不是人工判定）

| # | 答案中的关键事实 | 页面证据 | 审计观察 |
|---|---|---|---|
| 1 | 模型已读页面描述了 Carol/Alice/David 等员工 | `alice`/`carol`/`david` 页面分别给出三名员工的角色；三页都在 read_page_ids 中 | 有页面支持 |
| 2 | 公司总员工数未提供 | 9 个页面均无 company-wide total；bob 页面（模型未读）也只有 team of twelve | 页面未提及（答案正确拒答） |

**审计备注（不替代人工裁定）**：答案 “I do not know … none … state” 正确；refusal keyword 缺这些短语。模型 read_page_ids 不含 bob；bob 页面也没有 company-wide total，因此不影响拒答的正确性。注意 mh51 是 manual_review case，本卡片只做事实/证据核对。

### 10) 待人工填写的两个字段（留空）

- human_verdict: `________________`
- human_evidence_verdict: `________________`

### 11) review_notes（留空）

- review_notes: `________________`

---

## 22. mh51_total_employee_count · R3 · traditional_rag

- review_id: `s13_risk_r3_traditional_rag_mh51_total_employee_count`
- case_id: `mh51_total_employee_count`
- round: `3`
- group: `traditional_rag`
- 风险类型: `negation_or_refusal_recognition`
- 审计分类（非人工裁定）: 正确拒答（评分器漏判）
- 原始结果: `benchmark/results/stage12_multiround_20261101_120442_r3.json`
- checkpoint: `benchmark/results/stage12_multiround_20261101_120442_r3.checkpoint.jsonl`

### 2) 原始问题

> How many employees work at Acme Corp in total?

### 3) 参考答案

> The corpus does not state Acme Corp's total employee count. It only says that Bob Jones manages a team of twelve engineers, which is not the same as the company total.

### 4) 模型完整原始答案

```text
I do not know. The provided pages describe individual employees (Carol Chen, Alice Smith, David Kim) and some projects and buildings, but they do not state the total number of employees at Acme Corp.
```

### 5) 自动评分及具体评分细节

- auto_correct: `False`
- verified_correct: `False`（原有 grounded 指标，不是人工判定）
- status: `answered`
- keyword_groups: `无`
- structured_checks: `refusal_keyword_match=false; fabrication_keyword_match=false; fabrication_pattern_match=false; fabrication_pattern_cs_match=false; fabrication_positive_pattern_match=false`
- relation_chain_pairs: `n/a`
- evidence_recall: `None`；read_evidence_recall: `None`；source_precision: `0.0`
- usage: requests=1, total_tokens=1061, latency=1.32s
- 漏判机制: refusal_keyword_match=False: 'I do not know'/'none ... state' are not in refusal_keywords

- 备注: 该题 unanswerable；`read_evidence_recall=None` 不是失败，`source_precision=0.0` 是“无 required pages”的口径。

### 6) 模型提供的 source_ids

`carol, project_orion, building_b, alice, david`

### 7) 实际 read_page_ids

`carol, project_orion, building_b, alice, david`

- source_ids 是否都在 read_page_ids 中：是

### 8) 对应 Wiki 页面中的原始证据片段

**alice** (`benchmark/corpus/wiki/alice_smith.txt`)

> Alice Smith is a senior engineer at Acme Corp. She joined the company in 2021 and currently leads Project Phoenix, an internal platform migration initiative.

**carol** (`benchmark/corpus/wiki/carol_chen.txt`)

> Carol Chen is a staff engineer at Acme Corp, specializing in backend services and API design.

**david** (`benchmark/corpus/wiki/david_kim.txt`)

> David Kim is a principal engineer at Acme Corp and the technical lead for Project Orion.

**证据不足/未出现的项**

- 9 个 Wiki 页面中没有任何 company-wide total employee count/headcount。
- 模型 read_page_ids 不含 bob；bob 页面只写 “a team of twelve engineers”，也没有 company-wide total。

### 9) 逐项事实核对（审计观察，不是人工判定）

| # | 答案中的关键事实 | 页面证据 | 审计观察 |
|---|---|---|---|
| 1 | 模型已读页面描述了 Carol/Alice/David 等员工 | `alice`/`carol`/`david` 页面分别给出三名员工的角色；三页都在 read_page_ids 中 | 有页面支持 |
| 2 | 公司总员工数未提供 | 9 个页面均无 company-wide total；bob 页面（模型未读）也只有 team of twelve | 页面未提及（答案正确拒答） |

**审计备注（不替代人工裁定）**：答案 “I do not know … none … state” 正确；refusal keyword 缺这些短语。模型 read_page_ids 不含 bob；bob 页面也没有 company-wide total，因此不影响拒答的正确性。注意 mh51 是 manual_review case，本卡片只做事实/证据核对。

### 10) 待人工填写的两个字段（留空）

- human_verdict: `________________`
- human_evidence_verdict: `________________`

### 11) review_notes（留空）

- review_notes: `________________`

---

## 23. mh46_phoenix_reporting_chain · R1 · wiki_agent

- review_id: `s13_risk_r1_wiki_agent_mh46_phoenix_reporting_chain`
- case_id: `mh46_phoenix_reporting_chain`
- round: `1`
- group: `wiki_agent`
- 风险类型: `relation_direction_or_order`
- 审计分类（非人工裁定）: 语义正确但被评分器漏判
- 原始结果: `benchmark/results/stage12_multiround_20261101_120442_r1.json`
- checkpoint: `benchmark/results/stage12_multiround_20261101_120442_r1.checkpoint.jsonl`

### 2) 原始问题

> Starting from Project Phoenix, trace the chain from the project lead to that person's direct manager, and then to that manager's direct manager. List the three people in order, state which adjacent links are direct reporting relationships, and state whether the relationship between the first and last person is direct or indirect.

### 3) 参考答案

> Project Phoenix is led by Alice Smith. Alice Smith directly reports to Bob Jones, and Bob Jones directly reports to Eve Martinez. The chain is Alice Smith, then Bob Jones, then Eve Martinez. Eve Martinez is two reporting levels above Alice Smith, so their connection is indirect.

### 4) 模型完整原始答案

```text
The chain from Project Phoenix is:

1. Alice Smith — senior engineer at Acme Corp and the lead of Project Phoenix.
2. Bob Jones — Director of Engineering.
3. Eve Martinez — VP of Engineering.

Adjacent links:
- Alice Smith → Bob Jones: direct reporting relationship. Alice's page states she "reports directly to Bob Jones, the Director of Engineering," and Bob's page confirms he manages a team including Alice Smith.
- Bob Jones → Eve Martinez: direct reporting relationship. Bob's page states he "reports to Eve Martinez, the VP of Engineering."

Relationship between the first and last person (Alice Smith and Eve Martinez): indirect. Alice reports directly to Bob, who reports directly to Eve, so Alice is two levels below Eve rather than a direct report.
```

### 5) 自动评分及具体评分细节

- auto_correct: `False`
- verified_correct: `False`（原有 grounded 指标，不是人工判定）
- status: `answered`
- keyword_groups: `0:true, 1:true, 2:true, 3:true, 4:true`
- structured_checks: `relation_chain_check=false`
- relation_chain_pairs: `direct(alice->bob)=true; direct(bob->eve)=true; indirect(alice->eve)=true; forbidden_direct(alice->eve)=false; leadership(alice->project phoenix)=false`
- evidence_recall: `1.0`；read_evidence_recall: `1.0`；source_precision: `1.0`
- usage: requests=6, total_tokens=5615, latency=7.97s
- 漏判机制: relation_chain leadership_pairs=False: 'lead of Project Phoenix' is not in markers leads/led/leader

### 6) 模型提供的 source_ids

`project_phoenix, alice, bob`

### 7) 实际 read_page_ids

`alice, bob, project_phoenix`

- source_ids 是否都在 read_page_ids 中：是
- accessed_page_ids 中未被读取的页面（搜索/访问候选，不计为读取证据）：`building_a, building_b, carol, david, eve, project_orion`

### 8) 对应 Wiki 页面中的原始证据片段

**project_phoenix** (`benchmark/corpus/wiki/project_phoenix.txt`)

> The project is led by Alice Smith, with Carol Chen as a core team member.

**alice** (`benchmark/corpus/wiki/alice_smith.txt`)

> Alice reports directly to Bob Jones, the Director of Engineering.

**bob** (`benchmark/corpus/wiki/bob_jones.txt`)

> He reports to Eve Martinez, the VP of Engineering.

**eve** (`benchmark/corpus/wiki/eve_martinez.txt`)

> Eve manages two directors: Bob Jones, who leads the platform engineering group, and David Kim, who leads the data engineering group.

### 9) 逐项事实核对（审计观察，不是人工判定）

| # | 答案中的关键事实 | 页面证据 | 审计观察 |
|---|---|---|---|
| 1 | Alice Smith 领导 Project Phoenix | `project_phoenix`：The project is led by Alice Smith | 有页面支持 |
| 2 | Alice 直接向 Bob 汇报 | `alice`：Alice reports directly to Bob Jones | 有页面支持 |
| 3 | Bob 向 Eve 汇报 | `bob`：He reports to Eve Martinez | 有页面支持 |
| 4 | Eve 管理 Bob，Bob 领导 platform engineering group | `eve`：Eve manages two directors: Bob Jones… | 有页面支持 |
| 5 | Alice 到 Eve 是 indirect（两级） | Alice→Bob→Eve；页面分别给出两条 direct 关系 | 有页面支持（链式推导） |

**审计备注（不替代人工裁定）**：答案 chain 正确；relation_chain leadership_pairs 只收 leads/led/leader，答案用 “lead of Project Phoenix”，因此漏判。

### 10) 待人工填写的两个字段（留空）

- human_verdict: `________________`
- human_evidence_verdict: `________________`

### 11) review_notes（留空）

- review_notes: `________________`

---

## 24. mh46_phoenix_reporting_chain · R2 · wiki_agent

- review_id: `s13_risk_r2_wiki_agent_mh46_phoenix_reporting_chain`
- case_id: `mh46_phoenix_reporting_chain`
- round: `2`
- group: `wiki_agent`
- 风险类型: `relation_direction_or_order`
- 审计分类（非人工裁定）: 语义正确但被评分器漏判
- 原始结果: `benchmark/results/stage12_multiround_20261101_120442_r2.json`
- checkpoint: `benchmark/results/stage12_multiround_20261101_120442_r2.checkpoint.jsonl`

### 2) 原始问题

> Starting from Project Phoenix, trace the chain from the project lead to that person's direct manager, and then to that manager's direct manager. List the three people in order, state which adjacent links are direct reporting relationships, and state whether the relationship between the first and last person is direct or indirect.

### 3) 参考答案

> Project Phoenix is led by Alice Smith. Alice Smith directly reports to Bob Jones, and Bob Jones directly reports to Eve Martinez. The chain is Alice Smith, then Bob Jones, then Eve Martinez. Eve Martinez is two reporting levels above Alice Smith, so their connection is indirect.

### 4) 模型完整原始答案

```text
The chain starting from Project Phoenix is: Alice Smith (project lead) → Bob Jones → Eve Martinez.

1. Alice Smith — senior engineer and lead of Project Phoenix. She reports directly to Bob Jones, Director of Engineering. (Alice → Bob is a DIRECT reporting relationship.)
2. Bob Jones — Director of Engineering, who manages a team of twelve engineers including Alice. He reports to Eve Martinez, VP of Engineering. (Bob → Eve is a DIRECT reporting relationship.)
3. Eve Martinez — VP of Engineering, who manages Bob Jones as one of her two directors.

The relationship between the first person (Alice Smith) and the last person (Eve Martinez) is INDIRECT: Alice reports to Bob, and Bob reports to Eve, so Alice is two levels below Eve in the chain of command, not a direct report.
```

### 5) 自动评分及具体评分细节

- auto_correct: `False`
- verified_correct: `False`（原有 grounded 指标，不是人工判定）
- status: `answered`
- keyword_groups: `0:true, 1:true, 2:true, 3:true, 4:true`
- structured_checks: `relation_chain_check=false`
- relation_chain_pairs: `direct(alice->bob)=true; direct(bob->eve)=true; indirect(alice->eve)=true; forbidden_direct(alice->eve)=false; leadership(alice->project phoenix)=false`
- evidence_recall: `1.0`；read_evidence_recall: `1.0`；source_precision: `1.0`
- usage: requests=7, total_tokens=7241, latency=9.23s
- 漏判机制: relation_chain leadership_pairs=False: 'lead of Project Phoenix' is not in markers leads/led/leader

### 6) 模型提供的 source_ids

`project_phoenix, alice, bob, eve`

### 7) 实际 read_page_ids

`alice, bob, eve, project_phoenix`

- source_ids 是否都在 read_page_ids 中：是
- accessed_page_ids 中未被读取的页面（搜索/访问候选，不计为读取证据）：`building_a, building_b, carol, david, project_orion`

### 8) 对应 Wiki 页面中的原始证据片段

**project_phoenix** (`benchmark/corpus/wiki/project_phoenix.txt`)

> The project is led by Alice Smith, with Carol Chen as a core team member.

**alice** (`benchmark/corpus/wiki/alice_smith.txt`)

> Alice reports directly to Bob Jones, the Director of Engineering.

**bob** (`benchmark/corpus/wiki/bob_jones.txt`)

> He reports to Eve Martinez, the VP of Engineering.

**eve** (`benchmark/corpus/wiki/eve_martinez.txt`)

> Eve manages two directors: Bob Jones, who leads the platform engineering group, and David Kim, who leads the data engineering group.

### 9) 逐项事实核对（审计观察，不是人工判定）

| # | 答案中的关键事实 | 页面证据 | 审计观察 |
|---|---|---|---|
| 1 | Alice Smith 领导 Project Phoenix | `project_phoenix`：The project is led by Alice Smith | 有页面支持 |
| 2 | Alice 直接向 Bob 汇报 | `alice`：Alice reports directly to Bob Jones | 有页面支持 |
| 3 | Bob 向 Eve 汇报 | `bob`：He reports to Eve Martinez | 有页面支持 |
| 4 | Eve 管理 Bob，Bob 领导 platform engineering group | `eve`：Eve manages two directors: Bob Jones… | 有页面支持 |
| 5 | Alice 到 Eve 是 indirect（两级） | Alice→Bob→Eve；页面分别给出两条 direct 关系 | 有页面支持（链式推导） |

**审计备注（不替代人工裁定）**：答案 chain 正确；relation_chain leadership_pairs 只收 leads/led/leader，答案用 “lead of Project Phoenix”，因此漏判。

### 10) 待人工填写的两个字段（留空）

- human_verdict: `________________`
- human_evidence_verdict: `________________`

### 11) review_notes（留空）

- review_notes: `________________`

---

## 25. mh47_exec_level_to_phoenix_chain · R1 · traditional_rag

- review_id: `s13_risk_r1_traditional_rag_mh47_exec_level_to_phoenix_chain`
- case_id: `mh47_exec_level_to_phoenix_chain`
- round: `1`
- group: `traditional_rag`
- 风险类型: `relation_direction_or_order`
- 审计分类（非人工裁定）: 语义正确但被评分器漏判
- 原始结果: `benchmark/results/stage12_multiround_20261101_120442_r1.json`
- checkpoint: `benchmark/results/stage12_multiround_20261101_120442_r1.checkpoint.jsonl`

### 2) 原始问题

> Starting from the executive level of Building B, trace the relationships to the engineer who leads Project Phoenix: identify the VP whose office is on that floor, the director she manages who leads the platform engineering group, and the engineer who reports to that director and leads Project Phoenix. List the chain and state which relationship is indirect.

### 3) 参考答案

> Building B's fifth floor is the executive level, where Eve Martinez, the VP of Engineering, has her office. Eve directly manages Bob Jones, who leads the platform engineering group. Bob directly manages Alice Smith, and Alice leads Project Phoenix. Eve's relationship to Alice is indirect, because it passes through Bob Jones.

### 4) 模型完整原始答案

```text
Chain: Eve Martinez (VP of Engineering, office on the fifth/executive floor of Building B) → Bob Jones (the director she manages who leads platform engineering) → Alice Smith (the engineer who reports to Bob and leads Project Phoenix). The indirect relationship is between Eve Martinez and Alice Smith: Eve manages Alice only through Bob Jones, not directly.
```

### 5) 自动评分及具体评分细节

- auto_correct: `False`
- verified_correct: `False`（原有 grounded 指标，不是人工判定）
- status: `answered`
- keyword_groups: `0:true, 1:true, 2:true, 3:true, 4:true, 5:true`
- structured_checks: `relation_chain_check=false`
- relation_chain_pairs: `direct(eve->bob)=false; direct(bob->alice)=false; indirect(eve->alice)=true; forbidden_direct(eve->alice)=false; leadership(alice->project phoenix)=true`
- evidence_recall: `0.8`；read_evidence_recall: `0.8`；source_precision: `1.0`
- usage: requests=1, total_tokens=1512, latency=3.68s
- 漏判机制: relation_chain_check=False while all keyword groups match: direct/indirect pairs require literal direct(ly) near named pairs

- 备注: read_evidence_recall=0.8；已读页面仍覆盖答案事实，详见逐项核对。

### 6) 模型提供的 source_ids

`building_b, eve, bob, project_phoenix`

### 7) 实际 read_page_ids

`building_b, eve, bob, project_phoenix, building_a`

- source_ids 是否都在 read_page_ids 中：是

### 8) 对应 Wiki 页面中的原始证据片段

**building_b** (`benchmark/corpus/wiki/building_b.txt`)

> The fifth floor is the executive level, containing the VP of Engineering's office (currently Eve Martinez), the CEO's office, and a boardroom.

**eve** (`benchmark/corpus/wiki/eve_martinez.txt`)

> Eve manages two directors: Bob Jones, who leads the platform engineering group, and David Kim, who leads the data engineering group.

**bob** (`benchmark/corpus/wiki/bob_jones.txt`)

> He reports to Eve Martinez, the VP of Engineering.

**bob** (`benchmark/corpus/wiki/bob_jones.txt`)

> He manages a team of twelve engineers, including Alice Smith and Carol Chen.

**project_phoenix** (`benchmark/corpus/wiki/project_phoenix.txt`)

> The project is led by Alice Smith, with Carol Chen as a core team member.

### 9) 逐项事实核对（审计观察，不是人工判定）

| # | 答案中的关键事实 | 页面证据 | 审计观察 |
|---|---|---|---|
| 1 | Eve Martinez 的办公室在 Building B 五楼 | `building_b`：fifth floor … VP of Engineering’s office | 有页面支持 |
| 2 | Eve 管理 Bob，Bob 领导 platform engineering group | `eve`：Eve manages two directors: Bob Jones, who leads the platform engineering group | 有页面支持 |
| 3 | Bob 管理 Alice Smith | `bob`：He manages a team of twelve engineers, including Alice Smith；Agent 还读到 `alice` 的 reports directly to Bob | 有页面支持 |
| 4 | Alice 领导 Project Phoenix | `project_phoenix`：The project is led by Alice Smith | 有页面支持 |
| 5 | Eve 到 Alice 是 indirect（两级） | Eve→Bob→Alice；页面分别给出两条 direct/manages 关系 | 有页面支持（链式推导） |

**审计备注（不替代人工裁定）**：链 Eve→Bob→Alice 和 indirect 结论有页面支持；relation_chain 的 direct_pairs 需要字面 direct(ly) 靠近每对实体，答案用 “manages”/“both are direct” 时漏判。RAG read_evidence_recall=0.8（alice 页面未读），但 `bob` 页面说 manages team includes Alice，`project_phoenix` 说 Alice leads，事实有支持。

### 10) 待人工填写的两个字段（留空）

- human_verdict: `________________`
- human_evidence_verdict: `________________`

### 11) review_notes（留空）

- review_notes: `________________`

---

## 26. mh47_exec_level_to_phoenix_chain · R1 · wiki_agent

- review_id: `s13_risk_r1_wiki_agent_mh47_exec_level_to_phoenix_chain`
- case_id: `mh47_exec_level_to_phoenix_chain`
- round: `1`
- group: `wiki_agent`
- 风险类型: `relation_direction_or_order`
- 审计分类（非人工裁定）: 语义正确但被评分器漏判
- 原始结果: `benchmark/results/stage12_multiround_20261101_120442_r1.json`
- checkpoint: `benchmark/results/stage12_multiround_20261101_120442_r1.checkpoint.jsonl`

### 2) 原始问题

> Starting from the executive level of Building B, trace the relationships to the engineer who leads Project Phoenix: identify the VP whose office is on that floor, the director she manages who leads the platform engineering group, and the engineer who reports to that director and leads Project Phoenix. List the chain and state which relationship is indirect.

### 3) 参考答案

> Building B's fifth floor is the executive level, where Eve Martinez, the VP of Engineering, has her office. Eve directly manages Bob Jones, who leads the platform engineering group. Bob directly manages Alice Smith, and Alice leads Project Phoenix. Eve's relationship to Alice is indirect, because it passes through Bob Jones.

### 4) 模型完整原始答案

```text
The chain starts on the executive level (fifth floor) of Building B:

1. **Eve Martinez** — VP of Engineering. Her office is on the fifth floor (executive level) of Building B. She manages two directors: Bob Jones (platform engineering group) and David Kim (data engineering group).
2. **Bob Jones** — Director of Engineering, leads the platform engineering group and reports directly to Eve Martinez.
3. **Alice Smith** — Senior engineer who leads Project Phoenix. She reports directly to Bob Jones.

So the chain is: **Eve Martinez (VP) → Bob Jones (Director, platform engineering) → Alice Smith (engineer, Project Phoenix lead).**

The **indirect relationship is Eve Martinez → Alice Smith**: Eve does not manage Alice directly — Alice reports to Bob Jones, who reports to Eve. The two direct (supervisor→direct-report) links are Eve→Bob and Bob→Alice.
```

### 5) 自动评分及具体评分细节

- auto_correct: `False`
- verified_correct: `False`（原有 grounded 指标，不是人工判定）
- status: `answered`
- keyword_groups: `0:true, 1:true, 2:true, 3:true, 4:true, 5:true`
- structured_checks: `relation_chain_check=false`
- relation_chain_pairs: `direct(eve->bob)=false; direct(bob->alice)=false; indirect(eve->alice)=true; forbidden_direct(eve->alice)=false; leadership(alice->project phoenix)=true`
- evidence_recall: `1.0`；read_evidence_recall: `1.0`；source_precision: `1.0`
- usage: requests=4, total_tokens=4622, latency=7.06s
- 漏判机制: relation_chain direct_pairs=False: answer uses 'manages' and a summary 'both are direct'; no literal direct(ly) near each named pair

### 6) 模型提供的 source_ids

`building_b, eve, bob, alice, project_phoenix`

### 7) 实际 read_page_ids

`alice, bob, building_b, carol, eve, project_phoenix`

- source_ids 是否都在 read_page_ids 中：是
- accessed_page_ids 中未被读取的页面（搜索/访问候选，不计为读取证据）：`building_a, david, project_orion`

### 8) 对应 Wiki 页面中的原始证据片段

**building_b** (`benchmark/corpus/wiki/building_b.txt`)

> The fifth floor is the executive level, containing the VP of Engineering's office (currently Eve Martinez), the CEO's office, and a boardroom.

**eve** (`benchmark/corpus/wiki/eve_martinez.txt`)

> Eve manages two directors: Bob Jones, who leads the platform engineering group, and David Kim, who leads the data engineering group.

**bob** (`benchmark/corpus/wiki/bob_jones.txt`)

> He reports to Eve Martinez, the VP of Engineering.

**bob** (`benchmark/corpus/wiki/bob_jones.txt`)

> He manages a team of twelve engineers, including Alice Smith and Carol Chen.

**project_phoenix** (`benchmark/corpus/wiki/project_phoenix.txt`)

> The project is led by Alice Smith, with Carol Chen as a core team member.

**alice** (`benchmark/corpus/wiki/alice_smith.txt`)

> Alice reports directly to Bob Jones, the Director of Engineering.

### 9) 逐项事实核对（审计观察，不是人工判定）

| # | 答案中的关键事实 | 页面证据 | 审计观察 |
|---|---|---|---|
| 1 | Eve Martinez 的办公室在 Building B 五楼 | `building_b`：fifth floor … VP of Engineering’s office | 有页面支持 |
| 2 | Eve 管理 Bob，Bob 领导 platform engineering group | `eve`：Eve manages two directors: Bob Jones, who leads the platform engineering group | 有页面支持 |
| 3 | Bob 管理 Alice Smith | `bob`：He manages a team of twelve engineers, including Alice Smith；Agent 还读到 `alice` 的 reports directly to Bob | 有页面支持 |
| 4 | Alice 领导 Project Phoenix | `project_phoenix`：The project is led by Alice Smith | 有页面支持 |
| 5 | Eve 到 Alice 是 indirect（两级） | Eve→Bob→Alice；页面分别给出两条 direct/manages 关系 | 有页面支持（链式推导） |

**审计备注（不替代人工裁定）**：链 Eve→Bob→Alice 和 indirect 结论有页面支持；relation_chain 的 direct_pairs 需要字面 direct(ly) 靠近每对实体，答案用 “manages”/“both are direct” 时漏判。RAG read_evidence_recall=0.8（alice 页面未读），但 `bob` 页面说 manages team includes Alice，`project_phoenix` 说 Alice leads，事实有支持。

### 10) 待人工填写的两个字段（留空）

- human_verdict: `________________`
- human_evidence_verdict: `________________`

### 11) review_notes（留空）

- review_notes: `________________`

---

## 27. mh47_exec_level_to_phoenix_chain · R2 · traditional_rag

- review_id: `s13_risk_r2_traditional_rag_mh47_exec_level_to_phoenix_chain`
- case_id: `mh47_exec_level_to_phoenix_chain`
- round: `2`
- group: `traditional_rag`
- 风险类型: `relation_direction_or_order`
- 审计分类（非人工裁定）: 语义正确但被评分器漏判
- 原始结果: `benchmark/results/stage12_multiround_20261101_120442_r2.json`
- checkpoint: `benchmark/results/stage12_multiround_20261101_120442_r2.checkpoint.jsonl`

### 2) 原始问题

> Starting from the executive level of Building B, trace the relationships to the engineer who leads Project Phoenix: identify the VP whose office is on that floor, the director she manages who leads the platform engineering group, and the engineer who reports to that director and leads Project Phoenix. List the chain and state which relationship is indirect.

### 3) 参考答案

> Building B's fifth floor is the executive level, where Eve Martinez, the VP of Engineering, has her office. Eve directly manages Bob Jones, who leads the platform engineering group. Bob directly manages Alice Smith, and Alice leads Project Phoenix. Eve's relationship to Alice is indirect, because it passes through Bob Jones.

### 4) 模型完整原始答案

```text
The chain is: Eve Martinez (VP of Engineering, office on the fifth/executive level of Building B) → Bob Jones (Director of Engineering, whom Eve manages and who leads the platform engineering group) → Alice Smith (an engineer on Bob's team and the lead of Project Phoenix). The indirect relationship is between Eve Martinez and Alice Smith: Eve manages Bob directly, and Bob manages Alice directly, so Eve's relationship to Alice is indirect through Bob.
```

### 5) 自动评分及具体评分细节

- auto_correct: `False`
- verified_correct: `False`（原有 grounded 指标，不是人工判定）
- status: `answered`
- keyword_groups: `0:true, 1:true, 2:true, 3:true, 4:true, 5:true`
- structured_checks: `relation_chain_check=false`
- relation_chain_pairs: `direct(eve->bob)=true; direct(bob->alice)=true; indirect(eve->alice)=true; forbidden_direct(eve->alice)=true; leadership(alice->project phoenix)=true`
- evidence_recall: `0.8`；read_evidence_recall: `0.8`；source_precision: `1.0`
- usage: requests=1, total_tokens=1392, latency=3.06s
- 漏判机制: relation_chain_check=False while all keyword groups match: direct/indirect pairs require literal direct(ly) near named pairs

- 备注: read_evidence_recall=0.8；已读页面仍覆盖答案事实，详见逐项核对。

### 6) 模型提供的 source_ids

`building_b, eve, bob, project_phoenix`

### 7) 实际 read_page_ids

`building_b, eve, bob, project_phoenix, building_a`

- source_ids 是否都在 read_page_ids 中：是

### 8) 对应 Wiki 页面中的原始证据片段

**building_b** (`benchmark/corpus/wiki/building_b.txt`)

> The fifth floor is the executive level, containing the VP of Engineering's office (currently Eve Martinez), the CEO's office, and a boardroom.

**eve** (`benchmark/corpus/wiki/eve_martinez.txt`)

> Eve manages two directors: Bob Jones, who leads the platform engineering group, and David Kim, who leads the data engineering group.

**bob** (`benchmark/corpus/wiki/bob_jones.txt`)

> He reports to Eve Martinez, the VP of Engineering.

**bob** (`benchmark/corpus/wiki/bob_jones.txt`)

> He manages a team of twelve engineers, including Alice Smith and Carol Chen.

**project_phoenix** (`benchmark/corpus/wiki/project_phoenix.txt`)

> The project is led by Alice Smith, with Carol Chen as a core team member.

### 9) 逐项事实核对（审计观察，不是人工判定）

| # | 答案中的关键事实 | 页面证据 | 审计观察 |
|---|---|---|---|
| 1 | Eve Martinez 的办公室在 Building B 五楼 | `building_b`：fifth floor … VP of Engineering’s office | 有页面支持 |
| 2 | Eve 管理 Bob，Bob 领导 platform engineering group | `eve`：Eve manages two directors: Bob Jones, who leads the platform engineering group | 有页面支持 |
| 3 | Bob 管理 Alice Smith | `bob`：He manages a team of twelve engineers, including Alice Smith；Agent 还读到 `alice` 的 reports directly to Bob | 有页面支持 |
| 4 | Alice 领导 Project Phoenix | `project_phoenix`：The project is led by Alice Smith | 有页面支持 |
| 5 | Eve 到 Alice 是 indirect（两级） | Eve→Bob→Alice；页面分别给出两条 direct/manages 关系 | 有页面支持（链式推导） |

**审计备注（不替代人工裁定）**：链 Eve→Bob→Alice 和 indirect 结论有页面支持；relation_chain 的 direct_pairs 需要字面 direct(ly) 靠近每对实体，答案用 “manages”/“both are direct” 时漏判。RAG read_evidence_recall=0.8（alice 页面未读），但 `bob` 页面说 manages team includes Alice，`project_phoenix` 说 Alice leads，事实有支持。

### 10) 待人工填写的两个字段（留空）

- human_verdict: `________________`
- human_evidence_verdict: `________________`

### 11) review_notes（留空）

- review_notes: `________________`

---

## 28. mh47_exec_level_to_phoenix_chain · R2 · wiki_agent

- review_id: `s13_risk_r2_wiki_agent_mh47_exec_level_to_phoenix_chain`
- case_id: `mh47_exec_level_to_phoenix_chain`
- round: `2`
- group: `wiki_agent`
- 风险类型: `relation_direction_or_order`
- 审计分类（非人工裁定）: 语义正确但被评分器漏判
- 原始结果: `benchmark/results/stage12_multiround_20261101_120442_r2.json`
- checkpoint: `benchmark/results/stage12_multiround_20261101_120442_r2.checkpoint.jsonl`

### 2) 原始问题

> Starting from the executive level of Building B, trace the relationships to the engineer who leads Project Phoenix: identify the VP whose office is on that floor, the director she manages who leads the platform engineering group, and the engineer who reports to that director and leads Project Phoenix. List the chain and state which relationship is indirect.

### 3) 参考答案

> Building B's fifth floor is the executive level, where Eve Martinez, the VP of Engineering, has her office. Eve directly manages Bob Jones, who leads the platform engineering group. Bob directly manages Alice Smith, and Alice leads Project Phoenix. Eve's relationship to Alice is indirect, because it passes through Bob Jones.

### 4) 模型完整原始答案

```text
Chain from Building B's executive level (fifth floor):

1. Eve Martinez — VP of Engineering, whose office is on the fifth floor (executive level) of Building B.
2. Bob Jones — Director of Engineering, whom Eve manages and who leads the platform engineering group.
3. Alice Smith — senior engineer who reports directly to Bob Jones and leads Project Phoenix.

So: Eve Martinez → Bob Jones → Alice Smith.

Indirect relationship: The link between Eve Martinez and Alice Smith is indirect — Alice does not report to Eve directly; she reports to Bob Jones, who in turn reports to Eve. Both the Eve→Bob and Bob→Alice relationships are direct reporting relationships.
```

### 5) 自动评分及具体评分细节

- auto_correct: `False`
- verified_correct: `False`（原有 grounded 指标，不是人工判定）
- status: `answered`
- keyword_groups: `0:true, 1:true, 2:true, 3:true, 4:true, 5:true`
- structured_checks: `relation_chain_check=false`
- relation_chain_pairs: `direct(eve->bob)=false; direct(bob->alice)=false; indirect(eve->alice)=true; forbidden_direct(eve->alice)=false; leadership(alice->project phoenix)=true`
- evidence_recall: `1.0`；read_evidence_recall: `1.0`；source_precision: `1.0`
- usage: requests=4, total_tokens=4596, latency=7.23s
- 漏判机制: relation_chain direct_pairs=False: answer uses 'manages' and a summary 'both are direct'; no literal direct(ly) near each named pair

### 6) 模型提供的 source_ids

`building_b, eve, bob, alice, project_phoenix`

### 7) 实际 read_page_ids

`alice, bob, building_a, building_b, eve, project_phoenix`

- source_ids 是否都在 read_page_ids 中：是
- accessed_page_ids 中未被读取的页面（搜索/访问候选，不计为读取证据）：`carol, david, project_orion`

### 8) 对应 Wiki 页面中的原始证据片段

**building_b** (`benchmark/corpus/wiki/building_b.txt`)

> The fifth floor is the executive level, containing the VP of Engineering's office (currently Eve Martinez), the CEO's office, and a boardroom.

**eve** (`benchmark/corpus/wiki/eve_martinez.txt`)

> Eve manages two directors: Bob Jones, who leads the platform engineering group, and David Kim, who leads the data engineering group.

**bob** (`benchmark/corpus/wiki/bob_jones.txt`)

> He reports to Eve Martinez, the VP of Engineering.

**bob** (`benchmark/corpus/wiki/bob_jones.txt`)

> He manages a team of twelve engineers, including Alice Smith and Carol Chen.

**project_phoenix** (`benchmark/corpus/wiki/project_phoenix.txt`)

> The project is led by Alice Smith, with Carol Chen as a core team member.

**alice** (`benchmark/corpus/wiki/alice_smith.txt`)

> Alice reports directly to Bob Jones, the Director of Engineering.

### 9) 逐项事实核对（审计观察，不是人工判定）

| # | 答案中的关键事实 | 页面证据 | 审计观察 |
|---|---|---|---|
| 1 | Eve Martinez 的办公室在 Building B 五楼 | `building_b`：fifth floor … VP of Engineering’s office | 有页面支持 |
| 2 | Eve 管理 Bob，Bob 领导 platform engineering group | `eve`：Eve manages two directors: Bob Jones, who leads the platform engineering group | 有页面支持 |
| 3 | Bob 管理 Alice Smith | `bob`：He manages a team of twelve engineers, including Alice Smith；Agent 还读到 `alice` 的 reports directly to Bob | 有页面支持 |
| 4 | Alice 领导 Project Phoenix | `project_phoenix`：The project is led by Alice Smith | 有页面支持 |
| 5 | Eve 到 Alice 是 indirect（两级） | Eve→Bob→Alice；页面分别给出两条 direct/manages 关系 | 有页面支持（链式推导） |

**审计备注（不替代人工裁定）**：链 Eve→Bob→Alice 和 indirect 结论有页面支持；relation_chain 的 direct_pairs 需要字面 direct(ly) 靠近每对实体，答案用 “manages”/“both are direct” 时漏判。RAG read_evidence_recall=0.8（alice 页面未读），但 `bob` 页面说 manages team includes Alice，`project_phoenix` 说 Alice leads，事实有支持。

### 10) 待人工填写的两个字段（留空）

- human_verdict: `________________`
- human_evidence_verdict: `________________`

### 11) review_notes（留空）

- review_notes: `________________`

---

## 29. mh47_exec_level_to_phoenix_chain · R3 · traditional_rag

- review_id: `s13_risk_r3_traditional_rag_mh47_exec_level_to_phoenix_chain`
- case_id: `mh47_exec_level_to_phoenix_chain`
- round: `3`
- group: `traditional_rag`
- 风险类型: `relation_direction_or_order`
- 审计分类（非人工裁定）: 语义正确但被评分器漏判
- 原始结果: `benchmark/results/stage12_multiround_20261101_120442_r3.json`
- checkpoint: `benchmark/results/stage12_multiround_20261101_120442_r3.checkpoint.jsonl`

### 2) 原始问题

> Starting from the executive level of Building B, trace the relationships to the engineer who leads Project Phoenix: identify the VP whose office is on that floor, the director she manages who leads the platform engineering group, and the engineer who reports to that director and leads Project Phoenix. List the chain and state which relationship is indirect.

### 3) 参考答案

> Building B's fifth floor is the executive level, where Eve Martinez, the VP of Engineering, has her office. Eve directly manages Bob Jones, who leads the platform engineering group. Bob directly manages Alice Smith, and Alice leads Project Phoenix. Eve's relationship to Alice is indirect, because it passes through Bob Jones.

### 4) 模型完整原始答案

```text
Chain: Eve Martinez (VP of Engineering, office on the fifth/executive floor of Building B) → Bob Jones (Director of Engineering, leads the platform engineering group and reports to Eve) → Alice Smith (engineer who reports to Bob and leads Project Phoenix). The indirect relationship is Eve Martinez to Alice Smith: Eve manages Bob directly, and Bob manages Alice directly, so Eve does not directly manage Alice.
```

### 5) 自动评分及具体评分细节

- auto_correct: `False`
- verified_correct: `False`（原有 grounded 指标，不是人工判定）
- status: `answered`
- keyword_groups: `0:true, 1:true, 2:true, 3:true, 4:true, 5:true`
- structured_checks: `relation_chain_check=false`
- relation_chain_pairs: `direct(eve->bob)=true; direct(bob->alice)=true; indirect(eve->alice)=true; forbidden_direct(eve->alice)=true; leadership(alice->project phoenix)=true`
- evidence_recall: `0.8`；read_evidence_recall: `0.8`；source_precision: `1.0`
- usage: requests=1, total_tokens=1443, latency=3.20s
- 漏判机制: relation_chain_check=False while all keyword groups match: direct/indirect pairs require literal direct(ly) near named pairs

- 备注: read_evidence_recall=0.8；已读页面仍覆盖答案事实，详见逐项核对。

### 6) 模型提供的 source_ids

`building_b, eve, bob, project_phoenix`

### 7) 实际 read_page_ids

`building_b, eve, bob, project_phoenix, building_a`

- source_ids 是否都在 read_page_ids 中：是

### 8) 对应 Wiki 页面中的原始证据片段

**building_b** (`benchmark/corpus/wiki/building_b.txt`)

> The fifth floor is the executive level, containing the VP of Engineering's office (currently Eve Martinez), the CEO's office, and a boardroom.

**eve** (`benchmark/corpus/wiki/eve_martinez.txt`)

> Eve manages two directors: Bob Jones, who leads the platform engineering group, and David Kim, who leads the data engineering group.

**bob** (`benchmark/corpus/wiki/bob_jones.txt`)

> He reports to Eve Martinez, the VP of Engineering.

**bob** (`benchmark/corpus/wiki/bob_jones.txt`)

> He manages a team of twelve engineers, including Alice Smith and Carol Chen.

**project_phoenix** (`benchmark/corpus/wiki/project_phoenix.txt`)

> The project is led by Alice Smith, with Carol Chen as a core team member.

### 9) 逐项事实核对（审计观察，不是人工判定）

| # | 答案中的关键事实 | 页面证据 | 审计观察 |
|---|---|---|---|
| 1 | Eve Martinez 的办公室在 Building B 五楼 | `building_b`：fifth floor … VP of Engineering’s office | 有页面支持 |
| 2 | Eve 管理 Bob，Bob 领导 platform engineering group | `eve`：Eve manages two directors: Bob Jones, who leads the platform engineering group | 有页面支持 |
| 3 | Bob 管理 Alice Smith | `bob`：He manages a team of twelve engineers, including Alice Smith；Agent 还读到 `alice` 的 reports directly to Bob | 有页面支持 |
| 4 | Alice 领导 Project Phoenix | `project_phoenix`：The project is led by Alice Smith | 有页面支持 |
| 5 | Eve 到 Alice 是 indirect（两级） | Eve→Bob→Alice；页面分别给出两条 direct/manages 关系 | 有页面支持（链式推导） |

**审计备注（不替代人工裁定）**：链 Eve→Bob→Alice 和 indirect 结论有页面支持；relation_chain 的 direct_pairs 需要字面 direct(ly) 靠近每对实体，答案用 “manages”/“both are direct” 时漏判。RAG read_evidence_recall=0.8（alice 页面未读），但 `bob` 页面说 manages team includes Alice，`project_phoenix` 说 Alice leads，事实有支持。

### 10) 待人工填写的两个字段（留空）

- human_verdict: `________________`
- human_evidence_verdict: `________________`

### 11) review_notes（留空）

- review_notes: `________________`

---
## 停止状态

本文件只整理第二批复核材料，未修改第一批报告/裁定、Stage 13.0 JSON/CSV、评分器、数据集、Agent、Prompt 或知识库，未调用模型 API，未运行 Live，未提交 Git。

请人工填写每张卡片的 `human_verdict`、`human_evidence_verdict` 和 `review_notes` 后，再决定是否进入评分器修复阶段。
