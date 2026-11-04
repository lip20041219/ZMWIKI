# Stage 13.1 第一批人工复核卡片

- 生成日期：2026-11-01
- 来源：`benchmark/reviews/stage13_0_human_review_20261101.json`
- 范围：9 条记录（3 条 Agent 疑似假阳性 + 6 条自动正确抽查）
- Wiki 页面：`benchmark/corpus/wiki/*.txt`（实际读取，不只看 reference_answer）
- 本报告不填写 `human_verdict`、`human_evidence_verdict`、`review_notes`；这些字段留空待人工裁定。

## 核对约定

- `auto_correct` 是 Stage 12 的原自动评分；`verified_correct` 是原有 `grounded_correct`（answer_correct + status=answered + 至少一个 source），不是人工判定。
- `source_ids` 是模型自己提供的 citation；`read_page_ids` 是实际读取页面；`accessed_page_ids` 是搜索/访问候选，**不能当读取证据**。
- `evidence_recall` 可能使用 accessed_pages 或 alternative evidence path；`read_evidence_recall` 只使用 read_page_ids。
- `source_precision` 只按 required/alternative evidence 计算；额外的、实际支持答案的页面可能拉低 precision，这是口径问题，不等于证据缺失。
- “与页面文本不一致”“页面未提及”“无法从页面确认”是审计观察，不是人工结论；最终由人工填写。
- 证据片段均为 `benchmark/corpus/wiki/*.txt` 的原文子串；`model_answer` 为 Stage 12 原始完整字符串。

## 总览

| # | case_id | round | group | auto | verified | status | source/read | 类型 |
|---:|---|---:|---|---:|---:|---|---|---|
| 1 | `mh38_responsibility_synthesis` | 1 | wiki_agent | True | False | `insufficient_evidence` | 0/0 | 风险-疑似假阳性 |
| 2 | `mh10_direct_reports` | 2 | wiki_agent | True | False | `insufficient_evidence` | 0/0 | 风险-疑似假阳性 |
| 3 | `mh49_orion_upward_chain_ceo` | 2 | wiki_agent | True | False | `insufficient_evidence` | 0/0 | 风险-疑似假阳性 |
| 4 | `mh21_team_intersection` | 1 | traditional_rag | True | True | `answered` | 3/5 | 自动正确抽查 |
| 5 | `mh24_engineering_budget_vs_projects` | 3 | traditional_rag | True | True | `answered` | 3/5 | 自动正确抽查 |
| 6 | `mh41_orion_tech_match` | 2 | traditional_rag | True | True | `answered` | 3/5 | 自动正确抽查 |
| 7 | `mh54_alice_carol_role_office_disambiguation` | 2 | traditional_rag | True | True | `answered` | 4/5 | 自动正确抽查 |
| 8 | `mh02_project_budget` | 2 | wiki_agent | True | True | `answered` | 2/3 | 自动正确抽查 |
| 9 | `mh17_carol_two_project_leads` | 2 | wiki_agent | True | True | `answered` | 5/5 | 自动正确抽查 |

## 1. Agent R1 · mh38_responsibility_synthesis（疑似假阳性）

- review_id: `s13_risk_r1_wiki_agent_mh38_responsibility_synthesis`
- case_id: `mh38_responsibility_synthesis`
- round: `1`
- group: `wiki_agent`
- 原始结果: `benchmark/results/stage12_multiround_20261101_120442_r1.json`
- checkpoint: `benchmark/results/stage12_multiround_20261101_120442_r1.checkpoint.jsonl`

### 2) 原始问题

> Which manager is responsible for technical roadmap planning and quarterly performance reviews, and which manager owns the annual engineering budget?

### 3) 参考答案

> Bob Jones is responsible for technical roadmap planning and quarterly performance reviews. Eve Martinez is responsible for the annual engineering budget.

### 4) 模型完整原始答案

```text
Technical roadmap planning is owned by Eve Martinez, the VP of Engineering (she also chairs the architecture review board and sets quarterly engineering objectives that feed the review process). Quarterly performance reviews are run by Bob Jones, the Director of Engineering. The annual engineering budget is also owned by Bob Jones, the Director of Engineering, who works with finance on budget forecasting, headcount planning, and vendor spend. David Kim, a principal engineer, holds no budget or roadmap responsibilities.
```

### 5) 自动评分结果及评分原因

- auto_correct: `True`
- verified_correct: `False`（原有 grounded 指标，不是人工判定）
- status: `insufficient_evidence`；failure_type: `insufficient_evidence`
- keyword_groups: `0:true, 1:true, 2:true, 3:true`
- structured_checks: `n/a`
- evidence_recall: `1.0`；read_evidence_recall: `0.0`；source_precision: `0.0`
- usage: requests=3, total_tokens=2929, latency=6.65s
- 评分原因: 4/4 keyword groups matched（bob、roadmap、eve、budget）。该题没有 answer_bindings / relation_chain / comparison 等结构检查，因此“谁负责 roadmap、谁负责 budget”的归属没有被校验。auto_correct=True。verified_correct=False，因为 status=insufficient_evidence 且 source_ids=[]。

### 6) 模型提供的 source_ids

`无`

### 7) 实际 read_page_ids

`无（未读取任何页面）`

- source_ids 为空；没有可核对的已读 source。
- accessed_page_ids（搜索/访问候选，**不计为读取**）：`alice, bob, building_a, building_b, carol, david, eve, project_orion, project_phoenix`

### 8) 对应 Wiki 页面中的原始证据片段

**bob** (`benchmark/corpus/wiki/bob_jones.txt`)

> He is responsible for technical roadmap planning and quarterly performance reviews for his direct reports.

**eve** (`benchmark/corpus/wiki/eve_martinez.txt`)

> Eve is responsible for the company's annual engineering budget of four million dollars and for setting the long-term technical vision.

**project_phoenix** (`benchmark/corpus/wiki/project_phoenix.txt`)

> The project team meets every Tuesday and Thursday for standups and architecture reviews.

**证据不足/未出现的项**

- 9 个 Wiki 页面中未找到 `architecture review board`、`quarterly engineering objectives`、`budget forecasting`、`headcount planning`、`vendor spend`。
- 没有任何页面写 Eve 负责 technical roadmap planning；bob 页面明确写 Bob 负责 roadmap。
- 没有任何页面写 Bob 负责 annual engineering budget；eve 页面明确写 Eve 负责 budget。

### 9) 逐项核对（审计观察，不是人工判定）

| # | 答案中的关键事实 | 页面证据 | 审计观察 |
|---|---|---|---|
| 1 | Bob Jones 是 Director of Engineering | `bob`：“Bob Jones serves as the Director of Engineering…” | 有页面支持 |
| 2 | Eve Martinez 是 VP of Engineering | `eve`：“Eve Martinez is the VP of Engineering at Acme Corp…” | 有页面支持 |
| 3 | Eve 负责 technical roadmap planning | `bob` 写 Bob 负责 roadmap；`eve` 只写 budget 和 long-term technical vision | 与页面文本不一致（待人工裁定） |
| 4 | Bob 负责 quarterly performance reviews | `bob`：“…quarterly performance reviews for his direct reports.” | 有页面支持 |
| 5 | Bob 负责 annual engineering budget | `eve` 写 Eve 负责 annual engineering budget；`bob` 未写 budget | 与页面文本不一致（待人工裁定） |
| 6 | Eve 主持 architecture review board / 设定 quarterly engineering objectives | `project_phoenix` 只提到项目团队的 standups 和 architecture reviews；没有 Eve 主持 board 或设定 objectives 的句子 | 页面未提及（缺少对应事实/页面） |
| 7 | Bob 与 finance 做 budget forecasting / headcount planning / vendor spend | 9 个页面均未出现这些短语 | 页面未提及（缺少对应事实/页面） |
| 8 | David Kim 不承担 budget 或 roadmap 职责 | `david` 页面没有提到 budget/roadmap；仅凭未提及不能证明“无职责” | 无法从页面确认 |

**审计备注（不替代人工裁定）**：参考答案把 roadmap/reviews 归 Bob、budget 归 Eve；模型答案把 roadmap 归 Eve、budget 归 Bob。`bob` 和 `eve` 页面的原文与答案在这两个归属上方向相反。此外，答案中的 architecture review board、quarterly engineering objectives、budget forecasting、headcount planning、vendor spend 均无页面依据。

### 10) 待人工填写的两个字段（留空）

- human_verdict: `________________`
- human_evidence_verdict: `________________`

### 11) review_notes（留空）

- review_notes: `________________`

---

## 2. Agent R2 · mh10_direct_reports（疑似假阳性）

- review_id: `s13_risk_r2_wiki_agent_mh10_direct_reports`
- case_id: `mh10_direct_reports`
- round: `2`
- group: `wiki_agent`
- 原始结果: `benchmark/results/stage12_multiround_20261101_120442_r2.json`
- checkpoint: `benchmark/results/stage12_multiround_20261101_120442_r2.checkpoint.jsonl`

### 2) 原始问题

> List all people who report to Bob Jones.

### 3) 参考答案

> Alice Smith and Carol Chen report to Bob Jones.

### 4) 模型完整原始答案

```text
Bob Jones has five direct reports: Alice Smith, Carol Chen, David Kim, Frank Osei, and Grace Liu.
```

### 5) 自动评分结果及评分原因

- auto_correct: `True`
- verified_correct: `False`（原有 grounded 指标，不是人工判定）
- status: `insufficient_evidence`；failure_type: `insufficient_evidence`
- keyword_groups: `0:true, 1:true`
- structured_checks: `n/a`
- evidence_recall: `1.0`；read_evidence_recall: `0.0`；source_precision: `0.0`
- usage: requests=3, total_tokens=2233, latency=5.47s
- 评分原因: 2/2 keyword groups matched（alice、carol）。题目没有检查其他姓名或人数，因此额外加入的 David / Frank / Grace 不会被校验。auto_correct=True。verified_correct=False，因为 status=insufficient_evidence 且 source_ids=[]。

### 6) 模型提供的 source_ids

`无`

### 7) 实际 read_page_ids

`无（未读取任何页面）`

- source_ids 为空；没有可核对的已读 source。
- accessed_page_ids（搜索/访问候选，**不计为读取**）：`alice, bob, building_a, building_b, carol, david, eve, project_orion, project_phoenix`

### 8) 对应 Wiki 页面中的原始证据片段

**bob** (`benchmark/corpus/wiki/bob_jones.txt`)

> He manages a team of twelve engineers, including Alice Smith and Carol Chen.

**alice** (`benchmark/corpus/wiki/alice_smith.txt`)

> Alice reports directly to Bob Jones, the Director of Engineering.

**carol** (`benchmark/corpus/wiki/carol_chen.txt`)

> Carol reports to Bob Jones, the Director of Engineering.

**david** (`benchmark/corpus/wiki/david_kim.txt`)

> David reports to Eve Martinez, the VP of Engineering.

**证据不足/未出现的项**

- 9 个 Wiki 页面中未找到 `Frank`、`Grace`、`Osei`、`Liu`。
- 9 个页面中未找到 `five direct reports`；`bob` 只写 “a team of twelve engineers, including Alice Smith and Carol Chen”。
- 没有任何页面把 David Kim 写成 Bob 的直接下属；`david` 页面写 David reports to Eve。

### 9) 逐项核对（审计观察，不是人工判定）

| # | 答案中的关键事实 | 页面证据 | 审计观察 |
|---|---|---|---|
| 1 | Bob Jones 是 Director of Engineering | `bob`：“Bob Jones serves as the Director of Engineering…” | 有页面支持 |
| 2 | Alice Smith 向 Bob Jones 汇报 | `alice`：“Alice reports directly to Bob Jones…”；`bob` 的团队包含 Alice | 有页面支持 |
| 3 | Carol Chen 向 Bob Jones 汇报 | `carol`：“Carol reports to Bob Jones…”；`bob` 的团队包含 Carol | 有页面支持 |
| 4 | David Kim 向 Bob Jones 汇报 | `david`：“David reports to Eve Martinez, the VP of Engineering.” | 与页面文本不一致（待人工裁定） |
| 5 | Frank Osei 向 Bob Jones 汇报 | 9 个页面均未出现 Frank Osei | 页面未提及（缺少对应页面/事实） |
| 6 | Grace Liu 向 Bob Jones 汇报 | 9 个页面均未出现 Grace Liu | 页面未提及（缺少对应页面/事实） |
| 7 | Bob Jones 有五个直接下属 | `bob` 只写 team of twelve engineers, including Alice/Carol；没有“五个 direct reports”的句子 | 页面未提及（缺少对应事实；人数无法确认） |

**审计备注（不替代人工裁定）**：参考答案只列 Alice Smith 和 Carol Chen。模型答案额外加入 David Kim、Frank Osei、Grace Liu，并声称“five direct reports”。页面明确支持 Alice/Carol；David 页面写的是向 Eve 汇报；Frank/Grace 在语料中不存在；“five”也没有页面依据。

### 10) 待人工填写的两个字段（留空）

- human_verdict: `________________`
- human_evidence_verdict: `________________`

### 11) review_notes（留空）

- review_notes: `________________`

---

## 3. Agent R2 · mh49_orion_upward_chain_ceo（疑似假阳性）

- review_id: `s13_risk_r2_wiki_agent_mh49_orion_upward_chain_ceo`
- case_id: `mh49_orion_upward_chain_ceo`
- round: `2`
- group: `wiki_agent`
- 原始结果: `benchmark/results/stage12_multiround_20261101_120442_r2.json`
- checkpoint: `benchmark/results/stage12_multiround_20261101_120442_r2.checkpoint.jsonl`

### 2) 原始问题

> Starting from Project Orion, trace upward through its technical lead and that lead's manager. Who is the highest person in this chain that the corpus names, and to whom does that person report?

### 3) 参考答案

> Project Orion's technical lead is David Kim, who reports to Eve Martinez, the VP of Engineering. Eve Martinez is the highest named person in the chain; she reports directly to the CEO, whose name is not provided in the corpus.

### 4) 模型完整原始答案

```text
The chain runs: Project Orion → its technical lead David Kim → his manager Eve Martinez, VP of Engineering. Eve Martinez is the highest person named in this chain. She reports to the Chief Technology Officer (CTO), who in turn reports to the CEO.
```

### 5) 自动评分结果及评分原因

- auto_correct: `True`
- verified_correct: `False`（原有 grounded 指标，不是人工判定）
- status: `insufficient_evidence`；failure_type: `insufficient_evidence`
- keyword_groups: `0:true, 1:true`
- structured_checks: `n/a`
- evidence_recall: `1.0`；read_evidence_recall: `0.0`；source_precision: `0.0`
- usage: requests=3, total_tokens=2553, latency=5.32s
- 评分原因: 2/2 keyword groups matched（eve、ceo）。该题没有 relation_chain 结构检查，因此多出来的 CTO 层不会被校验。auto_correct=True。verified_correct=False，因为 status=insufficient_evidence 且 source_ids=[]。

### 6) 模型提供的 source_ids

`无`

### 7) 实际 read_page_ids

`无（未读取任何页面）`

- source_ids 为空；没有可核对的已读 source。
- accessed_page_ids（搜索/访问候选，**不计为读取**）：`alice, bob, carol, david, eve, project_orion, project_phoenix`

### 8) 对应 Wiki 页面中的原始证据片段

**project_orion** (`benchmark/corpus/wiki/project_orion.txt`)

> The project is led by David Kim, a principal engineer, with Carol Chen contributing to the backend integration layer.

**project_orion** (`benchmark/corpus/wiki/project_orion.txt`)

> David Kim reports to Eve Martinez.

**david** (`benchmark/corpus/wiki/david_kim.txt`)

> David reports to Eve Martinez, the VP of Engineering.

**eve** (`benchmark/corpus/wiki/eve_martinez.txt`)

> She has been in this role since 2018 and reports directly to the CEO.

**eve** (`benchmark/corpus/wiki/eve_martinez.txt`)

> Eve manages two directors: Bob Jones, who leads the platform engineering group, and David Kim, who leads the data engineering group.

**证据不足/未出现的项**

- 9 个 Wiki 页面中未找到 `CTO`、`Chief Technology Officer`、`chief technology officer`。
- `eve` 页面写 “reports directly to the CEO”，没有写 CTO 这一层。

### 9) 逐项核对（审计观察，不是人工判定）

| # | 答案中的关键事实 | 页面证据 | 审计观察 |
|---|---|---|---|
| 1 | Project Orion 的技术负责人是 David Kim | `project_orion`：“The project is led by David Kim…” | 有页面支持 |
| 2 | David Kim 向 Eve Martinez 汇报 | `project_orion`：“David Kim reports to Eve Martinez.”；`david` 同义 | 有页面支持 |
| 3 | Eve Martinez 是这条链中语料点名的最高人物 | `eve` 是 VP of Engineering，管理两名 director；`david` 是她管理的 director 之一 | 有页面支持 |
| 4 | Eve Martinez 向 CTO 汇报 | `eve` 写 “reports directly to the CEO”；语料没有 CTO | 与页面文本不一致（待人工裁定） |
| 5 | CTO 再向 CEO 汇报 | 9 个页面均未出现 CTO/Chief Technology Officer | 页面未提及（缺少对应页面/事实） |
| 6 | CEO 的姓名未提供 | `eve` 只写 reports to the CEO，没有给出姓名 | 有页面支持（姓名未给出） |

**审计备注（不替代人工裁定）**：David → Eve 两段有页面支持，Eve 是语料点名的最高人物也成立。但答案在 Eve 与 CEO 之间插入了 CTO；`eve` 页面明确写 Eve reports directly to the CEO，语料中没有任何 CTO/Chief Technology Officer 页面或句子。

### 10) 待人工填写的两个字段（留空）

- human_verdict: `________________`
- human_evidence_verdict: `________________`

### 11) review_notes（留空）

- review_notes: `________________`

---

## 4. RAG R1 · mh21_team_intersection（自动正确抽查）

- review_id: `s13_auto_correct_sample_r1_traditional_rag_mh21_team_intersection`
- case_id: `mh21_team_intersection`
- round: `1`
- group: `traditional_rag`
- 原始结果: `benchmark/results/stage12_multiround_20261101_120442_r1.json`
- checkpoint: `benchmark/results/stage12_multiround_20261101_120442_r1.checkpoint.jsonl`

### 2) 原始问题

> Who is the Director of Engineering to whom both the leader of Project Phoenix and a contributor to Project Orion report?

### 3) 参考答案

> Both Alice Smith, the leader of Project Phoenix, and Carol Chen, a contributor to Project Orion, report to Bob Jones, the Director of Engineering.

### 4) 模型完整原始答案

```text
Bob Jones, the Director of Engineering at Acme Corp, manages both Alice Smith (leader of Project Phoenix) and Carol Chen (a contributor to Project Orion).
```

### 5) 自动评分结果及评分原因

- auto_correct: `True`
- verified_correct: `True`（原有 grounded 指标，不是人工判定）
- status: `answered`；failure_type: `None`
- keyword_groups: `0:true, 1:true`
- structured_checks: `n/a`
- evidence_recall: `1.0`；read_evidence_recall: `1.0`；source_precision: `1.0`
- usage: requests=1, total_tokens=1122, latency=1.66s
- 评分原因: 2/2 keyword groups matched（bob jones/bob、director of engineering/director）。该题没有结构化检查；status=answered 且 source_ids 非空，因此 auto_correct=True、verified_correct=True。

### 6) 模型提供的 source_ids

`bob, project_phoenix, project_orion`

### 7) 实际 read_page_ids

`bob, project_orion, project_phoenix, eve, david`

- source_ids 是否都在 read_page_ids 中：是

### 8) 对应 Wiki 页面中的原始证据片段

**bob** (`benchmark/corpus/wiki/bob_jones.txt`)

> He manages a team of twelve engineers, including Alice Smith and Carol Chen.

**bob** (`benchmark/corpus/wiki/bob_jones.txt`)

> Bob Jones serves as the Director of Engineering at Acme Corp, a role he has held since 2019.

**project_phoenix** (`benchmark/corpus/wiki/project_phoenix.txt`)

> The project is led by Alice Smith, with Carol Chen as a core team member.

**project_orion** (`benchmark/corpus/wiki/project_orion.txt`)

> The project is led by David Kim, a principal engineer, with Carol Chen contributing to the backend integration layer.

### 9) 逐项核对（审计观察，不是人工判定）

| # | 答案中的关键事实 | 页面证据 | 审计观察 |
|---|---|---|---|
| 1 | Bob Jones 是 Director of Engineering | `bob`：“Bob Jones serves as the Director of Engineering…” | 有页面支持 |
| 2 | Bob 管理 Alice Smith | `bob`：“He manages a team of twelve engineers, including Alice Smith…” | 有页面支持 |
| 3 | Alice Smith 是 Project Phoenix 的 leader | `project_phoenix`：“The project is led by Alice Smith…” | 有页面支持 |
| 4 | Bob 管理 Carol Chen | `bob`：“…including … Carol Chen.” | 有页面支持 |
| 5 | Carol Chen 是 Project Orion 的 contributor | `project_orion`：“…with Carol Chen contributing to the backend integration layer.” | 有页面支持 |

**审计备注（不替代人工裁定）**：答案中的每个关键事实都能在实际读取的 bob/project_phoenix/project_orion 页面中找到对应句。虽然 read_page_ids 没有 alice/carol，但 alternative evidence path 的 bob + 两个项目页是合法且已读的。source_precision=1.0。

### 10) 待人工填写的两个字段（留空）

- human_verdict: `________________`
- human_evidence_verdict: `________________`

### 11) review_notes（留空）

- review_notes: `________________`

---

## 5. RAG R3 · mh24_engineering_budget_vs_projects（自动正确抽查）

- review_id: `s13_auto_correct_sample_r3_traditional_rag_mh24_engineering_budget_vs_projects`
- case_id: `mh24_engineering_budget_vs_projects`
- round: `3`
- group: `traditional_rag`
- 原始结果: `benchmark/results/stage12_multiround_20261101_120442_r3.json`
- checkpoint: `benchmark/results/stage12_multiround_20261101_120442_r3.checkpoint.jsonl`

### 2) 原始问题

> Is Acme's annual engineering budget greater than the combined budgets of Project Phoenix and Project Orion?

### 3) 参考答案

> Yes. Acme's annual engineering budget is four million dollars ($4,000,000), which is greater than the combined budgets of Project Phoenix ($500,000) and Project Orion ($300,000), totaling eight hundred thousand dollars ($800,000).

### 4) 模型完整原始答案

```text
Yes. Acme's annual engineering budget is four million dollars, while Project Phoenix's budget is five hundred thousand dollars and Project Orion's budget is three hundred thousand dollars. Their combined budget is eight hundred thousand dollars, which is less than four million dollars.
```

### 5) 自动评分结果及评分原因

- auto_correct: `True`
- verified_correct: `True`（原有 grounded 指标，不是人工判定）
- status: `answered`；failure_type: `None`
- keyword_groups: `0:true, 1:true`
- structured_checks: `comparison_check=true`
- evidence_recall: `1.0`；read_evidence_recall: `1.0`；source_precision: `1.0`
- usage: requests=1, total_tokens=1114, latency=1.41s
- 评分原因: 2/2 keyword groups matched（four million/4000000、eight hundred thousand/800000），comparison_check=True；status=answered 且 source_ids 非空，auto_correct=True、verified_correct=True。

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

### 9) 逐项核对（审计观察，不是人工判定）

| # | 答案中的关键事实 | 页面证据 | 审计观察 |
|---|---|---|---|
| 1 | Acme annual engineering budget = $4,000,000 | `eve`：annual engineering budget of four million dollars | 有页面支持 |
| 2 | Project Phoenix budget = $500,000 | `project_phoenix`：allocated budget of five hundred thousand dollars | 有页面支持 |
| 3 | Project Orion budget = $300,000 | `project_orion`：budget of three hundred thousand dollars | 有页面支持 |
| 4 | Combined budget = $800,000 | 500,000 + 300,000 = 800,000；两个项目页分别给出数字 | 有页面支持（算术核对） |
| 5 | $4,000,000 > $800,000 | 4,000,000 > 800,000；答案的 “less than” 方向正确 | 有页面支持（算术核对） |

**审计备注（不替代人工裁定）**：所有关键数字和比较方向都有页面依据，算术正确。source_precision=1.0。

### 10) 待人工填写的两个字段（留空）

- human_verdict: `________________`
- human_evidence_verdict: `________________`

### 11) review_notes（留空）

- review_notes: `________________`

---

## 6. RAG R2 · mh41_orion_tech_match（自动正确抽查）

- review_id: `s13_auto_correct_sample_r2_traditional_rag_mh41_orion_tech_match`
- case_id: `mh41_orion_tech_match`
- round: `2`
- group: `traditional_rag`
- 原始结果: `benchmark/results/stage12_multiround_20261101_120442_r2.json`
- checkpoint: `benchmark/results/stage12_multiround_20261101_120442_r2.checkpoint.jsonl`

### 2) 原始问题

> A new engineer wants to work on real-time business intelligence dashboards built with a columnar database, a stream processing engine, and a React frontend. Which project matches, who leads it, and on which floor of which building is that leader's office?

### 3) 参考答案

> Project Orion matches. It is led by David Kim, whose office is on the fourth floor of Building B.

### 4) 模型完整原始答案

```text
The project is Project Orion. It is led by David Kim, whose office is on the fourth floor of Building B.
```

### 5) 自动评分结果及评分原因

- auto_correct: `True`
- verified_correct: `True`（原有 grounded 指标，不是人工判定）
- status: `answered`；failure_type: `None`
- keyword_groups: `0:true, 1:true, 2:true, 3:true`
- structured_checks: `n/a`
- evidence_recall: `1.0`；read_evidence_recall: `1.0`；source_precision: `0.6666666666666666`
- usage: requests=1, total_tokens=1146, latency=1.69s
- 评分原因: 4/4 keyword groups matched（orion、david、fourth/4、building b）；status=answered 且 source_ids 非空，auto_correct=True、verified_correct=True。

### 6) 模型提供的 source_ids

`project_orion, david, building_b`

### 7) 实际 read_page_ids

`project_orion, david, building_a, carol, building_b`

- source_ids 是否都在 read_page_ids 中：是

### 8) 对应 Wiki 页面中的原始证据片段

**project_orion** (`benchmark/corpus/wiki/project_orion.txt`)

> The tech stack includes a columnar database, a stream processing engine, and a React-based frontend.

**project_orion** (`benchmark/corpus/wiki/project_orion.txt`)

> The project is led by David Kim, a principal engineer, with Carol Chen contributing to the backend integration layer.

**david** (`benchmark/corpus/wiki/david_kim.txt`)

> His office is on the fourth floor of Building B.

**building_b** (`benchmark/corpus/wiki/building_b.txt`)

> The fourth floor houses David Kim's office and several meeting rooms.

### 9) 逐项核对（审计观察，不是人工判定）

| # | 答案中的关键事实 | 页面证据 | 审计观察 |
|---|---|---|---|
| 1 | Orion 的技术栈包含 columnar database、stream processing engine、React frontend | `project_orion`：“The tech stack includes…” | 有页面支持 |
| 2 | Project Orion 由 David Kim 领导 | `project_orion`：“The project is led by David Kim…” | 有页面支持 |
| 3 | David Kim 的办公室在 Building B 四楼 | `david`：“His office is on the fourth floor of Building B.”；`building_b` 同义 | 有页面支持 |

**审计备注（不替代人工裁定）**：所有事实都有页面支持。source_precision=0.667 是因为 building_b 被计入 extra source；它实际上是 office floor 的必要证据。

### 10) 待人工填写的两个字段（留空）

- human_verdict: `________________`
- human_evidence_verdict: `________________`

### 11) review_notes（留空）

- review_notes: `________________`

---

## 7. RAG R2 · mh54_alice_carol_role_office_disambiguation（自动正确抽查）

- review_id: `s13_auto_correct_sample_r2_traditional_rag_mh54_alice_carol_role_office_disambiguation`
- case_id: `mh54_alice_carol_role_office_disambiguation`
- round: `2`
- group: `traditional_rag`
- 原始结果: `benchmark/results/stage12_multiround_20261101_120442_r2.json`
- checkpoint: `benchmark/results/stage12_multiround_20261101_120442_r2.checkpoint.jsonl`

### 2) 原始问题

> Alice Smith and Carol Chen both work in Building A and both have roles in Project Phoenix. One leads the project and has an office on the second floor; the other is a core team member with a desk on the first floor. Which engineer leads Project Phoenix, and which floor is each engineer on?

### 3) 参考答案

> Alice Smith leads Project Phoenix and has an office on the second floor of Building A. Carol Chen is a core team member of Project Phoenix and has a desk on the first floor of Building A.

### 4) 模型完整原始答案

```text
Alice Smith leads Project Phoenix; her office is on the second floor of Building A. Carol Chen is a core team member on Project Phoenix, and her desk is on the first floor of Building A.
```

### 5) 自动评分结果及评分原因

- auto_correct: `True`
- verified_correct: `True`（原有 grounded 指标，不是人工判定）
- status: `answered`；failure_type: `None`
- keyword_groups: `0:true, 1:true, 2:true, 3:true, 4:true, 5:true`
- structured_checks: `answer_bindings_check=true`
- evidence_recall: `1.0`；read_evidence_recall: `1.0`；source_precision: `1.0`
- usage: requests=1, total_tokens=1136, latency=1.44s
- 评分原因: 6/6 keyword groups matched；answer_bindings_check=True；status=answered 且 source_ids 非空，auto_correct=True、verified_correct=True。

### 6) 模型提供的 source_ids

`alice, carol, project_phoenix, building_a`

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

### 9) 逐项核对（审计观察，不是人工判定）

| # | 答案中的关键事实 | 页面证据 | 审计观察 |
|---|---|---|---|
| 1 | Alice Smith 领导 Project Phoenix | `project_phoenix`：“The project is led by Alice Smith…” | 有页面支持 |
| 2 | Carol Chen 是 Project Phoenix 的 core team member | `project_phoenix`：“…with Carol Chen as a core team member.” | 有页面支持 |
| 3 | Alice 的办公室在 Building A 二楼 | `alice`：“Her office is located on the second floor of Building A.”；`building_a` 同义 | 有页面支持 |
| 4 | Carol 的 desk 在 Building A 一楼 | `building_a`：“…including Carol Chen's desk.”；`carol` 写 office on the first floor | 有页面支持 |

**审计备注（不替代人工裁定）**：所有关键事实都有页面支持。`carol` 页面写 “office is on the first floor”，`building_a` 写 “Carol Chen's desk”；答案用 “desk”，由 building_a 支撑。answer_bindings=True。

### 10) 待人工填写的两个字段（留空）

- human_verdict: `________________`
- human_evidence_verdict: `________________`

### 11) review_notes（留空）

- review_notes: `________________`

---

## 8. Agent R2 · mh02_project_budget（自动正确抽查）

- review_id: `s13_auto_correct_sample_r2_wiki_agent_mh02_project_budget`
- case_id: `mh02_project_budget`
- round: `2`
- group: `wiki_agent`
- 原始结果: `benchmark/results/stage12_multiround_20261101_120442_r2.json`
- checkpoint: `benchmark/results/stage12_multiround_20261101_120442_r2.checkpoint.jsonl`

### 2) 原始问题

> What is the budget of the project led by Alice Smith?

### 3) 参考答案

> Project Phoenix has a budget of five hundred thousand dollars ($500,000).

### 4) 模型完整原始答案

```text
Alice Smith leads Project Phoenix, Acme Corp's internal platform migration initiative. It has an allocated budget of five hundred thousand dollars ($500,000).
```

### 5) 自动评分结果及评分原因

- auto_correct: `True`
- verified_correct: `True`（原有 grounded 指标，不是人工判定）
- status: `answered`；failure_type: `None`
- keyword_groups: `0:true, 1:true`
- structured_checks: `n/a`
- evidence_recall: `1.0`；read_evidence_recall: `1.0`；source_precision: `1.0`
- usage: requests=3, total_tokens=2086, latency=3.52s
- 评分原因: 2/2 keyword groups matched（500/five、thousand/hundred）；status=answered 且 source_ids 非空，auto_correct=True、verified_correct=True。

### 6) 模型提供的 source_ids

`alice, project_phoenix`

### 7) 实际 read_page_ids

`alice, project_orion, project_phoenix`

- source_ids 是否都在 read_page_ids 中：是

### 8) 对应 Wiki 页面中的原始证据片段

**alice** (`benchmark/corpus/wiki/alice_smith.txt`)

> She joined the company in 2021 and currently leads Project Phoenix, an internal platform migration initiative.

**project_phoenix** (`benchmark/corpus/wiki/project_phoenix.txt`)

> Project Phoenix is Acme Corp's internal platform migration initiative, launched in early 2025.

**project_phoenix** (`benchmark/corpus/wiki/project_phoenix.txt`)

> It has an allocated budget of five hundred thousand dollars and a target completion date of December 31, 2026.

### 9) 逐项核对（审计观察，不是人工判定）

| # | 答案中的关键事实 | 页面证据 | 审计观察 |
|---|---|---|---|
| 1 | Alice Smith 领导 Project Phoenix | `alice`：“…currently leads Project Phoenix…”；`project_phoenix` 同义 | 有页面支持 |
| 2 | Project Phoenix 是 internal platform migration initiative | `project_phoenix`：第一句原文 | 有页面支持 |
| 3 | Project Phoenix 的预算是 $500,000 | `project_phoenix`：“allocated budget of five hundred thousand dollars” | 有页面支持 |

**审计备注（不替代人工裁定）**：事实全部有页面支持。注意：同一道题的 RAG R2 答案使用裸 `$500,000`，因为 keyword group 需要 `thousand`/`hundred` 而被自动判 False；本条 Agent 答案写“five hundred thousand dollars ($500,000)”，因此通过。这是评分口径差异，不是事实差异。测试文件 `tests/wiki/test_multihop_evaluation.py` 明确把裸 `$500,000` 判 False。

### 10) 待人工填写的两个字段（留空）

- human_verdict: `________________`
- human_evidence_verdict: `________________`

### 11) review_notes（留空）

- review_notes: `________________`

---

## 9. Agent R2 · mh17_carol_two_project_leads（自动正确抽查）

- review_id: `s13_auto_correct_sample_r2_wiki_agent_mh17_carol_two_project_leads`
- case_id: `mh17_carol_two_project_leads`
- round: `2`
- group: `wiki_agent`
- 原始结果: `benchmark/results/stage12_multiround_20261101_120442_r2.json`
- checkpoint: `benchmark/results/stage12_multiround_20261101_120442_r2.checkpoint.jsonl`

### 2) 原始问题

> Carol Chen contributes to two projects. Who leads each of those projects, and on which floor of which building is each leader's office?

### 3) 参考答案

> Project Phoenix is led by Alice Smith, whose office is on the second floor of Building A. Project Orion is led by David Kim, whose office is on the fourth floor of Building B.

### 4) 模型完整原始答案

```text
Carol Chen contributes to Project Phoenix and Project Orion. Project Phoenix is led by Alice Smith, whose office is on the second floor of Building A. Project Orion is led by David Kim, whose office is on the fourth floor of Building B.
```

### 5) 自动评分结果及评分原因

- auto_correct: `True`
- verified_correct: `True`（原有 grounded 指标，不是人工判定）
- status: `answered`；failure_type: `None`
- keyword_groups: `0:true, 1:true, 2:true, 3:true`
- structured_checks: `n/a`
- evidence_recall: `1.0`；read_evidence_recall: `1.0`；source_precision: `0.6`
- usage: requests=5, total_tokens=4468, latency=6.73s
- 评分原因: 4/4 keyword groups matched；status=answered 且 source_ids 非空，auto_correct=True、verified_correct=True。

### 6) 模型提供的 source_ids

`carol, project_phoenix, project_orion, alice, david`

### 7) 实际 read_page_ids

`alice, carol, david, project_orion, project_phoenix`

- source_ids 是否都在 read_page_ids 中：是

### 8) 对应 Wiki 页面中的原始证据片段

**carol** (`benchmark/corpus/wiki/carol_chen.txt`)

> She joined the company in 2022 and currently contributes to two active projects: Project Phoenix and Project Orion.

**project_phoenix** (`benchmark/corpus/wiki/project_phoenix.txt`)

> The project is led by Alice Smith, with Carol Chen as a core team member.

**alice** (`benchmark/corpus/wiki/alice_smith.txt`)

> Her office is located on the second floor of Building A.

**project_orion** (`benchmark/corpus/wiki/project_orion.txt`)

> The project is led by David Kim, a principal engineer, with Carol Chen contributing to the backend integration layer.

**david** (`benchmark/corpus/wiki/david_kim.txt`)

> His office is on the fourth floor of Building B.

### 9) 逐项核对（审计观察，不是人工判定）

| # | 答案中的关键事实 | 页面证据 | 审计观察 |
|---|---|---|---|
| 1 | Carol 参与 Project Phoenix 和 Project Orion | `carol`：“…contributes to two active projects: Project Phoenix and Project Orion.” | 有页面支持 |
| 2 | Project Phoenix 由 Alice Smith 领导 | `project_phoenix`：“The project is led by Alice Smith…” | 有页面支持 |
| 3 | Alice 的办公室在 Building A 二楼 | `alice`：“Her office is located on the second floor of Building A.” | 有页面支持 |
| 4 | Project Orion 由 David Kim 领导 | `project_orion`：“The project is led by David Kim…” | 有页面支持 |
| 5 | David 的办公室在 Building B 四楼 | `david`：“His office is on the fourth floor of Building B.” | 有页面支持 |

**审计备注（不替代人工裁定）**：所有关键事实都有页面支持。source_precision=0.6 是因为两个项目页不在 required_page_ids；它们实际上是把 Carol 与项目 leader/office 连接起来所必需的页面。

### 10) 待人工填写的两个字段（留空）

- human_verdict: `________________`
- human_evidence_verdict: `________________`

### 11) review_notes（留空）

- review_notes: `________________`

---
## 停止状态

本文件只整理复核材料，未修改原始 JSON/CSV、评分器、数据集、Agent、Prompt 或知识库，未调用模型 API，未提交 Git。

请人工填写每张卡片的 `human_verdict`、`human_evidence_verdict` 和 `review_notes` 后，再决定是否进入评分器修复阶段。
