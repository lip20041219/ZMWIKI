# Stage 15.1 — 离线 EvidenceAudit v0 报告

- **日期**：2026-11-01
- **性质**：离线评测审计；不接入 WikiAgent 运行时，不改变回答行为，不调用模型 API。
- **范围**：claim 级支持/矛盾/无法判断判定；3 个历史错误 + 35 条 Stage 13 已知正确记录 + 10 条新增 cited-but-unsupported 案例。

---

## 1. 文件修改清单

| 文件 | 类型 | 说明 |
|---|---|---|
| `benchmark/evidence_audit.py` | 新增 | EvidenceAudit v0 独立模块 |
| `benchmark/run_evidence_audit_v0.py` | 新增 | 离线校准脚本（历史 3 条 + 35 条正确 + 10 条 fixture） |
| `benchmark/datasets/evidence_audit_v0_cases.json` | 新增 | 10 条 cited-but-unsupported 审计案例（独立 fixture，不改现有数据集） |
| `tests/wiki/test_evidence_audit.py` | 新增 | 14 个审计回归测试 |
| `docs/evaluation/STAGE15_1_EVIDENCE_AUDIT_REPORT.md` | 新增 | 本报告 |

未修改：`WikiAgent` 主流程、`score_answer` 正式评分语义、`multihop_wiki.json`、Stage 10–14 结果、人工裁定文件。

---

## 2. 审计算法与判定规则

### 2.1 输入边界

```text
answer          : 答案文本
sources         : 答案引用的页面 ID
read_page_ids   : 实际读取页面 ID
pages           : page_id -> 页面正文（WikiPage.content 或字符串）
```

- 只有 `sources ∩ read_page_ids` 的页面可作为支持/反证来源。
- `sources` 中未实际读取、或不在语料中的 ID 记为 `invalid_source_ids`，不参与证据判断。
- 搜索候选、模型自行编造的 Read results 不会进入 `pages`/`read_page_ids`，因此不会被当作证据。
- case annotations 不传入 `audit_answer`；fixture 的 expected 只用于测试校准，不作为证据。

### 2.2 事实抽取

对每个有效 source 页面：

1. 按句号/问号/感叹号/换行分句；
2. 用页面 id、id 词、首句中的专有名词序列构建实体别名索引；
3. 抽取简单事实：
   - `reports_to`：`X reports [directly] to Y`；
   - `manages` / `direct_report`：`X manages ...`、`team of N engineers ... including Y`；
   - `responsible_for`：`X is responsible for ...`、`X owns ...`、`... is owned/run by X`；
   - `leads`：`X leads Y`、`Y is led by X`；
   - `office`：`office is on the ... floor of Building ...`；
   - `team_size`、`budget`、`joined_year`、`role_since`。
4. 记录极性（否定词）与来源句。

### 2.3 claim 抽取

对答案分句：

- 跳过拒答/不确定表达（`I do not know`、`not stated`、`could not confirm` 等），它们不是事实 claim；
- 抽取 `reports_to`、`direct_report`、`team_size`、`responsible_for`、`leads`、`office`、`budget` 等；
- 句子含有实体和事实性动词但无法解析时，记为 `unverifiable` 的 fallback claim；
- 相同文本/主语/关系/值的重复 claim 去重；重复 source 去重。

### 2.4 判定规则

| verdict | 规则 |
|---|---|
| `supported` | source 中存在同一 subject + relation + target/value 的事实，且极性一致 |
| `contradicted` | source 对同一 subject 的**单值关系**给出不同对象/值；或同一责任被明确分配给另一主体；或 source 明确肯定被答案否定的同一事实 |
| `unverifiable` | 无法解析、无 source 事实、或只有“没有找到支持证据”（absence 不等于反证） |

单值关系：`reports_to`、`office`、`team_size`、`budget`、`leads`（按项目/人方向处理）、`joined_year`、`role_since`。

**明确不做的推断**：

- 不知道某个实体（如 Frank/Grace/CTO）不等于它不存在；仅当 source 对同一关系给出不同对象时才判 contradicted。
- 无法解析的复杂 claim 不猜测，保留 `unverifiable`。
- 答案中的否定只在与 source 的肯定事实直接冲突时判 contradicted；不同对象的否定 claim 不判矛盾。

### 2.5 输出指标

```text
claims                      : claim 列表
claim_support_rate          : supported / total claims
claim_contradiction_rate    : contradicted / total claims
citation_claim_precision    : supported / (supported + contradicted)
evidence_verdict            : supported | partially_supported | contradicted | unverifiable | no_evidence | no_answer
audit_coverage              : (supported + contradicted) / total claims（case 内）
valid_source_ids
invalid_source_ids
```

`citation_claim_precision` 的分母是**可判定 claim**，不是全部 claim；不能把它当成整体准确率。

---

## 3. 数据集构成及去重统计

### 3.1 原始记录 vs 去重记录 vs claim

| 集合 | 记录数 | 唯一 review_id / case | 唯一 case | claim 数 |
|---|---:|---:|---:|---:|
| 3 个历史错误 | 3 | 3 | 3 | 10 |
| 35 条 Stage 13 已知正确 | 35 | 35 | 17 | 62 |
| 10 条新增 cited-but-unsupported fixture | 10 | 10 | 10 | 18 |
| **合计** | **48** | **48** | **30**（历史/正确无 case 重叠，fixture 独立） | **90** |

> **重要**：90 个 claim 不是 90 个独立样本；同一记录的多个 claim 高度相关。误判率同时按 claim 和按记录/case 报告；不得把 claim 数当样本量夸大。

### 3.2 Stage 13 已知正确记录

- 来源：Stage 13.1 + 13.3 人工裁定 `correct` 的 35 条记录。
- 轮次：R1 11、R2 15、R3 9。
- 组别：traditional_rag 27、wiki_agent 8。
- 唯一 case：17；同一 case 的多轮记录不独立。
- claims：62，其中 supported 25、contradicted 0、unverifiable 37。

### 3.3 新增 fixture（10 条）

文件：`benchmark/datasets/evidence_audit_v0_cases.json`

| 类别 | 案例 |
|---|---|
| 错误人物归属/责任对调 | `ea01_cited_responsibility_swap` |
| 编造直接汇报关系 | `ea02_cited_fabricated_direct_reports` |
| 多跳关系错误连接 | `ea03_cited_inserted_cto` |
| 关系方向反转 | `ea04_cited_relation_reversal` |
| 错误金额 | `ea05_cited_wrong_budget` |
| 否定关系冲突 | `ea06_cited_negation_flip` |
| 引用无关页面 | `ea07_cited_irrelevant_page` |
| 多页面联合支持 | `ea08_multi_page_support` |
| 复杂/无法判断 claim | `ea09_complex_unverifiable` |
| 重复 claim / 重复 source | `ea10_duplicate_claim_and_source` |

fixture 与 `multihop_wiki.json` 分离，不修改原 50-case 数据集。

---

## 4. 逐案例审计结果

### 4.1 三个历史错误案例

给定其 required pages 作为**假设的有效引用**，审计错误答案本身（不是审计“无 source”）：

| case | verdict | claims | supported | contradicted | unverifiable | support_rate | contradiction_rate | citation_precision | coverage |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `mh38_responsibility_synthesis` | contradicted | 3 | 1 | 2 | 0 | 0.333 | 0.667 | 0.333 | 1.000 |
| `mh10_direct_reports` | contradicted | 4 | 2 | 2 | 0 | 0.500 | 0.500 | 0.500 | 1.000 |
| `mh49_orion_upward_chain_ceo` | contradicted | 3 | 2 | 1 | 0 | 0.667 | 0.333 | 0.667 | 1.000 |

逐 claim：

- mh38：Eve 负责 roadmap（contradicted）；Bob 负责 quarterly reviews（supported）；Bob 拥有 budget（contradicted）。
- mh10：Bob team_size=5（contradicted，source=12）；Alice/Carol 向 Bob 汇报（supported）；David 向 Bob 汇报（contradicted，source David→Eve）。
- mh49：David→Eve（supported）；Project Orion→David（supported）；Eve→CTO（contradicted，source Eve→CEO）。

### 4.2 35 条 Stage 13 已知正确记录

无 `contradicted`（0/35 记录，0/62 claim）。逐记录：

| review_id | verdict | claims | supported | contradicted | unverifiable |
|---|---|---:|---:|---:|---:|
| `s13_auto_correct_sample_r1_traditional_rag_mh21_team_intersection` | unverifiable | 0 | 0 | 0 | 0 |
| `s13_auto_correct_sample_r3_traditional_rag_mh24_engineering_budget_vs_projects` | unverifiable | 1 | 0 | 0 | 1 |
| `s13_auto_correct_sample_r2_traditional_rag_mh41_orion_tech_match` | partially_supported | 2 | 1 | 0 | 1 |
| `s13_auto_correct_sample_r2_traditional_rag_mh54_alice_carol_role_office_disambiguation` | partially_supported | 3 | 2 | 0 | 1 |
| `s13_auto_correct_sample_r2_wiki_agent_mh02_project_budget` | supported | 1 | 1 | 0 | 0 |
| `s13_auto_correct_sample_r2_wiki_agent_mh17_carol_two_project_leads` | partially_supported | 4 | 2 | 0 | 2 |
| `s13_risk_r2_traditional_rag_mh02_project_budget` | unverifiable | 1 | 0 | 0 | 1 |
| `s13_risk_r1_traditional_rag_mh18_orion_contributor_manager_office` | partially_supported | 2 | 1 | 0 | 1 |
| `s13_risk_r2_traditional_rag_mh20_phoenix_deadline_office` | partially_supported | 2 | 1 | 0 | 1 |
| `s13_risk_r3_traditional_rag_mh25_project_launch_order` | unverifiable | 0 | 0 | 0 | 0 |
| `s13_risk_r1_traditional_rag_mh40_walkway_data_team_synthesis` | unverifiable | 0 | 0 | 0 | 0 |
| `s13_risk_r2_traditional_rag_mh40_walkway_data_team_synthesis` | unverifiable | 0 | 0 | 0 | 0 |
| `s13_risk_r3_traditional_rag_mh40_walkway_data_team_synthesis` | unverifiable | 0 | 0 | 0 | 0 |
| `s13_risk_r3_traditional_rag_mh54_alice_carol_role_office_disambiguation` | supported | 2 | 2 | 0 | 0 |
| `s13_risk_r1_traditional_rag_mh24_engineering_budget_vs_projects` | supported | 1 | 1 | 0 | 0 |
| `s13_risk_r2_traditional_rag_mh24_engineering_budget_vs_projects` | unverifiable | 1 | 0 | 0 | 1 |
| `s13_risk_r3_wiki_agent_mh24_engineering_budget_vs_projects` | partially_supported | 2 | 1 | 0 | 1 |
| `s13_risk_r1_traditional_rag_mh08_ceo_unanswerable` | unverifiable | 1 | 0 | 0 | 1 |
| `s13_risk_r3_wiki_agent_mh08_ceo_unanswerable` | unverifiable | 0 | 0 | 0 | 0 |
| `s13_risk_r2_traditional_rag_mh29_alice_salary` | unverifiable | 1 | 0 | 0 | 1 |
| `s13_risk_r1_traditional_rag_mh30_david_university` | unverifiable | 1 | 0 | 0 | 1 |
| `s13_risk_r2_traditional_rag_mh30_david_university` | unverifiable | 0 | 0 | 0 | 0 |
| `s13_risk_r1_traditional_rag_mh50_orion_exact_completion_date` | unverifiable | 1 | 0 | 0 | 1 |
| `s13_risk_r2_traditional_rag_mh50_orion_exact_completion_date` | unverifiable | 0 | 0 | 0 | 0 |
| `s13_risk_r3_traditional_rag_mh50_orion_exact_completion_date` | unverifiable | 1 | 0 | 0 | 1 |
| `s13_risk_r1_traditional_rag_mh51_total_employee_count` | unverifiable | 0 | 0 | 0 | 0 |
| `s13_risk_r2_traditional_rag_mh51_total_employee_count` | unverifiable | 0 | 0 | 0 | 0 |
| `s13_risk_r3_traditional_rag_mh51_total_employee_count` | unverifiable | 0 | 0 | 0 | 0 |
| `s13_risk_r1_wiki_agent_mh46_phoenix_reporting_chain` | partially_supported | 5 | 4 | 0 | 1 |
| `s13_risk_r2_wiki_agent_mh46_phoenix_reporting_chain` | partially_supported | 8 | 3 | 0 | 5 |
| `s13_risk_r1_traditional_rag_mh47_exec_level_to_phoenix_chain` | unverifiable | 3 | 0 | 0 | 3 |
| `s13_risk_r1_wiki_agent_mh47_exec_level_to_phoenix_chain` | partially_supported | 8 | 3 | 0 | 5 |
| `s13_risk_r2_traditional_rag_mh47_exec_level_to_phoenix_chain` | unverifiable | 2 | 0 | 0 | 2 |
| `s13_risk_r2_wiki_agent_mh47_exec_level_to_phoenix_chain` | partially_supported | 6 | 2 | 0 | 4 |
| `s13_risk_r3_traditional_rag_mh47_exec_level_to_phoenix_chain` | partially_supported | 3 | 1 | 0 | 2 |

### 4.3 新增 fixture 10 条

| case | expected | actual | claim verdicts | 结果 |
|---|---|---|---|---|
| `ea01_cited_responsibility_swap` | contradicted | contradicted | contradicted, supported, contradicted | PASS |
| `ea02_cited_fabricated_direct_reports` | contradicted | contradicted | contradicted, supported, supported, contradicted | PASS |
| `ea03_cited_inserted_cto` | contradicted | contradicted | supported, supported, contradicted | PASS |
| `ea04_cited_relation_reversal` | contradicted | contradicted | contradicted | PASS |
| `ea05_cited_wrong_budget` | contradicted | contradicted | contradicted | PASS |
| `ea06_cited_negation_flip` | contradicted | contradicted | contradicted | PASS |
| `ea07_cited_irrelevant_page` | unverifiable | unverifiable | unverifiable | PASS |
| `ea08_multi_page_support` | supported | supported | supported, supported | PASS |
| `ea09_complex_unverifiable` | unverifiable | unverifiable | unverifiable | PASS |
| `ea10_duplicate_claim_and_source` | supported | supported | supported | PASS |

**fixture：10/10 匹配预期。**

---

## 5. 指标与误判率

### 5.1 已知正确记录（35 条 / 62 claims）

| 指标 | 数值 |
|---|---:|
| records | 35 |
| unique review_ids | 35 |
| unique cases | 17 |
| claims | 62 |
| supported claims | 25（40.3%） |
| contradicted claims | **0（0.0%）** |
| unverifiable claims | 37（59.7%） |
| claim_support_rate | 0.403 |
| claim_contradiction_rate | 0.000 |
| citation_claim_precision | 1.000（分母仅 25 条可判定 claim） |
| audit_coverage（case 级） | 14/35 = 0.400 |
| **已知正确 claim 的 contradicted 误判率** | **0/62 = 0.0%** |
| 已知正确 case 的 contradicted 误判率 | **0/35 = 0.0%** |

case-level verdict 分布：

| verdict | 数量 |
|---|---:|
| supported | 3 |
| partially_supported | 11 |
| unverifiable | 21 |
| contradicted | 0 |

> 0% 误判是保守规则的结果，不是高覆盖率的结果：59.7% 的 claim 是 unverifiable，不能据此声称 auditor 已能支持大多数答案。

### 5.2 历史错误（3 条 / 10 claims）

| 指标 | 数值 |
|---|---:|
| contradicted cases | **3/3** |
| contradicted claims | 5/10 |
| supported claims | 5/10 |
| unverifiable claims | 0/10 |
| 检出率（case） | 100% |

### 5.3 fixture（10 条 / 18 claims）

| 指标 | 数值 |
|---|---:|
| 预期匹配 | **10/10** |
| contradicted claims | 8/18 |
| supported claims | 8/18 |
| unverifiable claims | 2/18 |

---

## 6. 三个历史错误案例检出情况

| case | 检出 | 关键矛盾 claim | 关键 source |
|---|---|---|---|
| `mh38_responsibility_synthesis` | ✅ contradicted | Eve 负责 roadmap；Bob 拥有 budget | bob 页面：Bob 负责 roadmap/reviews；eve 页面：Eve 负责 budget |
| `mh10_direct_reports` | ✅ contradicted | Bob team_size=5（source 12）；David 向 Bob 汇报（source David→Eve） | bob 页面：team of twelve incl Alice/Carol；david 页面：David→Eve |
| `mh49_orion_upward_chain_ceo` | ✅ contradicted | Eve→CTO（source Eve→CEO） | eve 页面：Eve reports directly to CEO；无 CTO |

**结论：目标 3/3 达成。**

---

## 7. 误判和漏检分析

### 7.1 校准中修正的误判

v0 初版在 35 条正确记录上曾出现 3 例误判（mh46/mh47 多轮），原因是：

1. **代词主语误解析**：`She reports to Bob` 被归到错误的人；
2. **关系从句主语误选**：`Alice ... who reports to Bob ... leads Project Phoenix` 把 Bob 当成 leads 的主语；
3. **否定作用域过宽**：`Alice reports to Bob ... not a direct report [to Eve]` 把整句标为否定，导致把正确 claim 判成矛盾。

修正后：

- 代词开头的 claim 不再猜测主语；无法确定时跳过/保留 unverifiable；
- 关系从句取局部主语；
- 否定只在局部 subject 之后、关系词之前生效；
- 35 条正确记录 contradicted = 0。

### 7.2 已知正确记录的漏检/低覆盖

- 62 条 claim 中 37 条 unverifiable（59.7%）。
- 主要未覆盖类型：
  - 间接/隐含关系（“highest person”、“indirect”、“two levels above”）；
  - 多句组合/聚合（总和、比率、比较的完整表述）；
  - 同义改写（owns vs responsible for 之外的表达）；
  - 代词密集的链式答案；
  - 拒答/不确定答案（被跳过，不产生 claim）。
- 这不是“判错”，而是“没有判断”；报告必须保留 unverifiable，不能把它计入 support 或 contradiction。

### 7.3 历史错误的漏检风险

- mh10 的 Frank/Grace 无法被判 contradicted（语料没有他们的页面）；当前检出依赖 team_size 5 vs 12 和 David→Eve，而不是“Frank/Grace 不存在”。
- mh49 的 CTO 本身无法被证明不存在；检出依赖 Eve→CTO 与 Eve→CEO 的单值冲突。
- 如果 source 只引用 bob 页面、不引用 david 页面，mh10 的 David 矛盾可能变成 unverifiable；team_size 仍可检出。
- 如果 source 不包含 Eve 页面，mh49 的 CTO 矛盾会变成 unverifiable（没有 Eve→CEO 事实）。

### 7.4 审计器规则假设

- `reports_to`、`office`、`team_size`、`budget` 等被视为单值；
- “team of twelve engineers” 与 “five direct reports” 被视为同一 team-size 维度（启发式；mh10 同时有 David 关系矛盾，不单靠此规则）；
- 责任属性按关键词归并（roadmap/budget/quarterly_reviews/vision）；
- 这些假设对小语料有效，但不能直接外推到开放域。

---

## 8. 测试

### 8.1 命令

```bash
.venv/Scripts/python.exe -m pytest \
  tests/wiki/test_evidence_audit.py \
  tests/wiki/test_wiki_agent.py \
  tests/wiki/test_multihop_evaluation.py \
  tests/wiki/test_multihop_scoring.py \
  tests/wiki/test_multihop_comparison.py \
  tests/wiki/test_stage10_comparison.py \
  tests/wiki/test_stage11_ablation.py \
  tests/wiki/test_stage12_multiround.py \
  tests/web/test_ask.py \
  tests/tui/test_agent_flow.py -q
```

**结果：273 passed in 19.42s。**

`tests/wiki/test_evidence_audit.py` 单独运行：**14 passed**。

### 8.2 覆盖的测试场景

- 正确实体关系；
- 关系反转；
- 错误人物归属；
- 错误数值/金额；
- 否定冲突；
- 无关引用页面；
- 多页面联合支持；
- 无法判断的复杂 claim；
- 空答案、无来源、source 未实际读取；
- 重复 claim / 重复 source 去重；
- fixture 10 条预期匹配；
- 3 个历史错误 case 的 contradicted 检出。

### 8.3 校准命令

```bash
.venv/Scripts/python.exe benchmark/run_evidence_audit_v0.py \
  --output <path.json>
```

输出包含 historical / known_correct / fixture 三组逐案例结果。

---

## 9. 是否具备进入 Stage 15.2 的条件

### 9.1 验收标准对照

| 标准 | 结果 |
|---|---|
| 3 个历史错误至少检出 2，目标 3/3 | **3/3** |
| 已知正确 claim 误判 contradicted ≤10% | **0/62 = 0%** |
| 无法判断的 claim 保留 unverifiable | 是（37/62；fixture 2/18） |
| 报告未覆盖表达类型及失败案例 | 是（§7.2/7.3） |
| 逐案例结果，不只总体准确率 | 是（§4） |

### 9.2 结论

**具备进入 Stage 15.2 的条件，但仅限“评测扩展”，不具备 runtime gate 条件。**

- 评测层面：v0 能以 0% 误判检出 3/3 历史错误，且 fixture 10/10；可以作为离线审计基线。
- 但 coverage 只有 40%（case 级 14/35；claim 级 25/62），59.7% 的 claim 是 unverifiable；如果现在接入 runtime gate，会漏掉大量真实错误，或因为规则不全而误拒正常回答。
- 建议 Stage 15.2 范围：
  1. 扩展 paraphrase/否定/聚合/多句 claim 的审计规则；
  2. 在完整 50-case 已保存结果上校准（当前只有 35 条风险富集记录）；
  3. 明确 `unverifiable` 在门禁中的语义（不能当支持，也不能当反证）；
  4. 只有在新校准中保持误判 ≤10% 且 coverage 明显提升后，才讨论 runtime gate。
- 不建议在本阶段把 audit 结果写回 `answer_correct`/`grounded_correct`，也不建议接入 WikiAgent。

### 9.3 已知限制

1. 35 条正确记录来自 Stage 13 风险富集样本，不是全量 50-case 随机样本；17 个唯一 case，多轮记录不独立。
2. 90 个 claim 不是 90 个独立样本；指标是描述性的，不做统计显著性结论。
3. 审计器仅使用 cited sources；未引用但已读页面不参与支持判断。
4. 规则式审计不验证逐事实 entailment；`citation_claim_precision` 只对可判定 claim 有意义。
5. 未调用真实模型；使用已保存答案与页面内容。
6. fixture 是新增独立文件，不改现有数据集；未来若纳入正式 benchmark，需要单独评审。

---

## 附录：关键代码位置

| 文件:行 | 内容 |
|---|---|
| `benchmark/evidence_audit.py` | EvidenceAudit v0 实现 |
| `benchmark/run_evidence_audit_v0.py` | 离线校准脚本 |
| `benchmark/datasets/evidence_audit_v0_cases.json` | 10 条 fixture |
| `tests/wiki/test_evidence_audit.py` | 14 个回归测试 |
| `src/zmwiki/wiki/wiki_agent.py:579-619` | `_do_answer`（未修改；审计器不接入运行时） |
| `benchmark/run_multihop_evaluation.py:896-963` | `score_answer`（未修改） |
