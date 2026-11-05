# Stage 15.0 — Evidence-grounded Answering 可行性分析

- **分析日期**：2026-11-01
- **性质**：只读分析；未修改源码、测试、数据集或历史结果；未调用模型 API；未引入依赖。
- **目标**：评估从“有来源才能回答”升级到“答案事实必须受到来源支持”的最小可行方案。
- **结论摘要**：当前证据链是**页面级**而非**事实级**；benchmark 能识别“无来源回答”，但**不能识别“有来源但来源不支持答案”**。三个历史错误如果在读取页面后引用真实 source，现有评分仍会判为正确且 grounded。数据集已有 24 条 reasoning_chain 和 10 条结构化答案检查，足以支撑一个**离线、规则式 EvidenceAudit v0**；建议进入 Stage 15.1，但先只做评测审计，不做 runtime gate。

---

## 1. 当前证据链架构

### 1.1 数据流

```text
search → read/follow_links
        │
        ├─ WikiPage.content / WikiPage.sources  （页面正文与文档来源）
        ├─ AgentStep.read_page_ids             （本步实际读取的页面 ID）
        └─ AgentResult.read_page_ids           （全部实际读取页面 ID）
                │
                ▼
        answer 动作（只有纯文本 + 页面级 sources）
                │
                ├─ AgentResult.answer                  （答案全文）
                ├─ AgentResult.sources                 （引用页面 path）
                └─ AgentStep.params["text"][:200]      （截断的诊断副本）
                │
                ▼
        benchmark.run_agent
                ├─ CaseResult.answer
                ├─ CaseResult.sources                  （页面 ID）
                ├─ CaseResult.read_page_ids            （全部读取页面 ID）
                ├─ CaseResult.evidence_recall          （access recall）
                ├─ CaseResult.read_evidence_recall     （实际阅读 recall）
                └─ CaseResult.grounded_correct         （answer_correct + 至少一个 source）
```

### 1.2 关键字段的实际含义

| 字段 | 位置 | 实际内容 | 是否支持事实级映射 |
|---|---|---|---|
| `WikiPage.content` | `wiki_schema.py:57-83` | 页面正文，约 5 行/页 | 是事实来源，但不结构化 |
| `WikiPage.sources` | 同上 | 编译该页面的 `Document.id`，不是答案事实来源 | 否 |
| `AgentStep.read_page_ids` | `wiki_agent.py:130-157` | 本步 read/follow_links 实际返回的页面 ID | 只能定位页面，不能定位句子 |
| `AgentResult.read_page_ids` | `wiki_agent.py:160-181` | 全部实际读取页面 ID | 同上 |
| `AgentResult.sources` | 同上 | 答案引用的页面 path（已通过 read_pages 校验） | 页面级，不是逐事实 |
| `AgentResult.answer` | 同上 | 纯文本答案，仅 `status=answered` 时非空 | 无 claim/source 结构 |
| `CaseResult.sources` | `run_multihop_evaluation.py:980-1032` | 由 `sources` path 解析出的页面 ID | 页面级 |
| `CaseResult.read_page_ids` | 同上 | 实际读取页面 ID 全集 | 页面级 |
| `evidence_recall` | 同上 | `accessed_pages ∩ required`；包含搜索访问 | 否，是 access 指标 |
| `read_evidence_recall` | 同上 | `read_page_ids ∩ required`；实际阅读页面 | 页面级 |
| `grounded_correct` | 同上 | `answer_correct AND 至少一个 cited source` | **仅 citation presence，不验证内容支持** |
| Web `evidence` | `web/server.py:960-980` | 由 `result.sources` 生成的页面元数据 | 页面级展示，不是 claim evidence |

### 1.3 传播过程中丢失的信息

1. **页面正文不进入 `AgentResult`**：`read_pages` 只在 `answer()` 内部存在；结果只保留 `read_page_ids`。
2. **没有 claim → source 的映射**：答案是一个字符串，`sources` 是一个页面列表；无法知道哪句话来自哪一页。
3. **步骤级 text 被截断**：`AgentStep.params["text"]` 只保留前 200 字符，不能用于完整 claim 分析；完整文本只在 `AgentResult.answer`/`CaseResult.answer`。
4. **`WikiPage.sources` 不是证据来源**：它是编译页面的文档 ID，不能反查答案事实。
5. **`evidence` 字段不是 AgentResult 的一部分**：Web API 临时构造页面元数据，不参与 benchmark。

> 结论：当前链只能回答“答案是否引用了真实读取的页面”，不能回答“答案中的每个事实是否由这些页面的内容支持”。

---

## 2. 已有能力与缺失能力

### 2.1 已有能力

| 能力 | 实现 | 能做什么 |
|---|---|---|
| 实际阅读页面校验 | `_do_read` / `read_pages` | 搜索候选不算证据；只有 `wiki_read` 成功返回的页面进入 `read_pages` |
| source 有效性校验 | `_valid_source_ids` | 只接受实际读取页面的 ID；无效 source 被丢弃 |
| 零证据回答拦截 | Stage 14.2 修复 | `read_pages=[]` 时 answer=None，status=insufficient_evidence |
| 状态感知正确率 | `answer_correct = lexical_match AND status=answered` | 防止无证据答案计入正式正确率 |
| 结构化答案检查 | `score_answer` 的 `answer_bindings`、`relation_chain`、`ratio_checks`、`comparison_checks`、`shared_entities` | 检查答案文本中的实体绑定、关系顺序、数值比较 |
| 页面级证据指标 | `evidence_recall`、`read_evidence_recall`、`source_precision` | 衡量页面覆盖与引用重叠 |

### 2.2 缺失能力

| 缺失 | 影响 |
|---|---|
| claim 级答案表示 | 无法把“Bob 负责 roadmap”和“Eve 负责预算”分别映射到 bob/eve 页面 |
| 来源内容支持检查 | 引用真实页面但答案内容不支持时，仍可能通过 |
| 逐事实/逐句审计 | 只能看整段答案与关键词组，不能定位错误句 |
| 矛盾检测 | 答案与页面内容相反（如 Bob/Eve 责任对调）不一定会被关键词检查发现 |
| 引用-事实精度 | `source_precision` 只衡量页面是否属于 required，不衡量引用是否支持具体 claim |
| 通用 paraphrase 匹配 | 结构化检查只覆盖部分 case；其余主要是关键词组 |
| 统一的 abstention 指标 | `unanswerable_handling` 只覆盖 answered 拒答，不覆盖纯 insufficient_evidence |

---

## 3. 现有 50-case 数据集覆盖情况

### 3.1 基础分布

| 维度 | 数量 |
|---|---:|
| cases | 50 |
| answerable | 43 |
| unanswerable | 7 |
| 有 `reasoning_chain` | 24 |
| 有 `attribute_lookups` / `relation_edges` / `reasoning_hops` | 24 |
| 有 `answer_bindings` | 6 |
| 有 `relation_chain` | 2 |
| 有 `comparison_checks` | 1 |
| 有 `ratio_checks` | 1 |
| 有 `shared_entities` | 1 |
| 有 `alternative_evidence_sets` | 13 |
| `manual_review=true` | 6 |
| `distractor_page_ids` / `correct_evidence_page_ids` | 2 / 2 |
| `evidence_note` | 27 |

### 3.2 能覆盖事实混淆/关系颠倒/跨页推断的案例

| 风险类型 | 代表 cases | 现有结构化检查 |
|---|---|---|
| 事实/责任混淆 | `mh38_responsibility_synthesis`、`mh36_phoenix_carol_bob_responsibilities`、`mh42`–`mh45`、`mh53`、`mh54` | mh42–45/53/54 有 `answer_bindings`；mh38 只有 `reasoning_chain`（评分器不使用） |
| 关系方向颠倒 | `mh46`、`mh47`、`mh48`、`mh49`、`mh31`–`mh35` | mh46/mh47 有 `relation_chain`；其余主要靠关键词组 |
| 跨页错误推断 | `mh24`、`mh37`、`mh38`、`mh42`–`mh45`、`mh49`、`mh05`、`mh06`、`mh17` | mh24 comparison、mh37 ratio；其余部分有 reasoning_chain 但评分器不用 |
| 干扰页选择 | `mh53`、`mh54` | `distractor_page_ids` + `answer_bindings` |
| 不可回答/拒答 | `mh08`、`mh28`–`mh30`、`mh50`–`mh52` | refusal/fabrication patterns；mh50–52 有 manual_review |

### 3.3 三个历史错误的覆盖

| case | 有 reasoning_chain | 有可执行答案检查 | 当前自动评分能否抓错 |
|---|---|---|---|
| `mh38_responsibility_synthesis` | 是（明确 Bob/Eve 责任） | 否（无 answer_bindings/relation_chain） | 不能；关键词组可通过 |
| `mh10_direct_reports` | 否 | 否 | 不能；关键词组可通过 |
| `mh49_orion_upward_chain_ceo` | 是（Orion→David→Eve→CEO） | 否（无 relation_chain） | 不能；关键词组可通过 |

> 数据集“包含”这些风险案例，但**不等于 benchmark 能自动验证它们**。mh38/mh49 的 reasoning_chain 是人工审计注解，当前评分器没有使用；mh10 连 reasoning_chain 都没有。

---

## 4. 不依赖模型调用时，基础验证是否可行？

### 4.1 可行性判断

**可行，但只对部分事实类型成立。** 语料很小：9 个页面、约 54 个句子，事实以显式关系/属性为主，适合规则式抽取。

| 事实类型 | 可否规则验证 | 依据 |
|---|---|---|
| 实体-属性（职位、楼层、预算、日期、人数） | 高 | 页面句子显式；可抽取 subject + attribute + value |
| 实体-关系（reports_to、manages、leads、located_in） | 高 | 页面使用固定动词；方向可由主宾顺序判断 |
| 多实体比较（更长、更早、更高） | 中-高 | mh24/mh42–45 已有 comparison/bindings；需数值/年份解析 |
| 数值聚合/比率 | 中 | mh37 已有 ratio_checks；需少量算术 |
| 否定/不可回答 | 中 | 需要否定词与“全库不存在”检索；现有 fabrication 规则是启发式 |
| 自由文本 paraphrase | 中-低 | 规则只能覆盖显式实体/关系；同义表达会漏 |
| 多句/多跳综合 | 中-低 | 需要 claim 分解与跨页组合；没有模型时只能做限定模式 |

### 4.2 可复用的数据集注解

- `reasoning_chain`（24 条）：每条是 `page: sentence` 或 `synthesis: ...`，可作为**离线评测的期望事实链**。
- `answer_bindings`（6 条）：可直接复用现有 `_check_answer_bindings`，但它是答案文本检查，不是来源内容检查。
- `relation_chain`（2 条）：可直接复用方向/顺序检查。
- `comparison_checks` / `ratio_checks`（各 1 条）：可复用数值/方向检查。
- `alternative_evidence_sets`（13 条）：可用于判断 source 是否属于合法证据路径。

### 4.3 不能直接解决的部分

- mh10 没有 reasoning_chain，也没有结构化检查；需要通用关系抽取（Bob manages team incl Alice/Carol；David reports to Eve；Frank/Grace 不存在）。
- 自由文本答案的事实分解没有现成结构；必须做 sentence/claim segmentation。
- “引用存在但不支持”的场景需要把 answer claim 与 source sentence 做匹配，而不是只比较 answer 与 reference。
- 不能用 `reasoning_chain` 在运行时验证（会泄漏参考答案）；它只能用于**离线评测 oracle**。

---

## 5. Stage 14 修复后仍存在的路径

Stage 14.2 只拦截了 `read_pages=[]` 的 answer。以下路径仍然存在：

### 5.1 “读了页面，但答案编造/错误连接”

1. 模型读取 bob/eve，然后给出责任对调的答案，并引用 `["bob","eve"]`。
2. 模型读取 bob/alice/carol，然后编造五个 direct reports，并引用 `["bob"]`。
3. 模型读取 project_orion/david/eve，然后插入 CTO，并引用三页。
4. 模型引用真实页面，但某个 claim 来自模型先验或另一页；页面级 source 仍有效。
5. 模型引用多个页面，其中一页支持 A，另一页不支持 B；没有逐 claim 映射。

### 5.2 现有评分为什么抓不到

- `score_answer` 只比较答案文本与 reference keywords/结构化注解，不读取 source 内容。
- `grounded_correct` 只要求 `answer_correct + 至少一个 source`。
- `source_precision` 只衡量 cited pages 与 required/alternative 页面的交集，不衡量“该页是否支持该句”。
- 结构化检查只覆盖 10/50 个 case 的特定关系/绑定，且只看 answer 文本。

### 5.3 直接证据

用 Stage 13 的历史错误答案（错误内容）运行 `score_answer`：

| case | 历史错误答案 | `score_answer` | 如果 status=answered 且 sources 非空 |
|---|---|---|---|
| `mh38_responsibility_synthesis` | 把 Bob/Eve 责任对调 | **True** | `grounded_correct=True` |
| `mh10_direct_reports` | 编造 Frank/Grace/David 为 direct reports | **True** | `grounded_correct=True` |
| `mh49_orion_upward_chain_ceo` | 在 Eve 与 CEO 间插入 CTO | **True** | `grounded_correct=True` |

> 这是本阶段最重要的可行性依据：**Stage 14 修复解决了“无来源回答”，但没有解决“有来源但不支持”**。如果只用现有指标，模型只要读取并引用真实页面，仍可能通过错误答案。

---

## 6. 最小可行实现方案

### 6.1 推荐：离线 EvidenceAudit v0（评测层，不改 Agent）

**目标**：在不调用模型、不改变 Agent 行为的前提下，测量“答案 claim 是否被引用的已读页面支持”。

**输入**：

- `CaseResult.answer`（完整答案）
- `CaseResult.sources`（cited page IDs）
- `CaseResult.read_page_ids`（实际读取页面）
- `WikiIndex` / corpus 页面内容（离线可重建）
- case 的可选注解：`reasoning_chain`、`answer_bindings`、`relation_chain`、`comparison_checks`、`ratio_checks`

**处理步骤**：

1. **事实抽取**：对每个 source/read page 的句子抽取简单事实：
   - `(subject, relation, object, polarity, sentence)`
   - 关系模式：`reports to`、`manages`、`leads`、`is responsible for`、`is located in/on`、`works in`、`joined in`、`holds`、`has` 等。
   - 属性值：职位、楼层、建筑、预算、日期、人数、学位、项目角色。
2. **claim 分解**：按句号/分号/连接词拆分答案；对每个句子抽取实体、关系、属性值。
3. **支持判定**：
   - **supported**：存在同一 source 页面的 fact，实体别名匹配、关系/属性一致、值一致、极性一致；
   - **contradicted**：存在 source fact 但实体/关系/值相反（如 Bob/Eve 对调、关系方向反了、否定翻转）；
   - **unverifiable**：无法抽取或无法匹配（不是“错误”，需单独统计）；
   - **no_source_claim**：claim 未落在任何 cited source。
4. **输出**：
   - 每 case：`claims_total`、`claims_supported`、`claims_contradicted`、`claims_unverifiable`、`claim_support_rate`、`contradicted_claim_rate`、`evidence_verdict`。
   - 不改变现有 `answer_correct`/`grounded_correct`/`evidence_recall`。

**数据集注解的使用**：

- `reasoning_chain` 只用于**评测 oracle**（检查审核器是否覆盖该 chain），不进入 runtime。
- `answer_bindings`/`relation_chain`/`comparison_checks`/`ratio_checks` 可作为高置信度 case 的额外检查，与通用规则并用。
- 对 mh38/mh49，reasoning_chain 能直接给出期望事实；对 mh10，需要通用关系抽取。

**建议新增字段**：

```text
claim_count
claim_supported_count
claim_contradicted_count
claim_unverifiable_count
claim_support_rate
claim_contradiction_rate
citation_claim_precision
evidence_verdict  (supported | partially_supported | contradicted | unverifiable)
audit_coverage    (boolean: 至少一个 claim 可检查)
```

### 6.2 不建议现在做：runtime gate

- 在 `_do_answer` 内用同一 auditor 检查 `read_pages`，失败则返回 insufficient/None。
- 风险：规则召回不足会导致正常答案被错误拒绝；必须先测 false positive。
- 建议只在 Stage 15.1 评测通过后，作为 Stage 15.2 的可选 gate，且只对**高置信度矛盾**（实体对调、关系反向、否定翻转）拒绝。

### 6.3 不建议现在做：结构化 claim 输出协议

- 要求模型输出 `claims: [{text, source_id, quote}]`，可直接验证 quote substring 和 claim 绑定。
- 优点：最强、最可审计；缺点：改 Prompt/输出契约/benchmark/所有调用方，成本远高于离线 audit。
- 作为 Stage 15.1 的备选，不列入最小方案。

### 6.4 不建议：LLM-as-judge / 多 Agent 验证

- 需要模型调用，违反本阶段限制；成本与不可复现性高。
- 不作为最小方案。

---

## 7. 建议新增的评测案例

建议在后续阶段新增 8–10 条，专门覆盖“有引用但不支持”：

| 建议 ID | 目的 | 关键注解 |
|---|---|---|
| `mh55_cited_responsibility_swap` | 引用 bob/eve 但把责任对调 | `answer_bindings` + `reasoning_chain` + 期望 contradicted |
| `mh56_cited_fabricated_direct_reports` | 引用 bob 但编造 direct reports | `reasoning_chain` + fabrication patterns |
| `mh57_cited_inserted_cto` | 引用 Orion/David/Eve 但插入 CTO | `relation_chain` + unknown entity 检查 |
| `mh58_cited_relation_reversal` | Alice/Bob 汇报方向反了但引用两页 | `relation_chain` + forbidden_direct_pairs |
| `mh59_cited_negation_flip` | “Eve 不负责预算”但关键词齐全 | negation + `answer_bindings` |
| `mh60_cited_wrong_aggregation` | 引用三个预算页但合计/比率错 | `ratio_checks` + arithmetic |
| `mh61_cited_wrong_cross_page_binding` | 每个事实单独真，但绑错人/项目 | `answer_bindings` + `binding_entities` |
| `mh62_cited_unanswerable_fabrication` | 不可回答问题却引用真实页面编造答案 | refusal/fabrication + `manual_review` |
| `mh63_distractor_citation` | 引用 distractor 页而非 correct evidence 页 | `distractor_page_ids` + `correct_evidence_page_ids` |
| `mh64_correct_paraphrase_control` | 正确 paraphrase + 正确引用 | 用于测 auditor false positive |

每条应包含：`reference_answer`、`required_page_ids`、`reasoning_chain`、必要的 `answer_bindings`/`relation_chain`/`ratio_checks`/`comparison_checks`、`manual_review`；不要修改现有 case。

---

## 8. 预期指标

| 指标 | 定义 | 用途 |
|---|---|---|
| `claim_count` | 可分解的 answer claim 数 | 审计覆盖率分母 |
| `claim_support_rate` | supported / claim_count | 主要证据支持指标 |
| `claim_contradiction_rate` | contradicted / claim_count | 事实错误/关系颠倒 |
| `claim_unverifiable_rate` | unverifiable / claim_count | 规则无法判断的比例；不能当错误 |
| `citation_claim_precision` | supported claims / claims attributed to cited pages | 引用是否支持事实 |
| `evidence_verdict` | supported / partially_supported / contradicted / unverifiable | case 级结论 |
| `audit_coverage` | 至少一个 claim 可检查的 case 比例 | auditor 覆盖面 |
| `auditor_false_positive_rate` | 人工确认正确但 auditor 判 contradicted 的比例 | 关键安全阈值 |
| `auditor_false_negative_rate` | 人工确认错误但 auditor 未判 contradicted 的比例 | 漏检率 |

**建议的 Stage 15.1 初始目标**（proposed，需用实际数据校准）：

1. 3 个历史错误答案在“假设有合法 sources”时，至少 2 个被判 `contradicted` 或 `unsupported`；
2. 已知正确的 35 条 Stage 13 记录中，auditor 误判 `contradicted` 的比例 ≤ 10%；
3. `audit_coverage` ≥ 60%（至少覆盖有 reasoning_chain/结构化注解的案例）；
4. 所有新指标可复现、无模型调用、不影响旧指标。

---

## 9. 实现成本与性能风险

| 项 | 评估 |
|---|---|
| 新代码量 | 约 300–600 行 stdlib 模块 + 1 个测试文件（估计） |
| 依赖 | 无；正则/字符串/算术 |
| 数据变更 | 不需要；用现有 corpus + case annotations |
| 运行时成本 | 9 页/54 句，正则匹配毫秒级；50-case 审计 <1s |
| 主要风险 | paraphrase/别名/否定/聚合/多句 claim 导致漏判；规则过严导致误判正确回答 |
| 过拟合风险 | 9 页小语料容易针对案例调规则；需 paraphrase 对照与 adversarial cases |
| 评测泄漏 | `reasoning_chain` 只能做 oracle，不能进 runtime |
| 维护成本 | 规则集需要随 corpus/问题类型扩展；应保持最小、可测 |

---

## 10. 是否值得进入 Stage 15.1

**值得，但建议限定为“离线评测审计”，不是 runtime gate。**

理由：

1. **问题真实存在**：Stage 14 后仍有“读页面 + 引用 + 编造/错连”的路径；现有 `grounded_correct` 只检查 citation presence。
2. **技术可行**：语料只有 9 页、事实显式；24 条 reasoning_chain + 10 条结构化检查可做 bootstrap；无需模型调用或新依赖。
3. **风险可控**：先做离线审计，不改变 Agent 回答；用 35 条已知正确和 3 条已知错误校准 false positive/negative，再决定是否 runtime gate。
4. **成本合理**：一个小模块 + 测试，远低于结构化 claim 输出协议或 LLM judge。

**Stage 15.1 建议范围**：

- 实现 `EvidenceAudit v0`（离线，仅 benchmark 侧）；
- 新增 claim-level 字段与聚合指标；
- 新增 8–10 条“cited but unsupported”案例；
- 用 3 个历史错误 + 35 条正确记录做校准；
- **不修改 Agent 行为**，不把 audit 结果写回 `answer_correct`/`grounded_correct`；
- 若 false positive 可接受，再讨论 Stage 15.2 runtime gate 或结构化 claim 协议。

**停止条件**：

- 若 auditor 在 35 条正确记录上误判 contradicted >10%，或 3 个历史错误漏检 ≥2 个，则不上 runtime gate，先改进规则或转向结构化 claim 输出；
- 不把 `reasoning_chain` 用作 runtime 真值，避免泄漏；
- 不进入 Evidence Judge 多 Agent 架构。

---

## 附录：关键代码位置

| 文件:行 | 内容 |
|---|---|
| `src/zmwiki/wiki/wiki_schema.py:57-83` | `WikiPage` 字段定义 |
| `src/zmwiki/wiki/wiki_agent.py:130-157` | `AgentStep`（read_page_ids/search_hit_page_ids） |
| `src/zmwiki/wiki/wiki_agent.py:160-181` | `AgentResult`（answer/sources/read_page_ids） |
| `src/zmwiki/wiki/wiki_agent.py:275` | `read_pages` 累积 |
| `src/zmwiki/wiki/wiki_agent.py:500-528` | `_do_read` 成功读取页面 |
| `src/zmwiki/wiki/wiki_agent.py:579-619` | `_do_answer` 只校验 source ID，不校验内容 |
| `benchmark/run_multihop_evaluation.py:980-1032` | `CaseResult` 字段 |
| `benchmark/run_multihop_evaluation.py:1490-1545` | Agent 结果提取与 CaseResult 构造 |
| `benchmark/run_multihop_evaluation.py:896-963` | `score_answer`（文本匹配/结构化检查） |
| `benchmark/run_multihop_evaluation.py:1094-1110` | 页面级证据指标 |
| `benchmark/run_multihop_evaluation.py:1542` 附近 | `grounded_correct` 仅检查 source 存在 |
| `src/zmwiki/web/server.py:960-980` | Web `evidence` 页面元数据 |
