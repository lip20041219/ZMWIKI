# Stage 14.3 — 修复后行为验收

- **验收日期**：2026-11-01
- **验收对象**：Stage 14.2 的零证据回答修复
- **范围**：Agent 行为回归、评测指标语义、定向测试、Web/TUI 兼容性
- **未执行**：外部模型 API、Live 评测、Stage 13 的 29 条 scorer false negative 优化
- **结论摘要**：Agent 修复通过定向验收（259 passed、3/3 历史案例回放）；指标语义已核对，但 `answer_correct` 已从旧词法语义变为状态感知正式语义，旧结果不可直接与新结果比较；`abstention_accuracy` 当前不存在，需要用现有 `unanswerable_handling` 或后续单独定义。

---

## 1. Agent 行为验收

| 验收项 | 验证方式 | 结果 |
|---|---|---|
| 正常可回答：search → read → answer | `test_search_read_answer_success` 等既有测试 | 通过 |
| 不可回答：正确返回证据不足/拒答 | `test_correct_refusal_with_sources_still_returns_answer`；mh08 R3 历史路径 | 通过 |
| 只有 search、没有 read | `test_search_candidates_not_treated_as_evidence` | 通过：`status=insufficient_evidence`，`answer=None` |
| protocol_error 后直接 answer | `test_protocol_error_then_answer_is_rejected` | 通过：不能返回正式答案 |
| 伪造 Read results | `test_fabricated_read_results_are_not_evidence` | 通过：不写入 `read_page_ids`，`answer=None` |
| 已读取页面但 source 无效 | `test_answer_without_sources_is_insufficient`、`test_invalid_source_forms_are_not_accepted` | 通过：`answer=None` |
| `_do_answer` 直接调用且 `read_pages=[]` | `test_do_answer_without_read_pages_returns_none` | 通过 |
| Web/TUI 对 `answer=None` | `tests/web/test_ask.py`、`tests/tui/test_agent_flow.py` | 通过 |

### 1.1 三个历史错误案例回放

使用 Stage 12 保存的原始 `raw_output` 与答案，通过 FakeTransport 离线回放固定后的 WikiAgent，**未调用外部模型 API**：

| 案例 | status | answer | pages_read | 步骤 |
|---|---|---|---:|---|
| `mh38_responsibility_synthesis` R1 | `insufficient_evidence` | `None` | 0 | search → protocol_error → answer(rejected) |
| `mh10_direct_reports` R2 | `insufficient_evidence` | `None` | 0 | search → protocol_error → answer(rejected) |
| `mh49_orion_upward_chain_ceo` R2 | `insufficient_evidence` | `None` | 0 | search → protocol_error → answer(rejected) |

结果：**3/3 不再产生正式答案**。

---

## 2. 评测指标兼容性

### 2.1 字段语义

| 字段 | 当前定义 | 旧结果中的语义 | 兼容性结论 |
|---|---|---|---|
| `lexical_match` | `score_answer(answer, case)` 的纯文本匹配；忽略 status/source | 旧结果没有该字段；旧 `answer_correct` 等价于它 | 新字段保留旧词法分；跨版本比较用它 |
| `answer_correct` | `lexical_match AND status == 'answered'` | 旧结果是纯词法匹配，不看 status | **语义已改变，不能直接比较** |
| `grounded_correct` | `answer_correct AND 至少一个 cited source` | 旧结果同样基于“答案正确 + answered + source”，但旧 answer_correct 是词法 | 公式相同；**仅表示 citation presence，不是逐事实支持率** |
| `evidence_recall` | 旧 access recall：`accessed_pages ∩ required`；包含搜索候选访问 | 相同 | 保持不变；不能当“实际阅读证据” |
| `read_evidence_recall` | `read_page_ids ∩ required`；只算真实 read/follow_links | 相同 | 保持不变；适合衡量实际阅读证据 |
| `abstention_accuracy` | **不存在** | 不存在 | 不能核对；当前最接近的是 `unanswerable_handling` |

### 2.2 `answer_correct` 的新旧差异

- **旧 Stage 10–13 结果**：`answer_correct = score_answer(answer, case)`，即使 `status=insufficient_evidence`、`sources=[]`，只要文本匹配也可能为 `True`。
- **新代码**：`answer_correct` 还要求 `status == 'answered'`；纯文本匹配保存在 `lexical_match`。
- **影响**：旧结果文件中的 `answer_accuracy` 是旧词法口径；新运行中的 `answer_accuracy` 是正式口径。两者不可直接对比。
- **旧文件状态**：Stage 13.0 metadata 记录的 13 个数据/结果文件 hash 全部匹配，未被覆盖；仅 `run_multihop_evaluation.py` 与 `test_multihop_evaluation.py` 的 hash 因本次修复发生变化（预期）。

### 2.3 `grounded_correct` 的边界

`grounded_correct` 只做三件事：

1. 答案文本匹配；
2. `status == 'answered'`；
3. `sources` 非空。

它**不检查**每个事实是否被 source 支持，也不计算逐事实 entailment。因此：

- 它不能描述为“严格证据支持率”或“逐事实正确率”；
- 只能描述为“citation-backed correctness”或“有引用来源的正确回答率”；
- 代码注释与指标 definition 已按此口径修正。

### 2.4 `evidence_recall` 与 `read_evidence_recall`

- `evidence_recall` 是旧 access recall，`accessed_pages` 包含搜索候选访问；即使 `pages_read=0`，也可能为 1.0。
- `read_evidence_recall` 只使用 `AgentResult.read_page_ids`，即真实 `wiki_read`/`follow_links` 返回的页面。
- Stage 13 的 3 个错误案例正是 `evidence_recall=1.0` 但 `read_evidence_recall=0.0` 的例子；验收时继续以 `read_evidence_recall` 作为实际阅读证据指标。

### 2.5 `abstention_accuracy`

- 当前代码、历史结果、文档中均不存在 `abstention_accuracy`。
- 现有最接近的指标是 `unanswerable_handling`：只统计 answerable=False 的 case 中，`answer_correct` 为 True 的比例。
- `unanswerable_handling` 现在也使用正式 `answer_correct`，因此只有“answered + 拒答词 + 文本匹配”才算正确；纯 `insufficient_evidence` abstention 不会被计入。
- **未新增该指标**：本阶段不新增功能；如需真正区分“拒答正确”与“答案正确”，应在后续阶段单独定义 `abstention_accuracy`，不能把 `unanswerable_handling` 直接改名为 abstention。

### 2.6 新旧结果对比规则

- **旧–旧**：可以用旧词法 `answer_accuracy` 比较；不要与新的正式口径混用。
- **新–新**：使用新的正式 `answer_accuracy`；同时报告 `lexical_match_rate` 作为诊断。
- **旧–新**：不得直接比较 `answer_accuracy`。应使用 `lexical_match`/`lexical_match_rate` 对齐旧词法分，或先用 `benchmark/rescore_multihop_results.py` 重评分。
- `benchmark/compare_multihop_runs.py` 会拒绝 source hash 不同的两次运行，因此不会静默混比旧/新代码的结果；但同 hash 的历史运行内部仍可能是旧词法口径，报告时需注明。

---

## 3. 实际执行的测试

### 3.1 定向测试命令

```bash
.venv/Scripts/python.exe -m pytest \
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

**结果：259 passed in 20.39s**。

覆盖：

- Agent 正常回答、拒答、零证据、伪 Read results、协议错误恢复；
- benchmark 评分、grounded/unsupported、lexical_match/answer_correct 语义；
- Stage 10/11/12 结果构造与评分；
- 新旧运行比较工具；
- Web/TUI 对 `answer=None` 的处理。

### 3.2 历史案例回放

- 3/3 历史错误案例返回 `insufficient_evidence` + `answer=None`；
- 不调用外部模型 API；
- 回放输入来自 `benchmark/results/stage12_multiround_20261101_120442_r1/r2.json` 的保存记录。

### 3.3 历史文件完整性

- Stage 13.0 metadata 的 13 个数据/结果文件 hash：**13/13 匹配**；
- 仅 `benchmark/run_multihop_evaluation.py` 和 `tests/wiki/test_multihop_evaluation.py` 的 hash 与旧记录不同（本次修复的预期变化）；
- 未覆盖 dataset、Stage 10–13 结果、人工裁定文件。

---

## 4. 尚未解决的风险

1. **`answer_correct` 语义迁移**：旧结果仍是旧词法语义；任何跨版本报告必须显式标注，不能把旧 `answer_accuracy` 与新正式 `answer_accuracy` 直接比较。
2. **`abstention_accuracy` 缺失**：当前没有独立的拒答/abstention 指标；`unanswerable_handling` 只覆盖 answered 拒答，不覆盖纯 `insufficient_evidence` abstention。
3. **`grounded_correct` 不是逐事实支持率**：它只检查 citation presence；source 是否真正支持答案未被验证。
4. **protocol_error 仍丢弃前导 read**：本次阻止了零证据 answer，但没有执行被协议错误吞掉的 read；后续可改进恢复策略。
5. **无真实模型验证**：259 个测试和 3 个历史回放均为离线；真实模型在新门禁下的重试/补读行为未实测。
6. **Web/TUI 行为变化**：无证据文本不再作为“未验证备注”显示；TUI/Web 测试已更新，但产品侧需确认这是期望行为。
7. **29 条 scorer false negative 未处理**：属于后续评分器优化，不在本阶段。
8. **旧结果与新结果混合比较**：`compare_multihop_runs.py` 的 source hash 门禁能阻止不同代码版本混比，但不能识别同 hash 下的旧语义；需要在报告层继续标注。

---

## 5. 是否具备进入下一阶段的条件

**有条件具备。**

- Agent 零证据回答修复：**通过**（259 passed，3/3 历史案例）。
- 正常回答、拒答、协议错误、伪 Read results、无效 source：**通过**。
- 评测字段语义：**已核对并记录**；`answer_correct` 语义迁移与 `abstention_accuracy` 缺失是已知边界，不是本次修复的阻塞项。
- 建议在进入下一阶段前确认两件事：
  1. 接受 `answer_correct` 的新正式语义，并在所有新旧对比中使用 `lexical_match` 或重评分；
  2. 决定是否需要独立的 `abstention_accuracy`；若需要，另行定义并加测试，不把 `unanswerable_handling` 改名复用。
- 真实模型验证需单独授权；本阶段未调用 API，未进入 Stage 14.3 之后的阶段。
