# ZMWiki 多跳评测集人工扩题工作表

状态：人工设计用。首批 15 题中，mh17、mh18、mh20、mh21、mh23、mh24、mh25、mh26、mh28、mh29、mh30 已录入 `multihop_wiki.json`（正式集 26 题）；mh16、mh19、mh22、mh27 仍为草稿。本文件保留槽位、事实依据、设计统计和验收标准。

依据文件（全部现场核对）：

- `benchmark/datasets/multihop_wiki.json`：现有 26 题、字段、评分规则。
- `benchmark/run_multihop_evaluation.py`：数据校验、`_PAGE_LINKS`、`_CANDIDATE_LINKS`、评分逻辑、聚合指标。
- `benchmark/corpus/wiki/*.txt`：9 个知识页面，页面 ID 与文件对应关系见 `_PAGE_LINKS`。

使用顺序：

1. 从第 7 节挑槽位，按第 8 节模板填写。
2. 用第 4、5、6 节核对每个事实和页面 ID 是否真实存在。
3. 用第 9 节验收清单逐条检查。
4. 录入 JSON 后，先用**离线模式**验证 schema 和页面 ID（不调用模型），再决定是否跑真实模型。

---

## 1. 数据集硬约束

### 1.1 字段

| 字段 | 是否必需 | 说明 |
|---|---|---|
| `id` | 是 | 唯一，建议 `mhNN_short_slug`。现有 15 题已占用 `mh01`–`mh15`。 |
| `question` | 是 | 交给被测系统的自然语言问题。 |
| `reference_answer` | 是 | 人工标准答案，只给评测者，不传给被测系统。 |
| `answer_keyword_groups` | 可回答题必需 | 外层列表是 AND，内层列表是 OR；每个内层组至少命中一个关键词才算正确。 |
| `refusal_keywords` | 不可回答题必需 | 出现任一关键词才算正确拒绝。缺少该字段时不可回答题无法得分。 |
| `answerable` | 是 | `true` / `false`。 |
| `required_page_ids` | 是 | 必须是 `_PAGE_LINKS` 的键之一（见 1.2）。不可回答题按现有约定填 `[]`。 |
| `hops` | 是 | 现有约定：大致等于需要串联的页面数/关系步数；`hops >= 2` 会被计入多跳指标。 |
| `question_type` | 是 | 自由字符串，只记录、不参与评分；建议保持稳定词汇，便于后续分组统计。 |

### 1.2 合法页面 ID

`alice`、`bob`、`carol`、`project_phoenix`、`project_orion`、`david`、`eve`、`building_a`、`building_b`

对应文件：

| page_id | 文件 |
|---|---|
| `alice` | `alice_smith.txt` |
| `bob` | `bob_jones.txt` |
| `carol` | `carol_chen.txt` |
| `project_phoenix` | `project_phoenix.txt` |
| `project_orion` | `project_orion.txt` |
| `david` | `david_kim.txt` |
| `eve` | `eve_martinez.txt` |
| `building_a` | `building_a.txt` |
| `building_b` | `building_b.txt` |

### 1.3 评分规则

来自数据集 `scoring_rule`：

- 可回答题：`answer_correct =` 每个 `answer_keyword_groups` 组至少命中一个关键词。
- 不可回答题：`answer_correct =` 命中任一 `refusal_keywords`。
- 匹配前会统一转小写并去掉标点，**不做语义判断**。同义改写可能被判错，关键词堆砌可能被判对。
- 因此关键词要覆盖标准答案的多种自然表达，同时避免过于宽泛的词。
- 不要用单字母关键词区分楼栋：`"b"` 会命中 `"building"` 里的 `b`。要区分楼栋时用 `"building a"` / `"building b"` 这类短语，或改用更独特的词。

### 1.4 已有不可回答题的实际写法

参考 `mh08_ceo_unanswerable`：

```json
{
  "id": "mh08_ceo_unanswerable",
  "question": "Who is the CEO of Acme Corp?",
  "reference_answer": "The CEO is not named in any page. Eve Martinez reports to the CEO, but the CEO's identity is not provided.",
  "answer_keyword_groups": [],
  "refusal_keywords": [
    "cannot", "could not", "not mentioned", "no information", "unclear",
    "not found", "unknown", "not stated", "does not say", "don't know", "not provided"
  ],
  "answerable": false,
  "required_page_ids": [],
  "hops": 0,
  "question_type": "unanswerable"
}
```

---

## 2. 首批 15 题分布（历史记录）

> 本节统计的是扩题前的 15 题，仅作设计历史保留，不代表当前数据集。正式集为 26 题；mh03 现已改为 `two_page_chain` / `hops: 2`，当前分布以 `multihop_wiki.json` v1.2 为准。

### 2.1 题型

| question_type | 数量 | 题目 |
|---|---:|---|
| `single_page_fact` | 2 | mh13, mh15 |
| `single_page_by_title` | 1 | mh11 |
| `single_page_relation` | 1 | mh12 |
| `single_page_by_attribute` | 1 | mh07 |
| `two_page_relation` | 4 | mh01, mh02, mh04, mh14 |
| `two_page_negative` | 1 | mh09 |
| `three_page_chain` | 1 | mh03 |
| `three_page_aggregation` | 2 | mh05, mh10 |
| `three_page_intersection` | 1 | mh06 |
| `unanswerable` | 1 | mh08 |

### 2.2 跳数

| hops | 数量 | 题目 |
|---:|---:|---|
| 0 | 1 | mh08 |
| 1 | 5 | mh07, mh11, mh12, mh13, mh15 |
| 2 | 5 | mh01, mh02, mh04, mh09, mh14 |
| 3 | 4 | mh03, mh05, mh06, mh10 |
| 4+ | 0 | — |

### 2.3 覆盖结论

- **4 跳及以上为 0**；3 跳只有 4 题，其中真正的跨页链只有 mh03。
- **链接探索不足**：只有 mh03 需要沿“项目 → 负责人 → 上级”走；没有 4 页以上的链接链。
- **聚合只有求和**：mh05 是两个项目预算求和；没有差值、多少、先后、跨楼栋合计。
- **反向关系只有一处**：mh10 列举 Bob 的下属；没有“经理的经理”“某团队包含谁”“某楼住着哪个组”的反向题。
- **否定题只有一处**：mh09 是成员关系否定；没有地点、汇报关系、角色否定。
- **不可回答只有 1 题**：mh08 是实体缺失；没有属性缺失、虚假前提、跨页链断裂。
- **未使用的真实事实**：项目技术栈、团队例会日、学位、前雇主、年度工程预算、楼栋层数、健身房、行政层、Bob 管理的 12 人、项目启动时间等。

---

## 3. 推荐新增题型与优先级

| 优先级 | 建议 question_type | 解决什么缺口 | 说明 |
|---|---|---|---|
| P0 | `four_page_link_chain` | 4 页链接链为零 | 沿真实 `_PAGE_LINKS` 走 3 条边、4 个页面。 |
| P0 | `five_page_multi_branch` | 单链为主，缺少分叉聚合 | 从 Carol 出发分别走到两个项目的负责人，5 个页面。 |
| P0 | `three_page_reverse_chain` | 反向关系不足 | 从楼栋/办公室反查人物，再反查其团队或项目。 |
| P1 | `cross_page_comparison` | 只有求和，没有比较 | 预算差、楼层差、启动先后、目标日期先后。 |
| P1 | `reverse_relation` | 只有 mh10 | 经理的经理、团队包含谁、谁向谁汇报。 |
| P1 | `cross_page_negative` | 只有 mh09 | 同楼/不同楼、汇报关系否定、角色否定。 |
| P1 | `false_premise` | 没有 | 问题预设一个语料中不存在的“某项目由某人领导”。 |
| P1 | `unanswerable_attribute` | 只有实体缺失 | 薪资、学位、精确日期、人数、地名等属性缺失。 |
| P2 | `single_page_fact` | 单页事实覆盖偏窄 | 技术栈、例会日、学位、前雇主等未用事实。 |
| P2 | `two_page_relation` | 办公室/楼栋关系覆盖不足 | 人物 → 办公室 → 楼栋层数等。 |

不建议继续大量增加同构的 `two_page_relation` 或只替换人名的 `three_page_chain`；新增题应优先补上面的结构缺口。

---

## 4. 知识页面事实盘点

以下全部来自 9 个语料文件，未做外部推断。

| page_id | 实体 | 可用于出题的事实 | `links_to` |
|---|---|---|---|
| `alice` | Alice Smith | senior engineer；2021 年加入；领导 Project Phoenix；直接向 Bob Jones（Director of Engineering）汇报；办公室在 Building A 二层；计算机科学硕士；曾在分布式系统创业公司工作；擅长数据库内部与性能调优。 | `bob`, `project_phoenix` |
| `bob` | Bob Jones | Director of Engineering；2019 年起任；管理 12 名工程师，其中包括 Alice Smith 和 Carol Chen；办公室在 Building A 三层；向 Eve Martinez（VP of Engineering）汇报；此前是核心基础设施的 staff engineer；负责技术路线图和季度绩效评估。 | `alice`, `carol`, `building_a` |
| `carol` | Carol Chen | staff engineer；2022 年加入；参与 Project Phoenix 和 Project Orion；向 Bob Jones 汇报；办公室在 Building A 一层；负责认证服务、指导初级工程师；此前在金融科技公司做高吞吐支付系统。 | `bob`, `project_phoenix`, `project_orion` |
| `project_phoenix` | Project Phoenix | 内部平台迁移项目；2025 年初启动；由 Alice Smith 领导；Carol Chen 是核心成员；预算 50 万美元；目标完成日期 2026-12-31；单体应用迁移到 Kubernetes 微服务；团队每周二、周四开会。 | `alice`, `carol` |
| `project_orion` | Project Orion | 数据分析平台项目；2025 年中启动；由 David Kim（principal engineer）领导；Carol Chen 参与后端集成；预算 30 万美元；预计 2027 年 Q2 上线；实时商业智能仪表盘；技术栈含列式数据库、流处理引擎、React 前端；David 向 Eve Martinez 汇报。 | `david`, `carol` |
| `david` | David Kim | principal engineer；Project Orion 技术负责人；2020 年加入；数据基础设施；向 Eve Martinez 汇报；办公室在 Building B 四层；分布式系统博士；发表过流处理论文；此前在大型云厂商做托管数据库。 | `project_orion`, `eve` |
| `eve` | Eve Martinez | VP of Engineering；2018 年起任；直接向 CEO 汇报；管理两位总监：Bob Jones（platform engineering group）和 David Kim（data engineering group）；办公室在 Building B 五层；年度工程预算 400 万美元；负责长期技术愿景；此前在两家财富 500 强公司任高级工程管理。 | `bob`, `david`, `building_b` |
| `building_a` | Building A | Acme 两栋办公楼之一；容纳 platform engineering group；共 3 层；一层是员工和高级工程师开放工位，包括 Carol Chen 的工位；二层有额外办公室、小厨房、Alice Smith 的办公室；三层是 Director of Engineering 办公室（Bob Jones）和两间会议室；食堂在二层，紧邻小厨房。 | `bob` |
| `building_b` | Building B | Acme 第二栋办公楼；容纳行政和数据工程团队；共 5 层；一层有健身房、更衣室和淋浴；二、三层是数据工程团队工位；四层是 David Kim 的办公室和会议室；五层是行政层，含 VP of Engineering 办公室（Eve Martinez）、CEO 办公室和董事会会议室；Building B 与 Building A 通过二层的有盖走廊相连。 | `eve` |

### 4.1 语料内部冲突（出题前必须知道）

`eve` 页面把 David Kim 写成“两位总监之一，领导 data engineering group”；但 `david` 页面和 `project_orion` 页面都称 David 为 **principal engineer / technical lead**。这不是出题人可以自行消解的差异。

处理方式：

- 不要出依赖 David 头衔的题（例如“David 是总监吗”“Eve 手下的两位总监是谁”）。
- 不要用 `eve` 页面的“two directors”作为唯一证据设计标准答案。
- 如果确实想覆盖这个冲突，应单独设计成“语料一致性检测”实验，而不是普通多跳题；当前 15 题和评分规则没有这种类型。

---

## 5. 实际页面链接图

来自 `benchmark/run_multihop_evaluation.py` 的 `_PAGE_LINKS`（有向）：

```text
alice            -> bob, project_phoenix
bob              -> alice, carol, building_a
carol            -> bob, project_phoenix, project_orion
project_phoenix  -> alice, carol
project_orion    -> david, carol
david            -> project_orion, eve
eve              -> bob, david, building_b
building_a       -> bob
building_b       -> eve
```

注意：

- `links_from` 由构建逻辑反推，因此反向链接也可用；但 `_PAGE_LINKS` 本身是手动有向表。
- 链接图**不完整**：办公室位置、项目预算、日期、技术栈、团队人数、楼栋设施都不在链接里，只在页面内容里。
- `building_a` 只链接 `bob`，不链接 `alice` / `carol`；`building_b` 只链接 `eve`，不链接 `david`。因此“某办公室在几层”类问题即使涉及楼栋页，也不是纯链接跟随，需要搜索或读取。
- 真正的纯链接链（可用于 `link_chain` 题）：

| 链接链 | 页面数 |
|---|---:|
| `carol → project_orion → david → eve` | 4 |
| `carol → project_phoenix → alice → bob` | 4 |
| `david → project_orion → carol → bob → building_a` | 5 |
| `eve → bob → alice → project_phoenix` | 4 |
| `eve → david → project_orion → carol → bob` | 5 |
| `building_a → bob → alice → project_phoenix` | 4 |
| `building_b → eve → david → project_orion` | 4 |
| `alice → bob → building_a` | 3 |
| `carol → bob → building_a` | 3 |
| `project_orion → carol → bob → building_a` | 4 |
| `project_phoenix → alice → bob` | 3 |

---

## 6. 可扩写的真实关系链

| 关系链 | 页面 | 跳数 | 可出题方向 | 证据要点 |
|---|---|---:|---|---|
| Carol → Project Orion → David → Eve | `carol`, `project_orion`, `david`, `eve` | 4 | Orion 的负责人是谁、向谁汇报 | carol 参与 Orion；orion 由 David 领导；david 向 Eve 汇报 |
| Carol → Project Phoenix → Alice → Bob | `carol`, `project_phoenix`, `alice`, `bob` | 4 | Phoenix 的负责人是谁、向谁汇报 | carol 是 Phoenix 核心成员；phoenix 由 Alice 领导；alice 向 Bob 汇报 |
| David → Project Orion → Carol → Bob | `david`, `project_orion`, `carol`, `bob` | 4 | 与 David 共事的人向谁汇报 | orion 同时写 David 和 Carol；carol 向 Bob 汇报 |
| Building A 三层 → Bob → Eve | `building_a`, `bob`, `eve` | 3 | 三层办公室主人是哪个团队的负责人 | building_a 三层是 Director 办公室；eve 写 Bob 领导 platform engineering group |
| Building B 五层 → Eve → Bob/David | `building_b`, `eve` | 2–3 | 行政层 VP 管理谁 | building_b 五层是 Eve 办公室；eve 管理 Bob 和 David |
| Building B 四层 → David → Project Orion | `building_b`, `david`, `project_orion` | 3 | 四层办公室主人领导哪个项目 | building_b 四层是 David 办公室；david 领导 Orion |
| Building A 二层 → Alice → Project Phoenix | `building_a`, `alice`, `project_phoenix` | 3 | 二层办公室主人领导哪个项目 | building_a 二层是 Alice 办公室；alice 领导 Phoenix |
| Carol → Bob → Building A 三层 | `carol`, `bob`, `building_a` | 3 | Carol 的经理办公室在哪里 | carol 向 Bob 汇报；bob 办公室在 Building A 三层 |
| Alice → Bob → Eve | `alice`, `bob`, `eve` | 3 | Alice 的经理向谁汇报 | alice 向 Bob 汇报；bob 向 Eve 汇报 |
| Eve → Bob → Building A | `eve`, `bob`, `building_a` | 3 | platform engineering 负责人办公室在哪里 | eve 写 Bob 领导 platform group；bob 办公室在 Building A 三层 |
| Carol → 两个项目 → Alice/David | `carol`, `project_phoenix`, `alice`, `project_orion`, `david` | 5 | Carol 参与的两个项目分别由谁领导 | carol 页面列出两个项目；两个项目页分别写 Alice/David |
| Project Phoenix 与 Project Orion 的预算 | `project_phoenix`, `project_orion` | 2 | 差额、谁更大、较小项目由谁领导 | 50 万 vs 30 万；Orion 由 David 领导 |
| Building A 与 Building B 的层数 | `building_a`, `building_b` | 2 | 合计、差值、谁更高 | 3 层 vs 5 层 |
| Eve 的年度工程预算 vs 两个项目预算 | `eve`, `project_phoenix`, `project_orion` | 3 | 年度预算是否大于项目预算合计 | 400 万 vs 80 万 |
| 两个项目的启动/目标日期 | `project_phoenix`, `project_orion` | 2 | 谁先启动、谁先完成/上线 | Phoenix 2025 初、2026-12-31；Orion 2025 中、2027 Q2 |
| 人物办公室 → 楼栋层数 | `alice`/`carol`/`bob`/`david`/`eve` + `building_a`/`building_b` | 2–3 | 同楼判断、楼层差、楼栋层数 | 各人物页写办公室；楼栋页写层数与设施 |

---

## 7. 题目槽位

说明：

- “建议问题”是草稿，可改写；关键是“涉及页面”和“设计要点”中的事实必须成立。
- 标“二选一”的槽位是同一事实的顺向/反向版本，实际录入时只选一个。
- 槽位数量多于最低要求，目的是给人工筛选；**不要全部录入**。建议优先做 P0 的 8–10 题、P1 中选 8–10 题、P2 中选 4–6 题。

### 7.1 P0：三跳以上、链接链、跨页综合

| 槽位 | 建议 question_type | 跳数 | 涉及页面 | 建议问题（草稿） | 设计要点与证据 |
|---|---|---:|---|---|---|
| A01 | `four_page_link_chain` | 4 | `carol`, `project_orion`, `david`, `eve` | Carol Chen contributes to Project Orion. Who leads that project, and who does that person report to? | carol 页写参与 Orion；orion 页写 David 领导；david 页写向 Eve 汇报。答案 David Kim → Eve Martinez。 |
| A02 | `four_page_link_chain` | 4 | `carol`, `project_phoenix`, `alice`, `bob` | Carol Chen is a core member of Project Phoenix. Who leads that project, and who does that person report to? | phoenix 页写 Alice 领导、Carol 核心成员；alice 页写向 Bob 汇报。答案 Alice Smith → Bob Jones。 |
| A03 | `three_page_chain` | 3 | `project_orion`, `david`, `building_b` | Where is the office of the person who leads Project Orion? | orion 页写 David 领导；david 页写办公室在 Building B 四层；building_b 页再次确认四层是 David 办公室。 |
| A04 | `three_page_reverse_chain` | 3 | `building_b`, `david`, `project_orion` | Which project is led by the person whose office is on the fourth floor of Building B? | building_b 页写四层是 David 办公室；david 页写领导 Orion；orion 页确认。与 A03 同一事实的顺向/反向，**二选一**。 |
| A05 | `three_page_chain` | 3 | `eve`, `bob`, `building_a` | Where is the office of the director who leads the platform engineering group? | eve 页写 Bob 领导 platform engineering group；bob 页写办公室在 Building A 三层；building_a 页写三层是 Director 办公室。 |
| A06 | `three_page_chain` | 3 | `alice`, `bob`, `eve` | Who does the manager of Alice Smith report to? | alice 页写向 Bob 汇报；bob 页写向 Eve 汇报。答案 Eve Martinez。 |
| A07 | `three_page_chain` | 3 | `carol`, `bob`, `building_a` | Where is the office of the manager of Carol Chen? | carol 页写向 Bob 汇报；bob 页写办公室 Building A 三层；building_a 页确认三层是 Director 办公室。与 A05 同为 Bob 办公室题，**二选一**。 |
| A08 | `three_page_reverse_chain` | 3 | `building_a`, `bob`, `eve` | Which group is led by the Director of Engineering whose office is on the third floor of Building A? | building_a 页写三层是 Director 办公室；bob 页写自己是 Director of Engineering；eve 页写 Bob 领导 platform engineering group。答案 platform engineering group。 |
| A09 | `three_page_chain` | 3 | `project_orion`, `carol`, `bob` | Who does Carol Chen, a contributor to Project Orion, report to? | orion 页写 Carol 参与；carol 页写向 Bob 汇报。答案 Bob Jones。 |
| A10 | `three_page_chain` | 3 | `building_a`, `alice`, `project_phoenix` | Which project is led by the engineer whose office is on the second floor of Building A? | building_a 页写二层是 Alice 办公室；alice 页写领导 Phoenix；phoenix 页确认。答案 Project Phoenix。 |
| A11 | `five_page_multi_branch` | 5 | `carol`, `project_phoenix`, `alice`, `project_orion`, `david` | Carol Chen contributes to two projects. Who leads each of them? | carol 页列出 Phoenix 和 Orion；phoenix 页写 Alice 领导；orion 页写 David 领导。答案 Alice Smith（Phoenix）和 David Kim（Orion）。这是当前最缺的分叉聚合。 |
| A12 | `four_page_link_chain` | 4 | `project_orion`, `carol`, `bob`, `building_a` | Where is the office of the manager of the engineer who contributes to Project Orion? | orion 页写 Carol 参与；carol 页写向 Bob 汇报；bob 页写办公室 Building A 三层；building_a 页确认。答案 Building A 三层。 |

### 7.2 P1：比较、聚合、计算

| 槽位 | 建议 question_type | 跳数 | 涉及页面 | 建议问题（草稿） | 设计要点与证据 |
|---|---|---:|---|---|---|
| B01 | `cross_page_comparison` | 2 | `project_phoenix`, `project_orion` | Which named project has the larger budget, and by how much? | Phoenix 50 万，Orion 30 万；答案 Phoenix，多 20 万美元。关键词需覆盖 `200`/`two hundred`/`200000`。 |
| B02 | `cross_page_aggregation` | 2 | `building_a`, `building_b` | How many floors do Acme's two office buildings have in total? | A 3 层、B 5 层，合计 8 层。 |
| B03 | `cross_page_comparison` | 2 | `building_a`, `building_b` | Which office building has more floors, and how many more? | B 5 层、A 3 层，多 2 层。 |
| B04 | `cross_page_comparison` | 3 | `eve`, `project_phoenix`, `project_orion` | Is Acme's annual engineering budget larger than the combined budgets of its two named projects? | eve 页 400 万美元；两个项目 50 万 + 30 万 = 80 万；答案 yes，400 万 > 80 万。 |
| B05 | `cross_page_comparison` | 2 | `project_phoenix`, `project_orion` | Which named project was launched first? | Phoenix 2025 年初，Orion 2025 年中；答案 Phoenix。注意两页分别用 launched/initiated。 |
| B06 | `cross_page_comparison` | 2 | `project_phoenix`, `project_orion` | Which named project has the earlier target completion or launch date? | Phoenix 目标完成 2026-12-31；Orion 预计 2027 Q2 上线；答案 Phoenix。题干必须保留“completion or launch”，不要混用 deadline。 |
| B07 | `cross_page_comparison` | 3 | `project_phoenix`, `project_orion`, `david` | Which named project has the smaller budget, and who leads it? | Orion 30 万 < Phoenix 50 万；Orion 由 David 领导。答案 David Kim / Project Orion。 |
| B08 | `cross_page_aggregation` | 2 | `carol`, `bob` | How many floors apart are Carol Chen's and Bob Jones's offices? | Carol 一层、Bob 三层，楼层差为 2。题干用 “difference in floor numbers” 或 “floors apart”，并在标准答案中说明是 2 层差，避免“中间隔几层”的歧义。 |

### 7.3 P1：反向关系

| 槽位 | 建议 question_type | 跳数 | 涉及页面 | 建议问题（草稿） | 设计要点与证据 |
|---|---|---:|---|---|---|
| C01 | `reverse_relation` | 2 | `alice`, `bob` | Which director's team includes the person who leads Project Phoenix? | Alice 领导 Phoenix、向 Bob 汇报；Bob 是 Director of Engineering。答案 Bob Jones 的团队。与 A02 同链条，但问的是反向归属，可作为反向题保留。 |
| C02 | `reverse_relation` | 1 | `bob` | Which team includes both Alice Smith and Carol Chen? | bob 页写管理 12 名工程师，其中包括 Alice 和 Carol。答案 Bob Jones 的团队。单页可答，优先级低，且与 mh10 有重叠。 |
| C03 | `reverse_relation` | 1 | `eve` | Which two groups are led by the people who report to Eve Martinez? | eve 页写 Bob 领导 platform engineering group、David 领导 data engineering group。单页可答，优先级低；不要问“两位总监”的头衔，避免触发 4.1 的冲突。 |

### 7.4 P1：跨页否定

| 槽位 | 建议 question_type | 跳数 | 涉及页面 | 建议问题（草稿） | 设计要点与证据 |
|---|---|---:|---|---|---|
| D01 | `cross_page_negative` | 2 | `alice`, `david` | Do Alice Smith and David Kim work in the same building? | Alice 办公室 Building A 二层；David 办公室 Building B 四层；答案 No。 |
| D02 | `cross_page_negative` | 2 | `carol`, `eve` | Do Carol Chen and Eve Martinez work in the same building? | Carol Building A 一层；Eve Building B 五层；答案 No。 |
| D03 | `cross_page_negative` | 3 | `bob`, `david`, `eve` | Does Bob Jones report to David Kim? | Bob 向 Eve 汇报；David 也向 Eve 汇报；答案 No，Bob 不向 David 汇报。 |
| D04 | `cross_page_negative` | 3 | `building_b`, `david`, `eve` | Do David Kim and Eve Martinez have offices on the same floor of Building B? | Building B 页写四层是 David 办公室、五层是 Eve 办公室；david 和 eve 页分别确认。答案 No，David 四层、Eve 五层。 |
| D05 | `cross_page_negative` | 2 | `david`, `building_b` | Does David Kim work in Building A? | David 页写办公室在 Building B 四层；building_b 页确认；答案 No。 |
| D06 | `cross_page_negative` | 2 | `bob`, `eve` | Does Eve Martinez report to Bob Jones? | Bob 页写向 Eve 汇报；eve 页写 Bob 是她的下属总监之一；答案 No，方向相反。 |
| D07 | `cross_page_negative` | 2 | `carol`, `project_phoenix` | Does Carol Chen lead Project Phoenix? | phoenix 页写 Alice 领导、Carol 是核心成员；carol 页写参与两个项目；答案 No，Alice 领导。 |

食堂否定变体：`Is the cafeteria in Building B?` 只能由“Building A 页写食堂在二层 + Building B 楼层清单未列食堂”推出，依赖封闭世界假设；如果要用，标为低置信度，优先选 D01–D07 中证据更直接的题。

### 7.5 P1：不可回答与虚假前提

建议从 E01–E09 中选 4–6 题，不要全部录入；不可回答题过多会扭曲整体指标。

| 槽位 | 建议 question_type | 跳数 | 涉及页面 | 建议问题（草稿） | 设计要点与证据 |
|---|---|---:|---|---|---|
| E01 | `unanswerable_attribute` | 0 | — | What is Alice Smith's salary? | 9 个页面都没有薪资信息。`required_page_ids: []`，`hops: 0`，`refusal_keywords` 必填。 |
| E02 | `unanswerable_attribute` | 0 | — | What is Carol Chen's highest academic degree? | Carol 页没有学位；语料只给出 Alice 的硕士和 David 的博士。 |
| E03 | `unanswerable_attribute` | 0 | — | What is the exact completion date of Project Orion? | Orion 页只写“expected to launch in Q2 2027”，没有精确完成日期。标准答案要明确区分“预计上线季度”与“精确完成日期”。 |
| E04 | `unanswerable_attribute` | 0 | — | How many employees work at Acme Corp in total? | Bob 页只写管理 12 名工程师，没有公司总人数。 |
| E05 | `unanswerable_attribute` | 0 | — | How many people report directly to David Kim? | eve 页写 David 领导 data engineering group，但没有人数。注意 4.1 的头衔冲突；如果担心歧义，优先选别的题。 |
| E06 | `unanswerable_attribute` | 0 | — | What is Project Phoenix's budget in euros? | 语料只有美元金额，没有汇率或欧元金额。 |
| E07 | `unanswerable_attribute` | 0 | — | Which university did David Kim attend? | David 页只写分布式系统博士，没有学校名。 |
| E08 | `unanswerable_attribute` | 0 | — | What is the name of the startup where Alice Smith previously worked? | Alice 页只写“一家专注分布式系统的创业公司”，没有名字。 |
| E09 | `false_premise` | 0 | — | Which project is led by the person whose office is on the third floor of Building A? | Building A 三层是 Director of Engineering（Bob）办公室；Bob 是总监，语料中没有他领导某个项目。答案应为“没有这样的项目 / Bob 不领导项目”。这是虚假前提，不是普通属性缺失。 |

### 7.6 P2：单页/双页补充事实

这些题价值较低，只用于补单页事实覆盖；不要挤占 P0/P1 的篇幅。

| 槽位 | 建议 question_type | 跳数 | 涉及页面 | 建议问题（草稿） | 设计要点与证据 |
|---|---|---:|---|---|---|
| F01 | `single_page_fact` | 1 | `project_phoenix` | Which container orchestration platform does Project Phoenix use? | phoenix 页写迁移到运行在 Kubernetes 上的微服务架构。答案 Kubernetes。 |
| F02 | `single_page_fact` | 1 | `project_orion` | Which project uses a columnar database, a stream processing engine, and a React frontend? | orion 页的技术栈原文。答案 Project Orion。 |
| F03 | `single_page_fact` | 1 | `building_b` | Which groups are housed in Building B? | building_b 页写容纳 executive and data engineering groups。 |
| F04 | `single_page_fact` | 1 | `project_phoenix` | On which days does the Project Phoenix team meet? | phoenix 页写每周二、周四。答案 Tuesday and Thursday。 |
| F05 | `single_page_fact` | 1 | `alice` | What is Alice Smith's highest academic degree? | alice 页写计算机科学硕士。答案 master's degree。 |
| F06 | `single_page_fact` | 1 | `bob` | What was Bob Jones's previous role before becoming Director of Engineering? | bob 页写此前是 core infrastructure 的 staff engineer。答案 staff engineer。 |

---

## 8. 空白题目模板

### 8.1 可回答题

```json
{
  "id": "mhXX_short_slug",
  "question": "问题原文",
  "reference_answer": "人工标准答案，包含所有必要事实。",
  "answer_keyword_groups": [
    ["关键词1", "同义表达1"],
    ["关键词2", "同义表达2"]
  ],
  "answerable": true,
  "required_page_ids": ["page_id_1", "page_id_2"],
  "hops": 3,
  "question_type": "three_page_chain"
}
```

填写说明：

- `answer_keyword_groups` 的每个内层组必须至少命中一个；组与组之间是 AND。
- 数字答案同时给数字和英文拼写，例如 `["200", "two hundred", "200000"]`。
- 金额答案同时给带单位和纯数字形式；评分会去掉标点，`"$200,000"` 会归一化为 `200000`。
- 避免 `"a"` / `"b"` 这类单字母关键词；楼栋用 `"building a"` / `"building b"`。
- `required_page_ids` 只放**回答确实需要**的页面，不要为了拉高跳数而填充；填充会拉低 evidence recall，也会让题目验收失真。
- `hops` 与页面数/关系步数保持一致；`hops >= 2` 会计入多跳指标。

### 8.2 不可回答题

```json
{
  "id": "mhXX_unanswerable_slug",
  "question": "问题原文",
  "reference_answer": "语料没有给出该信息；标准答案要说明缺的是什么，并指出语料中相近但不充分的信息。",
  "answer_keyword_groups": [],
  "refusal_keywords": [
    "not stated", "no information", "cannot", "unknown", "not provided"
  ],
  "answerable": false,
  "required_page_ids": [],
  "hops": 0,
  "question_type": "unanswerable"
}
```

填写说明：

- `refusal_keywords` 必填，否则该题在现有脚本中无法得分。
- `required_page_ids` 按现有约定填 `[]`；即使题目需要读取某些页面才能确认信息缺失，也不要把它们塞进 `required_page_ids`，否则会改变 recall 口径。若未来要考核“定位过程”，应作为新的指标设计，而不是在当前数据集里硬塞。
- 虚假前提题的标准答案要直接否定前提，不要只写“不知道”。

### 8.3 一个完整示例（A01）

```json
{
  "id": "mh16_carol_orion_lead_chain",
  "question": "Carol Chen contributes to Project Orion. Who leads that project, and who does that person report to?",
  "reference_answer": "David Kim leads Project Orion, and he reports to Eve Martinez, the VP of Engineering.",
  "answer_keyword_groups": [
    ["david"],
    ["eve"]
  ],
  "answerable": true,
  "required_page_ids": ["carol", "project_orion", "david", "eve"],
  "hops": 4,
  "question_type": "four_page_link_chain"
}
```

注意：这个示例的 `answer_keyword_groups` 只要求 David 和 Eve，不要求 VP 头衔；如果希望答案必须带标题，可增加 `["vp", "vice president"]` 组，但会增加误判风险。

---

## 9. 题目验收清单

录入前逐条打勾；任何一条不通过，先改题再录。

### 9.1 事实与证据

- [ ] 标准答案中的每个事实都能在 `benchmark/corpus/wiki/` 的原文中找到，且已记录证据句。
- [ ] `required_page_ids` 中的每个 ID 都存在于 `_PAGE_LINKS`，没有拼写错误。
- [ ] `required_page_ids` 覆盖了回答所需的全部页面，没有遗漏。
- [ ] `required_page_ids` 可以包含确认性页面（现有 mh03 的 `eve` 页就是确认性证据），但不能加入与问题无关、也没有确认作用的页面。
- [ ] 如果某一页单独就能完整回答，且其他页面既不补充也不确认，就不要硬凑跳数；应改成真正需要跨页的问题。
- [ ] `hops` 与真实关系步数一致；`hops >= 2` 的题确实需要跨页。
- [ ] 没有依赖 4.1 中 David 头衔冲突的表述（“director”“两位总监”等）。

### 9.2 可验证性与评分

- [ ] `answer_keyword_groups` 能让标准答案通过；用脚本的归一化规则（小写、去标点）手工验算过。
- [ ] 关键词不会让明显错误的答案通过；特别是数字题、楼栋题、楼层题。
- [ ] 没有单字母关键词；没有过宽的关键词（如单独的 `"engineering"` 用来区分两个工程团队）。
- [ ] 不可回答题有非空 `refusal_keywords`，并且标准拒绝语能命中。
- [ ] 不可回答题的 `answer_keyword_groups` 为 `[]`，`required_page_ids` 为 `[]`，`hops` 为 `0`（按现有约定）。

### 9.3 去重与结构

- [ ] 没有与现有 15 题重复或只替换人名/项目名。
- [ ] 与同批新增题不重复；A03/A04、A05/A07、B05/B06 这类同事实顺反向或近似题，实际只选一个。
- [ ] 新题优先补 4 页链、5 页分叉、反向关系、比较、否定、虚假前提，而不是再加同构的两页题。
- [ ] `question_type` 使用稳定词汇；新增类型在本题表或数据集说明中有定义。

### 9.4 否定题与不可回答题的真实性

- [ ] 否定题的“No”有正面证据，而不是只靠“某页没写”。例如 D01 有 Alice 和 David 两个办公室地址，D04 有 David 和 Eve 的楼层地址。
- [ ] 不可回答题的信息在**全部 9 个页面**都不存在，不是只缺在某一个页面。
- [ ] 不可回答题的标准答案说明了“缺什么”，并区分“语料有相近信息但不充分”的情况（如 E03 的 Q2 2027 预计上线）。
- [ ] 虚假前提题确实预设了一个语料中不存在的实体/关系，而不是问一个可用常识补全的问题。

### 9.5 录入后验证

- [ ] JSON 可被 `load_dataset()` 通过：`id` 唯一，必需字段齐全，`required_page_ids` 是列表。
- [ ] 用离线模式跑一次（不调用模型），确认脚本能加载数据、构建 WikiIndex、完成 baseline，不出现 schema 或链接错误。示例命令（输出路径必须是新文件）：

```powershell
python benchmark/run_multihop_evaluation.py --output benchmark/results/<new_result>.json
```

- [ ] 如果题目针对 `--auto-links` 的链接解析，分别用默认 manual 和 `--auto-links` 各跑一次；不要把两种 `link_source` 的结果直接混在一起比较。
- [ ] 新增一定数量后，再决定是否更新数据集的 `version` / `description`；不要在不说明的情况下改变历史口径。

---

## 10. 建议的录入顺序

1. **先做 8–10 道 P0**：A01、A02、A03、A05、A06、A08、A10、A11、A12，以及 A04/A07 中二选一。
2. **再做 6–8 道 P1**：B01、B02、B03、B04、B07、C01、D01、D03。
3. **补 3–5 道不可回答/虚假前提**：E01、E03、E04、E07、E09 中选，保持不可回答题占比不要过高。
4. **最后补 3–5 道 P2 单页事实**：F01、F02、F03、F04、F05 中选。
5. 每批录入后先跑离线校验，再决定是否消耗真实模型额度。

---

## 人工录题区（首批 15 题）

本区记录首批 15 题的草稿与设计说明。mh17、mh18、mh20、mh21、mh23、mh24、mh25、mh26、mh28、mh29、mh30 已按验收报告整理为英文并录入 `multihop_wiki.json`（正式集 26 题）；mh16、mh19、mh22、mh27 仍为草稿。

填写约定：

- 新题 ID 从 `mh16` 开始，使用小写 snake_case，录入前确认与现有 `mh01`–`mh15` 不重复。
- `页面数 / 关系边数 / 属性查询 / 推理步骤数` 分开记录；属性查询单独计数，不并入关系边；当前数据集的 `hops` 实际等于页面数，不要把页面数直接写成“跳数”。
- 题型名是 `question_type` 的自由字符串；新增题型名（如 `four_page_link_chain`、`reverse_relation`、`cross_page_comparison`）与现有字段结构兼容。
- 可回答题使用 `answer_keyword_groups`；不可回答/错误前提题使用 `refusal_keywords`，并按现有约定填 `required_page_ids: []`、`hops: 0`。
- 题目状态按 `待编写 → 已录入` 推进；已录入题的正式英文版、必要页和关键词以 `multihop_wiki.json` 为准。

### 首批录题清单

| 新题 ID | 槽位 | 题型 | 页面数 / 关系边数 / 推理步骤数 | 状态 |
|---|---|---|---:|---|
| `mh16_carol_orion_lead_chain` | A01 | `four_page_link_chain` | 4 / 3 + 1 属性查询 / 4 | 待编写 |
| `mh17_carol_two_project_leads` | A11 | `five_page_multi_branch` | 5 / 4 + 2 次属性查询 / 6（扩展后） | 已录入 |
| `mh18_orion_contributor_manager_office` | A12 | `four_page_link_chain` | 4 / 3 + 1 次属性查询 / 4（扩展后） | 已录入 |
| `mh19_building_a_director_group` | A08 | `reverse_relation` | 2 / 2 / 2 | 待编写 |
| `mh20_phoenix_deadline_office` | A10 | `two_page_relation` | 2 / 1 + 2 属性 / 3 | 已录入 |
| `mh21_team_intersection` | C02 | `three_page_intersection` | 3 / 2 / 3 | 已录入 |
| `mh22_project_budget_difference` | B01 | `cross_page_comparison` | 2 / 0 / 3 | 待编写 |
| `mh23_building_floor_difference` | B03 | `cross_page_comparison` | 2 / 0 / 3 | 已录入 |
| `mh24_engineering_budget_vs_projects` | B04 | `cross_page_aggregation` | 3 / 0 / 4 | 已录入 |
| `mh25_project_launch_order` | B05 | `cross_page_comparison` | 2 / 0 / 3 | 已录入 |
| `mh26_alice_david_same_building` | D01 | `two_page_negative` | 2 / 0 / 3 | 已录入 |
| `mh27_carol_phoenix_lead` | D07 | `two_page_negative` | 2 / 2 / 2 | 待编写 |
| `mh28_first_floor_office_premise` | E09 | `false_premise` | 0（1 页验证）/ 0 / 1 | 已录入 |
| `mh29_alice_salary` | E01 | `unanswerable` | 0 / 0 / 1 | 已录入 |
| `mh30_david_university` | E07 | `unanswerable` | 0 / 0 / 1 | 已录入 |

> 正式录入说明：mh17、mh18、mh21 按验收报告采用“必要页”口径，正式 `required_page_ids` 分别为 3/3/2 页，题型相应改为 `three_page_multi_branch`、`three_page_chain`、`two_page_intersection`；本区统计表保留原设计稿数值。mh28/mh29/mh30 的拒答关键词已按现有格式补齐；M4.3.1 起评分器增加 `fabrication_keywords` 词法拦截“拒答后继续编造”，但它仍是启发式，不可回答题正式评测仍需人工复核。

---

### mh16 · A01 · 四页链 · 难度：高

- **新题 ID**：`mh16_carol_orion_lead_chain`
- **槽位 ID**：A01
- **题型**：`four_page_link_chain`
- **难度**：高
- **涉及页面**：4 页（`carol`、`project_orion`、`david`、`eve`）
- **页面关系边**：3 条（Carol → Project Orion → David → Eve；最后一条汇报关系以页面事实为依据，不要求存在显式 Wiki 链接）
- **属性查询**：1 次（Eve Martinez 的办公室）
- **推理步骤数**：4（3 条关系边 + 1 次属性查询）
- **说明**：以上是题目设计中的关系链和属性查询统计，不等于 Agent 实际执行时必然采用相同的搜索、读取或推理步骤。
- **必要证据页面 ID**：`carol`、`project_orion`、`david`、`eve`
- **题目状态**：待编写（待编写 / 待校验 / 已通过）

**预期答案要点**

- Carol Chen 参与 Project Orion。
- Project Orion 由 David Kim 领导。
- David Kim 向 Eve Martinez（VP of Engineering）汇报。
- 若采用扩展问法，还需覆盖 Eve Martinez 的办公室在 Building B 五层。

**评分规则设计提示**

- 至少需要 `david` 和 `eve` 两个关键词组；若问办公室，再加 `fifth`/`5` 与 `"building b"` 短语组。
- 不要用单字母 `"b"`；不要只要求 `"engineering"`（data engineering 也会命中）。
- 若不做扩展，把题型改为 `three_page_chain`，页面/边/步骤均改为 3，并从必要页面中删掉 `eve`。

**问题（待我填写）**

> Carol 参与的 Project Orion 由谁负责？该负责人直接向谁汇报？该汇报对象的办公室位于哪栋楼、几层？

**标准答案（待我填写）**

> Project Orion 由 David Kim 负责。David 直接向 Eve Martinez 汇报。Eve 的办公室位于 Building B 五层。

---

### mh17 · A11 · 五页多分支 · 难度：高

- **新题 ID**：`mh17_carol_two_project_leads`
- **槽位 ID**：A11
- **题型**：`five_page_multi_branch`
- **难度**：高
- **设计目标**：以 Carol 的两个项目为分叉，要求两位负责人及办公室；重点是多分支和办公室对比，不重复 mh16 的 David→Eve 汇报链。
- **涉及页面**：5 页（`carol`、`project_phoenix`、`alice`、`project_orion`、`david`）
- **页面关系边**：4 条（Carol→Phoenix→Alice；Carol→Orion→David）
- **属性查询**：2 次（Alice 的办公室、David 的办公室）
- **推理步骤数**：6（识别两个项目、两位负责人、两处办公室）
- **必要证据页面 ID**：`carol`、`project_phoenix`、`alice`、`project_orion`、`david`
- **题目状态**：已录入（正式数据集 26 题版本）

**预期答案要点**

- Carol Chen 参与 Project Phoenix 和 Project Orion。
- Project Phoenix 由 Alice Smith 领导；Alice 的办公室在 Building A 二层。
- Project Orion 由 David Kim 领导；David 的办公室在 Building B 四层。
- 只要求负责人时，`alice` 和 `david` 页是确认页；若保留 5 页，必须把“办公室”纳入评分要点。

**答案判定与评分注意事项**

- 人名分别用 `alice`、`david`；办公室用 `"second"/"2"`、`"fourth"/"4"` 与 `"building a"/"building b"` 短语。
- 不要把 `"project"` 或 `"engineering"` 单独作为判定词；不要让只答一个项目就通过。
- 若只问负责人，页面数改为 3（`carol`、`project_phoenix`、`project_orion`），`hops` 改为 2（分支深度）或 4（边数），并在备注中说明口径。

**问题（待我填写）**

> Carol Chen 参与的两个项目分别由谁领导？这两位负责人的办公室分别位于哪栋楼、几层？

**标准答案（待我填写）**

> Project Phoenix 由 Alice Smith 领导，她的办公室在 Building A 二层；Project Orion 由 David Kim 领导，他的办公室在 Building B 四层。

---

### mh18 · A12 · 四页链 · 难度：高

- **新题 ID**：`mh18_orion_contributor_manager_office`
- **槽位 ID**：A12
- **题型**：`four_page_link_chain`
- **难度**：高
- **设计目标**：从 Orion 贡献者 Carol 出发，走到其经理 Bob 的办公室与 Building A 楼层事实；与 mh16 的 David→Eve 分支区分。
- **涉及页面**：4 页（`project_orion`、`carol`、`bob`、`building_a`）
- **页面关系边**：3 条（Project Orion→Carol→Bob→Building A；Bob 的办公室位置是页面事实）
- **属性查询**：1 次（Building A 的楼层数或三层设施）
- **推理步骤数**：4（识别贡献者、经理、办公室、楼层事实）
- **必要证据页面 ID**：`project_orion`、`carol`、`bob`、`building_a`
- **题目状态**：已录入（正式数据集 26 题版本）

**预期答案要点**

- Carol Chen 是 Project Orion 的贡献者。
- Carol Chen 向 Bob Jones 汇报。
- Bob Jones 的办公室在 Building A 三层。
- 若采用扩展问法，还需覆盖 Building A 共三层，或三层另有会议室等设施。

**答案判定与评分注意事项**

- 使用 `"building a"` 短语和 `"third"/"3"`；若问楼层数，再加 `"three"/"3"` 组。
- 不要用 `"floor"` 单独判定；不要把 `"bob"` 与题干中的其他人物混淆。
- `building_a` 在原题中只是确认页；若不做扩展，删掉它并把页面/边/步骤改为 3/3/3。

**问题（待我填写）**

> Carol Chen 参与的 Project Orion 中，她向谁汇报？该经理的办公室位于 Building A 的哪一层？Building A 共有几层？

**标准答案（待我填写）**

> Carol Chen 向 Bob Jones 汇报。Bob Jones 的办公室位于 Building A 三层；Building A 共有三层。

---

### mh19 · A08 · 反向团队 · 难度：中

- **新题 ID**：`mh19_building_a_director_group`
- **槽位 ID**：A08
- **题型**：`reverse_relation`
- **难度**：中
- **页面数**：2
- **关系边数**：2
- **推理步骤数**：2
- **必要证据页面 ID**：`building_a`、`eve`
- **可选确认页**：`bob`（不列入必要页面）
- **题目状态**：待编写（待编写 / 待校验 / 已通过）

**预期答案要点**

- Building A 三层是 Director of Engineering 的办公室，当前占用者是 Bob Jones。
- Bob Jones 领导 platform engineering group。

**评分规则设计提示**

- 用 `"platform engineering"` 短语；不要用 `"engineering"` 单独判定（data engineering 也会命中）。
- 若答案要求人物，加 `"bob"`；若只问团队，不要让 `"bob"` 成为唯一得分点。
- 原槽位标注 3 页；审查后 `bob` 页只是确认页，建议删掉并把 `hops` 改为 2。

**问题（待我填写）**

> Building A 三层办公室的使用者领导哪个团队？

**标准答案（待我填写）**

> 办公室使用者是 Bob Jones，他领导 platform engineering group。

---

### mh20 · A10 · 截止日期→负责人办公室 · 难度：中

- **新题 ID**：`mh20_phoenix_deadline_office`
- **槽位 ID**：A10
- **题型**：`two_page_relation`
- **难度**：中
- **设计目标**：以 Project Phoenix 的目标完成日期为入口，要求项目、负责人及负责人办公室；避免重复 mh19 的“办公室→人物→角色”路径，也不复用 mh22 的预算差值。
- **涉及页面**：2 页（`project_phoenix`、`alice`）
- **页面关系边**：1 条（Project Phoenix→Alice）
- **属性查询**：2 次（目标完成日期、Alice 的办公室）
- **推理步骤数**：3（读取日期、确认项目与负责人、读取办公室）
- **必要证据页面 ID**：`project_phoenix`、`alice`
- **可选确认页**：`building_a`（不列入必要页面）
- **题目状态**：已录入（正式数据集 26 题版本）

**预期答案要点**

- Project Phoenix 的目标完成日期是 2026-12-31。
- Project Phoenix 由 Alice Smith 领导。
- Alice Smith 的办公室在 Building A 二层。

**答案判定与评分注意事项**

- 关键词至少覆盖 `"phoenix"`、`"alice"`、`"second"/"2"`、`"building a"` 四组。
- 不要用 `"project"` 单独判定；不要用单字母 `"a"` 或 `"b"`。
- 答案必须同时包含项目、负责人和办公室；只答项目或只答负责人应判错。
- 页面数 2、关系边 1、属性查询 2、推理步骤 3；不要把 2 页写成“2 跳”。

**问题（待我填写）**

> Project Phoenix 的目标完成日期是什么？该项目由谁领导，该负责人的办公室在哪栋楼、几层？

**标准答案（待我填写）**

> Project Phoenix 的目标完成日期是 2026 年 12 月 31 日；项目由 Alice Smith 领导，她的办公室在 Building A 二层。

---

### mh21 · C02 · 三页交集 · 难度：中

- **新题 ID**：`mh21_team_intersection`
- **槽位 ID**：C02
- **题型**：`three_page_intersection`
- **难度**：中
- **设计目标**：把 Phoenix 负责人 Alice 与 Orion 贡献者 Carol 归到同一总监团队；重点是团队交集，区别于 mh19 的单人团队题和 mh27 的 Carol 否定题。
- **涉及页面**：3 页（`alice`、`carol`、`bob`）
- **页面关系边**：2 条（Alice→Bob；Carol→Bob）
- **属性查询**：0 次
- **推理步骤数**：3（识别 Alice、识别 Carol、汇合到 Bob）
- **必要证据页面 ID**：`alice`、`carol`、`bob`
- **可选确认页**：`project_phoenix`、`project_orion`（不列入必要页面）
- **题目状态**：已录入（正式数据集 26 题版本）

**预期答案要点**

- Alice Smith 领导 Project Phoenix，并向 Bob Jones 汇报。
- Carol Chen 参与 Project Orion，并向 Bob Jones 汇报。
- 两人同属 Bob Jones 的团队；Bob Jones 是 Director of Engineering。

**答案判定与评分注意事项**

- 必须命中 `"bob"`；可选 `"director"`；不要让错误答案 “Alice” 或 “Carol” 命中。
- 不要用 `"team"` 单独判定；若问题要求“总监”，可加 `"director"` 组。
- 原槽位是单页 `bob` 题（1 页/0 边/1 步），没有多跳价值；本录题区按重设计的 3 页交集题记录，重设计后的题干由你填写。

**问题（待我填写）**

> Project Phoenix 的负责人和 Project Orion 的一位贡献者，共同向哪位总监汇报？

**标准答案（待我填写）**

> Project Phoenix 的负责人 Alice Smith 和 Project Orion 的贡献者 Carol Chen 都向 Bob Jones 汇报；Bob Jones 是 Director of Engineering。

---

### mh22 · B01 · 预算比较 · 难度：中

- **新题 ID**：`mh22_project_budget_difference`
- **槽位 ID**：B01
- **题型**：`cross_page_comparison`
- **难度**：中
- **页面数**：2
- **关系边数**：0
- **推理步骤数**：3（读取两个预算、比较、相减）
- **必要证据页面 ID**：`project_phoenix`、`project_orion`
- **题目状态**：待编写（待编写 / 待校验 / 已通过）

**预期答案要点**

- Project Phoenix 预算 500,000 美元。
- Project Orion 预算 300,000 美元。
- Phoenix 比 Orion 多 200,000 美元。

**评分规则设计提示**

- 用 `"phoenix"` + `["200", "two hundred", "200000"]` 两组；不要让 `"500"`/`"300"` 满足差值组。
- 错误答案 “Orion” 必须失败；如果答案同时列出两个预算，差值组仍必须出现。
- 页面数 2、关系边 0、推理步骤 3；`hops` 按现有页面口径填 2，但不要称为“2 跳图遍历”。

**问题（待我填写）**

> Project Phoenix 和 Project Orion 哪个项目的预算更高？两者相差多少？

**标准答案（待我填写）**

> Project Phoenix 的预算更高，比 Project Orion 多 200,000 美元。

---

### mh23 · B03 · 楼层比较 · 难度：低

- **新题 ID**：`mh23_building_floor_difference`
- **槽位 ID**：B03
- **题型**：`cross_page_comparison`
- **难度**：低
- **设计目标**：比较 Building A 与 Building B 的总层数；区别于 mh19 的“办公室所在楼层”和 mh22 的预算比较。
- **涉及页面**：2 页（`building_a`、`building_b`）
- **页面关系边**：0 条
- **属性查询**：2 次（两栋楼的层数）
- **推理步骤数**：3（读取两个层数、比较、相减）
- **必要证据页面 ID**：`building_a`、`building_b`
- **题目状态**：已录入（正式数据集 26 题版本）

**预期答案要点**

- Building A 有三层。
- Building B 有五层。
- Building B 比 Building A 多两层。

**答案判定与评分注意事项**

- 用 `"building b"` 短语 + `["2", "two"]`；不要用单字母 `"b"`。
- 不要让 `"floor"` 单独满足；错误答案 “Building A” 必须失败。
- 页面数 2、关系边 0、步骤 3；`hops` 按页面口径填 2。

**问题（待我填写）**

> Building A 和 Building B 哪栋楼层数更多？多几层？

**标准答案（待我填写）**

> Building B 有五层，Building A 有三层；Building B 比 Building A 多两层。

---

### mh24 · B04 · 预算聚合比较 · 难度：中

- **新题 ID**：`mh24_engineering_budget_vs_projects`
- **槽位 ID**：B04
- **题型**：`cross_page_aggregation`
- **难度**：中
- **设计目标**：比较 Acme 年度工程预算金额与两个项目预算合计；区别于 mh22 的两个项目预算差值，重点在跨页求和与不同预算口径。
- **涉及页面**：3 页（`eve`、`project_phoenix`、`project_orion`）
- **页面关系边**：0 条
- **属性查询**：3 次（年度工程预算、两个项目预算）
- **推理步骤数**：4（读取三项预算、求和、比较）
- **必要证据页面 ID**：`eve`、`project_phoenix`、`project_orion`
- **题目状态**：已录入（正式数据集 26 题版本）

**预期答案要点**

- Acme 的年度工程预算为 4,000,000 美元，由 Eve Martinez 负责。
- Project Phoenix 预算 500,000 美元；Project Orion 预算 300,000 美元。
- 两个项目合计 800,000 美元。
- 年度工程预算金额大于两个项目预算合计。
- 标准答案应说明这是“年度工程预算”与“项目预算合计”的不同口径比较。

**答案判定与评分注意事项**

- 用 `["yes"]`、`["4000000", "four million"]`、`["800000", "eight hundred thousand"]` 组；不要用裸 `"4"`/`"800"` 或 `"million"` 单独判定。
- 页面数 3、关系边 0、步骤 4；`hops` 按页面口径填 3，不要写成“3 跳”。

**问题（待我填写）**

> Acme 的年度工程预算金额是否大于 Project Phoenix 与 Project Orion 的预算合计？

**标准答案（待我填写）**

> 是。Acme 的年度工程预算为 4,000,000 美元，由 Eve Martinez 负责；Project Phoenix 预算 500,000 美元，Project Orion 预算 300,000 美元，两者合计 800,000 美元。年度工程预算金额大于两个项目预算合计。

---

### mh25 · B05 · 启动时间比较 · 难度：低

- **新题 ID**：`mh25_project_launch_order`
- **槽位 ID**：B05
- **题型**：`cross_page_comparison`
- **难度**：低
- **设计目标**：比较两个项目的启动/发起时间；区别于 mh22、mh24 的预算比较和 mh23 的楼层比较，重点在时间先后。
- **涉及页面**：2 页（`project_phoenix`、`project_orion`）
- **页面关系边**：0 条
- **属性查询**：2 次（两个项目的启动/发起时间）
- **推理步骤数**：3（读取两个时间、比较）
- **必要证据页面 ID**：`project_phoenix`、`project_orion`
- **题目状态**：已录入（正式数据集 26 题版本）

**预期答案要点**

- Project Phoenix 于 2025 年初启动。
- Project Orion 于 2025 年中启动/发起。
- 因此 Project Phoenix 更早。

**答案判定与评分注意事项**

- 用 `"phoenix"`；不要用 `"2025"` 单独判定（两页都有 2025）。
- 注意原文分别使用 launched/initiated；标准答案按“启动/发起”解释，不要求两个页面使用同一个英文词。
- 页面数 2、关系边 0、步骤 3；`hops` 按页面口径填 2。

**问题（待我填写）**

> Project Phoenix 和 Project Orion 哪个项目更早启动/发起？

**标准答案（待我填写）**

> Project Phoenix 更早。Project Phoenix 于 2025 年初启动，Project Orion 于 2025 年中发起。

---

### mh26 · D01 · 同楼否定 · 难度：中

- **新题 ID**：`mh26_alice_david_same_building`
- **槽位 ID**：D01
- **题型**：`two_page_negative`
- **难度**：中
- **设计目标**：用 Alice 与 David 的办公室位置做同楼否定；区别于 mh19 的 Building A 办公室→团队路径，也避免复用 mh27 的 Carol 角色否定。
- **涉及页面**：2 页（`alice`、`david`）
- **页面关系边**：0 条
- **属性查询**：2 次（两人的办公室位置）
- **推理步骤数**：3（读取两个位置、比较楼栋）
- **必要证据页面 ID**：`alice`、`david`
- **题目状态**：已录入（正式数据集 26 题版本）

**预期答案要点**

- Alice Smith 的办公室在 Building A 二层。
- David Kim 的办公室在 Building B 四层。
- 两人不在同一栋楼；答案应为否定。

**答案判定与评分注意事项**

- 用 `["no"]` + `"building a"` + `"building b"`；不要用单字母 `"a"`/`"b"`。
- 错误答案 “yes” 必须失败；标准答案说明“以办公室所在楼栋作为工作楼栋”。
- 页面数 2、关系边 0、步骤 3；`hops` 按页面口径填 2。

**问题（待我填写）**

> Alice Smith 和 David Kim 是否在同一栋楼工作？

**标准答案（待我填写）**

> 不是。Alice Smith 的办公室在 Building A 二层，David Kim 的办公室在 Building B 四层，因此两人不在同一栋楼工作。

---

### mh27 · D07 · 角色否定 · 难度：中

- **新题 ID**：`mh27_carol_phoenix_lead`
- **槽位 ID**：D07
- **题型**：`two_page_negative`
- **难度**：中
- **页面数**：2
- **关系边数**：2（Carol → Project Phoenix → Alice）
- **推理步骤数**：2（确认 Carol 参与、确认 Alice 领导）
- **必要证据页面 ID**：`carol`、`project_phoenix`
- **题目状态**：待编写（待编写 / 待校验 / 已通过）

**预期答案要点**

- Carol Chen 参与 Project Phoenix，但不是项目负责人。
- Alice Smith 才是 Project Phoenix 的负责人。
- 答案应为否定。

**评分规则设计提示**

- 用 `["no"]` + `["alice"]`；不要让题干里的 `"carol"` 满足答案。
- 错误答案 “yes” 必须失败；若答案只写 “Carol Chen” 而不否定，应判错。
- 页面数 2、关系边 2、步骤 2；这是纯链接链 `carol → project_phoenix → alice`。

**问题（待我填写）**

> Carol Chen 是否领导 Project Phoenix？如果不是，谁负责领导该项目？

**标准答案（待我填写）**

> 不是。Project Phoenix 由 Alice Smith 领导，Carol Chen 是该项目的核心成员。

---

### mh28 · E09 · 错误前提（Building B 一层） · 难度：高

- **新题 ID**：`mh28_first_floor_office_premise`
- **槽位 ID**：E09
- **题型**：`false_premise`
- **难度**：高
- **设计目标**：检查前提“Building B 一层有某人的办公室，且该人领导某个项目”是否成立；building_b 明确写一层是 gym，David 在四层、Eve 在五层，因此该前提不成立。与 mh19 的三层/Bob 团队题区分。
- **涉及页面**：0 页（按不可回答题约定录入 `required_page_ids: []`）
- **页面关系边**：0 条
- **属性查询**：0 次（读取楼层清单，不是查询人物属性）
- **推理步骤数**：1（确认 Building B 一层是 gym、没有人物办公室）
- **必要证据页面 ID（录入 `required_page_ids`）**：`[]`
- **实际验证页面**：`building_b`；`david`、`eve` 可选作旁证
- **题目状态**：已录入（正式数据集 26 题版本）

**预期答案要点**

- Building B 一层是带更衣室和淋浴的 gym，不是办公室。
- Building B 四层是 David Kim 的办公室，五层是 Eve Martinez 的办公室。
- 题目中的前提“一层办公室的使用者领导某个项目”不成立；不存在这样的人物或项目。
- 标准答案必须明确指出“前提不成立 / 没有这样的项目”，不能只给一个猜测的项目名。

**答案判定与评分注意事项**

- 使用 `refusal_keywords`，例如 `"no such project"`、`"does not lead"`、`"not stated"`、`"no project"`；不要用普通 `answer_keyword_groups` 接受一个项目名。
- 判定依据是 `building_b` 页的明确楼层清单：一层 gym、四层 David、五层 Eve；不是“现实中没有”，也不是“语料没提到所以现实不存在”。
- 如果回答 “Project Phoenix” 或 “Project Orion” 但没有指出前提不成立，应判错。
- 按现有不可回答题约定填 `required_page_ids: []`、`hops: 0`；实际验证页面只写在备注里，不写入 `required_page_ids`，否则会改变 recall 口径。
- 已知限制：M4.3.1 起 `fabrication_keywords` 可拦截“I cannot find a project; maybe Project Phoenix”这类已列出的编造表达，但仍是词法启发式（同义改写可能漏判）；不可回答题仍需人工复核。

**问题（待我填写）**

> Building B 一层的办公室由谁使用？该人领导哪个项目？

**标准答案（待我填写）**

> Building B 一层不是办公室，而是带更衣室和淋浴的 gym；该层没有人物办公室。David Kim 的办公室在四层，Eve Martinez 的办公室在五层。因此题目前提不成立，不存在这样的人物或项目。

---

### mh29 · E01 · 薪资不可回答 · 难度：低

- **新题 ID**：`mh29_alice_salary`
- **槽位 ID**：E01
- **题型**：`unanswerable`
- **难度**：低
- **设计目标**：验证当前 9 个语料页面都没有 Alice Smith 的薪资；这是语料缺失题，不是多跳题，也不等同于现实世界没有薪资信息。
- **涉及页面**：0 页
- **页面关系边**：0 条
- **属性查询**：0 次
- **推理步骤数**：1（检查全部 9 页，确认没有薪资信息）
- **必要证据页面 ID（录入 `required_page_ids`）**：`[]`
- **实际验证范围**：全部 9 个知识页面
- **题目状态**：已录入（正式数据集 26 题版本）

**预期答案要点**

- 当前 9 个语料页面都没有给出 Alice Smith 的薪资。
- 正确答案必须说明“语料未提供”，不能把现实世界可能存在的薪资信息当作语料证据。
- 标准答案不应出现具体金额。

**答案判定与评分注意事项**

- `answer_keyword_groups` 填 `[]`；`refusal_keywords` 必须非空，例如 `"not stated"`、`"no information"`、`"not provided"`。
- 理想情况下不应让数字答案通过；M4.3.1 起 `fabrication_keywords` 可拦截“not stated; maybe $X”这类已列出的混合回答，但仍是词法启发式，需要人工校验。
- 不要把题干里的 `"salary"` 放进 refusal keywords。
- 页面/边/属性/步骤为 0/0/0/1；这是不可回答覆盖题，不是多跳题。

**问题（待我填写）**

> Alice Smith 的薪资是多少？

**标准答案（待我填写）**

> 当前语料没有提供 Alice Smith 的薪资信息；9 个知识页面均未给出金额，不能据此推测。

---

### mh30 · E07 · 大学不可回答 · 难度：低

- **新题 ID**：`mh30_david_university`
- **槽位 ID**：E07
- **题型**：`unanswerable`
- **难度**：低
- **设计目标**：验证当前语料只给出 David Kim 的博士领域、没有大学名称；这是语料缺失题，不是多跳题，也不等同于现实世界不知道他的学校。
- **涉及页面**：0 页
- **页面关系边**：0 条
- **属性查询**：0 次
- **推理步骤数**：1（检查 David 页及全部页面，确认没有学校名）
- **必要证据页面 ID（录入 `required_page_ids`）**：`[]`
- **实际验证范围**：`david` 页及全部 9 个知识页面
- **题目状态**：已录入（正式数据集 26 题版本）

**预期答案要点**

- David Kim 的页面只写了“分布式系统博士”，没有给出大学名称。
- 其他页面也没有补充学校名。
- 正确答案必须说明“当前语料未提供”，不能把现实世界的信息当作语料证据。

**答案判定与评分注意事项**

- `answer_keyword_groups` 填 `[]`；`refusal_keywords` 必须非空，例如 `"not stated"`、`"no information"`、`"not provided"`。
- 不要把 `"phd"`/`"doctor"` 当作拒绝词：正常答案也会提到学位。
- 理想情况下不应让现实学校名通过；M4.3.1 起 `fabrication_keywords` 可拦截“not stated; maybe MIT”这类已列出的混合回答，但仍是词法启发式，需要人工校验。
- 页面/边/属性/步骤为 0/0/0/1；这是不可回答覆盖题，不是多跳题。

**问题（待我填写）**

> David Kim 就读于哪所大学？

**标准答案（待我填写）**

> 当前语料没有提供 David Kim 就读的大学名称；他的页面只写有分布式系统博士学位，没有学校信息。
