# zmwiki → ZMWiki 只读架构审计报告

> **文档状态（2026-10-28 修订）**
>
> 本报告是 ZMWiki 实现前的只读架构审计与实施计划。文中目录树、文件/测试数量、
> “当前缺失”“推荐目标架构”“分阶段计划”等描述均为审计当时状态；后续阶段已实现
> Wiki 编译、链接解析、WikiAgent、TUI 与本地文档生命周期等功能。当前能力、安装
> 与操作方式请以 README 为准；本报告保留作为设计依据，不再逐项更新。

> **审计性质**
>
> ：只读源码审计，未修改 / 创建 / 删除 / 移动任何文件，未运行测试，未安装依赖。
> **审计范围**
>
> ：
>
> `src/zmwiki/`
>
>  全部 33 个源码文件、
>
> `tests/`
>
>  42 个测试文件、
>
> `benchmark/`
>
>  语料与评测结果、
>
> `docs/`
>
>  架构文档，以及参考项目 
>
> `WeAgentAI/LLM-Wiki`
>
> （arXiv:2605.25480）。
> **证据标注**
>
> ：每项结论附文件路径与类 / 函数名；无源码证据的内容标注「待核实」。



***

## 一、当前架构与真实调用链

### 1.1 仓库目录树（实际）



```
ZMWiki\
├── README.md                          # 25KB，功能说明与评测结果
├── pyproject.toml                     # 零运行时依赖，semantic 为可选 extra
├── src\zmwiki\                         # 33 个 .py 模块
│   ├── __init__.py                    # 公开 API 重导出（62 个符号）
│   ├── schema.py                      # Document / Chunk / Source / Metadata
│   ├── pipeline.py                    # RAGPipeline / DeterministicRAGPipeline
│   ├── retrievers.py                  # Dense / Lexical / Hybrid Retriever
│   ├── stores.py                      # VectorStore / InMemoryVectorStore / SearchResult
│   ├── lexical.py                     # LexicalIndex / InMemoryLexicalIndex (BM25)
│   ├── fusion.py                      # WeightedScoreFusion / RRFFusion
│   ├── rerankers.py                   # Reranker / DeterministicReranker
│   ├── semantic_rerank.py             # CrossEncoderReranker（可选，非公开 API）
│   ├── rewriters.py                   # QueryRewriter / DeterministicQueryRewriter
│   ├── context.py                     # Context / ContextBuilder / DeterministicContextBuilder
│   ├── citations.py                   # Citation / AnswerResult / citations_from_context
│   ├── generators.py                  # Generator / DeterministicGenerator / build_prompt
│   ├── llm_generators.py              # OpenAICompatibleGenerator（stdlib HTTP）
│   ├── chunkers.py                    # Chunker / FixedSizeChunker
│   ├── embedders.py                   # EmbeddingProvider / DeterministicEmbedder
│   ├── semantic.py                    # SemanticEmbeddingProvider（可选，非公开 API）
│   ├── ingestion.py                   # TextIngestor / PlainTextIngestor
│   ├── file_ingestion.py              # FileIngestor / PlainTextFileIngestor / FileSource
│   ├── directory_ingestion.py         # DirectoryIngestor / PlainTextDirectoryIngestor
│   ├── tokenizers.py                  # Tokenizer / SimpleTokenizer
│   ├── text.py                        # 评测用文本工具（分句/分词/覆盖率）
│   ├── evaluation.py                  # RetrievalEvaluator（HR/Recall/Precision/MRR/MAP/NDCG）
│   ├── evaluation_dataset.py          # EvaluationCaseSet
│   ├── evaluation_harness.py          # EvaluationHarness
│   ├── evaluation_loader.py           # JsonEvaluationDatasetLoader
│   ├── evaluation_runner.py           # RetrievalEvaluationRunner
│   ├── benchmark.py                   # BenchmarkRunner / DeterministicBenchmarkRunner
│   ├── answer_evaluation.py           # AnswerEvaluator（lexical coverage/groundedness/citation）
│   └── attribution.py                 # ClaimExtractor / ClaimAttributor（claim-level）
├── tests\                              # 42 个测试文件，1281 passed / 5 skipped
├── benchmark\
│   ├── corpus\                         # 6 个 .txt 文档（doc_a ~ doc_f）
│   ├── datasets\                       # retrieval_small.json (8 cases), answer_small.json (12+ cases)
│   ├── results\                        # 13 个评测结果 JSON
│   └── run_*.py                        # 9 个评测运行脚本
└── docs\
    ├── architecture.md                 # 110KB，66 个章节，逐模块设计契约
    ├── claim_attribution_benchmark.md
    ├── query_rewriter_benchmark.md
    ├── retrieval_strategy_benchmark.md
    └── semantic_retrieval_benchmark.md
```

> **结构核对（2026-10-28）**：以上目录树为审计时快照。当前 `src/zmwiki/` 已新增
> `wiki/`、`tui/`、`web/` 三个子包，共 45 个 .py 文件；`tests/` 现有 61 个测试文件；
> 演示语料新增 `benchmark/corpus/wiki/`（9 篇）。测试通过/跳过数、评测结果文件数等
> 量化数据属历史审计记录，本次未复验。

### 1.2 主要模块职责与依赖关系

zmwiki 是一个**严格分层、依赖注入、确定性优先**的 RAG 检索 / 评测库。核心数据流：



```
Document (schema.py)
  → Chunk[] (chunkers.py: FixedSizeChunker)
    → Embedding (embedders.py) → VectorStore.upsert (stores.py)
    → LexicalIndex.upsert (lexical.py: BM25)

Query
  → QueryRewriter.rewrite (rewriters.py, 可选)
    → Retriever.retrieve (retrievers.py: Dense/Lexical/Hybrid)
      → FusionStrategy.fuse (fusion.py, Hybrid 内部)
    → Reranker.rerank (rerankers.py, 可选)
  → ContextBuilder.build (context.py) → Context
    → Generator.generate (generators.py / llm_generators.py, 可选) → str
    → citations_from_context (citations.py) → AnswerResult
```

**模块依赖方向（无循环依赖，源码已证实）**：



* `schema.py`：零依赖，纯数据契约

* `tokenizers.py`：零依赖

* `text.py`：依赖 `tokenizers.py`

* `embedders.py`：依赖 `schema.py`

* `stores.py`：依赖 `embedders.py`, `schema.py`

* `lexical.py`：依赖 `schema.py`, `stores.py`, `tokenizers.py`

* `fusion.py`：依赖 `stores.py`

* `retrievers.py`：依赖 `embedders.py`, `fusion.py`, `lexical.py`, `schema.py`, `stores.py`

* `rerankers.py`：依赖 `schema.py`, `stores.py`, `tokenizers.py`

* `rewriters.py`：依赖 `schema.py`

* `context.py`：依赖 `schema.py`, `stores.py`

* `citations.py`：依赖 `context.py`, `schema.py`

* `generators.py`：依赖 `context.py`, `schema.py`, `tokenizers.py`

* `llm_generators.py`：依赖 `context.py`, `generators.py`, `schema.py`

* `chunkers.py`：依赖 `schema.py`

* `ingestion.py` / `file_ingestion.py` / `directory_ingestion.py`：依赖 `schema.py`

* `pipeline.py`：依赖以上全部核心阶段

* `evaluation.py` / `evaluation_dataset.py` / `evaluation_harness.py` / `evaluation_loader.py` / `evaluation_runner.py` / `benchmark.py`：依赖 `schema.py`, `stores.py`, `pipeline.py`

* `answer_evaluation.py`：依赖 `attribution.py`, `citations.py`, `context.py`, `schema.py`, `text.py`, `tokenizers.py`

* `attribution.py`：依赖 `citations.py`, `context.py`, `schema.py`, `text.py`, `tokenizers.py`

* `semantic.py` / `semantic_rerank.py`：可选模块，懒加载 `sentence-transformers`，不在默认 import 图中

### 1.3 公开 API（`__init__.py` 已证实，62 个符号）

**数据契约**：`Document`, `Chunk`, `Source`, `Metadata`, `Embedding`, `SearchResult`, `Context`, `Citation`, `AnswerResult`, `Claim`, `ClaimAttribution`, `AttributionResult`, `EvaluationCase`, `EvaluationPrediction`, `EvaluationResult`, `AnswerEvaluation`, `EvaluationCaseSet`

**摄入**：`PlainTextIngestor`, `PlainTextFileIngestor`, `PlainTextDirectoryIngestor`, `FileSource`

**分块 / 嵌入**：`FixedSizeChunker`, `DeterministicEmbedder`, `EmbeddingProvider`

**存储 / 索引**：`InMemoryVectorStore`, `VectorStore`, `InMemoryLexicalIndex`, `LexicalIndex`

**检索 / 融合 / 重排**：`DenseRetriever`, `LexicalRetriever`, `HybridRetriever`, `Retriever`, `WeightedScoreFusion`, `RRFFusion`, `FusionStrategy`, `DeterministicReranker`, `Reranker`

**查询改写**：`DeterministicQueryRewriter`, `QueryRewriter`

**上下文 / 生成**：`DeterministicContextBuilder`, `ContextBuilder`, `DeterministicGenerator`, `Generator`, `build_prompt`, `OpenAICompatibleGenerator`, `HttpRequest`, `HttpResponse`, `Transport`, `urllib_transport`

**流水线**：`DeterministicRAGPipeline`, `RAGPipeline`

**评测**：`RetrievalEvaluator`, `Evaluator`, `EvaluationHarness`, `JsonEvaluationDatasetLoader`, `EvaluationDatasetLoader`, `RetrievalEvaluationRunner`, `BenchmarkRunner`, `DeterministicBenchmarkRunner`, `AnswerEvaluator`, `aggregate_evaluations`, `DeterministicClaimExtractor`, `ClaimExtractor`, `DeterministicClaimAttributor`, `ClaimAttributor`

**工具**：`SimpleTokenizer`, `Tokenizer`

**关键观察**：`semantic.py`（`SemanticEmbeddingProvider`）和 `semantic_rerank.py`（`CrossEncoderReranker`）**不在公开 API 中**，需 `from zmwiki.semantic import ...` 单独导入。

### 1.4 真实调用链（以 `answer_with_sources` 为例，源码已证实）



```
pipeline.answer_with_sources(query, top_k, candidate_k, max_chars)
  └─ pipeline.query(query, ...)
      └─ pipeline.retrieve(query, top_k, candidate_k)
          ├─ pipeline._retrieval_query(query)          # rewriter.rewrite() 或原样
          ├─ retriever.retrieve(retrieval_query, top_k=candidate_k, candidate_k=candidate_k)
          │   └─ [HybridRetriever] dense_retriever.retrieve() + lexical_retriever.retrieve() + fusion.fuse()
          └─ [reranker 非 None] reranker.rerank(retrieval_query, candidates, top_k)
      └─ context_builder.build(results, max_chars)     # → Context(text, chunks)
  ├─ generator.generate(query, context)                  # 原始 query，非 retrieval_query
  └─ citations_from_context(context)                     # → tuple[Citation]
```

**摄入调用链**：



```
pipeline.ingest(document)
  └─ for chunk in chunker.chunk(document):
      ├─ embedder.embed(chunk.text)          # 每 chunk 恰好一次
      ├─ vector_store.upsert(chunk, embedding)
      └─ lexical_index.upsert(chunk)
```



***

## 二、现有模块复用清单

按「可直接复用 / 可复用但需调整接口 / 需要重构 / 需要新增 / 可以暂缓或删除」五类分类。每项附源码证据。

### 2.1 可直接复用（无需修改，作为 ZMWiki 基础设施）



| 模块        | 文件路径                                                                                                                       | 类 / 函数                                                                                                                               | 当前职责                                            | 复用理由                                                                  |
| --------- | -------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------- | --------------------------------------------------------------------- |
| 数据契约      | `src/zmwiki/schema.py`                                                                                                      | `Document`, `Chunk`, `Source`, `Metadata`                                                                                            | 不可变值对象，Document→Chunk 1:N                       | Wiki 页面编译的输入仍是 Document；Chunk 可作为页面片段的中间表示。Metadata 是自由 dict，可承载页面元数据 |
| 分词器       | `src/zmwiki/tokenizers.py`                                                                                                  | `SimpleTokenizer`, `Tokenizer`                                                                                                       | 确定性分词，CJK 单字                                    | wiki\_search 的词法搜索后端可直接复用                                             |
| 文本工具      | `src/zmwiki/text.py`                                                                                                        | `split_sentences`, `token_coverage`, `tokenize_text`, `unique_tokens`, `safe_ratio`                                                  | 评测用文本归一化                                        | Error Book 的内容验证、页面质量检查可复用                                            |
| BM25 词法索引 | `src/zmwiki/lexical.py`                                                                                                     | `InMemoryLexicalIndex`, `LexicalIndex`                                                                                               | BM25 检索，全文本索引                                   | wiki\_search 的「正文搜索」后备路径可直接复用；页面元数据搜索需新建索引                            |
| 向量存储      | `src/zmwiki/stores.py`                                                                                                      | `InMemoryVectorStore`, `VectorStore`, `SearchResult`, `cosine_similarity`                                                            | 余弦相似度暴力搜索                                       | Wiki 页面的语义搜索可复用；但当前仅索引 Chunk，需扩展为索引 WikiPage                          |
| 嵌入提供者契约   | `src/zmwiki/embedders.py`                                                                                                   | `EmbeddingProvider`, `Embedding`, `DeterministicEmbedder`                                                                            | 文本→向量抽象                                         | Wiki 页面嵌入复用同一契约                                                       |
| 语义嵌入      | `src/zmwiki/semantic.py`                                                                                                    | `SemanticEmbeddingProvider`                                                                                                          | sentence-transformers 真实嵌入                      | Wiki 编译和搜索的语义嵌入后端                                                     |
| 融合策略      | `src/zmwiki/fusion.py`                                                                                                      | `WeightedScoreFusion`, `RRFFusion`, `FusionStrategy`                                                                                 | 稠密 + 词法结果融合                                     | wiki\_search 中元数据搜索 + 正文搜索 + 语义搜索的多路融合可复用                             |
| 评测指标      | `src/zmwiki/evaluation.py`                                                                                                  | `RetrievalEvaluator`                                                                                                                 | HR/Recall/Precision/MRR/MAP/NDCG                | ZMWiki 的检索质量评测直接复用                                                    |
| 答案评测      | `src/zmwiki/answer_evaluation.py`                                                                                           | `AnswerEvaluator`, `aggregate_evaluations`                                                                                           | lexical coverage/groundedness/citation coverage | ZMWiki 的答案质量评测直接复用                                                    |
| 声明级归因     | `src/zmwiki/attribution.py`                                                                                                 | `DeterministicClaimExtractor`, `DeterministicClaimAttributor`                                                                        | 句子级 claim 提取与归因                                 | ZMWiki 的证据充分性判断可复用其 claim 提取逻辑                                        |
| 评测基础设施    | `src/zmwiki/evaluation_dataset.py`, `evaluation_harness.py`, `evaluation_loader.py`, `evaluation_runner.py`, `benchmark.py` | `EvaluationCaseSet`, `EvaluationHarness`, `JsonEvaluationDatasetLoader`, `RetrievalEvaluationRunner`, `DeterministicBenchmarkRunner` | 离线评测编排                                          | ZMWiki 评测复用同一套基础设施，仅需扩展 dataset schema                                |
| HTTP 客户端  | `src/zmwiki/llm_generators.py`                                                                                              | `OpenAICompatibleGenerator`, `HttpRequest`, `HttpResponse`, `Transport`, `urllib_transport`                                          | stdlib HTTP，可注入 transport                       | Wiki 编译的 LLM 调用、Agent 的 tool-calling LLM 调用均可复用此 HTTP 边界和错误处理模式       |
| 摄入模块      | `src/zmwiki/ingestion.py`, `file_ingestion.py`, `directory_ingestion.py`                                                    | `PlainTextIngestor`, `PlainTextFileIngestor`, `PlainTextDirectoryIngestor`, `FileSource`                                             | 文本 / 文件 / 目录摄入                                  | Wiki 编译的原始文档摄入直接复用                                                    |

### 2.2 可复用但需要调整接口



| 模块    | 文件路径                      | 类 / 函数                                                               | 当前职责                                               | 需调整内容                                                                                                                                | 理由                                                                                        |
| ----- | ------------------------- | -------------------------------------------------------------------- | -------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------- |
| 分块器   | `src/zmwiki/chunkers.py`   | `FixedSizeChunker`, `Chunker`                                        | 固定字符窗口分块                                           | 需新增 `WikiPageChunker` 或扩展 `Chunker` 契约以支持「语义段落 / 标题感知」分块；当前仅按字符数切分，不适合 Wiki 页面编译的输入预处理                                               | LLM-Wiki 的编译输入是 Markdown article（`preprocess_bench.py`），需要保留标题结构；FixedSizeChunker 可作为基线保留 |
| 检索器   | `src/zmwiki/retrievers.py` | `DenseRetriever`, `LexicalRetriever`, `HybridRetriever`, `Retriever` | query→`list[SearchResult]`（Chunk 级）                | 需新增 `WikiSearchRetriever`，返回 `WikiPageSearchResult`（含页面元数据：title/aliases/tags/description），而非 Chunk；`Retriever` 抽象可保留但返回类型需泛化        | wiki\_search 先搜元数据再搜正文，返回候选页面 + 元数据，与当前 Chunk 级 SearchResult 不同                           |
| 上下文构建 | `src/zmwiki/context.py`    | `DeterministicContextBuilder`, `ContextBuilder`, `Context`           | SearchResult→Context（字符预算打包）                       | 需扩展为支持「Wiki 页面内容」作为上下文单元，而非 Chunk；`Context.chunks` 当前是 `tuple[Chunk, ...]`，需支持 `tuple[WikiPage, ...]` 或统一为 `ContextItem` 抽象          | Agent 读取的是 Wiki 页面全文，不是 Chunk；但 Context 的「打包 + 预算 + 来源追溯」设计可复用                            |
| 生成器   | `src/zmwiki/generators.py` | `Generator`, `DeterministicGenerator`, `build_prompt`                | query+Context→answer                               | `Generator` 契约可保留；但 ZMWiki 的 Agent 不是单次 generate，而是多轮 tool-calling 循环；需新增 `WikiAgent` 编排层，`Generator` 作为最终答案合成步骤                     | 当前 Generator 是单次调用，Agent 需要循环调用 LLM 并解析 tool call                                         |
| 流水线   | `src/zmwiki/pipeline.py`   | `DeterministicRAGPipeline`, `RAGPipeline`                            | ingest/retrieve/query/answer/answer\_with\_sources | 需新增 `WikiPipeline`（或 `ZMWikiPipeline`），包含 `compile()`（离线编译）和 `agent_query()`（在线 Agent 检索）两个新入口；`retrieve/query/answer` 可保留为 RAG 基线模式 | 当前 pipeline 是单次检索流水线，ZMWiki 需要离线编译 + 在线 Agent 双模式                                         |
| 引用    | `src/zmwiki/citations.py`  | `Citation`, `AnswerResult`, `citations_from_context`                 | Chunk→Citation 投影                                  | `Citation` 当前携带 `chunk_id`/`position`，需扩展为支持 `page_id`/`page_path`/`source_document_id`；`citations_from_context` 逻辑可复用               | Wiki 答案的引用应指向 Wiki 页面，而非原始 Chunk                                                          |
| 重排器   | `src/zmwiki/rerankers.py`  | `Reranker`, `DeterministicReranker`                                  | 候选重排序                                              | 可复用，但当前重排对象是 `SearchResult[Chunk]`；ZMWiki 中 wiki\_search 返回页面候选，重排对象变为 `WikiPageSearchResult`                                        | 重排逻辑（词法重叠 / CrossEncoder）与检索对象解耦，仅需适配输入类型                                                 |

### 2.3 需要重构



| 模块    | 文件路径                             | 类 / 函数                                        | 当前职责                                       | 重构方向                                                                                                                 | 理由                                                                                                                    |
| ----- | -------------------------------- | --------------------------------------------- | ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------- |
| 查询改写  | `src/zmwiki/rewriters.py`         | `QueryRewriter`, `DeterministicQueryRewriter` | query→retrieval query（单次）                  | 重构为 Agent 内部的「查询分解 / 子问题生成」能力，而非单次改写；当前 `rewrite(query)→str` 契约太窄                                                    | LLM-Wiki 的 Agent 通过 wiki\_search/wiki\_read 循环探索，不是单次 query rewrite；DeterministicQueryRewriter（仅空白归一化）在 ZMWiki 中无实际价值 |
| 评测数据集 | `src/zmwiki/evaluation_loader.py` | `JsonEvaluationDatasetLoader`                 | 加载 `{cases:[{query, relevant_chunk_ids}]}` | 扩展 schema 以支持 multi-hop 评测：`{query, answer, hop_count, required_pages, evidence_chain}`；当前仅支持单跳 chunk 级 ground truth | ZMWiki 的核心价值在 multi-hop 检索，当前评测数据集无法衡量                                                                                |
| 评测运行器 | `src/zmwiki/evaluation_runner.py` | `RetrievalEvaluationRunner`                   | 对每个 case 调用一次 `pipeline.retrieve()`        | 扩展为支持 Agent 模式评测：对每个 case 运行完整 Agent 循环（wiki\_search+wiki\_read+tool calls），记录 tool call 次数、读取页面数、延迟                 | 当前仅评测单次检索，无法衡量 Agent 的多步探索效果                                                                                          |

### 2.4 需要新增（ZMWiki 核心，当前完全缺失）



| 新模块                  | 职责                                                                                                                                                                       | 对应 LLM-Wiki 模块                                      | 不可缺少的理由                                           |
| -------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------- | ------------------------------------------------- |
| `wiki_schema.py`     | WikiPage 数据模型：id/path/title/aliases/tags/description/content/sources/links\_to/links\_from/metadata；页面类型目录（page\_types.yaml）                                             | `configs/wiki-schema.md`, `configs/page_types.yaml` | ZMWiki 的核心数据结构，当前完全没有「页面」概念，只有 Document 和 Chunk   |
| `wiki_compiler.py`   | 离线编译引擎：SelectPages（选相关已有页面）→ CompileWikiPages（生成页面 + 链接 + 索引）→ 结构验证→内容验证→ Error Book 反馈→增量更新                                                                             | `bench_ingest.py`（two-step LLM ingestion engine）    | 将原始文档编译为结构化 Wiki 页面是 ZMWiki 与 RAG 的根本区别           |
| `wiki_index.py`      | Wiki 页面索引：元数据索引（title/aliases/tags/description）+ 正文索引 + 反向链接索引 + 目录索引（\_index.md）                                                                                        | `wiki_retriever.py` 中的索引部分                          | wiki\_search 先搜元数据再搜正文，需要独立于 Chunk 索引的页面索引        |
| `wiki_retriever.py`  | `wiki_search(query)`：元数据优先搜索→正文搜索→返回候选页面 + 元数据；`wiki_read(paths)`：批量读取页面 / 目录索引，返回页面全文 + 链接                                                                              | `wiki_retriever.py`                                 | LLM-Wiki 的两个核心工具接口，当前完全缺失                         |
| `wiki_agent.py`      | Retrieval-as-Reasoning Agent：tool-calling 循环（wiki\_search/wiki\_read），三种遍历模式（Direct/Link-Following/Browse\&Aggregation），停止条件（T\_max=15, P=3 空搜索，至少一次 wiki\_read），证据充分性判断 | `wiki_agent.py`                                     | Agent-native 检索的核心编排，当前 pipeline 是单次检索，无 Agent 循环 |
| `error_book.py`      | Error Book：error\_book.yaml 持久化，5 步自校正（Discover→Attribute→Constrain→Inject→Verify），Layer1 代码自动修复（断链 / 格式 / 索引不一致），Layer2 LLM 周期修复（缺失页面 / 不完整摘要 / 无依据句子 / 页面矛盾）           | `bench_error_book.py`                               | LLM-Wiki 的自进化机制，当前完全缺失                            |
| `wiki_evaluation.py` | ZMWiki 专用评测：multi-hop F1/EM、hop-wise/type-wise 分解、tool call 预算消耗、页面读取数、延迟、编译成本（token / 时间 / 费用）                                                                          | `evaluate.py`, `run_qa.py`                          | 当前评测仅覆盖单跳检索和 lexical 答案评测，无法衡量 multi-hop Agent 检索 |

### 2.5 可以暂缓或删除



| 模块      | 文件路径                      | 类 / 函数                       | 理由                                                                                                      |
| ------- | ------------------------- | ---------------------------- | ------------------------------------------------------------------------------------------------------- |
| 确定性查询改写 | `src/zmwiki/rewriters.py`  | `DeterministicQueryRewriter` | 仅做空白归一化，README 已明确「不能改善检索」；ZMWiki 中 Agent 通过多步搜索探索，不需要单次 query rewrite。可保留接口但不作为主流程                     |
| 确定性生成器  | `src/zmwiki/generators.py` | `DeterministicGenerator`     | 抽取式（返回最高词法重叠 chunk 原文），仅用于测试 / CI；ZMWiki 的最终答案由 Agent 基于多页面证据合成，DeterministicGenerator 可保留为测试基线         |
| 确定性重排器  | `src/zmwiki/rerankers.py`  | `DeterministicReranker`      | 词法 token 重叠重排，仅用于测试；ZMWiki 中 wiki\_search 已有元数据优先级排序，可保留为测试基线                                           |
| 确定性嵌入器  | `src/zmwiki/embedders.py`  | `DeterministicEmbedder`      | sha256 哈希向量，无语义；ZMWiki 编译和搜索需要真实语义嵌入（`SemanticEmbeddingProvider`），DeterministicEmbedder 可保留为 CI 无模型测试基线 |

> **注意**
>
> ：以上「暂缓 / 删除」仅指不作为 ZMWiki 主流程，
>
> **全部保留以维持原有 RAG 基线可运行**
>
> 。zmwiki 的 1281 个测试必须继续通过。



***

## 三、与参考 LLM-Wiki 的差异表

### 3.1 核心架构差异



| 维度                  | zmwiki（当前）                                                                            | LLM-Wiki（参考）                                                                      | 差异性质                                |
| ------------------- | ------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------- | ----------------------------------- |
| **知识表示**            | Chunk（固定字符窗口，无结构）                                                                    | WikiPage（结构化 Markdown，含 title/aliases/tags/description/links/sources）             | 根本差异：zmwiki 无「页面」概念                  |
| **原始文档与知识分离**       | 否：Document→Chunk→索引，Chunk 既是检索单元又是存储单元                                               | 是：raw articles（原始）→ wiki pages（编译产物），二者分离，页面有 sources 指回原始                        | zmwiki 的 Chunk 是原始文档的切片，不是编译产物       |
| **页面 Schema**       | 无                                                                                    | `configs/wiki-schema.md` 定义：知识页面 + 来源页面（digests/articles），按主题目录组织，frontmatter 元数据 | 完全缺失                                |
| **页面生成 / 更新 / 重编译** | 无（仅 ingest 时 chunk+embed+index）                                                      | Algorithm 1：SelectPages→CompileWikiPages→验证→Error Book→增量提交；支持文档增量添加，不全量重建        | 完全缺失                                |
| **双向链接**            | 无（Chunk 间无关系，仅通过 document\_id 关联）                                                    | 是：页面内容中的 `[[wikilinks]]`，编译时维护 links\_to/links\_from，目录索引 `_index.md`             | 完全缺失                                |
| **搜索与读取区分**         | 否：`retrieve()` 返回 ranked Chunk 列表，`query()` 打包为 Context；无「读取页面」概念                    | 是：`wiki_search(query)` 返回候选页面 + 元数据（不返回全文）；`wiki_read(paths)` 批量读取页面全文 / 目录索引     | 根本差异：zmwiki 检索即返回全文，无 search/read 分离 |
| **Agent 多步检索**      | 无：单次 retrieve→rerank→context→generate，无循环                                            | 是：tool-calling 循环，三种遍历模式，T\_max=15 预算，P=3 空搜索停止，证据充分性判断                           | 完全缺失                                |
| **Error Book**      | 无                                                                                    | 是：error\_book.yaml 持久化，5 步自校正，Layer1 代码修复 + Layer2 LLM 周期修复                       | 完全缺失                                |
| **评测**              | 单跳检索（HR/Recall/MRR/MAP/NDCG）+ lexical 答案评测（coverage/groundedness/citation）+ claim 归因 | multi-hop QA（F1/EM，hop-wise/type-wise 分解）+ tool call 消耗 + 页面读取数 + 延迟 + 编译成本       | zmwiki 评测无法衡量 multi-hop              |

### 3.2 可复用的对应关系



| zmwiki 能力                                                                     | LLM-Wiki 对应                           | 复用方式                                             |
| ---------------------------------------------------------------------------- | ------------------------------------- | ------------------------------------------------ |
| `InMemoryLexicalIndex`（BM25）                                                 | wiki\_search 的正文搜索后备                  | 直接复用，索引对象从 Chunk 变为 WikiPage                     |
| `InMemoryVectorStore` + `SemanticEmbeddingProvider`                          | wiki\_search 的语义搜索                    | 直接复用，嵌入对象从 Chunk.text 变为 WikiPage.content        |
| `WeightedScoreFusion` / `RRFFusion`                                          | 多路搜索结果融合                              | 直接复用                                             |
| `OpenAICompatibleGenerator` 的 HTTP 边界                                        | 编译 LLM 调用 + Agent tool-calling LLM 调用 | 复用 Transport/HttpRequest/HttpResponse/ 错误处理模式    |
| `RetrievalEvaluator`                                                         | 检索质量评测                                | 直接复用                                             |
| `AnswerEvaluator`                                                            | 答案质量评测                                | 直接复用（lexical 基线），需补充 LLM-judge 或 multi-hop F1/EM |
| `EvaluationCaseSet` / `EvaluationHarness` / `RetrievalEvaluationRunner`      | 评测编排                                  | 复用基础设施，扩展 dataset schema 和 runner                |
| `PlainTextIngestor` / `PlainTextFileIngestor` / `PlainTextDirectoryIngestor` | 原始文档摄入                                | 直接复用                                             |
| `FixedSizeChunker`                                                           | 编译前的文本预处理                             | 可作为基线，需新增标题感知分块                                  |
| `Context`（text+chunks + 来源追溯）                                                | Agent 读取页面后的证据组装                      | 复用设计，扩展 chunks→pages                             |
| `Citation` / `citations_from_context`                                        | 答案引用                                  | 复用投影逻辑，扩展 chunk\_id→page\_id                     |

### 3.3 LLM-Wiki 源码中已证实但 zmwiki 不具备的能力

以下来自 `WeAgentAI/LLM-Wiki` GitHub 仓库 README（源码已证实）和论文分析：



1. `wiki_search(query)`：先搜索页面名称 / 别名 / 标签 / 描述等结构化元数据，无命中再搜索正文；返回候选页面及其元数据（不返回全文）。来源：`llm_wiki_bench/wiki_retriever.py`

2. `wiki_read(paths)`：批量读取目录索引（`_index.md`）或页面全文；返回的正文中包含的链接成为下一步探索路径（traversal affordance）。来源：`llm_wiki_bench/wiki_retriever.py`

3. `wiki_agent.py`：Retrieval-as-Reasoning tool-calling Agent，组合 wiki\_search/wiki\_read，遵循链接，检查证据充分性后生成最终答案。来源：`llm_wiki_bench/wiki_agent.py`

4. `bench_ingest.py`：two-step LLM ingestion engine，Algorithm 1 的批量编译实现。来源：`llm_wiki_bench/bench_ingest.py`

5. `bench_error_book.py`：Error Book 自校正机制。来源：`llm_wiki_bench/bench_error_book.py`

6. `configs/wiki-schema.md`：Wiki 页面 schema 规范。来源：`configs/wiki-schema.md`

7. `configs/page_types.yaml`：默认页面类型目录。来源：`configs/page_types.yaml`

8. **Agent 停止条件**：T\_max=15 tool calls，P=3 连续空搜索，回答前至少一次 wiki\_read。来源：README + 论文

9. **编译错误分布**：断链 55.8-63.8%，错误来源引用 18.9-28.5%，无依据句子 5.8-12.7%，索引不一致 2.2-10.9%。来源：论文表 7

10. **编译成本**：AuthTrace 语料 860 篇→wiki 消耗 194M tokens，约 22.6 万 tokens / 篇（来自同团队 WikiLoop 后续工作，非原论文直接数据）。来源：论文分析引用



***

## 四、推荐目标架构

### 4.1 ZMWiki 目标架构图



```
┌─────────────────────────────────────────────────────────────────┐
│                        离线编译阶段（一次性）                        │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  原始文档 (Document)                                              │
│     │                                                             │
│     ▼                                                             │
│  摄入 (ingestion.py / file_ingestion.py / directory_ingestion.py) │
│     │                                                             │
│     ▼                                                             │
│  预处理分块 (chunkers.py — 新增标题感知模式)                       │
│     │                                                             │
│     ▼                                                             │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │  WikiCompiler (wiki_compiler.py) — 新增                  │     │
│  │  ① SelectPages: 从已有 wiki 索引选相关页面 (≤5)          │     │
│  │  ② CompileWikiPages: LLM 生成页面+链接+索引更新           │     │
│  │     (注入 Error Book 累积约束)                             │     │
│  │  ③ 结构验证 (代码): 断链/格式/索引一致性                   │     │
│  │  ④ 内容验证 (LLM): 来源对照/无依据句子/页面矛盾            │     │
│  │  ⑤ Error Book 更新 → 约束重提取 → 代码自动修复             │     │
│  │  ⑥ 提交到 WikiStore                                       │     │
│  └─────────────────────────────────────────────────────────┘     │
│     │                                                             │
│     ▼                                                             │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │  WikiStore (wiki_index.py) — 新增                        │     │
│  │  ┌──────────────┐ ┌──────────────┐ ┌────────────────┐  │     │
│  │  │ 元数据索引    │ │ 正文 BM25索引 │ │ 反向链接索引    │  │     │
│  │  │ title/aliases│ │ (lexical.py  │ │ links_from     │  │     │
│  │  │ tags/desc    │ │  复用)        │ │ 目录索引_index │  │     │
│  │  └──────────────┘ └──────────────┘ └────────────────┘  │     │
│  │  ┌──────────────┐                                        │     │
│  │  │ 语义向量索引  │ (stores.py + semantic.py 复用)        │     │
│  │  └──────────────┘                                        │     │
│  └─────────────────────────────────────────────────────────┘     │
│                                                                   │
│  ErrorBook (error_book.py) — 新增: error_book.yaml 持久化       │
│    Discover → Attribute → Constrain → Inject → Verify            │
│    Layer1 代码修复(每批) + Layer2 LLM 周期修复(每10批)          │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        在线检索阶段（每次查询）                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  用户 Query                                                        │
│     │                                                             │
│     ▼                                                             │
│  ┌─────────────────────────────────────────────────────────┐     │
│  │  WikiAgent (wiki_agent.py) — 新增                        │     │
│  │                                                           │     │
│  │  tool-calling 循环 (T_max=15, P=3 空搜索停止):           │     │
│  │  ┌──────────────┐  ┌──────────────┐                     │     │
│  │  │ wiki_search  │  │ wiki_read    │                     │     │
│  │  │ (wiki_retriever│ │ (wiki_retriever│                   │     │
│  │  │  .py) 新增    │  │  .py) 新增    │                     │     │
│  │  └──────┬───────┘  └──────┬───────┘                     │     │
│  │         │                   │                              │     │
│  │         ▼                   ▼                              │     │
│  │  候选页面+元数据      页面全文+链接                         │     │
│  │         │                   │                              │     │
│  │         └─────────┬─────────┘                              │     │
│  │                   ▼                                        │     │
│  │  证据充分性判断 (复用 attribution.py claim 提取)             │     │
│  │                   │                                        │     │
│  │         ┌─────────┴─────────┐                              │     │
│  │         ▼                     ▼                              │     │
│  │    证据不足，继续循环      证据充分，停止                     │     │
│  └─────────────────────────────────────────────────────────┘     │
│     │                                                             │
│     ▼                                                             │
│  答案合成 (Generator — 复用 generators.py / llm_generators.py)   │
│     │                                                             │
│     ▼                                                             │
│  AnswerResult (answer + sources) — 复用 citations.py             │
│    sources 指向 WikiPage (page_id/page_path) 而非 Chunk          │
│                                                                   │
│  三种遍历模式 (Agent 自主选择):                                    │
│  - Direct Retrieval: 1次 search + 1次 read                       │
│  - Link-Following Traversal: 沿页面链接逐跳探索                   │
│  - Browse & Aggregation: 先读目录索引，再选页面读取               │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                     RAG 基线模式（保留，不删除）                    │
├─────────────────────────────────────────────────────────────────┤
│  DeterministicRAGPipeline 全部入口保留：                           │
│  ingest / retrieve / query / answer / answer_with_sources        │
│  所有 1281 个测试继续通过，作为 ZMWiki 的公平对照基线              │
└─────────────────────────────────────────────────────────────────┘
```

### 4.2 旧模块到新模块映射



| zmwiki 旧模块                                         | ZMWiki 新角色                              | 变更类型        |
| ------------------------------------------------- | --------------------------------------- | ----------- |
| `schema.py: Document`                             | Wiki 编译的输入单元                            | 原样保留        |
| `schema.py: Chunk`                                | 编译前预处理的中间单元；不再作为最终检索单元                  | 保留但降级       |
| `chunkers.py: FixedSizeChunker`                   | 编译前文本预处理基线                              | 保留，新增标题感知模式 |
| `embedders.py: EmbeddingProvider`                 | Wiki 页面嵌入 + 原始 Chunk 嵌入                 | 原样保留        |
| `semantic.py: SemanticEmbeddingProvider`          | Wiki 页面语义嵌入后端                           | 原样保留        |
| `stores.py: VectorStore`                          | Wiki 页面语义索引 + 原始 Chunk 向量索引             | 保留，索引对象扩展   |
| `lexical.py: LexicalIndex`                        | Wiki 页面正文 BM25 索引 + 原始 Chunk BM25 索引    | 保留，索引对象扩展   |
| `fusion.py: FusionStrategy`                       | wiki\_search 多路结果融合（元数据 + 正文 + 语义）      | 原样保留        |
| `retrievers.py: Retriever`                        | 抽象保留；新增 `WikiSearchRetriever`           | 扩展          |
| `rerankers.py: Reranker`                          | wiki\_search 结果重排                       | 保留，输入类型扩展   |
| `context.py: Context`                             | Agent 证据组装（pages 替代 chunks）             | 扩展          |
| `citations.py: Citation`                          | 答案引用（page\_id 替代 chunk\_id）             | 扩展          |
| `generators.py: Generator`                        | 最终答案合成                                  | 原样保留        |
| `llm_generators.py: OpenAICompatibleGenerator`    | 编译 LLM + Agent LLM + 答案合成 LLM 的 HTTP 边界 | 原样保留        |
| `pipeline.py: DeterministicRAGPipeline`           | RAG 基线模式，保留全部入口                         | 原样保留        |
| `pipeline.py: RAGPipeline`                        | 抽象保留；新增 `WikiPipeline` 子类               | 扩展          |
| `rewriters.py: QueryRewriter`                     | 降级为可选，不作为主流程                            | 保留但降级       |
| `evaluation.py: RetrievalEvaluator`               | 检索质量评测                                  | 原样保留        |
| `answer_evaluation.py: AnswerEvaluator`           | 答案质量评测（lexical 基线）                      | 原样保留        |
| `attribution.py: ClaimAttributor`                 | 证据充分性判断的 claim 提取                       | 复用          |
| `evaluation_runner.py: RetrievalEvaluationRunner` | 扩展为支持 Agent 模式评测                        | 扩展          |
| `benchmark.py: BenchmarkRunner`                   | 评测编排                                    | 原样保留        |
| **新增:&#x20;**`wiki_schema.py`                     | WikiPage 数据模型                           | 全新          |
| **新增:&#x20;**`wiki_compiler.py`                   | 离线编译引擎                                  | 全新          |
| **新增:&#x20;**`wiki_index.py`                      | Wiki 页面多维索引                             | 全新          |
| **新增:&#x20;**`wiki_retriever.py`                  | wiki\_search + wiki\_read 工具            | 全新          |
| **新增:&#x20;**`wiki_agent.py`                      | Agent 检索循环                              | 全新          |
| **新增:&#x20;**`error_book.py`                      | Error Book 自校正                          | 全新          |
| **新增:&#x20;**`wiki_evaluation.py`                 | multi-hop 评测 + 成本分析                     | 全新          |

### 4.3 设计决定（必须在编码前确认）



1. **WikiPage 存储格式**：Markdown 文件（LLM-Wiki 方式，人类可读 + 可 diff）还是 Python 对象 + JSON 序列化（与 zmwiki 现有 in-memory 风格一致）？

* 建议：Phase 1 用 Python dataclass + JSON 持久化，与 zmwiki 现有风格一致；Markdown 导出作为 Phase 2 可选功能。

* 理由：zmwiki 当前全部是 in-memory + stdlib，无文件持久化层；引入 Markdown 文件系统会增加路径管理、文件锁、并发写入等复杂度。

1. **编译 LLM 与 Agent LLM 是否分离**：LLM-Wiki 使用 premium model（编译）+ fast model（分析），zmwiki 当前只有一个 `Generator` 契约。

* 建议：复用 `OpenAICompatibleGenerator`，通过不同实例（不同 base\_url/model）区分编译和 Agent；不新增 LLM 客户端抽象。

1. **wiki\_search 的元数据索引实现**：独立倒排索引还是复用 `InMemoryLexicalIndex` 索引元数据字段？

* 建议：复用 `InMemoryLexicalIndex`，为每个页面的元数据（title+aliases+tags+description 拼接）建立一个独立的 lexical index 实例；正文用另一个实例。不新增索引引擎。

1. **Agent 的 tool-calling 实现**：依赖 LLM 的 function calling 能力还是手动解析 JSON？

* 建议：使用 OpenAI-compatible chat completions 的 `tools` 参数（function calling），与 `llm_generators.py` 的 HTTP 边界一致；不引入新依赖。

* 注意：当前 `OpenAICompatibleGenerator` 不支持 `tools` 参数，需扩展 `_build_request` 以支持可选的 `tools` 和 `tool_choice`。

1. **Error Book 的持久化**：YAML 文件（LLM-Wiki 方式）还是 JSON？

* 建议：JSON，与 zmwiki 现有评测结果格式一致（`benchmark/results/*.json`），不引入 YAML 依赖。

1. **RAG 基线与 ZMWiki 的共存方式**：同一 `pipeline.py` 中新增 `WikiPipeline` 还是新建 `zmwiki` 子包？

* 建议：在 `src/zmwiki/` 下新增 `wiki/` 子目录（`zmwiki.wiki`），包含 wiki\_schema/wiki\_compiler/wiki\_index/wiki\_retriever/wiki\_agent/error\_book/wiki\_evaluation；不新建顶层包，保持单一安装包。

* 理由：pyproject.toml 当前只有一个包 `zmwiki`；新建包需要修改构建配置。子目录方式最小侵入。



***

## 五、分阶段实施计划及验收条件

### Phase 0：基础设施准备（不改现有代码）

**目标**：建立 ZMWiki 的数据模型和测试骨架，不触碰现有 RAG 流程。

**任务**：



1. 新增 `src/zmwiki/wiki/__init__.py`

2. 新增 `src/zmwiki/wiki/wiki_schema.py`：`WikiPage` dataclass（id/path/title/aliases/tags/description/content/sources/links\_to/links\_from/metadata），`WikiPageType` 枚举

3. 新增 `tests/wiki/test_wiki_schema.py`：WikiPage 构造、验证、不可变性测试

**验收条件**：



* `WikiPage` 可构造、可序列化（to\_dict）、可反序列化

* 所有现有 1281 个测试不受影响（零修改现有文件）

* 新增测试全部通过

### Phase 1：Wiki 索引与检索工具（wiki\_search /wiki\_read）

**目标**：实现 Wiki 页面的多维索引和两个核心工具接口，不涉及 LLM 编译（用手工构造的 WikiPage 测试）。

**任务**：



1. 新增 `src/zmwiki/wiki/wiki_index.py`：

* `WikiPageIndex`：元数据索引（复用 `InMemoryLexicalIndex`）+ 正文索引（复用 `InMemoryLexicalIndex`）+ 反向链接索引（dict）+ 目录索引

* `upsert(page)`, `get(page_id)`, `search_metadata(query, top_k)`, `search_content(query, top_k)`, `get_backlinks(page_id)`, `get_directory_index(dir_path)`

1. 新增 `src/zmwiki/wiki/wiki_retriever.py`：

* `wiki_search(query, *, top_k=5) -> list[WikiPageSearchResult]`：元数据优先→正文后备，返回候选页面 + 元数据（不返回全文）

* `wiki_read(paths: list[str]) -> list[WikiPage]`：批量读取页面 / 目录索引，返回页面全文

* `WikiPageSearchResult`：page\_id/path/title/aliases/tags/description/score（不含 content）

1. 新增 `tests/wiki/test_wiki_index.py` 和 `tests/wiki/test_wiki_retriever.py`

**验收条件**：



* `wiki_search` 先命中元数据的页面排名高于仅命中正文的页面

* `wiki_read` 返回的页面包含完整 content 和 links\_to

* 反向链接索引一致性：A.links\_to 包含 B ↔ B.links\_from 包含 A

* 断链检测：wiki\_read 一个不存在的 path 返回明确错误或空结果

* 所有现有测试继续通过

### Phase 2：Wiki 编译引擎（离线，LLM 驱动）

**目标**：实现将原始 Document 编译为 WikiPage 的核心流程，包括 SelectPages、CompileWikiPages、结构验证、增量更新。

**任务**：



1. 新增 `src/zmwiki/wiki/wiki_compiler.py`：

* `WikiCompiler`：依赖注入（llm\_client, wiki\_index, embedder, chunker）

* `select_pages(document, *, max_pages=5) -> list[WikiPage]`：从已有索引选相关页面

* `compile_pages(document, related_pages) -> list[WikiPage]`：LLM 生成页面 + 链接 + 元数据

* `validate_structure(pages) -> list[StructureError]`：断链 / 格式 / 索引一致性检查（代码规则）

* `ingest_document(document)`：完整编译流程（select→compile→validate→index→upsert）

1. 扩展 `llm_generators.py: OpenAICompatibleGenerator`：支持 `tools` 参数（function calling），用于编译和 Agent

2. 新增 `tests/wiki/test_wiki_compiler.py`：使用 fake transport 测试编译流程

**验收条件**：



* 编译一个 Document 产出至少 1 个 WikiPage，页面包含 title/aliases/tags/description/content/sources/links\_to

* 页面 sources 指回原始 Document（document\_id）

* 结构验证能检测断链（links\_to 指向不存在的 page\_id）

* 增量编译：ingest 第二个 Document 时，select\_pages 能返回第一个 Document 编译出的相关页面

* 编译过程不修改原始 Document 或现有 Chunk 索引

* 所有现有测试继续通过

### Phase 3：Error Book 自校正

**目标**：实现 Error Book 的持久化、5 步自校正、Layer1 代码自动修复。

**任务**：



1. 新增 `src/zmwiki/wiki/error_book.py`：

* `ErrorBook`：JSON 持久化（error\_book.json），`ErrorEntry`（id/error\_type/description/cause/constraint/verification\_method/status）

* `discover(errors)`, `attribute(error)`, `constrain(error) -> str`, `inject() -> list[str]`（返回所有 open 约束）, `verify(error_id) -> bool`

* Layer1 代码修复：`fix_broken_links(pages, index)`, `fix_format_errors(pages)`, `fix_index_inconsistency(pages, index)`

1. 集成到 `wiki_compiler.py`：编译时注入 Error Book 约束，验证后更新 Error Book

2. 新增 `tests/wiki/test_error_book.py`

**验收条件**：



* Error Book 可持久化到 JSON 文件并重新加载

* 编译时注入的约束出现在 LLM prompt 中（通过 fake transport 验证请求体）

* Layer1 代码修复能修复断链（将指向不存在页面的链接移除或标记）

* Error entry 的状态流转：open → verified → closed

* 所有现有测试继续通过

### Phase 4：WikiAgent 在线检索

**目标**：实现 Agent 的 tool-calling 循环、三种遍历模式、停止条件、证据充分性判断。

**任务**：



1. 新增 `src/zmwiki/wiki/wiki_agent.py`：

* `WikiAgent`：依赖注入（llm\_client, wiki\_search, wiki\_read, generator, max\_tool\_calls=15, patience=3）

* `answer(query) -> AgentAnswer`：tool-calling 循环

* 停止条件：证据充分 / T\_max 到达 / P 次连续空搜索 / 至少一次 wiki\_read

* `AgentAnswer`：answer/sources/tool\_calls\_log/pages\_read/latency

1. 扩展 `context.py` 或新增 `wiki_context.py`：支持 WikiPage 作为上下文单元

2. 扩展 `citations.py`：`Citation` 支持 `page_id`/`page_path`（可选字段，不破坏现有 chunk\_id）

3. 新增 `tests/wiki/test_wiki_agent.py`：使用 fake transport 测试 Agent 循环

**验收条件**：



* Agent 对简单问题执行 Direct Retrieval（1 search + 1 read + answer）

* Agent 对 multi-hop 问题执行 Link-Following（search→read→follow link→read→...）

* Agent 对宽泛问题执行 Browse & Aggregation（read directory index→select pages→read）

* T\_max=15 时强制停止，返回已收集证据的部分答案

* P=3 连续空搜索时提前停止

* 回答前至少执行一次 wiki\_read（无 read 直接 answer 被拒绝）

* 所有现有测试继续通过

### Phase 5：评测与成本分析

**目标**：实现 multi-hop 评测、工具调用预算分析、编译成本统计、与 RAG 基线的公平对照。

**任务**：



1. 新增 `src/zmwiki/wiki/wiki_evaluation.py`：

* `WikiEvaluationCase`：query/reference\_answer/hop\_count/required\_page\_ids

* `WikiEvaluator`：F1/EM（multi-hop），hop-wise 分解，type-wise 分解

* `CostAnalyzer`：编译成本（tokens/time），查询成本（tool\_calls/pages\_read/latency/tokens）

1. 扩展 `evaluation_loader.py`：支持 multi-hop dataset schema（向后兼容现有 retrieval\_small.json）

2. 新增 `benchmark/wiki/` 目录：multi-hop 评测数据集（可从 HotpotQA/MuSiQue 子集构造，或手工构造小规模）

3. 新增 `benchmark/run_wiki_evaluation.py`：ZMWiki vs RAG 基线对照评测脚本

4. 新增 `tests/wiki/test_wiki_evaluation.py`

**验收条件**：



* multi-hop F1/EM 可计算，hop-wise 分解正确（2-hop vs 4-hop 分别统计）

* 编译成本统计：总 tokens、总时间、每文档平均 tokens

* 查询成本统计：tool\_calls 次数、pages\_read 数量、延迟、输入 / 输出 tokens

* 公平对照：同一 corpus、同一 query set、同一 answer model，ZMWiki（Agent 循环）vs RAG（单次 top-k）

* 所有现有测试继续通过

### Phase 6：集成与文档

**目标**：将 ZMWiki 集成到公开 API，更新文档，确保 RAG 基线与 ZMWiki 双模式共存。

**任务**：



1. 更新 `src/zmwiki/__init__.py`：导出 ZMWiki 核心类（WikiPage, WikiCompiler, WikiAgent, wiki\_search, wiki\_read, ErrorBook, WikiEvaluator）

2. 新增 `docs/zmwiki_architecture.md`：ZMWiki 架构文档

3. 更新 `README.md`：新增 ZMWiki 章节（不删除 RAG 内容）

4. 新增 `examples/zmwiki_quickstart.py`：ZMWiki 快速上手示例

5. 全量测试验证：1281 现有测试 + 新增 ZMWiki 测试全部通过

**验收条件**：



* `from zmwiki import WikiPage, WikiCompiler, WikiAgent` 可正常导入

* RAG 基线快速上手示例（README 中现有）继续运行

* ZMWiki 快速上手示例可运行（编译→Agent 查询→答案）

* 所有测试通过

* 文档完整：架构图、API 参考、评测结果、与 RAG 的差异说明



***

## 六、验收与评测设计（仅方案，不运行实验）

### 6.1 文档编译完整性与来源可追溯性



| 指标     | 定义                                          | 验证方法                                                                           |
| ------ | ------------------------------------------- | ------------------------------------------------------------------------------ |
| 页面覆盖率  | 编译出的 WikiPage 数量 / 预期页面数量                   | 对已知 corpus 手工标注预期页面数，编译后对比                                                     |
| 来源追溯率  | 包含有效 sources（指回原始 Document）的页面比例            | 检查每个 WikiPage.sources 中的 document\_id 是否存在于原始 corpus                           |
| 内容保真度  | 页面内容中可追溯到原始文档的句子比例                          | 复用 `attribution.py` 的 token\_coverage，对每个页面 content 与原始 Document content 计算覆盖率 |
| 元数据完整率 | 包含 title/aliases/tags/description 全部字段的页面比例 | 结构化检查                                                                          |

### 6.2 Wiki 链接有效性与双向关系一致性



| 指标      | 定义                                                | 验证方法                                            |
| ------- | ------------------------------------------------- | ----------------------------------------------- |
| 断链率     | links\_to 中指向不存在 page\_id 的比例                     | 遍历所有页面的 links\_to，检查目标 page\_id 是否存在于 WikiIndex |
| 反向链接一致性 | 对所有 (A,B) 对：A.links\_to 包含 B ↔ B.links\_from 包含 A | 遍历验证，不一致计数                                      |
| 目录索引完整性 | \_index.md 中列出的页面与实际存在页面对应                        | 检查目录索引中的每个 path 是否可 wiki\_read                  |
| 链接密度    | 平均每个页面的出链数 / 入链数                                  | 统计分布，检测孤立页面（出链 = 0 且入链 = 0）                     |

### 6.3 多步检索能否找到跨文档证据



| 指标              | 定义                            | 验证方法                                                                       |
| --------------- | ----------------------------- | -------------------------------------------------------------------------- |
| multi-hop F1/EM | 对需要 2+ hop 的问题，答案与参考答案的 F1/EM | 使用 HotpotQA/MuSiQue/2Wiki 风格的 multi-hop 数据集                                |
| hop-wise 分解     | 按问题所需 hop 数（2/3/4）分别统计 F1/EM  | 数据集标注 hop\_count，分组统计                                                      |
| 跨文档证据召回率        | Agent 实际读取的页面中包含所需证据页面的比例     | 数据集标注 required\_page\_ids，检查 Agent 的 pages\_read 是否覆盖                      |
| 遍历路径有效性         | Agent 遵循的链接路径中，每一步的链接确实存在于源页面 | 检查 tool\_calls\_log 中每次 wiki\_read 的 path 是否来自上一次 wiki\_read 返回的 links\_to |

### 6.4 Agent 的工具调用预算和停止条件



| 指标                   | 定义                                | 验证方法                            |
| -------------------- | --------------------------------- | ------------------------------- |
| 平均 tool\_calls/query | 每个问题的 wiki\_search+wiki\_read 总次数 | 统计 AgentAnswer.tool\_calls\_log |
| 平均 pages\_read/query | 每个问题读取的 WikiPage 数量               | 统计 AgentAnswer.pages\_read      |
| T\_max 触发率           | 因达到 15 次 tool call 而停止的问题比例       | 检查停止原因                          |
| P 触发率                | 因 3 次连续空搜索而停止的问题比例                | 检查停止原因                          |
| 证据充分停止率              | 因 Agent 判断证据充分而停止的问题比例            | 检查停止原因                          |
| 延迟分布                 | 每个问题的端到端延迟（p50/p95/p99）           | 统计 AgentAnswer.latency          |

### 6.5 错误知识的记录、修正和失效处理



| 指标             | 定义                                            | 验证方法                                      |
| -------------- | --------------------------------------------- | ----------------------------------------- |
| Error Book 覆盖率 | 编译过程中发现的错误中被记录到 Error Book 的比例                | 对比验证发现的错误数与 Error Book 条目数                |
| 约束注入率          | Error Book 中 open 状态的约束被注入编译 prompt 的比例       | 检查编译请求体中的约束列表                             |
| Layer1 修复率     | 代码自动修复成功的结构错误比例                               | 对比修复前后的结构错误数                              |
| 错误复发率          | 已 closed 的错误在后续编译中重新出现的比例                     | 长期跟踪 Error Book 状态变化                      |
| 页面失效处理         | 原始 Document 删除 / 修改后，相关 WikiPage 的标记 / 更新是否正确 | 测试：删除一个 Document 后，检查其衍生页面是否被标记为 orphaned |

### 6.6 与现有 zmwiki 的公平对照评测

**对照设计原则**：



* 同一 corpus（zmwiki 现有 6 文档 / 33 chunks，或更大的 multi-hop corpus）

* 同一 query set

* 同一 answer LLM（model/temperature）

* 同一 embedding model（如使用语义检索）

* ZMWiki：Agent 循环（T\_max=15, P=3）

* RAG 基线：单次 retrieve（top\_k=5, candidate\_k=33）→ context → generate



| 维度              | ZMWiki                        | RAG 基线                   | 说明                      |
| --------------- | ----------------------------- | ------------------------ | ----------------------- |
| 单跳 F1/EM        | 需测量                           | 已有基线（answer\_small.json） | 单跳问题上 ZMWiki 不应显著差于 RAG |
| multi-hop F1/EM | 需测量                           | 不适用（RAG 无 multi-hop 能力）  | ZMWiki 的核心优势场景          |
| 证据质量            | 页面级引用，含来源追溯                   | Chunk 级引用                | 引用粒度不同，需分别评估            |
| Token 消耗（查询）    | 输入 + 输出 tokens（含多轮 tool call） | 单次 generate 的 tokens     | ZMWiki 预期更高             |
| 延迟              | 端到端（含多轮 LLM 调用）               | 单次检索 + 生成                | ZMWiki 预期更高             |
| 编译成本            | 一次性 tokens + 时间               | 零（RAG 无编译）               | ZMWiki 的额外成本            |
| 可解释性            | 页面可读 + 链接可追溯 + Agent 路径可审计    | Chunk 列表                 | ZMWiki 预期更高             |

### 6.7 如何区分 Wiki 结构带来的收益与单纯增加模型调用带来的收益

这是 LLM-Wiki 论文自身承认的方法论难点（论文中 ablation 显示「去掉遍历」F1 下降 11.7-13.8，但同时也减少了 tool call 预算）。

**消融实验设计**：



| 实验组          | Wiki 结构     | Agent 循环                           | 预算           | 目的                 |
| ------------ | ----------- | ---------------------------------- | ------------ | ------------------ |
| A: RAG 基线    | 无           | 无（单次 top-k）                        | 1 次 generate | 基线                 |
| B: 迭代 RAG    | 无（Chunk 级）  | 有（search+read Chunk 循环）            | T\_max=15    | 单纯增加模型调用的收益        |
| C: Wiki 单次读取 | 有（WikiPage） | 无（wiki\_search top-1 + wiki\_read） | 1 次 generate | 单纯 Wiki 结构的收益（无循环） |
| D: ZMWiki 完整 | 有（WikiPage） | 有（wiki\_search+wiki\_read + 链接跟随）  | T\_max=15    | Wiki 结构 + 循环的总收益   |

**归因公式**：



* 单纯增加模型调用的收益 = B - A

* 单纯 Wiki 结构的收益 = C - A

* Wiki 结构 + 循环的交互收益 = D - C - (B - A)

* ZMWiki 总收益 = D - A

**关键判断**：如果 D ≈ B（Wiki 结构无额外收益），则 ZMWiki 的价值主要来自 Agent 循环而非 Wiki 结构；如果 D > B 且 C > A，则 Wiki 结构确实带来独立收益。



***

## 七、主要风险、技术债和不应实现的功能

### 7.1 主要风险



| 风险                           | 严重度 | 说明                                                                                  | 缓解措施                                                                                                      |
| ---------------------------- | --- | ----------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| **编译质量不可控**                  | 高   | LLM 编译出的 WikiPage 可能包含幻觉、断链、矛盾；Error Book 只能减少不能消除                                  | Phase 1 先实现索引和检索工具（不依赖编译质量），用手工构造的 WikiPage 验证 Agent 循环；Phase 2 再引入 LLM 编译                                |
| **multi-hop 评测数据集缺失**        | 高   | zmwiki 现有评测集（retrieval\_small 8 cases /answer\_small 12 cases）都是单跳，无法衡量 ZMWiki 的核心价值 | Phase 5 构造小规模 multi-hop 数据集（可从 HotpotQA dev 子集采样 50-100 条，或手工构造）；不追求大规模，先验证能力                             |
| **Agent 循环的 token 成本**       | 中   | T\_max=15 次 tool call 意味着最多 15 轮 LLM 调用，token 成本和延迟远高于单次 RAG                        | 实现预算控制和提前停止；评测中明确报告 token 消耗；对单跳问题回退到 RAG 模式                                                              |
| **与现有 RAG 基线的回归风险**          | 中   | 新增代码可能意外影响现有 1281 个测试                                                               | 所有新增代码放在 `zmwiki/wiki/` 子包，不修改现有模块（除 `__init__.py` 导出和 `llm_generators.py` 可选 tools 参数扩展）；每 Phase 结束运行全量测试 |
| **LLM function calling 兼容性** | 中   | 不是所有 OpenAI-compatible 端点都支持 `tools` 参数（如部分本地模型）                                    | `OpenAICompatibleGenerator` 扩展时保持 `tools=None` 时行为不变；Agent 可降级为手动 JSON 解析模式                               |
| **Wiki 页面规模扩展性**             | 低   | LLM-Wiki 论文承认「页面数万时目录索引难以处理」；zmwiki 当前 in-memory 实现更不适合大规模                           | Phase 1-6 均不追求大规模（目标 < 1000 页面）；扩展性问题标注为「待核实」，不在本阶段解决                                                     |

### 7.2 现有技术债（zmwiki 已存在，非 ZMWiki 引入）



| 技术债                | 位置                               | 说明                                                           | 对 ZMWiki 的影响                                                 |
| ------------------ | -------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 无持久化层              | 全部 stores/indexes                | `InMemoryVectorStore` 和 `InMemoryLexicalIndex` 都是纯内存，进程退出即丢失 | Wiki 编译结果需要持久化，需新增 JSON / 文件持久化（ErrorBook 已有此需求）             |
| 无并发控制              | pipeline.py                      | `ingest` 非事务性，多线程 ingest 可能导致索引不一致                           | Wiki 编译是批量离线过程，并发风险低；但在线 Agent 查询与离线编译同时进行时需注意               |
| lexical index 全量重建 | lexical.py: `_rebuild()`         | 每次 upsert/delete 重新分词所有 chunks，O (total tokens) per write    | Wiki 页面编译是批量 upsert，全量重建可接受；但增量更新时性能可能成为问题                   |
| 无 token 计数         | 全部                               | `ContextBuilder` 用字符预算（max\_chars），无 LLM token 计数            | ZMWiki 的成本分析需要 token 计数，需新增（可通过 tiktoken 或简单估算）              |
| 评测数据集过小            | benchmark/datasets/              | retrieval\_small 仅 8 cases，answer\_small 仅 12 cases，且都是单跳    | 无法衡量 multi-hop，需新增数据集                                        |
| semantic 模块非公开 API | semantic.py, semantic\_rerank.py | 不在 `__init__.py` 导出，需单独 import                               | ZMWiki 编译和搜索需要语义嵌入，应考虑将 `SemanticEmbeddingProvider` 加入公开 API |

### 7.3 不应实现的功能（本阶段明确排除）



| 功能                               | 排除理由                                                    |
| -------------------------------- | ------------------------------------------------------- |
| **Web UI / Chat UI**             | zmwiki 定位是库，不是产品；README 明确「Not in scope: Chat UI」        |
| **MCP 服务器**                      | zmwiki 无 MCP 依赖；LLM-Wiki 的 MCP 适配是独立层，不属于核心架构            |
| **流式生成（streaming）**              | `llm_generators.py` 明确不实现 streaming；Agent 循环不需要流式       |
| **重试 / 退避（retry/backoff）**       | `llm_generators.py` 明确「one attempt, no retry loop」；保持一致 |
| **向量数据库后端（FAISS/Qdrant/Milvus）** | zmwiki 当前只有 `InMemoryVectorStore`；引入外部向量数据库超出最小改造范围      |
| **PDF/Markdown/HTML 解析**         | 当前仅支持 `.txt`；Wiki 编译的输入预处理可复用现有摄入，不新增解析器                |
| **多租户 / 权限控制**                   | zmwiki 是单用户库，无权限模型                                       |
| **分布式编译**                        | 编译是单机离线批量过程，不需要分布式                                      |
| **实时索引更新**                       | Wiki 编译是离线批量，不需要 watch 文件系统实时更新                         |
| **LLM-judge 评测**                 | 当前评测是 deterministic lexical 基线；LLM-judge 作为可选扩展，不在最小方案中 |
| **GraphRAG 风格的社区检测 / 层次摘要**      | ZMWiki 的核心是 Wiki 页面 + 双向链接，不是图社区摘要；不引入 Louvain / 层次聚类   |
| **多语言支持（非中文 / 英文）**              | 当前 `SimpleTokenizer` 对 CJK 单字处理，对其他语言支持有限；不在本阶段扩展       |



***

## 八、尚未核实的问题

以下问题在本次只读审计中无法从源码确认，需要在实施前或实施中进一步核实：



1. **LLM-Wiki 的&#x20;**`wiki-schema.md`**&#x20;具体内容**：GitHub 仓库的 `configs/wiki-schema.md` 未在 README 中展示全文，仅知其存在。WikiPage 的精确字段定义（如 aliases/tags 的格式、sources 的结构、links\_to 的表示方式）需要直接读取该文件确认。

* 当前状态：根据论文分析和 README 推断了字段，但精确 schema 待核实。

1. **LLM-Wiki 的&#x20;**`bench_ingest.py`**&#x20;编译 prompt 具体内容**：编译 prompt（SelectPages 和 CompileWikiPages 的 system/user prompt）是编译质量的关键，但论文和 README 均未给出全文。

* 当前状态：仅知 Algorithm 1 的步骤，具体 prompt 模板待核实。

1. **LLM-Wiki 的&#x20;**`wiki_agent.py`**&#x20;Agent system prompt 具体内容**：Agent 的 system prompt（如何描述工具、如何指导证据充分性判断、如何选择遍历模式）未公开。

* 当前状态：仅知工具接口和停止条件，Agent prompt 待核实。

1. **zmwiki 的&#x20;**`docs/architecture.md`**&#x20;中是否有关于未来扩展的规划**：architecture.md 共 110KB/66 章节，本次审计未逐章阅读，可能包含与 ZMWiki 相关的设计预留或排除声明。

* 当前状态：已阅读全部源码，architecture.md 仅查看了目录结构；如有「明确不负责」的声明可能影响 ZMWiki 设计决策。

1. **zmwiki 测试中是否有针对「页面」「链接」「Agent」的预留测试**：42 个测试文件中，本次仅详细阅读了 `test_pipeline.py` 的前 100 行，其他测试文件未逐一阅读。

* 当前状态：测试文件名均为 `test_<module>.py`，未见 wiki/agent 相关测试；但不排除某些测试中有预留接口测试。

1. `benchmark/results/`**&#x20;中 13 个评测结果 JSON 的完整内容**：本次仅阅读了 `retrieval_small_baseline.json`，其他 12 个结果文件未逐一解析。

* 当前状态：已知评测覆盖 retrieval strategy comparison、semantic profile、query rewriter profile、answer evaluation profile、attribution profile、truncation study 等；具体数值未全部提取。

1. **zmwiki 是否有 Git 分支或未合并的 ZMWiki 相关工作**：本次审计仅检查了工作目录的文件，未检查 Git 历史或其他分支。

* 当前状态：工作目录中未见 ZMWiki 相关代码；Git 历史待核实。

1. `pyproject.toml`**&#x20;中&#x20;**`dependencies = []`**&#x20;是否为最终决定**：零运行时依赖是 zmwiki 的核心设计原则，但 ZMWiki 的编译需要 LLM API 调用（已有 stdlib HTTP）、可能需要 JSON 持久化（stdlib json）、可能需要 token 计数（可能需要 tiktoken）。

* 当前状态：stdlib 可覆盖大部分需求；token 计数是否引入 tiktoken 待确认。

1. **LLM-Wiki 论文中报告的编译成本具体数值**：原论文未给出编译 token / 时间 / 费用数值（论文分析明确指出这是「最大的空白」）；同团队 WikiLoop 后续工作报告了 AuthTrace 语料 194M tokens，但这是不同语料、不同系统。

* 当前状态：ZMWiki 的编译成本需要自行测量，无可参考基准。

1. **zmwiki 的&#x20;**`semantic.py`**&#x20;和&#x20;**`semantic_rerank.py`**&#x20;在实际使用中的性能表现**：这两个模块是可选的，CI 不运行（测试用 fake backend），实际模型性能未在仓库中验证。

* 当前状态：代码结构正确，但实际语义检索质量待核实。



***

## 九、总结

### 9.1 核心判断



1. **zmwiki 是一个工程质量极高的 RAG 检索 / 评测库**：33 个模块严格分层、依赖注入、零循环依赖、1281 个测试、零运行时依赖。其数据契约（Document/Chunk）、存储（VectorStore/LexicalIndex）、检索（Dense/Lexical/Hybrid）、融合、重排、上下文、引用、生成、评测等模块均可直接复用于 ZMWiki。

2. **zmwiki 与 LLM-Wiki 的根本差异在于「知识表示」和「检索范式」**：

* zmwiki：Chunk（扁平、无结构、无关系）+ 单次检索（retrieval-as-lookup）

* LLM-Wiki：WikiPage（结构化、含元数据、双向链接）+ Agent 多步检索（retrieval-as-reasoning）

1. **最小改造的核心是「新增 7 个模块，修改 0 个现有模块」**：

* 新增：`wiki_schema.py`, `wiki_compiler.py`, `wiki_index.py`, `wiki_retriever.py`, `wiki_agent.py`, `error_book.py`, `wiki_evaluation.py`

* 现有模块全部保留，RAG 基线完全可运行

* 仅需扩展 `llm_generators.py`（支持 tools 参数）和 `__init__.py`（导出新模块），均为向后兼容的增量修改

1. **ZMWiki 的价值主要在 multi-hop 场景**：对单跳问题，ZMWiki 不应显著优于 RAG（且成本更高）；对需要跨文档、多跳推理的问题，Wiki 结构 + Agent 循环是核心优势。评测设计必须包含 multi-hop 数据集，并通过消融实验区分「Wiki 结构收益」与「增加模型调用收益」。

### 9.2 实施优先级



| 优先级 | Phase   | 核心产出                                  | 依赖                            |
| --- | ------- | ------------------------------------- | ----------------------------- |
| P0  | Phase 0 | WikiPage 数据模型                         | 无                             |
| P0  | Phase 1 | wiki\_search + wiki\_read + WikiIndex | Phase 0                       |
| P1  | Phase 2 | WikiCompiler（离线编译）                    | Phase 1                       |
| P1  | Phase 4 | WikiAgent（在线检索）                       | Phase 1, Phase 2（需要编译好的 Wiki） |
| P2  | Phase 3 | Error Book 自校正                        | Phase 2                       |
| P2  | Phase 5 | multi-hop 评测 + 成本分析                   | Phase 4                       |
| P3  | Phase 6 | 集成 + 文档                               | 全部                            |

> **关键路径**
>
> ：Phase 0 → Phase 1 → Phase 2 → Phase 4 → Phase 5。Error Book（Phase 3）可并行于 Phase 4，不阻塞 Agent 实现。

### 9.3 等待人工确认的决策点



1. WikiPage 存储格式：Python dataclass+JSON vs Markdown 文件

2. 编译 LLM 与 Agent LLM 是否分离实例

3. Agent tool-calling 实现：OpenAI tools API vs 手动 JSON 解析

4. Error Book 持久化格式：JSON vs YAML

5. 代码组织：`zmwiki/wiki/` 子包 vs 独立 `zmwiki` 包

6. multi-hop 评测数据集来源：手工构造 vs 采样公开数据集

7. 是否将 `SemanticEmbeddingProvider` 加入公开 API



***

*本报告为只读审计结果，未执行任何代码修改。所有结论均附源码证据；标注「待核实」的内容需要进一步确认。等待人工确认后进入实施阶段。*