# Claim-Level Attribution Benchmark

Evidence: `benchmark/results/answer_attribution_profile.json`
Runner: `benchmark/run_attribution_profile.py`

## 1. Motivation

Retrieval evaluation says whether the right context was found. Answer
evaluation says whether the answer covers the reference and stays grounded in
that context. Neither says *which source carries which sentence* -- and neither
distinguishes "the answer is supported by the evidence" from "the answer's
citations actually point at that evidence".

Those are different failures with the same symptom. An answer can be fully
supported by the retrieved context while carrying no citations at all, and it
can cite a source that does not contain a word of the claim. This benchmark
measures that gap with a deterministic, inspectable baseline.

**It is a baseline evaluator, not a semantic factuality judge.** Everything here
is token overlap and sentence splitting. See [Limitations](#7-limitations).

## 2. Claim extraction

`DeterministicClaimExtractor` splits the answer into sentences and emits one
`Claim` per non-empty sentence:

| Field | Meaning |
| --- | --- |
| `claim_id` | `claim-0`, `claim-1`, ... -- unique within the answer |
| `text` | the sentence as written, terminator included |
| `sentence_index` | position in the answer's sentence sequence |

Terminators are `.`, `!`, `?`, `;`, `。`, `！`, `？`, `；` and the line break. No
clause splitting, no subject/predicate analysis, no coreference resolution, no
LLM decomposition. An answer with two facts in one sentence is one claim.

```
"Redis uses an in-memory data structure. Redis supports persistence."
  -> claim-0: "Redis uses an in-memory data structure."
  -> claim-1: "Redis supports persistence."
```

Naive by design: `"3.5"` and `"e.g."` split too, which can only produce shorter,
more numerous sentences. That is a stated property of a baseline with no
sentence-boundary model.

## 3. Attribution definition

For each claim, over the same NFC + `SimpleTokenizer` path answer evaluation
uses:

```
claim_tokens  = unique tokens of the claim
support_score = |claim_tokens ∩ context_tokens| / |claim_tokens|      (0.0 if no tokens)
supported     = support_score >= threshold                            (default 0.5)

supporting_citation_ids = [citation.chunk_id for each citation
                           whose own coverage of the claim >= threshold]
attributed    = supporting_citation_ids is not empty
```

Rates: `supported_claims / total_claims` and `attributed_claims / total_claims`,
both `0.0` when there are no claims. `mean_support_score` is the mean of every
claim's `support_score`, so unsupported claims pull it down.

Two properties worth stating because they are what the two rates buy:

- **supported ≠ attributed.** Support is measured against the Context; attribution
  only against the Citations projected from it. With no citations a claim is
  supported and unattributed; with a citation that does not cover it, the same.
- **Multiple sources are all listed.** A claim under a citation whose own text
  covers it is attributed; if two citations do, both are kept, in citation input
  order, never re-sorted and never deduplicated by score.

## 4. Dataset

The answer dataset, unchanged and shared with the answer profile:
`benchmark/datasets/answer_small.json` -- 16 cases over the frozen Phase 10A
corpus (6 documents / 33 chunks), run through the canonical deterministic
pipeline at `top_k=5`, `candidate_k=33`.

Seven cases are the pipeline's own output. Nine are probes: they may carry
`answer_override` (an answer the generator would never produce) and
`citation_chunk_ids` (pin the citations to exactly these context chunks, in this
order; an empty list means no citations). Probe rows are labelled in the
evidence, and a pinned id that is not in the retrieved Context stops the run
rather than being silently dropped.

## 5. Results

Pooled over all 25 claims (a claim is the unit, so a three-claim answer weighs
three times a one-claim answer):

| Metric | Value |
| --- | ---: |
| total claims | 25 |
| supported claims | 24 |
| attributed claims | 21 |
| supported claim rate | 0.9600 |
| attributed claim rate | 0.8400 |
| mean support score | 0.9427 |

Where claims are lost on the way from "the answer said it" to "a cited source
carries it":

| Funnel step | Claims |
| --- | ---: |
| claims extracted | 25 |
| … supported by the context | 24 |
| … attributed to a citation | 21 |
| unsupported | 1 |
| **supported but uncited** | **3** |

The high support rate is a property of this generator, not of the metric: the
bundled `DeterministicGenerator` is extractive and returns a chunk's text
verbatim, so its sentences are literally in the context. A generative model
would produce a different profile, and this number should not be read as "96% of
answers are grounded" in any general sense.

## 6. Negative cases

| Probe | What it constructs | Result |
| --- | --- | --- |
| `n04_correct_without_sources` | covering answer, no citations | supported, **unattributed** -- the two rates separate |
| `p01_correct_wrong_citation` | covering answer cited to a chunk that does not contain it | supported, **unattributed** (`supporting_citation_ids: []`) |
| `p02_claim_supported_but_uncited` | two claims, one cited | claim 1 supported + attributed; claim 2 supported, unattributed |
| `p03_claim_with_two_sources` | one claim, two covering citations | both listed, `["doc_b::1", "doc_b::2"]` |
| `p04_wrong_but_supported_and_cited` | verbatim context sentences that answer a different question | both claims fully supported *and* attributed, reference coverage 0.33 |
| `n02_correct_but_ungrounded` | answer matching the reference, unrelated context | unsupported *and* unattributed, `support_score` 0.0 |
| `n05_negation_grounded` | `"LRU never evicts the least recently used entry."` against a source saying it does | **supported (0.88) and attributed to the source it contradicts** |

The last row is the limitation, kept as evidence: one word flips the meaning and
no token-overlap baseline can see it. `p04` is the same limitation from the
other side -- fully attributed, and still the wrong answer.

## 7. Limitations

- **No entailment.** Token overlap cannot distinguish a claim from its negation,
  and cannot tell a citation that *supports* a claim from one that merely shares
  its vocabulary.
- **No paraphrase.** A correct sentence with different words scores 0.0. No
  synonyms, no embeddings, no stemming.
- **Claim granularity is the sentence.** A sentence with two facts is one claim;
  a fragment is a claim of its own.
- **Function words count.** No stopword list, so `the` and `and` contribute to
  coverage. Removing them would change every number here, and the choice is
  recorded rather than tuned.
- **Source-level, not span-level.** A citation is credited for a claim if its
  whole text covers it; nothing points at an offset or a passage inside a chunk.
- **No LLM judge, no NLI, no embeddings, no automated fact checking.** Adding any
  of those would be a different layer behind the same value type, with its own
  optional dependency.

## 8. Relationship to Retrieval / Answer Evaluation

Four different questions, four different numbers, deliberately never merged:

| Layer | Question | Measure |
| --- | --- | --- |
| Retrieval evaluation | did we retrieve the right context? | hit rate, recall, MRR, MAP, NDCG |
| Answer evaluation | did the answer cover the reference and stay grounded? | reference coverage, groundedness, **source-level** citation coverage |
| Claim attribution (this) | is this claim supported, and by which citations? | supported / attributed claim rates, per-claim ids |

The path an answer takes, with the layer that measures each step:

```
Retrieval hit -> Context available -> Answer generated -> Answer grounded
              -> Claim supported -> Claim attributed to a source
```

Retrieval outcome is measured in the retrieval benchmark; groundedness in the
answer profile; the last two steps here. The profiles share one corpus, one
dataset and one pipeline, so the numbers describe one system -- but no layer's
score is an input to another's, and none of this is a complete fact-checking
pipeline.

## 9. Reproducibility

```bash
python benchmark/run_attribution_profile.py
python -m pytest -q
```

The runner builds its pipeline from scratch, executes the whole profile twice,
and writes evidence only when the two runs are identical -- the committed file
is byte-identical to a fresh run. It opens no socket, reads no key, loads no
model, and adds no dependency: `dependencies = []` still holds. The corpus, the
retrieval dataset and the frozen retrieval evidence are pinned by sha256 in the
test suite, so this benchmark cannot quietly rewrite history.
