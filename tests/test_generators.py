"""Generator tests: Generator contract, DeterministicGenerator, build_prompt."""

import pytest

from zmrag import (
    Chunk,
    Context,
    DeterministicGenerator,
    Generator,
    SimpleTokenizer,
    build_prompt,
)


def _chunk(cid, text):
    return Chunk(id=cid, document_id="doc", text=text, position=0)


def _context(*entries):
    chunks = tuple(_chunk(cid, text) for cid, text in entries)
    text = "\n\n".join(f"[{i + 1}] {c.text}" for i, c in enumerate(chunks))
    return Context(text=text, chunks=chunks)


def _generate(query, context, **kw):
    return DeterministicGenerator(**kw).generate(query, context)


# --- contract shape ---


def test_generator_is_abstract():
    with pytest.raises(TypeError):
        Generator()  # type: ignore[abstract]
    assert isinstance(DeterministicGenerator(), Generator)


def test_deterministic_defaults_and_tokenizer_injection():
    gen = DeterministicGenerator()
    assert isinstance(gen.tokenizer, SimpleTokenizer)
    assert DeterministicGenerator(tokenizer=SimpleTokenizer()).tokenizer is not None
    with pytest.raises(TypeError):
        DeterministicGenerator(tokenizer="nope")  # type: ignore[arg-type]


# --- validation ---


@pytest.mark.parametrize("bad", ["", "   \n\t", 42, None, ["q"]])
def test_invalid_query_rejected(bad):
    with pytest.raises(TypeError):
        _generate(bad, _context(("A", "pie")))  # type: ignore[arg-type]


@pytest.mark.parametrize("bad", [None, "ctx", 42, ("A", "pie")])
def test_context_must_be_a_context(bad):
    with pytest.raises(TypeError):
        _generate("pie", bad)  # type: ignore[arg-type]


# --- generation ---


def test_extracts_the_best_matching_chunk_verbatim():
    ctx = _context(
        ("A", "banana split"),
        ("B", "apple pie recipe"),
        ("C", "apple sauce"),
    )
    assert _generate("apple pie", ctx) == "apple pie recipe"


def test_ties_keep_the_upstream_ranked_order():
    # Both overlap "apple" once -> the earlier (better-ranked) chunk wins.
    ctx = _context(("first", "apple crumble"), ("second", "apple tart"))
    assert _generate("apple", ctx) == "apple crumble"


def test_answer_is_verbatim_context_text_never_fabricated():
    ctx = _context(("A", "banana"), ("B", "apple"))
    answer = _generate("apple", ctx)
    assert answer in [c.text for c in ctx.chunks]
    # No prompt scaffolding, no marker, no invented sentence -- it is a copy.
    assert answer == "apple"


def test_answer_tracks_the_query():
    ctx = _context(("A", "apple pie"), ("B", "banana bread"))
    assert _generate("banana", ctx) == "banana bread"
    assert _generate("apple", ctx) == "apple pie"


def test_no_overlap_falls_back_to_the_first_chunk():
    ctx = _context(("A", "zzz"), ("B", "qqq"))
    assert _generate("apple", ctx) == "zzz"


def test_empty_context_yields_empty_answer():
    assert _generate("apple", Context(text="", chunks=())) == ""


def test_context_is_never_mutated():
    ctx = _context(("A", "apple"), ("B", "banana"))
    before = (ctx.text, ctx.chunks)
    _generate("apple", ctx)
    assert (ctx.text, ctx.chunks) == before


def test_deterministic_across_calls_and_instances():
    ctx = _context(("A", "apple pie"), ("B", "apple"))
    gen = DeterministicGenerator()
    assert gen.generate("apple", ctx) == gen.generate("apple", ctx)
    assert gen.generate("apple", ctx) == DeterministicGenerator().generate("apple", ctx)


def test_not_a_language_model_no_shape_beyond_the_contract():
    # The stand-in must not pretend to be an LLM: no sampling knobs, no
    # temperature/max_tokens/model attributes, no network client.
    gen = DeterministicGenerator()
    for attr in ("model", "model_name", "temperature", "max_tokens", "client"):
        assert not hasattr(gen, attr)


# --- prompt construction ---


def test_prompt_delimits_context_and_question():
    prompt = build_prompt("apple pie?", _context(("A", "apple pie recipe")))
    assert "<context>" in prompt and "</context>" in prompt
    assert "<question>" in prompt and "</question>" in prompt
    assert "apple pie recipe" in prompt  # context text, verbatim
    assert prompt.index("<context>") < prompt.index("<question>")  # evidence first
    assert "apple pie?" in prompt


def test_prompt_uses_context_text_verbatim_including_numbering():
    ctx = _context(("A", "first passage"), ("B", "second passage"))
    prompt = build_prompt("q", ctx)
    assert ctx.text in prompt


def test_prompt_with_empty_context_keeps_an_empty_context_section():
    prompt = build_prompt("q", Context(text="", chunks=()))
    assert "<context>\n\n</context>" in prompt


def test_prompt_is_deterministic_and_validated():
    ctx = _context(("A", "apple"))
    assert build_prompt("q", ctx) == build_prompt("q", ctx)
    with pytest.raises(TypeError):
        build_prompt("   ", ctx)  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        build_prompt("q", "not a context")  # type: ignore[arg-type]


# --- public API ---


def test_public_import():
    from zmrag import __all__

    assert {"Generator", "DeterministicGenerator", "build_prompt"} <= set(__all__)
    assert isinstance(DeterministicGenerator(), Generator)
