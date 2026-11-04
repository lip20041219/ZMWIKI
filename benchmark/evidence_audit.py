"""Offline claim-level evidence audit (EvidenceAudit v0).

This module is deliberately independent of WikiAgent and of the benchmark
scorer. It answers one question only:

    Does the answer text contain claims that are supported by, contradicted by,
    or unverifiable from the pages that the answer actually cited?

It does not call a model, does not use case annotations as evidence, and does
not modify any existing metric. Case annotations may be used by callers/tests
to describe expected outcomes, but ``audit_answer`` ignores them.

Verdicts
--------
supported
    A source fact with the same subject/relation/attribute/value exists.
contradicted
    A source fact explicitly assigns a different value/object to the same
    single-valued subject/relation, or explicitly negates the claimed fact.
unverifiable
    No supporting or contradicting source fact can be established. Absence of
    support is NOT treated as contradiction.

The v0 extractor is intentionally conservative: complex or paraphrased claims
that cannot be parsed are ``unverifiable`` rather than guessed.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Iterable, Mapping, Sequence

__all__ = [
    "AuditClaim",
    "AuditResult",
    "audit_answer",
    "SUPPORTED",
    "CONTRADICTED",
    "UNVERIFIABLE",
]

SUPPORTED = "supported"
CONTRADICTED = "contradicted"
UNVERIFIABLE = "unverifiable"

_ALL_FEATURES = frozenset({"paraphrase", "clause_split", "dimensions"})


def _enabled_features(features: Sequence[str] | None) -> frozenset[str]:
    if features is None:
        return _ALL_FEATURES
    unknown = set(features) - _ALL_FEATURES
    if unknown:
        raise ValueError(f"unknown audit features: {sorted(unknown)}")
    return frozenset(features)

#: Relations whose value is treated as single-valued in this corpus.
_SINGLE_VALUED = {
    "reports_to",
    "office",
    "team_size",
    "direct_report_count",
    "director_count",
    "project_member_count",
    "budget",
    "joined_year",
    "role_since",
}

_GENERIC_ALIAS_TOKENS = {
    "acme", "corp", "director", "engineering", "engineer", "vp",
    "president", "company", "chief",
}

_REFUSAL_RE = re.compile(
    r"\b(?:i do not know|i don't know|not stated|not provided|not named|"
    r"could not confirm|cannot confirm|does not (?:give|state|provide|name)|"
    r"no page|not available|not listed|insufficient|unknown|not found)\b"
)

_NEGATION_RE = re.compile(
    r"\b(?:not|never|does not|doesn't|did not|didn't|no longer|without)\b"
)

_NUMBER_WORDS = {
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
    "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
    "twenty": 20,
}

_FLOOR_WORDS = ("first", "second", "third", "fourth", "fifth")


@dataclass(frozen=True)
class AuditClaim:
    claim_id: str
    text: str
    subject: str | None
    relation: str | None
    object: str | None
    value: str | None
    verdict: str
    supporting_page_ids: tuple[str, ...]
    contradicting_page_ids: tuple[str, ...]
    reason: str
    negated: bool = False

    def to_dict(self) -> dict:
        return {
            "claim_id": self.claim_id,
            "text": self.text,
            "subject": self.subject,
            "relation": self.relation,
            "object": self.object,
            "value": self.value,
            "verdict": self.verdict,
            "negated": self.negated,
            "supporting_page_ids": list(self.supporting_page_ids),
            "contradicting_page_ids": list(self.contradicting_page_ids),
            "reason": self.reason,
        }


@dataclass(frozen=True)
class AuditResult:
    answer: str | None
    claims: tuple[AuditClaim, ...]
    claim_support_rate: float
    claim_contradiction_rate: float
    citation_claim_precision: float
    evidence_verdict: str
    audit_coverage: float
    valid_source_ids: tuple[str, ...]
    invalid_source_ids: tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "answer": self.answer,
            "claims": [c.to_dict() for c in self.claims],
            "claim_support_rate": self.claim_support_rate,
            "claim_contradiction_rate": self.claim_contradiction_rate,
            "citation_claim_precision": self.citation_claim_precision,
            "evidence_verdict": self.evidence_verdict,
            "audit_coverage": self.audit_coverage,
            "valid_source_ids": list(self.valid_source_ids),
            "invalid_source_ids": list(self.invalid_source_ids),
        }


@dataclass(frozen=True)
class _Fact:
    subject: str
    relation: str
    object: str | None
    value: str | None
    polarity: bool
    page_id: str
    sentence: str


def _normalize(text: str) -> str:
    text = text.lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9\s']+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _page_content(page: object) -> str:
    if isinstance(page, str):
        return page
    content = getattr(page, "content", None)
    if not isinstance(content, str):
        raise TypeError("pages values must be str or objects with .content")
    return content


def _build_alias_index(pages: Mapping[str, object]) -> dict[str, str]:
    """Map normalized entity aliases to canonical page ids.

    Aliases come from the page id, its id words, and capitalized multi-token
    sequences in the page's first sentence. Generic organization/role phrases
    are excluded; ambiguous aliases are dropped.
    """
    raw: dict[str, set[str]] = {}
    for page_id, page in pages.items():
        text = _page_content(page)
        aliases = {page_id.lower(), page_id.replace("_", " ").lower()}
        first = _sentences(text)[0] if _sentences(text) else ""
        for seq in re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b", first):
            if len(seq.split()) < 2:
                continue
            if set(seq.lower().split()) & _GENERIC_ALIAS_TOKENS:
                continue
            aliases.add(seq.lower())
        for alias in aliases:
            alias = alias.strip()
            if alias:
                raw.setdefault(alias, set()).add(page_id)
    index: dict[str, str] = {}
    for alias, ids in raw.items():
        if len(ids) == 1:
            index[alias] = next(iter(ids))
    index.setdefault("ceo", "ceo")
    index.setdefault("chief executive officer", "ceo")
    index.setdefault("cto", "cto")
    index.setdefault("chief technology officer", "cto")
    return index


def _entity_mentions(text: str, aliases: Mapping[str, str]) -> list[tuple[int, int, str]]:
    """Return non-overlapping (start, end, canonical) entity mentions."""
    norm = _normalize(text)
    hits: list[tuple[int, int, str]] = []
    for alias, canonical in sorted(aliases.items(), key=lambda kv: -len(kv[0])):
        pattern = r"(?<!\w)" + re.escape(_normalize(alias)) + r"(?:'s|s)?(?!\w)"
        for match in re.finditer(pattern, norm):
            hits.append((match.start(), match.end(), canonical))
    hits.sort(key=lambda h: (h[0], -(h[1] - h[0])))
    chosen: list[tuple[int, int, str]] = []
    for hit in hits:
        if any(not (hit[1] <= c[0] or hit[0] >= c[1]) for c in chosen):
            continue
        chosen.append(hit)
    return chosen


def _first_entity(text: str, aliases: Mapping[str, str]) -> str | None:
    mentions = _entity_mentions(text, aliases)
    return mentions[0][2] if mentions else None


def _role_subject(text: str, aliases: Mapping[str, str], phrase: str) -> str | None:
    """Pick the person nearest a role phrase; ignore projects/buildings."""
    norm = _normalize(text)
    pos = norm.find(_normalize(phrase))
    if pos < 0:
        return _first_entity(text, aliases)
    candidates = [
        (abs(start - pos), start, canonical)
        for start, _, canonical in _entity_mentions(text, aliases)
        if not canonical.startswith(("project_", "building_"))
        and canonical not in ("ceo", "cto")
    ]
    return min(candidates)[2] if candidates else _first_entity(text, aliases)


def _last_entity(text: str, aliases: Mapping[str, str]) -> str | None:
    mentions = _entity_mentions(text, aliases)
    return mentions[-1][2] if mentions else None


def _number(value: str) -> int | None:
    value = _normalize(value)
    if value.isdigit():
        return int(value)
    return _NUMBER_WORDS.get(value)


def _money(value: str) -> str | None:
    """Normalize common money phrases to a comparable string."""
    norm = _normalize(value)
    digits = re.sub(r"\D", "", norm)
    if digits:
        return str(int(digits))
    words = {
        "one million": 1_000_000, "two million": 2_000_000,
        "three million": 3_000_000, "four million": 4_000_000,
        "five million": 5_000_000,
        "one hundred thousand": 100_000,
        "two hundred thousand": 200_000,
        "three hundred thousand": 300_000,
        "four hundred thousand": 400_000,
        "five hundred thousand": 500_000,
        "eight hundred thousand": 800_000,
        "nine hundred thousand": 900_000,
    }
    for phrase, number in words.items():
        if phrase in norm:
            return str(number)
    return None


def _attribute_keys(text: str) -> tuple[str, ...]:
    norm = _normalize(text)
    keys: list[str] = []
    if "budget" in norm:
        keys.append("budget")
    if "roadmap" in norm:
        keys.append("roadmap")
    if "quarterly" in norm or "performance review" in norm:
        keys.append("quarterly_reviews")
    if "vision" in norm:
        keys.append("vision")
    if not keys:
        keys.append(norm)
    return tuple(dict.fromkeys(keys))


def _polarity(sentence: str) -> bool:
    return not bool(_NEGATION_RE.search(_normalize(sentence)))


def _has_neg(text: str) -> bool:
    return bool(_NEGATION_RE.search(_normalize(text)))


def _negated_after_last_entity(text: str, aliases: Mapping[str, str]) -> bool:
    """Negation that scopes to the local subject, not an earlier clause."""
    mentions = _entity_mentions(text, aliases)
    if mentions:
        return _has_neg(text[mentions[-1][1]:])
    return _has_neg(text)


def _source_facts(
    page_id: str, content: str, aliases: Mapping[str, str],
    features: frozenset[str],
) -> list[_Fact]:
    facts: list[_Fact] = []
    for raw in _sentences(content):
        s = _normalize(raw)
        mentions = _entity_mentions(s, aliases)
        subject: str | None = None
        if re.match(r"^(he|she|his|her|it)\b", s):
            subject = page_id
        elif mentions:
            subject = mentions[0][2]
        if subject is None:
            continue

        # reports_to
        m = re.search(r"\breports?\s+(?:directly\s+)?to\s+(.+)$", s)
        if m:
            tail = m.group(1)
            obj = "ceo" if "ceo" in tail else _first_entity(tail, aliases)
            if obj:
                facts.append(_Fact(subject, "reports_to", obj, None, _polarity(s), page_id, raw))
        if "paraphrase" in features:
            for pattern in (
                r"\bworks under\s+(.+)$",
                r"\bis managed by\s+(.+)$",
                r"\bis under\s+(.+)$",
            ):
                m = re.search(pattern, s)
                if m:
                    obj = _first_entity(m.group(1), aliases)
                    if obj:
                        facts.append(_Fact(subject, "reports_to", obj, None, _polarity(s), page_id, raw))
                        break

        # manages / team size / direct reports
        manage_match = re.search(r"\bmanages?\b", s)
        if manage_match:
            tail = s[manage_match.end():]
            for _, _, obj in _entity_mentions(tail, aliases):
                if obj != subject:
                    facts.append(_Fact(subject, "manages", obj, None, _polarity(s), page_id, raw))
            tm = re.search(r"team of\s+([a-z0-9]+)\s+engineers", s)
            if tm and _number(tm.group(1)) is not None:
                facts.append(_Fact(subject, "team_size", None, str(_number(tm.group(1))), _polarity(s), page_id, raw))
            if "dimensions" in features:
                dm = re.search(r"manages?\s+([a-z0-9]+)\s+directors", s)
                if dm and _number(dm.group(1)) is not None:
                    facts.append(_Fact(subject, "director_count", None, str(_number(dm.group(1))), _polarity(s), page_id, raw))

        if "dimensions" in features:
            dc = re.search(r"has\s+([a-z0-9]+)\s+direct reports", s)
            if dc and _number(dc.group(1)) is not None:
                facts.append(_Fact(subject, "direct_report_count", None, str(_number(dc.group(1))), _polarity(s), page_id, raw))
            rc = re.search(r"([a-z0-9]+)\s+(?:employees|people)\s+report", s)
            if rc and _number(rc.group(1)) is not None:
                facts.append(_Fact(subject, "direct_report_count", None, str(_number(rc.group(1))), _polarity(s), page_id, raw))
        else:
            # v0.1 compatibility: direct-report counts share the team-size slot.
            dc = re.search(r"has\s+([a-z0-9]+)\s+direct reports", s)
            if dc and _number(dc.group(1)) is not None:
                facts.append(_Fact(subject, "team_size", None, str(_number(dc.group(1))), _polarity(s), page_id, raw))

        if "paraphrase" in features:
            for pattern, relation in (
                (r"\bis a member of\s+(.+)$", "member_of"),
                (r"\bis part of\s+(.+)$", "member_of"),
                (r"\bcontributes to\s+(.+)$", "member_of"),
                (r"\bworks on\s+(.+)$", "member_of"),
            ):
                m = re.search(pattern, s)
                if m:
                    obj = _first_entity(m.group(1), aliases)
                    if obj:
                        facts.append(_Fact(subject, relation, obj, None, _polarity(s), page_id, raw))
                        break
            if "core team member" in s or "contributing" in s:
                for _, _, person in mentions:
                    if person != page_id:
                        facts.append(_Fact(person, "member_of", page_id, None, _polarity(s), page_id, raw))

        # role / title
        if "paraphrase" in features:
            for phrase, role in _ROLE_MAP.items():
                if phrase in s:
                    role_subject = _role_subject(s, aliases, phrase) or subject
                    facts.append(_Fact(role_subject, "role", role, None, _polarity(s), page_id, raw))

        # responsible paraphrases
        if "paraphrase" in features:
            for pattern in (
                r"\bis in charge of\s+(.+)$",
                r"\bhandles\s+(.+)$",
                r"\bis accountable for\s+(.+)$",
                r"\btakes care of\s+(.+)$",
            ):
                m = re.search(pattern, s)
                if m:
                    for key in _attribute_keys(m.group(1)):
                        facts.append(_Fact(subject, "responsible_for", key, key, _polarity(s), page_id, raw))
                    break

        # leads (active/passive)
        lead_match = re.search(
            r"(?<!technical )(?<!project )(?<!team )\bleads?\b", s
        )
        if lead_match:
            tail = s[lead_match.end():]
            head = " ".join(tail.split()[:5])
            obj = _first_entity(head, aliases) or page_id
            facts.append(_Fact(subject, "leads", obj, None, _polarity(s), page_id, raw))
        if "paraphrase" in features:
            m = re.search(r"\bheads\s+(.+)$", s)
            if m:
                obj = _first_entity(m.group(1), aliases)
                if obj:
                    facts.append(_Fact(subject, "leads", obj, None, _polarity(s), page_id, raw))
        m = re.search(r"\bis led by\s+(.+)$", s)
        if m:
            lead = _first_entity(m.group(1), aliases)
            if lead:
                facts.append(_Fact(lead, "leads", page_id, None, _polarity(s), page_id, raw))

        # responsibility / ownership
        for pattern, passive in (
            (r"\bis responsible for\s+(.+)$", False),
            (r"\nowns?\s+(.+)$", False),
            (r"^(.+?)\s+(?:is|are)\s+(?:also\s+)?(?:owned|run) by\s+(.+)$", True),
        ):
            m = re.search(pattern, s)
            if not m:
                continue
            if passive:
                attr_text, subj_tail = m.group(1), m.group(2)
                subj = _first_entity(subj_tail, aliases)
            else:
                attr_text, subj = m.group(1), subject
            if not subj:
                continue
            for key in _attribute_keys(attr_text):
                facts.append(_Fact(subj, "responsible_for", key, key, _polarity(s), page_id, raw))

        # office floor/building
        office_re = re.compile(
            rf"office is (?:located )?(?:on|in) (?:the )?({'|'.join(_FLOOR_WORDS)}) floor of (building [ab])"
            rf"|office is in (building [ab]) on the ({'|'.join(_FLOOR_WORDS)}) floor"
        )
        for om in office_re.finditer(s):
            if om.group(1):
                floor, building = om.group(1), om.group(2)
            else:
                building, floor = om.group(3), om.group(4)
            prefix = s[:om.start()]
            office_subject = _last_entity(prefix, aliases) or subject
            if office_subject:
                facts.append(_Fact(
                    office_subject, "office", f"{floor}|{building}", None,
                    not _has_neg(prefix), page_id, raw,
                ))

        # team size without manages
        tm = re.search(r"team of\s+([a-z0-9]+)\s+engineers", s)
        if tm and _number(tm.group(1)) is not None:
            facts.append(_Fact(subject, "team_size", None, str(_number(tm.group(1))), _polarity(s), page_id, raw))

        # budget
        bm = re.search(r"(?:annual engineering budget|budget) (?:of|is)\s+\$?\s*([\d][\d\s,]*)", s)
        if not bm:
            bm = re.search(r"(?:annual engineering budget|budget) (?:of|is)\s+(.+?)\s+dollars", s)
        if bm:
            amount = _money(bm.group(1))
            budget_subject = _last_entity(s[:bm.start()], aliases) or subject
            if amount and budget_subject:
                facts.append(_Fact(budget_subject, "budget", None, amount, not _has_neg(s[:bm.start()]), page_id, raw))

        # years
        for pattern, relation in (
            (r"\bsince\s+(\d{4})\b", "role_since"),
            (r"\bjoined .*?\bin\s+(\d{4})\b", "joined_year"),
            (r"\bheld since\s+(\d{4})\b", "role_since"),
        ):
            ym = re.search(pattern, s)
            if ym:
                facts.append(_Fact(subject, relation, None, ym.group(1), _polarity(s), page_id, raw))

    return facts


def _split_clauses(raw: str) -> list[str]:
    """Split compound answer sentences without splitting entity lists."""
    parts = re.split(
        r";\s*|\s+(?:but|while)\s+|\s+and\s+(?=(?:he|she|they|it|also|"
        r"reports|manages|leads|is|are|has|owns|works|joined|handles|heads|"
        r"responsible|serves|does)\b)",
        raw,
        flags=re.IGNORECASE,
    )
    return [p.strip() for p in parts if p.strip()]


def _answer_claims(
    claim_id_start: int, answer: str, aliases: Mapping[str, str],
    features: frozenset[str],
) -> list[AuditClaim]:
    claims: list[AuditClaim] = []
    n = claim_id_start
    seen: set[tuple] = set()

    def add(text: str, subject: str | None, relation: str | None,
            obj: str | None, value: str | None, reason: str = "",
            negated: bool | None = None) -> None:
        nonlocal n
        key = (_normalize(text), subject, relation, obj, value)
        if key in seen:
            return
        seen.add(key)
        claims.append(AuditClaim(
            claim_id=f"C{n}", text=text, subject=subject, relation=relation,
            object=obj, value=value, verdict=UNVERIFIABLE,
            supporting_page_ids=(), contradicting_page_ids=(),
            reason=reason or "no matching source fact",
            negated=(not _polarity(text)) if negated is None else negated,
        ))
        n += 1

    last_subject: str | None = None
    previous_mention_count = 0
    units: list[str] = []
    for raw in _sentences(answer):
        if "clause_split" in features:
            units.extend(_split_clauses(raw))
        else:
            units.append(raw)
    for raw in units:
        s = _normalize(raw)
        if _REFUSAL_RE.search(s):
            continue
        mentions = _entity_mentions(s, aliases)
        subject = mentions[0][2] if mentions else None
        pronoun = bool(re.match(r"^(he|she|his|her|they)\b", s))
        pronoun_ambiguous = pronoun and previous_mention_count > 1
        verb_start = bool(re.match(
            r"^(reports|manages|leads|is|are|has|owns|works|joined|handles|"
            r"heads|serves|does)\b", s
        ))
        if verb_start and last_subject:
            subject = last_subject
        elif not mentions and last_subject and verb_start:
            subject = last_subject
        if pronoun and last_subject:
            subject = None if pronoun_ambiguous else last_subject
        if pronoun_ambiguous:
            last_subject = None
        if mentions:
            last_subject = mentions[0][2]
        previous_mention_count = len(mentions)

        # direct reports + team size
        m = re.search(r"(.+?)\s+has\s+([a-z0-9]+)\s+direct reports(?:\s*:?\s*(.*))?$", s)
        if m:
            subj = _last_entity(m.group(1), aliases) or (None if pronoun else subject)
            count = _number(m.group(2))
            if subj and count is not None:
                relation = (
                    "direct_report_count" if "dimensions" in features
                    else "team_size"
                )
                add(raw, subj, relation, None, str(count),
                    negated=_negated_after_last_entity(m.group(1), aliases))
                raw_tail = raw.split("direct reports", 1)[1] if "direct reports" in raw else ""
                for _, _, obj in _entity_mentions(raw_tail, aliases):
                    if obj != subj:
                        add(raw, subj, "direct_report", obj, None,
                            negated=_negated_after_last_entity(m.group(1), aliases))
        # reports to
        m = re.search(r"(.+?)\s+reports?\s+(?:directly\s+)?to\s+(.+)$", s)
        if m:
            subj = _last_entity(m.group(1), aliases) or (None if pronoun else subject)
            tail = m.group(2)
            obj = _first_entity(tail, aliases)
            if subj and obj:
                add(raw, subj, "reports_to", obj, None,
                    negated=_negated_after_last_entity(m.group(1), aliases))
        else:
            m = re.match(r"^reports?\s+(?:directly\s+)?to\s+(.+)$", s)
            if m and subject:
                obj = _first_entity(m.group(1), aliases)
                if obj:
                    add(raw, subject, "reports_to", obj, None, negated=_has_neg(s))
        if "paraphrase" in features:
            for pattern in (
                r"(.+?)\s+works under\s+(.+)$",
                r"(.+?)\s+is managed by\s+(.+)$",
                r"(.+?)\s+is under\s+(.+)$",
            ):
                m = re.search(pattern, s)
                if m:
                    subj = _last_entity(m.group(1), aliases) or (None if pronoun else subject)
                    obj = _first_entity(m.group(2), aliases)
                    if subj and obj:
                        add(raw, subj, "reports_to", obj, None,
                            negated=_negated_after_last_entity(m.group(1), aliases))
                        break
        if "paraphrase" in features:
            m = re.search(r"([a-z0-9]+)\s+(?:employees|people)\s+report\s+(?:directly\s+)?to\s+(.+)$", s)
            if m:
                subj = _first_entity(m.group(2), aliases)
                count = _number(m.group(1))
                if subj and count is not None:
                    add(raw, subj, "direct_report_count", None, str(count))
            m = re.search(r"(.+?)\s+has\s+([a-z0-9]+)\s+(?:employees|people)\s+under", s)
            if m:
                subj = _last_entity(m.group(1), aliases) or (None if pronoun else subject)
                count = _number(m.group(2))
                if subj and count is not None:
                    add(raw, subj, "direct_report_count", None, str(count))
        # manages / team of N
        m = re.search(r"(.+?)\s+(?:manages?|has)\s+(?:a\s+)?team of\s+([a-z0-9]+)\s+engineers", s)
        if m:
            subj = _last_entity(m.group(1), aliases) or (None if pronoun else subject)
            count = _number(m.group(2))
            if subj and count is not None:
                add(raw, subj, "team_size", None, str(count),
                    negated=_negated_after_last_entity(m.group(1), aliases))
        m = re.search(r"(.+?)\s+team has\s+([a-z0-9]+)\s+engineers", s)
        if m:
            subj = _last_entity(m.group(1), aliases) or (None if pronoun else subject)
            count = _number(m.group(2))
            if subj and count is not None:
                add(raw, subj, "team_size", None, str(count),
                    negated=_negated_after_last_entity(m.group(1), aliases))
        if "dimensions" in features:
            m = re.search(r"(.+?)\s+(?:manages?|oversees)\s+([a-z0-9]+)\s+directors", s)
            if m:
                subj = _last_entity(m.group(1), aliases) or (None if pronoun else subject)
                count = _number(m.group(2))
                if subj and count is not None:
                    add(raw, subj, "director_count", None, str(count),
                        negated=_negated_after_last_entity(m.group(1), aliases))
        # responsibility active
        m = re.search(r"(.+?)\s+is responsible for\s+(.+)$", s)
        if m:
            subj = _first_entity(m.group(1), aliases) or (None if pronoun else subject)
            for key in _attribute_keys(m.group(2)):
                add(raw, subj, "responsible_for", key, key, negated=_negated_after_last_entity(m.group(1), aliases))
        if "paraphrase" in features:
            for pattern in (
                r"(.+?)\s+is in charge of\s+(.+)$",
                r"(.+?)\s+handles\s+(.+)$",
                r"(.+?)\s+is accountable for\s+(.+)$",
            ):
                m = re.search(pattern, s)
                if m:
                    subj = _last_entity(m.group(1), aliases) or (None if pronoun else subject)
                    for key in _attribute_keys(m.group(2)):
                        add(raw, subj, "responsible_for", key, key,
                            negated=_negated_after_last_entity(m.group(1), aliases))
                    break
        # responsibility passive
        m = re.search(r"^(.+?)\s+(?:is|are)\s+(?:also\s+)?(?:owned|run) by\s+(.+)$", s)
        if m:
            subj = _first_entity(m.group(2), aliases)
            for key in _attribute_keys(m.group(1)):
                add(raw, subj, "responsible_for", key, key, negated=_negated_after_last_entity(m.group(1), aliases))
        # owns
        m = re.search(r"(.+?)\s+owns?\s+(.+)$", s)
        if m:
            subj = _first_entity(m.group(1), aliases) or (None if pronoun else subject)
            for key in _attribute_keys(m.group(2)):
                add(raw, subj, "responsible_for", key, key, negated=_negated_after_last_entity(m.group(1), aliases))
        if "paraphrase" in features:
            for pattern in (
                r"(.+?)\s+is a member of\s+(.+)$",
                r"(.+?)\s+is part of\s+(.+)$",
                r"(.+?)\s+contributes to\s+(.+)$",
                r"(.+?)\s+works on\s+(.+)$",
            ):
                m = re.search(pattern, s)
                if m:
                    subj = _last_entity(m.group(1), aliases) or (None if pronoun else subject)
                    obj = _first_entity(m.group(2), aliases)
                    if subj and obj:
                        add(raw, subj, "member_of", obj, None,
                            negated=_negated_after_last_entity(m.group(1), aliases))
                        break
        # role / title
        if "paraphrase" in features:
            for phrase, role in _ROLE_MAP.items():
                if phrase in s:
                    role_subject = _role_subject(s, aliases, phrase) or subject
                    add(raw, role_subject, "role", role, None)
        # leads active/passive
        m = re.search(r"(.+?)\s+(?<!technical )(?<!project )(?<!team )leads?\s+(.+)$", s)
        if m:
            subj = _last_entity(m.group(1), aliases) or (None if pronoun else subject)
            head = " ".join(m.group(2).split()[:5])
            obj = _first_entity(head, aliases)
            if subj and obj:
                add(raw, subj, "leads", obj, None, negated=_negated_after_last_entity(m.group(1), aliases))
        if "paraphrase" in features:
            m = re.search(r"(.+?)\s+heads\s+(.+)$", s)
            if m:
                subj = _first_entity(m.group(1), aliases) or (None if pronoun else subject)
                obj = _first_entity(m.group(2), aliases)
                if subj and obj:
                    add(raw, subj, "leads", obj, None,
                        negated=_negated_after_last_entity(m.group(1), aliases))
        m = re.search(r"(.+?)\s+is led by\s+(.+)$", s)
        if m:
            obj = _first_entity(m.group(1), aliases)
            subj = _first_entity(m.group(2), aliases)
            if subj and obj:
                add(raw, subj, "leads", obj, None, negated=_negated_after_last_entity(m.group(1), aliases))
        # office
        office_re = re.compile(
            rf"office is (?:located )?(?:on|in) (?:the )?({'|'.join(_FLOOR_WORDS)}) floor of (building [ab])"
            rf"|office is in (building [ab]) on the ({'|'.join(_FLOOR_WORDS)}) floor"
        )
        for om in office_re.finditer(s):
            if om.group(1):
                floor, building = om.group(1), om.group(2)
            else:
                building, floor = om.group(3), om.group(4)
            prefix = s[:om.start()]
            office_subject = _last_entity(prefix, aliases) or (None if pronoun else subject)
            if office_subject:
                add(raw, office_subject, "office", f"{floor}|{building}", None,
                    negated=_has_neg(prefix))
        # budget
        m = re.search(r"(?:annual engineering budget|budget) (?:of|is)\s+\$?\s*([\d][\d\s,]*)", s)
        if not m:
            m = re.search(r"(?:annual engineering budget|budget) (?:of|is)\s+(.+?)\s+dollars", s)
        if m:
            amount = _money(m.group(1))
            prefix = s[:m.start()]
            subj = _last_entity(prefix, aliases)
            if amount and subj:
                add(raw, subj, "budget", None, amount,
                    negated=_has_neg(prefix))
        # fallback: factual-looking sentence with entity but no parsed claim
        if not any(c.text == raw for c in claims):
            if mentions and re.search(r"\b(is|are|has|have|reports|managed|leads|located)\b", s):
                add(raw, subject, None, None, None)

    return claims


_VALUE_RELATIONS = {
    "team_size", "direct_report_count", "director_count",
    "project_member_count", "budget", "joined_year", "role_since",
}

_ROLE_MAP = {
    "director of engineering": "director_of_engineering",
    "vp of engineering": "vp_of_engineering",
    "vice president of engineering": "vp_of_engineering",
    "senior engineer": "senior_engineer",
    "staff engineer": "staff_engineer",
    "principal engineer": "principal_engineer",
    "technical lead": "technical_lead",
}


def _claim_target(claim: AuditClaim) -> str | None:
    return claim.value if claim.relation in _VALUE_RELATIONS else claim.object


def _fact_target(fact: _Fact, relation: str) -> str | None:
    return fact.value if relation in _VALUE_RELATIONS else fact.object


def _match_claim(claim: AuditClaim, facts: Sequence[_Fact], aliases: Mapping[str, str]) -> AuditClaim:
    if not claim.subject or not claim.relation:
        return claim
    if claim.relation in ("reports_to", "leads") and claim.subject == claim.object:
        return claim
    relevant = [f for f in facts if f.relation == claim.relation and f.subject == claim.subject]
    target = _claim_target(claim)

    if claim.relation == "responsible_for":
        key = claim.object or claim.value
        same = [f for f in relevant if f.object == key or f.value == key]
        good = [f for f in same if f.polarity == (not claim.negated)]
        if good:
            pages = tuple(sorted({f.page_id for f in good}))
            return AuditClaim(**{**claim.__dict__, "verdict": SUPPORTED,
                                 "supporting_page_ids": pages,
                                 "reason": "source fact assigns the same responsibility"})
        if claim.negated:
            positive_same = [f for f in same if f.polarity]
            if positive_same:
                pages = tuple(sorted({f.page_id for f in positive_same}))
                return AuditClaim(**{**claim.__dict__, "verdict": CONTRADICTED,
                                     "contradicting_page_ids": pages,
                                     "reason": "source assigns the responsibility the answer denies"})
            return claim
        conflicting = [f for f in facts if f.relation == "responsible_for"
                       and (f.object == key or f.value == key)
                       and f.subject != claim.subject and f.polarity]
        if conflicting:
            pages = tuple(sorted({f.page_id for f in conflicting}))
            return AuditClaim(**{**claim.__dict__, "verdict": CONTRADICTED,
                                 "contradicting_page_ids": pages,
                                 "reason": "source assigns this responsibility to another subject"})
        return claim

    if claim.relation in ("role", "member_of"):
        same = [f for f in relevant
                if f.object == claim.object and f.polarity]
        if same:
            pages = tuple(sorted({f.page_id for f in same}))
            return AuditClaim(**{**claim.__dict__, "verdict": SUPPORTED,
                                 "supporting_page_ids": pages,
                                 "reason": "source states the same role/membership"})
        return claim

    if claim.relation in (
        "reports_to", "office", "team_size", "direct_report_count",
        "director_count", "project_member_count", "budget", "leads",
        "direct_report",
    ):
        same = [f for f in relevant if _fact_target(f, claim.relation) == target]
        good = [f for f in same if f.polarity == (not claim.negated)]
        if good:
            pages = tuple(sorted({f.page_id for f in good}))
            return AuditClaim(**{**claim.__dict__, "verdict": SUPPORTED,
                                 "supporting_page_ids": pages,
                                 "reason": "source fact matches"})
        if claim.negated:
            positive_same = [f for f in same if f.polarity]
            if positive_same:
                pages = tuple(sorted({f.page_id for f in positive_same}))
                return AuditClaim(**{**claim.__dict__, "verdict": CONTRADICTED,
                                     "contradicting_page_ids": pages,
                                     "reason": "source states the fact the answer denies"})
            return claim
        if claim.relation == "leads":
            reverse = [f for f in facts if f.relation == "leads"
                       and f.subject == claim.object and f.object == claim.subject]
            if reverse:
                pages = tuple(sorted({f.page_id for f in reverse}))
                return AuditClaim(**{**claim.__dict__, "verdict": SUPPORTED,
                                     "supporting_page_ids": pages,
                                     "reason": "source states the reverse leadership pair"})
            other_lead = [f for f in facts if f.relation == "leads"
                          and f.object == claim.object and f.subject != claim.subject]
            if other_lead:
                pages = tuple(sorted({f.page_id for f in other_lead}))
                return AuditClaim(**{**claim.__dict__, "verdict": CONTRADICTED,
                                     "contradicting_page_ids": pages,
                                     "reason": "source names a different lead"})
        if claim.relation in _SINGLE_VALUED:
            different = [f for f in relevant
                         if _fact_target(f, claim.relation) != target
                         and f.polarity]
            if different:
                pages = tuple(sorted({f.page_id for f in different}))
                return AuditClaim(**{**claim.__dict__, "verdict": CONTRADICTED,
                                     "contradicting_page_ids": pages,
                                     "reason": "source gives a different value for this single-valued relation"})
        if claim.relation == "direct_report":
            managed = [f for f in facts if f.relation == "manages"
                       and f.subject == claim.subject and f.object == claim.object]
            if managed:
                pages = tuple(sorted({f.page_id for f in managed}))
                return AuditClaim(**{**claim.__dict__, "verdict": SUPPORTED,
                                     "supporting_page_ids": pages,
                                     "reason": "source states this management pair"})
            relation = [f for f in facts if f.relation == "reports_to" and f.subject == claim.object
                        and f.polarity]
            if relation and all(f.object != claim.subject for f in relation):
                pages = tuple(sorted({f.page_id for f in relation}))
                return AuditClaim(**{**claim.__dict__, "verdict": CONTRADICTED,
                                     "contradicting_page_ids": pages,
                                     "reason": "source reports this person to another manager"})
        return claim
    return claim


def audit_answer(
    answer: str | None,
    sources: Sequence[str],
    read_page_ids: Sequence[str],
    pages: Mapping[str, object],
    *,
    features: Sequence[str] | None = None,
) -> AuditResult:
    """Audit ``answer`` against the cited, actually-read source pages.

    ``pages`` maps page id -> content string or WikiPage-like object.
    ``sources`` are cited page ids; ``read_page_ids`` are pages actually read.
    A source that was not read is invalid and cannot support a claim.
    """
    read = {str(p) for p in read_page_ids}
    cited = [str(s) for s in sources]
    valid_sources = tuple(dict.fromkeys(s for s in cited if s in pages and s in read))
    invalid_sources = tuple(s for s in cited if s not in valid_sources)

    if answer is None or not answer.strip():
        return AuditResult(None, (), 0.0, 0.0, 0.0, "no_answer", 0.0, valid_sources, invalid_sources)

    enabled = _enabled_features(features)
    aliases = _build_alias_index(pages)
    facts: list[_Fact] = []
    for page_id in valid_sources:
        facts.extend(_source_facts(
            page_id, _page_content(pages[page_id]), aliases, enabled,
        ))

    claims = _answer_claims(0, answer, aliases, enabled)
    audited = [_match_claim(c, facts, aliases) for c in claims]
    total = len(audited)
    supported = sum(1 for c in audited if c.verdict == SUPPORTED)
    contradicted = sum(1 for c in audited if c.verdict == CONTRADICTED)
    checkable = supported + contradicted
    support_rate = supported / total if total else 0.0
    contradiction_rate = contradicted / total if total else 0.0
    precision = supported / checkable if checkable else 0.0
    coverage = checkable / total if total else 0.0

    if not valid_sources:
        verdict = "no_evidence"
    elif not audited:
        verdict = "unverifiable"
    elif contradicted:
        verdict = "contradicted"
    elif supported == total:
        verdict = "supported"
    elif supported:
        verdict = "partially_supported"
    else:
        verdict = "unverifiable"

    return AuditResult(
        answer=answer, claims=tuple(audited),
        claim_support_rate=support_rate,
        claim_contradiction_rate=contradiction_rate,
        citation_claim_precision=precision,
        evidence_verdict=verdict,
        audit_coverage=coverage,
        valid_source_ids=valid_sources,
        invalid_source_ids=invalid_sources,
    )
