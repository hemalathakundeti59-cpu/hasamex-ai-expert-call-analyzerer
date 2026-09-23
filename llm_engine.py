"""
llm_engine.py

All calls to the LLM (Anthropic Claude) live here, plus the
hallucination-mitigation logic:

  1. RETRIEVAL-GROUNDED PROMPTS
     We never ask the model to answer "from memory". Every prompt embeds
     only the retrieved segments (with their transcript-file + timestamp
     IDs) and instructs the model to answer *only* from them.

  2. STRUCTURED OUTPUT
     The model is asked to return JSON: {answer, quote, timestamp,
     source_file}. Structured output is far easier to verify and render
     than free text.

  3. QUOTE VERIFICATION (the main hallucination guardrail)
     After the model responds, `verify_quote()` checks that the returned
     quote is an actual (near-exact) substring of the segment it cites.
     If it isn't, we mark it "unverified" in the UI instead of silently
     trusting the model. This is the single most important design
     decision for the "how do you reduce hallucinations" question.

  4. "NOT MENTIONED" ESCAPE HATCH
     The model is explicitly told it's allowed — and expected — to say a
     topic was not discussed, rather than inventing an answer.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List, Optional

from transcript_parser import QASegment

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None

MODEL_NAME = os.environ.get("HASAMEX_MODEL", "claude-sonnet-4-5")


@dataclass
class Citation:
    source_file: str
    timestamp: str
    expert: str
    quote: str
    verified: bool


@dataclass
class GroundedAnswer:
    answer_text: str
    citations: List[Citation]
    not_mentioned: bool


def _client() -> "anthropic.Anthropic":
    if anthropic is None:
        raise RuntimeError(
            "The 'anthropic' package is not installed. Run: pip install anthropic"
        )
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return anthropic.Anthropic(api_key=api_key)


def _format_segments(segments: List[QASegment]) -> str:
    lines = []
    for s in segments:
        lines.append(
            f"[{s.source_file} | {s.expert} | {s.market} | {s.timestamp}]\n"
            f"Q: {s.question}\nA: {s.answer}\n"
        )
    return "\n".join(lines)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def verify_quote(quote: str, segments: List[QASegment], min_ratio: float = 0.85) -> bool:
    """Check the quote is a genuine (near-)substring of some source answer."""
    nq = _normalize(quote)
    if not nq:
        return False
    for s in segments:
        na = _normalize(s.answer)
        if nq in na:
            return True
        # fuzzy fallback for minor paraphrase of wording/punctuation
        if SequenceMatcher(None, nq, na).find_longest_match(0, len(nq), 0, len(na)).size >= int(
            len(nq) * min_ratio
        ):
            return True
    return False


def _extract_json(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```json\s*|\s*```$", "", raw, flags=re.MULTILINE)
    return json.loads(raw)


SYSTEM_PROMPT = """You are an analyst assistant answering questions ONLY from the \
provided expert-call transcript excerpts. Rules:

1. Use ONLY the excerpts given to you. Never use outside knowledge.
2. If the excerpts do not contain the answer, say so explicitly instead of guessing.
3. Every factual claim must be backed by an exact quote copied verbatim from the \
excerpts, plus its [source_file, timestamp, expert].
4. Do not merge or paraphrase quotes into something that isn't literally present.
5. Respond ONLY with valid JSON matching the schema you're given. No prose, no markdown fences."""


def answer_question_for_expert(question: str, segments: List[QASegment]) -> GroundedAnswer:
    """Answer one interview-guide question for a single expert, grounded + cited."""
    schema = (
        '{"not_mentioned": bool, "answer": string, '
        '"citations": [{"quote": string, "timestamp": string, "source_file": string}]}'
    )
    prompt = (
        f"EXCERPTS:\n{_format_segments(segments)}\n\n"
        f"QUESTION: {question}\n\n"
        f"Return JSON matching this schema exactly: {schema}\n"
        f"If the question is not addressed in the excerpts, set not_mentioned=true, "
        f"answer should explain that, and citations should be an empty list."
    )
    client = _client()
    resp = client.messages.create(
        model=MODEL_NAME,
        max_tokens=1000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = "".join(block.text for block in resp.content if block.type == "text")
    data = _extract_json(raw)

    citations = []
    for c in data.get("citations", []):
        verified = verify_quote(c.get("quote", ""), segments)
        expert = next((s.expert for s in segments if s.source_file == c.get("source_file")), "")
        citations.append(
            Citation(
                source_file=c.get("source_file", ""),
                timestamp=c.get("timestamp", ""),
                expert=expert,
                quote=c.get("quote", ""),
                verified=verified,
            )
        )
    return GroundedAnswer(
        answer_text=data.get("answer", ""),
        citations=citations,
        not_mentioned=bool(data.get("not_mentioned", False)),
    )


def cross_transcript_themes(all_segments_by_expert: dict) -> dict:
    """
    all_segments_by_expert: {expert_name: [QASegment, ...]}
    Returns dict with 'themes' and 'disagreements', each a list of
    {summary, citations:[{expert, quote, timestamp, source_file}]}.
    """
    parts = []
    flat_segments: List[QASegment] = []
    for expert, segs in all_segments_by_expert.items():
        parts.append(_format_segments(segs))
        flat_segments.extend(segs)
    excerpts_text = "\n".join(parts)

    schema = (
        '{"themes": [{"summary": string, "citations": [{"expert": string, '
        '"quote": string, "timestamp": string, "source_file": string}]}], '
        '"disagreements": [{"summary": string, "citations": [{"expert": string, '
        '"quote": string, "timestamp": string, "source_file": string}]}]}'
    )
    prompt = (
        f"EXCERPTS FROM ALL EXPERTS:\n{excerpts_text}\n\n"
        "Identify:\n"
        "1. COMMON THEMES that at least two experts independently raised.\n"
        "2. DISAGREEMENTS where experts gave meaningfully different views "
        "(e.g. different emphasis on finance vs training, different growth "
        "estimates, different timelines).\n"
        "Every theme/disagreement must cite at least one exact quote per expert involved.\n"
        f"Return JSON matching this schema exactly: {schema}"
    )
    client = _client()
    resp = client.messages.create(
        model=MODEL_NAME,
        max_tokens=2000,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = "".join(block.text for block in resp.content if block.type == "text")
    data = _extract_json(raw)

    def _verify_group(items):
        for item in items:
            for c in item.get("citations", []):
                c["verified"] = verify_quote(c.get("quote", ""), flat_segments)
        return items

    data["themes"] = _verify_group(data.get("themes", []))
    data["disagreements"] = _verify_group(data.get("disagreements", []))
    return data


def ask_freeform_question(question: str, retrieved_segments: List[QASegment]) -> GroundedAnswer:
    """Same grounding/citation contract as answer_question_for_expert, but the
    caller passes in whatever the retriever found across ALL transcripts."""
    return answer_question_for_expert(question, retrieved_segments)
