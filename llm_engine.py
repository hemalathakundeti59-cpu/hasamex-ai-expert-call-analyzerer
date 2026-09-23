"""
llm_engine.py

Gemini-powered LLM layer with retrieval grounding,
structured JSON output, and quote verification.
"""

from __future__ import annotations
import json
import os
import re
import time
from dotenv import load_dotenv
load_dotenv()
from dataclasses import dataclass
from difflib import SequenceMatcher
from typing import List

from transcript_parser import QASegment

try:
    from google import genai
except ImportError:  # pragma: no cover
    genai = None


MODEL_NAME = os.environ.get("HASAMEX_MODEL", "gemini-3.6-flash")

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


def _client():
    if genai is None:
        raise RuntimeError(
            "The 'google-genai' package is not installed. "
            "Run: pip install -U google-genai"
        )

    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Add your Gemini API key to .env."
        )

    return genai.Client(api_key=api_key)


def _format_segments(segments: List[QASegment]) -> str:
    lines = []

    for s in segments:
        lines.append(
            f"[{s.source_file} | {s.expert} | {s.market} | {s.timestamp}]\n"
            f"Q: {s.question}\n"
            f"A: {s.answer}\n"
        )

    return "\n".join(lines)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def verify_quote(
    quote: str,
    segments: List[QASegment],
    min_ratio: float = 0.85
) -> bool:
    """Check that the quote is a genuine or near-exact substring."""

    nq = _normalize(quote)

    if not nq:
        return False

    for s in segments:
        na = _normalize(s.answer)

        if nq in na:
            return True

        match = SequenceMatcher(
            None,
            nq,
            na
        ).find_longest_match(
            0,
            len(nq),
            0,
            len(na)
        )

        if match.size >= int(len(nq) * min_ratio):
            return True

    return False


def _extract_json(raw: str) -> dict:
    raw = raw.strip()

    raw = re.sub(
        r"^```json\s*|\s*```$",
        "",
        raw,
        flags=re.MULTILINE
    )

    return json.loads(raw)


SYSTEM_PROMPT = """
You are an analyst assistant answering questions ONLY from the
provided expert-call transcript excerpts.

Rules:

1. Use ONLY the excerpts provided.
2. Never use outside knowledge.
3. If the excerpts do not contain the answer, say so explicitly.
4. Every factual claim must be supported by an exact quote from the excerpts.
5. Quotes must be copied verbatim from the excerpts.
6. Do not invent quotes.
7. Do not merge or paraphrase quotes.
8. Include the source file and timestamp for every citation.
9. Return ONLY valid JSON.
10. Do not use markdown fences.
"""


def _generate(prompt: str, max_tokens: int = 2000) -> str:
    client = _client()

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt,
                config={
                    "system_instruction": SYSTEM_PROMPT,
                    "temperature": 0,
                    "max_output_tokens": max_tokens,
                    "response_mime_type": "application/json",
                },
            )
            return response.text

        except Exception as e:
            if "503" not in str(e) or attempt == 2:
                raise

            time.sleep(3 * (attempt + 1))

def answer_question_for_expert(
    question: str,
    segments: List[QASegment]
) -> GroundedAnswer:
    """Answer one interview-guide question for one expert."""

    schema = """
{
  "not_mentioned": true or false,
  "answer": "string",
  "citations": [
    {
      "quote": "exact quote",
      "timestamp": "timestamp",
      "source_file": "source file"
    }
  ]
}
"""

    prompt = f"""
EXPERT TRANSCRIPT EXCERPTS:

{_format_segments(segments)}

QUESTION:
{question}

Return JSON matching this schema exactly:

{schema}

If the question is not addressed in the excerpts:

- set "not_mentioned" to true
- explain that it was not mentioned
- return an empty citations list

Do not invent information or quotes.
"""

    raw = _generate(prompt, max_tokens=1000)
    data = _extract_json(raw)

    citations = []

    for c in data.get("citations", []):
        quote = c.get("quote", "")
        source_file = c.get("source_file", "")

        verified = verify_quote(
            quote,
            segments
        )

        expert = next(
            (
                s.expert
                for s in segments
                if s.source_file == source_file
            ),
            ""
        )

        citations.append(
            Citation(
                source_file=source_file,
                timestamp=c.get("timestamp", ""),
                expert=expert,
                quote=quote,
                verified=verified,
            )
        )

    return GroundedAnswer(
        answer_text=data.get("answer", ""),
        citations=citations,
        not_mentioned=bool(
            data.get("not_mentioned", False)
        ),
    )

def answer_all_questions_for_expert(
    questions: List[str],
    segments: List[QASegment]
) -> List[GroundedAnswer]:

    schema = """
{
  "answers": [
    {
      "question_number": 1,
      "not_mentioned": false,
      "answer": "string",
      "citations": [
        {
          "quote": "exact quote",
          "timestamp": "timestamp",
          "source_file": "source file"
        }
      ]
    }
  ]
}
"""

    questions_text = "\n".join(
        f"{i + 1}. {q}" for i, q in enumerate(questions)
    )

    prompt = f"""
EXPERT TRANSCRIPT EXCERPTS:

{_format_segments(segments)}

INTERVIEW QUESTIONS:

{questions_text}

Answer ALL questions in ONE response.

For each question:
- Use ONLY the transcript excerpts.
- Give an evidence-based answer.
- Every factual claim must be supported by an exact quote.
- Quotes must be copied verbatim.
- Include the timestamp and source file.
- If the transcript does not address the question, set
  "not_mentioned" to true and use an empty citations list.
- Keep question_number exactly aligned with the numbered questions.

Return JSON matching this schema exactly:

{schema}

Do not invent information or quotes.
"""

    raw = _generate(prompt, max_tokens=3000)
    data = _extract_json(raw)

    returned = {
        int(item.get("question_number", 0)): item
        for item in data.get("answers", [])
    }

    results = []

    for number, question in enumerate(questions, start=1):
        item = returned.get(number, {})

        citations = []

        for c in item.get("citations", []):
            quote = c.get("quote", "")
            source_file = c.get("source_file", "")

            verified = verify_quote(
                quote,
                segments
            )

            expert = next(
                (
                    s.expert
                    for s in segments
                    if s.source_file == source_file
                ),
                ""
            )

            citations.append(
                Citation(
                    source_file=source_file,
                    timestamp=c.get("timestamp", ""),
                    expert=expert,
                    quote=quote,
                    verified=verified,
                )
            )

        results.append(
            GroundedAnswer(
                answer_text=item.get("answer", ""),
                citations=citations,
                not_mentioned=bool(
                    item.get("not_mentioned", False)
                ),
            )
        )

    return results
def cross_transcript_themes(
    all_segments_by_expert: dict
) -> dict:
    """
    Find common themes and disagreements across all experts.
    """

    parts = []
    flat_segments: List[QASegment] = []

    for expert, segs in all_segments_by_expert.items():
        parts.append(_format_segments(segs))
        flat_segments.extend(segs)

    excerpts_text = "\n".join(parts)

    schema = """
{
  "themes": [
    {
      "summary": "string",
      "citations": [
        {
          "expert": "string",
          "quote": "exact quote",
          "timestamp": "timestamp",
          "source_file": "source file"
        }
      ]
    }
  ],
  "disagreements": [
    {
      "summary": "string",
      "citations": [
        {
          "expert": "string",
          "quote": "exact quote",
          "timestamp": "timestamp",
          "source_file": "source file"
        }
      ]
    }
  ]
}
"""

    prompt = f"""
EXCERPTS FROM ALL EXPERTS:

{excerpts_text}

Identify:

1. COMMON THEMES:
Themes independently raised by at least two experts.

2. DISAGREEMENTS:
Meaningfully different views between experts, such as differences
in emphasis, growth estimates, or adoption timelines.

Every theme or disagreement must contain exact quotes from the
experts involved.

Do not invent quotes.

Return JSON matching this schema exactly:

{schema}
"""

    raw = _generate(prompt, max_tokens=2000)
    data = _extract_json(raw)

    def _verify_group(items):
        for item in items:
            for c in item.get("citations", []):
                c["verified"] = verify_quote(
                    c.get("quote", ""),
                    flat_segments
                )

        return items

    data["themes"] = _verify_group(
        data.get("themes", [])
    )

    data["disagreements"] = _verify_group(
        data.get("disagreements", [])
    )

    return data


def ask_freeform_question(
    question: str,
    retrieved_segments: List[QASegment]
) -> GroundedAnswer:
    """
    Answer a free-form question using retrieved segments
    across all transcripts.
    """

    return answer_question_for_expert(
        question,
        retrieved_segments
    )