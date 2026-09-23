"""
transcript_parser.py

Parses the raw expert-call transcript .txt files into structured data.

Expected input format (one of the given files):

    Expert 1 - Dr. Jean Martin
    Role: Head of Urology
    Market: France

    00:00
    Interviewer: Thanks for joining ...

    00:18
    Dr. Martin: Adoption is growing ...

Design notes
------------
- Every spoken turn becomes a `Turn` (timestamp, speaker, text, is_expert).
- Every expert answer is additionally paired with the interviewer question
  that immediately preceded it, producing a `QASegment`. This is the unit
  we retrieve and cite from, because an answer alone is often ambiguous
  without knowing which question it responds to.
- Nothing here calls an LLM. Parsing is 100% deterministic so that every
  timestamp/quote shown later is guaranteed to be traceable back to the
  source file (this is what the case brief calls "do not invent info").
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List


TIMESTAMP_RE = re.compile(r"^(\d{1,2}:\d{2})$")
HEADER_EXPERT_RE = re.compile(r"^Expert\s+\d+\s*[-–]\s*(.+)$", re.IGNORECASE)
HEADER_ROLE_RE = re.compile(r"^Role:\s*(.+)$", re.IGNORECASE)
HEADER_MARKET_RE = re.compile(r"^Market:\s*(.+)$", re.IGNORECASE)


@dataclass
class Turn:
    timestamp: str
    speaker: str
    text: str
    is_expert: bool


@dataclass
class QASegment:
    """One expert answer, grounded with its timestamp and preceding question."""
    seg_id: str
    expert: str
    role: str
    market: str
    timestamp: str
    question: str
    answer: str
    source_file: str


@dataclass
class Transcript:
    expert: str
    role: str
    market: str
    source_file: str
    turns: List[Turn] = field(default_factory=list)
    segments: List[QASegment] = field(default_factory=list)


def _parse_single_transcript(path: str) -> Transcript:
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.rstrip("\n") for ln in f]

    expert, role, market = "Unknown", "Unknown", "Unknown"
    turns: List[Turn] = []

    i = 0
    # --- header block (before the first timestamp line) ---
    while i < len(lines):
        line = lines[i].strip()
        if TIMESTAMP_RE.match(line):
            break
        m = HEADER_EXPERT_RE.match(line)
        if m:
            expert = m.group(1).strip()
        m = HEADER_ROLE_RE.match(line)
        if m:
            role = m.group(1).strip()
        m = HEADER_MARKET_RE.match(line)
        if m:
            market = m.group(1).strip()
        i += 1

    # --- body: timestamp, then one speaker turn, repeated ---
    current_ts = None
    current_text_lines: List[str] = []

    def flush():
        if current_ts is None or not current_text_lines:
            return
        raw = " ".join(l.strip() for l in current_text_lines if l.strip())
        if ":" in raw:
            speaker, text = raw.split(":", 1)
            speaker = speaker.strip()
            text = text.strip()
        else:
            speaker, text = "Unknown", raw.strip()
        is_expert = speaker.lower() not in ("interviewer", "interviewer:", "moderator")
        turns.append(Turn(timestamp=current_ts, speaker=speaker, text=text, is_expert=is_expert))

    while i < len(lines):
        line = lines[i].strip()
        if TIMESTAMP_RE.match(line):
            flush()
            current_ts = line
            current_text_lines = []
        elif line:
            current_text_lines.append(line)
        i += 1
    flush()

    transcript = Transcript(expert=expert, role=role, market=market,
                             source_file=os.path.basename(path), turns=turns)

    # --- pair each expert turn with the preceding interviewer turn ---
    last_question = ""
    seg_count = 0
    for turn in turns:
        if not turn.is_expert:
            last_question = turn.text
            continue
        seg_count += 1
        seg_id = f"{transcript.source_file}#{turn.timestamp}"
        transcript.segments.append(
            QASegment(
                seg_id=seg_id,
                expert=expert,
                role=role,
                market=market,
                timestamp=turn.timestamp,
                question=last_question,
                answer=turn.text,
                source_file=transcript.source_file,
            )
        )

    return transcript


def load_transcripts(paths: List[str]) -> List[Transcript]:
    return [_parse_single_transcript(p) for p in paths]


def all_segments(transcripts: List[Transcript]) -> List[QASegment]:
    segs: List[QASegment] = []
    for t in transcripts:
        segs.extend(t.segments)
    return segs


if __name__ == "__main__":
    # Quick self-test against the /data folder — run: python transcript_parser.py
    data_dir = os.path.join(os.path.dirname(__file__), "data")
    files = sorted(
        os.path.join(data_dir, f)
        for f in os.listdir(data_dir)
        if f.lower().startswith("transcript")
    )
    ts = load_transcripts(files)
    for t in ts:
        print(f"\n=== {t.expert} ({t.role}, {t.market}) — {t.source_file} ===")
        for seg in t.segments:
            print(f"  [{seg.timestamp}] Q: {seg.question[:60]}...")
            print(f"          A: {seg.answer[:80]}...")
    total = sum(len(t.segments) for t in ts)
    print(f"\nParsed {len(ts)} transcripts, {total} answer segments total.")
