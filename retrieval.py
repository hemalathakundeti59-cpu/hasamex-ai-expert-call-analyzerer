"""
retrieval.py

A dependency-free TF-IDF + cosine-similarity retriever.

Why not a vector DB / embeddings API for a 3-transcript case study?
- The whole corpus is ~21 short segments. A full vector DB is overkill and
  adds an external dependency + API cost for no real accuracy gain here.
- TF-IDF still forces us to build a genuine *retrieval* layer (the thing
  that matters architecturally) instead of just stuffing everything into
  one prompt, which is what "scaling to 30+ transcripts" is really asking
  about — see README "Scaling" section for how this swaps to a real
  vector store.

This module is intentionally small and pure Python (no numpy/sklearn) so
it needs zero extra installs beyond the LLM SDK.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Dict, List, Sequence

from transcript_parser import QASegment

_WORD_RE = re.compile(r"[a-zA-Z']+")


def _tokenize(text: str) -> List[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


class TfidfIndex:
    def __init__(self, segments: Sequence[QASegment]):
        self.segments = list(segments)
        self._doc_tokens: List[List[str]] = [
            _tokenize(f"{s.question} {s.answer}") for s in self.segments
        ]
        self._df: Counter = Counter()
        for toks in self._doc_tokens:
            for term in set(toks):
                self._df[term] += 1
        self._n_docs = max(len(self._doc_tokens), 1)
        self._doc_vecs: List[Dict[str, float]] = [
            self._vectorize(toks) for toks in self._doc_tokens
        ]

    def _idf(self, term: str) -> float:
        df = self._df.get(term, 0)
        return math.log((self._n_docs + 1) / (df + 1)) + 1.0

    def _vectorize(self, tokens: List[str]) -> Dict[str, float]:
        tf = Counter(tokens)
        vec = {term: count * self._idf(term) for term, count in tf.items()}
        norm = math.sqrt(sum(v * v for v in vec.values())) or 1.0
        return {k: v / norm for k, v in vec.items()}

    @staticmethod
    def _cosine(a: Dict[str, float], b: Dict[str, float]) -> float:
        if len(a) > len(b):
            a, b = b, a
        return sum(v * b.get(k, 0.0) for k, v in a.items())

    def search(self, query: str, top_k: int = 6,
               expert_filter: str | None = None) -> List[QASegment]:
        q_vec = self._vectorize(_tokenize(query))
        scored = []
        for seg, doc_vec in zip(self.segments, self._doc_vecs):
            if expert_filter and seg.expert != expert_filter:
                continue
            score = self._cosine(q_vec, doc_vec)
            scored.append((score, seg))
        scored.sort(key=lambda x: x[0], reverse=True)
        # Fallback: if TF-IDF finds nothing (e.g. very short/odd query),
        # still return the top_k most recent segments for that expert so
        # the LLM has *something* grounded to work with rather than
        # silently returning zero context.
        results = [seg for score, seg in scored[:top_k] if score > 0]
        if not results:
            results = [seg for _, seg in scored[:top_k]]
        return results

    def all_for_expert(self, expert: str) -> List[QASegment]:
        return [s for s in self.segments if s.expert == expert]
