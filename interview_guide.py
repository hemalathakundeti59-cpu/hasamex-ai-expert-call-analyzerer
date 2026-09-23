"""interview_guide.py — parses the numbered questions out of Interview_Guide.txt"""

import re
from typing import List

_Q_RE = re.compile(r"^\s*\d+\.\s*(.+)$")


def load_questions(path: str) -> List[str]:
    questions = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            m = _Q_RE.match(line)
            if m:
                questions.append(m.group(1).strip())
    return questions
