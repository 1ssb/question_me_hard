"""Stores existing notions (Q&A pairs) to avoid redundant questioning."""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple


class Memory:
    """Stores question-answer pairs from previous clarification sessions.

    By reusing known answers the clarifier avoids re-asking questions that
    have already been answered, saving LLM thinking tokens.
    """

    def __init__(self) -> None:
        self._store: Dict[str, str] = {}

    def remember(self, question: str, answer: str) -> None:
        """Store a question-answer pair."""
        self._store[question.strip()] = answer.strip()

    def recall(self, question: str) -> Optional[str]:
        """Return the stored answer for *question*, or ``None`` if not found."""
        return self._store.get(question.strip())

    def all_pairs(self) -> List[Tuple[str, str]]:
        """Return all stored question-answer pairs."""
        return list(self._store.items())

    def __len__(self) -> int:
        return len(self._store)

    def __repr__(self) -> str:  # pragma: no cover
        return f"Memory({len(self)} entries)"
