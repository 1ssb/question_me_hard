"""Core clarification module – questions the prompt author until clarity is reached."""
from __future__ import annotations

import re
from typing import Callable, List, Optional

from .memory import Memory


# Type alias for any LLM callable: takes a string prompt, returns a string.
LLMCallable = Callable[[str], str]

_QUESTION_PROMPT_TEMPLATE = """\
You are a helpful assistant that clarifies prompts. Given the prompt below and \
any previously known context, identify up to {max_questions} distinct questions \
(if any) that would help clarify the prompt's intent or requirements. If the \
prompt is already sufficiently clear, respond only with the word "CLEAR".

Previously known context (do NOT re-ask these):\n{context}

Prompt to clarify:\n{prompt}

Respond with a numbered list of questions, one per line, or "CLEAR" if no \
clarification is needed."""

_REFINE_PROMPT_TEMPLATE = """\
Rewrite the following prompt to incorporate the clarifications provided. \
The rewritten prompt should be clear, unambiguous, and comprehensive.

Original prompt:\n{prompt}

Clarifications:\n{clarifications}

Rewritten prompt:"""


def _parse_questions(response: str) -> List[str]:
    """Parse a numbered or bulleted list of questions from the LLM response."""
    if "CLEAR" in response.upper():
        return []
    questions: List[str] = []
    for line in response.strip().splitlines():
        line = line.strip()
        match = re.match(r"^(?:\d+[.)]\s*|-\s*)(.+)", line)
        if match:
            q = match.group(1).strip()
            if q:
                questions.append(q)
    return questions


class Clarifier:
    """Questions the prompt author until clarity is reached.

    Uses a provided LLM callable to generate clarifying questions, collects
    answers from the prompt author (via *ask_fn*), and stores them in a
    :class:`Memory` instance.  On subsequent calls existing answers are reused
    so that already-answered questions are never re-asked, saving LLM thinking
    tokens.

    Args:
        memory: Optional :class:`Memory` instance.  Share one across sessions
            to persist existing notions between calls.
        max_questions: Maximum number of questions to request per round.
        max_rounds: Maximum number of questioning rounds before refinement.
        ask_fn: Callable used to prompt the human for answers.  Receives the
            question string and returns the answer string.  Defaults to the
            built-in :func:`input`.
    """

    def __init__(
        self,
        memory: Optional[Memory] = None,
        max_questions: int = 5,
        max_rounds: int = 3,
        ask_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.memory: Memory = memory if memory is not None else Memory()
        self.max_questions = max_questions
        self.max_rounds = max_rounds
        self._ask_fn: Callable[[str], str] = ask_fn if ask_fn is not None else input

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_context(self) -> str:
        pairs = self.memory.all_pairs()
        if not pairs:
            return "(none)"
        return "\n".join(f"Q: {q}\nA: {a}" for q, a in pairs)

    def _generate_questions(self, prompt: str, llm_fn: LLMCallable) -> List[str]:
        query = _QUESTION_PROMPT_TEMPLATE.format(
            max_questions=self.max_questions,
            context=self._build_context(),
            prompt=prompt,
        )
        return _parse_questions(llm_fn(query))

    def _refine_prompt(self, prompt: str, llm_fn: LLMCallable) -> str:
        pairs = self.memory.all_pairs()
        if not pairs:
            return prompt
        clarifications = "\n".join(f"- {q}: {a}" for q, a in pairs)
        query = _REFINE_PROMPT_TEMPLATE.format(
            prompt=prompt,
            clarifications=clarifications,
        )
        return llm_fn(query).strip()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clarify(self, prompt: str, llm_fn: LLMCallable) -> str:
        """Interactively clarify *prompt* by asking the author targeted questions.

        Iterates for up to ``max_rounds`` rounds.  Each round the LLM generates
        clarifying questions; any already answered via :attr:`memory` are
        skipped.  New answers are collected via :attr:`_ask_fn` and stored.
        Once no new questions arise the refined prompt is returned.

        Args:
            prompt: The initial, potentially ambiguous prompt.
            llm_fn: A callable ``(str) -> str`` backed by any LLM.

        Returns:
            A refined version of *prompt* that incorporates all clarifications.
        """
        for _ in range(self.max_rounds):
            questions = self._generate_questions(prompt, llm_fn)
            if not questions:
                break

            new_questions = [q for q in questions if self.memory.recall(q) is None]
            if not new_questions:
                break

            for q in new_questions:
                answer = self._ask_fn(f"{q}\n> ")
                self.memory.remember(q, answer)

        return self._refine_prompt(prompt, llm_fn)
