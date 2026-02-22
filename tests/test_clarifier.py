"""Tests for question_me_hard.Clarifier."""
from unittest.mock import MagicMock

import pytest

from question_me_hard import Clarifier, Memory
from question_me_hard.clarifier import _parse_questions


# ---------------------------------------------------------------------------
# _parse_questions helper
# ---------------------------------------------------------------------------


def test_parse_questions_numbered():
    response = "1. What is the goal?\n2. Who is the user?\n3. What format?"
    qs = _parse_questions(response)
    assert qs == ["What is the goal?", "Who is the user?", "What format?"]


def test_parse_questions_bullet():
    response = "- What is the goal?\n- Who is the user?"
    qs = _parse_questions(response)
    assert qs == ["What is the goal?", "Who is the user?"]


def test_parse_questions_clear_keyword():
    assert _parse_questions("CLEAR") == []
    assert _parse_questions("The prompt is CLEAR enough.") == []
    assert _parse_questions("clear") == []


def test_parse_questions_empty():
    assert _parse_questions("") == []


# ---------------------------------------------------------------------------
# Clarifier
# ---------------------------------------------------------------------------


def _seq_llm(*responses: str) -> MagicMock:
    """Return a mock LLM callable that returns *responses* in sequence."""
    return MagicMock(side_effect=list(responses))


def test_clear_prompt_skips_asking():
    """When LLM says CLEAR in round 1, no questions are asked."""
    llm = _seq_llm("CLEAR")
    ask_fn = MagicMock()
    clarifier = Clarifier(ask_fn=ask_fn)
    result = clarifier.clarify("write a poem", llm)
    ask_fn.assert_not_called()
    # No clarifications stored → original prompt returned unchanged
    assert result == "write a poem"


def test_questions_are_asked_and_stored():
    """Clarifier asks questions, stores answers, then refines the prompt."""
    mem = Memory()
    questions_response = "1. Who is the target audience?\n2. What tone should be used?"
    refine_response = "Write a formal poem for software engineers."
    llm = _seq_llm(questions_response, "CLEAR", refine_response)
    ask_fn = MagicMock(side_effect=["software engineers\n", "formal\n"])

    clarifier = Clarifier(memory=mem, max_rounds=2, ask_fn=ask_fn)
    result = clarifier.clarify("write a poem", llm)

    assert ask_fn.call_count == 2
    assert len(mem) == 2
    assert mem.recall("Who is the target audience?") == "software engineers"
    assert "software engineers" in result or "formal" in result


def test_existing_memory_skips_repeated_question():
    """Questions already in memory must not be re-asked."""
    mem = Memory()
    mem.remember("Who is the target audience?", "developers")

    # LLM returns the same question that is already answered
    questions_response = "1. Who is the target audience?"
    refine_response = "Write a poem for developers."
    llm = _seq_llm(questions_response, refine_response)
    ask_fn = MagicMock()

    clarifier = Clarifier(memory=mem, max_rounds=1, ask_fn=ask_fn)
    result = clarifier.clarify("write a poem", llm)

    # ask_fn should NOT be called since the question is already answered
    ask_fn.assert_not_called()
    assert "developers" in result


def test_max_rounds_respected():
    """Clarifier must not exceed max_rounds questioning rounds."""
    # Always return new questions so rounds would be infinite without cap.
    always_questions = "1. Clarify this?\n2. And that?"
    refine_response = "refined"
    # Provide enough responses for max_rounds + refine
    llm = MagicMock(side_effect=[always_questions, always_questions, refine_response])
    ask_fn = MagicMock(return_value="some answer")

    clarifier = Clarifier(max_rounds=2, ask_fn=ask_fn)
    result = clarifier.clarify("vague prompt", llm)

    # After 2 rounds the questions become "known", so the 3rd LLM call is
    # the refine step.  Total LLM calls should be ≤ max_rounds + 1.
    assert llm.call_count <= 3


def test_shared_memory_across_sessions():
    """Memory shared between two Clarifier instances reuses existing answers."""
    mem = Memory()

    # First clarifier learns one answer
    llm1 = _seq_llm(
        "1. What language?\n",  # round 1 question
        "CLEAR",               # round 2: no more questions
        "Write Python code.",  # refine
    )
    ask_fn1 = MagicMock(return_value="Python")
    Clarifier(memory=mem, max_rounds=2, ask_fn=ask_fn1).clarify("write code", llm1)
    assert mem.recall("What language?") == "Python"

    # Second clarifier: LLM returns the same question → should be skipped
    llm2 = _seq_llm(
        "1. What language?\n",  # same question
        "Write Python code.",   # refine
    )
    ask_fn2 = MagicMock()
    Clarifier(memory=mem, max_rounds=1, ask_fn=ask_fn2).clarify("write code", llm2)
    ask_fn2.assert_not_called()
