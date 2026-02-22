"""Tests for question_me_hard.Memory."""
import pytest

from question_me_hard.memory import Memory


def test_remember_and_recall():
    mem = Memory()
    mem.remember("What is the target audience?", "Software engineers")
    assert mem.recall("What is the target audience?") == "Software engineers"


def test_recall_missing_returns_none():
    mem = Memory()
    assert mem.recall("Nonexistent question") is None


def test_all_pairs():
    mem = Memory()
    mem.remember("Q1?", "A1")
    mem.remember("Q2?", "A2")
    pairs = mem.all_pairs()
    assert len(pairs) == 2
    assert ("Q1?", "A1") in pairs
    assert ("Q2?", "A2") in pairs


def test_len():
    mem = Memory()
    assert len(mem) == 0
    mem.remember("Q?", "A")
    assert len(mem) == 1


def test_strips_whitespace_on_remember():
    mem = Memory()
    mem.remember("  Question?  ", "  Answer  ")
    assert mem.recall("Question?") == "Answer"
    assert mem.recall("  Question?  ") == "Answer"


def test_overwrite_existing_answer():
    mem = Memory()
    mem.remember("Q?", "old answer")
    mem.remember("Q?", "new answer")
    assert mem.recall("Q?") == "new answer"
    assert len(mem) == 1
