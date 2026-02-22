"""Integration tests – verify all question_me_hard components work together."""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Public API import smoke test
# ---------------------------------------------------------------------------


def test_public_api_all_symbols_importable():
    """Every symbol listed in __all__ must be importable from the top-level package."""
    import question_me_hard as qmh

    expected = [
        "Clarifier",
        "ClarificationSubagent",
        "FieldStatus",
        "LLMCallable",
        "Memory",
        "QNode",
        "QuestionFn",
        "Spec",
        "SpecEntry",
        "StopFn",
        "build_binary_question_tree",
        "default_spec_check_bank",
        "render_dot",
        "to_dot",
    ]
    for name in expected:
        assert hasattr(qmh, name), f"Missing public symbol: {name}"


def test_public_api_all_matches_expected():
    """__all__ must contain exactly the documented public symbols."""
    import question_me_hard as qmh

    assert set(qmh.__all__) == {
        "Clarifier",
        "ClarificationSubagent",
        "FieldStatus",
        "LLMCallable",
        "Memory",
        "QNode",
        "QuestionFn",
        "Spec",
        "SpecEntry",
        "StopFn",
        "build_binary_question_tree",
        "default_spec_check_bank",
        "render_dot",
        "to_dot",
    }


# ---------------------------------------------------------------------------
# Clarifier + Memory end-to-end
# ---------------------------------------------------------------------------


def test_clarifier_memory_full_loop():
    """Clarifier asks questions, stores answers in Memory, then refines the prompt."""
    from question_me_hard import Clarifier, Memory

    mem = Memory()
    questions_response = "1. What language?\n2. What framework?"
    refine_response = "Write a Python FastAPI REST API."
    llm = MagicMock(side_effect=[questions_response, "CLEAR", refine_response])
    ask_fn = MagicMock(side_effect=["Python", "FastAPI"])

    clarifier = Clarifier(memory=mem, max_rounds=2, ask_fn=ask_fn)
    result = clarifier.clarify("write a REST API", llm)

    assert ask_fn.call_count == 2
    assert mem.recall("What language?") == "Python"
    assert mem.recall("What framework?") == "FastAPI"
    assert isinstance(result, str)
    assert len(result) > 0


def test_clarifier_memory_no_re_ask_on_second_session():
    """Memory persisted from session 1 prevents re-asking in session 2."""
    from question_me_hard import Clarifier, Memory

    mem = Memory()

    # Session 1
    llm1 = MagicMock(side_effect=["1. What language?", "CLEAR", "Write Python."])
    ask_fn1 = MagicMock(return_value="Python")
    Clarifier(memory=mem, max_rounds=2, ask_fn=ask_fn1).clarify("write code", llm1)

    # Session 2 – same question returned by LLM should be skipped
    llm2 = MagicMock(side_effect=["1. What language?", "Write Python code."])
    ask_fn2 = MagicMock()
    result2 = Clarifier(memory=mem, max_rounds=1, ask_fn=ask_fn2).clarify("write code", llm2)

    ask_fn2.assert_not_called()
    assert isinstance(result2, str)


# ---------------------------------------------------------------------------
# ClarificationSubagent + Spec + Memory end-to-end
# ---------------------------------------------------------------------------


def _make_subagent_llm():
    """Return a mock LLM for a full ClarificationSubagent run.

    The mock provides:
    - One question per unknown field
    - One extraction response per question
    - Assumption values for remaining fields
    - A contract response
    """
    from question_me_hard import Spec

    n_fields = len(Spec.FIELD_NAMES)
    # For each of 7 fields: one question + one extraction = 14 calls,
    # plus 1 contract call.
    responses = []
    for i, field in enumerate(Spec.FIELD_NAMES):
        responses.append(f"What about {field}?")     # question
        responses.append(f"Spec value for {field}.")  # extraction
    responses.append(
        "FUNCTION_SIGNATURE: def solve(data: list) -> list:\n"
        "DOCSTRING: Solve the problem.\n"
        "EXAMPLES: solve([1, 2])\n"
        "TEST_CASES: - solve([]) == []\n"
        "ERROR_HANDLING: Raise ValueError on invalid input.\n"
    )
    return MagicMock(side_effect=responses)


def test_subagent_full_pipeline():
    """ClarificationSubagent clarifies all fields, then produces a contract."""
    from question_me_hard import ClarificationSubagent, Memory, Spec

    mem = Memory()
    llm = _make_subagent_llm()
    ask_fn = MagicMock(return_value="user answer")

    agent = ClarificationSubagent(memory=mem, max_questions=20, ask_fn=ask_fn)
    spec = agent.clarify("write a sort function", llm)

    # All required fields must be resolved
    for f in Spec.REQUIRED_FIELDS:
        assert spec.get_entry(f).status.value in ("known", "assumed")

    # Contract should contain all five keys
    contract = agent.to_contract("write a sort function", llm)
    assert set(contract.keys()) == {
        "function_signature",
        "docstring",
        "examples",
        "test_cases",
        "error_handling",
    }
    assert contract["function_signature"] != ""


def test_subagent_memory_reuse_across_instances():
    """Answers stored by one ClarificationSubagent are reused by a second."""
    from question_me_hard import ClarificationSubagent, FieldStatus, Memory, Spec

    mem = Memory()

    # Agent 1: only "inputs" is unknown; it asks one question and caches the answer.
    spec1 = Spec()
    for f in Spec.FIELD_NAMES:
        if f != "inputs":
            spec1.update(f, f"value for {f}")

    llm1 = MagicMock(side_effect=[
        "What are the inputs?",  # _generate_question for "inputs"
        "A list of integers.",   # _extract_spec_value for "inputs"
    ])
    ask_fn1 = MagicMock(return_value="list of ints")
    agent1 = ClarificationSubagent(spec=spec1, memory=mem, max_questions=5, ask_fn=ask_fn1)
    agent1.clarify("sort function", llm1)
    assert mem.recall("What are the inputs?") == "list of ints"

    # Agent 2: also has only "inputs" unknown; LLM returns the same question → cache hit.
    spec2 = Spec()
    for f in Spec.FIELD_NAMES:
        if f != "inputs":
            spec2.update(f, f"value for {f}")

    llm2 = MagicMock(side_effect=[
        "What are the inputs?",  # same question → cache hit, no re-asking
        "A list of integers.",   # _extract_spec_value (called with cached answer)
    ])
    ask_fn2 = MagicMock()
    agent2 = ClarificationSubagent(spec=spec2, memory=mem, max_questions=5, ask_fn=ask_fn2)
    agent2.clarify("sort function", llm2)

    ask_fn2.assert_not_called()
    assert spec2.get_entry("inputs").status == FieldStatus.KNOWN


# ---------------------------------------------------------------------------
# Tree pipeline: default_spec_check_bank → build_binary_question_tree → to_dot
# ---------------------------------------------------------------------------


def test_tree_pipeline_default_bank():
    """default_spec_check_bank feeds into build_binary_question_tree and to_dot."""
    from question_me_hard import build_binary_question_tree, default_spec_check_bank, to_dot

    checks = default_spec_check_bank()
    root_text, _ = checks[0]
    root = build_binary_question_tree(root_text, max_depth=2, spec_checks=checks)

    dot = to_dot(root)
    assert "digraph" in dot
    assert root.node_id in dot


def test_tree_pipeline_all_nodes_in_dot():
    """Every node produced by build_binary_question_tree appears in the DOT string."""
    from question_me_hard import build_binary_question_tree, to_dot

    root = build_binary_question_tree("Root check?", max_depth=2)
    dot = to_dot(root)

    queue = [root]
    while queue:
        node = queue.pop()
        assert node.node_id in dot
        queue.extend(node.children.values())


def test_tree_pipeline_ask_user_leaves_are_in_dot():
    """ASK USER leaf nodes must appear in the DOT output."""
    from question_me_hard import build_binary_question_tree, to_dot

    root = build_binary_question_tree("Root?", max_depth=1)
    dot = to_dot(root)

    no_child = root.children.get("No")
    assert no_child is not None
    assert no_child.text.upper().startswith("ASK USER:")
    assert no_child.node_id in dot


# ---------------------------------------------------------------------------
# Cross-component: Clarifier output drives tree construction
# ---------------------------------------------------------------------------


def test_clarifier_questions_used_as_tree_root():
    """Simulate a flow: clarify a prompt, use first question as tree root."""
    from question_me_hard import (
        Clarifier,
        Memory,
        build_binary_question_tree,
        default_spec_check_bank,
        to_dot,
    )

    mem = Memory()
    clarifier_question = "1. What input format is expected?"
    llm = MagicMock(side_effect=[clarifier_question, "CLEAR", "Refined prompt."])
    ask_fn = MagicMock(return_value="JSON array")

    clarifier = Clarifier(memory=mem, max_rounds=2, ask_fn=ask_fn)
    clarifier.clarify("process data", llm)

    # Use the stored question as the root of a question tree
    answered_questions = [q for q, _ in mem.all_pairs()]
    assert len(answered_questions) > 0

    root_text = answered_questions[0]
    checks = default_spec_check_bank()
    root = build_binary_question_tree(root_text, max_depth=1, spec_checks=checks)
    dot = to_dot(root)

    assert root.node_id in dot
    assert "digraph" in dot


# ---------------------------------------------------------------------------
# ClarificationSubagent + tree: spec field names used as custom tree checks
# ---------------------------------------------------------------------------


def test_subagent_spec_fields_drive_tree():
    """Spec field names can be used to build a custom spec-check tree."""
    from question_me_hard import (
        Spec,
        build_binary_question_tree,
        to_dot,
    )

    # Build spec-check tuples from Spec.FIELD_NAMES
    checks = [
        (f"Is '{f}' specified?", f"Ask the author to specify {f}.")
        for f in Spec.FIELD_NAMES
    ]
    root = build_binary_question_tree(checks[0][0], max_depth=2, spec_checks=checks)
    dot = to_dot(root)

    assert "digraph" in dot
    # All node IDs should appear in the DOT output
    queue = [root]
    while queue:
        node = queue.pop()
        assert node.node_id in dot
        queue.extend(node.children.values())


# ---------------------------------------------------------------------------
# render_dot raises RuntimeError when graphviz package is absent
# ---------------------------------------------------------------------------


def test_render_dot_raises_without_graphviz(monkeypatch):
    """render_dot must raise RuntimeError when the graphviz package is missing."""
    import builtins
    import importlib

    from question_me_hard import build_binary_question_tree, render_dot, to_dot

    root = build_binary_question_tree("Root?", max_depth=0)
    dot = to_dot(root)

    original_import = builtins.__import__

    def _no_graphviz(name, *args, **kwargs):
        if name == "graphviz":
            raise ImportError("graphviz not installed")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_graphviz)
    with pytest.raises(RuntimeError, match="graphviz"):
        render_dot(dot, "/tmp/test_render_output")
