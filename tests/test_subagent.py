"""Tests for question_me_hard.subagent (ClarificationSubagent, _parse_contract)."""
from __future__ import annotations

from typing import Optional
from unittest.mock import MagicMock

import pytest

from question_me_hard.memory import Memory
from question_me_hard.spec import FieldStatus, Spec
from question_me_hard.subagent import ClarificationSubagent, _parse_contract


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seq_llm(*responses: str) -> MagicMock:
    """Return a mock LLM callable that returns *responses* in sequence."""
    return MagicMock(side_effect=list(responses))


def _spec_with_all_known(except_field: Optional[str] = None) -> Spec:
    """Return a Spec with all 7 fields KNOWN, optionally leaving one UNKNOWN."""
    spec = Spec()
    for f in Spec.FIELD_NAMES:
        if f != except_field:
            spec.update(f, f"value for {f}")
    return spec


# ---------------------------------------------------------------------------
# _parse_contract
# ---------------------------------------------------------------------------


def test_parse_contract_all_sections():
    response = (
        "FUNCTION_SIGNATURE: def foo(x: int) -> int:\n"
        "DOCSTRING: Does foo.\n  Args:\n    x: an integer\n"
        "EXAMPLES: foo(1)  # -> 1\n"
        "TEST_CASES: - foo(1) == 1\n- foo(0) == 0\n"
        "ERROR_HANDLING: Raise ValueError for negative x.\n"
    )
    result = _parse_contract(response)
    assert result["function_signature"] == "def foo(x: int) -> int:"
    assert "Does foo." in result["docstring"]
    assert "foo(1)" in result["examples"]
    assert "foo(1) == 1" in result["test_cases"]
    assert "ValueError" in result["error_handling"]


def test_parse_contract_empty_response():
    result = _parse_contract("")
    for key in ("function_signature", "docstring", "examples", "test_cases", "error_handling"):
        assert result[key] == ""


def test_parse_contract_partial_response():
    response = "FUNCTION_SIGNATURE: def bar() -> None:\nDOCSTRING: Does bar."
    result = _parse_contract(response)
    assert result["function_signature"] == "def bar() -> None:"
    assert result["docstring"] == "Does bar."
    assert result["examples"] == ""


def test_parse_contract_multiline_section():
    response = (
        "FUNCTION_SIGNATURE: def baz():\n"
        "DOCSTRING: Line one.\nLine two.\nLine three.\n"
        "ERROR_HANDLING: None.\n"
    )
    result = _parse_contract(response)
    assert "Line one." in result["docstring"]
    assert "Line two." in result["docstring"]
    assert "Line three." in result["docstring"]


def test_parse_contract_case_insensitive_headers():
    """Section header matching is case-insensitive."""
    response = "function_signature: def qux():\nDOCSTRING: Docs."
    result = _parse_contract(response)
    assert result["function_signature"] == "def qux():"


# ---------------------------------------------------------------------------
# ClarificationSubagent – construction
# ---------------------------------------------------------------------------


def test_subagent_default_construction():
    agent = ClarificationSubagent()
    assert isinstance(agent.spec, Spec)
    assert isinstance(agent.memory, Memory)
    assert agent.max_questions == 10


def test_subagent_accepts_shared_spec():
    spec = Spec()
    agent = ClarificationSubagent(spec=spec)
    assert agent.spec is spec


def test_subagent_accepts_shared_memory():
    mem = Memory()
    agent = ClarificationSubagent(memory=mem)
    assert agent.memory is mem


# ---------------------------------------------------------------------------
# _pick_next_field
# ---------------------------------------------------------------------------


def test_pick_next_field_all_unknown_returns_first():
    agent = ClarificationSubagent()
    assert agent._pick_next_field() == Spec.FIELD_NAMES[0]


def test_pick_next_field_prefers_conflicting_over_unknown():
    spec = _spec_with_all_known()
    # Trigger a conflict on "outputs" (not the first field)
    spec.update("outputs", "different value")
    agent = ClarificationSubagent(spec=spec)
    assert agent._pick_next_field() == "outputs"


def test_pick_next_field_returns_none_when_all_settled():
    spec = _spec_with_all_known()
    agent = ClarificationSubagent(spec=spec)
    assert agent._pick_next_field() is None


def test_pick_next_field_skips_known_fields():
    spec = Spec()
    spec.update("inputs", "v")  # KNOWN
    agent = ClarificationSubagent(spec=spec)
    field = agent._pick_next_field()
    assert field != "inputs"


# ---------------------------------------------------------------------------
# clarify() – stopping rules
# ---------------------------------------------------------------------------


def test_clarify_stops_immediately_when_already_known_enough():
    """All fields pre-populated: no questions asked, zero LLM calls."""
    spec = _spec_with_all_known()
    llm = MagicMock()
    ask_fn = MagicMock()
    agent = ClarificationSubagent(spec=spec, ask_fn=ask_fn)
    result = agent.clarify("write a sort function", llm)
    ask_fn.assert_not_called()
    llm.assert_not_called()
    assert result is spec


def test_clarify_budget_zero_fills_all_with_assumptions():
    """max_questions=0: all fields assumed, ask_fn never called."""
    llm = MagicMock(return_value="assumed value")
    ask_fn = MagicMock()
    agent = ClarificationSubagent(max_questions=0, ask_fn=ask_fn)
    spec = agent.clarify("write a function", llm)
    ask_fn.assert_not_called()
    assert llm.call_count == len(Spec.FIELD_NAMES)
    for f in Spec.FIELD_NAMES:
        assert spec.get_entry(f).status == FieldStatus.ASSUMED


def test_clarify_budget_zero_assumption_notes_set():
    llm = MagicMock(return_value="assumed value")
    agent = ClarificationSubagent(max_questions=0)
    spec = agent.clarify("prompt", llm)
    for f in Spec.FIELD_NAMES:
        assert spec.get_entry(f).assumption_note  # non-empty


def test_clarify_asks_question_and_updates_spec():
    """Single unknown field: one question asked, spec updated to KNOWN."""
    spec = _spec_with_all_known(except_field="inputs")
    llm = _seq_llm(
        "What are the input types?",  # _generate_question
        "A list of integers",  # _extract_spec_value
    )
    ask_fn = MagicMock(return_value="list of ints")
    agent = ClarificationSubagent(spec=spec, max_questions=5, ask_fn=ask_fn)
    agent.clarify("write a sort function", llm)
    ask_fn.assert_called_once()
    assert spec.get_entry("inputs").status == FieldStatus.KNOWN
    assert spec.get_entry("inputs").value == "A list of integers"


def test_clarify_answer_stored_in_memory():
    """Answer for a new question is persisted in memory."""
    spec = _spec_with_all_known(except_field="inputs")
    llm = _seq_llm("What are the input types?", "A list of integers")
    ask_fn = MagicMock(return_value="list of ints")
    mem = Memory()
    agent = ClarificationSubagent(spec=spec, memory=mem, max_questions=5, ask_fn=ask_fn)
    agent.clarify("write a sort function", llm)
    assert mem.recall("What are the input types?") == "list of ints"


def test_clarify_llm_returns_known_marks_field_without_asking():
    """When LLM returns 'KNOWN', field is inferred; ask_fn not called."""
    spec = _spec_with_all_known(except_field="inputs")
    llm = _seq_llm("KNOWN")
    ask_fn = MagicMock()
    agent = ClarificationSubagent(spec=spec, max_questions=5, ask_fn=ask_fn)
    agent.clarify("write a sort function", llm)
    ask_fn.assert_not_called()
    assert spec.get_entry("inputs").status == FieldStatus.KNOWN
    assert spec.get_entry("inputs").value == "(inferred from prompt)"


def test_clarify_reuses_cached_answer_without_asking():
    """A question already in memory is not re-asked."""
    spec = _spec_with_all_known(except_field="inputs")
    mem = Memory()
    mem.remember("What are the input types?", "list of ints")
    llm = _seq_llm(
        "What are the input types?",  # _generate_question matches cached question
        "A list of integers",  # _extract_spec_value
    )
    ask_fn = MagicMock()
    agent = ClarificationSubagent(spec=spec, memory=mem, max_questions=5, ask_fn=ask_fn)
    agent.clarify("write a sort function", llm)
    ask_fn.assert_not_called()
    assert spec.get_entry("inputs").value == "A list of integers"


def test_clarify_conflict_triggers_reconciliation():
    """A conflicting field is resolved before unknown ones."""
    spec = _spec_with_all_known()
    # Trigger a conflict in "outputs"
    spec.update("outputs", "different value")
    llm = _seq_llm(
        "Which is correct?",  # reconciliation question from _generate_question
        "original value",  # _extract_spec_value
    )
    ask_fn = MagicMock(return_value="original value")
    agent = ClarificationSubagent(spec=spec, max_questions=5, ask_fn=ask_fn)
    agent.clarify("write a sort function", llm)
    ask_fn.assert_called_once()
    assert spec.get_entry("outputs").status == FieldStatus.KNOWN
    assert spec.get_entry("outputs").conflict_candidate is None


def test_clarify_returns_spec_object():
    spec = _spec_with_all_known()
    llm = MagicMock()
    agent = ClarificationSubagent(spec=spec)
    result = agent.clarify("prompt", llm)
    assert result is spec


def test_clarify_budget_partially_exhausted_fills_remaining():
    """Budget of 1 answers one field; the rest get LLM assumptions."""
    spec = Spec()
    # Only mark "inputs" as known; everything else unknown
    spec.update("inputs", "known value")
    llm = _seq_llm(
        "Outputs question?",  # _generate_question for "outputs"
        "Returns a sorted list",  # _extract_spec_value for "outputs"
        # After budget=1, _fill_with_assumptions for remaining 5 unknown fields:
        "assumed constraints",
        "assumed edge_cases",
        "assumed environment",
        "assumed determinism",
        "assumed failure_modes",
    )
    ask_fn = MagicMock(return_value="a sorted list")
    agent = ClarificationSubagent(spec=spec, max_questions=1, ask_fn=ask_fn)
    agent.clarify("write a sort", llm)
    ask_fn.assert_called_once()
    # "outputs" was answered
    assert spec.get_entry("outputs").status == FieldStatus.KNOWN
    # remaining unknown fields became ASSUMED
    for f in ("constraints", "edge_cases", "environment", "determinism", "failure_modes"):
        assert spec.get_entry(f).status == FieldStatus.ASSUMED
    # 2 LLM calls for outputs question+extract + 5 for assumptions
    assert llm.call_count == 7


# ---------------------------------------------------------------------------
# to_contract
# ---------------------------------------------------------------------------

_CONTRACT_RESPONSE = (
    "FUNCTION_SIGNATURE: def sort_list(items: list) -> list:\n"
    "DOCSTRING: Sort a list.\n"
    "EXAMPLES: sort_list([3, 1])  # -> [1, 3]\n"
    "TEST_CASES: - sort_list([]) == []\n"
    "ERROR_HANDLING: Raise TypeError for non-list inputs.\n"
)


def test_to_contract_returns_all_keys():
    spec = _spec_with_all_known()
    llm = _seq_llm(_CONTRACT_RESPONSE)
    agent = ClarificationSubagent(spec=spec)
    contract = agent.to_contract("sort a list", llm)
    assert set(contract.keys()) == {
        "function_signature",
        "docstring",
        "examples",
        "test_cases",
        "error_handling",
    }


def test_to_contract_contains_expected_values():
    spec = _spec_with_all_known()
    llm = _seq_llm(_CONTRACT_RESPONSE)
    agent = ClarificationSubagent(spec=spec)
    contract = agent.to_contract("sort a list", llm)
    assert "sort_list" in contract["function_signature"]
    assert "Sort" in contract["docstring"]
    assert "sort_list" in contract["examples"]
    assert "sort_list" in contract["test_cases"]
    assert "TypeError" in contract["error_handling"]


def test_to_contract_calls_llm_exactly_once():
    spec = _spec_with_all_known()
    llm = _seq_llm(_CONTRACT_RESPONSE)
    agent = ClarificationSubagent(spec=spec)
    agent.to_contract("sort a list", llm)
    assert llm.call_count == 1


def test_to_contract_includes_assumption_note_in_llm_input():
    """Assumed fields should be flagged in the prompt sent to the LLM."""
    spec = _spec_with_all_known()
    spec.assume("environment", "Python 3.11", "inferred")
    llm = _seq_llm(_CONTRACT_RESPONSE)
    agent = ClarificationSubagent(spec=spec)
    agent.to_contract("sort a list", llm)
    # Inspect what was passed to the LLM
    call_arg = llm.call_args[0][0]
    assert "NEEDS CONFIRMATION" in call_arg
