"""Tests for question_me_hard.spec (FieldStatus, SpecEntry, Spec)."""
from __future__ import annotations

import pytest

from question_me_hard.spec import FieldStatus, Spec, SpecEntry


# ---------------------------------------------------------------------------
# FieldStatus
# ---------------------------------------------------------------------------


def test_field_status_values():
    assert FieldStatus.UNKNOWN == "unknown"
    assert FieldStatus.KNOWN == "known"
    assert FieldStatus.CONFLICTING == "conflicting"
    assert FieldStatus.ASSUMED == "assumed"


def test_field_status_is_str():
    assert isinstance(FieldStatus.KNOWN, str)


# ---------------------------------------------------------------------------
# SpecEntry defaults
# ---------------------------------------------------------------------------


def test_spec_entry_defaults():
    entry = SpecEntry()
    assert entry.value is None
    assert entry.status == FieldStatus.UNKNOWN
    assert entry.assumption_note is None
    assert entry.conflict_candidate is None


# ---------------------------------------------------------------------------
# Spec – initialisation
# ---------------------------------------------------------------------------


def test_spec_all_fields_start_unknown():
    spec = Spec()
    for name in Spec.FIELD_NAMES:
        assert spec.get_entry(name).status == FieldStatus.UNKNOWN


def test_spec_field_names_contains_expected():
    for expected in (
        "inputs",
        "outputs",
        "constraints",
        "edge_cases",
        "environment",
        "determinism",
        "failure_modes",
    ):
        assert expected in Spec.FIELD_NAMES


def test_spec_required_fields_subset_of_field_names():
    assert set(Spec.REQUIRED_FIELDS).issubset(set(Spec.FIELD_NAMES))


def test_spec_required_fields_contains_expected():
    for expected in ("inputs", "outputs", "constraints", "edge_cases", "failure_modes"):
        assert expected in Spec.REQUIRED_FIELDS


# ---------------------------------------------------------------------------
# Spec.update – happy paths
# ---------------------------------------------------------------------------


def test_spec_update_unknown_to_known():
    spec = Spec()
    spec.update("inputs", "list of integers")
    entry = spec.get_entry("inputs")
    assert entry.status == FieldStatus.KNOWN
    assert entry.value == "list of integers"
    assert entry.assumption_note is None
    assert entry.conflict_candidate is None


def test_spec_update_assumed_flag():
    spec = Spec()
    spec.update("inputs", "list of integers", assumed=True, assumption_note="rough guess")
    entry = spec.get_entry("inputs")
    assert entry.status == FieldStatus.ASSUMED
    assert entry.assumption_note == "rough guess"


def test_spec_update_assumed_default_note():
    spec = Spec()
    spec.update("inputs", "v", assumed=True)
    assert spec.get_entry("inputs").assumption_note  # non-empty default


def test_spec_update_same_value_no_conflict():
    spec = Spec()
    spec.update("inputs", "list of integers")
    spec.update("inputs", "list of integers")
    assert spec.get_entry("inputs").status == FieldStatus.KNOWN


# ---------------------------------------------------------------------------
# Spec.update – conflict detection
# ---------------------------------------------------------------------------


def test_spec_update_conflict_detection():
    spec = Spec()
    spec.update("inputs", "list of integers")
    spec.update("inputs", "numpy array")  # triggers conflict
    entry = spec.get_entry("inputs")
    assert entry.status == FieldStatus.CONFLICTING
    assert entry.value == "list of integers"  # old value retained
    assert entry.conflict_candidate == "numpy array"


def test_spec_update_conflict_only_when_previously_known():
    # Starting from UNKNOWN should not trigger conflict
    spec = Spec()
    spec.update("inputs", "v1")  # UNKNOWN -> KNOWN
    spec.update("inputs", "v2")  # KNOWN -> CONFLICTING
    assert spec.get_entry("inputs").status == FieldStatus.CONFLICTING
    # But updating an ASSUMED field with a different value should overwrite, not conflict
    spec2 = Spec()
    spec2.update("inputs", "assumed", assumed=True)
    spec2.update("inputs", "new value")  # ASSUMED -> KNOWN (not CONFLICTING)
    assert spec2.get_entry("inputs").status == FieldStatus.KNOWN


def test_spec_update_invalid_field_raises():
    spec = Spec()
    with pytest.raises(KeyError, match="nonexistent"):
        spec.update("nonexistent", "value")


# ---------------------------------------------------------------------------
# Spec.assume
# ---------------------------------------------------------------------------


def test_spec_assume():
    spec = Spec()
    spec.assume("outputs", "sorted list", "inferred from context")
    entry = spec.get_entry("outputs")
    assert entry.status == FieldStatus.ASSUMED
    assert entry.value == "sorted list"
    assert "inferred from context" in entry.assumption_note


def test_spec_assume_default_note():
    spec = Spec()
    spec.assume("outputs", "v")
    assert spec.get_entry("outputs").assumption_note  # non-empty


def test_spec_assume_clears_conflict_candidate():
    spec = Spec()
    spec.update("outputs", "a")
    spec.update("outputs", "b")  # triggers conflict
    assert spec.get_entry("outputs").status == FieldStatus.CONFLICTING
    spec.assume("outputs", "c", "resolution")
    assert spec.get_entry("outputs").conflict_candidate is None
    assert spec.get_entry("outputs").status == FieldStatus.ASSUMED


def test_spec_assume_invalid_field_raises():
    spec = Spec()
    with pytest.raises(KeyError):
        spec.assume("bad_field", "value")


# ---------------------------------------------------------------------------
# Spec.resolve_conflict
# ---------------------------------------------------------------------------


def test_spec_resolve_conflict():
    spec = Spec()
    spec.update("constraints", "n <= 1000")
    spec.update("constraints", "n <= 10000")  # triggers conflict
    spec.resolve_conflict("constraints", "n <= 1000")
    entry = spec.get_entry("constraints")
    assert entry.status == FieldStatus.KNOWN
    assert entry.value == "n <= 1000"
    assert entry.conflict_candidate is None
    assert entry.assumption_note is None


def test_spec_resolve_conflict_invalid_field_raises():
    spec = Spec()
    with pytest.raises(KeyError):
        spec.resolve_conflict("bad_field", "value")


# ---------------------------------------------------------------------------
# Spec.is_known_enough
# ---------------------------------------------------------------------------


def test_spec_not_known_enough_initially():
    assert not Spec().is_known_enough()


def test_spec_known_enough_all_required_known():
    spec = Spec()
    for f in Spec.REQUIRED_FIELDS:
        spec.update(f, f"value for {f}")
    assert spec.is_known_enough()


def test_spec_known_enough_with_assumed():
    spec = Spec()
    for f in Spec.REQUIRED_FIELDS:
        spec.assume(f, f"assumed {f}")
    assert spec.is_known_enough()


def test_spec_not_known_enough_with_conflicting_required():
    spec = Spec()
    for f in Spec.REQUIRED_FIELDS:
        spec.update(f, "v1")
    # Introduce a conflict in one required field
    spec.update("inputs", "v2")  # triggers CONFLICTING
    assert not spec.is_known_enough()


def test_spec_known_enough_optional_fields_unknown():
    """Optional fields (environment, determinism) do not block is_known_enough."""
    spec = Spec()
    for f in Spec.REQUIRED_FIELDS:
        spec.update(f, f"value for {f}")
    # environment and determinism remain UNKNOWN
    assert spec.is_known_enough()


# ---------------------------------------------------------------------------
# Spec query helpers
# ---------------------------------------------------------------------------


def test_spec_unknown_fields():
    spec = Spec()
    spec.update("inputs", "v")
    unknowns = spec.unknown_fields()
    assert "inputs" not in unknowns
    assert "outputs" in unknowns


def test_spec_conflicting_fields():
    spec = Spec()
    spec.update("inputs", "a")
    spec.update("inputs", "b")
    assert "inputs" in spec.conflicting_fields()
    assert "outputs" not in spec.conflicting_fields()


def test_spec_assumed_fields():
    spec = Spec()
    spec.assume("determinism", "deterministic by default")
    assert "determinism" in spec.assumed_fields()
    assert "inputs" not in spec.assumed_fields()


def test_spec_field_label_contains_field_name():
    spec = Spec()
    label = spec.field_label("inputs")
    assert "input" in label.lower()


def test_spec_field_label_unknown_field_returns_name():
    spec = Spec()
    assert spec.field_label("nonexistent") == "nonexistent"


# ---------------------------------------------------------------------------
# Spec.to_dict
# ---------------------------------------------------------------------------


def test_spec_to_dict_keys():
    spec = Spec()
    d = spec.to_dict()
    assert set(d.keys()) == set(Spec.FIELD_NAMES)
    for name in Spec.FIELD_NAMES:
        assert "value" in d[name]
        assert "status" in d[name]
        assert "assumption_note" in d[name]


def test_spec_to_dict_reflects_updates():
    spec = Spec()
    spec.update("inputs", "list of ints")
    d = spec.to_dict()
    assert d["inputs"]["value"] == "list of ints"
    assert d["inputs"]["status"] == "known"
    assert d["outputs"]["status"] == "unknown"


def test_spec_to_dict_assumption_note_present():
    spec = Spec()
    spec.assume("environment", "Python 3.11", "standard assumption")
    d = spec.to_dict()
    assert "standard assumption" in d["environment"]["assumption_note"]
