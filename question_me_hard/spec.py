"""Structured specification state for a coding task.

Tracks each field of a task specification (inputs, outputs, constraints,
edge_cases, environment, determinism, failure_modes) with an explicit status:
UNKNOWN, KNOWN, CONFLICTING, or ASSUMED.

This is the *state* component of the clarification subagent – it makes
ambiguity measurable and gives the agent a clear termination criterion.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional


class FieldStatus(str, Enum):
    """Status of a single spec field."""

    UNKNOWN = "unknown"
    KNOWN = "known"
    CONFLICTING = "conflicting"
    ASSUMED = "assumed"


@dataclass
class SpecEntry:
    """Value and status for one field of the spec."""

    value: Optional[str] = None
    status: FieldStatus = FieldStatus.UNKNOWN
    assumption_note: Optional[str] = None
    #: New value that caused a CONFLICTING transition; set by :meth:`Spec.update`.
    conflict_candidate: Optional[str] = None


# Human-readable labels for each field (used in LLM prompts).
_FIELD_LABELS: Dict[str, str] = {
    "inputs": "input format (types, shapes, examples)",
    "outputs": "output format (types, shapes, examples)",
    "constraints": "constraints and bounds (n, value ranges, memory/time)",
    "edge_cases": "edge cases (empty, ties, invalid input)",
    "environment": "environment and dependencies (Python version, libs)",
    "determinism": "determinism and randomness requirements",
    "failure_modes": "error handling (raise vs return, failure rules)",
}


class Spec:
    """Structured specification state for a coding task.

    Maintains seven standard spec fields.  Each field carries a
    :class:`FieldStatus` that starts at UNKNOWN and transitions to KNOWN,
    CONFLICTING, or ASSUMED as the clarification session progresses.

    ``REQUIRED_FIELDS`` (inputs, outputs, constraints, edge_cases,
    failure_modes) must be KNOWN or ASSUMED before
    :meth:`is_known_enough` returns ``True``.
    """

    FIELD_NAMES: List[str] = list(_FIELD_LABELS.keys())
    REQUIRED_FIELDS: List[str] = [
        "inputs",
        "outputs",
        "constraints",
        "edge_cases",
        "failure_modes",
    ]

    def __init__(self) -> None:
        self._fields: Dict[str, SpecEntry] = {
            name: SpecEntry() for name in self.FIELD_NAMES
        }

    # ------------------------------------------------------------------
    # Mutation helpers
    # ------------------------------------------------------------------

    def update(
        self,
        field_name: str,
        value: str,
        *,
        assumed: bool = False,
        assumption_note: str = "",
    ) -> None:
        """Update *field_name* with *value*.

        If the field is already KNOWN and *value* differs from the stored
        value, the field transitions to CONFLICTING and the new value is
        kept as ``conflict_candidate`` for later reconciliation.

        Args:
            field_name: One of :attr:`FIELD_NAMES`.
            value: The new spec value.
            assumed: If ``True`` the field is tagged ASSUMED instead of KNOWN.
            assumption_note: Optional rationale stored with an assumption.

        Raises:
            KeyError: If *field_name* is not a recognised spec field.
        """
        if field_name not in self._fields:
            raise KeyError(f"Unknown spec field: {field_name!r}")
        entry = self._fields[field_name]
        if (
            entry.status == FieldStatus.KNOWN
            and entry.value is not None
            and entry.value != value
        ):
            entry.conflict_candidate = value
            entry.status = FieldStatus.CONFLICTING
        else:
            entry.value = value
            entry.conflict_candidate = None
            if assumed:
                entry.status = FieldStatus.ASSUMED
                entry.assumption_note = assumption_note or "Assumed; needs confirmation."
            else:
                entry.status = FieldStatus.KNOWN
                entry.assumption_note = None

    def assume(self, field_name: str, value: str, note: str = "") -> None:
        """Record an explicit assumption for *field_name*.

        The field is marked ASSUMED and *note* is stored as the assumption
        rationale (visible in :meth:`to_dict` and the contract output).

        Args:
            field_name: One of :attr:`FIELD_NAMES`.
            value: The assumed value.
            note: Human-readable rationale for the assumption.

        Raises:
            KeyError: If *field_name* is not a recognised spec field.
        """
        if field_name not in self._fields:
            raise KeyError(f"Unknown spec field: {field_name!r}")
        entry = self._fields[field_name]
        entry.value = value
        entry.status = FieldStatus.ASSUMED
        entry.assumption_note = note or "Assumed; needs confirmation."
        entry.conflict_candidate = None

    def resolve_conflict(self, field_name: str, resolved_value: str) -> None:
        """Resolve a conflicting field with the authoritative *resolved_value*.

        The field transitions back to KNOWN and ``conflict_candidate`` is
        cleared.

        Args:
            field_name: One of :attr:`FIELD_NAMES`.
            resolved_value: The value agreed upon after reconciliation.

        Raises:
            KeyError: If *field_name* is not a recognised spec field.
        """
        if field_name not in self._fields:
            raise KeyError(f"Unknown spec field: {field_name!r}")
        entry = self._fields[field_name]
        entry.value = resolved_value
        entry.status = FieldStatus.KNOWN
        entry.conflict_candidate = None
        entry.assumption_note = None

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_entry(self, field_name: str) -> SpecEntry:
        """Return the :class:`SpecEntry` for *field_name*.

        Raises:
            KeyError: If *field_name* is not a recognised spec field.
        """
        if field_name not in self._fields:
            raise KeyError(f"Unknown spec field: {field_name!r}")
        return self._fields[field_name]

    def is_known_enough(self) -> bool:
        """Return ``True`` when every required field is KNOWN or ASSUMED."""
        return all(
            self._fields[f].status in (FieldStatus.KNOWN, FieldStatus.ASSUMED)
            for f in self.REQUIRED_FIELDS
        )

    def unknown_fields(self) -> List[str]:
        """Return names of all UNKNOWN fields."""
        return [n for n in self.FIELD_NAMES if self._fields[n].status == FieldStatus.UNKNOWN]

    def conflicting_fields(self) -> List[str]:
        """Return names of all CONFLICTING fields."""
        return [
            n for n in self.FIELD_NAMES if self._fields[n].status == FieldStatus.CONFLICTING
        ]

    def assumed_fields(self) -> List[str]:
        """Return names of all ASSUMED fields."""
        return [n for n in self.FIELD_NAMES if self._fields[n].status == FieldStatus.ASSUMED]

    def field_label(self, field_name: str) -> str:
        """Return the human-readable label for *field_name*."""
        return _FIELD_LABELS.get(field_name, field_name)

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the spec to a plain dict (JSON-serialisable)."""
        return {
            name: {
                "value": entry.value,
                "status": entry.status.value,
                "assumption_note": entry.assumption_note,
            }
            for name, entry in self._fields.items()
        }

    def __repr__(self) -> str:  # pragma: no cover
        counts: Dict[FieldStatus, int] = {s: 0 for s in FieldStatus}
        for e in self._fields.values():
            counts[e.status] += 1
        return (
            f"Spec(known={counts[FieldStatus.KNOWN]}, "
            f"assumed={counts[FieldStatus.ASSUMED]}, "
            f"conflicting={counts[FieldStatus.CONFLICTING]}, "
            f"unknown={counts[FieldStatus.UNKNOWN]})"
        )
