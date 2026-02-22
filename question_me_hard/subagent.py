"""Clarification subagent – turns prompt ambiguity into managed spec-state.

Unlike a simple question-answer loop, :class:`ClarificationSubagent`:

* Maintains a structured :class:`~question_me_hard.spec.Spec` with explicit
  field statuses (KNOWN / UNKNOWN / CONFLICTING / ASSUMED).
* Selects the highest-value next question at each step: conflicting fields are
  resolved before unknown ones, and unknown fields are filled in priority order.
* Stops when the required spec fields are known or the question budget runs out.
* Records auditable assumptions rather than silently proceeding with ambiguity.
* Exports a drop-in coding contract (function signature, docstring, examples,
  test cases, error-handling rules) via :meth:`ClarificationSubagent.to_contract`.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

from .clarifier import LLMCallable
from .memory import Memory
from .spec import FieldStatus, Spec


# ---------------------------------------------------------------------------
# LLM prompt templates
# ---------------------------------------------------------------------------

_TARGETED_QUESTION_TEMPLATE = """\
You are clarifying a coding task specification. Generate exactly one focused \
question about the "{field_label}" to help implement the task.

Prompt:
{prompt}

Already-known spec:
{context}

Return exactly one question targeting "{field_label}", or the single word \
"KNOWN" if this aspect is already sufficiently clear from the prompt and context."""

_SPEC_EXTRACT_TEMPLATE = """\
Extract the "{field_label}" specification value from the answer below. \
Return a concise, implementable specification (1-2 sentences maximum).

Question asked: {question}
Answer given: {answer}

Extracted "{field_label}" spec value:"""

_RECONCILE_TEMPLATE = """\
The "{field_label}" specification has conflicting information:
  Earlier answer: {old_value}
  Conflicting answer: {new_value}

Generate exactly one question to resolve this conflict and establish the \
authoritative value."""

_ASSUMPTION_TEMPLATE = """\
Based on the prompt and any known context, provide a reasonable default \
assumption for the "{field_label}" of this task. This assumption will be \
recorded explicitly and flagged for confirmation. \
Return only the assumed value (1-2 sentences).

Prompt:
{prompt}

Known context:
{context}"""

_CONTRACT_TEMPLATE = """\
Given the specification below, produce a coding contract using these exact \
labeled sections (each section starts at the beginning of a line):

FUNCTION_SIGNATURE: <Python function signature, one line>
DOCSTRING: <description with Args, Returns, and Raises sections>
EXAMPLES: <2-3 usage examples as Python expressions>
TEST_CASES: <3-5 test case descriptions, each on its own line starting with "- ">
ERROR_HANDLING: <rules for invalid inputs and failure conditions>

Specification:
{spec}

Original prompt:
{prompt}"""

_CONTRACT_SECTIONS = (
    "FUNCTION_SIGNATURE",
    "DOCSTRING",
    "EXAMPLES",
    "TEST_CASES",
    "ERROR_HANDLING",
)


def _parse_contract(response: str) -> Dict[str, str]:
    """Parse a structured contract response from the LLM into a plain dict.

    Recognises the five canonical section headers (FUNCTION_SIGNATURE,
    DOCSTRING, EXAMPLES, TEST_CASES, ERROR_HANDLING) and collects all lines
    that follow each header until the next header (or end of string).

    Args:
        response: Raw LLM response containing the labelled sections.

    Returns:
        A dict with keys ``function_signature``, ``docstring``, ``examples``,
        ``test_cases``, ``error_handling``; values default to ``""`` for any
        section not found in *response*.
    """
    result: Dict[str, str] = {s.lower(): "" for s in _CONTRACT_SECTIONS}
    current: Optional[str] = None
    pending: List[str] = []

    for line in response.splitlines():
        matched = False
        for section in _CONTRACT_SECTIONS:
            if line.upper().startswith(section + ":"):
                if current is not None:
                    result[current] = "\n".join(pending).strip()
                current = section.lower()
                rest = line[len(section) + 1 :].strip()
                pending = [rest] if rest else []
                matched = True
                break
        if not matched and current is not None:
            pending.append(line)

    if current is not None:
        result[current] = "\n".join(pending).strip()

    return result


# ---------------------------------------------------------------------------
# ClarificationSubagent
# ---------------------------------------------------------------------------


class ClarificationSubagent:
    """Manages prompt clarification as a structured spec-state estimation problem.

    Unlike a simple Q&A loop, this subagent maintains a :class:`Spec` with
    seven fields (inputs, outputs, constraints, edge_cases, environment,
    determinism, failure_modes), each carrying an explicit status.

    **Policy**: conflicting fields are resolved first; unknown fields are
    filled in priority order (inputs → outputs → constraints → edge_cases →
    environment → determinism → failure_modes).

    **Stopping rules** (whichever fires first):

    1. All required spec fields are KNOWN or ASSUMED
       (:meth:`Spec.is_known_enough`).
    2. The question budget (:attr:`max_questions`) is exhausted.
    3. No UNKNOWN or CONFLICTING fields remain.

    **Auditable assumptions**: any field still UNKNOWN when the budget is
    exhausted is filled via an LLM-generated assumption tagged
    ASSUMED / needs confirmation.

    **No repeats**: previously answered questions stored in :attr:`memory` are
    reused without re-asking the user.

    **Contract export**: :meth:`to_contract` synthesises the final spec into a
    function signature, docstring, usage examples, test-case descriptions, and
    error-handling rules – a drop-in artefact for a coding agent.

    Args:
        spec: Optional :class:`Spec` to populate.  A fresh one is created if
            not supplied.
        memory: Optional :class:`Memory` for reusing previous answers.
        max_questions: Maximum number of *new* questions to ask the user per
            :meth:`clarify` call.
        ask_fn: Callable used to prompt the human.  Defaults to built-in
            :func:`input`.
    """

    def __init__(
        self,
        spec: Optional[Spec] = None,
        memory: Optional[Memory] = None,
        max_questions: int = 10,
        ask_fn: Optional[Callable[[str], str]] = None,
    ) -> None:
        self.spec: Spec = spec if spec is not None else Spec()
        self.memory: Memory = memory if memory is not None else Memory()
        self.max_questions = max_questions
        self._ask_fn: Callable[[str], str] = ask_fn if ask_fn is not None else input

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _build_context(self) -> str:
        known = [(name, e) for name, e in self.spec._fields.items() if e.value is not None]
        if not known:
            return "(none)"
        lines = []
        for name, e in known:
            note = " [NEEDS CONFIRMATION]" if e.status == FieldStatus.ASSUMED else ""
            lines.append(f"- {name}: {e.value}{note}")
        return "\n".join(lines)

    def _pick_next_field(self) -> Optional[str]:
        """Return the highest-priority field still needing clarification.

        Conflicting fields take precedence over unknown ones.  Returns
        ``None`` when every field is KNOWN or ASSUMED.
        """
        for name in Spec.FIELD_NAMES:
            if self.spec.get_entry(name).status == FieldStatus.CONFLICTING:
                return name
        for name in Spec.FIELD_NAMES:
            if self.spec.get_entry(name).status == FieldStatus.UNKNOWN:
                return name
        return None

    def _generate_question(
        self, prompt: str, field: str, llm_fn: LLMCallable
    ) -> Optional[str]:
        """Ask the LLM to produce a targeted question for *field*.

        Returns ``None`` when the LLM responds with the single word
        ``"KNOWN"`` (the field is already clear from context).
        """
        entry = self.spec.get_entry(field)
        label = self.spec.field_label(field)

        if entry.status == FieldStatus.CONFLICTING:
            query = _RECONCILE_TEMPLATE.format(
                field_label=label,
                old_value=entry.value or "",
                new_value=entry.conflict_candidate or "(conflicting value)",
            )
        else:
            query = _TARGETED_QUESTION_TEMPLATE.format(
                field_label=label,
                prompt=prompt,
                context=self._build_context(),
            )

        response = llm_fn(query).strip()
        return None if response.upper() == "KNOWN" else response

    def _extract_spec_value(
        self, field: str, question: str, answer: str, llm_fn: LLMCallable
    ) -> str:
        """Use the LLM to extract a concise spec value from a raw *answer*."""
        query = _SPEC_EXTRACT_TEMPLATE.format(
            field_label=self.spec.field_label(field),
            question=question,
            answer=answer,
        )
        return llm_fn(query).strip()

    def _fill_with_assumptions(self, prompt: str, llm_fn: LLMCallable) -> None:
        """Record LLM-generated assumptions for every still-UNKNOWN field."""
        for field in Spec.FIELD_NAMES:
            if self.spec.get_entry(field).status == FieldStatus.UNKNOWN:
                query = _ASSUMPTION_TEMPLATE.format(
                    field_label=self.spec.field_label(field),
                    prompt=prompt,
                    context=self._build_context(),
                )
                value = llm_fn(query).strip()
                self.spec.assume(
                    field, value, note="Assumed from context; needs confirmation."
                )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def clarify(self, prompt: str, llm_fn: LLMCallable) -> Spec:
        """Run the clarification loop until the spec converges or the budget is hit.

        On each iteration the subagent:

        1. Checks whether all required spec fields are already known enough
           (fast-exit).
        2. Picks the highest-priority field still needing clarification.
        3. Generates a targeted question via *llm_fn*.
        4. If the question was answered before (memory hit), reuses the answer
           without calling *ask_fn*.
        5. Otherwise asks the user via :attr:`_ask_fn`, stores the answer, and
           extracts a concise spec value via *llm_fn*.
        6. Updates or resolves the spec field.

        After the loop exits, any remaining UNKNOWN fields are filled with
        explicit LLM-generated assumptions tagged ``ASSUMED / needs
        confirmation``.

        Args:
            prompt: The original, potentially ambiguous prompt.
            llm_fn: A callable ``(str) -> str`` backed by any LLM.

        Returns:
            The populated :class:`Spec` (also accessible as :attr:`spec`).
        """
        questions_asked = 0

        while questions_asked < self.max_questions:
            if self.spec.is_known_enough():
                break

            field = self._pick_next_field()
            if field is None:
                break

            question = self._generate_question(prompt, field, llm_fn)
            if question is None:
                # LLM says the field is already clear from context.
                self.spec.update(field, "(inferred from prompt)")
                continue

            # Reuse a previously-given answer without re-asking.
            cached = self.memory.recall(question)
            if cached is not None:
                spec_value = self._extract_spec_value(field, question, cached, llm_fn)
                if self.spec.get_entry(field).status == FieldStatus.CONFLICTING:
                    self.spec.resolve_conflict(field, spec_value)
                else:
                    self.spec.update(field, spec_value)
                continue

            # Ask the user for a fresh answer.
            answer = self._ask_fn(f"{question}\n> ")
            self.memory.remember(question, answer)
            questions_asked += 1

            spec_value = self._extract_spec_value(field, question, answer, llm_fn)
            if self.spec.get_entry(field).status == FieldStatus.CONFLICTING:
                self.spec.resolve_conflict(field, spec_value)
            else:
                self.spec.update(field, spec_value)

        # Ensure no field is left silently unknown.
        self._fill_with_assumptions(prompt, llm_fn)
        return self.spec

    def to_contract(self, prompt: str, llm_fn: LLMCallable) -> Dict[str, str]:
        """Synthesise a drop-in coding contract from the populated spec.

        The contract is the primary output of the clarification session – a
        machine-readable artefact a coding agent can compile into an
        implementation.  Any ASSUMED fields are flagged in the spec summary
        passed to the LLM so that the contract makes assumptions visible.

        Args:
            prompt: The original prompt (for context in the LLM call).
            llm_fn: A callable ``(str) -> str`` backed by any LLM.

        Returns:
            A dict with keys: ``function_signature``, ``docstring``,
            ``examples``, ``test_cases``, ``error_handling``.
        """
        spec_lines = []
        for name, entry in self.spec._fields.items():
            value = entry.value or "(unknown)"
            note = (
                f" [NEEDS CONFIRMATION: {entry.assumption_note}]"
                if entry.assumption_note
                else ""
            )
            spec_lines.append(f"{name}: {value}{note}")

        query = _CONTRACT_TEMPLATE.format(
            spec="\n".join(spec_lines),
            prompt=prompt,
        )
        return _parse_contract(llm_fn(query))
