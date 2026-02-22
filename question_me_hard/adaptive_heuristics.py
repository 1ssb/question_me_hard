"""
Smart Convergence Heuristics - LLM-driven, not WIDTH/DEPTH facades

Instead of hardcoding "7 fields" and "max_depth 5", the system uses intelligent
heuristics where the LLM decides:
- Which dimensions matter for THIS prompt
- When each dimension has sufficient clarity
- When the overall spec is ready for implementation
- Why it's stopping (explained to the user)
"""

from typing import Dict, List, Tuple

from .clarifier import LLMCallable
from .spec import Spec, FieldStatus


class ClarificationStrategy:
    """Smart clarification strategy with LLM-driven convergence"""

    def __init__(self, clarification_level: str = "normal"):
        """
        Args:
            clarification_level: "quick" (minimal), "normal" (standard), "thorough" (deep)
                - Informs the LLM's tolerance for ambiguity
        """
        self.clarification_level = clarification_level
        self.tolerance_map = {
            "quick": 0.6,  # LLM can stop at 60% clarity
            "normal": 0.75,  # 75% clarity threshold
            "thorough": 0.95,  # 95% clarity threshold (very deep)
        }
        self.clarity_threshold = self.tolerance_map.get(clarification_level, 0.75)

    def assess_prompt_dimensions(self, prompt: str, llm_fn: LLMCallable) -> Dict[str, float]:
        """
        LLM assesses which dimensions are relevant and how clear each is.
        Returns: dict of dimension_name -> clarity_score (0-1)
        
        Note: LLM decides which dimensions exist, not hardcoded 7.
        """
        query = f"""\
Analyze this specification prompt and identify ALL dimensions that need clarification:

"{prompt}"

For EACH dimension you identify, assess its current clarity (0.0 = completely vague, 1.0 = crystal clear).
A "dimension" is any aspect/requirement the system needs to understand.

Format your response EXACTLY as:
dimension_name: <score>
dimension_name: <score>
...

Examples of dimensions: input_types, output_format, performance_bounds, error_handling, database_persistence, etc.

Respond with 2-8 dimensions (be concise, only list what matters for THIS prompt):
"""
        response = llm_fn(query).strip()

        dimensions = {}
        for line in response.split("\n"):
            line = line.strip()
            if ":" in line and not line.startswith("#"):
                try:
                    dim_name, score_str = line.split(":", 1)
                    dim_name = dim_name.strip().lower().replace(" ", "_")
                    score = float(score_str.strip())
                    score = max(0.0, min(1.0, score))
                    dimensions[dim_name] = score
                except (ValueError, IndexError):
                    pass

        return dimensions if dimensions else {"input_specification": 0.3, "output_specification": 0.3}

    def assess_field_clarity(
        self, field_name: str, conversation: List[Tuple[str, str]], llm_fn: LLMCallable
    ) -> Tuple[float, str]:
        """
        Assess clarity of a single field based on conversation history.
        
        Returns:
            (clarity_score, explanation)
        """
        conversation_text = "\n".join(
            [f"Q: {q}\nA: {a}" for q, a in conversation]
        )

        query = f"""\
Based on the conversation below about "{field_name}", assess the clarity of understanding:

{conversation_text}

Rate clarity on 0.0 (still vague) to 1.0 (fully understood).
Also briefly explain what you understand or what's still unclear.

Format:
CLARITY: <0-1 score>
EXPLANATION: <1 sentence>
"""
        response = llm_fn(query).strip()

        score = 0.5
        explanation = "Unable to assess"

        for line in response.split("\n"):
            if line.startswith("CLARITY:"):
                try:
                    score = float(line.replace("CLARITY:", "").strip())
                    score = max(0.0, min(1.0, score))
                except ValueError:
                    pass
            elif line.startswith("EXPLANATION:"):
                explanation = line.replace("EXPLANATION:", "").strip()

        return (score, explanation)

    def should_continue_drilling(
        self, field_name: str, clarity_score: float, conversation: List[Tuple[str, str]],
        llm_fn: LLMCallable
    ) -> Tuple[bool, str]:
        """
        Should we ask more questions about this field?
        
        Returns:
            (should_continue, reason)
        """
        if clarity_score >= self.clarity_threshold:
            return (False, f"Clarity {clarity_score:.0%} meets threshold {self.clarity_threshold:.0%}")

        if len(conversation) >= 8:
            return (False, f"Max conversation length (8) reached for {field_name}")

        query = f"""\
Given the conversation so far about "{field_name}":

{chr(10).join([f"Q: {q}\\nA: {a}" for q, a in conversation])}

Would asking MORE questions about this dimension yield valuable insight?
Or is further drilling likely to be redundant?

Respond with ONLY:
- "CONTINUE: [your next question]" - ask more
- "STOP: [reason]" - sufficient clarity achieved
"""
        response = llm_fn(query).strip()

        if response.startswith("STOP:"):
            reason = response.replace("STOP:", "").strip()
            return (False, reason)
        elif response.startswith("CONTINUE:"):
            return (True, "LLM recommends further drilling")
        else:
            return (clarity_score < 0.8, "Unclear response, continuing")

    def assess_overall_convergence(
        self, prompt: str, spec: Spec, llm_fn: LLMCallable
    ) -> Tuple[bool, str]:
        """
        Is the overall specification ready for implementation?
        
        Returns:
            (is_converged, explanation)
        """
        # Build specification summary
        spec_summary = "\n".join(
            [
                f"{field}: {spec.get_entry(field).value or '(unknown)'}"
                for field in Spec.FIELD_NAMES
            ]
        )

        query = f"""\
Original prompt: "{prompt}"

Current specification:
{spec_summary}

Can a competent engineer implement this based on the current spec?
Consider: ambiguities, missing edge cases, unclear requirements.

Respond with:
- "READY: [brief reason]" - sufficient to code
- "NEEDS_MORE: [specific gaps]" - needs more clarification
"""
        response = llm_fn(query).strip()

        if response.startswith("READY:"):
            reason = response.replace("READY:", "").strip()
            return (True, reason)
        else:
            reason = response.replace("NEEDS_MORE:", "").strip()
            return (False, reason)

    def explain_convergence(self, spec: Spec, stop_reasons: Dict[str, str]) -> str:
        """Build explainable summary of why/where clarification stopped"""
        output = []
        output.append(f"\nClarification Tolerance: {self.clarification_level}")
        output.append(f"Clarity Threshold: {self.clarity_threshold:.0%}\n")

        output.append("Convergence Summary:")
        output.append("-" * 80)

        for field in Spec.FIELD_NAMES:
            entry = spec.get_entry(field)
            status_icon = {
                FieldStatus.KNOWN: "✓",
                FieldStatus.ASSUMED: "⊙",
                FieldStatus.CONFLICTING: "⚠",
                FieldStatus.UNKNOWN: "?",
            }.get(entry.status, "?")

            reason = stop_reasons.get(field, "")
            if reason:
                output.append(f"  {status_icon} {field:<20} {reason}")
            else:
                output.append(f"  {status_icon} {field:<20} {entry.status.value}")

        return "\n".join(output)
