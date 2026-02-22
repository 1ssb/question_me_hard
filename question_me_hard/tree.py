"""Binary question-tree builder and Graphviz DOT exporter.

Builds a binary tree of "spec-check" questions, stopping when *max_depth* is
reached or a node's text starts with ``ASK USER:`` (the default "No" leaf).
The tree can be serialised to Graphviz DOT format; optional SVG/PNG rendering
requires the ``graphviz`` Python package and the Graphviz system binaries.

Install optional rendering support::

    pip install graphviz
    # and the system graphviz package (e.g. apt-get install graphviz)
"""

from __future__ import annotations

import re
import textwrap
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------


@dataclass
class QNode:
    """A node in a binary question tree."""

    node_id: str
    text: str
    depth: int
    # children keys are branch labels (e.g. "Yes"/"No") mapping to child nodes
    children: Dict[str, "QNode"] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sanitize_id(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_]+", "_", s.strip())
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "node"


def _wrap_label(s: str, width: int = 36) -> str:
    """Wrap *s* at *width* characters, inserting newlines for DOT labels."""
    return "\n".join(textwrap.wrap(s, width=width)) if s else ""


def default_spec_check_bank() -> Sequence[Tuple[str, str]]:
    """Return the default bank of (check_question, ask_user_prompt) tuples.

    These are "meta-questions" an agent asks itself about a prompt's quality.
    """
    return [
        (
            "Is the input format (types/shapes) explicitly specified?",
            "Ask the author to specify input types/shapes and examples.",
        ),
        (
            "Is the output format (types/shapes) explicitly specified?",
            "Ask the author to specify output types/shapes and examples.",
        ),
        (
            "Are constraints/bounds stated (n, value ranges, memory/time)?",
            "Ask the author for constraints/bounds and performance targets.",
        ),
        (
            "Are edge cases clarified (empty, ties, invalid input)?",
            "Ask the author how edge cases should be handled.",
        ),
        (
            "Are determinism and randomness requirements specified?",
            "Ask the author about determinism, seeding, and acceptable nondeterminism.",
        ),
        (
            "Are allowed dependencies/environment specified (Python version, libs)?",
            "Ask the author about environment and allowed dependencies.",
        ),
        (
            "Are error handling expectations specified (raise vs return)?",
            "Ask the author what to do on invalid inputs / failures.",
        ),
    ]


# ---------------------------------------------------------------------------
# Tree builder
# ---------------------------------------------------------------------------

QuestionFn = Callable[["QNode", str], str]
StopFn = Callable[["QNode"], bool]


def build_binary_question_tree(
    root_text: str,
    max_depth: int,
    branch_labels: Tuple[str, str] = ("Yes", "No"),
    spec_checks: Optional[Sequence[Tuple[str, str]]] = None,
    *,
    question_fn: Optional[QuestionFn] = None,
    stop_fn: Optional[StopFn] = None,
) -> QNode:
    """Build a binary question tree up to *max_depth* (root is depth 0).

    Semantics (default, for prompt-ambiguity handling):

    * Each internal node is a "spec-check" question.
    * **Yes** branch: spec exists – proceed to the next check.
    * **No** branch: spec missing – create a leaf ``ASK USER: …`` and stop
      there.

    Behaviour can be overridden with:

    * ``question_fn(parent_node, branch_label) -> child_text``
    * ``stop_fn(node) -> bool``  –  return ``True`` to make *node* a leaf.

    Args:
        root_text: Text for the root node (typically a spec-check question).
        max_depth: Maximum depth of the tree (root is depth 0).
        branch_labels: ``(yes_label, no_label)`` pair.
        spec_checks: Sequence of ``(question, ask_user_prompt)`` tuples.
            Defaults to :func:`default_spec_check_bank`.
        question_fn: Optional override for child-text generation.
        stop_fn: Optional override for leaf detection.

    Returns:
        The root :class:`QNode` of the built tree.

    Raises:
        ValueError: If *max_depth* is negative.
    """
    if max_depth < 0:
        raise ValueError("max_depth must be >= 0")

    yes_label, no_label = branch_labels
    checks = list(spec_checks) if spec_checks is not None else list(default_spec_check_bank())

    def _get_check(d: int) -> Tuple[str, str]:
        return checks[d % len(checks)]

    def _default_question_fn(parent: QNode, branch: str) -> str:
        next_q, ask_prompt = _get_check(parent.depth + 1)
        if branch == yes_label:
            return next_q
        if branch == no_label:
            return f"ASK USER: {ask_prompt}"
        return next_q  # fallback

    def _default_stop_fn(node: QNode) -> bool:
        if node.depth >= max_depth:
            return True
        if node.text.strip().upper().startswith("ASK USER:"):
            return True
        return False

    _question_fn = question_fn or _default_question_fn
    _stop_fn = stop_fn or _default_stop_fn

    root = QNode(node_id="q0", text=root_text, depth=0)
    queue = [root]
    next_id = 1

    while queue:
        node = queue.pop(0)
        if _stop_fn(node):
            continue

        for branch in (yes_label, no_label):
            child_text = _question_fn(node, branch)
            child = QNode(node_id=f"q{next_id}", text=child_text, depth=node.depth + 1)
            next_id += 1
            node.children[branch] = child
            queue.append(child)

    return root


# ---------------------------------------------------------------------------
# DOT export + optional render
# ---------------------------------------------------------------------------


def to_dot(
    root: QNode,
    *,
    graph_name: str = "QuestionTree",
    rankdir: str = "TB",
    node_shape: str = "box",
    fontname: str = "Helvetica",
) -> str:
    """Serialise *root* and its descendants to Graphviz DOT format.

    The returned string can be saved to a ``.dot`` file and rendered with any
    DOT-compatible viewer, or passed to :func:`render_dot`.

    Args:
        root: Root node of the tree.
        graph_name: Name embedded in the DOT ``digraph`` declaration.
        rankdir: Graph layout direction (``"TB"``, ``"LR"``, etc.).
        node_shape: Graphviz node shape attribute.
        fontname: Font used for nodes and edges.

    Returns:
        A DOT-format string.
    """
    lines: list[str] = []
    lines.append(f"digraph {_sanitize_id(graph_name)} {{")
    lines.append(f"  rankdir={rankdir};")
    lines.append(f'  node [shape={node_shape}, fontname="{fontname}"];')
    lines.append(f'  edge [fontname="{fontname}"];')

    queue = [root]
    seen: set[str] = set()

    while queue:
        node = queue.pop(0)
        if node.node_id in seen:
            continue
        seen.add(node.node_id)

        label = _wrap_label(f"[d={node.depth}] {node.text}")
        label = label.replace('"', r"\"")
        lines.append(f'  {node.node_id} [label="{label}"];')

        for branch, child in node.children.items():
            edge_label = _wrap_label(branch, width=12).replace('"', r"\"")
            lines.append(f'  {node.node_id} -> {child.node_id} [label="{edge_label}"];')
            queue.append(child)

    lines.append("}")
    return "\n".join(lines)


def render_dot(dot: str, out_path_no_ext: str, *, fmt: str = "svg") -> str:
    """Render a DOT string to an image file using the ``graphviz`` package.

    Requires the ``graphviz`` Python package (``pip install graphviz``) and the
    Graphviz system binaries (the ``dot`` executable).

    Args:
        dot: DOT-format string (e.g. from :func:`to_dot`).
        out_path_no_ext: Output file path *without* extension.
        fmt: Output format – ``"svg"`` (default), ``"png"``, etc.

    Returns:
        The path to the rendered file (with extension).

    Raises:
        RuntimeError: If the ``graphviz`` Python package is not installed.
    """
    try:
        from graphviz import Source  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "Missing Python package 'graphviz'. Install with: pip install graphviz"
        ) from exc

    src = Source(dot)
    return src.render(filename=out_path_no_ext, format=fmt, cleanup=True)
