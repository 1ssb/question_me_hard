"""Tests for question_me_hard.tree (binary question-tree visualizer)."""
from __future__ import annotations

import pytest

from question_me_hard.tree import (
    QNode,
    _sanitize_id,
    _wrap_label,
    build_binary_question_tree,
    default_spec_check_bank,
    to_dot,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def test_sanitize_id_basic():
    assert _sanitize_id("Hello World!") == "Hello_World"


def test_sanitize_id_empty():
    assert _sanitize_id("") == "node"


def test_sanitize_id_strips_underscores():
    assert _sanitize_id("  __hello__  ") == "hello"


def test_wrap_label_short():
    result = _wrap_label("short")
    assert result == "short"


def test_wrap_label_empty():
    assert _wrap_label("") == ""


def test_wrap_label_long():
    long_text = "a" * 100
    result = _wrap_label(long_text, width=36)
    assert "\n" in result


# ---------------------------------------------------------------------------
# default_spec_check_bank
# ---------------------------------------------------------------------------


def test_default_spec_check_bank_nonempty():
    bank = default_spec_check_bank()
    assert len(bank) > 0


def test_default_spec_check_bank_tuples():
    for item in default_spec_check_bank():
        q, ask = item
        assert isinstance(q, str) and q
        assert isinstance(ask, str) and ask


# ---------------------------------------------------------------------------
# build_binary_question_tree – structural tests
# ---------------------------------------------------------------------------


def test_build_depth_zero():
    """At depth 0, root becomes a leaf immediately."""
    root = build_binary_question_tree("Root question?", max_depth=0)
    assert root.depth == 0
    assert root.children == {}


def test_build_negative_depth_raises():
    with pytest.raises(ValueError, match="max_depth must be >= 0"):
        build_binary_question_tree("Root?", max_depth=-1)


def test_build_depth_one():
    """At depth 1 the root gets two children."""
    root = build_binary_question_tree("Root?", max_depth=1)
    assert set(root.children.keys()) == {"Yes", "No"}
    yes_child = root.children["Yes"]
    no_child = root.children["No"]
    assert yes_child.depth == 1
    assert no_child.depth == 1
    # No-branch must be an ASK USER leaf
    assert no_child.text.upper().startswith("ASK USER:")
    # ASK USER leaf must have no children
    assert no_child.children == {}


def test_build_yes_branch_continues():
    """Yes branch should expand further; No branch becomes a leaf immediately."""
    root = build_binary_question_tree("Root?", max_depth=2)
    yes1 = root.children["Yes"]
    # Yes child at depth 1 should itself have children (depth < max_depth and not ASK USER)
    assert set(yes1.children.keys()) == {"Yes", "No"}


def test_build_custom_branch_labels():
    root = build_binary_question_tree(
        "Root?", max_depth=1, branch_labels=("True", "False")
    )
    assert set(root.children.keys()) == {"True", "False"}


def test_build_custom_spec_checks():
    checks = [
        ("Is X defined?", "Please define X."),
        ("Is Y defined?", "Please define Y."),
    ]
    root = build_binary_question_tree("Root?", max_depth=2, spec_checks=checks)
    assert root.children  # tree was built


def test_build_custom_stop_fn():
    """Custom stop_fn can halt expansion early."""
    def stop_fn(node: QNode) -> bool:
        # stop after root
        return node.depth > 0

    root = build_binary_question_tree("Root?", max_depth=5, stop_fn=stop_fn)
    # Root should have children, but those children are leaves.
    assert root.children
    for child in root.children.values():
        assert child.children == {}


def test_build_custom_question_fn():
    """Custom question_fn controls child text."""

    def question_fn(parent: QNode, branch: str) -> str:
        return f"custom-{branch}-depth{parent.depth + 1}"

    root = build_binary_question_tree(
        "Root?", max_depth=1, question_fn=question_fn
    )
    assert root.children["Yes"].text == "custom-Yes-depth1"
    assert root.children["No"].text == "custom-No-depth1"


def test_build_unique_node_ids():
    """All nodes in the tree must have unique IDs."""
    root = build_binary_question_tree("Root?", max_depth=3)
    ids: list[str] = []
    queue = [root]
    while queue:
        node = queue.pop(0)
        ids.append(node.node_id)
        queue.extend(node.children.values())
    assert len(ids) == len(set(ids))


def test_build_node_id_prefix():
    """Node IDs should start with 'q'."""
    root = build_binary_question_tree("Root?", max_depth=2)
    queue = [root]
    while queue:
        node = queue.pop(0)
        assert node.node_id.startswith("q")
        queue.extend(node.children.values())


# ---------------------------------------------------------------------------
# to_dot
# ---------------------------------------------------------------------------


def test_to_dot_returns_string():
    root = build_binary_question_tree("Root?", max_depth=1)
    dot = to_dot(root)
    assert isinstance(dot, str)


def test_to_dot_contains_digraph():
    root = build_binary_question_tree("Root?", max_depth=1)
    dot = to_dot(root, graph_name="TestGraph")
    assert "digraph TestGraph" in dot


def test_to_dot_contains_all_node_ids():
    root = build_binary_question_tree("Root?", max_depth=1)
    dot = to_dot(root)
    queue = [root]
    while queue:
        node = queue.pop(0)
        assert node.node_id in dot
        queue.extend(node.children.values())


def test_to_dot_contains_edges():
    root = build_binary_question_tree("Root?", max_depth=1)
    dot = to_dot(root)
    assert "->" in dot


def test_to_dot_rankdir_option():
    root = build_binary_question_tree("Root?", max_depth=1)
    dot = to_dot(root, rankdir="LR")
    assert "rankdir=LR" in dot


def test_to_dot_depth_label_in_nodes():
    root = build_binary_question_tree("Root?", max_depth=1)
    dot = to_dot(root)
    assert "[d=0]" in dot
    assert "[d=1]" in dot


def test_to_dot_single_node():
    """A depth-0 tree (just the root, no children) should produce valid DOT."""
    root = build_binary_question_tree("Root?", max_depth=0)
    dot = to_dot(root)
    assert "digraph" in dot
    assert root.node_id in dot


# ---------------------------------------------------------------------------
# QNode dataclass
# ---------------------------------------------------------------------------


def test_qnode_defaults():
    node = QNode(node_id="q0", text="Hello?", depth=0)
    assert node.children == {}


def test_qnode_children_isolation():
    """Two QNodes must not share the same default children dict."""
    n1 = QNode(node_id="q1", text="A", depth=0)
    n2 = QNode(node_id="q2", text="B", depth=0)
    n1.children["Yes"] = QNode(node_id="q3", text="C", depth=1)
    assert "Yes" not in n2.children
