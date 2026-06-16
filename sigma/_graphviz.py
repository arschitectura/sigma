"""Graphviz-based tree visualization for conditional inference trees."""

from __future__ import annotations

import typing

import graphviz
import numpy
import numpy.typing

from . import _extension
from . import _node
from . import _palette
from . import _partition
from . import _tree_text

_DEFAULT_ROOT_COLORS = ("white", "black", "black")
_DEFAULT_SPLIT_COLORS = ("black", "lightgray", "black")
_DEFAULT_LEAF_PALETTE = _palette._DEFAULT_LEAF_PALETTE


def _per_node_displayed_indices(
    node: _node.Node, top_displayed_items: None | int
) -> list[int]:
    """Return the node's own top items by lowest non-NaN expected rank.

    Args:
        node: Tree node. Returns [] for non-ranking nodes.
        top_displayed_items: Maximum number of items to return. None
            yields [].

    Returns:
        Sorted-by-expected-rank-ascending list of item indices (length
        at most top_displayed_items). Items with NaN expected rank are
        excluded.
    """
    if top_displayed_items is None:
        return []
    if not isinstance(node, _node.RankingNode):
        return []
    values = numpy.array([metric.value for metric in node.metrics], dtype=float)
    valid_indices = numpy.flatnonzero(~numpy.isnan(values))
    if valid_indices.size == 0:
        return []
    take = min(top_displayed_items, valid_indices.size)
    valid_values = values[valid_indices]
    order = numpy.argsort(valid_values, kind="stable")[:take]
    result = [int(idx) for idx in valid_indices[order]]
    return result


def _escape_html_with_bold(text: str) -> str:
    """Escape special characters and turn bold sentinels into HTML <b> tags."""
    escaped = (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br/>")
        .replace(_tree_text._BOLD_OPEN, "<b>")
        .replace(_tree_text._BOLD_CLOSE, "</b>")
    )
    return escaped


def _to_html_label(text: str) -> str:
    """Wrap an escaped HTML body in graphviz HTML-label angle brackets."""
    body = _escape_html_with_bold(text)
    result = f"<{body}>"
    return result


def _make_leaf_html_label(
    text_label: str,
    badge_number: int,
    leaf_foreground: str,
    leaf_background: str,
    width_pt: None | float = None,
) -> str:
    """Build an HTML-like graphviz label with a numbered badge."""
    escaped = _escape_html_with_bold(text_label)
    match width_pt:
        case None:
            outer_table_open = (
                '<<table border="0" cellborder="0" cellspacing="4"'
                ' cellpadding="0">'
            )
        case _:
            width_attr = int(round(width_pt))
            outer_table_open = (
                '<<table border="0" cellborder="0" cellspacing="4"'
                f' cellpadding="0" width="{width_attr}">'
            )
    html = (
        f"{outer_table_open}"
        f"<tr>"
        f'<td valign="top">'
        f'<table border="0" cellborder="0" cellspacing="0" cellpadding="0">'
        f"<tr>"
        f'<td width="20" height="20" fixedsize="true"'
        f' bgcolor="{leaf_foreground}" style="rounded"'
        f' align="center" valign="middle">'
        f'<font color="{leaf_background}" point-size="12">'
        f"<b>{badge_number}</b></font>"
        f"</td>"
        f"</tr>"
        f"</table>"
        f"</td>"
        f'<td align="left">{escaped}</td>'
        f"</tr>"
        f"</table>>"
    )
    return html


def _build_digraph(
    root: _node.Node,
    category_labels: None | dict[int, dict[float, str]],
    class_names: None | list[str],
    root_colors: tuple[str, str, str],
    split_colors: tuple[str, str, str],
    leaf_palette: tuple[str, str, str],
    foreground_color: str,
    feature_names: None | numpy.typing.NDArray = None,
    response_name: None | str = None,
    prediction_formatter: None | typing.Callable[[float], str] = None,
    background_color: None | str = None,
    max_depth: None | int = None,
    precision: int = 3,
    dpi: int = 192,
    orientation: typing.Literal["top-down", "left-to-right"] = "top-down",
    reverse_order: bool = False,
    top_displayed_items: None | int = None,
    max_branch_length: int = 60,
) -> graphviz.Digraph:
    """Build a graphviz Digraph from a fitted tree."""
    natural_dot = _emit_digraph(
        root,
        category_labels,
        class_names,
        root_colors,
        split_colors,
        leaf_palette,
        foreground_color,
        feature_names=feature_names,
        response_name=response_name,
        prediction_formatter=prediction_formatter,
        background_color=background_color,
        max_depth=max_depth,
        precision=precision,
        dpi=dpi,
        orientation=orientation,
        reverse_order=reverse_order,
        uniform_width=None,
        top_displayed_items=top_displayed_items,
        max_branch_length=max_branch_length,
    )
    uniform_width = _max_content_node_width(natural_dot)
    if uniform_width is None:
        return natural_dot
    final_dot = _emit_digraph(
        root,
        category_labels,
        class_names,
        root_colors,
        split_colors,
        leaf_palette,
        foreground_color,
        feature_names=feature_names,
        response_name=response_name,
        prediction_formatter=prediction_formatter,
        background_color=background_color,
        max_depth=max_depth,
        precision=precision,
        dpi=dpi,
        orientation=orientation,
        reverse_order=reverse_order,
        uniform_width=uniform_width,
        top_displayed_items=top_displayed_items,
        max_branch_length=max_branch_length,
    )
    return final_dot


def _emit_digraph(
    root: _node.Node,
    category_labels: None | dict[int, dict[float, str]],
    class_names: None | list[str],
    root_colors: tuple[str, str, str],
    split_colors: tuple[str, str, str],
    leaf_palette: tuple[str, str, str],
    foreground_color: str,
    feature_names: None | numpy.typing.NDArray = None,
    response_name: None | str = None,
    prediction_formatter: None | typing.Callable[[float], str] = None,
    background_color: None | str = None,
    max_depth: None | int = None,
    precision: int = 3,
    dpi: int = 192,
    orientation: typing.Literal["top-down", "left-to-right"] = "top-down",
    reverse_order: bool = False,
    uniform_width: None | float = None,
    top_displayed_items: None | int = None,
    max_branch_length: int = 60,
) -> graphviz.Digraph:
    """Emit a graphviz Digraph in a single pass, optionally forcing a width."""
    display_reverse = reverse_order ^ (orientation == "left-to-right")
    leaves = root.leaves()
    leaf_count = len(leaves)
    background_color = background_color or "white"
    match orientation:
        case "top-down":
            rankdir = "TB"
        case "left-to-right":
            rankdir = "LR"
    dot = graphviz.Digraph(
        graph_attr={
            "bgcolor": background_color,
            "rankdir": rankdir,
            "dpi": str(dpi),
        },
        node_attr={
            "shape": "box",
            "style": "filled",
            "fontsize": "9",
            "fontname": "sans-serif",
            "margin": "0.11,0.055",
        },
        edge_attr={
            "fontsize": "9",
            "fontname": "sans-serif",
        },
    )
    match uniform_width:
        case None:
            uniform_attrs: dict[str, str] = {}
            leaf_width_pt: None | float = None
        case _:
            uniform_attrs = {"width": f"{uniform_width:.4f}"}
            leaf_width_pt = max((uniform_width - 0.22) * 72.0, 1.0)
    stack: list[_node.Node] = [root]
    while stack:
        node = stack.pop()
        node_displayed = _per_node_displayed_indices(node, top_displayed_items)
        label = _tree_text._format_prediction(
            node,
            class_names,
            response_name,
            prediction_formatter,
            separator="\n",
            precision=precision,
            bold_value=True,
            displayed_indices=node_displayed,
        )
        node_object_id = id(node)
        node_id = str(node_object_id)
        decoration = node.decoration
        if decoration is None:
            decoration_suffix = ""
        else:
            decoration_text = str(decoration)
            decoration_suffix = f"\n{decoration_text}"
        match node.extension:
            case _partition.Partition() as partition:
                p_value_line = _tree_text._format_p_value(partition)
                label = f"{label}\n{p_value_line}{decoration_suffix}"
                foreground, background, border = (
                    root_colors if node is root else split_colors
                )
                split_html_label = _to_html_label(label)
                dot.node(
                    node_id,
                    label=split_html_label,
                    fontcolor=foreground,
                    fillcolor=background,
                    color=border,
                    **uniform_attrs,
                )
                if max_depth is None or node.depth < max_depth:
                    left_label, right_label = _tree_text._format_branch_labels(
                        partition,
                        category_labels,
                        feature_names,
                        precision=precision,
                    )
                    left_label = _tree_text._ellipsize(
                        left_label, max_branch_length
                    )
                    right_label = _tree_text._ellipsize(
                        right_label, max_branch_length
                    )
                    left_child, right_child, swapped = (
                        _node.ordered_display_children(
                            node, partition, display_reverse
                        )
                    )
                    if swapped:
                        left_label, right_label = right_label, left_label
                    left_child_object_id = id(left_child)
                    left_child_id = str(left_child_object_id)
                    dot.edge(
                        node_id,
                        left_child_id,
                        label=f"  {left_label}",
                        color=foreground_color,
                        fontcolor=foreground_color,
                    )
                    stack.append(left_child)
                    right_child_object_id = id(right_child)
                    right_child_id = str(right_child_object_id)
                    dot.edge(
                        node_id,
                        right_child_id,
                        label=f"  {right_label}",
                        color=foreground_color,
                        fontcolor=foreground_color,
                    )
                    stack.append(right_child)
                    continue
            case _extension.Leaf() as leaf:
                label = f"{label}{decoration_suffix}"
                leaf_id = leaf.leaf_id
                badge_number = leaf_id + 1
                leaf_background = _palette._leaf_color(
                    leaf_id, leaf_count, leaf_palette
                )
                leaf_foreground = _palette._contrast_foreground(leaf_background)
                html_label = _make_leaf_html_label(
                    label,
                    badge_number,
                    leaf_foreground,
                    leaf_background,
                    leaf_width_pt,
                )
                dot.node(
                    node_id,
                    label=html_label,
                    fontcolor=leaf_foreground,
                    fillcolor=leaf_background,
                    color=foreground_color,
                    **uniform_attrs,
                )
        if max_depth is not None:
            placeholder_id = f"trunc_{node_id}"
            dot.node(
                placeholder_id,
                label="...",
                fontcolor=split_colors[0],
                fillcolor=split_colors[1],
                color=split_colors[2],
            )
            dot.edge(node_id, placeholder_id, color=foreground_color)
    return dot


def _max_content_node_width(dot: graphviz.Digraph) -> None | float:
    """Return the max non-trunc node width (inches) from plain-format render.

    Returns None when fewer than two content nodes are emitted, signalling
    that there is nothing to unify.
    """
    plain_output = dot.pipe(format="plain").decode("utf-8")
    widths: list[float] = []
    for line in plain_output.splitlines():
        if not line.startswith("node "):
            continue
        tokens = line.split(None, 5)
        if len(tokens) < 5:
            continue
        name = tokens[1]
        if name.startswith("trunc_"):
            continue
        widths.append(float(tokens[4]))
    if len(widths) < 2:
        return None
    max_width = max(widths)
    return max_width
