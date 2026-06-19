"""SQL CASE-expression rendering for fitted Tree estimators."""

from __future__ import annotations

import typing

import numpy
import numpy.typing

from . import _extension
from . import _node
from . import _partition
from . import _tree_text

if typing.TYPE_CHECKING:
    from . import _tree


def _collect_sql(
    root: _node.Node,
    feature_names: None | numpy.typing.NDArray,
    category_labels: None | dict[int, dict[float, str]],
    target_class_index: None | int,
    max_depth: None | int,
    best_first: bool,
) -> str:
    """Build the SQL CASE expression for the tree rooted at root."""
    result = _build_sql_case(
        root,
        feature_names,
        category_labels,
        target_class_index,
        max_depth,
        best_first,
        indent_level=0,
    )
    return result


def _build_sql_case(
    node: _node.Node,
    feature_names: None | numpy.typing.NDArray,
    category_labels: None | dict[int, dict[float, str]],
    target_class_index: None | int,
    max_depth: None | int,
    best_first: bool,
    indent_level: int,
) -> str:
    """Recursively render the SQL expression for the subtree at node."""
    extension = node.extension
    is_truncated = max_depth is not None and node.depth >= max_depth
    if isinstance(extension, _extension.Leaf) or is_truncated:
        line = _format_leaf_line(
            node, target_class_index, indent_level, is_truncated
        )
        return line
    partition = typing.cast(_partition.Partition, extension)
    branches = _node.display_branches(node, partition, best_first)
    raw_name = _tree_text._resolve_feature_name(partition, feature_names)
    feature = _format_sql_identifier(raw_name)
    labels = _tree_text._feature_category_labels(partition, category_labels)
    indent = "    " * indent_level
    when_indent = "    " * (indent_level + 1)
    lines: list[str] = [f"{indent}CASE"]
    for condition, child in branches:
        sql_condition = _format_sql_condition(condition, feature, labels)
        subexpression = _build_sql_case(
            child,
            feature_names,
            category_labels,
            target_class_index,
            max_depth,
            best_first,
            indent_level + 2,
        )
        lines.append(f"{when_indent}WHEN {sql_condition} THEN")
        lines.append(subexpression)
    if isinstance(partition, _partition.CategoricalPartition):
        fallback_value = _leaf_numeric_value(node, target_class_index)
        fallback_literal = _format_sql_numeric_literal(fallback_value)
        else_clause = f"{when_indent}ELSE {fallback_literal}"
    else:
        else_clause = f"{when_indent}ELSE NULL"
    lines.append(else_clause)
    lines.append(f"{indent}END")
    result = "\n".join(lines)
    return result


def _format_leaf_line(
    node: _node.Node,
    target_class_index: None | int,
    indent_level: int,
    is_truncated: bool,
) -> str:
    """Render a single leaf (or truncated subtree) line with trailing comment."""
    indent = "    " * indent_level
    value = _leaf_numeric_value(node, target_class_index)
    value_literal = _format_sql_numeric_literal(value)
    if is_truncated:
        comment = f"Truncated at depth {node.depth}"
    else:
        leaf_extension = typing.cast(_extension.Leaf, node.extension)
        leaf_number = leaf_extension.leaf_id + 1
        comment = f"Leaf {leaf_number}"
    line = f"{indent}{value_literal} -- {comment}"
    return line


def _leaf_numeric_value(
    node: _node.Node, target_class_index: None | int
) -> float:
    """Return the numeric value emitted at a SQL terminal (leaf, truncated
    subtree, or internal-node categorical fallback)."""
    if isinstance(node, _node.ClassificationNode):
        if target_class_index is None:
            raise RuntimeError(
                "target_class_index must be resolved before rendering a"
                " classification leaf"
            )
        value = float(node.class_distribution[target_class_index])
        return value
    if isinstance(node, _node.SurvivalNode):
        value = float(node.prediction)
        return value
    if isinstance(node, _node.RankingNode):
        raise NotImplementedError(
            "SQL export is not supported for RankingTree: a single SQL"
            " scalar cannot represent the per-item expected-rank vector"
            " predicted at each leaf."
        )
    regression_node = typing.cast(_node.RegressionNode, node)
    value = float(regression_node.prediction)
    return value


def _format_sql_condition(
    condition: _partition.BranchCondition,
    feature: str,
    labels: None | dict[float, str],
) -> str:
    """Return the SQL predicate fragment for a single branch condition."""
    match condition:
        case _partition.BooleanValue() as boolean:
            if boolean.value:
                return f"{feature}"
            return f"NOT {feature}"
        case _partition.NumericInterval() as interval:
            fragment = _format_sql_interval_condition(feature, interval)
            return fragment
        case _:
            subset = typing.cast(_partition.CategorySubset, condition)
            fragment = _format_sql_subset_condition(feature, subset, labels)
            return fragment


def _format_sql_interval_condition(
    feature: str, interval: _partition.NumericInterval
) -> str:
    """Return the SQL predicate for a numeric interval branch."""
    lower = interval.lower
    upper = interval.upper
    if lower is None:
        upper_literal = _format_sql_numeric_literal(
            typing.cast(int | float, upper)
        )
        return f"{feature} <= {upper_literal}"
    if upper is None:
        lower_literal = _format_sql_numeric_literal(lower)
        return f"{feature} > {lower_literal}"
    lower_literal = _format_sql_numeric_literal(lower)
    upper_literal = _format_sql_numeric_literal(upper)
    return f"{feature} > {lower_literal} AND {feature} <= {upper_literal}"


def _format_sql_subset_condition(
    feature: str,
    subset: _partition.CategorySubset,
    labels: None | dict[float, str],
) -> str:
    """Return the SQL predicate for a categorical subset branch."""
    sorted_cats = sorted(subset.categories)
    if labels is None:
        items = [
            _format_sql_category_literal(category) for category in sorted_cats
        ]
    else:
        items = [
            _format_sql_category_literal(labels.get(category, category))
            for category in sorted_cats
        ]
    if len(items) == 1:
        condition = f"{feature} = {items[0]}"
    else:
        listing = ", ".join(items)
        condition = f"{feature} IN ({listing})"
    return condition


def _format_sql_identifier(name: str) -> str:
    """Quote name as a SQL-92 double-quoted identifier with embedded escaping."""
    escaped = name.replace('"', '""')
    quoted = f'"{escaped}"'
    return quoted


def _format_sql_category_literal(value: object) -> str:
    """Quote a category value as a SQL literal."""
    if isinstance(value, str):
        escaped = value.replace("'", "''")
        literal = f"'{escaped}'"
        return literal
    literal = _format_sql_numeric_literal(value)
    return literal


def _format_sql_numeric_literal(value: object) -> str:
    """Format a numeric value as a SQL literal (NULL for non-finite floats)."""
    if isinstance(value, bool):
        literal = "TRUE" if value else "FALSE"
        return literal
    if isinstance(value, int):
        literal = repr(value)
        return literal
    float_value = float(typing.cast(float, value))
    if not numpy.isfinite(float_value):
        return "NULL"
    literal = repr(float_value)
    return literal


def _resolve_target_class_index(
    tree: _tree.Tree, target_class: None | object
) -> None | int:
    """Resolve target_class to an integer index into tree.classes_."""
    from . import _tree_classification

    is_classification = isinstance(
        tree, _tree_classification.ClassificationTree
    )
    if not is_classification:
        if target_class is not None:
            raise ValueError(
                "target_class is only valid for ClassificationTree;"
                f" got {type(tree).__name__}"
            )
        return None
    classes = tree.classes_
    if target_class is None:
        index = len(classes) - 1
        return index
    matches = numpy.where(classes == target_class)[0]
    if len(matches) == 0:
        raise ValueError(
            f"target_class={target_class!r} not found in tree.classes_;"
            f" valid options are {list(classes)}"
        )
    index = int(matches[0])
    return index
