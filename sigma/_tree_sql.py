"""SQL CASE-expression rendering for fitted Tree estimators."""

from __future__ import annotations

import typing

import numpy
import numpy.typing

from . import _extension, _feature, _metric, _node, _partition, _tree_text


def _collect_sql(
    root: _node.Node,
    metrics: tuple[_metric.Metric, ...],
    feature_names: None | numpy.typing.NDArray,
    category_labels: None | dict[int, dict[float, str]],
    target_class_index: None | int,
    max_depth: None | int,
    best_first: bool,
) -> str:
    """Build the expectations comment and SQL CASE expression for the tree
    rooted at root."""
    body = _build_sql_case(
        root,
        metrics,
        feature_names,
        category_labels,
        target_class_index,
        max_depth,
        best_first,
        indent_level=0,
    )
    header = _format_sql_expectations(
        root, feature_names, category_labels, max_depth
    )
    if header is None:
        return body
    result = f"{header}\n{body}"
    return result


def _format_sql_expectations(
    root: _node.Node,
    feature_names: None | numpy.typing.NDArray,
    category_labels: None | dict[int, dict[float, str]],
    max_depth: None | int,
) -> None | str:
    """Build the leading comment listing each referenced column and the SQL
    column type the expression expects, or None when the rendered expression
    references no column."""
    partitions: dict[int, _partition.Partition] = {}
    _collect_rendered_partitions(root, max_depth, partitions)
    if not partitions:
        return None
    parts: list[str] = []
    for index in sorted(partitions):
        partition = partitions[index]
        raw_name = _tree_text._resolve_feature_name(partition, feature_names)
        identifier = _format_sql_identifier(raw_name)
        kind = _expected_sql_kind(partition, category_labels)
        parts.append(f"{identifier} {kind}")
    listing = ", ".join(parts)
    line = f"-- Expects: {listing}"
    return line


def _collect_rendered_partitions(
    node: _node.Node,
    max_depth: None | int,
    partitions: dict[int, _partition.Partition],
) -> None:
    """Record the partition of each internal node rendered within max_depth."""
    extension = node.extension
    if isinstance(extension, _extension.Leaf):
        return
    if max_depth is not None and node.depth >= max_depth:
        return
    partition = typing.cast(_partition.Partition, extension)
    if partition.feature_index not in partitions:
        partitions[partition.feature_index] = partition
    for child in partition.children:
        _collect_rendered_partitions(child, max_depth, partitions)


def _expected_sql_kind(
    partition: _partition.Partition,
    category_labels: None | dict[int, dict[float, str]],
) -> str:
    """SQL column kind the partition's rendered conditions compare against."""
    match partition.feature:
        case _feature.PromotedBooleanFeature() | _feature.BooleanFeature():
            return "boolean"
        case _feature.CategoricalFeature():
            labels = _tree_text._feature_category_labels(
                partition, category_labels
            )
            if labels is None:
                return "numeric"
            return "text"
        case _:
            return "numeric"


def _build_sql_case(
    node: _node.Node,
    metrics: tuple[_metric.Metric, ...],
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
    branches = _node.display_branches(node, partition, best_first, metrics)
    raw_name = _tree_text._resolve_feature_name(partition, feature_names)
    feature = _format_sql_identifier(raw_name)
    labels = _tree_text._feature_category_labels(partition, category_labels)
    na_code = _tree_text._feature_missing_code(partition)
    is_promoted = isinstance(partition.feature, _feature.PromotedBooleanFeature)
    nan_child_node = None
    if (
        isinstance(partition, _partition.NumericalPartition)
        and partition.nan_child is not None
    ):
        nan_child_node = partition.children[partition.nan_child]
    indent = "    " * indent_level
    when_indent = "    " * (indent_level + 1)
    lines: list[str] = [f"{indent}CASE"]
    for condition, child in branches:
        sql_condition = _format_sql_condition(
            condition, feature, labels, na_code, is_promoted
        )
        if child is nan_child_node and isinstance(
            condition, _partition.NumericInterval
        ):
            sql_condition = f"{sql_condition} OR {feature} IS NULL"
        subexpression = _build_sql_case(
            child,
            metrics,
            feature_names,
            category_labels,
            target_class_index,
            max_depth,
            best_first,
            indent_level + 2,
        )
        lines.append(f"{when_indent}WHEN {sql_condition} THEN")
        lines.append(subexpression)
    fallback_value = node._sql_value(target_class_index)
    fallback_literal = _format_sql_numeric_literal(fallback_value)
    lines.append(f"{when_indent}ELSE {fallback_literal}")
    lines.append(f"{indent}END")
    result = "\n".join(lines)
    return result


def _format_leaf_line(
    node: _node.Node,
    target_class_index: None | int,
    indent_level: int,
    is_truncated: bool,
) -> str:
    """Render a single leaf (or truncated subtree) line with trailing
    comment."""
    indent = "    " * indent_level
    value = node._sql_value(target_class_index)
    value_literal = _format_sql_numeric_literal(value)
    if is_truncated:
        comment = f"Truncated at depth {node.depth}"
    else:
        leaf_extension = typing.cast(_extension.Leaf, node.extension)
        leaf_number = leaf_extension.leaf_id + 1
        comment = f"Leaf {leaf_number}"
    line = f"{indent}{value_literal} -- {comment}"
    return line


def _format_sql_condition(
    condition: _partition.BranchCondition,
    feature: str,
    labels: None | dict[float, str],
    na_code: None | float,
    is_promoted: bool,
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
        case _partition.MissingValue():
            return f"{feature} IS NULL"
        case _:
            subset = typing.cast(_partition.CategorySubset, condition)
            fragment = _format_sql_subset_condition(
                feature, subset, labels, na_code, is_promoted
            )
            return fragment


def _format_sql_interval_condition(
    feature: str, interval: _partition.NumericInterval
) -> str:
    """Return the SQL predicate for a numeric interval branch."""
    lower = interval.lower
    upper = interval.upper
    if lower is None and upper is None:
        return f"{feature} IS NOT NULL"
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
    na_code: None | float,
    is_promoted: bool,
) -> str:
    """Return the SQL predicate for a categorical subset branch. The N/A code
    is emitted as IS NULL; a promoted boolean is emitted with boolean
    literals rather than a string membership test.
    """
    sorted_cats = sorted(subset.categories)
    real_cats = [category for category in sorted_cats if category != na_code]
    parts: list[str] = []
    if is_promoted:
        for category in real_cats:
            literal = "TRUE" if category == 1.0 else "FALSE"
            parts.append(f"{feature} = {literal}")
    elif real_cats:
        if labels is None:
            items = [_format_sql_category_literal(c) for c in real_cats]
        else:
            items = [
                _format_sql_category_literal(labels.get(c, c))
                for c in real_cats
            ]
        if len(items) == 1:
            parts.append(f"{feature} = {items[0]}")
        else:
            listing = ", ".join(items)
            parts.append(f"{feature} IN ({listing})")
    if na_code is not None and na_code in subset.categories:
        parts.append(f"{feature} IS NULL")
    condition = " OR ".join(parts)
    return condition


def _format_sql_identifier(name: str) -> str:
    """Quote name as a SQL-92 double-quoted identifier with embedded
    escaping."""
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
