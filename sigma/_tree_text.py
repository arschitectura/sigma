"""Text/rendering helpers shared by the Tree estimators and exporters.

These helpers format predictions, branch labels, p-values, and tabular
rows.
"""

from __future__ import annotations

import collections.abc
import typing

import numpy
import numpy.typing

from . import _extension
from . import _node
from . import _partition


def _format_p_value_number(value: float) -> str:
    """Format a p-value as '<1e-300', a percentage, or scientific notation."""
    match value:
        case 0.0:
            formatted = "<1e-300"
        case v if v >= 0.0001:
            formatted = f"{v * 100:.2f}%"
        case _:
            formatted = f"{value:.2e}"
    return formatted


def _format_p_value(partition: _partition.Partition) -> str:
    """Format 'Split p-value = ...' for a split node ('<1e-300' on underflow)."""
    value = partition.p_value
    formatted = _format_p_value_number(value)
    result = f"Split p-value = {formatted}"
    return result


def _format_share(share: float) -> str:
    """Format a share fraction as a percentage."""
    pct = 100.0 * share
    if pct < 0.1:
        return "<0.1%"
    return f"{pct:.1f}%"


def _format_repr(value: object) -> str:
    """Format a value with repr, using double quotes for strings."""
    raw = repr(value)
    if isinstance(value, str):
        inner = value.replace("\\", "\\\\").replace('"', '\\"')
        raw = f'"{inner}"'
    return raw


def _format_value(value: float, precision: int = 3) -> str:
    """Format a float with the given number of digits after the decimal point.

    Non-finite values (inf, -inf, NaN) render as the literal "unknown".
    """
    if not numpy.isfinite(value):
        return "unknown"
    result = f"{value:.{precision}f}"
    return result


def _format_threshold(value: int | float, precision: int = 3) -> str:
    """Format a split threshold, using scientific notation outside [1e-3, 1e6)."""
    if isinstance(value, int):
        precision = 0
    abs_value = abs(value)
    if abs_value == 0 or 0.001 <= abs_value < 1_000_000:
        result = f"{value:.{precision}f}"
    else:
        result = f"{value:.{precision}e}"
    return result


def _format_probability(value: float, precision: int = 3) -> str:
    """Format a probability as a percentage with the given precision.

    Non-finite values (inf, -inf, NaN) render as the literal "unknown".
    """
    if not numpy.isfinite(value):
        return "unknown"
    if value == 0.0:
        return "0%"
    if value == 1.0:
        return "100%"
    pct = 100.0 * value
    if pct >= 0.1:
        result = f"{pct:.{precision}f}%"
    else:
        result = f"{pct:.{precision}e}%"
    return result


def _format_ci_pair(
    formatter: typing.Callable[[float], str],
    ci_low: float,
    ci_high: float,
) -> str:
    """Render a confidence-interval bound pair, including the leading space.

    When both bounds are non-finite, returns the shorthand
    " (unknown bounds)". Otherwise returns " (low to high)" with each bound
    routed through the supplied formatter (which itself produces "unknown"
    for any non-finite bound).
    """
    if not numpy.isfinite(ci_low) and not numpy.isfinite(ci_high):
        return " (unknown bounds)"
    pair = f" ({formatter(ci_low)} to {formatter(ci_high)})"
    return pair


_BOLD_OPEN = "\x01"
_BOLD_CLOSE = "\x02"


def _bold(text: str, bold: bool) -> str:
    """Wrap text with bold sentinel markers when bold is True."""
    if bold:
        result = f"{_BOLD_OPEN}{text}{_BOLD_CLOSE}"
    else:
        result = text
    return result


def _format_prediction(
    node: _node.Node,
    class_names: None | list[str] = None,
    response_name: None | str = None,
    prediction_formatter: None | typing.Callable[[float], str] = None,
    separator: str = ", ",
    precision: int = 3,
    bold_value: bool = False,
    displayed_indices: None | list[int] = None,
) -> str:
    """Format prediction value with optional CI for display.

    Args:
        node: Tree node whose prediction to format.
        class_names: Optional display names for class labels (classification
            only).
        response_name: Optional name for the response variable (regression
            and survival only). For regression the label reads "{response_name}
            mean" instead of "Predicted mean"; for survival it reads "Median
            {response_name}" instead of "Median survival".
        prediction_formatter: Optional callable taking a float and returning a
            string. For regression and survival, applied to the prediction and
            each confidence interval bound. For classification, applied to each
            class probability and its confidence interval bounds.
        separator: String inserted between parts. Defaults to ", " for
            single-line text output; pass "\n" for multi-line graphviz labels.
        precision: Number of digits after the decimal point used by the default
            formatters for predictions, CI bounds, and class probabilities.
            Ignored when prediction_formatter is supplied.
        bold_value: When True, the predicted value (regression mean,
            classification probability, or survival metric value) is wrapped
            with bold sentinel markers, suitable for HTML rendering by the
            graphviz visualizer. Confidence interval bounds are not bolded.
    """
    share = _format_share(node.share)
    count_part = f"Obs. count = {node.n_samples}, Obs. share = {share}"
    if isinstance(node, _node.ClassificationNode):
        text = _format_classification_prediction(
            node,
            class_names,
            prediction_formatter,
            separator,
            precision,
            count_part,
            bold_value,
        )
        return text
    if isinstance(node, _node.SurvivalNode):
        text = _format_survival_prediction(
            node,
            response_name,
            prediction_formatter,
            separator,
            precision,
            count_part,
            bold_value,
        )
        return text
    if isinstance(node, _node.RankingNode):
        ranking_indices = [] if displayed_indices is None else displayed_indices
        text = _format_ranking_prediction(
            node,
            prediction_formatter,
            separator,
            precision,
            count_part,
            bold_value,
            ranking_indices,
        )
        return text
    regression_node = typing.cast(_node.RegressionNode, node)
    text = _format_regression_prediction(
        regression_node,
        response_name,
        prediction_formatter,
        separator,
        precision,
        count_part,
        bold_value,
    )
    return text


def _capitalize_first_letter(text: str) -> str:
    """Return text with its first character upper-cased."""
    result = text[:1].upper() + text[1:]
    return result


def _ellipsize(s: str, max_length: int) -> str:
    """Truncate s to at most max_length characters with a trailing '...' on truncation."""
    if len(s) <= max_length:
        return s
    if max_length >= 3:
        result = s[: max_length - 3] + "..."
        return result
    return "..."[:max_length]


def _format_classification_prediction(
    node: _node.ClassificationNode,
    class_names: None | list[str],
    prediction_formatter: None | typing.Callable[[float], str],
    separator: str,
    precision: int,
    count_part: str,
    bold_value: bool,
) -> str:
    """Format the classification line of a node display."""
    formatter = prediction_formatter or (
        lambda v: _format_probability(v, precision)
    )
    class_distribution = node.class_distribution
    ci_low = node.ci_low
    ci_high = node.ci_high
    parts: list[str] = []
    for i in range(len(class_distribution)):
        raw = str(i) if class_names is None else class_names[i]
        name = _capitalize_first_letter(raw)
        value_text = _bold(formatter(class_distribution[i]), bold_value)
        part = f"{name} proba. = {value_text}"
        if ci_low is not None and ci_high is not None:
            part += _format_ci_pair(
                formatter, float(ci_low[i]), float(ci_high[i])
            )
        parts.append(part)
    parts.append(count_part)
    text = separator.join(parts)
    return text


def _format_regression_prediction(
    node: _node.RegressionNode,
    response_name: None | str,
    prediction_formatter: None | typing.Callable[[float], str],
    separator: str,
    precision: int,
    count_part: str,
    bold_value: bool,
) -> str:
    """Format the regression line of a node display."""
    formatter = prediction_formatter or (lambda v: _format_value(v, precision))
    if response_name is None:
        label = "Predicted"
    else:
        label = _capitalize_first_letter(response_name)
    value_text = _bold(formatter(node.prediction), bold_value)
    prediction_part = f"{label} mean = {value_text}"
    ci_low = node.ci_low
    ci_high = node.ci_high
    if ci_low is not None and ci_high is not None:
        prediction_part += _format_ci_pair(formatter, ci_low, ci_high)
    text = f"{prediction_part}{separator}{count_part}"
    return text


def _format_survival_prediction(
    node: _node.SurvivalNode,
    response_name: None | str,
    prediction_formatter: None | typing.Callable[[float], str],
    separator: str,
    precision: int,
    count_part: str,
    bold_value: bool,
) -> str:
    """Format one labeled line per configured survival metric."""
    metrics = node.metrics
    parts: list[str] = []
    for metric in metrics:
        line = _format_survival_metric_line(
            label=metric.label,
            value=metric.value,
            ci_low=metric.ci_low,
            ci_high=metric.ci_high,
            style=metric.style,
            response_name=response_name,
            prediction_formatter=prediction_formatter,
            precision=precision,
            bold_value=bold_value,
        )
        parts.append(line)
    parts.append(count_part)
    text = separator.join(parts)
    return text


def _format_ranking_prediction(
    node: _node.RankingNode,
    prediction_formatter: None | typing.Callable[[float], str],
    separator: str,
    precision: int,
    count_part: str,
    bold_value: bool,
    displayed_indices: list[int],
) -> str:
    """Format one labeled line per displayed ranking metric."""
    parts: list[str] = []
    for index in displayed_indices:
        metric = node.metrics[index]
        line = _format_survival_metric_line(
            label=f"{_capitalize_first_letter(metric.label)} rank",
            value=metric.value,
            ci_low=metric.ci_low,
            ci_high=metric.ci_high,
            style=metric.style,
            response_name=None,
            prediction_formatter=prediction_formatter,
            precision=precision,
            bold_value=bold_value,
        )
        parts.append(line)
    parts.append(count_part)
    text = separator.join(parts)
    return text


def _format_survival_metric_line(
    label: str,
    value: float,
    ci_low: None | float,
    ci_high: None | float,
    style: typing.Literal["value", "probability"],
    response_name: None | str,
    prediction_formatter: None | typing.Callable[[float], str],
    precision: int,
    bold_value: bool,
) -> str:
    """Format a single metric line with optional CI."""

    def default_formatter(v: float) -> str:
        if style == "probability":
            return _format_probability(v, precision)
        return _format_value(v, precision)

    formatter = prediction_formatter or default_formatter
    display_label = _apply_response_name(label, response_name)
    value_text = _bold(formatter(value), bold_value)
    line = f"{display_label} = {value_text}"
    if ci_low is not None and ci_high is not None:
        line += _format_ci_pair(formatter, ci_low, ci_high)
    return line


def _apply_response_name(label: str, response_name: None | str) -> str:
    """Substitute response_name into the canonical 'Median survival' label."""
    if response_name is None:
        return label
    if label == "Median survival":
        capitalized = _capitalize_first_letter(response_name)
        text = f"Median {capitalized}"
        return text
    return label


class _TextRow(typing.NamedTuple):
    """One displayed line of the textual tree representation."""

    prefix: str
    leaf_index_cell: None | str
    prediction_cells: list[str]
    p_value_cell: None | str
    decoration: None | str


def _table_prediction_headers(
    node: _node.Node,
    class_names: None | list[str],
    response_name: None | str,
    displayed_indices: None | list[int],
) -> list[str]:
    """Header strings for the prediction columns of node."""
    if isinstance(node, _node.ClassificationNode):
        result = _table_classification_prediction_headers(node, class_names)
        return result
    if isinstance(node, _node.SurvivalNode):
        result = _table_survival_prediction_headers(node, response_name)
        return result
    if isinstance(node, _node.RankingNode):
        ranking_indices = [] if displayed_indices is None else displayed_indices
        result = _table_ranking_prediction_headers(node, ranking_indices)
        return result
    result = _table_regression_prediction_headers(response_name)
    return result


def _table_classification_prediction_headers(
    node: _node.ClassificationNode,
    class_names: None | list[str],
) -> list[str]:
    """Headers for a classification node's prediction columns."""
    headers: list[str] = []
    for i in range(len(node.class_distribution)):
        raw = str(i) if class_names is None else class_names[i]
        name = _capitalize_first_letter(raw)
        headers.append(f"{name} proba.")
    headers.append("Obs. count")
    headers.append("Obs. share")
    return headers


def _table_regression_prediction_headers(
    response_name: None | str,
) -> list[str]:
    """Headers for a regression node's prediction columns."""
    if response_name is None:
        label = "Predicted"
    else:
        label = _capitalize_first_letter(response_name)
    headers = [f"{label} mean", "Obs. count", "Obs. share"]
    return headers


def _table_survival_prediction_headers(
    node: _node.SurvivalNode,
    response_name: None | str,
) -> list[str]:
    """Headers for a survival node's prediction columns."""
    headers = [
        _apply_response_name(metric.label, response_name)
        for metric in node.metrics
    ]
    headers.append("Obs. count")
    headers.append("Obs. share")
    return headers


def _table_ranking_prediction_headers(
    node: _node.RankingNode,
    displayed_indices: list[int],
) -> list[str]:
    """Headers for a ranking node's displayed-item prediction columns."""
    headers = [
        f"{_capitalize_first_letter(node.metrics[index].label)} rank"
        for index in displayed_indices
    ]
    headers.append("Obs. count")
    headers.append("Obs. share")
    return headers


def _table_prediction_cells(
    node: _node.Node,
    prediction_formatter: None | typing.Callable[[float], str],
    precision: int,
    displayed_indices: None | list[int],
) -> list[str]:
    """Cell strings for the prediction columns of node."""
    if isinstance(node, _node.ClassificationNode):
        result = _table_classification_prediction_cells(
            node, prediction_formatter, precision
        )
        return result
    if isinstance(node, _node.SurvivalNode):
        result = _table_survival_prediction_cells(
            node, prediction_formatter, precision
        )
        return result
    if isinstance(node, _node.RankingNode):
        ranking_indices = [] if displayed_indices is None else displayed_indices
        result = _table_ranking_prediction_cells(
            node, prediction_formatter, precision, ranking_indices
        )
        return result
    regression_node = typing.cast(_node.RegressionNode, node)
    result = _table_regression_prediction_cells(
        regression_node, prediction_formatter, precision
    )
    return result


def _table_classification_prediction_cells(
    node: _node.ClassificationNode,
    prediction_formatter: None | typing.Callable[[float], str],
    precision: int,
) -> list[str]:
    """Cells for a classification node's prediction columns."""
    formatter = prediction_formatter or (
        lambda v: _format_probability(v, precision)
    )
    class_distribution = node.class_distribution
    ci_low = node.ci_low
    ci_high = node.ci_high
    cells: list[str] = []
    for i in range(len(class_distribution)):
        cell = formatter(class_distribution[i])
        if ci_low is not None and ci_high is not None:
            cell += _format_ci_pair(
                formatter, float(ci_low[i]), float(ci_high[i])
            )
        cells.append(cell)
    cells.append(str(node.n_samples))
    cells.append(_format_share(node.share))
    return cells


def _table_regression_prediction_cells(
    node: _node.RegressionNode,
    prediction_formatter: None | typing.Callable[[float], str],
    precision: int,
) -> list[str]:
    """Cells for a regression node's prediction columns."""
    formatter = prediction_formatter or (lambda v: _format_value(v, precision))
    cell = formatter(node.prediction)
    ci_low = node.ci_low
    ci_high = node.ci_high
    if ci_low is not None and ci_high is not None:
        cell += _format_ci_pair(formatter, ci_low, ci_high)
    cells = [cell, str(node.n_samples), _format_share(node.share)]
    return cells


def _table_survival_prediction_cells(
    node: _node.SurvivalNode,
    prediction_formatter: None | typing.Callable[[float], str],
    precision: int,
) -> list[str]:
    """Cells for a survival node's prediction columns."""
    cells: list[str] = []
    for metric in node.metrics:
        cell = _format_survival_metric_cell(
            metric.value,
            metric.ci_low,
            metric.ci_high,
            metric.style,
            prediction_formatter,
            precision,
        )
        cells.append(cell)
    cells.append(str(node.n_samples))
    cells.append(_format_share(node.share))
    return cells


def _format_survival_metric_cell(
    value: float,
    ci_low: None | float,
    ci_high: None | float,
    style: typing.Literal["value", "probability"],
    prediction_formatter: None | typing.Callable[[float], str],
    precision: int,
) -> str:
    """Render one survival metric cell with optional CI bounds."""

    def default_formatter(v: float) -> str:
        if style == "probability":
            return _format_probability(v, precision)
        return _format_value(v, precision)

    formatter = prediction_formatter or default_formatter
    cell = formatter(value)
    if ci_low is not None and ci_high is not None:
        cell += _format_ci_pair(formatter, ci_low, ci_high)
    return cell


def _table_ranking_prediction_cells(
    node: _node.RankingNode,
    prediction_formatter: None | typing.Callable[[float], str],
    precision: int,
    displayed_indices: list[int],
) -> list[str]:
    """Cells for a ranking node's displayed-item prediction columns."""
    cells: list[str] = []
    for index in displayed_indices:
        metric = node.metrics[index]
        cell = _format_survival_metric_cell(
            metric.value,
            metric.ci_low,
            metric.ci_high,
            metric.style,
            prediction_formatter,
            precision,
        )
        cells.append(cell)
    cells.append(str(node.n_samples))
    cells.append(_format_share(node.share))
    return cells


def _table_p_value_cell(partition: _partition.Partition) -> str:
    """Cell string for Split p-value, e.g. '0.01%' or '<1e-300'."""
    value = partition.p_value
    formatted = _format_p_value_number(value)
    return formatted


def _format_aligned_headers(
    headers: collections.abc.Sequence[str],
    rows: collections.abc.Sequence[collections.abc.Sequence[str]],
) -> list[str]:
    """Render headers, dashed separator, and padded rows as a list of lines."""
    column_count = len(headers)
    widths: list[int] = []
    for column_index in range(column_count):
        max_cell_width = max(
            (len(row[column_index]) for row in rows), default=0
        )
        column_width = max(len(headers[column_index]), max_cell_width)
        widths.append(column_width)
    header_cells = [
        _pad_cell(headers[i], widths[i], i) for i in range(column_count)
    ]
    separator_cells = [
        ("-" * widths[i] if headers[i] else " " * widths[i])
        for i in range(column_count)
    ]
    lines = [" ".join(header_cells), " ".join(separator_cells)]
    for row in rows:
        cells = [_pad_cell(row[i], widths[i], i) for i in range(column_count)]
        lines.append(" ".join(cells))
    stripped = [line.rstrip() for line in lines]
    return stripped


def _pad_cell(text: str, width: int, column_index: int) -> str:
    """Left-align the first column and right-align all subsequent columns."""
    if column_index == 0:
        return text.ljust(width)
    return text.rjust(width)


def _collect_text_rows(
    root: _node.Node,
    category_labels: None | dict[int, dict[float, str]],
    feature_names: None | numpy.typing.NDArray,
    prediction_formatter: None | typing.Callable[[float], str],
    max_depth: None | int,
    precision: int,
    best_first: bool,
    displayed_indices: None | list[int],
) -> list[_TextRow]:
    """Walk the tree and collect one _TextRow per displayed line."""
    rows: list[_TextRow] = []
    _append_text_row(
        root,
        "All records",
        prediction_formatter,
        precision,
        rows,
        displayed_indices,
    )
    prediction_column_count = len(rows[0].prediction_cells)
    _append_child_text_rows(
        root,
        category_labels,
        feature_names,
        prediction_formatter,
        max_depth,
        precision,
        best_first,
        "",
        rows,
        displayed_indices,
        prediction_column_count,
    )
    return rows


def _append_text_row(
    node: _node.Node,
    prefix: str,
    prediction_formatter: None | typing.Callable[[float], str],
    precision: int,
    rows: list[_TextRow],
    displayed_indices: None | list[int],
) -> None:
    """Append one _TextRow representing node to rows."""
    cells = _table_prediction_cells(
        node, prediction_formatter, precision, displayed_indices
    )
    match node.extension:
        case _partition.Partition() as partition:
            p_value_cell: None | str = _table_p_value_cell(partition)
            leaf_index_cell: None | str = None
        case _extension.Leaf() as leaf_extension:
            p_value_cell = None
            leaf_index_cell = str(leaf_extension.leaf_id + 1)
        case _:
            p_value_cell = None
            leaf_index_cell = None
    if node.decoration is None:
        decoration: None | str = None
    else:
        decoration = str(node.decoration)
    rows.append(
        _TextRow(prefix, leaf_index_cell, cells, p_value_cell, decoration)
    )


def _append_child_text_rows(
    node: _node.Node,
    category_labels: None | dict[int, dict[float, str]],
    feature_names: None | numpy.typing.NDArray,
    prediction_formatter: None | typing.Callable[[float], str],
    max_depth: None | int,
    precision: int,
    best_first: bool,
    indent: str,
    rows: list[_TextRow],
    displayed_indices: None | list[int],
    prediction_column_count: int,
) -> None:
    """Recursively append text rows for each child of node."""
    match node.extension:
        case _partition.Partition() as partition:
            pass
        case _:
            return
    if max_depth is not None and node.depth >= max_depth:
        empty_prediction_cells = [""] * prediction_column_count
        rows.append(
            _TextRow(
                f"{indent}└── ...",
                None,
                empty_prediction_cells,
                None,
                None,
            )
        )
        return
    left_label, right_label = _format_branch_labels(
        partition, category_labels, feature_names, precision=precision
    )
    ordered_children = node.display_children(best_first)
    left_child, right_child = ordered_children or (
        partition.left,
        partition.right,
    )
    if left_child is not partition.left:
        left_label, right_label = right_label, left_label
    for branch_label, child, is_last in [
        (left_label, left_child, False),
        (right_label, right_child, True),
    ]:
        connector = "└──" if is_last else "├──"
        prefix = f"{indent}{connector} {branch_label}"
        _append_text_row(
            child,
            prefix,
            prediction_formatter,
            precision,
            rows,
            displayed_indices,
        )
        indent_extension = "    " if is_last else "│   "
        _append_child_text_rows(
            child,
            category_labels,
            feature_names,
            prediction_formatter,
            max_depth,
            precision,
            best_first,
            indent + indent_extension,
            rows,
            displayed_indices,
            prediction_column_count,
        )


def _format_categorical_condition(name: str, items: list[str]) -> str:
    """Format a categorical split condition."""
    match len(items):
        case 1:
            listing = items[0]
        case 2:
            listing = f"{items[0]} or {items[1]}"
        case _:
            listing = ", ".join(items[:-1]) + f", or {items[-1]}"
    return f"{name} is {listing}"


def _format_branch_labels(
    partition: _partition.Partition,
    category_labels: None | dict[int, dict[float, str]],
    feature_names: None | numpy.typing.NDArray = None,
    precision: int = 3,
) -> tuple[str, str]:
    """Return display labels for the left and right branches of a partition."""
    feature_index = partition.feature_index
    if feature_names is None:
        if partition.feature_name is None:
            name = f"X[{feature_index}]"
        else:
            name = partition.feature_name
    else:
        name = str(feature_names[feature_index])
    if isinstance(partition, _partition.BooleanPartition):
        left_label = f"{name} is false"
        right_label = f"{name} is true"
        return (left_label, right_label)
    if isinstance(partition, _partition.NumericalPartition):
        threshold = _format_threshold(partition.threshold, precision)
        left_label = f"{name} <= {threshold}"
        right_label = f"{name} > {threshold}"
        return (left_label, right_label)
    categorical = typing.cast(_partition.CategoricalPartition, partition)
    labels = (
        category_labels.get(feature_index)
        if category_labels is not None
        else None
    )
    included_cats = sorted(categorical.left_categories)
    excluded_cats = sorted(categorical.right_categories)
    if labels is None:
        included_items = [_format_repr(c) for c in included_cats]
        excluded_items = [_format_repr(c) for c in excluded_cats]
    else:
        included_items = [
            _format_repr(labels.get(c, str(c))) for c in included_cats
        ]
        excluded_items = [
            _format_repr(labels.get(c, str(c))) for c in excluded_cats
        ]
    left_label = _format_categorical_condition(name, included_items)
    right_label = _format_categorical_condition(name, excluded_items)
    return (left_label, right_label)
