"""Module-level free functions to render a fitted Tree.

Four free functions take a fitted Tree estimator as their first
positional argument:

- export_text: produce a textual report of the tree as a string or
  write it to a file.
- export_graphviz: produce the DOT source describing the tree as a
  string or write it to a file.
- export_image: render the tree as GIF, PDF, PNG, or SVG bytes or
  write the bytes to a file.
- export_sql: produce a SQL CASE expression that reproduces the
  tree's predictions, as a string or written to a file.
"""

from __future__ import annotations

import typing

import sklearn.utils.validation

from . import _tree
from . import _tree_text


@typing.overload
def export_text(
    tree: _tree.Tree,
    out_file: None = None,
    feature_names: None | list[str] = None,
    class_names: None | list[str] = None,
    response_name: None | str = None,
    category_labels: None | _tree._CategoryLabels = None,
    prediction_formatter: None | typing.Callable[[float], str] = None,
    max_depth: None | int = None,
    precision: int = 3,
) -> str: ...


@typing.overload
def export_text(
    tree: _tree.Tree,
    out_file: str | typing.IO[str],
    feature_names: None | list[str] = None,
    class_names: None | list[str] = None,
    response_name: None | str = None,
    category_labels: None | _tree._CategoryLabels = None,
    prediction_formatter: None | typing.Callable[[float], str] = None,
    max_depth: None | int = None,
    precision: int = 3,
) -> None: ...


def export_text(
    tree: _tree.Tree,
    out_file: None | str | typing.IO[str] = None,
    feature_names: None | list[str] = None,
    class_names: None | list[str] = None,
    response_name: None | str = None,
    category_labels: None | _tree._CategoryLabels = None,
    prediction_formatter: None | typing.Callable[[float], str] = None,
    max_depth: None | int = None,
    precision: int = 3,
) -> None | str:
    """Build a text report of the fitted tree.

    Returns the same content that Tree.to_text returns. When out_file is
    None, the text is returned as a string; otherwise it is written to
    the destination and the function returns None. Children are ordered
    with the best leaf first by default; pass reverse_order=True to the
    Tree constructor to invert this.

    Args:
        tree: A fitted Tree estimator (RegressionTree or ClassificationTree).
        out_file: Where to write the text. When None (the default), the
            text is returned as a string. When a string, it is treated as
            a filesystem path; the file is opened with UTF-8 encoding,
            written, closed, and the function returns None. When a
            file-like object (anything with a write method), the text is
            written to it and the function returns None.
        feature_names: Optional display names for the covariates, one per
            column of X. When None, falls back to feature_names_in_ (set at
            fit time when X is a pandas DataFrame). When neither source is
            available, splits show "X[i]" placeholders.
        class_names: Optional display names for class labels (classification
            only). When None, falls back to [str(c) for c in tree.classes_]
            (the labels seen at fit time, e.g., the Categorical y order or
            the unique string values of y).
        response_name: Optional name for the response variable. When None,
            falls back to response_name_in_ (set at fit time when y is a
            named pandas Series, or for survival when y is a DataFrame).
            When neither source is available, the regression header is
            "Predicted mean" and the survival header keeps the canonical
            "Median survival" wording.
        category_labels: Optional mapping from a categorical feature
            (column-name string or integer index) to a dict of {code: label}.
            String keys are resolved against feature_names (or
            feature_names_in_). Merged over category_labels_in_ (set at
            fit time when X has pandas categorical / object columns), with
            caller-provided keys winning.
        prediction_formatter: Optional callable taking a float and returning a
            string. For regression, applied to the prediction and each
            confidence interval bound. For classification, applied to each
            class probability.
        max_depth: Maximum depth to render, with the root counted as depth 0.
            When None (the default), the full tree is rendered. When a
            non-negative integer, subtrees rooted below this depth are
            replaced with a single "..." marker line.
        precision: Number of digits after the decimal point used by the
            default formatters for predictions, CI bounds, split thresholds,
            and class probabilities. Does not affect p-values or observation
            shares. Defaults to 3.

    Returns:
        The textual tree representation as a string when out_file is
        None; otherwise None.

    Raises:
        sklearn.exceptions.NotFittedError: If tree has not been fitted.
        ValueError: If max_depth is a negative integer or not an integer.
        ValueError: If precision is not a non-negative integer.
        TypeError: If out_file is neither None, a string, nor a file-like
            object with a write method.
    """
    if max_depth is not None and (
        not isinstance(max_depth, int) or max_depth < 0
    ):
        raise ValueError("max_depth must be None or a non-negative integer")
    if not isinstance(precision, int) or precision < 0:
        raise ValueError("precision must be a non-negative integer")
    if (
        out_file is not None
        and not isinstance(out_file, str)
        and not hasattr(out_file, "write")
    ):
        raise TypeError(
            "out_file must be None, a string path, or a file-like object"
            " with a write method"
        )
    sklearn.utils.validation.check_is_fitted(tree, "content_")
    names = tree._effective_feature_names(feature_names)
    resolved_category_labels = tree._resolve_category_labels(
        category_labels, names
    )
    effective_response_name = tree._effective_response_name(response_name)
    effective_class_names = tree._effective_class_names(class_names)
    root = tree.content_
    text_rows = _tree_text._collect_text_rows(
        root,
        resolved_category_labels,
        names,
        prediction_formatter,
        max_depth,
        precision,
        not tree.reverse_order,
    )
    prediction_headers = _tree_text._table_prediction_headers(
        root, effective_class_names, effective_response_name
    )
    # TODO XXX optimize/review these 2
    has_split = any(row.p_value_cell is not None for row in text_rows)
    has_decoration = any(row.decoration is not None for row in text_rows)
    headers = ["", *prediction_headers]
    if has_split:
        headers.append("Split p-value")
    if has_decoration:
        headers.append("")
    rendered_rows: list[list[str]] = []
    for row in text_rows:
        cells = [row.prefix]
        prediction_cells = row.prediction_cells
        if prediction_cells:
            cells.extend(prediction_cells)
        else:
            # TODO XXX make it useless for the table utility function
            cells.extend([""] * len(prediction_headers))
        if has_split:
            cells.append(row.p_value_cell or "")
        if has_decoration:
            cells.append(row.decoration or "")
        rendered_rows.append(cells)
    lines = _tree_text._format_aligned_headers(headers, rendered_rows)
    result = "\n".join(lines)
    match out_file:
        case None:
            return result
        case str():
            with open(out_file, "w", encoding="utf-8") as file_handle:
                file_handle.write(result)
            return None
        case _:
            out_file.write(result)
            return None


@typing.overload
def export_sql(
    tree: _tree.Tree,
    out_file: None = None,
    target_class: None | object = None,
    feature_names: None | list[str] = None,
    category_labels: None | _tree._CategoryLabels = None,
    max_depth: None | int = None,
) -> str: ...


@typing.overload
def export_sql(
    tree: _tree.Tree,
    out_file: str | typing.IO[str],
    target_class: None | object = None,
    feature_names: None | list[str] = None,
    category_labels: None | _tree._CategoryLabels = None,
    max_depth: None | int = None,
) -> None: ...


def export_sql(
    tree: _tree.Tree,
    out_file: None | str | typing.IO[str] = None,
    target_class: None | object = None,
    feature_names: None | list[str] = None,
    category_labels: None | _tree._CategoryLabels = None,
    max_depth: None | int = None,
) -> None | str:
    """Build a SQL CASE expression that reproduces the tree's predictions.

    When out_file is None, the SQL is returned as a string; otherwise it
    is written to the destination and the function returns None. The
    emitted expression is a single, recursively nested CASE that routes
    each row to its leaf and returns the leaf's prediction:

    - RegressionTree: each leaf returns its predicted mean.
    - ClassificationTree: each leaf returns the probability of the
      target_class.
    - SurvivalTree: each leaf returns its first metric value (typically
      the median survival).

    A NULL input feature (or a categorical value not observed during
    training) matches no WHEN clause and falls through to ELSE NULL, so
    the prediction is NULL rather than a silent misprediction. Branch
    ordering follows the tree's reverse_order attribute exactly like
    to_text and to_image.

    Args:
        tree: A fitted Tree estimator (RegressionTree, ClassificationTree,
            or SurvivalTree).
        out_file: Where to write the SQL. When None (the default), the SQL
            is returned as a string. When a string, it is treated as a
            filesystem path; the file is opened with UTF-8 encoding,
            written, closed, and the function returns None. When a
            file-like object (anything with a write method), the SQL is
            written to it and the function returns None.
        target_class: Classification-only. Selects which class's
            probability is emitted at each leaf. Must equal one of the
            values in tree.classes_. When None (the default), the last
            class (tree.classes_[-1]) is used. Passing a non-None value
            for a non-ClassificationTree raises ValueError.
        feature_names: Optional display names for the covariates, one per
            column of X, used as the double-quoted SQL column identifiers
            in the emitted conditions. When None, falls back to
            feature_names_in_ (set at fit time when X is a pandas
            DataFrame). When neither source is available, identifiers
            read "X[i]".
        category_labels: Optional mapping from a categorical feature
            (column-name string or integer index) to a dict of
            {code: label}. String keys are resolved against feature_names
            (or feature_names_in_). Merged over category_labels_in_ (set
            at fit time when X has pandas categorical / object columns),
            with caller-provided keys winning. When labels are known, the
            IN (...) lists hold label strings; otherwise they hold the
            numeric codes Sigma stored at fit time.
        max_depth: Maximum depth to render, with the root counted as depth
            0. When None (the default), the full tree is rendered. When a
            non-negative integer, subtrees rooted below this depth are
            collapsed to a single leaf line carrying the truncated node's
            own prediction and a "-- Truncated at depth N" comment.

    Returns:
        The SQL CASE expression as a string when out_file is None;
        otherwise None.

    Raises:
        sklearn.exceptions.NotFittedError: If tree has not been fitted.
        ValueError: If max_depth is a negative integer or not an integer.
        ValueError: If target_class is provided for a non-classification
            tree, or is not a value in tree.classes_.
        TypeError: If out_file is neither None, a string, nor a file-like
            object with a write method.
    """
    if max_depth is not None and (
        not isinstance(max_depth, int) or max_depth < 0
    ):
        raise ValueError("max_depth must be None or a non-negative integer")
    if (
        out_file is not None
        and not isinstance(out_file, str)
        and not hasattr(out_file, "write")
    ):
        raise TypeError(
            "out_file must be None, a string path, or a file-like object"
            " with a write method"
        )
    sklearn.utils.validation.check_is_fitted(tree, "content_")
    from . import _tree_sql

    names = tree._effective_feature_names(feature_names)
    resolved_category_labels = tree._resolve_category_labels(
        category_labels, names
    )
    target_class_index = _tree_sql._resolve_target_class_index(
        tree, target_class
    )
    result = _tree_sql._collect_sql(
        tree.content_,
        names,
        resolved_category_labels,
        target_class_index,
        max_depth,
        not tree.reverse_order,
    )
    match out_file:
        case None:
            return result
        case str():
            with open(out_file, "w", encoding="utf-8") as file_handle:
                file_handle.write(result)
            return None
        case _:
            out_file.write(result)
            return None


@typing.overload
def export_graphviz(
    tree: _tree.Tree,
    out_file: None = None,
    feature_names: None | list[str] = None,
    class_names: None | list[str] = None,
    response_name: None | str = None,
    category_labels: None | _tree._CategoryLabels = None,
    prediction_formatter: None | typing.Callable[[float], str] = None,
    max_depth: None | int = None,
    precision: int = 3,
    orientation: typing.Literal["top-down", "left-to-right"] = "top-down",
    dpi: int = 192,
    root_colors: None | tuple[str, str, str] = None,
    split_colors: None | tuple[str, str, str] = None,
    leaf_colors: None | tuple[str, str, str] = None,
    background_color: None | str = None,
) -> str: ...


@typing.overload
def export_graphviz(
    tree: _tree.Tree,
    out_file: str | typing.IO[str],
    feature_names: None | list[str] = None,
    class_names: None | list[str] = None,
    response_name: None | str = None,
    category_labels: None | _tree._CategoryLabels = None,
    prediction_formatter: None | typing.Callable[[float], str] = None,
    max_depth: None | int = None,
    precision: int = 3,
    orientation: typing.Literal["top-down", "left-to-right"] = "top-down",
    dpi: int = 192,
    root_colors: None | tuple[str, str, str] = None,
    split_colors: None | tuple[str, str, str] = None,
    leaf_colors: None | tuple[str, str, str] = None,
    background_color: None | str = None,
) -> None: ...


def export_graphviz(
    tree: _tree.Tree,
    out_file: None | str | typing.IO[str] = None,
    feature_names: None | list[str] = None,
    class_names: None | list[str] = None,
    response_name: None | str = None,
    category_labels: None | _tree._CategoryLabels = None,
    prediction_formatter: None | typing.Callable[[float], str] = None,
    max_depth: None | int = None,
    precision: int = 3,
    orientation: typing.Literal["top-down", "left-to-right"] = "top-down",
    dpi: int = 192,
    root_colors: None | tuple[str, str, str] = None,
    split_colors: None | tuple[str, str, str] = None,
    leaf_colors: None | tuple[str, str, str] = None,
    background_color: None | str = None,
) -> None | str:
    """Export the fitted tree as a Graphviz DOT source.

    When out_file is None, the DOT source is returned as a string;
    otherwise it is written to the destination and the function returns
    None.

    Args:
        tree: A fitted Tree estimator (RegressionTree or ClassificationTree).
        out_file: Where to write the DOT source. When None (the default), the
            DOT source is returned as a string. When a string, it is treated
            as a filesystem path; the file is opened with UTF-8 encoding,
            written, closed, and the function returns None. When a file-like
            object (anything with a write method), the DOT source is written
            to it and the function returns None.
        feature_names: Optional display names for the covariates, one per
            column of X. When None, falls back to feature_names_in_ (set at
            fit time when X is a pandas DataFrame). When neither source is
            available, splits show "X[i]" placeholders.
        class_names: Optional display names for class labels (classification
            only). When None, falls back to [str(c) for c in tree.classes_]
            (the labels seen at fit time, e.g., the Categorical y order or
            the unique string values of y).
        response_name: Optional name for the response variable. When None,
            falls back to response_name_in_ (set at fit time when y is a
            named pandas Series, or for survival when y is a DataFrame).
            When neither source is available, regression node labels read
            "Predicted mean = ..." and survival nodes keep the canonical
            "Median survival" wording.
        category_labels: Optional mapping from a categorical feature
            (column-name string or integer index) to a dict of {code: label}.
            String keys are resolved against feature_names (or
            feature_names_in_). Merged over category_labels_in_ (set at fit
            time when X has pandas categorical / object columns), with
            caller-provided keys winning.
        prediction_formatter: Optional callable taking a float and returning a
            string. For regression, applied to the prediction and each
            confidence interval bound. For classification, applied to each
            class probability and its confidence interval bounds.
        max_depth: Maximum depth to render, with the root counted as depth 0.
            When None (the default), the full tree is rendered. When a
            non-negative integer, subtrees rooted below this depth are
            replaced with a single "..." placeholder node.
        precision: Number of digits after the decimal point used by the
            default formatters for predictions, CI bounds, split thresholds,
            and class probabilities. Does not affect p-values or observation
            shares. Defaults to 3.
        orientation: Layout direction for the rendered tree, one of
            "top-down" (the default, with the root at the top and leaves
            below) or "left-to-right" (with the root at the left and leaves
            growing rightward). In "left-to-right" mode children render
            top-to-bottom in descending prediction order.
        dpi: Output resolution in dots per inch. Proportionally scales the
            width and height of any SVG produced from the returned DOT, and
            the pixel dimensions of any raster output. Defaults to 192.
        root_colors: (font, fill, border) colors for the root node. Defaults
            to ("white", "black", "black").
        split_colors: (font, fill, border) colors for internal (split) nodes.
            Defaults to ("black", "lightgray", "black").
        leaf_colors: (font, fill, border) colors for leaf nodes. Defaults to
            ("white", "#0F62FE", "#0F62FE").
        background_color: Optional graphviz color string for the canvas
            background (e.g., "white", "#ffeecc"). Defaults to "white" when
            None. Pass "transparent" to request a transparent background.

    Returns:
        The DOT source as a string when out_file is None; otherwise None.

    Raises:
        sklearn.exceptions.NotFittedError: If tree has not been fitted.
        ValueError: If max_depth is a negative integer or not an integer.
        ValueError: If precision is not a non-negative integer.
        ValueError: If dpi is not a positive integer.
        TypeError: If out_file is neither None, a string, nor a file-like
            object with a write method.
        ImportError: If graphviz is not installed.
    """
    if max_depth is not None and (
        not isinstance(max_depth, int) or max_depth < 0
    ):
        raise ValueError("max_depth must be None or a non-negative integer")
    if not isinstance(precision, int) or precision < 0:
        raise ValueError("precision must be a non-negative integer")
    if not isinstance(dpi, int) or dpi <= 0:
        raise ValueError("dpi must be a positive integer")
    if (
        out_file is not None
        and not isinstance(out_file, str)
        and not hasattr(out_file, "write")
    ):
        raise TypeError(
            "out_file must be None, a string path, or a file-like object"
            " with a write method"
        )
    sklearn.utils.validation.check_is_fitted(tree, "content_")
    from . import _graphviz

    names = tree._effective_feature_names(feature_names)
    resolved_category_labels = tree._resolve_category_labels(
        category_labels, names
    )
    effective_response_name = tree._effective_response_name(response_name)
    effective_class_names = tree._effective_class_names(class_names)
    effective_root_colors = root_colors or _graphviz._DEFAULT_ROOT_COLORS
    effective_split_colors = split_colors or _graphviz._DEFAULT_SPLIT_COLORS
    effective_leaf_colors = leaf_colors or _graphviz._DEFAULT_LEAF_COLORS
    dot = _graphviz._build_digraph(
        tree.content_,
        resolved_category_labels,
        effective_class_names,
        effective_root_colors,
        effective_split_colors,
        effective_leaf_colors,
        feature_names=names,
        response_name=effective_response_name,
        prediction_formatter=prediction_formatter,
        background_color=background_color,
        max_depth=max_depth,
        precision=precision,
        dpi=dpi,
        orientation=orientation,
        reverse_order=tree.reverse_order,
    )
    dot_source = dot.source.rstrip("\n")
    match out_file:
        case None:
            return dot_source
        case str():
            with open(out_file, "w", encoding="utf-8") as file_handle:
                file_handle.write(dot_source)
            return None
        case _:
            out_file.write(dot_source)
            return None


@typing.overload
def export_image(
    tree: _tree.Tree,
    format: typing.Literal["gif", "pdf", "png", "svg"],
    out_file: None = None,
    kind: typing.Literal["tree", "response"] = "tree",
    feature_names: None | list[str] = None,
    class_names: None | list[str] = None,
    response_name: None | str = None,
    category_labels: None | _tree._CategoryLabels = None,
    prediction_formatter: None | typing.Callable[[float], str] = None,
    max_depth: None | int = None,
    precision: int = 3,
    orientation: typing.Literal["top-down", "left-to-right"] = "top-down",
    dpi: int = 192,
    root_colors: None | tuple[str, str, str] = None,
    split_colors: None | tuple[str, str, str] = None,
    leaf_colors: None | tuple[str, str, str] = None,
    background_color: None | str = None,
) -> bytes: ...


@typing.overload
def export_image(
    tree: _tree.Tree,
    format: typing.Literal["gif", "pdf", "png", "svg"],
    out_file: str | typing.IO[bytes],
    kind: typing.Literal["tree", "response"] = "tree",
    feature_names: None | list[str] = None,
    class_names: None | list[str] = None,
    response_name: None | str = None,
    category_labels: None | _tree._CategoryLabels = None,
    prediction_formatter: None | typing.Callable[[float], str] = None,
    max_depth: None | int = None,
    precision: int = 3,
    orientation: typing.Literal["top-down", "left-to-right"] = "top-down",
    dpi: int = 192,
    root_colors: None | tuple[str, str, str] = None,
    split_colors: None | tuple[str, str, str] = None,
    leaf_colors: None | tuple[str, str, str] = None,
    background_color: None | str = None,
) -> None: ...


def export_image(
    tree: _tree.Tree,
    format: typing.Literal["gif", "pdf", "png", "svg"],
    out_file: None | str | typing.IO[bytes] = None,
    kind: typing.Literal["tree", "response"] = "tree",
    feature_names: None | list[str] = None,
    class_names: None | list[str] = None,
    response_name: None | str = None,
    category_labels: None | _tree._CategoryLabels = None,
    prediction_formatter: None | typing.Callable[[float], str] = None,
    max_depth: None | int = None,
    precision: int = 3,
    orientation: typing.Literal["top-down", "left-to-right"] = "top-down",
    dpi: int = 192,
    root_colors: None | tuple[str, str, str] = None,
    split_colors: None | tuple[str, str, str] = None,
    leaf_colors: None | tuple[str, str, str] = None,
    background_color: None | str = None,
) -> None | bytes:
    """Render the fitted tree as a GIF, PDF, PNG, or SVG image.

    When out_file is None, the rendered bytes are returned; otherwise
    they are written to the destination and the function returns None.
    Two visualizations are supported via the kind parameter: the default
    "tree" view emits the full decision-tree diagram and the "response"
    view emits a per-leaf summary of the tree's predictions.

    Args:
        tree: A fitted Tree estimator (RegressionTree, ClassificationTree,
            or SurvivalTree).
        format: Output format, one of "gif", "pdf", "png", or "svg". For the
            DOT source see export_graphviz.
        out_file: Where to write the rendered bytes. When None (the default),
            the bytes are returned. When a string, it is treated as a
            filesystem path; the file is opened in binary write mode,
            written, closed, and the function returns None. When a file-like
            object (anything with a write method), the bytes are written to
            it and the function returns None.
        kind: Which visualization to render. "tree" (the default) renders
            the decision tree diagram. "response" renders a per-leaf
            response summary chosen by task: per-leaf box of [ci_low,
            ci_high] with a tick at the predicted mean for regression;
            per-leaf vertical stacked bar of class proportions (top
            segment = first entry of class_names) for classification; one
            Kaplan-Meier step curve per leaf, with the response_name as
            the time axis, for survival. When kind is "response", the
            tree-rendering parameters feature_names, category_labels,
            prediction_formatter, root_colors, split_colors, leaf_colors,
            orientation, max_depth, and precision are ignored. The
            tree's reverse_order attribute is honored using the same
            convention as the tree plot.
        feature_names: Optional display names for the covariates, one per
            column of X. When None, falls back to feature_names_in_ (set at
            fit time when X is a pandas DataFrame). When neither source is
            available, splits show "X[i]" placeholders.
        class_names: Optional display names for class labels (classification
            only). When None, falls back to [str(c) for c in tree.classes_]
            (the labels seen at fit time, e.g., the Categorical y order or
            the unique string values of y).
        response_name: Optional name for the response variable. When None,
            falls back to response_name_in_ (set at fit time when y is a
            named pandas Series, or for survival when y is a DataFrame).
            When neither source is available, regression node labels read
            "Predicted mean = ..." and survival nodes keep the canonical
            "Median survival" wording.
        category_labels: Optional mapping from a categorical feature
            (column-name string or integer index) to a dict of {code: label}.
            String keys are resolved against feature_names (or
            feature_names_in_). Merged over category_labels_in_ (set at fit
            time when X has pandas categorical / object columns), with
            caller-provided keys winning.
        prediction_formatter: Optional callable taking a float and returning
            a string. For regression, applied to the prediction and each
            confidence interval bound. For classification, applied to each
            class probability and its confidence interval bounds.
        max_depth: Maximum depth to render, with the root counted as depth 0.
            When None (the default), the full tree is rendered without depth
            markers. When a non-negative integer, subtrees rooted below this
            depth are replaced with a single "..." placeholder node.
        precision: Number of digits after the decimal point used by the
            default formatters for predictions, CI bounds, split thresholds,
            and class probabilities. Does not affect p-values or observation
            shares. Defaults to 3.
        orientation: Layout direction for the rendered tree, one of
            "top-down" (the default, with the root at the top and leaves
            below) or "left-to-right" (with the root at the left and leaves
            growing rightward). Maps to graphviz's rankdir attribute ("TB"
            or "LR" respectively). In "left-to-right" mode children render
            top-to-bottom in descending prediction order.
        dpi: Output resolution in dots per inch. Proportionally scales the
            width and height of the SVG output and the pixel dimensions of
            the GIF, PNG, and PDF outputs, regardless of kind. Defaults to
            192.
        root_colors: (font, fill, border) colors for the root node. Defaults
            to ("white", "black", "black").
        split_colors: (font, fill, border) colors for internal (split) nodes.
            Defaults to ("black", "lightgray", "black").
        leaf_colors: (font, fill, border) colors for leaf nodes. Defaults to
            ("white", "#0F62FE", "#0F62FE").
        background_color: Optional graphviz color string for the canvas
            background (e.g., "white", "#ffeecc"). Defaults to "white" when
            None. Pass "transparent" to request a transparent background,
            provided the output format supports transparency (PDF, PNG, and
            SVG do; GIF supports 1-bit transparency).

    Returns:
        The rendered image bytes when out_file is None; otherwise None.

    Raises:
        sklearn.exceptions.NotFittedError: If tree has not been fitted.
        ValueError: If format is not one of the supported formats.
        ValueError: If kind is not one of the supported kinds.
        ValueError: If max_depth is a negative integer or not an integer.
        ValueError: If precision is not a non-negative integer.
        ValueError: If dpi is not a positive integer.
        TypeError: If out_file is neither None, a string, nor a file-like
            object with a write method.
        ImportError: If graphviz is not installed when kind is "tree", or
            if cairosvg is not installed when requesting PDF or PNG
            output with kind="tree", or if matplotlib (which transitively
            installs Pillow used to produce GIF output) is not installed
            when kind is "response".
    """
    if format not in ("gif", "pdf", "png", "svg"):
        raise ValueError("format must be one of 'gif', 'pdf', 'png', or 'svg'")
    if kind not in ("tree", "response"):
        raise ValueError("kind must be one of 'tree' or 'response'")
    if max_depth is not None and (
        not isinstance(max_depth, int) or max_depth < 0
    ):
        raise ValueError("max_depth must be None or a non-negative integer")
    if not isinstance(precision, int) or precision < 0:
        raise ValueError("precision must be a non-negative integer")
    if not isinstance(dpi, int) or dpi <= 0:
        raise ValueError("dpi must be a positive integer")
    if (
        out_file is not None
        and not isinstance(out_file, str)
        and not hasattr(out_file, "write")
    ):
        raise TypeError(
            "out_file must be None, a string path, or a file-like object"
            " with a write method"
        )
    sklearn.utils.validation.check_is_fitted(tree, "content_")
    effective_response_name = tree._effective_response_name(response_name)
    effective_class_names = tree._effective_class_names(class_names)
    if kind == "response":
        from . import _response_plot

        payload = _response_plot._render_response_image(
            tree,
            format,
            response_name=effective_response_name,
            class_names=effective_class_names,
            dpi=dpi,
            background_color=background_color,
        )
        match out_file:
            case None:
                return payload
            case str():
                with open(out_file, "wb") as file_handle:
                    file_handle.write(payload)
                return None
            case _:
                out_file.write(payload)
                return None
    from . import _graphviz

    names = tree._effective_feature_names(feature_names)
    resolved_category_labels = tree._resolve_category_labels(
        category_labels, names
    )
    effective_root_colors = root_colors or _graphviz._DEFAULT_ROOT_COLORS
    effective_split_colors = split_colors or _graphviz._DEFAULT_SPLIT_COLORS
    effective_leaf_colors = leaf_colors or _graphviz._DEFAULT_LEAF_COLORS
    dot = _graphviz._build_digraph(
        tree.content_,
        resolved_category_labels,
        effective_class_names,
        effective_root_colors,
        effective_split_colors,
        effective_leaf_colors,
        feature_names=names,
        response_name=effective_response_name,
        prediction_formatter=prediction_formatter,
        background_color=background_color,
        max_depth=max_depth,
        precision=precision,
        dpi=dpi,
        orientation=orientation,
        reverse_order=tree.reverse_order,
    )
    match format:
        case "gif":
            payload = dot.pipe(format="gif")
        case "svg":
            payload = dot.pipe(format="svg")
        case "pdf" | "png":
            svg_bytes = dot.pipe(format="svg")
            try:
                import cairosvg
            except ImportError as import_error:
                raise ImportError(
                    "cairosvg is required for PDF and PNG output. "
                    "Install it with: pip install cairosvg"
                ) from import_error
            match format:
                case "pdf":
                    payload = cairosvg.svg2pdf(bytestring=svg_bytes, dpi=96)
                case "png":
                    payload = cairosvg.svg2png(bytestring=svg_bytes, dpi=96)
    match out_file:
        case None:
            return payload
        case str():
            with open(out_file, "wb") as file_handle:
                file_handle.write(payload)
            return None
        case _:
            out_file.write(payload)
            return None
