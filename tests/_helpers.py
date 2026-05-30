"""Shared fixtures and helpers used across the split test_tree_*.py files."""

import typing

import numpy
import pandas

import sigma._node
import sigma._partition
import sigma._tree_regression


_NodeT = typing.TypeVar("_NodeT", bound=sigma._node.Node)


_RegressionTreeCiMethodLiteral = typing.Literal[
    "bayesian_bootstrap",
    "bca",
    "beta",
    "exponential",
    "gamma",
    "log_normal",
    "log_normal_gci",
    "normal",
    "poisson",
    "student_t",
]
_REGRESSION_TREE_CI_METHODS: list[_RegressionTreeCiMethodLiteral] = [
    "bayesian_bootstrap",
    "bca",
    "beta",
    "exponential",
    "gamma",
    "log_normal",
    "log_normal_gci",
    "normal",
    "poisson",
    "student_t",
]
_REGRESSION_TREE_CI_METHODS_BRACKET_MEAN: list[
    _RegressionTreeCiMethodLiteral
] = [
    "bayesian_bootstrap",
    "bca",
    "beta",
    "exponential",
    "gamma",
    "normal",
    "poisson",
    "student_t",
]
_REGRESSION_TREE_CI_METHODS_COLLAPSE_ON_CONSTANT: list[
    _RegressionTreeCiMethodLiteral
] = [
    "bayesian_bootstrap",
    "bca",
    "gamma",
    "log_normal",
    "log_normal_gci",
    "normal",
    "student_t",
]
_CLASSIFICATION_TREE_CI_METHODS: list[
    typing.Literal[
        "agresti_coull",
        "clopper_pearson",
        "jeffreys",
        "mid_p_exact",
        "wilson",
        "wilson_cc",
    ]
] = [
    "agresti_coull",
    "clopper_pearson",
    "jeffreys",
    "mid_p_exact",
    "wilson",
    "wilson_cc",
]


def _fit_step_regression_tree(**kwargs):
    """Fit a simple step-function regression tree for name-related tests."""
    X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
    y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
    regression_tree = sigma._tree_regression.RegressionTree(
        correlation="normal",
        min_splits=2,
        min_buckets=1,
        **kwargs,
    )
    regression_tree.fit(X, y)
    return regression_tree


def _fit_three_step_regression_tree(**kwargs):
    """Fit a regression tree on a 3-step response, yielding a depth >= 2 tree."""
    X = numpy.arange(1, 81, dtype=float).reshape(-1, 1)
    y = numpy.where(X.ravel() < 20, 0.0, numpy.where(X.ravel() < 60, 5.0, 10.0))
    regression_tree = sigma._tree_regression.RegressionTree(
        correlation="normal",
        min_splits=2,
        min_buckets=1,
        **kwargs,
    )
    regression_tree.fit(X, y)
    return regression_tree


def _fit_categorical_regression_tree(X_is_dataframe=False, **kwargs):
    """Fit a regression tree with a two-column categorical signal."""
    rng = numpy.random.default_rng(42)
    n = 40
    categorical_column = numpy.repeat([0.0, 1.0], n // 2)
    noise = rng.standard_normal(n)
    y = numpy.where(categorical_column == 0.0, 0.0, 10.0)
    if X_is_dataframe:
        X = pandas.DataFrame({"category": categorical_column, "noise": noise})
    else:
        X = numpy.column_stack([categorical_column, noise])
    regression_tree = sigma._tree_regression.RegressionTree(
        correlation="normal",
        categorical_features=[0] if not X_is_dataframe else ["category"],
        min_splits=2,
        min_buckets=1,
        **kwargs,
    )
    regression_tree.fit(X, y)
    return regression_tree


def _collect_nodes(node: _NodeT) -> list[_NodeT]:
    """Collect all nodes in a tree via pre-order traversal."""
    nodes = [node]
    extension = node.extension
    if isinstance(extension, sigma._partition.Partition):
        nodes.extend(_collect_nodes(typing.cast(_NodeT, extension.left)))
        nodes.extend(_collect_nodes(typing.cast(_NodeT, extension.right)))
    return nodes


def _step_X_y_regression(n: int = 40) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Build a clear two-step regression problem on a single feature."""
    X = numpy.arange(1, n + 1, dtype=float).reshape(-1, 1)
    y = numpy.where(X.ravel() <= n // 2, 0.0, 10.0)
    return X, y


def _step_X_y_classification(
    n: int = 40,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """Build a clear two-step classification problem on a single feature."""
    X = numpy.arange(1, n + 1, dtype=float).reshape(-1, 1)
    y = numpy.where(X.ravel() <= n // 2, 0, 1)
    return X, y


def _make_multifeature_X(n: int = 200, seed: int = 0) -> numpy.ndarray:
    """Build a 4-feature i.i.d. uniform[0, 10] design matrix."""
    rng = numpy.random.default_rng(seed)
    X = rng.uniform(0.0, 10.0, size=(n, 4))
    return X


def _biased_quadrant_mask(X: numpy.ndarray) -> numpy.ndarray:
    """Boolean mask for the (X[:, 0] > 5) AND (X[:, 1] > 5) quadrant."""
    mask = (X[:, 0] > 5.0) & (X[:, 1] > 5.0)
    return mask


def _collect_split_features(node) -> set[int]:
    """Walk the tree and collect every internal node's partition.feature_index."""
    features: set[int] = set()
    extension = node.extension
    if not isinstance(extension, sigma._partition.Partition):
        return features
    features.add(int(extension.feature_index))
    features |= _collect_split_features(extension.left)
    features |= _collect_split_features(extension.right)
    return features
