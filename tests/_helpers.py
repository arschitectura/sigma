"""Shared fixtures and helpers used across the test suite."""

import os
import typing
import zipfile

import numpy
import pandas

import sigma._node
import sigma._partition
import sigma._tree_classification
import sigma._tree_regression
import sigma._tree_survival

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_DEMO_DATA_DIR = os.path.join(_REPO_ROOT, ".demo_data")


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


def _fit_step_classification_tree(**kwargs):
    """Fit a binary classification tree on a perfectly separable step function."""
    X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
    y = numpy.where(X.ravel() <= 20, 0.0, 1.0)
    classification_tree = sigma._tree_classification.ClassificationTree(
        correlation="normal",
        min_splits=2,
        min_buckets=1,
        **kwargs,
    )
    classification_tree.fit(X, y)
    return classification_tree


def _fit_step_survival_tree(**kwargs):
    """Fit a survival tree on a binary categorical signal with median metric."""
    times = numpy.linspace(1.0, 10.0, 60)
    events = numpy.tile([1.0, 0.0], 30)
    y = numpy.column_stack([times, events])
    X = numpy.column_stack([numpy.repeat([0.0, 1.0], 30)])
    survival_tree = sigma._tree_survival.SurvivalTree(
        categorical_features=[0],
        min_splits=2,
        min_buckets=1,
        **kwargs,
    )
    survival_tree.fit(X, y)
    return survival_tree


def _collect_nodes(node: _NodeT) -> list[_NodeT]:
    """Collect all nodes in a tree via pre-order traversal."""
    nodes = [node]
    extension = node.extension
    if isinstance(extension, sigma._partition.Partition):
        for child in extension.children:
            nodes.extend(_collect_nodes(child))
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
    for child in extension.children:
        features |= _collect_split_features(child)
    return features


def _load_sushi3a():
    """Load the full SUSHI3A demo dataset as (X, rankings)."""
    zip_path = os.path.join(_DEMO_DATA_DIR, "sushi3-2016.zip")
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open("sushi3-2016/sushi3a.5000.10.order") as order_file:
            order_lines = order_file.read().decode("utf-8").splitlines()
        with archive.open("sushi3-2016/sushi3.udata") as udata_file:
            udata_lines = udata_file.read().decode("utf-8").splitlines()
        with archive.open("sushi3-2016/sushi3.idata") as idata_file:
            idata_lines = idata_file.read().decode("utf-8").splitlines()
    item_count = 10
    all_item_names = [
        line.split("\t")[1] for line in idata_lines if line.strip()
    ]
    item_names = all_item_names[:item_count]
    n_kept_rows = sum(1 for line in order_lines[1:] if line.strip())
    ranks_in_cell = numpy.full(
        (n_kept_rows, item_count), numpy.nan, dtype=float
    )
    row_index = 0
    for line in order_lines[1:]:
        if not line.strip():
            continue
        tokens = line.split()
        for position in range(item_count):
            item_id = int(tokens[2 + position])
            ranks_in_cell[row_index, item_id] = float(position + 1)
        row_index += 1
    rankings = pandas.DataFrame(ranks_in_cell, columns=item_names)
    demographic_rows = [
        line.split("\t") for line in udata_lines if line.strip()
    ]
    demographic_frame = pandas.DataFrame(
        demographic_rows,
        columns=[
            "user_id",
            "gender",
            "age_group",
            "completion_seconds",
            "childhood_prefecture",
            "childhood_region",
            "childhood_east_west",
            "current_prefecture",
            "current_region",
            "current_east_west",
            "migrated",
        ],
    )
    gender_codes = demographic_frame["gender"].map({"0": "male", "1": "female"})
    X = pandas.DataFrame(
        {
            "Gender": pandas.Categorical(
                gender_codes, categories=["female", "male"]
            ),
            "Childhood region": pandas.Categorical(
                demographic_frame["childhood_region"]
            ),
        }
    )
    return X, rankings


def _load_sushi3a_subset(n_users):
    """Load the first n_users rows of the SUSHI3A demo dataset as (X, rankings)."""
    X, rankings = _load_sushi3a()
    X_sub = X.iloc[:n_users].reset_index(drop=True)
    rankings_sub = rankings.iloc[:n_users].reset_index(drop=True)
    return X_sub, rankings_sub


def _random_alpha(n_items, rng):
    """Sample a worth vector with geometric mean approximately 1."""
    log_alpha = rng.normal(size=n_items) * 0.5
    log_alpha -= log_alpha.mean()
    alpha = numpy.exp(log_alpha)
    return alpha
