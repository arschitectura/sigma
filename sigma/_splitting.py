"""Split search for conditional inference trees.

Implements Step 2 of the algorithm in Hothorn, Hornik, and Zeileis (2006),
"Unbiased Recursive Partitioning: A Conditional Inference Framework," *Journal
of Computational and Graphical Statistics*, 15(3), 651-674: given the selected
covariate, find the binary partition that maximizes the test statistic.
"""

import itertools

import numpy
import numpy.typing

from . import _statistics
from . import _types


def find_best_split_numeric(
    X_j: numpy.typing.NDArray[numpy.floating],
    h: numpy.typing.NDArray[numpy.floating],
    weights: numpy.typing.NDArray[numpy.floating],
    test_stat: _types.TestStat,
    min_buckets: int,
    is_integer: bool,
    correlation: _types.Correlation = _types.Correlation.RANK,
) -> None | tuple[float, float]:
    """Find the optimal numeric split threshold for covariate X_j.

    Args:
        X_j: Numeric covariate values, shape (n,).
        h: Influence function values, shape (n, q).
        weights: Case weights, shape (n,).
        test_stat: Type of test statistic.
        min_buckets: Minimum sum of weights in each child node.
        is_integer: True when the covariate's observed values are all
            integer-valued.
        correlation: Correlation type for the test statistic.

    Returns:
        (threshold, test_statistic) if a valid split exists, None otherwise.
    """
    h = _maybe_rank_h(h, weights, correlation)
    active = weights > 0
    unique_values = numpy.unique(X_j[active])
    if len(unique_values) < 2:
        return None
    if is_integer:
        candidate_thresholds = unique_values[:-1]
    else:
        candidate_thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0
    best_statistic = -numpy.inf
    best_threshold = None
    for threshold in candidate_thresholds:
        left_mask = X_j <= threshold
        left_weight = weights[left_mask].sum()
        right_weight = weights[~left_mask].sum()
        if left_weight < min_buckets or right_weight < min_buckets:
            continue
        g_j = left_mask.astype(float).reshape(-1, 1)
        T = _statistics.compute_linear_statistic(g_j, h, weights)
        mu = _statistics.compute_conditional_expectation(g_j, h, weights)
        Sigma = _statistics.compute_conditional_covariance(g_j, h, weights)
        statistic = _statistics.compute_test_statistic(T, mu, Sigma, test_stat)
        if statistic > best_statistic:
            best_statistic = statistic
            best_threshold = threshold
    if best_threshold is None:
        return None
    result = (float(best_threshold), float(best_statistic))
    return result


def find_best_split_boolean(
    X_j: numpy.typing.NDArray[numpy.floating],
    h: numpy.typing.NDArray[numpy.floating],
    weights: numpy.typing.NDArray[numpy.floating],
    test_stat: _types.TestStat,
    min_buckets: int,
    correlation: _types.Correlation = _types.Correlation.RANK,
) -> None | tuple[bool, float]:
    """Find the unique split for a boolean covariate X_j.

    Args:
        X_j: Boolean covariate values, shape (n,), encoded as 0.0 / 1.0
            floats.
        h: Influence function values, shape (n, q).
        weights: Case weights, shape (n,).
        test_stat: Type of test statistic.
        min_buckets: Minimum sum of weights in each child node.
        correlation: Correlation type for the test statistic.

    Returns:
        (True, test_statistic) when the unique partition is feasible, None
        otherwise.
    """
    del correlation
    active = weights > 0
    unique_values = numpy.unique(X_j[active])
    if len(unique_values) < 2:
        return None
    left_mask = X_j <= 0.5
    left_weight = weights[left_mask].sum()
    right_weight = weights[~left_mask].sum()
    if left_weight < min_buckets or right_weight < min_buckets:
        return None
    g_j = left_mask.astype(float).reshape(-1, 1)
    T = _statistics.compute_linear_statistic(g_j, h, weights)
    mu = _statistics.compute_conditional_expectation(g_j, h, weights)
    Sigma = _statistics.compute_conditional_covariance(g_j, h, weights)
    statistic = _statistics.compute_test_statistic(T, mu, Sigma, test_stat)
    result = (True, float(statistic))
    return result


def find_best_split_categorical(
    X_j: numpy.typing.NDArray[numpy.floating],
    h: numpy.typing.NDArray[numpy.floating],
    weights: numpy.typing.NDArray[numpy.floating],
    test_stat: _types.TestStat,
    min_buckets: int,
    correlation: _types.Correlation = _types.Correlation.RANK,
) -> None | tuple[frozenset, float]:
    """Find the optimal categorical partition for covariate X_j.

    Args:
        X_j: Categorical covariate values, shape (n,).
        h: Influence function values, shape (n, q).
        weights: Case weights, shape (n,).
        test_stat: Type of test statistic.
        min_buckets: Minimum sum of weights in each child node.
        correlation: Correlation type for the test statistic.

    Returns:
        (left_categories, test_statistic) if a valid split exists, None
        otherwise.
    """
    h = _maybe_rank_h(h, weights, correlation)
    active = weights > 0
    categories = numpy.unique(X_j[active])
    K = len(categories)
    if K < 2:
        return None
    if K <= 10:
        candidates = _exhaustive_partitions(categories)
    else:
        candidates = _mean_ordered_partitions(categories, X_j, h, weights)
    best_statistic = -numpy.inf
    best_left: None | frozenset = None
    for left_set in candidates:
        left_mask = numpy.isin(X_j, list(left_set))
        left_weight = weights[left_mask].sum()
        right_weight = weights[~left_mask].sum()
        if left_weight < min_buckets or right_weight < min_buckets:
            continue
        g_j = left_mask.astype(float).reshape(-1, 1)
        T = _statistics.compute_linear_statistic(g_j, h, weights)
        mu = _statistics.compute_conditional_expectation(g_j, h, weights)
        Sigma = _statistics.compute_conditional_covariance(g_j, h, weights)
        statistic = _statistics.compute_test_statistic(T, mu, Sigma, test_stat)
        if statistic > best_statistic:
            best_statistic = statistic
            best_left = frozenset(left_set)
    if best_left is None:
        return None
    result = (best_left, float(best_statistic))
    return result


def find_best_split(
    X: numpy.typing.NDArray[numpy.floating],
    h: numpy.typing.NDArray[numpy.floating],
    weights: numpy.typing.NDArray[numpy.floating],
    feature_index: int,
    feature_types: numpy.typing.NDArray,
    test_stat: _types.TestStat,
    min_buckets: int,
    correlation: _types.Correlation = _types.Correlation.RANK,
) -> None | tuple[float | frozenset | bool, float]:
    """Find the optimal binary split for the given feature.

    Args:
        X: Covariate matrix, shape (n, m).
        h: Influence function values, shape (n, q).
        weights: Case weights, shape (n,).
        feature_index: Index of the feature to split on.
        feature_types: Per-feature CovariateType, shape (m,).
        test_stat: Type of test statistic.
        min_buckets: Minimum sum of weights in each child node.
        correlation: Correlation type for the test statistic.

    Returns:
        (split_criterion, test_statistic) if a valid split exists, None
        otherwise. split_criterion is the literal True sentinel for boolean
        features, a float threshold for numeric features, or a frozenset of
        left categories for categorical features.
    """
    X_j = X[:, feature_index]
    result: None | tuple[float | frozenset | bool, float]
    match feature_types[feature_index]:
        case _types.CovariateType.BOOLEAN:
            result = find_best_split_boolean(
                X_j, h, weights, test_stat, min_buckets, correlation
            )
        case _types.CovariateType.CATEGORICAL:
            result = find_best_split_categorical(
                X_j, h, weights, test_stat, min_buckets, correlation
            )
        case _types.CovariateType.INTEGER:
            result = find_best_split_numeric(
                X_j,
                h,
                weights,
                test_stat,
                min_buckets,
                is_integer=True,
                correlation=correlation,
            )
        case _types.CovariateType.REAL:
            result = find_best_split_numeric(
                X_j,
                h,
                weights,
                test_stat,
                min_buckets,
                is_integer=False,
                correlation=correlation,
            )
    return result


def _maybe_rank_h(
    h: numpy.typing.NDArray[numpy.floating],
    weights: numpy.typing.NDArray[numpy.floating],
    correlation: _types.Correlation,
) -> numpy.typing.NDArray[numpy.floating]:
    """Rank-transform h if correlation is RANK and h is continuous."""
    is_rank = correlation == _types.Correlation.RANK
    is_regression = h.shape[1] == 1
    if is_rank and is_regression:
        h = _statistics._rank_transform(h, weights)
    return h


def _exhaustive_partitions(
    categories: numpy.typing.NDArray[numpy.floating],
) -> list[list[float]]:
    """Generate all non-trivial binary partitions."""
    rest = list(categories[:-1])
    subsets = itertools.chain.from_iterable(
        itertools.combinations(rest, r) for r in range(1, len(rest) + 1)
    )
    partitions = [list(subset) for subset in subsets]
    return partitions


def _mean_ordered_partitions(
    categories: numpy.typing.NDArray[numpy.floating],
    X_j: numpy.typing.NDArray[numpy.floating],
    h: numpy.typing.NDArray[numpy.floating],
    weights: numpy.typing.NDArray[numpy.floating],
) -> list[list[float]]:
    """Generate contiguous partitions ordered by mean influence."""
    h_first_column = h[:, 0]
    means = numpy.empty(len(categories))
    for i, category in enumerate(categories):
        mask = X_j == category
        w = weights[mask]
        w_sum = w.sum()
        if w_sum > 0:
            means[i] = (w * h_first_column[mask]).sum() / w_sum
        else:
            means[i] = 0.0
    order = numpy.argsort(means)
    sorted_categories = categories[order]
    partitions = [
        list(sorted_categories[: k + 1])
        for k in range(len(sorted_categories) - 1)
    ]
    return partitions
