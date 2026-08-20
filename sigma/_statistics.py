"""Statistical engine for conditional inference.

Implements the linear statistics, conditional moments, test statistics, p-value
computation, and variable selection procedure described in Hothorn, Hornik, and
Zeileis (2006), "Unbiased Recursive Partitioning: A Conditional Inference
Framework," *Journal of Computational and Graphical Statistics*, 15(3), 651-674.
The conditional distribution used for the p-value follows Hothorn, Hornik,
van de Wiel, and Zeileis (2006), "A Lego System for Conditional Inference,"
*The American Statistician*, 60(3), 257-263.
"""

import collections.abc
import typing

import numpy
import numpy.typing
import scipy.stats

from . import _feature, _types

_INTEGRATION_SEED = 0


class _VariableSelection(typing.NamedTuple):
    """Selected variable and its test inputs at one tree node."""

    feature_index: int
    p_value: float
    T: numpy.typing.NDArray[numpy.floating]
    mu: numpy.typing.NDArray[numpy.floating]
    Sigma: numpy.typing.NDArray[numpy.floating]


def compute_linear_statistic(
    g_j: numpy.typing.NDArray[numpy.floating],
    h: numpy.typing.NDArray[numpy.floating],
    weights: numpy.typing.NDArray[numpy.floating],
) -> numpy.typing.NDArray[numpy.floating]:
    """Compute the linear statistic T_j.

    Args:
        g_j: Covariate transformation values, shape (n, p_j).
        h: Influence function values, shape (n, q).
        weights: Case weights, shape (n,).

    Returns:
        Linear statistic vector, shape (p_j * q,).
    """
    weighted_g = g_j * weights[:, numpy.newaxis]
    matrix = weighted_g.T @ h
    result = matrix.ravel(order="F")
    return result


def compute_conditional_expectation(
    g_j: numpy.typing.NDArray[numpy.floating],
    h: numpy.typing.NDArray[numpy.floating],
    weights: numpy.typing.NDArray[numpy.floating],
) -> numpy.typing.NDArray[numpy.floating]:
    """Compute the conditional expectation mu_j = E(T_j | S).

    Args:
        g_j: Covariate transformation values, shape (n, p_j).
        h: Influence function values, shape (n, q).
        weights: Case weights, shape (n,).

    Returns:
        Conditional expectation vector, shape (p_j * q,).
    """
    w_dot = weights.sum()
    weighted_g_sum = (g_j * weights[:, numpy.newaxis]).sum(axis=0)
    e_h = (h * weights[:, numpy.newaxis]).sum(axis=0) / w_dot
    matrix = numpy.outer(weighted_g_sum, e_h)
    result = matrix.ravel(order="F")
    return result


def compute_conditional_covariance(
    g_j: numpy.typing.NDArray[numpy.floating],
    h: numpy.typing.NDArray[numpy.floating],
    weights: numpy.typing.NDArray[numpy.floating],
) -> numpy.typing.NDArray[numpy.floating]:
    """Compute the conditional covariance Sigma_j = V(T_j | S).

    Args:
        g_j: Covariate transformation values, shape (n, p_j).
        h: Influence function values, shape (n, q).
        weights: Case weights, shape (n,).

    Returns:
        Conditional covariance matrix, shape (p_j*q, p_j*q).
    """
    w_dot = weights.sum()
    e_h = (h * weights[:, numpy.newaxis]).sum(axis=0) / w_dot
    h_centered = h - e_h
    weighted_h = h_centered * weights[:, numpy.newaxis]
    v_h = weighted_h.T @ h_centered / w_dot
    weighted_g = g_j * weights[:, numpy.newaxis]
    g_quad = weighted_g.T @ g_j
    g_sum = weighted_g.sum(axis=0)
    g_outer = numpy.outer(g_sum, g_sum)
    inner = w_dot * g_quad - g_outer
    result = numpy.kron(v_h, inner) / (w_dot - 1)
    return result


def compute_test_statistic(
    T: numpy.typing.NDArray[numpy.floating],
    mu: numpy.typing.NDArray[numpy.floating],
    Sigma: numpy.typing.NDArray[numpy.floating],
    test_stat: _types.TestStat,
) -> float:
    """Compute the test statistic.

    Args:
        T: Linear statistic, shape (d,).
        mu: Conditional expectation, shape (d,).
        Sigma: Conditional covariance, shape (d, d).
        test_stat: Type of test statistic.

    Returns:
        The computed test statistic value.
    """
    difference = T - mu
    if test_stat == _types.TestStat.QUADRATIC:
        sigma_pseudoinverse = numpy.linalg.pinv(Sigma)
        result = float(difference @ sigma_pseudoinverse @ difference)
        return result
    diagonal = numpy.diag(Sigma)
    valid = diagonal > 0
    if not valid.any():
        return 0.0
    standardized = numpy.abs(difference[valid]) / numpy.sqrt(diagonal[valid])
    result = float(standardized.max())
    return result


def compute_p_value(
    statistic: float,
    Sigma: numpy.typing.NDArray[numpy.floating],
    test_stat: _types.TestStat,
) -> float:
    """Compute the p-value for the test statistic.

    Args:
        statistic: The observed test statistic value.
        Sigma: Conditional covariance matrix, shape (d, d). May be
            rank-deficient; both branches handle the singular case.
        test_stat: Type of test statistic.

    Returns:
        The computed p-value.
    """
    if test_stat == _types.TestStat.QUADRATIC:
        rank = numpy.linalg.matrix_rank(Sigma)
        if rank == 0:
            return 1.0
        p_value = float(scipy.stats.chi2.sf(statistic, df=rank))
        return p_value
    diagonal = numpy.diag(Sigma)
    valid = diagonal > 0
    if not valid.any():
        return 1.0
    valid_indices = numpy.where(valid)[0]
    Sigma_valid = Sigma[numpy.ix_(valid_indices, valid_indices)]
    standard_deviations = numpy.sqrt(numpy.diag(Sigma_valid))
    R = Sigma_valid / numpy.outer(standard_deviations, standard_deviations)
    d = len(valid_indices)
    lower = numpy.full(d, -statistic)
    upper = numpy.full(d, statistic)
    means = numpy.zeros(d)
    # scipy integrates the orthant probability with a randomized
    # quasi-Monte Carlo rule and draws its randomization from the global
    # numpy random state when no seed is given, which would make the
    # p-value depend on unrelated code. A fresh distribution seeded with
    # the same constant on every call keeps the rule fixed.
    distribution = scipy.stats.multivariate_normal(
        mean=means,
        cov=R,
        allow_singular=True,
        seed=_INTEGRATION_SEED,
        maxpts=25000,
        abseps=1e-3,
        releps=0,
    )
    orthant = distribution.cdf(upper, lower_limit=lower)
    prob = float(orthant)
    p_value = 1.0 - prob
    return p_value


def select_variable(
    X: numpy.typing.NDArray[numpy.floating],
    h: numpy.typing.NDArray[numpy.floating],
    weights: numpy.typing.NDArray[numpy.floating],
    features: collections.abc.Sequence[_feature.Feature],
    test_stat: _types.TestStat,
    test_type: _types.TestType,
    alpha: float = 0.05,
    correlation: _types.Correlation = _types.Correlation.RANK,
    resamples: None | int = None,
    rng: None | numpy.random.Generator = None,
) -> None | _VariableSelection:
    """Select the covariate with strongest association to the response.

    Applies multiplicity adjustment and returns None if the global null
    hypothesis of independence cannot be rejected.

    Args:
        X: Covariate matrix, shape (n, m).
        h: Influence function values, shape (n, q).
        weights: Case weights, shape (n,).
        features: One Feature per column of X, in column order.
        test_stat: Type of test statistic.
        test_type: Multiplicity adjustment method.
        alpha: Significance level.
        correlation: Correlation type.
        resamples: Number of permutations for min-P resampling. Required when
            test_type is MONTE_CARLO.
        rng: Random number generator for permutation resampling. Required when
            test_type is MONTE_CARLO.

    Returns:
        A _VariableSelection if the global null is rejected, None otherwise.

    Raises:
        ValueError: If test_type is MONTE_CARLO but resamples or rng is None.
    """
    is_rank = correlation == _types.Correlation.RANK
    is_regression = h.shape[1] == 1
    if is_rank and is_regression:
        h = _rank_transform(h, weights)
    m = X.shape[1]
    p_values = numpy.ones(m)
    g_list: list[None | numpy.typing.NDArray[numpy.floating]] = []
    T_list: list[None | numpy.typing.NDArray[numpy.floating]] = []
    mu_list: list[None | numpy.typing.NDArray[numpy.floating]] = []
    Sigma_list: list[None | numpy.typing.NDArray[numpy.floating]] = []
    for j in range(m):
        match features[j]:
            case _feature.CategoricalFeature():
                categories = _active_unique_levels(X, weights, j)
                if len(categories) < 2:
                    _append_skipped_feature(g_list, T_list, mu_list, Sigma_list)
                    continue
                g_j = (X[:, j : j + 1] == categories).astype(float)
            case _feature.BooleanFeature():
                unique_values = _active_unique_levels(X, weights, j)
                if len(unique_values) < 2:
                    _append_skipped_feature(g_list, T_list, mu_list, Sigma_list)
                    continue
                g_j = X[:, j : j + 1]
            case _:
                g_j = _numeric_selection_design(X[:, j], weights, is_rank)
        g_list.append(g_j)
        T = compute_linear_statistic(g_j, h, weights)
        mu = compute_conditional_expectation(g_j, h, weights)
        Sigma = compute_conditional_covariance(g_j, h, weights)
        T_list.append(T)
        mu_list.append(mu)
        Sigma_list.append(Sigma)
        statistic = compute_test_statistic(T, mu, Sigma, test_stat)
        p_values[j] = compute_p_value(statistic, Sigma, test_stat)
    match test_type:
        case _types.TestType.MONTE_CARLO:
            if resamples is None or rng is None:
                raise ValueError(
                    "resamples and rng are required for MONTE_CARLO test_type"
                )
            valid_indices = [j for j, g in enumerate(g_list) if g is not None]
            valid_g = [g for g in g_list if g is not None]
            valid_mu = [mu_j for mu_j in mu_list if mu_j is not None]
            valid_Sigma = [
                Sigma_j for Sigma_j in Sigma_list if Sigma_j is not None
            ]
            valid_p = p_values[valid_indices]
            adjusted_valid = _adjust_p_values_monte_carlo(
                valid_p,
                valid_g,
                h,
                weights,
                valid_mu,
                valid_Sigma,
                test_stat,
                resamples,
                rng,
            )
            adjusted_p_values: numpy.typing.NDArray[numpy.floating] = (
                numpy.ones(m)
            )
            for k, j in enumerate(valid_indices):
                adjusted_p_values[j] = adjusted_valid[k]
        case _:
            adjusted_p_values = _adjust_p_values(p_values, test_type)
    j_star = int(numpy.argmin(adjusted_p_values))
    if adjusted_p_values[j_star] > alpha:
        return None
    array_type = numpy.typing.NDArray[numpy.floating]
    result = _VariableSelection(
        feature_index=j_star,
        p_value=float(adjusted_p_values[j_star]),
        T=typing.cast(array_type, T_list[j_star]),
        mu=typing.cast(array_type, mu_list[j_star]),
        Sigma=typing.cast(array_type, Sigma_list[j_star]),
    )
    return result


def _active_unique_levels(
    X: numpy.typing.NDArray[numpy.floating],
    weights: numpy.typing.NDArray[numpy.floating],
    j: int,
) -> numpy.typing.NDArray[numpy.floating]:
    """Return the distinct values of column j over positive-weight rows."""
    active = weights > 0
    # Callers pass NaN-free columns only (categorical missing is already coded
    # to the N/A level, clean booleans carry no NaN), so numpy.unique needs no
    # missing-value filter here.
    levels = numpy.unique(X[active, j])
    return levels


def _append_skipped_feature(
    g_list: list[None | numpy.typing.NDArray[numpy.floating]],
    T_list: list[None | numpy.typing.NDArray[numpy.floating]],
    mu_list: list[None | numpy.typing.NDArray[numpy.floating]],
    Sigma_list: list[None | numpy.typing.NDArray[numpy.floating]],
) -> None:
    """Append None placeholders for a skipped feature to the result lists."""
    g_list.append(None)
    T_list.append(None)
    mu_list.append(None)
    Sigma_list.append(None)


def _adjust_p_values(
    p_values: numpy.typing.NDArray[numpy.floating],
    test_type: _types.TestType,
) -> numpy.typing.NDArray[numpy.floating]:
    """Apply multiplicity adjustment to raw p-values."""
    m = len(p_values)
    match test_type:
        case _types.TestType.BONFERRONI:
            adjusted_p_values = numpy.minimum(p_values * m, 1.0)
        case _types.TestType.SIDAK:
            adjusted_p_values = 1.0 - (1.0 - p_values) ** m
        case _types.TestType.MONTE_CARLO:
            raise ValueError(
                "MONTE_CARLO adjustment requires resampling;"
                " use select_variable() instead"
            )
    return adjusted_p_values


def _adjust_p_values_monte_carlo(
    observed_p_values: numpy.typing.NDArray[numpy.floating],
    g_list: collections.abc.Sequence[numpy.typing.NDArray[numpy.floating]],
    h: numpy.typing.NDArray[numpy.floating],
    weights: numpy.typing.NDArray[numpy.floating],
    mu_list: collections.abc.Sequence[numpy.typing.NDArray[numpy.floating]],
    Sigma_list: collections.abc.Sequence[numpy.typing.NDArray[numpy.floating]],
    test_stat: _types.TestStat,
    resamples: int,
    rng: numpy.random.Generator,
) -> numpy.typing.NDArray[numpy.floating]:
    """Compute min-P value resampling adjusted p-values by permutation."""
    active_indices = numpy.where(weights > 0)[0]
    n_active = len(active_indices)
    m = len(observed_p_values)
    exceed_count = numpy.zeros(m, dtype=int)
    for _ in range(resamples):
        permutation = rng.permutation(n_active)
        h_permuted = h.copy()
        h_permuted[active_indices] = h[active_indices[permutation]]
        permuted_p_values = numpy.ones(m)
        for j in range(m):
            g_j = g_list[j]
            mu_j = mu_list[j]
            Sigma_j = Sigma_list[j]
            T = compute_linear_statistic(g_j, h_permuted, weights)
            statistic = compute_test_statistic(T, mu_j, Sigma_j, test_stat)
            permuted_p_values[j] = compute_p_value(
                statistic, Sigma_j, test_stat
            )
        min_p_value = permuted_p_values.min()
        exceed_count += (min_p_value <= observed_p_values).astype(int)
    adjusted_p_values = (exceed_count + 1) / (resamples + 1)
    result = numpy.minimum(adjusted_p_values, 1.0)
    return result


def _numeric_selection_design(
    column: numpy.typing.NDArray[numpy.floating],
    weights: numpy.typing.NDArray[numpy.floating],
    is_rank: bool,
) -> numpy.typing.NDArray[numpy.floating]:
    """Build the selection design for a numeric covariate column."""
    mask = ~numpy.isnan(column)
    g_value = column.reshape(-1, 1)
    if is_rank:
        g_value = _rank_transform(g_value, weights)
    if mask.all():
        return g_value
    g_value = _fill_missing_observed_mean(g_value, mask, weights)
    indicator = (~mask).astype(float).reshape(-1, 1)
    g_j = numpy.hstack([g_value, indicator])
    return g_j


def _fill_missing_observed_mean(
    column: numpy.typing.NDArray[numpy.floating],
    mask: numpy.typing.NDArray[numpy.bool_],
    weights: numpy.typing.NDArray[numpy.floating],
) -> numpy.typing.NDArray[numpy.floating]:
    """Fill the missing rows of a single-column design with the weighted mean
    of its observed positive-weight rows."""
    observed = mask & (weights > 0)
    w_observed = weights[observed]
    w_sum = w_observed.sum()
    if w_sum > 0:
        mean_value = (column[observed, 0] * w_observed).sum() / w_sum
    else:
        mean_value = 0.0
    filled = column.copy()
    filled[~mask, 0] = mean_value
    return filled


def _rank_transform(
    matrix: numpy.typing.NDArray[numpy.floating],
    weights: numpy.typing.NDArray[numpy.floating],
) -> numpy.typing.NDArray[numpy.floating]:
    """Rank-transform columns among active (weight > 0) observed samples."""
    active = weights > 0
    ranked = numpy.zeros_like(matrix)
    for column_index in range(matrix.shape[1]):
        column = matrix[:, column_index]
        rows = active & ~numpy.isnan(column)
        ranked[rows, column_index] = scipy.stats.rankdata(
            column[rows], method="average"
        )
    return ranked
