"""Ranking-data primitives used by RankingTree.

`compute_pl_mle` and `pl_expected_rank` together form the leaf-level
statistic of `RankingTree`: each leaf fits a Plackett-Luce model on its
weighted partial rankings by Hunter (2004) MM iterations, regularised by
Turner et al. (2020) ghost-item pseudo-rankings, then reports the per-item
expected rank under the fitted worth vector. `compute_mean_rank_vector`
is a legacy utility retained as a public helper but no longer used by
`RankingTree`. The response matrix uses the ranks-in-cell layout: rows
are observations, columns are items, each cell carries the rank position
of that item for that observation. Unranked items are NaN.
"""

import numpy
import numpy.typing


def compute_pl_mle(
    y: numpy.typing.NDArray[numpy.floating],
    weights: numpy.typing.NDArray[numpy.floating],
    npseudo: float = 0.5,
    tolerance: float = 1.0e-6,
    max_iter: int = 100,
) -> numpy.typing.NDArray[numpy.floating]:
    """Fit the regularised Plackett-Luce worth vector to weighted rankings.

    Args:
        y: Ranks-in-cell array of shape (n_obs, n_items). NaN entries
            flag items the row did not rank; the corresponding row's
            partial ranking is read from the non-NaN cells sorted in
            ascending rank order.
        weights: Per-row weights, shape (n_obs,). Rows with non-positive
            weight or entirely NaN cells contribute nothing.
        npseudo: Weight of the Turner ghost-item pseudo-comparisons.
            Each real item receives `npseudo` pseudo-wins and `npseudo`
            pseudo-losses against a hypothetical ghost item of fixed
            worth equal to the geometric mean of real worths, providing
            shrinkage toward equal worth. Must be strictly positive.
            Defaults to 0.5 (Turner et al. 2020 default).
        tolerance: Convergence tolerance on the maximum absolute change
            in log-worth between successive MM iterations. Must be
            strictly positive.
        max_iter: Maximum number of MM iterations before the algorithm
            returns the current iterate without further refinement.
            Must be a positive integer.

    Returns:
        Length-n_items vector of worth estimates, normalised so that
        the geometric mean across items equals 1. Returns a vector of
        NaN when no row has positive weight or every row with positive
        weight has all-NaN cells.
    """
    n_obs, n_items = y.shape
    if n_obs == 0:
        empty = numpy.full(n_items, numpy.nan, dtype=float)
        return empty
    orderings, ordering_weights = _extract_orderings(y, weights)
    if len(orderings) == 0:
        empty = numpy.full(n_items, numpy.nan, dtype=float)
        return empty
    real_w = numpy.zeros(n_items, dtype=float)
    for ordering, w in zip(orderings, ordering_weights):
        real_w[ordering] += w
    alpha = numpy.ones(n_items, dtype=float)
    alpha_ghost = 1.0
    log_alpha = numpy.zeros(n_items, dtype=float)
    log_alpha_cap = 30.0
    for _ in range(max_iter):
        real_d = _accumulate_pl_denominator(
            alpha, orderings, ordering_weights, n_items
        )
        pseudo_d = 2.0 * npseudo / (alpha + alpha_ghost)
        total_d = real_d + pseudo_d
        total_w = real_w + npseudo
        alpha_new = total_w / total_d
        log_alpha_new = numpy.log(alpha_new)
        log_alpha_new -= log_alpha_new.mean()
        numpy.clip(
            log_alpha_new, -log_alpha_cap, log_alpha_cap, out=log_alpha_new
        )
        delta = float(numpy.max(numpy.abs(log_alpha - log_alpha_new)))
        log_alpha = log_alpha_new
        alpha = numpy.exp(log_alpha_new)
        if delta < tolerance:
            break
    return alpha


def pl_expected_rank(
    alpha: numpy.typing.NDArray[numpy.floating],
) -> numpy.typing.NDArray[numpy.floating]:
    """Compute the per-item expected rank under a Plackett-Luce worth vector.

    Args:
        alpha: Length-K worth vector. NaN entries propagate to the
            corresponding output positions.

    Returns:
        Length-K vector of expected ranks. Each finite entry lies in
        [1, K]; entries corresponding to NaN inputs are NaN.
    """
    n_items = alpha.size
    result = numpy.full(n_items, numpy.nan, dtype=float)
    valid_mask = ~numpy.isnan(alpha)
    valid_indices = numpy.flatnonzero(valid_mask)
    n_valid = valid_indices.size
    if n_valid == 0:
        return result
    a = alpha[valid_indices].astype(float, copy=False)
    a_column = a.reshape(-1, 1)
    a_row = a.reshape(1, -1)
    pair_sum = a_column + a_row
    pairwise = a_row / pair_sum
    row_sum = pairwise.sum(axis=1)
    expected_valid = 0.5 + row_sum
    result[valid_indices] = expected_valid
    return result


def compute_mean_rank_vector(
    y: numpy.typing.NDArray[numpy.floating],
    weights: numpy.typing.NDArray[numpy.floating],
) -> numpy.typing.NDArray[numpy.floating]:
    """Compute the weighted mean rank of each item across rows.

    Args:
        y: Ranks-in-cell array of shape (n_obs, n_items). NaN entries are
            ignored on a per-item basis: an item's mean rank is taken
            only over the rows that ranked it.
        weights: Per-row weights, shape (n_obs,).

    Returns:
        Length-n_items vector of weighted mean ranks. Items with zero
        total weight from rows that ranked them are reported as NaN.
    """
    n_obs, n_items = y.shape
    if n_obs == 0:
        empty = numpy.full(n_items, numpy.nan, dtype=float)
        return empty
    valid_mask = ~numpy.isnan(y)
    y_safe = numpy.where(valid_mask, y, 0.0)
    w_column = weights.reshape(-1, 1)
    numerator = (y_safe * w_column).sum(axis=0)
    denominator = (valid_mask * w_column).sum(axis=0)
    safe_denominator = numpy.where(denominator > 0, denominator, 1.0)
    mean_rank = numpy.where(
        denominator > 0, numerator / safe_denominator, numpy.nan
    )
    return mean_rank


def _extract_orderings(
    y: numpy.typing.NDArray[numpy.floating],
    weights: numpy.typing.NDArray[numpy.floating],
) -> tuple[list[numpy.typing.NDArray[numpy.intp]], list[float]]:
    """Return per-row item orderings and their weights, dropping empty rows."""
    n_obs = y.shape[0]
    orderings: list[numpy.typing.NDArray[numpy.intp]] = []
    ordering_weights: list[float] = []
    for row_index in range(n_obs):
        row_weight = float(weights[row_index])
        if row_weight <= 0.0:
            continue
        row = y[row_index]
        valid_mask = ~numpy.isnan(row)
        valid_indices = numpy.flatnonzero(valid_mask)
        if valid_indices.size == 0:
            continue
        sort_order = numpy.argsort(row[valid_indices], kind="stable")
        ordering = valid_indices[sort_order].astype(numpy.intp, copy=False)
        orderings.append(ordering)
        ordering_weights.append(row_weight)
    return orderings, ordering_weights


def _accumulate_pl_denominator(
    alpha: numpy.typing.NDArray[numpy.floating],
    orderings: list[numpy.typing.NDArray[numpy.intp]],
    ordering_weights: list[float],
    n_items: int,
) -> numpy.typing.NDArray[numpy.floating]:
    """Compute Σ_i w_i · Σ_{j: k ∈ A_j(i)} 1/S_j(i) for every item k."""
    real_d = numpy.zeros(n_items, dtype=float)
    global_add = 0.0
    full_sum = float(alpha.sum())
    at_risk_floor = max(full_sum, 1.0) * 1.0e-300
    for ordering, w in zip(orderings, ordering_weights):
        d = ordering.size
        prefix = numpy.cumsum(alpha[ordering])
        at_risk_sum = numpy.empty(d, dtype=float)
        at_risk_sum[0] = full_sum
        if d > 1:
            at_risk_sum[1:] = full_sum - prefix[:-1]
        numpy.maximum(at_risk_sum, at_risk_floor, out=at_risk_sum)
        cumulative_reciprocal = numpy.cumsum(1.0 / at_risk_sum)
        full_cumulative = float(cumulative_reciprocal[-1])
        global_add += w * full_cumulative
        real_d[ordering] -= w * (full_cumulative - cumulative_reciprocal)
    real_d += global_add
    return real_d
