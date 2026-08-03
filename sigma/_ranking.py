"""Ranking-data primitives used by RankingTree.

`compute_pl_mle` and `pl_expected_rank` together form the per-node
statistic of `RankingTree`: each node fits a Plackett-Luce model on its
weighted partial rankings by Hunter (2004) MM iterations, regularised by
Turner et al. (2020) ghost-item pseudo-rankings, then reports the per-item
expected rank under the fitted worth vector. `compute_mean_rank_vector`
is a legacy utility retained as a public helper but no longer used by
`RankingTree`. The response matrix uses the ranks-in-cell layout: rows
are observations, columns are items, each cell carries the rank position
of that item for that observation. Unranked items are NaN.
"""

import dataclasses

import numpy
import numpy.typing

_PL_EXPECTED_RANK_CHUNK = 256


@dataclasses.dataclass(frozen=True, eq=False, slots=True)
class _OrderingsCache:
    """Vectorised representation of per-row partial-permutation orderings."""

    flat_idx: numpy.typing.NDArray[numpy.intp]
    row_sizes: numpy.typing.NDArray[numpy.intp]
    row_starts: numpy.typing.NDArray[numpy.intp]
    row_ends_inclusive: numpy.typing.NDArray[numpy.intp]
    row_indices_in_y: numpy.typing.NDArray[numpy.intp]
    ordering_weights: numpy.typing.NDArray[numpy.floating]


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
    cache = _extract_orderings_cache(y, weights)
    alpha = _compute_pl_mle_from_cache(
        cache,
        cache.ordering_weights,
        n_items,
        npseudo=npseudo,
        tolerance=tolerance,
        max_iter=max_iter,
    )
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
    a_row = a.reshape(1, -1)
    row_sum = numpy.empty(n_valid, dtype=float)
    chunk = _PL_EXPECTED_RANK_CHUNK
    for start in range(0, n_valid, chunk):
        end = min(start + chunk, n_valid)
        a_col_chunk = a[start:end].reshape(-1, 1)
        pairwise_chunk = _pl_pairwise_ratio(a_col_chunk, a_row)
        row_sum[start:end] = pairwise_chunk.sum(axis=1)
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


def _extract_orderings_cache(
    y: numpy.typing.NDArray[numpy.floating],
    weights: numpy.typing.NDArray[numpy.floating],
) -> _OrderingsCache:
    """Pack per-row partial-permutation orderings into a vectorised cache."""
    n_obs = y.shape[0]
    if n_obs == 0:
        return _empty_cache()
    weights_arr = numpy.asarray(weights, dtype=float)
    weight_positive = weights_arr > 0.0
    valid_per_row = (~numpy.isnan(y)).sum(axis=1)
    keep_row = weight_positive & (valid_per_row > 0)
    if not bool(numpy.any(keep_row)):
        return _empty_cache()
    kept_indices = numpy.flatnonzero(keep_row).astype(numpy.intp, copy=False)
    orderings: list[numpy.typing.NDArray[numpy.intp]] = []
    row_sizes_list: list[int] = []
    for row_index in kept_indices:
        row = y[row_index]
        valid_mask = ~numpy.isnan(row)
        valid_indices_row = numpy.flatnonzero(valid_mask)
        sort_order = numpy.argsort(row[valid_indices_row], kind="stable")
        ordering = valid_indices_row[sort_order].astype(numpy.intp, copy=False)
        orderings.append(ordering)
        row_sizes_list.append(int(ordering.size))
    flat_idx = numpy.concatenate(orderings).astype(numpy.intp, copy=False)
    row_sizes = numpy.asarray(row_sizes_list, dtype=numpy.intp)
    row_starts = _row_starts(row_sizes)
    row_ends_inclusive = row_starts + row_sizes - 1
    ordering_weights = weights_arr[kept_indices].astype(float, copy=False)
    cache = _OrderingsCache(
        flat_idx=flat_idx,
        row_sizes=row_sizes,
        row_starts=row_starts,
        row_ends_inclusive=row_ends_inclusive,
        row_indices_in_y=kept_indices,
        ordering_weights=ordering_weights,
    )
    return cache


def _empty_cache() -> _OrderingsCache:
    """Return an _OrderingsCache representing zero orderings."""
    empty_intp = numpy.empty(0, dtype=numpy.intp)
    empty_float = numpy.empty(0, dtype=float)
    cache = _OrderingsCache(
        flat_idx=empty_intp,
        row_sizes=empty_intp,
        row_starts=empty_intp,
        row_ends_inclusive=empty_intp,
        row_indices_in_y=empty_intp,
        ordering_weights=empty_float,
    )
    return cache


def _subset_cache(
    cache: _OrderingsCache,
    indices: numpy.typing.NDArray[numpy.intp],
) -> _OrderingsCache:
    """Gather a subset of cached orderings by ordering-index array."""
    indices_arr = numpy.asarray(indices, dtype=numpy.intp)
    if indices_arr.size == 0:
        return _empty_cache()
    new_row_sizes = cache.row_sizes[indices_arr]
    new_row_starts = _row_starts(new_row_sizes)
    new_row_ends_inclusive = new_row_starts + new_row_sizes - 1
    parts = [
        cache.flat_idx[
            int(cache.row_starts[i]) : int(cache.row_ends_inclusive[i]) + 1
        ]
        for i in indices_arr
    ]
    new_flat_idx = numpy.concatenate(parts).astype(numpy.intp, copy=False)
    new_row_indices_in_y = cache.row_indices_in_y[indices_arr]
    new_ordering_weights = cache.ordering_weights[indices_arr]
    subset = _OrderingsCache(
        flat_idx=new_flat_idx,
        row_sizes=new_row_sizes,
        row_starts=new_row_starts,
        row_ends_inclusive=new_row_ends_inclusive,
        row_indices_in_y=new_row_indices_in_y,
        ordering_weights=new_ordering_weights,
    )
    return subset


def _compute_pl_mle_from_cache(
    cache: _OrderingsCache,
    ordering_weights: numpy.typing.NDArray[numpy.floating],
    n_items: int,
    npseudo: float,
    tolerance: float,
    max_iter: int,
) -> numpy.typing.NDArray[numpy.floating]:
    """Run the Hunter MM iterations on a prebuilt orderings cache."""
    if cache.row_sizes.size == 0:
        empty = numpy.full(n_items, numpy.nan, dtype=float)
        return empty
    weights_arr = numpy.asarray(ordering_weights, dtype=float)
    real_w = numpy.bincount(
        cache.flat_idx,
        weights=numpy.repeat(weights_arr, cache.row_sizes),
        minlength=n_items,
    ).astype(float, copy=False)
    alpha = numpy.ones(n_items, dtype=float)
    alpha_ghost = 1.0
    log_alpha = numpy.zeros(n_items, dtype=float)
    log_alpha_cap = 30.0
    for _ in range(max_iter):
        real_d = _accumulate_pl_denominator(alpha, cache, weights_arr, n_items)
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


@dataclasses.dataclass(frozen=True, eq=False, slots=True)
class _AtRiskAux:
    """Per-flat-step PL at-risk quantities derived from a cache and alpha."""

    at_risk_flat: numpy.typing.NDArray[numpy.floating]
    inv_risk_flat: numpy.typing.NDArray[numpy.floating]
    cum_recip_flat: numpy.typing.NDArray[numpy.floating]
    full_cum_per_row: numpy.typing.NDArray[numpy.floating]
    full_sum: float


def _compute_at_risk_aux(
    cache: _OrderingsCache,
    alpha: numpy.typing.NDArray[numpy.floating],
) -> _AtRiskAux:
    """Compute S_step, 1/S_step, cumulative reciprocal and per-row totals."""
    row_sizes = cache.row_sizes
    row_starts = cache.row_starts
    row_ends_inclusive = cache.row_ends_inclusive
    alpha_flat = alpha[cache.flat_idx]
    full_sum = float(alpha.sum())
    at_risk_floor = max(full_sum, 1.0) * 1.0e-300
    prefix_flat = _segment_prefix(alpha_flat, row_sizes, row_starts)
    at_risk_flat = full_sum - prefix_flat + alpha_flat
    numpy.maximum(at_risk_flat, at_risk_floor, out=at_risk_flat)
    inv_risk_flat = 1.0 / at_risk_flat
    cum_recip_flat = _segment_prefix(inv_risk_flat, row_sizes, row_starts)
    full_cum_per_row = cum_recip_flat[row_ends_inclusive]
    aux = _AtRiskAux(
        at_risk_flat=at_risk_flat,
        inv_risk_flat=inv_risk_flat,
        cum_recip_flat=cum_recip_flat,
        full_cum_per_row=full_cum_per_row,
        full_sum=full_sum,
    )
    return aux


def _accumulate_pl_denominator(
    alpha: numpy.typing.NDArray[numpy.floating],
    cache: _OrderingsCache,
    ordering_weights: numpy.typing.NDArray[numpy.floating],
    n_items: int,
) -> numpy.typing.NDArray[numpy.floating]:
    """Compute Σ_i w_i · Σ_{j: k ∈ A_j(i)} 1/S_j(i) for every item k."""
    if cache.row_sizes.size == 0:
        empty = numpy.zeros(n_items, dtype=float)
        return empty
    aux = _compute_at_risk_aux(cache, alpha)
    weights_arr = numpy.asarray(ordering_weights, dtype=float)
    global_add = float((weights_arr * aux.full_cum_per_row).sum())
    w_flat = numpy.repeat(weights_arr, cache.row_sizes)
    full_cum_repeat = numpy.repeat(aux.full_cum_per_row, cache.row_sizes)
    contrib_flat = -w_flat * (full_cum_repeat - aux.cum_recip_flat)
    real_d = numpy.bincount(
        cache.flat_idx, weights=contrib_flat, minlength=n_items
    ).astype(float, copy=False)
    real_d += global_add
    return real_d


def _compute_pl_score_per_row(
    cache: _OrderingsCache,
    alpha: numpy.typing.NDArray[numpy.floating],
    n_items: int,
) -> numpy.typing.NDArray[numpy.floating]:
    """Per-row PL log-likelihood score on log-alpha, shape (n_orderings, n_items)."""
    n_orderings = int(cache.row_sizes.size)
    if n_orderings == 0:
        empty = numpy.zeros((0, n_items), dtype=float)
        return empty
    aux = _compute_at_risk_aux(cache, alpha)
    score = -numpy.outer(aux.full_cum_per_row, alpha)
    row_of_flat = _row_of_flat(cache.row_sizes)
    full_cum_repeat = aux.full_cum_per_row[row_of_flat]
    tail_recip_flat = full_cum_repeat - aux.cum_recip_flat
    alpha_flat = alpha[cache.flat_idx]
    delta_flat = 1.0 + alpha_flat * tail_recip_flat
    score[row_of_flat, cache.flat_idx] += delta_flat
    return score


def _compute_pl_fisher_info(
    cache: _OrderingsCache,
    alpha: numpy.typing.NDArray[numpy.floating],
    ordering_weights: numpy.typing.NDArray[numpy.floating],
    n_items: int,
    chunk_size: int = 1024,
) -> numpy.typing.NDArray[numpy.floating]:
    """Observed Fisher information of the PL log-likelihood on log-alpha (n_items, n_items)."""
    if cache.row_sizes.size == 0:
        empty = numpy.zeros((n_items, n_items), dtype=float)
        return empty
    aux = _compute_at_risk_aux(cache, alpha)
    weights_arr = numpy.asarray(ordering_weights, dtype=float)
    w_flat = numpy.repeat(weights_arr, cache.row_sizes)
    full_cum_repeat = numpy.repeat(aux.full_cum_per_row, cache.row_sizes)
    tail_recip_flat = full_cum_repeat - aux.cum_recip_flat
    w_total_full_cum = float((weights_arr * aux.full_cum_per_row).sum())
    tail_contrib = numpy.bincount(
        cache.flat_idx,
        weights=w_flat * tail_recip_flat,
        minlength=n_items,
    ).astype(float, copy=False)
    diag_part = alpha * (w_total_full_cum - tail_contrib)
    h = numpy.diag(diag_part)
    n_total = int(cache.flat_idx.size)
    row_of_flat = _row_of_flat(cache.row_sizes)
    step_in_row_flat = (
        numpy.arange(n_total, dtype=numpy.intp) - cache.row_starts[row_of_flat]
    )
    inv_risk_flat = aux.inv_risk_flat
    for t_start in range(0, n_total, chunk_size):
        t_end = min(t_start + chunk_size, n_total)
        chunk_len = t_end - t_start
        u_chunk = alpha[None, :] * inv_risk_flat[t_start:t_end, None]
        counts = step_in_row_flat[t_start:t_end].astype(numpy.intp, copy=False)
        total_priors = int(counts.sum())
        if total_priors > 0:
            chunk_row = row_of_flat[t_start:t_end]
            chunk_step_repeated = numpy.repeat(
                numpy.arange(chunk_len, dtype=numpy.intp), counts
            )
            cumcounts = numpy.empty(chunk_len + 1, dtype=numpy.intp)
            cumcounts[0] = 0
            numpy.cumsum(counts, out=cumcounts[1:])
            offsets_within_row = numpy.arange(
                total_priors, dtype=numpy.intp
            ) - numpy.repeat(cumcounts[:-1], counts)
            chunk_row_repeated = numpy.repeat(chunk_row, counts)
            prior_flat_positions = (
                cache.row_starts[chunk_row_repeated] + offsets_within_row
            )
            prior_item_indices = cache.flat_idx[prior_flat_positions]
            u_chunk[chunk_step_repeated, prior_item_indices] = 0.0
        w_chunk = w_flat[t_start:t_end]
        weighted_u = w_chunk[:, None] * u_chunk
        h -= weighted_u.T @ u_chunk
    return h


def _compute_pl_expected_rank_jacobian(
    alpha: numpy.typing.NDArray[numpy.floating],
) -> numpy.typing.NDArray[numpy.floating]:
    """K-by-K Jacobian g[k, j] = ∂ E[R_k] / ∂ log alpha_j of pl_expected_rank."""
    alpha_arr = numpy.asarray(alpha, dtype=float)
    n_items = alpha_arr.size
    g = numpy.empty((n_items, n_items), dtype=float)
    chunk = _PL_EXPECTED_RANK_CHUNK
    a_row = alpha_arr.reshape(1, -1)
    for start in range(0, n_items, chunk):
        end = min(start + chunk, n_items)
        a_col_chunk = alpha_arr[start:end].reshape(-1, 1)
        s_chunk = _pl_pairwise_ratio(a_col_chunk, a_row)
        g[start:end] = s_chunk * (1.0 - s_chunk)
    diag_correction = g.sum(axis=1) - numpy.diag(g)
    numpy.fill_diagonal(g, -diag_correction)
    return g


def _pl_pairwise_ratio(
    a_col_chunk: numpy.typing.NDArray[numpy.floating],
    a_row: numpy.typing.NDArray[numpy.floating],
) -> numpy.typing.NDArray[numpy.floating]:
    """Plackett-Luce pairwise win probabilities a_row / (a_col_chunk + a_row)."""
    pair_sum_chunk = a_col_chunk + a_row
    ratio = a_row / pair_sum_chunk
    return ratio


def _row_starts(
    row_sizes: numpy.typing.NDArray[numpy.intp],
) -> numpy.typing.NDArray[numpy.intp]:
    """Exclusive prefix offsets where each row begins in the flat layout."""
    starts = numpy.zeros(row_sizes.size, dtype=numpy.intp)
    if row_sizes.size > 1:
        numpy.cumsum(row_sizes[:-1], out=starts[1:])
    return starts


def _segment_prefix(
    values_flat: numpy.typing.NDArray[numpy.floating],
    row_sizes: numpy.typing.NDArray[numpy.intp],
    row_starts: numpy.typing.NDArray[numpy.intp],
) -> numpy.typing.NDArray[numpy.floating]:
    """Within-row cumulative sum of a row-segmented flat array."""
    flat_cumsum = numpy.cumsum(values_flat)
    prev_cumsum = numpy.zeros(row_sizes.size, dtype=float)
    prev_cumsum[1:] = flat_cumsum[row_starts[1:] - 1]
    prev_cumsum_flat = numpy.repeat(prev_cumsum, row_sizes)
    prefix_flat = flat_cumsum - prev_cumsum_flat
    return prefix_flat


def _row_of_flat(
    row_sizes: numpy.typing.NDArray[numpy.intp],
) -> numpy.typing.NDArray[numpy.intp]:
    """Row index for each element of the row-segmented flat layout."""
    n_orderings = row_sizes.size
    indices = numpy.repeat(
        numpy.arange(n_orderings, dtype=numpy.intp), row_sizes
    )
    return indices
