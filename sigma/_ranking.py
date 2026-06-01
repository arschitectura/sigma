"""Ranking-data primitives used by RankingTree.

`compute_mean_rank_vector` computes per-item weighted mean ranks for the
leaf-level node statistics inside RankingTree. The response matrix uses
the ranks-in-cell layout: rows are observations, columns are items, and
each cell carries the rank position of that item for that observation.
Unranked items are NaN.
"""

import numpy
import numpy.typing


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
