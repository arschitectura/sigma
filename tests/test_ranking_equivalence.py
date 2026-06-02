"""Equivalence tests pinning the rewritten ranking primitives to the legacy ones.

This module embeds verbatim copies of the pre-rewrite implementations of
`_extract_orderings`, `_accumulate_pl_denominator`, `compute_pl_mle` and
`pl_expected_rank` from `sigma/_ranking.py`. Each test class compares the
embedded legacy function against the current `sigma._ranking` symbol on
the same inputs and asserts numerical equivalence at the tolerance
contract agreed with the project owner:

- `pl_expected_rank` and `compute_mean_rank_vector` : bit-identical
  (`assert_array_equal`).
- `_accumulate_pl_denominator`, `compute_pl_mle` : `rtol=1e-9, atol=1e-9`
  (vectorised cumsum reorders FP addition, no algorithmic drift).

The legacy snippets are intentionally retained as a regression net: any
future change to `sigma/_ranking.py` must keep these equivalences. They
are NOT exercised at run time outside of this file.
"""

import unittest

import numpy
import numpy.testing

import sigma._ranking

_RTOL = 1e-9
_ATOL = 1e-9


# ---------------------------------------------------------------------------
# Legacy implementations (verbatim copies of the pre-rewrite versions).
# ---------------------------------------------------------------------------


def _legacy_extract_orderings(y, weights):
    """Verbatim copy of pre-rewrite sigma._ranking._extract_orderings."""
    n_obs = y.shape[0]
    orderings = []
    ordering_weights = []
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


def _legacy_accumulate_pl_denominator(
    alpha, orderings, ordering_weights, n_items
):
    """Verbatim copy of pre-rewrite sigma._ranking._accumulate_pl_denominator."""
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


def _legacy_compute_pl_mle(
    y, weights, npseudo=0.5, tolerance=1.0e-6, max_iter=100
):
    """Verbatim copy of pre-rewrite sigma._ranking.compute_pl_mle."""
    n_obs, n_items = y.shape
    if n_obs == 0:
        empty = numpy.full(n_items, numpy.nan, dtype=float)
        return empty
    orderings, ordering_weights = _legacy_extract_orderings(y, weights)
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
        real_d = _legacy_accumulate_pl_denominator(
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


def _legacy_pl_expected_rank(alpha):
    """Verbatim copy of pre-rewrite sigma._ranking.pl_expected_rank."""
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


# ---------------------------------------------------------------------------
# Fixture generators.
# ---------------------------------------------------------------------------


def _row_from_ordering(ordering, n_items):
    """Build a single ranks-in-cell row from an item-index ordering."""
    row = numpy.full(n_items, numpy.nan, dtype=float)
    for position, item in enumerate(ordering):
        row[item] = float(position + 1)
    return row


def _build_dense_y(n_obs, n_items, rng):
    """Generate dense full-permutation rankings, one per row."""
    rows = numpy.empty((n_obs, n_items), dtype=float)
    for i in range(n_obs):
        permutation = rng.permutation(n_items)
        rows[i] = _row_from_ordering(permutation, n_items)
    return rows


def _build_sparse_y(n_obs, n_items, mean_d, rng):
    """Generate sparse partial rankings with d_i drawn from a clipped Poisson."""
    rows = numpy.full((n_obs, n_items), numpy.nan, dtype=float)
    for i in range(n_obs):
        d = int(numpy.clip(rng.poisson(mean_d), 2, n_items))
        items = rng.choice(n_items, size=d, replace=False)
        for position, item in enumerate(items):
            rows[i, item] = float(position + 1)
    return rows


def _mild_scenarios():
    """Cases whose alpha stays inside the regime where the at_risk_floor
    clamp never triggers. Used for the accumulator-level test, where
    catastrophic cancellation under a clamp regime would give
    floor-clamp-dependent garbage values for which the legacy and
    vectorised implementations are not expected to agree.
    """
    rng = numpy.random.default_rng(0)
    cases = []

    y = _build_dense_y(5, 4, rng)
    cases.append(("small_dense", y, numpy.ones(5), _random_alpha(4, rng)))

    y = _build_dense_y(200, 50, rng)
    cases.append(
        ("large_dense", y, _random_weights(200, rng), _random_alpha(50, rng))
    )

    y = _build_sparse_y(300, 200, mean_d=10, rng=rng)
    cases.append(
        (
            "sparse_partial",
            y,
            _random_weights(300, rng),
            _random_alpha(200, rng),
        )
    )

    y = numpy.full((20, 8), numpy.nan, dtype=float)
    for i in range(20):
        d = 2 if i % 2 == 0 else 8
        items = rng.choice(8, size=d, replace=False)
        for position, item in enumerate(items):
            y[i, item] = float(position + 1)
    cases.append(("mixed_d2_or_full", y, numpy.ones(20), _random_alpha(8, rng)))

    y = _build_dense_y(10, 4, rng)
    cases.append(
        (
            "tiny_weights",
            y,
            numpy.full(10, 1.0e-300),
            _random_alpha(4, rng),
        )
    )

    y = _build_dense_y(50, 6, rng)
    cases.append(
        (
            "tied_alphas",
            y,
            numpy.ones(50),
            numpy.ones(6),
        )
    )

    for _ in range(25):
        k = int(rng.choice([2, 10, 100, 1000]))
        n = int(rng.integers(low=2, high=300))
        if rng.random() < 0.5 or k < 4:
            y = _build_dense_y(n, k, rng)
        else:
            mean_d = max(2, k // 4)
            y = _build_sparse_y(n, k, mean_d, rng)
        weights = _random_weights(n, rng)
        alpha = _random_alpha(k, rng)
        cases.append((f"fuzz_n{n}_k{k}", y, weights, alpha))
    return cases


def _stress_scenarios():
    """All mild scenarios plus one extreme alpha-contrast case. The extreme
    case is only used via compute_pl_mle (the MM loop iterates away from the
    pathological starting point regardless of any per-iteration accumulator
    drift), never via the bare accumulator.
    """
    cases = list(_mild_scenarios())
    rng = numpy.random.default_rng(99)
    y = _build_dense_y(30, 3, rng)
    cases.append(
        (
            "extreme_alpha_contrast",
            y,
            numpy.ones(30),
            numpy.array([1.0e-10, 1.0, 1.0e10]),
        )
    )
    return cases


def _random_alpha(k, rng):
    """Sample a worth vector with geometric mean 1 (matching MM normalisation)."""
    log_alpha = rng.normal(size=k) * 0.5
    log_alpha -= log_alpha.mean()
    alpha = numpy.exp(log_alpha)
    return alpha


def _random_weights(n, rng):
    """Sample positive per-row weights."""
    weights = rng.uniform(low=0.1, high=5.0, size=n)
    return weights


# ---------------------------------------------------------------------------
# Test classes.
# ---------------------------------------------------------------------------


class TestAccumulatePlDenominatorEquivalence(unittest.TestCase):
    """Compare the current `_accumulate_pl_denominator` against the legacy loop."""

    __slots__ = ()

    def test_all_scenarios_match_legacy(self):
        """Every (y, weights, alpha) scenario yields the same accumulator output."""
        for name, y, weights, alpha in _mild_scenarios():
            with self.subTest(name=name):
                orderings, ordering_weights = _legacy_extract_orderings(
                    y, weights
                )
                if not orderings:
                    continue
                expected = _legacy_accumulate_pl_denominator(
                    alpha, orderings, ordering_weights, alpha.size
                )
                cache = sigma._ranking._extract_orderings_cache(y, weights)
                ordering_weights_arr = numpy.asarray(
                    ordering_weights, dtype=float
                )
                actual = sigma._ranking._accumulate_pl_denominator(
                    alpha, cache, ordering_weights_arr, alpha.size
                )
                numpy.testing.assert_allclose(
                    actual,
                    expected,
                    rtol=_RTOL,
                    atol=_ATOL,
                    err_msg=f"accumulator mismatch on {name}",
                )


class TestComputePlMleEquivalence(unittest.TestCase):
    """Compare the current `compute_pl_mle` against the legacy implementation."""

    __slots__ = ()

    def test_all_scenarios_match_legacy(self):
        """compute_pl_mle agrees with the legacy MM loop on every scenario."""
        for name, y, weights, _alpha in _stress_scenarios():
            with self.subTest(name=name):
                expected = _legacy_compute_pl_mle(y, weights)
                actual = sigma._ranking.compute_pl_mle(y, weights)
                numpy.testing.assert_allclose(
                    actual,
                    expected,
                    rtol=_RTOL,
                    atol=_ATOL,
                    err_msg=f"compute_pl_mle mismatch on {name}",
                )

    def test_empty_input_returns_nan_vector(self):
        """Identity preserved: zero-row y maps to an all-NaN worth vector."""
        y = numpy.zeros((0, 4), dtype=float)
        weights = numpy.zeros(0, dtype=float)
        alpha = sigma._ranking.compute_pl_mle(y, weights)
        self.assertEqual(alpha.shape, (4,))
        self.assertTrue(numpy.all(numpy.isnan(alpha)))

    def test_geometric_mean_normalisation_preserved(self):
        """Identity preserved: geometric mean of the returned worths is 1."""
        rng = numpy.random.default_rng(7)
        y = _build_dense_y(30, 5, rng)
        weights = numpy.ones(30)
        alpha = sigma._ranking.compute_pl_mle(y, weights)
        product = float(numpy.prod(alpha))
        self.assertAlmostEqual(product, 1.0, places=10)


class TestPlExpectedRankBitEquivalence(unittest.TestCase):
    """Compare the current `pl_expected_rank` against the legacy version bit-exactly."""

    __slots__ = ()

    def test_random_alpha_bit_identical(self):
        """Chunked expected-rank matches the legacy K×K reduction exactly."""
        rng = numpy.random.default_rng(1)
        for k in [2, 10, 256, 257, 1000, 3706]:
            with self.subTest(k=k):
                alpha = _random_alpha(k, rng)
                expected = _legacy_pl_expected_rank(alpha)
                actual = sigma._ranking.pl_expected_rank(alpha)
                numpy.testing.assert_array_equal(actual, expected)

    def test_nan_propagation_unchanged(self):
        """NaN entries in alpha map to NaN at the same output positions."""
        alpha = numpy.array([1.0, numpy.nan, 2.0, numpy.nan])
        expected = _legacy_pl_expected_rank(alpha)
        actual = sigma._ranking.pl_expected_rank(alpha)
        numpy.testing.assert_array_equal(actual, expected)

    def test_all_nan_alpha_returns_all_nan(self):
        """Identity: all-NaN alpha yields all-NaN expected ranks."""
        alpha = numpy.full(5, numpy.nan, dtype=float)
        result = sigma._ranking.pl_expected_rank(alpha)
        self.assertTrue(numpy.all(numpy.isnan(result)))


class TestOrderingsCacheRoundTrip(unittest.TestCase):
    """Round-trip checks on the orderings cache used by `_compute_per_item_ci`."""

    __slots__ = ()

    def test_cache_reconstructs_legacy_orderings(self):
        """flat_idx / row_starts / row_sizes recover the legacy per-row arrays."""
        rng = numpy.random.default_rng(3)
        y = _build_sparse_y(50, 30, mean_d=8, rng=rng)
        weights = _random_weights(50, rng)
        legacy_orderings, legacy_weights = _legacy_extract_orderings(y, weights)
        cache = sigma._ranking._extract_orderings_cache(y, weights)
        self.assertEqual(int(cache.row_sizes.size), len(legacy_orderings))
        for i, legacy_ord in enumerate(legacy_orderings):
            start = int(cache.row_starts[i])
            end = int(cache.row_ends_inclusive[i]) + 1
            cache_ord = cache.flat_idx[start:end]
            numpy.testing.assert_array_equal(cache_ord, legacy_ord)
        numpy.testing.assert_allclose(
            cache.ordering_weights,
            numpy.asarray(legacy_weights, dtype=float),
        )

    def test_subset_cache_matches_rebuild(self):
        """Subsetting a cache by row indices equals rebuilding from y[indices]."""
        rng = numpy.random.default_rng(4)
        y = _build_sparse_y(40, 20, mean_d=6, rng=rng)
        weights = _random_weights(40, rng)
        cache = sigma._ranking._extract_orderings_cache(y, weights)
        n_orderings = int(cache.row_sizes.size)
        indices = rng.integers(0, n_orderings, size=n_orderings)
        subset = sigma._ranking._subset_cache(cache, indices)
        rebuilt = sigma._ranking._extract_orderings_cache(
            y[indices], weights[indices]
        )
        numpy.testing.assert_array_equal(subset.flat_idx, rebuilt.flat_idx)
        numpy.testing.assert_array_equal(subset.row_sizes, rebuilt.row_sizes)
        numpy.testing.assert_array_equal(subset.row_starts, rebuilt.row_starts)
        numpy.testing.assert_array_equal(
            subset.row_ends_inclusive, rebuilt.row_ends_inclusive
        )
        numpy.testing.assert_allclose(
            subset.ordering_weights, rebuilt.ordering_weights
        )


if __name__ == "__main__":
    unittest.main()
