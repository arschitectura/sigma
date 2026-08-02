"""Homologation tests for the Wald and Gaussian-multiplier ranking CIs.

Covers, in order:

* Analytic invariants of the expected-rank Jacobian and the PL Fisher
  information.
* Finite-difference cross-checks of the analytic Jacobian and Fisher
  information against numerical derivatives of `pl_expected_rank` and
  the PL log-likelihood.
* Per-row score helper consistency (aggregate score is approximately
  zero at the MLE).
* Bootstrap-asymptotic agreement tests (env-gated; slow): Wald per-item
  SE agrees with sigma's Bayesian-bootstrap SE within Monte-Carlo
  noise, and B=2000 GMB SE collapses to Wald SE.
* Regression and dispatch checks: all four `ci_method` values produce
  finite, monotone, point-bracketing CIs on a Sushi subset, and the
  CI dispatch never affects tree structure under a fixed
  `random_state`.
"""

import math
import os
import unittest

import _helpers
import numpy
import numpy.testing

import sigma
import sigma._ranking
import sigma._types


def _quiet_fisher_info(*args, **kwargs):
    """Call _compute_pl_fisher_info while suppressing spurious BLAS FPE flags.

    The chunked matmul inside `_compute_pl_fisher_info` triggers benign
    divide/overflow/invalid FPE flags on Apple Silicon's Accelerate BLAS.
    Production callers (`_wald_per_item_ci`, `_gaussian_multiplier_per_item_ci`)
    already wrap; tests do not, so suppress here.
    """
    with numpy.errstate(divide="ignore", over="ignore", invalid="ignore"):
        return sigma._ranking._compute_pl_fisher_info(*args, **kwargs)


def _draw_pl_order(alpha, rng):
    """Sequential Plackett-Luce draw over all items in `alpha`."""
    n_items = alpha.size
    remaining_mask = numpy.ones(n_items, dtype=bool)
    order = numpy.empty(n_items, dtype=numpy.intp)
    for step in range(n_items):
        idx = numpy.flatnonzero(remaining_mask)
        sub = alpha[idx]
        prob = sub / sub.sum()
        choice = rng.choice(idx.size, p=prob)
        winner = idx[choice]
        order[step] = winner
        remaining_mask[winner] = False
    return order


def _simulate_full_rankings(alpha, n_rows, rng):
    """Generate n_rows full Plackett-Luce rankings."""
    n_items = alpha.size
    rankings = numpy.empty((n_rows, n_items), dtype=float)
    for i in range(n_rows):
        order = _draw_pl_order(alpha, rng)
        for position, item in enumerate(order):
            rankings[i, int(item)] = float(position + 1)
    return rankings


class TestJacobianInvariants(unittest.TestCase):
    """Analytic invariants of `_compute_pl_expected_rank_jacobian`."""

    __slots__ = ()

    def test_rows_sum_to_zero(self):
        """Each row of g should sum to zero (Σ_k E[R_k] is constant)."""
        rng = numpy.random.default_rng(0)
        for n_items in (2, 5, 20, 257):
            with self.subTest(n_items=n_items):
                alpha = _helpers._random_alpha(n_items, rng)
                g = sigma._ranking._compute_pl_expected_rank_jacobian(alpha)
                row_sums = g.sum(axis=1)
                numpy.testing.assert_allclose(
                    row_sums,
                    numpy.zeros(n_items),
                    atol=1e-12,
                )

    def test_uniform_alpha_is_symmetric(self):
        """g[k, j] = 1/4 for j != k and g[k, k] = -(K-1)/4 at α = ones."""
        for n_items in (2, 5, 20):
            with self.subTest(n_items=n_items):
                alpha = numpy.ones(n_items, dtype=float)
                g = sigma._ranking._compute_pl_expected_rank_jacobian(alpha)
                off_diag = g[~numpy.eye(n_items, dtype=bool)]
                numpy.testing.assert_allclose(
                    off_diag,
                    numpy.full(n_items * (n_items - 1), 0.25),
                )
                diag = numpy.diag(g)
                expected_diag = numpy.full(n_items, -(n_items - 1) / 4.0)
                numpy.testing.assert_allclose(diag, expected_diag)


class TestFisherInfoInvariants(unittest.TestCase):
    """Analytic invariants of `_compute_pl_fisher_info`."""

    __slots__ = ()

    def test_kernel_contains_constant_vector(self):
        """H · ones is zero (PL is invariant under a constant log-alpha shift)."""
        rng = numpy.random.default_rng(1)
        alpha = _helpers._random_alpha(6, rng)
        rankings = _simulate_full_rankings(alpha, 40, rng)
        weights = numpy.ones(rankings.shape[0])
        cache = sigma._ranking._extract_orderings_cache(rankings, weights)
        h = _quiet_fisher_info(cache, alpha, cache.ordering_weights, alpha.size)
        product = h @ numpy.ones(alpha.size)
        numpy.testing.assert_allclose(
            product, numpy.zeros(alpha.size), atol=1e-10
        )

    def test_symmetric_and_psd(self):
        """H is symmetric and positive semi-definite."""
        rng = numpy.random.default_rng(2)
        alpha = _helpers._random_alpha(8, rng)
        rankings = _simulate_full_rankings(alpha, 60, rng)
        weights = numpy.ones(rankings.shape[0])
        cache = sigma._ranking._extract_orderings_cache(rankings, weights)
        h = _quiet_fisher_info(cache, alpha, cache.ordering_weights, alpha.size)
        numpy.testing.assert_allclose(h, h.T, rtol=1e-12, atol=1e-12)
        eigenvalues = numpy.linalg.eigvalsh(h)
        self.assertGreaterEqual(float(eigenvalues.min()), -1e-10)


class TestWaldSeAtUniformAlpha(unittest.TestCase):
    """SE is uniform across items when α is uniform on a symmetric dataset."""

    __slots__ = ()

    def test_uniform_se_at_uniform_alpha(self):
        """Wald SE(E[R_k]) is identical across k when α=ones on a balanced dataset.

        A "balanced" dataset is one that includes every permutation of the K
        items the same number of times. Such a dataset is permutation-symmetric
        in k, so the observed PL Fisher information at α=ones is symmetric in
        k and the delta-method SE on E[R_k] is identical for every k.
        """
        import itertools

        n_items = 4
        alpha = numpy.ones(n_items, dtype=float)
        permutations = list(itertools.permutations(range(n_items)))
        rankings = numpy.empty((len(permutations), n_items), dtype=float)
        for index, perm in enumerate(permutations):
            for position, item in enumerate(perm):
                rankings[index, item] = float(position + 1)
        weights = numpy.ones(rankings.shape[0])
        cache = sigma._ranking._extract_orderings_cache(rankings, weights)
        h = _quiet_fisher_info(cache, alpha, cache.ordering_weights, n_items)
        ridge = 1e-9 * float(numpy.trace(h)) / n_items
        h_reg = h + ridge * numpy.eye(n_items)
        sigma_theta = numpy.linalg.inv(h_reg)
        g = sigma._ranking._compute_pl_expected_rank_jacobian(alpha)
        var = numpy.einsum("ki,ij,kj->k", g, sigma_theta, g)
        se = numpy.sqrt(numpy.maximum(var, 0.0))
        numpy.testing.assert_allclose(
            se, se[0] * numpy.ones(n_items), rtol=1e-7
        )


class TestFiniteDifferenceCrossChecks(unittest.TestCase):
    """Analytic Jacobian and Fisher info match central differences."""

    __slots__ = ()

    def test_jacobian_matches_finite_difference(self):
        """g[k, j] equals central difference of E[R_k] in log-α direction j."""
        rng = numpy.random.default_rng(4)
        n_items = 6
        alpha = _helpers._random_alpha(n_items, rng)
        g_analytic = sigma._ranking._compute_pl_expected_rank_jacobian(alpha)
        epsilon = 1e-5
        log_alpha = numpy.log(alpha)
        for j in range(n_items):
            shift = numpy.zeros(n_items)
            shift[j] = epsilon
            er_plus = sigma._ranking.pl_expected_rank(
                numpy.exp(log_alpha + shift)
            )
            er_minus = sigma._ranking.pl_expected_rank(
                numpy.exp(log_alpha - shift)
            )
            g_numeric_col = (er_plus - er_minus) / (2.0 * epsilon)
            numpy.testing.assert_allclose(
                g_analytic[:, j], g_numeric_col, rtol=1e-7, atol=1e-9
            )

    def test_fisher_info_matches_numerical_hessian(self):
        """H matches the central-difference Hessian of the PL log-likelihood.

        Loose rtol because numerical Hessians of a sparse partial ranking
        accumulate ε-noise at the second-difference scale; ε is chosen as
        1e-3 to balance truncation vs cancellation error.
        """
        rng = numpy.random.default_rng(5)
        n_items = 5
        alpha = _helpers._random_alpha(n_items, rng)
        rankings = _simulate_full_rankings(alpha, 40, rng)
        weights = numpy.ones(rankings.shape[0])
        cache = sigma._ranking._extract_orderings_cache(rankings, weights)
        h_analytic = _quiet_fisher_info(
            cache, alpha, cache.ordering_weights, n_items
        )
        epsilon = 1e-3

        def log_likelihood(alpha_local):
            ll = 0.0
            for i in range(rankings.shape[0]):
                row = rankings[i]
                ranked_mask = ~numpy.isnan(row)
                order = numpy.argsort(numpy.where(ranked_mask, row, numpy.inf))
                order = order[: int(ranked_mask.sum())]
                at_risk_sum = float(alpha_local.sum())
                for position in order:
                    ll += math.log(alpha_local[int(position)])
                    ll -= math.log(at_risk_sum)
                    at_risk_sum -= alpha_local[int(position)]
            return ll

        log_alpha = numpy.log(alpha)
        for j in range(n_items):
            for k in range(j, n_items):
                shift_jk = numpy.zeros(n_items)
                shift_jk[j] += epsilon
                shift_jk[k] += epsilon
                shift_j = numpy.zeros(n_items)
                shift_j[j] += epsilon
                shift_k = numpy.zeros(n_items)
                shift_k[k] += epsilon
                ll_pp = log_likelihood(numpy.exp(log_alpha + shift_jk))
                ll_pm = log_likelihood(numpy.exp(log_alpha + shift_j - shift_k))
                ll_mp = log_likelihood(numpy.exp(log_alpha - shift_j + shift_k))
                ll_mm = log_likelihood(numpy.exp(log_alpha - shift_jk))
                d2_ll = (ll_pp - ll_pm - ll_mp + ll_mm) / (
                    4.0 * epsilon * epsilon
                )
                # Observed Fisher info = -d2(log L) / d theta_j d theta_k.
                h_numeric = -d2_ll
                numpy.testing.assert_allclose(
                    h_analytic[j, k],
                    h_numeric,
                    rtol=1e-3,
                    atol=5e-4,
                )


class TestScorePerRowSumsToAggregateScore(unittest.TestCase):
    """Aggregate per-row score at the fitted MLE is near zero."""

    __slots__ = ()

    def test_aggregate_score_near_zero_at_mle(self):
        """U.sum(axis=0) is small at α̂ (sums to zero at the unregularised MLE)."""
        rng = numpy.random.default_rng(6)
        alpha_true = _helpers._random_alpha(8, rng)
        rankings = _simulate_full_rankings(alpha_true, 200, rng)
        weights = numpy.ones(rankings.shape[0])
        cache = sigma._ranking._extract_orderings_cache(rankings, weights)
        alpha = sigma._ranking._compute_pl_mle_from_cache(
            cache,
            cache.ordering_weights,
            8,
            npseudo=0.5,
            tolerance=1e-10,
            max_iter=500,
        )
        score = sigma._ranking._compute_pl_score_per_row(cache, alpha, 8)
        aggregate = score.sum(axis=0)
        # Regularised MM optimum shifts the score by the pseudo-comparison
        # contribution; assert the residual is small in absolute terms.
        self.assertLess(float(numpy.abs(aggregate).max()), 1.0)


class TestCiMethodDispatch(unittest.TestCase):
    """Dispatch accepts the four CI methods and rejects unknown names."""

    __slots__ = ()

    def test_accepts_all_four_methods(self):
        """All four `ci_method` values pass construction validation."""
        for method in (
            "bayesian_bootstrap",
            "bca",
            "wald",
            "gaussian_multiplier",
        ):
            with self.subTest(method=method):
                sigma._types._validate_literal_param(
                    method,
                    sigma._types.CiMethodRankingTree,
                    "ci_method",
                )

    def test_rejects_unknown_method(self):
        """Construction validation rejects a name outside the menu."""
        with self.assertRaises(ValueError):
            sigma._types._validate_literal_param(
                "frequentist_dolphins",
                sigma._types.CiMethodRankingTree,
                "ci_method",
            )


class TestAllFourMethodsOnSushiSubset(unittest.TestCase):
    """Each ci_method produces finite, monotone, bracketing CIs on Sushi."""

    __slots__ = ()

    @classmethod
    def setUpClass(cls):
        """Load 800 rows of SUSHI3A once for the whole class."""
        cls.X, cls.rankings = _helpers._load_sushi3a_subset(800)

    def _check_metrics(self, tree):
        """Each metric satisfies low ≤ value ≤ high and is finite per item."""
        for metric in tree.metrics_:
            self.assertTrue(metric.has_ci)
        for node in tree.nodes_:
            for index, value in enumerate(node.values):
                ci_low = node.ci_low[index]
                ci_high = node.ci_high[index]
                self.assertTrue(numpy.isfinite(value))
                self.assertTrue(numpy.isfinite(ci_low))
                self.assertTrue(numpy.isfinite(ci_high))
                self.assertLessEqual(ci_low, value + 1e-9)
                self.assertGreaterEqual(ci_high, value - 1e-9)

    def test_bayesian_bootstrap(self):
        """Default Bayesian bootstrap produces valid CIs."""
        tree = sigma.RankingTree(
            random_state=0,
            max_depth=1,
            ci_method="bayesian_bootstrap",
            ci_replicates=20,
        )
        tree.fit(self.X, self.rankings)
        self._check_metrics(tree)

    def test_bca(self):
        """BCa CI dispatch still produces valid CIs after the menu expansion."""
        tree = sigma.RankingTree(
            random_state=0, max_depth=1, ci_method="bca", ci_replicates=20
        )
        tree.fit(self.X, self.rankings)
        self._check_metrics(tree)

    def test_wald(self):
        """Wald CI dispatch produces valid CIs."""
        tree = sigma.RankingTree(
            random_state=0, max_depth=1, ci_method="wald", ci_replicates=20
        )
        tree.fit(self.X, self.rankings)
        self._check_metrics(tree)

    def test_gaussian_multiplier(self):
        """Gaussian-multiplier CI dispatch produces valid CIs."""
        tree = sigma.RankingTree(
            random_state=0,
            max_depth=1,
            ci_method="gaussian_multiplier",
            ci_replicates=20,
        )
        tree.fit(self.X, self.rankings)
        self._check_metrics(tree)


class TestCiMethodDoesNotAffectTreeShape(unittest.TestCase):
    """Switching ci_method does not change which splits are chosen."""

    __slots__ = ()

    def test_tree_shape_constant_across_methods(self):
        """At fixed random_state the split features and partitions match."""
        X, rankings = _helpers._load_sushi3a_subset(500)

        def shape_signature(tree):
            return [
                (
                    node.node_id,
                    node.depth,
                    type(node.extension).__name__,
                    getattr(node.extension, "feature_index", None),
                    getattr(node.extension, "thresholds", None),
                    tuple(
                        tuple(sorted(group))
                        for group in getattr(
                            node.extension, "category_groups", ()
                        )
                    )
                    if hasattr(node.extension, "category_groups")
                    else None,
                )
                for node in tree.nodes_
            ]

        references = None
        for method in (
            "bayesian_bootstrap",
            "bca",
            "wald",
            "gaussian_multiplier",
        ):
            with self.subTest(method=method):
                tree = sigma.RankingTree(
                    random_state=123,
                    max_depth=2,
                    ci_method=method,
                    ci_replicates=5,
                )
                tree.fit(X, rankings)
                signature = shape_signature(tree)
                if references is None:
                    references = signature
                else:
                    self.assertEqual(signature, references)


@unittest.skipUnless(
    os.environ.get("SIGMA_RUN_SLOW_RANKING_CI"),
    "slow; opt in by setting SIGMA_RUN_SLOW_RANKING_CI=1",
)
class TestBootstrapAsymptoticAgreement(unittest.TestCase):
    """Slow tests: Wald SE matches Bayesian bootstrap SE; GMB collapses to Wald."""

    __slots__ = ()

    @classmethod
    def setUpClass(cls):
        """Pre-fit a tree and extract its single-leaf CI inputs."""
        rng = numpy.random.default_rng(7)
        n_items = 20
        n_rows = 2000
        alpha_true = _helpers._random_alpha(n_items, rng)
        rankings = _simulate_full_rankings(alpha_true, n_rows, rng)
        weights = numpy.ones(n_rows)
        cls.cache = sigma._ranking._extract_orderings_cache(rankings, weights)
        cls.alpha = sigma._ranking._compute_pl_mle_from_cache(
            cls.cache,
            cls.cache.ordering_weights,
            n_items,
            npseudo=0.5,
            tolerance=1e-6,
            max_iter=100,
        )
        cls.point_rank = sigma._ranking.pl_expected_rank(cls.alpha)
        cls.n_items = n_items
        cls.weights = weights

    def _wald_se(self):
        """Compute Wald per-item SE from the cached fit."""
        h = _quiet_fisher_info(
            self.cache,
            self.alpha,
            self.cache.ordering_weights,
            self.n_items,
        )
        ridge = 1e-9 * float(numpy.trace(h)) / self.n_items
        h_reg = h + ridge * numpy.eye(self.n_items)
        sigma_theta = numpy.linalg.inv(h_reg)
        g = sigma._ranking._compute_pl_expected_rank_jacobian(self.alpha)
        var = numpy.einsum("ki,ij,kj->k", g, sigma_theta, g)
        se = numpy.sqrt(numpy.maximum(var, 0.0))
        return se

    def test_wald_se_matches_bayesian_bootstrap_se(self):
        """Per-item SE from Wald matches the std across B=500 Dirichlet refits."""
        wald_se = self._wald_se()
        rng = numpy.random.default_rng(11)
        b = 500
        weights_total = float(self.cache.ordering_weights.sum())
        dirichlet_alpha = self.cache.ordering_weights.astype(float, copy=False)
        replicate_ranks = numpy.empty((b, self.n_items), dtype=float)
        for replicate in range(b):
            bootstrap_weights = rng.dirichlet(dirichlet_alpha) * weights_total
            replicate_alpha = sigma._ranking._compute_pl_mle_from_cache(
                self.cache,
                bootstrap_weights[self.cache.row_indices_in_y],
                self.n_items,
                npseudo=0.5,
                tolerance=1e-6,
                max_iter=100,
            )
            replicate_ranks[replicate] = sigma._ranking.pl_expected_rank(
                replicate_alpha
            )
        bootstrap_se = replicate_ranks.std(axis=0, ddof=1)
        numpy.testing.assert_allclose(
            wald_se, bootstrap_se, rtol=0.15, atol=1e-3
        )

    def test_gmb_collapses_to_wald_at_large_b(self):
        """GMB per-item SE at B=2000 matches Wald SE within Monte-Carlo noise."""
        wald_se = self._wald_se()
        h = _quiet_fisher_info(
            self.cache,
            self.alpha,
            self.cache.ordering_weights,
            self.n_items,
        )
        ridge = 1e-9 * float(numpy.trace(h)) / self.n_items
        h_reg = h + ridge * numpy.eye(self.n_items)
        sigma_theta = numpy.linalg.inv(h_reg)
        g = sigma._ranking._compute_pl_expected_rank_jacobian(self.alpha)
        score = sigma._ranking._compute_pl_score_per_row(
            self.cache, self.alpha, self.n_items
        )
        weighted_score = self.cache.ordering_weights[:, None] * score
        rng = numpy.random.default_rng(13)
        b = 2000
        n_orderings = self.cache.row_sizes.size
        omega = rng.normal(size=(b, n_orderings))
        score_sums = omega @ weighted_score
        delta_theta = score_sums @ sigma_theta.T
        delta_er = delta_theta @ g.T
        gmb_se = delta_er.std(axis=0, ddof=1)
        numpy.testing.assert_allclose(gmb_se, wald_se, rtol=0.10, atol=1e-3)


if __name__ == "__main__":
    unittest.main()
