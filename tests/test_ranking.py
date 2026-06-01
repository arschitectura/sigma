"""Unit tests for the ranking-data helpers and RankingTree estimator."""

import inspect
import typing
import unittest

import numpy
import numpy.testing

import sigma
import sigma._partition
import sigma._ranking
import sigma._survival
import sigma._types


def _ranks_in_cell(ranking_per_item, n_items):
    """Helper: build a ranks-in-cell row from a per-item rank dict."""
    row = numpy.full(n_items, numpy.nan, dtype=float)
    for item, rank in ranking_per_item.items():
        row[item] = float(rank)
    return row


class TestComputeMeanRankVector(unittest.TestCase):
    """Tests for the per-item weighted-mean-rank leaf summary."""

    __slots__ = ()

    def test_single_ranking_recovers_positions(self):
        """One complete ranking returns each item's stated rank value."""
        y = numpy.array([[2.0, 3.0, 1.0]])
        weights = numpy.array([1.0])
        means = sigma._ranking.compute_mean_rank_vector(y, weights)
        numpy.testing.assert_allclose(means, [2.0, 3.0, 1.0])

    def test_two_complete_rankings_average(self):
        """Mean rank is the arithmetic average over complete rankings."""
        y = numpy.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        weights = numpy.array([1.0, 1.0])
        means = sigma._ranking.compute_mean_rank_vector(y, weights)
        numpy.testing.assert_allclose(means, [2.0, 2.0, 2.0])

    def test_weights_are_honored(self):
        """A rank-1 outcome dominates when its weight is much larger."""
        y = numpy.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]])
        weights = numpy.array([9.0, 1.0])
        means = sigma._ranking.compute_mean_rank_vector(y, weights)
        numpy.testing.assert_allclose(means, [1.2, 2.0, 2.8])

    def test_unranked_items_are_dropped(self):
        """NaN cells are dropped per item; items with no rank report NaN."""
        y = numpy.array([[1.0, 2.0, numpy.nan, numpy.nan]])
        weights = numpy.array([1.0])
        means = sigma._ranking.compute_mean_rank_vector(y, weights)
        self.assertAlmostEqual(means[0], 1.0)
        self.assertAlmostEqual(means[1], 2.0)
        self.assertTrue(numpy.isnan(means[2]))
        self.assertTrue(numpy.isnan(means[3]))


class TestRankingTreeFit(unittest.TestCase):
    """End-to-end smoke tests for the RankingTree estimator."""

    __slots__ = ()

    def test_splits_on_covariate_that_flips_preferences(self):
        """A covariate that flips PL worths drives a high-significance split."""
        rng = numpy.random.default_rng(0)
        n_samples = 400
        n_items = 4
        X = rng.normal(size=(n_samples, 1))
        ascending = numpy.array([1.0, 2.0, 3.0, 4.0])
        descending = numpy.array([4.0, 3.0, 2.0, 1.0])
        y = numpy.empty((n_samples, n_items), dtype=float)
        for i in range(n_samples):
            base = ascending if X[i, 0] > 0 else descending
            if rng.random() < 0.2:
                shuffled = rng.permutation(n_items).astype(float) + 1.0
                y[i] = shuffled
            else:
                y[i] = base
        tree = sigma.RankingTree(
            item_names=["A", "B", "C", "D"],
            random_state=123,
        )
        tree.fit(X, y)
        self.assertGreaterEqual(len(tree.leaves_), 2)
        leaf_predictions = sorted(
            {int(leaf.prediction) for leaf in tree.leaves_}
        )
        self.assertIn(0, leaf_predictions)
        self.assertIn(3, leaf_predictions)
        first_split = tree.content_.extension
        self.assertIsInstance(first_split, sigma._partition.Partition)
        partition = typing.cast(sigma._partition.Partition, first_split)
        self.assertLess(partition.p_value, 1e-3)

    def test_predict_returns_favorite_item_label(self):
        """predict returns the favorite-item label per sample."""
        rng = numpy.random.default_rng(1)
        n_samples = 100
        n_items = 3
        X = rng.normal(size=(n_samples, 1))
        y = numpy.empty((n_samples, n_items), dtype=float)
        for i in range(n_samples):
            if X[i, 0] > 0:
                y[i] = [1.0, 2.0, 3.0]
            else:
                y[i] = [3.0, 2.0, 1.0]
        tree = sigma.RankingTree(random_state=0)
        tree.fit(X, y)
        predictions = tree.predict(X)
        self.assertEqual(predictions.shape, (n_samples,))
        self.assertTrue(
            set(predictions.tolist()).issubset(set(tree.item_names_))
        )

    def test_item_names_fall_back_to_integer_indices(self):
        """Bare numpy y with no constructor names yields integer indices."""
        rng = numpy.random.default_rng(11)
        n_samples = 80
        n_items = 4
        X = rng.normal(size=(n_samples, 1))
        y = numpy.empty((n_samples, n_items), dtype=float)
        for i in range(n_samples):
            if X[i, 0] > 0:
                y[i] = [1.0, 2.0, 3.0, 4.0]
            else:
                y[i] = [4.0, 3.0, 2.0, 1.0]
        tree = sigma.RankingTree(random_state=0)
        tree.fit(X, y)
        self.assertTrue(numpy.issubdtype(tree.item_names_.dtype, numpy.integer))
        numpy.testing.assert_array_equal(
            tree.item_names_, numpy.arange(n_items)
        )
        predictions = tree.predict(X)
        self.assertTrue(numpy.issubdtype(predictions.dtype, numpy.integer))

    def test_predict_rank_returns_per_item_matrix(self):
        """predict_rank returns a (n_samples, n_items) matrix."""
        rng = numpy.random.default_rng(2)
        n_samples = 60
        n_items = 3
        X = rng.normal(size=(n_samples, 1))
        y = numpy.empty((n_samples, n_items), dtype=float)
        for i in range(n_samples):
            y[i] = [1.0, 2.0, 3.0] if X[i, 0] > 0 else [3.0, 2.0, 1.0]
        tree = sigma.RankingTree(random_state=0)
        tree.fit(X, y)
        ranks = tree.predict_rank(X)
        self.assertEqual(ranks.shape, (n_samples, 3))

    def test_default_ci_method_is_bayesian_bootstrap(self):
        """The default CI method matches the Rubin 1981 Bayesian bootstrap."""
        tree = sigma.RankingTree()
        self.assertEqual(tree.ci_method, "bayesian_bootstrap")

    def test_default_pca_components_is_ten(self):
        """The constructor default for pca_components is 10."""
        tree = sigma.RankingTree()
        self.assertEqual(tree.pca_components, 10)

    def test_rejects_distribution_specific_ci_method_at_construction(self):
        """RankingTree refuses the seven distribution-specific CI names."""
        rejected = [
            "beta",
            "exponential",
            "gamma",
            "log_normal",
            "log_normal_gci",
            "poisson",
            "poisson_jeffreys",
        ]
        for name in rejected:
            with self.assertRaises(ValueError):
                sigma._types._validate_literal_param(
                    name,
                    sigma._types.CiMethodRankingTree,
                    "ci_method",
                )

    def test_rejects_pca_components_below_one(self):
        """pca_components < 1 is rejected with ValueError."""
        with self.assertRaises(ValueError):
            sigma.RankingTree(pca_components=0)

    def test_no_offset_support(self):
        """Passing an offset to fit raises ValueError."""
        rng = numpy.random.default_rng(0)
        n = 30
        X = rng.normal(size=(n, 1))
        y = numpy.tile(numpy.array([1.0, 2.0, 3.0]), (n, 1))
        tree = sigma.RankingTree()
        with self.assertRaises(ValueError):
            tree.fit(X, y, offset=numpy.zeros(n))

    def test_fit_with_two_items(self):
        """Binary K=2 fit splits on the covariate and flips the favorite."""
        rng = numpy.random.default_rng(7)
        n_samples = 200
        X = rng.normal(size=(n_samples, 1))
        y = numpy.where(
            X[:, 0:1] > 0,
            numpy.array([[1.0, 2.0]]),
            numpy.array([[2.0, 1.0]]),
        )
        tree = sigma.RankingTree(random_state=0)
        tree.fit(X, y)
        self.assertGreaterEqual(len(tree.leaves_), 2)
        leaf_predictions = {int(leaf.prediction) for leaf in tree.leaves_}
        self.assertEqual(leaf_predictions, {0, 1})

    def test_ci_coverage_none_disables_per_item_ci(self):
        """ci_coverage=None makes every per-item ci_low / ci_high be None."""
        rng = numpy.random.default_rng(8)
        n_samples = 60
        n_items = 3
        X = rng.normal(size=(n_samples, 1))
        y = numpy.empty((n_samples, n_items), dtype=float)
        for i in range(n_samples):
            y[i] = [1.0, 2.0, 3.0] if X[i, 0] > 0 else [3.0, 2.0, 1.0]
        tree = sigma.RankingTree(ci_coverage=None, random_state=0)
        tree.fit(X, y)
        for leaf in tree.leaves_:
            for metric in leaf.metrics:
                self.assertIsNone(metric.ci_low)
                self.assertIsNone(metric.ci_high)

    def test_pickle_roundtrip_preserves_predictions(self):
        """A fitted RankingTree round-trips through pickle without drift."""
        import pickle

        rng = numpy.random.default_rng(9)
        n_samples = 100
        n_items = 3
        X = rng.normal(size=(n_samples, 1))
        y = numpy.empty((n_samples, n_items), dtype=float)
        for i in range(n_samples):
            y[i] = [1.0, 2.0, 3.0] if X[i, 0] > 0 else [3.0, 2.0, 1.0]
        tree = sigma.RankingTree(random_state=0)
        tree.fit(X, y)
        predictions_before = tree.predict(X)
        ranks_before = tree.predict_rank(X)
        restored = pickle.loads(pickle.dumps(tree))
        predictions_after = restored.predict(X)
        ranks_after = restored.predict_rank(X)
        numpy.testing.assert_array_equal(predictions_before, predictions_after)
        numpy.testing.assert_allclose(ranks_before, ranks_after)

    def test_sample_weight_none_matches_ones_array(self):
        """sample_weight=None and ones-array produce identical fits."""
        rng = numpy.random.default_rng(10)
        n_samples = 80
        n_items = 3
        X = rng.normal(size=(n_samples, 1))
        y = numpy.empty((n_samples, n_items), dtype=float)
        for i in range(n_samples):
            y[i] = [1.0, 2.0, 3.0] if X[i, 0] > 0 else [3.0, 2.0, 1.0]
        tree_none = sigma.RankingTree(random_state=0)
        tree_none.fit(X, y)
        tree_ones = sigma.RankingTree(random_state=0)
        tree_ones.fit(X, y, sample_weight=numpy.ones(n_samples))
        numpy.testing.assert_array_equal(
            tree_none.predict(X), tree_ones.predict(X)
        )
        numpy.testing.assert_allclose(
            tree_none.predict_rank(X), tree_ones.predict_rank(X)
        )

    def test_rejects_transmuter_at_construction(self):
        """Passing a non-None transmuter raises ValueError at construction."""

        def some_transmuter(X, y, w, offset, side_data):
            return y, w, offset

        with self.assertRaises(ValueError):
            sigma.RankingTree(transmuter=some_transmuter)

    def test_rejects_min_obs_per_item_kwarg(self):
        """min_obs_per_item was removed; not listed in the signature."""
        parameters = inspect.signature(sigma.RankingTree).parameters
        self.assertNotIn("min_obs_per_item", parameters)

    def test_pc_loadings_attribute_set_after_fit(self):
        """After fit, _pc_loadings_ has shape (n_items, R = min(pca_components, n_items))."""
        rng = numpy.random.default_rng(13)
        n_samples = 200
        n_items = 30
        X = rng.normal(size=(n_samples, 1))
        y = numpy.empty((n_samples, n_items), dtype=float)
        for i in range(n_samples):
            y[i] = (
                numpy.arange(1.0, n_items + 1.0)
                if X[i, 0] > 0
                else numpy.arange(n_items, 0.0, -1.0)
            )
        tree = sigma.RankingTree(pca_components=5, random_state=0)
        tree.fit(X, y)
        self.assertEqual(tree._pc_loadings_.shape, (n_items, 5))

    def test_pc_loadings_orthonormal(self):
        """V.T @ V is approximately the identity matrix (PCA invariant)."""
        rng = numpy.random.default_rng(14)
        n_samples = 200
        n_items = 30
        X = rng.normal(size=(n_samples, 1))
        y = numpy.empty((n_samples, n_items), dtype=float)
        for i in range(n_samples):
            y[i] = rng.permutation(n_items).astype(float) + 1.0
        tree = sigma.RankingTree(pca_components=10, random_state=0)
        tree.fit(X, y)
        V = tree._pc_loadings_
        product = V.T @ V
        numpy.testing.assert_allclose(
            product, numpy.eye(product.shape[0]), atol=1e-6
        )

    def test_per_node_metrics_cover_full_catalogue(self):
        """node.metrics has length equal to the full catalogue, not M+1."""
        rng = numpy.random.default_rng(12)
        n_samples = 200
        n_items = 30
        X = rng.normal(size=(n_samples, 1))
        y = numpy.empty((n_samples, n_items), dtype=float)
        for i in range(n_samples):
            base = numpy.arange(1.0, n_items + 1.0)
            if X[i, 0] < 0:
                base = base[::-1].copy()
            y[i] = base
        tree = sigma.RankingTree(pca_components=5, random_state=0)
        tree.fit(X, y)
        for node in tree.nodes_:
            self.assertEqual(len(node.metrics), n_items)


class TestTopDisplayedItems(unittest.TestCase):
    """Tests for the render-time top_displayed_items knob."""

    __slots__ = ()

    def _build_two_leaf_tree(self, top_a, top_b):
        """Fit a depth-1 tree whose two leaves prefer disjoint items."""
        rng = numpy.random.default_rng(31)
        n_per_leaf = 100
        n_items = 10
        X_left = numpy.full((n_per_leaf, 1), -1.0)
        X_right = numpy.full((n_per_leaf, 1), 1.0)
        X = numpy.vstack([X_left, X_right]) + rng.normal(
            scale=0.01, size=(2 * n_per_leaf, 1)
        )
        y_left = numpy.tile(
            numpy.arange(1, n_items + 1, dtype=float), (n_per_leaf, 1)
        )
        y_left[:, top_a], y_left[:, 0 : len(top_a)] = (
            y_left[:, 0 : len(top_a)].copy(),
            y_left[:, top_a].copy(),
        )
        y_right = numpy.tile(
            numpy.arange(1, n_items + 1, dtype=float), (n_per_leaf, 1)
        )
        y_right[:, top_b], y_right[:, 0 : len(top_b)] = (
            y_right[:, 0 : len(top_b)].copy(),
            y_right[:, top_b].copy(),
        )
        y = numpy.vstack([y_left, y_right])
        tree = sigma.RankingTree(random_state=0)
        tree.fit(X, y)
        return tree

    def test_to_text_default_top_displayed_items_is_three(self):
        """Default top_displayed_items=3 picks the union of per-leaf top-3."""
        tree = self._build_two_leaf_tree([0, 1, 2], [7, 8, 9])
        text = tree.to_text(precision=2)
        first_row = text.splitlines()[0]
        rank_columns = first_row.count("Rank of")
        union_count = len(tree._compute_displayed_indices(3))
        self.assertEqual(rank_columns, union_count)

    def test_to_text_top_displayed_items_respects_argument(self):
        """Passing top_displayed_items=5 widens the column union."""
        tree = self._build_two_leaf_tree([0, 1, 2], [7, 8, 9])
        text = tree.to_text(precision=2, top_displayed_items=5)
        first_row = text.splitlines()[0]
        rank_columns = first_row.count("Rank of")
        union_count = len(tree._compute_displayed_indices(5))
        self.assertEqual(rank_columns, union_count)

    def test_per_leaf_top_items_picked_by_lowest_mean_rank(self):
        """Each leaf's contribution to the union is its top items by mean rank."""
        tree = self._build_two_leaf_tree([0, 1, 2], [7, 8, 9])
        union = set(tree._compute_displayed_indices(3))
        for leaf in tree.leaves_:
            values = numpy.array([metric.value for metric in leaf.metrics])
            leaf_top = set(numpy.argsort(values, kind="stable")[:3].tolist())
            self.assertTrue(leaf_top.issubset(union))

    def test_top_displayed_items_rejected_for_non_ranking_trees(self):
        """Passing top_displayed_items to a non-ranking tree raises."""
        rng = numpy.random.default_rng(32)
        X = rng.normal(size=(40, 1))
        y = (X[:, 0] > 0).astype(int)
        tree = sigma.ClassificationTree(random_state=0)
        tree.fit(X, y)
        with self.assertRaises(ValueError):
            tree.to_text(top_displayed_items=3)

    def test_top_displayed_items_rejects_zero_and_negative(self):
        """top_displayed_items must be at least 1."""
        tree = self._build_two_leaf_tree([0, 1, 2], [7, 8, 9])
        with self.assertRaises(ValueError):
            tree.to_text(top_displayed_items=0)
        with self.assertRaises(ValueError):
            tree.to_text(top_displayed_items=-1)


if __name__ == "__main__":
    unittest.main()
