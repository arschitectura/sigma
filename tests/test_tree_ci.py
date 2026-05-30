"""Unit tests for confidence intervals across Tree estimators."""

import unittest

import numpy
import numpy.testing
import scipy.stats
import sklearn.datasets
import sklearn.exceptions
import sklearn.model_selection

import sigma._node
import sigma._partition
import sigma._tree
import sigma._tree_classification
import sigma._tree_regression
import sigma._tree_survival
import sigma._tree_text
import sigma._types

import _helpers


class TestRegressionTreeCI(unittest.TestCase):
    """Tests for confidence interval computation in RegressionTree."""

    __slots__ = ()

    def test_all_nodes_have_ci(self):
        """Verify every node has ci_low and ci_high after fitting."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        regression_tree.fit(X, y)
        nodes = _helpers._collect_nodes(regression_tree.content_)
        for index, node in enumerate(nodes):
            self.assertIsNotNone(node.ci_low, f"node {index} ci_low")
            self.assertIsNotNone(node.ci_high, f"node {index} ci_high")

    def test_ci_brackets_prediction(self):
        """Verify ci_low <= prediction <= ci_high for every node."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        regression_tree.fit(X, y)
        nodes = _helpers._collect_nodes(regression_tree.content_)
        for node in nodes:
            ci_low = node.ci_low
            ci_high = node.ci_high
            assert ci_low is not None
            assert ci_high is not None
            self.assertLessEqual(ci_low, node.prediction)
            self.assertGreaterEqual(ci_high, node.prediction)

    def test_ci_coverage_none_disables_ci(self):
        """Verify ci_low and ci_high are None when ci_coverage is None."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            ci_coverage=None,
        )
        regression_tree.fit(X, y)
        nodes = _helpers._collect_nodes(regression_tree.content_)
        for node in nodes:
            self.assertIsNone(node.ci_low)
            self.assertIsNone(node.ci_high)

    def test_wider_coverage_gives_wider_interval(self):
        """Verify a 99% CI is at least as wide as a 90% CI."""
        X, y = sklearn.datasets.load_diabetes(return_X_y=True)
        reg_90 = sigma._tree_regression.RegressionTree(
            correlation="normal",
            ci_coverage=0.90,
        )
        reg_99 = sigma._tree_regression.RegressionTree(
            correlation="normal",
            ci_coverage=0.99,
        )
        reg_90.fit(X, y)
        reg_99.fit(X, y)
        ci_high_90 = reg_90.content_.ci_high
        ci_low_90 = reg_90.content_.ci_low
        ci_high_99 = reg_99.content_.ci_high
        ci_low_99 = reg_99.content_.ci_low
        assert ci_high_90 is not None
        assert ci_low_90 is not None
        assert ci_high_99 is not None
        assert ci_low_99 is not None
        width_90 = ci_high_90 - ci_low_90
        width_99 = ci_high_99 - ci_low_99
        self.assertGreater(width_99, width_90)

    def test_to_text_shows_ci(self):
        """Verify to_text output contains CI in parenthesized format."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        regression_tree.fit(X, y)
        import re

        output = regression_tree.to_text()
        has_ci = re.search(r"\(\S+ to \S+\)", output)
        self.assertIsNotNone(has_ci)
        self.assertIn("All records", output)
        self.assertIn("<=", output)
        self.assertIn(">", output)

    def test_to_text_default_response_label(self):
        """Default regression label reads 'Predicted mean'."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        regression_tree.fit(X, y)

        output = regression_tree.to_text()
        self.assertIn("Predicted mean", output)

    def test_to_text_custom_response_name(self):
        """Display-time response_name replaces 'Predicted' in the label."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
        )
        regression_tree.fit(X, y)
        output = regression_tree.to_text(response_name="Price")
        self.assertIn("Price mean", output)
        self.assertNotIn("Predicted", output)

    def test_to_text_capitalizes_lowercase_response_name(self):
        """Lowercase response_name renders with a leading uppercase letter."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
        )
        regression_tree.fit(X, y)
        output = regression_tree.to_text(response_name="price")
        self.assertIn("Price mean", output)
        self.assertNotIn("price mean", output)


class TestClassificationTreeCI(unittest.TestCase):
    """Tests for CI behavior in ClassificationTree."""

    __slots__ = ()

    def test_classification_nodes_have_per_class_ci_arrays(self):
        """ClassificationNode.ci_low / .ci_high are per-class arrays."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 1.0)
        classification_tree = sigma._tree_classification.ClassificationTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        classification_tree.fit(X, y)
        nodes = _helpers._collect_nodes(classification_tree.content_)
        for node in nodes:
            ci_low = node.ci_low
            ci_high = node.ci_high
            assert ci_low is not None
            assert ci_high is not None
            self.assertEqual(ci_low.shape, (2,))
            self.assertEqual(ci_high.shape, (2,))

    def test_classification_tree_has_per_class_ci(self):
        """All nodes have per-class CI arrays of length n_classes."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 1.0)
        classification_tree = sigma._tree_classification.ClassificationTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        classification_tree.fit(X, y)
        nodes = _helpers._collect_nodes(classification_tree.content_)
        for node in nodes:
            ci_low = node.ci_low
            ci_high = node.ci_high
            self.assertIsNotNone(ci_low)
            self.assertIsNotNone(ci_high)
            assert ci_low is not None
            assert ci_high is not None
            self.assertEqual(len(ci_low), 2)
            self.assertEqual(len(ci_high), 2)

    def test_per_class_ci_brackets_distribution(self):
        """Per-class CI brackets the class distribution values."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 1.0)
        classification_tree = sigma._tree_classification.ClassificationTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        classification_tree.fit(X, y)
        nodes = _helpers._collect_nodes(classification_tree.content_)
        for node in nodes:
            if node.class_distribution is None:
                continue
            ci_low = node.ci_low
            ci_high = node.ci_high
            assert ci_low is not None
            assert ci_high is not None
            for k in range(len(node.class_distribution)):
                self.assertLessEqual(ci_low[k], node.class_distribution[k])
                self.assertGreaterEqual(ci_high[k], node.class_distribution[k])

    def test_per_class_ci_none_when_disabled(self):
        """Per-class CI fields are None when ci_coverage is None."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 1.0)
        classification_tree = sigma._tree_classification.ClassificationTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            ci_coverage=None,
        )
        classification_tree.fit(X, y)
        nodes = _helpers._collect_nodes(classification_tree.content_)
        for node in nodes:
            self.assertIsNone(node.ci_low)
            self.assertIsNone(node.ci_high)

    def test_wider_coverage_gives_wider_per_class_interval(self):
        """A 99% per-class CI is at least as wide as a 90% CI."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 1.0)
        clf_90 = sigma._tree_classification.ClassificationTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            ci_coverage=0.90,
        )
        clf_99 = sigma._tree_classification.ClassificationTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            ci_coverage=0.99,
        )
        clf_90.fit(X, y)
        clf_99.fit(X, y)
        lo_90 = clf_90.content_.ci_low
        hi_90 = clf_90.content_.ci_high
        lo_99 = clf_99.content_.ci_low
        hi_99 = clf_99.content_.ci_high
        assert lo_90 is not None and hi_90 is not None
        assert lo_99 is not None and hi_99 is not None
        width_90 = hi_90 - lo_90
        width_99 = hi_99 - lo_99
        for k in range(len(width_90)):
            self.assertGreaterEqual(width_99[k], width_90[k])

    def test_to_text_shows_per_class_ci(self):
        """Classification print output contains per-class CI intervals."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 1.0)
        classification_tree = sigma._tree_classification.ClassificationTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
        )
        classification_tree.fit(X, y)
        import re

        output = classification_tree.to_text(class_names=["cat", "dog"])
        has_ci = re.search(r"\S+ \(\S+ to \S+\)", output)
        self.assertIsNotNone(has_ci)


class TestRegressionTreeCiMethod(unittest.TestCase):
    """Tests for each regression tree ci_method including extreme cases."""

    __slots__ = ()

    def test_default_ci_method_is_bayesian_bootstrap(self):
        """Default ci_method for the regression tree is bayesian_bootstrap."""
        regression_tree = sigma._tree_regression.RegressionTree()
        self.assertEqual(regression_tree.ci_method, "bayesian_bootstrap")

    def test_each_method_produces_finite_ci(self):
        """Each method returns finite bounds on well-formed positive data."""
        rng = numpy.random.default_rng(0)
        y = rng.uniform(0.1, 0.9, 30)
        weights = numpy.ones(30)
        for method in _helpers._REGRESSION_TREE_CI_METHODS:
            regression_tree = sigma._tree_regression.RegressionTree(
                ci_method=method
            )
            regression_tree._rng_ci_ = numpy.random.default_rng(0)
            ci_low, ci_high = regression_tree._compute_ci(y, weights, None)
            assert ci_low is not None and ci_high is not None
            self.assertTrue(numpy.isfinite(ci_low))
            self.assertTrue(numpy.isfinite(ci_high))
            self.assertLessEqual(ci_low, ci_high)

    def test_bracket_mean_methods_bracket_sample_mean(self):
        """Bracket-mean methods satisfy ci_low <= Y_bar <= ci_high."""
        rng = numpy.random.default_rng(0)
        y = rng.uniform(0.1, 0.9, 30)
        weights = numpy.ones(30)
        p_hat = float(numpy.average(y, weights=weights))
        for method in _helpers._REGRESSION_TREE_CI_METHODS_BRACKET_MEAN:
            regression_tree = sigma._tree_regression.RegressionTree(
                ci_method=method
            )
            regression_tree._rng_ci_ = numpy.random.default_rng(0)
            ci_low, ci_high = regression_tree._compute_ci(y, weights, None)
            assert ci_low is not None and ci_high is not None
            self.assertLessEqual(ci_low, p_hat + 1e-12)
            self.assertGreaterEqual(ci_high, p_hat - 1e-12)

    def test_ci_coverage_none_disables_ci_for_each_method(self):
        """ci_coverage=None returns (None, None) for every method."""
        y = numpy.array([1.0, 2.0, 3.0])
        weights = numpy.ones(3)
        for method in _helpers._REGRESSION_TREE_CI_METHODS:
            regression_tree = sigma._tree_regression.RegressionTree(
                ci_method=method, ci_coverage=None
            )
            ci_low, ci_high = regression_tree._compute_ci(y, weights, None)
            self.assertIsNone(ci_low)
            self.assertIsNone(ci_high)

    def test_n_active_zero_returns_zero_point(self):
        """All-zero weights yield (0.0, 0.0) for every method."""
        y = numpy.array([1.0, 2.0, 3.0])
        weights = numpy.zeros(3)
        for method in _helpers._REGRESSION_TREE_CI_METHODS:
            regression_tree = sigma._tree_regression.RegressionTree(
                ci_method=method
            )
            ci_low, ci_high = regression_tree._compute_ci(y, weights, None)
            self.assertEqual(ci_low, 0.0)
            self.assertEqual(ci_high, 0.0)

    def test_n_active_one_returns_point(self):
        """A single active sample collapses the CI to that value."""
        y = numpy.array([4.0, 2.0, 3.0])
        weights = numpy.array([0.0, 1.0, 0.0])
        for method in _helpers._REGRESSION_TREE_CI_METHODS:
            regression_tree = sigma._tree_regression.RegressionTree(
                ci_method=method
            )
            ci_low, ci_high = regression_tree._compute_ci(y, weights, None)
            assert ci_low is not None and ci_high is not None
            self.assertAlmostEqual(ci_low, 2.0)
            self.assertAlmostEqual(ci_high, 2.0)

    def test_constant_response_collapses_ci(self):
        """Constant-response collapses the CI for zero-variance methods."""
        y = numpy.full(20, 0.5)
        weights = numpy.ones(20)
        for method in _helpers._REGRESSION_TREE_CI_METHODS_COLLAPSE_ON_CONSTANT:
            regression_tree = sigma._tree_regression.RegressionTree(
                ci_method=method
            )
            regression_tree._rng_ci_ = numpy.random.default_rng(0)
            ci_low, ci_high = regression_tree._compute_ci(y, weights, None)
            assert ci_low is not None and ci_high is not None
            self.assertAlmostEqual(ci_low, 0.5, places=6)
            self.assertAlmostEqual(ci_high, 0.5, places=6)

    def test_fractional_weights_produce_finite_ci(self):
        """Non-integer weights do not produce NaN or inf for any method."""
        rng = numpy.random.default_rng(1)
        y = rng.uniform(0.1, 0.9, size=25)
        weights = rng.uniform(0.1, 2.0, size=25)
        for method in _helpers._REGRESSION_TREE_CI_METHODS:
            regression_tree = sigma._tree_regression.RegressionTree(
                ci_method=method
            )
            regression_tree._rng_ci_ = numpy.random.default_rng(0)
            ci_low, ci_high = regression_tree._compute_ci(y, weights, None)
            assert ci_low is not None and ci_high is not None
            self.assertTrue(numpy.isfinite(ci_low))
            self.assertTrue(numpy.isfinite(ci_high))

    def test_wider_coverage_gives_wider_ci_for_each_method(self):
        """Higher ci_coverage yields a wider CI for every method."""
        rng = numpy.random.default_rng(2)
        y = rng.uniform(0.1, 0.9, 50)
        weights = numpy.ones(50)
        for method in _helpers._REGRESSION_TREE_CI_METHODS:
            reg_narrow = sigma._tree_regression.RegressionTree(
                ci_method=method, ci_coverage=0.5
            )
            reg_wide = sigma._tree_regression.RegressionTree(
                ci_method=method, ci_coverage=0.99
            )
            reg_narrow._rng_ci_ = numpy.random.default_rng(0)
            reg_wide._rng_ci_ = numpy.random.default_rng(0)
            low_n, high_n = reg_narrow._compute_ci(y, weights, None)
            low_w, high_w = reg_wide._compute_ci(y, weights, None)
            assert low_n is not None and high_n is not None
            assert low_w is not None and high_w is not None
            self.assertGreater(high_w - low_w, high_n - low_n)

    def test_normal_ci_widens_under_unequal_weights(self):
        """Kish n_eff shrinks with unequal weights, widening the normal CI."""
        rng = numpy.random.default_rng(3)
        n = 100
        y = rng.standard_normal(n)
        w_equal = numpy.ones(n)
        w_unequal = numpy.ones(n) * 0.01
        w_unequal[0] = 100.0
        regression_tree = sigma._tree_regression.RegressionTree(
            ci_method="normal"
        )
        low_e, high_e = regression_tree._compute_ci(y, w_equal, None)
        low_u, high_u = regression_tree._compute_ci(y, w_unequal, None)
        assert low_e is not None and high_e is not None
        assert low_u is not None and high_u is not None
        self.assertGreater(high_u - low_u, high_e - low_e)

    def test_fitted_tree_brackets_prediction_for_bracket_mean_methods(self):
        """After fit, bracket-mean methods satisfy ci_low <= prediction <=
        ci_high.
        """
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.2, 0.8)
        for method in _helpers._REGRESSION_TREE_CI_METHODS_BRACKET_MEAN:
            regression_tree = sigma._tree_regression.RegressionTree(
                correlation="normal",
                min_splits=2,
                min_buckets=1,
                ci_method=method,
            )
            regression_tree.fit(X, y)
            nodes = _helpers._collect_nodes(regression_tree.content_)
            for node in nodes:
                ci_low = node.ci_low
                ci_high = node.ci_high
                assert ci_low is not None and ci_high is not None
                self.assertLessEqual(ci_low, node.prediction + 1e-12)
                self.assertGreaterEqual(ci_high, node.prediction - 1e-12)


class TestRegressionTreeBayesianBootstrapCi(unittest.TestCase):
    """Tests specific to ci_method='bayesian_bootstrap'."""

    __slots__ = ()

    def test_random_state_makes_ci_reproducible(self):
        """Two fits with the same random_state produce identical leaf CIs."""
        rng = numpy.random.default_rng(42)
        n = 100
        X = rng.standard_normal((n, 2))
        y = 2.0 * X[:, 0] + rng.standard_normal(n) * 0.5
        tree1 = sigma._tree_regression.RegressionTree(
            correlation="normal",
            random_state=123,
            min_splits=10,
            min_buckets=5,
        )
        tree2 = sigma._tree_regression.RegressionTree(
            correlation="normal",
            random_state=123,
            min_splits=10,
            min_buckets=5,
        )
        tree1.fit(X, y)
        tree2.fit(X, y)
        self.assertEqual(len(tree1.leaves_), len(tree2.leaves_))
        for leaf1, leaf2 in zip(tree1.leaves_, tree2.leaves_):
            self.assertEqual(leaf1.ci_low, leaf2.ci_low)
            self.assertEqual(leaf1.ci_high, leaf2.ci_high)


class TestRegressionTreeBcaCi(unittest.TestCase):
    """Tests specific to ci_method='bca'."""

    __slots__ = ()

    def test_brackets_weighted_mean_on_skew_sample(self):
        """BCa CI contains the weighted mean on a log-normal sample."""
        rng = numpy.random.default_rng(7)
        y = numpy.exp(rng.standard_normal(50))
        weights = numpy.ones(50)
        weighted_mean = float(numpy.average(y, weights=weights))
        regression_tree = sigma._tree_regression.RegressionTree(ci_method="bca")
        regression_tree._rng_ci_ = numpy.random.default_rng(0)
        ci_low, ci_high = regression_tree._compute_ci(y, weights, None)
        assert ci_low is not None and ci_high is not None
        self.assertLessEqual(ci_low, weighted_mean)
        self.assertGreaterEqual(ci_high, weighted_mean)

    def test_collapses_when_all_responses_identical(self):
        """CI degenerates to the point estimate when y is constant."""
        y = numpy.full(10, 3.5)
        weights = numpy.ones(10)
        regression_tree = sigma._tree_regression.RegressionTree(ci_method="bca")
        regression_tree._rng_ci_ = numpy.random.default_rng(0)
        ci_low, ci_high = regression_tree._compute_ci(y, weights, None)
        self.assertEqual(ci_low, 3.5)
        self.assertEqual(ci_high, 3.5)

    def test_collapses_when_only_one_positive_weight(self):
        """CI degenerates when only a single observation has positive weight."""
        y = numpy.array([1.0, 2.0, 3.0, 4.0, 5.0])
        weights = numpy.array([0.0, 0.0, 1.0, 0.0, 0.0])
        regression_tree = sigma._tree_regression.RegressionTree(ci_method="bca")
        ci_low, ci_high = regression_tree._compute_ci(y, weights, None)
        self.assertEqual(ci_low, 3.0)
        self.assertEqual(ci_high, 3.0)

    def test_bounds_finite_at_extreme_coverage(self):
        """Adjusted percentiles stay in [0, 1] and CI bounds are finite at 99% coverage."""
        rng = numpy.random.default_rng(11)
        y = numpy.exp(rng.standard_normal(30))
        weights = numpy.ones(30)
        regression_tree = sigma._tree_regression.RegressionTree(
            ci_method="bca", ci_coverage=0.99
        )
        regression_tree._rng_ci_ = numpy.random.default_rng(0)
        ci_low, ci_high = regression_tree._compute_ci(y, weights, None)
        assert ci_low is not None and ci_high is not None
        self.assertTrue(numpy.isfinite(ci_low))
        self.assertTrue(numpy.isfinite(ci_high))
        self.assertLessEqual(ci_low, ci_high)

    def test_random_state_makes_ci_reproducible(self):
        """Two fits with the same random_state produce identical leaf CIs."""
        rng = numpy.random.default_rng(42)
        n = 100
        X = rng.standard_normal((n, 2))
        y = 2.0 * X[:, 0] + rng.standard_normal(n) * 0.5
        tree1 = sigma._tree_regression.RegressionTree(
            correlation="normal",
            ci_method="bca",
            random_state=123,
            min_splits=10,
            min_buckets=5,
        )
        tree2 = sigma._tree_regression.RegressionTree(
            correlation="normal",
            ci_method="bca",
            random_state=123,
            min_splits=10,
            min_buckets=5,
        )
        tree1.fit(X, y)
        tree2.fit(X, y)
        self.assertEqual(len(tree1.leaves_), len(tree2.leaves_))
        for leaf1, leaf2 in zip(tree1.leaves_, tree2.leaves_):
            self.assertEqual(leaf1.ci_low, leaf2.ci_low)
            self.assertEqual(leaf1.ci_high, leaf2.ci_high)


class TestRegressionTreeStudentTCi(unittest.TestCase):
    """Tests specific to ci_method='student_t'."""

    __slots__ = ()

    def test_wider_than_normal_on_small_sample(self):
        """t-quantile is larger than z for small df, giving a wider CI."""
        rng = numpy.random.default_rng(0)
        y = rng.standard_normal(5)
        weights = numpy.ones(5)
        reg_t = sigma._tree_regression.RegressionTree(ci_method="student_t")
        reg_n = sigma._tree_regression.RegressionTree(ci_method="normal")
        low_t, high_t = reg_t._compute_ci(y, weights, None)
        low_n, high_n = reg_n._compute_ci(y, weights, None)
        assert low_t is not None and high_t is not None
        assert low_n is not None and high_n is not None
        self.assertGreater(high_t - low_t, high_n - low_n)

    def test_matches_normal_asymptotically(self):
        """On large samples, student_t and normal intervals almost coincide."""
        rng = numpy.random.default_rng(1)
        y = rng.standard_normal(500)
        weights = numpy.ones(500)
        low_t, high_t = sigma._tree_regression.RegressionTree(
            ci_method="student_t"
        )._compute_ci(y, weights, None)
        low_n, high_n = sigma._tree_regression.RegressionTree(
            ci_method="normal"
        )._compute_ci(y, weights, None)
        assert low_t is not None and high_t is not None
        assert low_n is not None and high_n is not None
        self.assertAlmostEqual(low_t, low_n, places=3)
        self.assertAlmostEqual(high_t, high_n, places=3)


class TestRegressionTreeLogNormalCi(unittest.TestCase):
    """Tests specific to ci_method='log_normal'."""

    __slots__ = ()

    def test_bounds_positive(self):
        """Log-normal CI bounds are strictly positive."""
        rng = numpy.random.default_rng(2)
        y = numpy.exp(rng.standard_normal(50))
        weights = numpy.ones(50)
        regression_tree = sigma._tree_regression.RegressionTree(
            ci_method="log_normal"
        )
        ci_low, ci_high = regression_tree._compute_ci(y, weights, None)
        assert ci_low is not None and ci_high is not None
        self.assertGreater(ci_low, 0.0)
        self.assertGreater(ci_high, 0.0)

    def test_brackets_cox_mean_estimate(self):
        """CI brackets the log-normal mean MLE exp(mu + sigma^2 / 2)."""
        rng = numpy.random.default_rng(3)
        y = numpy.exp(rng.standard_normal(50))
        weights = numpy.ones(50)
        log_y = numpy.log(y)
        mu_log = float(numpy.average(log_y, weights=weights))
        sigma_sq = float(numpy.average((log_y - mu_log) ** 2, weights=weights))
        cox_mean = float(numpy.exp(mu_log + sigma_sq / 2.0))
        regression_tree = sigma._tree_regression.RegressionTree(
            ci_method="log_normal"
        )
        ci_low, ci_high = regression_tree._compute_ci(y, weights, None)
        assert ci_low is not None and ci_high is not None
        self.assertLessEqual(ci_low, cox_mean + 1e-9)
        self.assertGreaterEqual(ci_high, cox_mean - 1e-9)

    def test_nonpositive_y_raises(self):
        """Any y <= 0 raises ValueError during CI computation."""
        y = numpy.array([1.0, 0.0, 2.0])
        weights = numpy.ones(3)
        regression_tree = sigma._tree_regression.RegressionTree(
            ci_method="log_normal"
        )
        with self.assertRaises(ValueError):
            regression_tree._compute_ci(y, weights, None)

    def test_uses_t_quantile_not_z(self):
        """CI uses the Student-t quantile with df = n_eff - 1, not the standard normal."""
        log_y = numpy.array([0.0, 0.5, 1.0, 0.2, -0.3, 0.8, -0.1, 0.4])
        y = numpy.exp(log_y)
        weights = numpy.ones(8)
        w_sum = float(weights.sum())
        mu_log = float(numpy.dot(weights, log_y) / w_sum)
        sigma_sq = float(numpy.dot(weights, (log_y - mu_log) ** 2) / w_sum)
        n_eff = float(w_sum**2 / numpy.dot(weights, weights))
        log_mean = mu_log + sigma_sq / 2.0
        se = float(
            numpy.sqrt(sigma_sq / n_eff + sigma_sq**2 / (2.0 * (n_eff - 1.0)))
        )
        alpha = (1.0 - 0.95) / 2.0
        t_quantile = float(scipy.stats.t.ppf(1.0 - alpha, df=n_eff - 1.0))
        expected_low = float(numpy.exp(log_mean - t_quantile * se))
        expected_high = float(numpy.exp(log_mean + t_quantile * se))
        regression_tree = sigma._tree_regression.RegressionTree(
            ci_method="log_normal"
        )
        ci_low, ci_high = regression_tree._compute_ci(y, weights, None)
        assert ci_low is not None and ci_high is not None
        self.assertAlmostEqual(ci_low, expected_low, places=9)
        self.assertAlmostEqual(ci_high, expected_high, places=9)
        z_quantile = float(scipy.stats.norm.ppf(1.0 - alpha))
        z_low = float(numpy.exp(log_mean - z_quantile * se))
        z_high = float(numpy.exp(log_mean + z_quantile * se))
        self.assertLess(ci_low, z_low)
        self.assertGreater(ci_high, z_high)


class TestRegressionTreeLogNormalGciCi(unittest.TestCase):
    """Tests specific to ci_method='log_normal_gci'."""

    __slots__ = ()

    def test_bounds_positive(self):
        """Generalized CI bounds are strictly positive."""
        rng = numpy.random.default_rng(2)
        y = numpy.exp(rng.standard_normal(50))
        weights = numpy.ones(50)
        regression_tree = sigma._tree_regression.RegressionTree(
            ci_method="log_normal_gci"
        )
        regression_tree._rng_ci_ = numpy.random.default_rng(0)
        ci_low, ci_high = regression_tree._compute_ci(y, weights, None)
        assert ci_low is not None and ci_high is not None
        self.assertGreater(ci_low, 0.0)
        self.assertGreater(ci_high, 0.0)

    def test_brackets_cox_mean_estimate(self):
        """Generalized CI brackets the log-normal mean MLE exp(mu + sigma^2 / 2)."""
        rng = numpy.random.default_rng(3)
        y = numpy.exp(rng.standard_normal(50))
        weights = numpy.ones(50)
        log_y = numpy.log(y)
        mu_log = float(numpy.average(log_y, weights=weights))
        sigma_sq = float(numpy.average((log_y - mu_log) ** 2, weights=weights))
        cox_mean = float(numpy.exp(mu_log + sigma_sq / 2.0))
        regression_tree = sigma._tree_regression.RegressionTree(
            ci_method="log_normal_gci"
        )
        regression_tree._rng_ci_ = numpy.random.default_rng(0)
        ci_low, ci_high = regression_tree._compute_ci(y, weights, None)
        assert ci_low is not None and ci_high is not None
        self.assertLessEqual(ci_low, cox_mean)
        self.assertGreaterEqual(ci_high, cox_mean)

    def test_nonpositive_y_raises(self):
        """Any y <= 0 raises ValueError during CI computation."""
        y = numpy.array([1.0, 0.0, 2.0])
        weights = numpy.ones(3)
        regression_tree = sigma._tree_regression.RegressionTree(
            ci_method="log_normal_gci"
        )
        regression_tree._rng_ci_ = numpy.random.default_rng(0)
        with self.assertRaises(ValueError):
            regression_tree._compute_ci(y, weights, None)

    def test_random_state_makes_ci_reproducible(self):
        """Two fits with the same random_state produce identical leaf CIs."""
        rng = numpy.random.default_rng(42)
        n = 100
        X = rng.standard_normal((n, 2))
        y = numpy.exp(2.0 * X[:, 0] + rng.standard_normal(n) * 0.5)
        tree1 = sigma._tree_regression.RegressionTree(
            correlation="normal",
            ci_method="log_normal_gci",
            random_state=123,
            min_splits=10,
            min_buckets=5,
        )
        tree2 = sigma._tree_regression.RegressionTree(
            correlation="normal",
            ci_method="log_normal_gci",
            random_state=123,
            min_splits=10,
            min_buckets=5,
        )
        tree1.fit(X, y)
        tree2.fit(X, y)
        self.assertEqual(len(tree1.leaves_), len(tree2.leaves_))
        for leaf1, leaf2 in zip(tree1.leaves_, tree2.leaves_):
            self.assertEqual(leaf1.ci_low, leaf2.ci_low)
            self.assertEqual(leaf1.ci_high, leaf2.ci_high)


class TestRegressionTreeGammaCi(unittest.TestCase):
    """Tests specific to ci_method='gamma'."""

    __slots__ = ()

    def test_brackets_sample_mean(self):
        """Gamma CI brackets the weighted sample mean."""
        rng = numpy.random.default_rng(4)
        y = rng.gamma(shape=2.0, scale=1.0, size=50)
        weights = numpy.ones(50)
        p_hat = float(numpy.average(y, weights=weights))
        regression_tree = sigma._tree_regression.RegressionTree(
            ci_method="gamma"
        )
        ci_low, ci_high = regression_tree._compute_ci(y, weights, None)
        assert ci_low is not None and ci_high is not None
        self.assertLessEqual(ci_low, p_hat + 1e-12)
        self.assertGreaterEqual(ci_high, p_hat - 1e-12)

    def test_zero_mean_returns_zero_point(self):
        """All-zero y yields (0.0, 0.0)."""
        y = numpy.zeros(10)
        weights = numpy.ones(10)
        regression_tree = sigma._tree_regression.RegressionTree(
            ci_method="gamma"
        )
        ci_low, ci_high = regression_tree._compute_ci(y, weights, None)
        self.assertEqual(ci_low, 0.0)
        self.assertEqual(ci_high, 0.0)

    def test_negative_y_raises(self):
        """Any y < 0 raises ValueError."""
        y = numpy.array([1.0, -0.1, 2.0])
        weights = numpy.ones(3)
        regression_tree = sigma._tree_regression.RegressionTree(
            ci_method="gamma"
        )
        with self.assertRaises(ValueError):
            regression_tree._compute_ci(y, weights, None)


class TestRegressionTreePoissonCi(unittest.TestCase):
    """Tests specific to ci_method='poisson'."""

    __slots__ = ()

    def test_brackets_sample_mean(self):
        """Poisson CI brackets the weighted sample mean on count data."""
        rng = numpy.random.default_rng(5)
        y = rng.poisson(lam=3.0, size=50).astype(float)
        weights = numpy.ones(50)
        p_hat = float(numpy.average(y, weights=weights))
        regression_tree = sigma._tree_regression.RegressionTree(
            ci_method="poisson"
        )
        ci_low, ci_high = regression_tree._compute_ci(y, weights, None)
        assert ci_low is not None and ci_high is not None
        self.assertLessEqual(ci_low, p_hat + 1e-12)
        self.assertGreaterEqual(ci_high, p_hat - 1e-12)

    def test_all_zero_y_gives_zero_lower_positive_upper(self):
        """All-zero y: ci_low == 0 and ci_high is strictly positive."""
        y = numpy.zeros(10)
        weights = numpy.ones(10)
        regression_tree = sigma._tree_regression.RegressionTree(
            ci_method="poisson"
        )
        ci_low, ci_high = regression_tree._compute_ci(y, weights, None)
        assert ci_low is not None and ci_high is not None
        self.assertEqual(ci_low, 0.0)
        self.assertGreater(ci_high, 0.0)

    def test_negative_y_raises(self):
        """Any y < 0 raises ValueError."""
        y = numpy.array([1.0, -0.5, 2.0])
        weights = numpy.ones(3)
        regression_tree = sigma._tree_regression.RegressionTree(
            ci_method="poisson"
        )
        with self.assertRaises(ValueError):
            regression_tree._compute_ci(y, weights, None)


class TestRegressionTreePoissonJeffreysCi(unittest.TestCase):
    """Tests specific to ci_method='poisson_jeffreys'."""

    __slots__ = ()

    def test_brackets_sample_mean(self):
        """Jeffreys CI brackets the weighted sample mean on count data."""
        rng = numpy.random.default_rng(5)
        y = rng.poisson(lam=3.0, size=50).astype(float)
        weights = numpy.ones(50)
        p_hat = float(numpy.average(y, weights=weights))
        regression_tree = sigma._tree_regression.RegressionTree(
            ci_method="poisson_jeffreys"
        )
        ci_low, ci_high = regression_tree._compute_ci(y, weights, None)
        assert ci_low is not None and ci_high is not None
        self.assertLessEqual(ci_low, p_hat + 1e-12)
        self.assertGreaterEqual(ci_high, p_hat - 1e-12)

    def test_all_zero_y_gives_positive_lower_bound(self):
        """All-zero y: ci_low > 0 (Jeffreys does not clamp at zero)."""
        y = numpy.zeros(10)
        weights = numpy.ones(10)
        regression_tree = sigma._tree_regression.RegressionTree(
            ci_method="poisson_jeffreys"
        )
        ci_low, ci_high = regression_tree._compute_ci(y, weights, None)
        assert ci_low is not None and ci_high is not None
        self.assertGreater(ci_low, 0.0)
        self.assertGreater(ci_high, ci_low)

    def test_negative_y_raises(self):
        """Any y < 0 raises ValueError."""
        y = numpy.array([1.0, -0.5, 2.0])
        weights = numpy.ones(3)
        regression_tree = sigma._tree_regression.RegressionTree(
            ci_method="poisson_jeffreys"
        )
        with self.assertRaises(ValueError):
            regression_tree._compute_ci(y, weights, None)

    def test_shorter_than_garwood_at_moderate_rate(self):
        """Jeffreys interval is strictly shorter than Garwood at lambda in [2, 4]."""
        rng = numpy.random.default_rng(11)
        y = rng.poisson(lam=3.0, size=10).astype(float)
        weights = numpy.ones(10)
        garwood_tree = sigma._tree_regression.RegressionTree(
            ci_method="poisson"
        )
        jeffreys_tree = sigma._tree_regression.RegressionTree(
            ci_method="poisson_jeffreys"
        )
        garwood_low, garwood_high = garwood_tree._compute_ci(y, weights, None)
        jeffreys_low, jeffreys_high = jeffreys_tree._compute_ci(
            y, weights, None
        )
        assert garwood_low is not None and garwood_high is not None
        assert jeffreys_low is not None and jeffreys_high is not None
        garwood_width = garwood_high - garwood_low
        jeffreys_width = jeffreys_high - jeffreys_low
        self.assertLess(jeffreys_width, garwood_width)


class TestRegressionTreeExponentialCi(unittest.TestCase):
    """Tests specific to ci_method='exponential'."""

    __slots__ = ()

    def test_brackets_sample_mean(self):
        """Exponential CI brackets the weighted sample mean on positive data."""
        rng = numpy.random.default_rng(6)
        y = rng.exponential(scale=2.0, size=50)
        weights = numpy.ones(50)
        p_hat = float(numpy.average(y, weights=weights))
        regression_tree = sigma._tree_regression.RegressionTree(
            ci_method="exponential"
        )
        ci_low, ci_high = regression_tree._compute_ci(y, weights, None)
        assert ci_low is not None and ci_high is not None
        self.assertLessEqual(ci_low, p_hat + 1e-12)
        self.assertGreaterEqual(ci_high, p_hat - 1e-12)

    def test_zero_mean_returns_zero_point(self):
        """All-zero y yields (0.0, 0.0)."""
        y = numpy.zeros(10)
        weights = numpy.ones(10)
        regression_tree = sigma._tree_regression.RegressionTree(
            ci_method="exponential"
        )
        ci_low, ci_high = regression_tree._compute_ci(y, weights, None)
        self.assertEqual(ci_low, 0.0)
        self.assertEqual(ci_high, 0.0)

    def test_negative_y_raises(self):
        """Any y < 0 raises ValueError."""
        y = numpy.array([1.0, -2.0, 3.0])
        weights = numpy.ones(3)
        regression_tree = sigma._tree_regression.RegressionTree(
            ci_method="exponential"
        )
        with self.assertRaises(ValueError):
            regression_tree._compute_ci(y, weights, None)


class TestRegressionTreeBetaCi(unittest.TestCase):
    """Tests specific to ci_method='beta'."""

    __slots__ = ()

    def test_brackets_sample_mean(self):
        """Beta CI brackets the weighted sample mean on [0, 1] data."""
        rng = numpy.random.default_rng(7)
        y = rng.beta(2.0, 5.0, size=50)
        weights = numpy.ones(50)
        p_hat = float(numpy.average(y, weights=weights))
        regression_tree = sigma._tree_regression.RegressionTree(
            ci_method="beta"
        )
        ci_low, ci_high = regression_tree._compute_ci(y, weights, None)
        assert ci_low is not None and ci_high is not None
        self.assertLessEqual(ci_low, p_hat + 1e-12)
        self.assertGreaterEqual(ci_high, p_hat - 1e-12)

    def test_bounds_in_unit_interval(self):
        """Beta bounds remain within [0, 1]."""
        rng = numpy.random.default_rng(8)
        y = rng.uniform(0.0, 1.0, size=30)
        weights = numpy.ones(30)
        regression_tree = sigma._tree_regression.RegressionTree(
            ci_method="beta"
        )
        ci_low, ci_high = regression_tree._compute_ci(y, weights, None)
        assert ci_low is not None and ci_high is not None
        self.assertGreaterEqual(ci_low, 0.0)
        self.assertLessEqual(ci_high, 1.0)

    def test_all_zero_y(self):
        """All y == 0 gives ci_low == 0 and ci_high > 0."""
        y = numpy.zeros(10)
        weights = numpy.ones(10)
        regression_tree = sigma._tree_regression.RegressionTree(
            ci_method="beta"
        )
        ci_low, ci_high = regression_tree._compute_ci(y, weights, None)
        assert ci_low is not None and ci_high is not None
        self.assertEqual(ci_low, 0.0)
        self.assertGreater(ci_high, 0.0)

    def test_all_one_y(self):
        """All y == 1 gives ci_high == 1 and ci_low < 1."""
        y = numpy.ones(10)
        weights = numpy.ones(10)
        regression_tree = sigma._tree_regression.RegressionTree(
            ci_method="beta"
        )
        ci_low, ci_high = regression_tree._compute_ci(y, weights, None)
        assert ci_low is not None and ci_high is not None
        self.assertEqual(ci_high, 1.0)
        self.assertLess(ci_low, 1.0)

    def test_y_outside_unit_interval_raises(self):
        """Any y outside [0, 1] raises ValueError."""
        regression_tree = sigma._tree_regression.RegressionTree(
            ci_method="beta"
        )
        for bad_y in [
            numpy.array([0.5, -0.1, 0.3]),
            numpy.array([0.5, 1.1, 0.3]),
        ]:
            with self.assertRaises(ValueError):
                regression_tree._compute_ci(bad_y, numpy.ones(3), None)


class TestClassificationTreeCiMethod(unittest.TestCase):
    """Tests for each classification tree ci_method including extreme cases."""

    __slots__ = ()

    def test_default_ci_method_is_jeffreys(self):
        """Default ci_method for the classification tree is jeffreys."""
        classification_tree = sigma._tree_classification.ClassificationTree()
        self.assertEqual(classification_tree.ci_method, "jeffreys")

    def test_ci_coverage_none_disables_per_class_ci(self):
        """ci_coverage=None returns (None, None) for every method."""
        y = numpy.array([0.0, 1.0, 0.0])
        weights = numpy.ones(3)
        for method in _helpers._CLASSIFICATION_TREE_CI_METHODS:
            classification_tree = sigma._tree_classification.ClassificationTree(
                ci_method=method, ci_coverage=None
            )
            classification_tree.n_classes_ = 2
            ci_low, ci_high = classification_tree._compute_per_class_ci(
                y, weights
            )
            self.assertIsNone(ci_low)
            self.assertIsNone(ci_high)

    def test_empty_class_has_zero_lower_bound(self):
        """w_k == 0 gives ci_low == 0 and a valid non-trivial ci_high."""
        y = numpy.array([1.0, 1.0, 1.0, 1.0])
        weights = numpy.ones(4)
        for method in _helpers._CLASSIFICATION_TREE_CI_METHODS:
            classification_tree = sigma._tree_classification.ClassificationTree(
                ci_method=method
            )
            classification_tree.n_classes_ = 2
            ci_low, ci_high = classification_tree._compute_per_class_ci(
                y, weights
            )
            assert ci_low is not None and ci_high is not None
            self.assertEqual(ci_low[0], 0.0)
            self.assertGreater(ci_high[0], 0.0)
            self.assertLess(ci_high[0], 1.0)
            self.assertEqual(ci_high[1], 1.0)

    def test_single_observation_total(self):
        """w_total = 1 with w_k = 1 gives bounds in [0, 1]."""
        y = numpy.array([0.0])
        weights = numpy.array([1.0])
        for method in _helpers._CLASSIFICATION_TREE_CI_METHODS:
            classification_tree = sigma._tree_classification.ClassificationTree(
                ci_method=method
            )
            classification_tree.n_classes_ = 2
            ci_low, ci_high = classification_tree._compute_per_class_ci(
                y, weights
            )
            assert ci_low is not None and ci_high is not None
            self.assertEqual(ci_low[1], 0.0)
            self.assertEqual(ci_high[0], 1.0)
            self.assertGreaterEqual(ci_low[0], 0.0)
            self.assertLessEqual(ci_high[1], 1.0)

    def test_two_observations_one_each_are_symmetric(self):
        """One sample per class: the two per-class CIs mirror around 0.5."""
        y = numpy.array([0.0, 1.0])
        weights = numpy.ones(2)
        for method in _helpers._CLASSIFICATION_TREE_CI_METHODS:
            classification_tree = sigma._tree_classification.ClassificationTree(
                ci_method=method
            )
            classification_tree.n_classes_ = 2
            ci_low, ci_high = classification_tree._compute_per_class_ci(
                y, weights
            )
            assert ci_low is not None and ci_high is not None
            self.assertAlmostEqual(ci_low[0], 1.0 - ci_high[1], places=10)
            self.assertAlmostEqual(ci_high[0], 1.0 - ci_low[1], places=10)

    def test_fractional_weights_are_finite_and_bracketed(self):
        """Fractional weights produce finite bounds that bracket p_hat."""
        y = numpy.array([0.0, 1.0, 0.0, 1.0])
        weights = numpy.array([0.5, 0.5, 0.3, 0.7])
        w_total = weights.sum()
        p_hat = numpy.array(
            [
                weights[y == 0].sum() / w_total,
                weights[y == 1].sum() / w_total,
            ]
        )
        for method in _helpers._CLASSIFICATION_TREE_CI_METHODS:
            classification_tree = sigma._tree_classification.ClassificationTree(
                ci_method=method
            )
            classification_tree.n_classes_ = 2
            ci_low, ci_high = classification_tree._compute_per_class_ci(
                y, weights
            )
            assert ci_low is not None and ci_high is not None
            self.assertTrue(numpy.all(numpy.isfinite(ci_low)))
            self.assertTrue(numpy.all(numpy.isfinite(ci_high)))
            self.assertTrue(numpy.all(ci_low >= 0.0))
            self.assertTrue(numpy.all(ci_high <= 1.0))
            for k in range(2):
                self.assertLessEqual(ci_low[k], p_hat[k] + 1e-12)
                self.assertGreaterEqual(ci_high[k], p_hat[k] - 1e-12)

    def test_tiny_total_weight(self):
        """Very small total weight keeps bounds in [0, 1] and finite."""
        y = numpy.array([0.0, 1.0])
        weights = numpy.array([0.005, 0.005])
        for method in _helpers._CLASSIFICATION_TREE_CI_METHODS:
            classification_tree = sigma._tree_classification.ClassificationTree(
                ci_method=method
            )
            classification_tree.n_classes_ = 2
            ci_low, ci_high = classification_tree._compute_per_class_ci(
                y, weights
            )
            assert ci_low is not None and ci_high is not None
            self.assertTrue(numpy.all(numpy.isfinite(ci_low)))
            self.assertTrue(numpy.all(numpy.isfinite(ci_high)))
            self.assertTrue(numpy.all(ci_low >= 0.0))
            self.assertTrue(numpy.all(ci_high <= 1.0))

    def test_huge_weights_concentrate_around_half(self):
        """Very large balanced weights give a tight CI around 0.5."""
        y = numpy.concatenate([numpy.zeros(1_000_000), numpy.ones(1_000_000)])
        weights = numpy.ones(2_000_000)
        for method in _helpers._CLASSIFICATION_TREE_CI_METHODS:
            classification_tree = sigma._tree_classification.ClassificationTree(
                ci_method=method
            )
            classification_tree.n_classes_ = 2
            ci_low, ci_high = classification_tree._compute_per_class_ci(
                y, weights
            )
            assert ci_low is not None and ci_high is not None
            self.assertTrue(numpy.all(numpy.isfinite(ci_low)))
            self.assertTrue(numpy.all(numpy.isfinite(ci_high)))
            for k in range(2):
                self.assertAlmostEqual(ci_low[k], 0.5, places=2)
                self.assertAlmostEqual(ci_high[k], 0.5, places=2)

    def test_all_zero_weights_gives_uninformative_interval(self):
        """w_total == 0 returns the uninformative (zeros, ones) interval."""
        y = numpy.array([0.0, 1.0, 0.0])
        weights = numpy.zeros(3)
        for method in _helpers._CLASSIFICATION_TREE_CI_METHODS:
            classification_tree = sigma._tree_classification.ClassificationTree(
                ci_method=method
            )
            classification_tree.n_classes_ = 2
            ci_low, ci_high = classification_tree._compute_per_class_ci(
                y, weights
            )
            assert ci_low is not None and ci_high is not None
            numpy.testing.assert_array_equal(ci_low, numpy.zeros(2))
            numpy.testing.assert_array_equal(ci_high, numpy.ones(2))

    def test_multiclass_empty_class(self):
        """In 3-class data, an empty class gets (0, non-trivial) bounds."""
        y = numpy.array([0.0, 0.0, 1.0, 1.0])
        weights = numpy.ones(4)
        for method in _helpers._CLASSIFICATION_TREE_CI_METHODS:
            classification_tree = sigma._tree_classification.ClassificationTree(
                ci_method=method
            )
            classification_tree.n_classes_ = 3
            ci_low, ci_high = classification_tree._compute_per_class_ci(
                y, weights
            )
            assert ci_low is not None and ci_high is not None
            self.assertEqual(ci_low[2], 0.0)
            self.assertGreater(ci_high[2], 0.0)
            self.assertLess(ci_high[2], 1.0)

    def test_bracketing_for_each_method_via_fit(self):
        """After fit, ci_low <= p_hat <= ci_high for every class on every
        node.
        """
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 1.0)
        for method in _helpers._CLASSIFICATION_TREE_CI_METHODS:
            classification_tree = sigma._tree_classification.ClassificationTree(
                correlation="normal",
                min_splits=2,
                min_buckets=1,
                ci_method=method,
            )
            classification_tree.fit(X, y)
            nodes = _helpers._collect_nodes(classification_tree.content_)
            for node in nodes:
                if node.class_distribution is None:
                    continue
                ci_low = node.ci_low
                ci_high = node.ci_high
                assert ci_low is not None and ci_high is not None
                for k in range(len(node.class_distribution)):
                    self.assertLessEqual(
                        ci_low[k], node.class_distribution[k] + 1e-12
                    )
                    self.assertGreaterEqual(
                        ci_high[k], node.class_distribution[k] - 1e-12
                    )

    def test_mid_p_exact_is_narrower_than_clopper_pearson(self):
        """Mid-p endpoints lie strictly inside Clopper-Pearson endpoints."""
        alpha = 0.025
        grid = [
            (1.0, 9.0),
            (5.0, 15.0),
            (10.0, 10.0),
            (50.0, 50.0),
            (81.0, 182.0),
        ]
        for w_k, w_rest in grid:
            cp_low, cp_high = (
                sigma._tree_classification._compute_class_ci_clopper_pearson(
                    w_k, w_rest, alpha
                )
            )
            mid_low, mid_high = (
                sigma._tree_classification._compute_class_ci_mid_p_exact(
                    w_k, w_rest, alpha
                )
            )
            self.assertGreater(mid_low, cp_low)
            self.assertLess(mid_high, cp_high)

    def test_wilson_cc_is_wider_than_wilson(self):
        """Wilson-CC endpoints lie outside the plain Wilson endpoints."""
        alpha = 0.025
        grid = [
            (1.0, 9.0),
            (5.0, 15.0),
            (10.0, 10.0),
            (50.0, 50.0),
            (81.0, 182.0),
        ]
        for w_k, w_rest in grid:
            w_total = w_k + w_rest
            w_low, w_high = sigma._tree_classification._compute_class_ci_wilson(
                w_k, w_total, alpha
            )
            cc_low, cc_high = (
                sigma._tree_classification._compute_class_ci_wilson_cc(
                    w_k, w_total, alpha
                )
            )
            self.assertLessEqual(cc_low, w_low)
            self.assertGreaterEqual(cc_high, w_high)

    def test_wilson_cc_matches_wilson_asymptotically(self):
        """Continuity correction vanishes for large w_total."""
        alpha = 0.025
        w_k = 3_000.0
        w_total = 10_000.0
        w_low, w_high = sigma._tree_classification._compute_class_ci_wilson(
            w_k, w_total, alpha
        )
        cc_low, cc_high = (
            sigma._tree_classification._compute_class_ci_wilson_cc(
                w_k, w_total, alpha
            )
        )
        self.assertLess(abs(cc_low - w_low), 1e-3)
        self.assertLess(abs(cc_high - w_high), 1e-3)

    def test_newcombe_1998_table_i_reference_values(self):
        """Wilson-CC and mid-p match Newcombe (1998) Table I across four (n, r)."""
        alpha = 0.025
        cases = [
            (263, 81, (0.2535, 0.3682), (0.2544, 0.3658)),
            (148, 15, (0.0598, 0.1644), (0.0601, 0.1581)),
            (20, 0, (0.0000, 0.2005), (0.0000, 0.1391)),
            (29, 1, (0.0018, 0.1963), (0.0017, 0.1585)),
        ]
        for n, r, (m4_low, m4_high), (m6_low, m6_high) in cases:
            w_k = float(r)
            w_rest = float(n - r)
            w_total = float(n)
            cc_low, cc_high = (
                sigma._tree_classification._compute_class_ci_wilson_cc(
                    w_k, w_total, alpha
                )
            )
            self.assertAlmostEqual(cc_low, m4_low, places=4)
            self.assertAlmostEqual(cc_high, m4_high, places=4)
            if r == 0 or r == n:
                continue
            mid_low, mid_high = (
                sigma._tree_classification._compute_class_ci_mid_p_exact(
                    w_k, w_rest, alpha
                )
            )
            self.assertAlmostEqual(mid_low, m6_low, places=4)
            self.assertAlmostEqual(mid_high, m6_high, places=4)

    def test_agresti_coull_center_is_pseudo_proportion(self):
        """Endpoint midpoint equals (w_k + z^2/2) / (w_total + z^2) when no clipping."""
        alpha = 0.025
        z = float(scipy.stats.norm.ppf(1.0 - alpha))
        z_sq = z * z
        grid = [
            (5.0, 20.0),
            (10.0, 20.0),
            (10.0, 100.0),
            (50.0, 100.0),
            (81.0, 263.0),
        ]
        for w_k, w_total in grid:
            ci_low, ci_high = (
                sigma._tree_classification._compute_class_ci_agresti_coull(
                    w_k, w_total, alpha
                )
            )
            expected_center = (w_k + z_sq / 2.0) / (w_total + z_sq)
            self.assertAlmostEqual(
                (ci_low + ci_high) / 2.0, expected_center, places=12
            )

    def test_agresti_coull_endpoints_clipped_to_unit_interval(self):
        """Endpoints lie in [0, 1] for boundary and interior cases."""
        alpha = 0.025
        cases = [
            (0.0, 5.0),
            (5.0, 5.0),
            (0.5, 1.0),
            (1.0, 100.0),
            (99.0, 100.0),
        ]
        for w_k, w_total in cases:
            ci_low, ci_high = (
                sigma._tree_classification._compute_class_ci_agresti_coull(
                    w_k, w_total, alpha
                )
            )
            self.assertGreaterEqual(ci_low, 0.0)
            self.assertLessEqual(ci_high, 1.0)
            self.assertLessEqual(ci_low, ci_high)

    def test_agresti_coull_at_zero_successes_lower_endpoint_zero(self):
        """Zero successes clip the lower endpoint to exactly 0."""
        alpha = 0.025
        for w_total in (5.0, 20.0, 100.0):
            ci_low, _ = (
                sigma._tree_classification._compute_class_ci_agresti_coull(
                    0.0, w_total, alpha
                )
            )
            self.assertEqual(ci_low, 0.0)

    def test_agresti_coull_at_full_successes_upper_endpoint_one(self):
        """Full successes clip the upper endpoint to exactly 1."""
        alpha = 0.025
        for w_total in (5.0, 20.0, 100.0):
            _, ci_high = (
                sigma._tree_classification._compute_class_ci_agresti_coull(
                    w_total, w_total, alpha
                )
            )
            self.assertEqual(ci_high, 1.0)

    def test_agresti_coull_converges_to_wilson_for_large_n(self):
        """At w_total = 10_000 Agresti-Coull matches Wilson within 1e-3."""
        alpha = 0.025
        w_k = 3_000.0
        w_total = 10_000.0
        w_low, w_high = sigma._tree_classification._compute_class_ci_wilson(
            w_k, w_total, alpha
        )
        ac_low, ac_high = (
            sigma._tree_classification._compute_class_ci_agresti_coull(
                w_k, w_total, alpha
            )
        )
        self.assertLess(abs(ac_low - w_low), 1e-3)
        self.assertLess(abs(ac_high - w_high), 1e-3)

    def test_agresti_coull_explicit_formula_at_balanced_sample(self):
        """At (w_k=50, w_total=100, alpha=0.025) endpoints match the closed form."""
        alpha = 0.025
        w_k = 50.0
        w_total = 100.0
        z = float(scipy.stats.norm.ppf(1.0 - alpha))
        z_sq = z * z
        n_tilde = w_total + z_sq
        p_tilde = (w_k + z_sq / 2.0) / n_tilde
        half = z * float(numpy.sqrt(p_tilde * (1.0 - p_tilde) / n_tilde))
        expected_low = p_tilde - half
        expected_high = p_tilde + half
        ci_low, ci_high = (
            sigma._tree_classification._compute_class_ci_agresti_coull(
                w_k, w_total, alpha
            )
        )
        self.assertAlmostEqual(ci_low, expected_low, places=12)
        self.assertAlmostEqual(ci_high, expected_high, places=12)
