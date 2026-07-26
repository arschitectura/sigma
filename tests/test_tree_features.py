"""Unit tests for sample weight, side data, decorators, Monte Carlo, and edge-case rendering."""

import unittest

import _helpers
import numpy
import numpy.testing

import sigma._extension
import sigma._node
import sigma._partition
import sigma._statistics
import sigma._tree
import sigma._tree_classification
import sigma._tree_ranking
import sigma._tree_regression
import sigma._tree_survival
import sigma._tree_text
import sigma._types


class TestSampleWeight(unittest.TestCase):
    """Tests for sample_weight support in fit()."""

    __slots__ = ()

    def test_sample_weight_wrong_length_raises(self):
        """Raises ValueError when sample_weight length mismatches n_samples."""
        X = numpy.arange(1, 21, dtype=float).reshape(-1, 1)
        y = numpy.ones(20)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal"
        )
        with self.assertRaises(ValueError):
            regression_tree.fit(X, y, sample_weight=numpy.ones(10))

    def test_sample_weight_2d_raises(self):
        """Raises ValueError when sample_weight is not 1D."""
        X = numpy.arange(1, 21, dtype=float).reshape(-1, 1)
        y = numpy.ones(20)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal"
        )
        with self.assertRaises(ValueError):
            regression_tree.fit(X, y, sample_weight=numpy.ones((20, 1)))

    def test_sample_weight_negative_raises(self):
        """Raises ValueError when sample_weight contains negative values."""
        X = numpy.arange(1, 21, dtype=float).reshape(-1, 1)
        y = numpy.ones(20)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal"
        )
        weights = numpy.ones(20)
        weights[5] = -1.0
        with self.assertRaises(ValueError):
            regression_tree.fit(X, y, sample_weight=weights)

    def test_sample_weight_zero_allowed(self):
        """Zero weights are allowed and effectively exclude the observation."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        weights = numpy.ones(40)
        weights[0] = 0.0
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            ci_coverage=None,
        )
        regression_tree.fit(X, y, sample_weight=weights)

    def test_uniform_weights_match_default(self):
        """Uniform sample_weight produces the same predictions as omitting
        it.
        """
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        reg_default = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            ci_coverage=None,
        )
        reg_weighted = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            ci_coverage=None,
        )
        reg_default.fit(X, y)
        reg_weighted.fit(X, y, sample_weight=numpy.ones(40))
        preds_default = reg_default.predict(X)
        preds_weighted = reg_weighted.predict(X)
        numpy.testing.assert_allclose(preds_weighted, preds_default)

    def test_integer_weights_match_repeated_data(self):
        """Integer weights produce the same tree as duplicating observations."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        repeat_counts = numpy.ones(40, dtype=int)
        repeat_counts[:10] = 3
        repeat_counts[30:] = 2
        X_expanded = numpy.repeat(X, repeat_counts, axis=0)
        y_expanded = numpy.repeat(y, repeat_counts)
        reg_expanded = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            ci_coverage=None,
        )
        reg_weighted = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            ci_coverage=None,
        )
        reg_expanded.fit(X_expanded, y_expanded)
        reg_weighted.fit(X, y, sample_weight=repeat_counts.astype(float))
        expanded_partition = reg_expanded.content_.extension
        weighted_partition = reg_weighted.content_.extension
        assert isinstance(
            expanded_partition, sigma._partition.NumericalPartition
        )
        assert isinstance(
            weighted_partition, sigma._partition.NumericalPartition
        )
        self.assertEqual(
            expanded_partition.feature_index,
            weighted_partition.feature_index,
        )
        self.assertAlmostEqual(
            expanded_partition.thresholds[0],
            weighted_partition.thresholds[0],
        )
        preds_expanded = reg_expanded.predict(X)
        preds_weighted = reg_weighted.predict(X)
        numpy.testing.assert_allclose(preds_weighted, preds_expanded)

    def test_sample_weight_shifts_prediction(self):
        """Heavy weights on high-value samples shift the leaf prediction."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        weights = numpy.ones(40)
        weights[20:] = 5.0
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", alpha=0.0
        )
        regression_tree.fit(X, y, sample_weight=weights)
        unweighted_mean = y.mean()
        weighted_mean = numpy.dot(weights, y) / weights.sum()
        numpy.testing.assert_allclose(
            regression_tree.content_.prediction, weighted_mean
        )
        self.assertGreater(regression_tree.content_.prediction, unweighted_mean)

    def test_classification_tree_sample_weight_shifts_majority(self):
        """Heavy weight on one class changes the majority prediction."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 1.0)
        weights = numpy.ones(40)
        weights[20:] = 10.0
        classification_tree = sigma._tree_classification.ClassificationTree(
            correlation="normal", alpha=0.0
        )
        classification_tree.fit(X, y, sample_weight=weights)
        self.assertEqual(classification_tree.content_.prediction, 1.0)

    def test_min_splits_uses_weight_sum(self):
        """Few observations with high weights pass the min_splits threshold."""
        X = numpy.arange(1, 11, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 5, 0.0, 10.0)
        weights = numpy.full(10, 10.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_buckets=1,
            ci_coverage=None,
        )
        regression_tree.fit(X, y, sample_weight=weights)
        assert isinstance(
            regression_tree.content_.extension, sigma._partition.Partition
        )

    def test_ci_with_dominant_weight_narrows_interval(self):
        """A dominant-weight observation produces a narrower CI."""
        rng = numpy.random.default_rng(42)
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = rng.standard_normal(40)
        reg_uniform = sigma._tree_regression.RegressionTree(
            correlation="normal", alpha=0.0
        )
        reg_uniform.fit(X, y)
        assert reg_uniform.content_.ci_high is not None
        assert reg_uniform.content_.ci_low is not None
        ci_width_uniform = (
            reg_uniform.content_.ci_high - reg_uniform.content_.ci_low
        )
        weights = numpy.ones(40)
        weights[0] = 1000.0
        reg_heavy = sigma._tree_regression.RegressionTree(
            correlation="normal", alpha=0.0
        )
        reg_heavy.fit(X, y, sample_weight=weights)
        assert reg_heavy.content_.ci_high is not None
        assert reg_heavy.content_.ci_low is not None
        ci_width_heavy = reg_heavy.content_.ci_high - reg_heavy.content_.ci_low
        self.assertLess(ci_width_heavy, ci_width_uniform)


class TestSideData(unittest.TestCase):
    """Tests for the side_data parameter on fit()."""

    __slots__ = ()

    def test_side_data_wrong_rows_raises(self):
        """Passing side_data with wrong row count raises ValueError."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        side_data = numpy.ones((30, 2))

        def identity(X, y, sample_weight, offset, side_data):
            """Return y unchanged."""
            return y, sample_weight, offset

        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            transmuter=identity,
            ci_coverage=None,
        )
        with self.assertRaises(ValueError):
            regression_tree.fit(X, y, side_data=side_data)

    def test_side_data_none_by_default(self):
        """Transmuter receives side_data=None when fit was called without it."""
        captured = {}

        def capture(X, y, sample_weight, offset, side_data):
            """Record whether side_data was None on the first call."""
            if "side_data" not in captured:
                captured["side_data"] = side_data
            return y, sample_weight, offset

        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            transmuter=capture,
            ci_coverage=None,
        )
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        regression_tree.fit(X, y)
        self.assertIsNone(captured["side_data"])

    def test_side_data_passed_to_transmuter(self):
        """Transmuter receives the active subset of side_data."""
        received = {}

        def capture(X, y, sample_weight, offset, side_data):
            """Capture all arguments on first call."""
            if "side_data" not in received:
                received["side_data"] = side_data
                received["n_samples"] = len(y)
            return y, sample_weight, offset

        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        side_data = numpy.arange(80, dtype=float).reshape(40, 2)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            transmuter=capture,
            ci_coverage=None,
        )
        regression_tree.fit(X, y, side_data=side_data)
        self.assertIsNotNone(received["side_data"])
        self.assertEqual(received["side_data"].shape[0], 40)
        numpy.testing.assert_array_equal(received["side_data"], side_data)

    def test_side_data_split_correctly(self):
        """Transmuter receives correctly subsetted side_data."""
        calls = []

        def capture_all(X, y, sample_weight, offset, side_data):
            """Record side_data for each call."""
            calls.append(side_data.copy() if side_data is not None else None)
            return y, sample_weight, offset

        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        side_data = numpy.arange(40, dtype=float).reshape(-1, 1)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            transmuter=capture_all,
            ci_coverage=None,
        )
        regression_tree.fit(X, y, side_data=side_data)
        assert isinstance(
            regression_tree.content_.extension, sigma._partition.Partition
        )
        self.assertGreater(len(calls), 1)
        for call_sd in calls:
            self.assertIsNotNone(call_sd)
        full_sd = calls[0]
        numpy.testing.assert_array_equal(full_sd, side_data)
        left_sd = calls[1]
        right_sd = calls[2]
        combined = numpy.sort(numpy.concatenate([left_sd, right_sd]).ravel())
        numpy.testing.assert_array_equal(combined, side_data.ravel())
        self.assertLess(len(left_sd), 40)
        self.assertLess(len(right_sd), 40)

    def test_side_data_without_transmuter_flows_to_decorator(self):
        """side_data flows to the decorator even when no transmuter is set."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        side_data = numpy.arange(40, dtype=float).reshape(-1, 1)
        seen = []

        def decorator(X_sub, y_sub, w_sub, offset_sub, sd_sub):
            """Record the side_data subset seen at each node."""
            seen.append(sd_sub)

        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            decorator=decorator,
        )
        regression_tree.fit(X, y, side_data=side_data)
        self.assertGreater(len(seen), 0)
        for sd_sub in seen:
            self.assertIsNotNone(sd_sub)


class TestTreeMonteCarlo(unittest.TestCase):
    """Integration tests for Tree with monte_carlo test_type."""

    __slots__ = ()

    def test_regression_tree_splits_correctly(self):
        """RegressionTree splits correctly with monte_carlo adjustment."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            test_type="monte_carlo",
            resamples=99,
            random_state=123,
            min_splits=2,
            min_buckets=1,
        )
        regression_tree.fit(X, y)
        assert isinstance(
            regression_tree.content_.extension, sigma._partition.Partition
        )
        assert regression_tree.content_.extension.feature_index == 0

    def test_classification_tree_splits_correctly(self):
        """ClassificationTree splits correctly with monte_carlo adjustment."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 1.0)
        classification_tree = sigma._tree_classification.ClassificationTree(
            correlation="normal",
            test_type="monte_carlo",
            resamples=99,
            random_state=123,
            min_splits=2,
            min_buckets=1,
        )
        classification_tree.fit(X, y)
        assert isinstance(
            classification_tree.content_.extension, sigma._partition.Partition
        )
        assert classification_tree.content_.extension.feature_index == 0

    def test_reproducible_with_random_state(self):
        """Two fits with same random_state yield identical predictions."""
        rng = numpy.random.default_rng(42)
        n = 100
        X = rng.standard_normal((n, 2))
        y = 2.0 * X[:, 0] + rng.standard_normal(n) * 0.5
        reg1 = sigma._tree_regression.RegressionTree(
            correlation="normal",
            test_type="monte_carlo",
            resamples=99,
            random_state=123,
            min_splits=10,
            min_buckets=5,
        )
        reg2 = sigma._tree_regression.RegressionTree(
            correlation="normal",
            test_type="monte_carlo",
            resamples=99,
            random_state=123,
            min_splits=10,
            min_buckets=5,
        )
        reg1.fit(X, y)
        reg2.fit(X, y)
        pred1 = reg1.predict(X)
        pred2 = reg2.predict(X)
        numpy.testing.assert_array_equal(pred1, pred2)

    def test_raises_monte_carlo_without_resamples(self):
        """Raises ValueError when test_type='monte_carlo' and resamples is
        None.
        """
        X = numpy.array([[1.0], [2.0], [3.0]])
        y = numpy.array([1.0, 2.0, 3.0])
        regression_tree = sigma._tree_regression.RegressionTree(
            test_type="monte_carlo"
        )
        with self.assertRaises(ValueError):
            regression_tree.fit(X, y)

    def test_raises_resamples_zero(self):
        """Raises ValueError when resamples is 0."""
        X = numpy.array([[1.0], [2.0], [3.0]])
        y = numpy.array([1.0, 2.0, 3.0])
        regression_tree = sigma._tree_regression.RegressionTree(
            test_type="monte_carlo", resamples=0
        )
        with self.assertRaises(ValueError):
            regression_tree.fit(X, y)

    def test_monte_carlo_with_transmuter(self):
        """monte_carlo test_type works with a transmuter."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)

        def identity(X, y, sample_weight, offset, side_data):
            """Return y unchanged."""
            return y, sample_weight, offset

        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            test_type="monte_carlo",
            resamples=99,
            random_state=123,
            min_splits=2,
            min_buckets=1,
            transmuter=identity,
            ci_coverage=None,
        )
        regression_tree.fit(X, y)
        assert isinstance(
            regression_tree.content_.extension, sigma._partition.Partition
        )
        self.assertEqual(regression_tree.content_.extension.feature_index, 0)

    def test_monte_carlo_transmuter_destroys_signal(self):
        """monte_carlo with signal-destroying transmuter rejects split."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)

        def constant_y(X, y, sample_weight, offset, side_data):
            """Replace y with a constant."""
            y_transmuted = numpy.ones_like(y) * 5.0
            return y_transmuted, sample_weight, offset

        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            test_type="monte_carlo",
            resamples=99,
            random_state=123,
            min_splits=2,
            min_buckets=1,
            transmuter=constant_y,
            ci_coverage=None,
        )
        regression_tree.fit(X, y)
        assert isinstance(
            regression_tree.content_.extension, sigma._extension.Leaf
        )


class TestTreeDecorator(unittest.TestCase):
    """Tests for the decorator parameter on Tree estimators."""

    __slots__ = ()

    def test_default_decoration_is_none(self):
        """Without a decorator, every node has decoration equal to None."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        regression_tree.fit(X, y)
        nodes = _helpers._collect_nodes(regression_tree.content_)
        for node in nodes:
            self.assertIsNone(node.decoration)

    def test_decorator_called_on_every_node(self):
        """A decorator is invoked once per node in the fitted tree."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        calls = []

        def decorator(X_sub, y_sub, w_sub, offset_sub, sd_sub):
            """Record the call and return a marker string."""
            calls.append(len(y_sub))
            return f"n={len(y_sub)}"

        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            decorator=decorator,
        )
        regression_tree.fit(X, y)
        nodes = _helpers._collect_nodes(regression_tree.content_)
        self.assertEqual(len(calls), len(nodes))
        for node in nodes:
            self.assertEqual(node.decoration, f"n={node.n_samples}")

    def test_decorator_receives_consistent_lengths(self):
        """All per-node subsets passed to the decorator share one length."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        sample_weight = numpy.ones(40)
        side_data = numpy.arange(40, dtype=float).reshape(-1, 1)

        def identity_transmuter(X, y, sample_weight, offset, side_data):
            """Return y and sample_weight unchanged."""
            return y, sample_weight, offset

        lengths = []

        def decorator(X_sub, y_sub, w_sub, offset_sub, sd_sub):
            """Check that all four subsets have the same first-axis length."""
            assert sd_sub is not None
            lengths.append(
                (
                    X_sub.shape[0],
                    y_sub.shape[0],
                    w_sub.shape[0],
                    sd_sub.shape[0],
                )
            )

        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            transmuter=identity_transmuter,
            decorator=decorator,
        )
        regression_tree.fit(
            X, y, sample_weight=sample_weight, side_data=side_data
        )
        self.assertGreater(len(lengths), 0)
        for nx, ny, nw, nt in lengths:
            self.assertEqual(nx, ny)
            self.assertEqual(ny, nw)
            self.assertEqual(nw, nt)

    def test_decorator_receives_raw_weights(self):
        """The weights subset is the raw sample weight restricted to the
        node.
        """
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        sample_weight = numpy.full(40, 3.5)

        seen_weights = []

        def decorator(X_sub, y_sub, w_sub, offset_sub, sd_sub):
            """Record the distinct weight values seen at this node."""
            seen_weights.append(numpy.unique(w_sub).tolist())

        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            decorator=decorator,
        )
        regression_tree.fit(X, y, sample_weight=sample_weight)
        for uniques in seen_weights:
            self.assertEqual(uniques, [3.5])

    def test_decorator_receives_none_when_no_side_data(self):
        """side_data subset is None when fit was called without it."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        observed = []

        def decorator(X_sub, y_sub, w_sub, offset_sub, sd_sub):
            """Record whether side_data subset is None."""
            observed.append(sd_sub)

        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            decorator=decorator,
        )
        regression_tree.fit(X, y)
        self.assertGreater(len(observed), 0)
        for sd_sub in observed:
            self.assertIsNone(sd_sub)

    def test_decorator_receives_raw_data_not_transmuted(self):
        """Decorator receives the raw y, not the transmuter output."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)

        def shift_transmuter(X, y, sample_weight, offset, side_data):
            """Shift y by +100 to be distinguishable from raw y."""
            y_transmuted = y + 100.0
            return y_transmuted, sample_weight, offset

        seen_maxes = []

        def decorator(X_sub, y_sub, w_sub, offset_sub, sd_sub):
            """Record max of the raw y subset."""
            seen_maxes.append(float(y_sub.max()))

        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            transmuter=shift_transmuter,
            decorator=decorator,
        )
        regression_tree.fit(X, y)
        for m in seen_maxes:
            self.assertLess(m, 100.0)

    def test_to_text_appends_decoration(self):
        """to_text shows str(decoration) in the unnamed column on every node
        line."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)

        def decorator(X_sub, y_sub, w_sub, offset_sub, sd_sub):
            """Return a fixed tag for every node."""
            return "TAG"

        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            decorator=decorator,
        )
        regression_tree.fit(X, y)

        output = regression_tree.to_text()
        lines = output.splitlines()
        header_line = lines[0]
        p_value_end = header_line.index("Split p-value") + len("Split p-value")
        leaf_index_start = header_line.index("Leaf index")
        for line in lines[2:]:
            decoration_slice = line[p_value_end:leaf_index_start]
            self.assertEqual(decoration_slice.strip(), "TAG", line)

    def test_to_text_skips_decoration_when_none(self):
        """to_text omits suffix on nodes whose decoration is None."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)

        def decorator(X_sub, y_sub, w_sub, offset_sub, sd_sub):
            """Return a tag only on the root (full dataset)."""
            if len(y_sub) == 40:
                return "ROOT"
            return None

        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            decorator=decorator,
        )
        regression_tree.fit(X, y)

        output = regression_tree.to_text()
        lines = output.strip().split("\n")
        self.assertTrue(lines[2].endswith(" ROOT"))
        for line in lines[3:]:
            self.assertFalse(line.endswith(" ROOT"), line)

    def test_decoration_applied_via_str(self):
        """Non-string decorations are rendered via str() in to_text."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)

        def decorator(X_sub, y_sub, w_sub, offset_sub, sd_sub):
            """Return the node sample count as an int, not a string."""
            count = len(y_sub)
            return count

        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            decorator=decorator,
        )
        regression_tree.fit(X, y)

        output = regression_tree.to_text()
        self.assertTrue(output.strip().split("\n")[2].endswith(" 40"))


class TestNonFiniteRendering(unittest.TestCase):
    """Tests for the "unknown" rendering of non-finite predictions and CI bounds."""

    __slots__ = ()

    def _value_formatter(self, value: float) -> str:
        """Format a value at precision=1 using the production helper."""
        return sigma._tree_text._format_value(value, precision=1)

    def test_format_value_inf_returns_unknown(self):
        """_format_value renders +inf as the literal 'unknown'."""
        rendered = sigma._tree_text._format_value(float("inf"))
        self.assertEqual(rendered, "unknown")

    def test_format_value_negative_inf_returns_unknown(self):
        """_format_value renders -inf as the literal 'unknown'."""
        rendered = sigma._tree_text._format_value(float("-inf"))
        self.assertEqual(rendered, "unknown")

    def test_format_value_nan_returns_unknown(self):
        """_format_value renders NaN as the literal 'unknown'."""
        rendered = sigma._tree_text._format_value(float("nan"))
        self.assertEqual(rendered, "unknown")

    def test_format_value_finite_unchanged(self):
        """_format_value preserves the existing format for finite values."""
        rendered = sigma._tree_text._format_value(1.5, precision=2)
        self.assertEqual(rendered, "1.50")

    def test_format_probability_inf_returns_unknown(self):
        """_format_probability renders +inf as the literal 'unknown'."""
        rendered = sigma._tree_text._format_probability(float("inf"))
        self.assertEqual(rendered, "unknown")

    def test_format_probability_nan_returns_unknown(self):
        """_format_probability renders NaN as the literal 'unknown'."""
        rendered = sigma._tree_text._format_probability(float("nan"))
        self.assertEqual(rendered, "unknown")

    def test_format_probability_finite_unchanged(self):
        """_format_probability preserves the existing percentage format."""
        rendered = sigma._tree_text._format_probability(0.75, precision=1)
        self.assertEqual(rendered, "75.0%")

    def test_format_ci_pair_both_finite(self):
        """_format_ci_pair returns ' (low to high)' when both bounds are finite."""
        rendered = sigma._tree_text._format_ci_pair(
            self._value_formatter, 1.0, 2.0
        )
        self.assertEqual(rendered, " (1.0 to 2.0)")

    def test_format_ci_pair_low_unknown(self):
        """_format_ci_pair renders the lower bound as 'unknown' when non-finite."""
        rendered = sigma._tree_text._format_ci_pair(
            self._value_formatter, float("-inf"), 2.0
        )
        self.assertEqual(rendered, " (unknown to 2.0)")

    def test_format_ci_pair_high_unknown(self):
        """_format_ci_pair renders the upper bound as 'unknown' when non-finite."""
        rendered = sigma._tree_text._format_ci_pair(
            self._value_formatter, 1.0, float("inf")
        )
        self.assertEqual(rendered, " (1.0 to unknown)")

    def test_format_ci_pair_both_non_finite(self):
        """_format_ci_pair returns the shorthand ' (unknown bounds)' when both bounds are non-finite."""
        rendered = sigma._tree_text._format_ci_pair(
            self._value_formatter, float("inf"), float("-inf")
        )
        self.assertEqual(rendered, " (unknown bounds)")

    def test_format_ci_pair_both_nan(self):
        """_format_ci_pair returns the shorthand ' (unknown bounds)' when both bounds are NaN."""
        rendered = sigma._tree_text._format_ci_pair(
            self._value_formatter, float("nan"), float("nan")
        )
        self.assertEqual(rendered, " (unknown bounds)")

    def test_survival_root_unreached_median_renders_unknown(self):
        """An unreached survival median renders as 'unknown' in to_text."""
        rng = numpy.random.default_rng(0)
        n = 80
        time = rng.uniform(1.0, 5.0, n)
        event = numpy.zeros(n)
        y = numpy.column_stack([time, event])
        X = rng.uniform(0.0, 1.0, (n, 1))
        survival_tree = sigma._tree_survival.SurvivalTree(
            min_splits=10, min_buckets=5
        )
        survival_tree.fit(X, y)
        output = survival_tree.to_text()
        self.assertIn("unknown", output)
        self.assertNotIn("= inf", output)
        self.assertNotIn(" inf ", output)
        self.assertNotIn("nan", output)


class TestNoRawTrainingDataOnFittedTree(unittest.TestCase):
    """Tests that fit() does not retain caller-provided training inputs."""

    __slots__ = ()

    def _assert_no_fit_inputs(self, tree, n_samples):
        """Assert no raw fit-time array is reachable on the fitted estimator."""
        self.assertFalse(hasattr(tree, "_sample_weight"))
        self.assertFalse(hasattr(tree, "_side_data"))
        self.assertFalse(hasattr(tree, "_offset"))
        sized_like_inputs = [
            name
            for name, value in vars(tree).items()
            if isinstance(value, numpy.ndarray)
            and value.ndim >= 1
            and value.shape[0] == n_samples
        ]
        self.assertEqual(sized_like_inputs, [])

    def test_regression_tree_does_not_retain_fit_inputs(self):
        """RegressionTree.fit leaves no per-sample fit-time array on the estimator."""
        rng = numpy.random.RandomState(0)
        n = 80
        X = rng.randn(n, 3)
        y = X[:, 0] + 0.1 * rng.randn(n)
        sample_weight = rng.uniform(0.5, 1.5, size=n)
        side_data = rng.randn(n, 2)
        offset = rng.randn(n)
        tree = sigma._tree_regression.RegressionTree(
            min_splits=10, min_buckets=5
        )
        tree.fit(
            X,
            y,
            sample_weight=sample_weight,
            offset=offset,
            side_data=side_data,
        )
        self._assert_no_fit_inputs(tree, n)

    def test_classification_tree_does_not_retain_fit_inputs(self):
        """ClassificationTree.fit leaves no per-sample fit-time array on the estimator."""
        rng = numpy.random.RandomState(0)
        n = 80
        X = rng.randn(n, 3)
        y = (X[:, 0] > 0).astype(int)
        sample_weight = rng.uniform(0.5, 1.5, size=n)
        side_data = rng.randn(n, 2)
        offset_proba = rng.uniform(0.1, 0.9, size=(n, 2))
        offset_proba = offset_proba / offset_proba.sum(axis=1, keepdims=True)
        tree = sigma._tree_classification.ClassificationTree(
            min_splits=10, min_buckets=5
        )
        tree.fit(
            X,
            y,
            sample_weight=sample_weight,
            offset=offset_proba,
            side_data=side_data,
        )
        self._assert_no_fit_inputs(tree, n)

    def test_survival_tree_does_not_retain_fit_inputs(self):
        """SurvivalTree.fit leaves no per-sample fit-time array on the estimator."""
        rng = numpy.random.RandomState(0)
        n = 80
        X = rng.randn(n, 3)
        time = rng.exponential(scale=2.0, size=n)
        event = (rng.uniform(size=n) < 0.7).astype(float)
        y = numpy.column_stack([time, event])
        sample_weight = rng.uniform(0.5, 1.5, size=n)
        side_data = rng.randn(n, 2)
        offset = rng.uniform(0.5, 1.0, size=n)
        tree = sigma._tree_survival.SurvivalTree(min_splits=10, min_buckets=5)
        tree.fit(
            X,
            y,
            sample_weight=sample_weight,
            offset=offset,
            side_data=side_data,
        )
        self._assert_no_fit_inputs(tree, n)

    def test_ranking_tree_does_not_retain_fit_inputs(self):
        """RankingTree.fit leaves no per-sample fit-time array on the estimator."""
        rng = numpy.random.RandomState(0)
        n = 80
        n_items = 5
        X = rng.randn(n, 3)
        ascending = numpy.arange(1.0, n_items + 1.0)
        descending = ascending[::-1]
        y = numpy.where((X[:, 0] > 0).reshape(-1, 1), ascending, descending)
        sample_weight = rng.uniform(0.5, 1.5, size=n)
        side_data = rng.randn(n, 2)
        tree = sigma._tree_ranking.RankingTree(
            min_splits=10, min_buckets=5, ci_coverage=None
        )
        tree.fit(X, y, sample_weight=sample_weight, side_data=side_data)
        self._assert_no_fit_inputs(tree, n)


class TestRegressionTreeResponseSamples(unittest.TestCase):
    """Tests for RegressionTree.response_sample_size and per-leaf samples."""

    __slots__ = ()

    def test_default_size_is_1000(self):
        """Default constructor sets response_sample_size to 1000."""
        tree = sigma._tree_regression.RegressionTree()
        self.assertEqual(tree.response_sample_size, 1000)

    def test_size_zero_yields_empty_arrays(self):
        """response_sample_size=0 stores an empty array on every leaf."""
        rng = numpy.random.default_rng(0)
        n = 200
        X = rng.normal(size=(n, 2))
        y = X[:, 0] * 2.0 + 0.3 * rng.normal(size=n)
        tree = sigma._tree_regression.RegressionTree(
            response_sample_size=0, min_splits=10, min_buckets=5
        )
        tree.fit(X, y)
        for leaf in tree.leaves_:
            self.assertEqual(leaf.response_samples.size, 0)

    def test_capped_at_size(self):
        """Leaves with more than N active samples store exactly N values."""
        rng = numpy.random.default_rng(0)
        n = 500
        X = rng.normal(size=(n, 1))
        y = (X[:, 0] > 0).astype(float)
        tree = sigma._tree_regression.RegressionTree(
            response_sample_size=20,
            min_splits=10,
            min_buckets=5,
            max_depth=1,
        )
        tree.fit(X, y)
        for leaf in tree.leaves_:
            if leaf.n_samples > 20:
                self.assertEqual(leaf.response_samples.size, 20)
            else:
                self.assertEqual(leaf.response_samples.size, leaf.n_samples)

    def test_residuals_when_offset_set(self):
        """Stored values are post-transmutation residuals (y - offset)."""
        rng = numpy.random.default_rng(0)
        n = 200
        X = rng.normal(size=(n, 2))
        y = X[:, 0] * 2.0 + 0.3 * rng.normal(size=n)
        offset = 0.5 * X[:, 0]
        tree = sigma._tree_regression.RegressionTree(
            response_sample_size=50, min_splits=10, min_buckets=5
        )
        tree.fit(X, y, offset=offset)
        residuals = y - offset
        for leaf in tree.leaves_:
            samples = leaf.response_samples
            self.assertTrue(
                numpy.all(numpy.isin(samples, residuals)),
                "leaf response_samples must come from y - offset",
            )

    def test_no_offset_stores_y(self):
        """Without offset, stored values are drawn from y directly."""
        rng = numpy.random.default_rng(0)
        n = 200
        X = rng.normal(size=(n, 2))
        y = X[:, 0] * 2.0 + 0.3 * rng.normal(size=n)
        tree = sigma._tree_regression.RegressionTree(
            response_sample_size=50, min_splits=10, min_buckets=5
        )
        tree.fit(X, y)
        for leaf in tree.leaves_:
            samples = leaf.response_samples
            self.assertTrue(
                numpy.all(numpy.isin(samples, y)),
                "leaf response_samples must come from y",
            )

    def test_post_transmutation(self):
        """Samples reflect post-transmutation y, not raw y."""
        rng = numpy.random.default_rng(0)
        n = 200
        X = rng.normal(size=(n, 2))
        y = X[:, 0] * 2.0 + 0.3 * rng.normal(size=n)

        def double_y(X, y, sample_weight, offset, side_data):
            """Double the response."""
            return 2.0 * y, sample_weight, offset

        tree = sigma._tree_regression.RegressionTree(
            response_sample_size=50,
            min_splits=10,
            min_buckets=5,
            transmuter=double_y,
        )
        tree.fit(X, y)
        doubled = 2.0 * y
        for leaf in tree.leaves_:
            samples = leaf.response_samples
            if samples.size == 0:
                continue
            self.assertTrue(
                numpy.all(numpy.isin(samples, doubled)),
                "leaf response_samples must come from the transmuted y",
            )

    @staticmethod
    def _construct(target_class: type, **kwargs: object) -> object:
        """Untyped wrapper that funnels kwargs into the class at runtime."""
        instance = target_class(**kwargs)
        return instance

    def test_classification_tree_rejects_response_sample_size_kwarg(self):
        """ClassificationTree does not accept response_sample_size."""
        with self.assertRaises(TypeError):
            self._construct(
                sigma._tree_classification.ClassificationTree,
                response_sample_size=10,
            )

    def test_survival_tree_rejects_response_sample_size_kwarg(self):
        """SurvivalTree does not accept response_sample_size."""
        with self.assertRaises(TypeError):
            self._construct(
                sigma._tree_survival.SurvivalTree, response_sample_size=10
            )

    def test_negative_size_raises(self):
        """response_sample_size must be a non-negative integer."""
        with self.assertRaises(ValueError):
            sigma._tree_regression.RegressionTree(response_sample_size=-1)

    def test_non_int_size_raises(self):
        """response_sample_size must be an int (not float, bool, or None)."""
        with self.assertRaises(ValueError):
            self._construct(
                sigma._tree_regression.RegressionTree, response_sample_size=10.5
            )
        with self.assertRaises(ValueError):
            sigma._tree_regression.RegressionTree(response_sample_size=True)


class TestSplitPValueMatchesVariableSelection(unittest.TestCase):
    """Tests the p-value stored on a partition is the adjusted selection one."""

    __slots__ = ()

    def _select_at_root(self, tree, X, y):
        """Run variable selection on the whole sample with the tree's settings."""
        weights = numpy.ones(X.shape[0])
        selection = sigma._statistics.select_variable(
            X,
            y.reshape(-1, 1),
            weights,
            tree.feature_types_,
            sigma._types.TestStat.QUADRATIC,
            sigma._types.TestType.SIDAK,
            tree.alpha,
            sigma._types.Correlation.NORMAL,
        )
        return selection

    def test_root_p_value_reproduced_from_selection_triple(self):
        """Round-tripping the selection T, mu, and Sigma through compute_test_statistic and compute_p_value reproduces the root partition's Sidak-adjusted p-value."""
        rng = numpy.random.default_rng(0)
        n = 200
        x_signal = numpy.linspace(0.0, 10.0, n)
        x_noise = rng.standard_normal(n)
        X = numpy.column_stack([x_noise, x_signal])
        y = 3.0 * x_signal + rng.standard_normal(n) * 0.1
        tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            random_state=123,
        )
        tree.fit(X, y)
        root_partition = tree.content_.extension
        assert isinstance(root_partition, sigma._partition.Partition)
        statistics = root_partition.statistics
        assert statistics is not None
        selection = self._select_at_root(tree, X, y)
        assert selection is not None
        self.assertEqual(selection.T.shape, selection.mu.shape)
        self.assertEqual(
            selection.Sigma.shape, (selection.T.size, selection.T.size)
        )
        c = sigma._statistics.compute_test_statistic(
            selection.T,
            selection.mu,
            selection.Sigma,
            sigma._types.TestStat.QUADRATIC,
        )
        p_raw = sigma._statistics.compute_p_value(
            c, selection.Sigma, sigma._types.TestStat.QUADRATIC
        )
        p_adj = 1.0 - (1.0 - p_raw) ** tree.n_features_in_
        numpy.testing.assert_allclose(p_adj, statistics.p_value, rtol=1e-10)

    def test_categorical_split_p_value_matches_selection(self):
        """A categorical split stores the p-value that variable selection reports for that covariate."""
        rng = numpy.random.default_rng(7)
        n = 300
        categories = rng.integers(0, 3, size=n).astype(float)
        y = numpy.where(
            categories == 0,
            10.0,
            numpy.where(categories == 1, 20.0, 30.0),
        )
        y = y + rng.standard_normal(n) * 0.5
        X = categories.reshape(-1, 1)
        tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            categorical_features=[0],
            random_state=123,
        )
        tree.fit(X, y)
        root_partition = tree.content_.extension
        assert isinstance(root_partition, sigma._partition.CategoricalPartition)
        statistics = root_partition.statistics
        assert statistics is not None
        selection = self._select_at_root(tree, X, y)
        assert selection is not None
        self.assertEqual(selection.feature_index, 0)
        numpy.testing.assert_allclose(
            selection.p_value, statistics.p_value, rtol=1e-10
        )

    def test_partition_carries_no_test_moments(self):
        """A fitted partition's statistics expose the p-value only, not the T, mu, and Sigma of the selection test."""
        rng = numpy.random.default_rng(0)
        n = 200
        X = rng.standard_normal((n, 2))
        y = 3.0 * X[:, 1] + rng.standard_normal(n) * 0.1
        tree = sigma._tree_regression.RegressionTree(random_state=123)
        tree.fit(X, y)
        root_partition = tree.content_.extension
        assert isinstance(root_partition, sigma._partition.Partition)
        statistics = root_partition.statistics
        assert statistics is not None
        for name in ("T", "mu", "Sigma"):
            self.assertFalse(hasattr(statistics, name))
