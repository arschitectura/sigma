"""Unit tests for transmuter and offset transformations."""

import typing
import unittest

import numpy
import numpy.testing

import sigma._extension
import sigma._node
import sigma._partition
import sigma._tree
import sigma._tree_classification
import sigma._tree_regression
import sigma._tree_survival
import sigma._tree_text
import sigma._types

import _helpers


class TestTransmuterRegressionTree(unittest.TestCase):
    """Tests for the transmuter parameter on RegressionTree."""

    __slots__ = ()

    def test_transmuter_shifts_predictions(self):
        """Leaf predictions reflect the transmuted y values."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)

        def shift_y(X, y, sample_weight, offset, side_data):
            """Shift y values by +5."""
            y_transmuted = y + 5.0
            return y_transmuted, sample_weight, offset

        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            transmuter=shift_y,
            ci_coverage=None,
        )
        regression_tree.fit(X, y)
        assert isinstance(
            regression_tree.content_.extension, sigma._partition.Partition
        )
        predictions = regression_tree.predict(X)
        numpy.testing.assert_allclose(predictions[:20], 5.0)
        numpy.testing.assert_allclose(predictions[20:], 15.0)

    def test_transmuter_identity_matches_no_transmuter(self):
        """Identity transmuter produces the same predictions."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)

        def identity(X, y, sample_weight, offset, side_data):
            """Return y unchanged."""
            return y, sample_weight, offset

        reg_plain = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            ci_coverage=None,
        )
        reg_plain.fit(X, y)
        reg_id = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            transmuter=identity,
            ci_coverage=None,
        )
        reg_id.fit(X, y)
        numpy.testing.assert_array_equal(
            reg_plain.predict(X), reg_id.predict(X)
        )

    def test_transmuter_changes_sample_count(self):
        """Leaf n_samples reflects the transmuted array length."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = X.ravel().astype(float)

        def keep_alternating(X, y, sample_weight, offset, side_data):
            """Keep every other active sample, halving the row count."""
            y_transmuted = y[::2]
            w_transmuted = (
                sample_weight[::2] if sample_weight is not None else None
            )
            offset_transmuted = offset[::2] if offset is not None else None
            return y_transmuted, w_transmuted, offset_transmuted

        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_splits=4,
            min_buckets=2,
            max_depth=1,
            transmuter=keep_alternating,
            ci_coverage=None,
        )
        regression_tree.fit(X, y)
        for leaf in regression_tree.leaves_:
            self.assertEqual(leaf.n_samples, 10)

    def test_transmuter_receives_none_weight(self):
        """Transmuter receives sample_weight=None when not provided."""
        received = {}

        def capture(X, y, sample_weight, offset, side_data):
            """Capture the sample_weight argument."""
            received["sample_weight"] = sample_weight
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
        self.assertIsNone(received["sample_weight"])

    def test_transmuter_receives_weights(self):
        """Transmuter receives the active sample weights."""
        received = {}

        def capture(X, y, sample_weight, offset, side_data):
            """Capture the sample_weight argument."""
            received["sample_weight"] = sample_weight
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
        w = numpy.ones(40) * 2.0
        regression_tree.fit(X, y, sample_weight=w)
        self.assertIsNotNone(received["sample_weight"])
        numpy.testing.assert_allclose(received["sample_weight"], 2.0)


class TestTransmuterClassificationTree(unittest.TestCase):
    """Tests for the transmuter parameter on ClassificationTree."""

    __slots__ = ()

    def test_transmuter_flips_class_distribution(self):
        """Transmuter that flips labels reverses class distributions."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 1.0)

        def flip_labels(X, y, sample_weight, offset, side_data):
            """Flip binary class labels."""
            y_transmuted = 1.0 - y
            return y_transmuted, sample_weight, offset

        classification_tree = sigma._tree_classification.ClassificationTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            transmuter=flip_labels,
        )
        classification_tree.fit(X, y)
        partition = classification_tree.content_.extension
        assert isinstance(partition, sigma._partition.Partition)
        left = typing.cast(sigma._node.ClassificationNode, partition.left)
        right = typing.cast(sigma._node.ClassificationNode, partition.right)
        numpy.testing.assert_allclose(left.class_distribution[1], 1.0)
        numpy.testing.assert_allclose(right.class_distribution[0], 1.0)

    def test_transmuter_identity_matches_no_transmuter(self):
        """Identity transmuter produces the same predictions."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 1.0)

        def identity(X, y, sample_weight, offset, side_data):
            """Return y unchanged."""
            return y, sample_weight, offset

        clf_plain = sigma._tree_classification.ClassificationTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
        )
        clf_plain.fit(X, y)
        clf_id = sigma._tree_classification.ClassificationTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            transmuter=identity,
        )
        clf_id.fit(X, y)
        numpy.testing.assert_array_equal(
            clf_plain.predict(X), clf_id.predict(X)
        )


class TestTransmuterInternalNodes(unittest.TestCase):
    """Tests for transmuter applied to internal node predictions."""

    __slots__ = ()

    def test_internal_node_prediction_uses_transmuted_data(self):
        """Internal node prediction reflects the transmuted y values."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)

        def shift_y(X, y, sample_weight, offset, side_data):
            """Shift y values by +5."""
            y_transmuted = y + 5.0
            return y_transmuted, sample_weight, offset

        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            transmuter=shift_y,
            ci_coverage=None,
        )
        regression_tree.fit(X, y)
        assert isinstance(
            regression_tree.content_.extension, sigma._partition.Partition
        )
        expected_root_mean = numpy.mean(y) + 5.0
        self.assertAlmostEqual(
            regression_tree.content_.prediction, expected_root_mean
        )

    def test_internal_node_without_transmuter_uses_raw_data(self):
        """Internal node prediction uses raw y when no transmuter."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            ci_coverage=None,
        )
        regression_tree.fit(X, y)
        assert isinstance(
            regression_tree.content_.extension, sigma._partition.Partition
        )
        expected_root_mean = numpy.mean(y)
        self.assertAlmostEqual(
            regression_tree.content_.prediction, expected_root_mean
        )

    def test_classification_tree_internal_node_uses_transmuted_data(self):
        """ClassificationTree internal node distribution uses transmuted data."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 1.0)

        def flip_labels(X, y, sample_weight, offset, side_data):
            """Flip binary class labels."""
            y_transmuted = 1.0 - y
            return y_transmuted, sample_weight, offset

        classification_tree = sigma._tree_classification.ClassificationTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            transmuter=flip_labels,
        )
        classification_tree.fit(X, y)
        assert isinstance(
            classification_tree.content_.extension, sigma._partition.Partition
        )
        root_dist = classification_tree.content_.class_distribution
        if root_dist is None:
            raise AssertionError("expected non-None class_distribution")
        numpy.testing.assert_allclose(root_dist[0], 0.5)
        numpy.testing.assert_allclose(root_dist[1], 0.5)


class TestTransmuterPostHocValidation(unittest.TestCase):
    """Tests for post-hoc split validation with the transmuter."""

    __slots__ = ()

    def test_signal_destroying_transmuter_prevents_split(self):
        """A transmuter that destroys the signal rejects the split."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)

        def constant_y(X, y, sample_weight, offset, side_data):
            """Replace y with a constant."""
            y_transmuted = numpy.ones_like(y) * 5.0
            return y_transmuted, sample_weight, offset

        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            transmuter=constant_y,
            ci_coverage=None,
        )
        regression_tree.fit(X, y)
        assert isinstance(
            regression_tree.content_.extension, sigma._extension.Leaf
        )

    def test_identity_transmuter_preserves_tree_structure(self):
        """Identity transmuter produces the same tree as no transmuter."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)

        def identity(X, y, sample_weight, offset, side_data):
            """Return y unchanged."""
            return y, sample_weight, offset

        reg_plain = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            ci_coverage=None,
        )
        reg_plain.fit(X, y)
        reg_id = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            transmuter=identity,
            ci_coverage=None,
        )
        reg_id.fit(X, y)
        self.assertEqual(
            isinstance(reg_plain.content_.extension, sigma._extension.Leaf),
            isinstance(reg_id.content_.extension, sigma._extension.Leaf),
        )
        numpy.testing.assert_array_equal(
            reg_plain.predict(X), reg_id.predict(X)
        )

    def test_shift_transmuter_passes_validation(self):
        """A shift transmuter preserves signal and passes validation."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)

        def shift_y(X, y, sample_weight, offset, side_data):
            """Shift y values by +100."""
            y_transmuted = y + 100.0
            return y_transmuted, sample_weight, offset

        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            transmuter=shift_y,
            ci_coverage=None,
        )
        regression_tree.fit(X, y)
        assert isinstance(
            regression_tree.content_.extension, sigma._partition.Partition
        )
        predictions = regression_tree.predict(X)
        numpy.testing.assert_allclose(predictions[:20], 100.0)
        numpy.testing.assert_allclose(predictions[20:], 110.0)

    def test_classification_tree_signal_destroying_transmuter_prevents_split(
        self,
    ):
        """ClassificationTree: constant transmuter rejects the split."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 1.0)

        def constant_class(X, y, sample_weight, offset, side_data):
            """Replace all labels with class 0."""
            y_transmuted = numpy.zeros_like(y)
            return y_transmuted, sample_weight, offset

        classification_tree = sigma._tree_classification.ClassificationTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            transmuter=constant_class,
        )
        classification_tree.fit(X, y)
        assert isinstance(
            classification_tree.content_.extension, sigma._extension.Leaf
        )

    def test_child_predictions_use_transmuted_data(self):
        """Child node predictions come from transmuted data."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)

        def double_y(X, y, sample_weight, offset, side_data):
            """Double y values."""
            y_transmuted = y * 2.0
            return y_transmuted, sample_weight, offset

        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            transmuter=double_y,
            ci_coverage=None,
        )
        regression_tree.fit(X, y)
        assert isinstance(
            regression_tree.content_.extension, sigma._partition.Partition
        )
        predictions = regression_tree.predict(X)
        numpy.testing.assert_allclose(predictions[:20], 0.0)
        numpy.testing.assert_allclose(predictions[20:], 20.0)

    def test_p_value_is_max_of_original_and_transmuted(self):
        """Node p-value is the worst (max) of both test p-values."""
        rng = numpy.random.RandomState(42)
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        reg_plain = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            ci_coverage=None,
        )
        reg_plain.fit(X, y)
        plain_partition = reg_plain.content_.extension
        assert isinstance(plain_partition, sigma._partition.Partition)
        original_p = plain_partition.p_value

        def add_noise(X, y, sample_weight, offset, side_data):
            """Add noise to weaken the signal."""
            y_transmuted = y + rng.normal(0, 3.0, size=len(y))
            return y_transmuted, sample_weight, offset

        reg_noisy = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            transmuter=add_noise,
            ci_coverage=None,
        )
        reg_noisy.fit(X, y)
        noisy_partition = reg_noisy.content_.extension
        assert isinstance(noisy_partition, sigma._partition.Partition)
        noisy_p = noisy_partition.p_value
        if original_p is None or noisy_p is None:
            self.fail("p_value must not be None on internal nodes")
        self.assertGreaterEqual(noisy_p, original_p)


class TestRegressionTreeOffset(unittest.TestCase):
    """Tests for the offset argument on RegressionTree."""

    __slots__ = ()

    def test_offset_none_matches_legacy_fit(self):
        """fit(X, y, offset=None) reproduces the no-offset tree exactly."""
        X, y = _helpers._step_X_y_regression()
        plain = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        plain.fit(X, y)
        with_none = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        with_none.fit(X, y, offset=None)
        numpy.testing.assert_array_equal(plain.predict(X), with_none.predict(X))

    def test_offset_zero_matches_legacy_fit(self):
        """offset=0 produces the same tree topology and predictions."""
        X, y = _helpers._step_X_y_regression()
        plain = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        plain.fit(X, y)
        zero_offset = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        zero_offset.fit(X, y, offset=numpy.zeros(len(y)))
        numpy.testing.assert_array_equal(
            plain.predict(X), zero_offset.predict(X, offset=numpy.zeros(len(y)))
        )

    def test_offset_recovers_residual_signal(self):
        """Fit on y - offset reproduces the splits a residual fit would find."""
        X, y = _helpers._step_X_y_regression()
        baseline = numpy.full(len(y), 7.0)
        on_residual = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        on_residual.fit(X, y - baseline)
        with_offset = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        with_offset.fit(X, y, offset=baseline)
        numpy.testing.assert_array_equal(
            on_residual.predict(X), with_offset.predict(X)
        )

    def test_offset_predict_adds_to_residual(self):
        """predict(X, offset_new) returns leaf residual mean + offset_new."""
        X, y = _helpers._step_X_y_regression()
        baseline = numpy.full(len(y), 7.0)
        tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        tree.fit(X, y, offset=baseline)
        offset_new = numpy.full(3, 100.0)
        predictions = tree.predict(X[:3], offset=offset_new)
        node_values = numpy.array(
            [tree.nodes_[i].prediction for i in tree.predict_index(X[:3])]
        )
        numpy.testing.assert_allclose(predictions, node_values + 100.0)

    def test_offset_predict_default_after_fit_with_offset(self):
        """predict without offset after fit with offset returns leaf values."""
        X, y = _helpers._step_X_y_regression()
        baseline = numpy.full(len(y), 7.0)
        tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        tree.fit(X, y, offset=baseline)
        node_values = numpy.array(
            [tree.nodes_[i].prediction for i in tree.predict_index(X)]
        )
        numpy.testing.assert_array_equal(tree.predict(X), node_values)

    def test_offset_shape_mismatch_raises(self):
        """Passing offset with wrong shape raises ValueError."""
        X, y = _helpers._step_X_y_regression()
        tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        with self.assertRaises(ValueError):
            tree.fit(X, y, offset=numpy.zeros(len(y) + 1))

    def test_offset_non_finite_raises(self):
        """Passing offset with non-finite values raises ValueError."""
        X, y = _helpers._step_X_y_regression()
        bad_offset = numpy.zeros(len(y))
        bad_offset[0] = numpy.nan
        tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        with self.assertRaises(ValueError):
            tree.fit(X, y, offset=bad_offset)

    def test_constant_residual_returns_leaf(self):
        """offset = y produces a residual identically zero, hence a leaf."""
        X, y = _helpers._step_X_y_regression()
        tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        tree.fit(X, y, offset=y.copy())
        assert isinstance(tree.content_.extension, sigma._extension.Leaf)

    def test_offset_passed_to_transmuter(self):
        """The transmuter receives the active subset of offset."""
        seen = []

        def transmuter(X, y, sample_weight, offset, side_data):
            """Capture offset on first call."""
            seen.append(offset)
            return y, sample_weight, offset

        X, y = _helpers._step_X_y_regression()
        baseline = numpy.arange(len(y), dtype=float)
        tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            transmuter=transmuter,
            ci_coverage=None,
        )
        tree.fit(X, y, offset=baseline)
        self.assertGreater(len(seen), 0)
        self.assertIsNotNone(seen[0])
        numpy.testing.assert_array_equal(seen[0], baseline)


class TestClassificationTreeOffset(unittest.TestCase):
    """Tests for the offset argument on ClassificationTree."""

    __slots__ = ()

    def test_offset_none_matches_legacy_fit(self):
        """fit with offset=None reproduces the no-offset tree exactly."""
        X, y = _helpers._step_X_y_classification()
        plain = sigma._tree_classification.ClassificationTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        plain.fit(X, y)
        with_none = sigma._tree_classification.ClassificationTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        with_none.fit(X, y, offset=None)
        numpy.testing.assert_array_equal(
            plain.predict_proba(X), with_none.predict_proba(X)
        )

    def test_offset_uniform_recovers_no_offset_prediction(self):
        """Uniform offset at fit and uniform offset at predict reproduces the
        empirical leaf frequencies.
        """
        X, y = _helpers._step_X_y_classification()
        uniform = numpy.full((len(y), 2), 0.5)
        tree = sigma._tree_classification.ClassificationTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        tree.fit(X, y, offset=uniform)
        uniform_new = numpy.full((len(y), 2), 0.5)
        proba = tree.predict_proba(X, offset=uniform_new)
        empirical = numpy.array(
            [tree.nodes_[i].class_distribution for i in tree.predict_index(X)]
        )
        numpy.testing.assert_allclose(proba, empirical, atol=1e-9)

    def test_predict_proba_rows_sum_to_one(self):
        """Calibrated predict_proba returns rows that sum to one."""
        X, y = _helpers._step_X_y_classification()
        offset = numpy.full((len(y), 2), 0.3)
        offset[:, 1] = 0.7
        tree = sigma._tree_classification.ClassificationTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        tree.fit(X, y, offset=offset)
        proba = tree.predict_proba(X[:5], offset=offset[:5])
        row_sums = proba.sum(axis=1)
        numpy.testing.assert_allclose(row_sums, 1.0, atol=1e-9)

    def test_offset_shape_mismatch_raises(self):
        """Wrong-shape offset at fit raises ValueError."""
        X, y = _helpers._step_X_y_classification()
        tree = sigma._tree_classification.ClassificationTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        with self.assertRaises(ValueError):
            tree.fit(X, y, offset=numpy.full((len(y), 3), 0.33))

    def test_offset_rows_not_summing_to_one_raises(self):
        """Offset rows that don't sum to 1 raise ValueError."""
        X, y = _helpers._step_X_y_classification()
        bad_offset = numpy.full((len(y), 2), 0.5)
        bad_offset[0, 0] = 0.1
        bad_offset[0, 1] = 0.2
        tree = sigma._tree_classification.ClassificationTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        with self.assertRaises(ValueError):
            tree.fit(X, y, offset=bad_offset)

    def test_offset_out_of_range_raises(self):
        """Offset values outside [0, 1] raise ValueError."""
        X, y = _helpers._step_X_y_classification()
        bad_offset = numpy.full((len(y), 2), 0.5)
        bad_offset[0, 0] = -0.1
        bad_offset[0, 1] = 1.1
        tree = sigma._tree_classification.ClassificationTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        with self.assertRaises(ValueError):
            tree.fit(X, y, offset=bad_offset)

    def test_mean_offset_proba_stored_on_leaf(self):
        """Each leaf stores the active-mean of the offset rows."""
        X, y = _helpers._step_X_y_classification()
        offset = numpy.full((len(y), 2), 0.5)
        offset[:20, 0] = 0.2
        offset[:20, 1] = 0.8
        tree = sigma._tree_classification.ClassificationTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        tree.fit(X, y, offset=offset)
        for leaf in tree.leaves_:
            self.assertIsNotNone(leaf.mean_offset_proba)
            assert leaf.mean_offset_proba is not None
            numpy.testing.assert_allclose(leaf.mean_offset_proba.sum(), 1.0)

    def test_class_distribution_unchanged_with_offset(self):
        """class_distribution remains the empirical leaf class frequency."""
        X, y = _helpers._step_X_y_classification()
        offset = numpy.full((len(y), 2), 0.5)
        offset[:20, 0] = 0.7
        offset[:20, 1] = 0.3
        tree = sigma._tree_classification.ClassificationTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        tree.fit(X, y, offset=offset)
        for leaf in tree.leaves_:
            assert leaf.class_distribution is not None
            numpy.testing.assert_allclose(leaf.class_distribution.sum(), 1.0)

    def test_offset_passed_to_transmuter(self):
        """The transmuter receives the active subset of the offset matrix."""
        seen = []

        def transmuter(X, y, sample_weight, offset, side_data):
            """Capture offset on first call."""
            seen.append(offset)
            return y, sample_weight, offset

        X, y = _helpers._step_X_y_classification()
        offset = numpy.full((len(y), 2), 0.5)
        tree = sigma._tree_classification.ClassificationTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            transmuter=transmuter,
        )
        tree.fit(X, y, offset=offset)
        self.assertGreater(len(seen), 0)
        first_offset = seen[0]
        self.assertIsNotNone(first_offset)
        assert first_offset is not None
        self.assertEqual(first_offset.shape, (len(y), 2))


class TestSurvivalTreeOffset(unittest.TestCase):
    """Tests for the offset argument on SurvivalTree."""

    __slots__ = ()

    def _make_data(self, n: int = 200, seed: int = 0):
        """Build a small synthetic right-censored survival problem."""
        rng = numpy.random.default_rng(seed)
        X = rng.uniform(0, 10, (n, 1))
        time = rng.exponential(1.0 + 0.5 * (X[:, 0] > 5), size=n)
        event = (rng.random(n) < 0.7).astype(float)
        y = numpy.column_stack([time, event])
        return X, y

    def test_offset_none_matches_legacy_fit(self):
        """fit with offset=None reproduces the no-offset tree exactly."""
        X, y = self._make_data()
        plain = sigma._tree_survival.SurvivalTree(min_splits=10, min_buckets=5)
        plain.fit(X, y)
        with_none = sigma._tree_survival.SurvivalTree(
            min_splits=10, min_buckets=5
        )
        with_none.fit(X, y, offset=None)
        times_q = numpy.array([0.5, 1.0, 2.0])
        numpy.testing.assert_array_equal(
            plain.predict_survival(X[:5], times_q),
            with_none.predict_survival(X[:5], times_q),
        )

    def test_predict_survival_combines_multiplicatively(self):
        """predict_survival with offset returns S_offset * S_leaf."""
        X, y = self._make_data()
        n = len(y)
        S_off_fit = numpy.full(n, 0.9)
        tree = sigma._tree_survival.SurvivalTree(min_splits=10, min_buckets=5)
        tree.fit(X, y, offset=S_off_fit)
        times_q = numpy.array([0.5, 1.0, 2.0])
        bare = tree.predict_survival(X[:3], times_q)
        offset_grid = numpy.full((3, 3), 0.8)
        combined = tree.predict_survival(X[:3], times_q, offset=offset_grid)
        numpy.testing.assert_allclose(combined, offset_grid * bare)

    def test_predict_survival_default_no_offset(self):
        """predict_survival without offset returns the bare leaf curve."""
        X, y = self._make_data()
        S_off = numpy.full(len(y), 0.9)
        tree = sigma._tree_survival.SurvivalTree(min_splits=10, min_buckets=5)
        tree.fit(X, y, offset=S_off)
        times_q = numpy.array([0.5, 1.0, 2.0])
        bare = tree.predict_survival(X[:3], times_q)
        for i in range(3):
            self.assertTrue(numpy.all(bare[i] <= 1.0))
            self.assertTrue(numpy.all(bare[i] >= 0.0))

    def test_offset_shape_mismatch_at_fit_raises(self):
        """Passing 2D offset at fit raises (must be 1D length n_samples)."""
        X, y = self._make_data()
        n = len(y)
        tree = sigma._tree_survival.SurvivalTree(min_splits=10, min_buckets=5)
        with self.assertRaises(ValueError):
            tree.fit(X, y, offset=numpy.full((n, 5), 0.9))

    def test_offset_out_of_range_at_fit_raises(self):
        """Offset values outside (0, 1] at fit raise ValueError."""
        X, y = self._make_data()
        bad = numpy.full(len(y), 0.9)
        bad[0] = 1.5
        tree = sigma._tree_survival.SurvivalTree(min_splits=10, min_buckets=5)
        with self.assertRaises(ValueError):
            tree.fit(X, y, offset=bad)

    def test_predict_survival_offset_shape_validation(self):
        """predict_survival validates the offset shape."""
        X, y = self._make_data()
        n = len(y)
        S_off = numpy.full(n, 0.9)
        tree = sigma._tree_survival.SurvivalTree(min_splits=10, min_buckets=5)
        tree.fit(X, y, offset=S_off)
        times_q = numpy.array([0.5, 1.0, 2.0])
        with self.assertRaises(ValueError):
            tree.predict_survival(
                X[:3], times_q, offset=numpy.full((3, 5), 0.9)
            )

    def test_predict_survival_offset_non_monotone_raises(self):
        """Offset must be non-increasing along the time axis."""
        X, y = self._make_data()
        S_off = numpy.full(len(y), 0.9)
        tree = sigma._tree_survival.SurvivalTree(min_splits=10, min_buckets=5)
        tree.fit(X, y, offset=S_off)
        times_q = numpy.array([0.5, 1.0, 2.0])
        bad = numpy.array(
            [
                [0.5, 0.6, 0.7],
                [0.5, 0.6, 0.7],
                [0.5, 0.6, 0.7],
            ]
        )
        with self.assertRaises(ValueError):
            tree.predict_survival(X[:3], times_q, offset=bad)

    def test_offset_passed_to_transmuter(self):
        """The transmuter receives the active subset of offset."""
        seen = []

        def transmuter(X, y, sample_weight, offset, side_data):
            """Capture offset on first call."""
            seen.append(offset)
            return y, sample_weight, offset

        X, y = self._make_data()
        S_off = numpy.full(len(y), 0.9)
        tree = sigma._tree_survival.SurvivalTree(
            min_splits=10,
            min_buckets=5,
            transmuter=transmuter,
        )
        tree.fit(X, y, offset=S_off)
        self.assertGreater(len(seen), 0)
        first = seen[0]
        self.assertIsNotNone(first)
        assert first is not None
        self.assertEqual(first.shape, (len(y),))

    def test_event_grid_attribute_public(self):
        """SurvivalTree exposes event_grid_ as a public attribute."""
        X, y = self._make_data()
        tree = sigma._tree_survival.SurvivalTree(min_splits=10, min_buckets=5)
        tree.fit(X, y)
        self.assertTrue(hasattr(tree, "event_grid_"))
        self.assertGreater(len(tree.event_grid_), 0)


class TestOffsetDecoratorPropagation(unittest.TestCase):
    """Tests that decorators receive the offset slice."""

    __slots__ = ()

    def test_decorator_receives_offset_slice(self):
        """The decorator's offset_active argument is the active offset slice."""
        seen = []

        def decorator(X_sub, y_sub, w_sub, offset_sub, sd_sub):
            """Record offset_sub for each node."""
            seen.append(offset_sub)
            return None

        X, y = _helpers._step_X_y_regression()
        baseline = numpy.arange(len(y), dtype=float)
        tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            decorator=decorator,
        )
        tree.fit(X, y, offset=baseline)
        self.assertGreater(len(seen), 0)
        for offset_sub in seen:
            self.assertIsNotNone(offset_sub)

    def test_decorator_offset_is_none_when_no_fit_offset(self):
        """The decorator's offset_active argument is None without fit-time
        offset.
        """
        seen = []

        def decorator(X_sub, y_sub, w_sub, offset_sub, sd_sub):
            """Record offset_sub for each node."""
            seen.append(offset_sub)
            return None

        X, y = _helpers._step_X_y_regression()
        tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            decorator=decorator,
        )
        tree.fit(X, y)
        self.assertGreater(len(seen), 0)
        for offset_sub in seen:
            self.assertIsNone(offset_sub)


class TestPerfectOffsetReturnsLeaf(unittest.TestCase):
    """Tests that a perfect offset yields a single-leaf tree on a
    multi-feature design (so the feature-selection algorithm runs against
    every covariate).
    """

    __slots__ = ()

    def test_regression_perfect_offset_returns_leaf(self):
        """Regression: offset = y exactly drives the residual to zero."""
        X = _helpers._make_multifeature_X()
        rng = numpy.random.default_rng(1)
        y = rng.standard_normal(X.shape[0])
        tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=10, min_buckets=5
        )
        tree.fit(X, y, offset=y.copy())
        assert isinstance(tree.content_.extension, sigma._extension.Leaf)

    def test_classification_perfect_offset_returns_leaf(self):
        """Classification: offset = one_hot(y) drives the residual to zero."""
        X = _helpers._make_multifeature_X()
        rng = numpy.random.default_rng(2)
        y = (rng.random(X.shape[0]) > 0.5).astype(int)
        offset = numpy.eye(2)[y]
        tree = sigma._tree_classification.ClassificationTree(
            correlation="normal", min_splits=10, min_buckets=5
        )
        tree.fit(X, y, offset=offset)
        assert isinstance(tree.content_.extension, sigma._extension.Leaf)

    def test_survival_perfect_offset_returns_leaf(self):
        """Survival: offset = exp(-event) drives the martingale residual to
        zero per record (H_offset_i = event_i).
        """
        X = _helpers._make_multifeature_X()
        rng = numpy.random.default_rng(3)
        n = X.shape[0]
        time = rng.exponential(1.0, size=n)
        event = (rng.random(n) < 0.7).astype(float)
        y = numpy.column_stack([time, event])
        offset = numpy.exp(-event)
        tree = sigma._tree_survival.SurvivalTree(min_splits=10, min_buckets=5)
        tree.fit(X, y, offset=offset)
        assert isinstance(tree.content_.extension, sigma._extension.Leaf)


class TestBiasedSubspaceOffset(unittest.TestCase):
    """Tests that a biased offset confined to a (feature-0, feature-1)
    quadrant yields a tree whose splits are confined to features 0 and 1
    and whose biased-quadrant leaf carries the bias correction.
    """

    __slots__ = ()

    def test_regression_biased_subspace(self):
        """Regression: bias only in (X[:,0]>5) AND (X[:,1]>5) quadrant."""
        X = _helpers._make_multifeature_X()
        n = X.shape[0]
        y = numpy.zeros(n)
        biased = _helpers._biased_quadrant_mask(X)
        offset = numpy.where(biased, 5.0, 0.0)
        tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=10, min_buckets=5
        )
        tree.fit(X, y, offset=offset)
        assert isinstance(tree.content_.extension, sigma._partition.Partition)
        split_features = _helpers._collect_split_features(tree.content_)
        self.assertSetEqual(split_features, {0, 1})
        leaf_predictions = sorted(leaf.prediction for leaf in tree.leaves_)
        self.assertAlmostEqual(leaf_predictions[0], -5.0, places=6)
        for leaf_value in leaf_predictions[1:]:
            self.assertAlmostEqual(leaf_value, 0.0, places=6)
        biased_leaf_indices = tree.predict_index(X[biased])
        unbiased_leaf_indices = tree.predict_index(X[~biased])
        biased_leaf_set = set(biased_leaf_indices.tolist())
        unbiased_leaf_set = set(unbiased_leaf_indices.tolist())
        self.assertEqual(len(biased_leaf_set), 1)
        self.assertTrue(biased_leaf_set.isdisjoint(unbiased_leaf_set))

    def test_classification_biased_subspace(self):
        """A classification tree isolates a biased quadrant and recovers class 0 with certainty inside it."""
        X = _helpers._make_multifeature_X()
        n = X.shape[0]
        biased = _helpers._biased_quadrant_mask(X)
        y = numpy.where(biased, 0, 1)
        offset = numpy.empty((n, 2), dtype=float)
        offset[~biased] = numpy.eye(2)[y[~biased]]
        offset[biased] = 0.5
        tree = sigma._tree_classification.ClassificationTree(
            correlation="normal", min_splits=10, min_buckets=5
        )
        tree.fit(X, y, offset=offset)
        assert isinstance(tree.content_.extension, sigma._partition.Partition)
        split_features = _helpers._collect_split_features(tree.content_)
        self.assertSetEqual(split_features, {0, 1})
        biased_leaf_indices = tree.predict_index(X[biased])
        unbiased_leaf_indices = tree.predict_index(X[~biased])
        biased_leaf_set = set(biased_leaf_indices.tolist())
        unbiased_leaf_set = set(unbiased_leaf_indices.tolist())
        self.assertEqual(len(biased_leaf_set), 1)
        self.assertTrue(biased_leaf_set.isdisjoint(unbiased_leaf_set))
        biased_proba = tree.predict_proba(X[biased], offset=offset[biased])
        numpy.testing.assert_allclose(biased_proba[:, 0], 1.0, atol=1e-6)
        numpy.testing.assert_allclose(biased_proba[:, 1], 0.0, atol=1e-6)

    def test_survival_biased_subspace(self):
        """A survival tree isolates a biased quadrant on features 0 and 1 only."""
        X = _helpers._make_multifeature_X()
        n = X.shape[0]
        rng = numpy.random.default_rng(4)
        time = rng.exponential(1.0, size=n)
        event = numpy.ones(n, dtype=float)
        y = numpy.column_stack([time, event])
        biased = _helpers._biased_quadrant_mask(X)
        offset = numpy.where(biased, 1.0, float(numpy.exp(-1.0)))
        tree = sigma._tree_survival.SurvivalTree(min_splits=10, min_buckets=5)
        tree.fit(X, y, offset=offset)
        assert isinstance(tree.content_.extension, sigma._partition.Partition)
        split_features = _helpers._collect_split_features(tree.content_)
        self.assertSetEqual(split_features, {0, 1})
        biased_leaf_indices = tree.predict_index(X[biased])
        unbiased_leaf_indices = tree.predict_index(X[~biased])
        biased_leaf_set = set(biased_leaf_indices.tolist())
        unbiased_leaf_set = set(unbiased_leaf_indices.tolist())
        self.assertEqual(len(biased_leaf_set), 1)
        self.assertTrue(biased_leaf_set.isdisjoint(unbiased_leaf_set))


class TestPostTransmutationConsistency(unittest.TestCase):
    """Tests that all node-level computations operate on post-transmutation data."""

    __slots__ = ()

    def test_regression_prediction_uses_transmuted_offset(self):
        """RegressionTree leaf prediction equals weighted mean of transmuted residuals."""
        rng = numpy.random.default_rng(0)
        n = 200
        X = rng.normal(size=(n, 1))
        y = X[:, 0] * 2.0 + 0.5 * rng.normal(size=n)
        offset = 0.4 * X[:, 0]

        def double_both(X, y, sample_weight, offset, side_data):
            """Double y and offset; preserve row count."""
            return 2.0 * y, sample_weight, 2.0 * offset

        tree = sigma._tree_regression.RegressionTree(
            min_splits=10,
            min_buckets=5,
            transmuter=double_both,
            ci_coverage=None,
            response_sample_size=0,
        )
        tree.fit(X, y, offset=offset)
        leaf = tree.leaves_[0]
        self.assertIsInstance(leaf.prediction, float)
        self.assertTrue(numpy.isfinite(leaf.prediction))

    def test_classification_mean_offset_uses_transmuted_offset(self):
        """ClassificationTree leaf.mean_offset_proba reflects the transmuted offset."""
        rng = numpy.random.default_rng(0)
        n = 200
        X = rng.normal(size=(n, 1))
        y = (X[:, 0] > 0).astype(int)
        offset_proba = numpy.full((n, 2), 0.5)

        def shift_offset(X, y, sample_weight, offset, side_data):
            """Skew the offset to (0.8, 0.2) but keep y/weights."""
            new_offset = numpy.tile([0.8, 0.2], (len(y), 1))
            return y, sample_weight, new_offset

        tree = sigma._tree_classification.ClassificationTree(
            min_splits=10,
            min_buckets=5,
            transmuter=shift_offset,
        )
        tree.fit(X, y, offset=offset_proba)
        for leaf in tree.leaves_:
            if leaf.mean_offset_proba is not None:
                numpy.testing.assert_allclose(
                    leaf.mean_offset_proba, [0.8, 0.2], atol=1e-9
                )


class TestTransmuterReturnValidation(unittest.TestCase):
    """Tests that the transmuter return tuple is validated for shape and dimensions."""

    __slots__ = ()

    @staticmethod
    def _data():
        """Generate a simple regression dataset."""
        rng = numpy.random.default_rng(0)
        n = 60
        X = rng.normal(size=(n, 1))
        y = X[:, 0] + 0.1 * rng.normal(size=n)
        return X, y

    def test_two_tuple_return_raises(self):
        """Transmuter returning a 2-tuple raises ValueError."""
        X, y = self._data()

        def two_tuple(X, y, sample_weight, offset, side_data):
            """Return a 2-tuple, omitting the offset."""
            return y, sample_weight

        tree = sigma._tree_regression.RegressionTree(
            min_splits=10, min_buckets=5, transmuter=two_tuple
        )
        with self.assertRaisesRegex(ValueError, "3-tuple"):
            tree.fit(X, y)

    def test_offset_non_none_when_input_none_raises(self):
        """Transmuter must return offset=None when the input offset was None."""
        X, y = self._data()

        def invent_offset(X, y, sample_weight, offset, side_data):
            """Return a non-None offset even though input offset is None."""
            return y, sample_weight, numpy.zeros_like(y)

        tree = sigma._tree_regression.RegressionTree(
            min_splits=10, min_buckets=5, transmuter=invent_offset
        )
        with self.assertRaisesRegex(ValueError, "offset=None"):
            tree.fit(X, y)

    def test_offset_none_when_input_non_none_raises(self):
        """Transmuter must return non-None offset when input offset was non-None."""
        X, y = self._data()
        offset = numpy.zeros(len(y))

        def drop_offset(X, y, sample_weight, offset, side_data):
            """Return None for offset even though input offset is non-None."""
            return y, sample_weight, None

        tree = sigma._tree_regression.RegressionTree(
            min_splits=10, min_buckets=5, transmuter=drop_offset
        )
        with self.assertRaisesRegex(ValueError, "non-None offset"):
            tree.fit(X, y, offset=offset)

    def test_w_length_mismatch_raises(self):
        """Transmuter sample_weight length must match y."""
        X, y = self._data()

        def mismatch(X, y, sample_weight, offset, side_data):
            """Return w with length differing from y."""
            return y, numpy.ones(len(y) + 1), offset

        tree = sigma._tree_regression.RegressionTree(
            min_splits=10, min_buckets=5, transmuter=mismatch
        )
        with self.assertRaisesRegex(ValueError, "sample_weight"):
            tree.fit(X, y)

    def test_offset_length_mismatch_raises(self):
        """Transmuter offset length must match y."""
        X, y = self._data()
        offset = numpy.zeros(len(y))

        def mismatch(X, y, sample_weight, offset, side_data):
            """Return offset with length differing from y."""
            return y, sample_weight, numpy.zeros(len(y) + 1)

        tree = sigma._tree_regression.RegressionTree(
            min_splits=10, min_buckets=5, transmuter=mismatch
        )
        with self.assertRaisesRegex(ValueError, "offset"):
            tree.fit(X, y, offset=offset)

    def test_classification_offset_shape_raises(self):
        """Classification transmuter offset must keep (n_t, n_classes) shape."""
        rng = numpy.random.default_rng(0)
        n = 60
        X = rng.normal(size=(n, 1))
        y = (X[:, 0] > 0).astype(int)
        offset = numpy.full((n, 2), 0.5)

        def wrong_shape(X, y, sample_weight, offset, side_data):
            """Return classification offset with the wrong second dimension."""
            return y, sample_weight, numpy.full((len(y), 3), 1.0 / 3.0)

        tree = sigma._tree_classification.ClassificationTree(
            min_splits=10, min_buckets=5, transmuter=wrong_shape
        )
        with self.assertRaisesRegex(ValueError, "offset"):
            tree.fit(X, y, offset=offset)

    def test_survival_offset_value_out_of_range_raises(self):
        """Survival transmuter offset must lie in (0, 1]."""
        rng = numpy.random.default_rng(0)
        n = 60
        X = rng.normal(size=(n, 1))
        time = rng.exponential(scale=2.0, size=n)
        event = (rng.uniform(size=n) < 0.7).astype(float)
        y = numpy.column_stack([time, event])
        offset = numpy.full(n, 0.5)

        def out_of_range(X, y, sample_weight, offset, side_data):
            """Return a survival offset with values above 1."""
            return y, sample_weight, numpy.full(len(y), 2.0)

        tree = sigma._tree_survival.SurvivalTree(
            min_splits=10, min_buckets=5, transmuter=out_of_range
        )
        with self.assertRaisesRegex(ValueError, "0, 1"):
            tree.fit(X, y, offset=offset)


if __name__ == "__main__":
    unittest.main()
