"""Unit tests for SurvivalTree (gated on the optional lifelines dependency)."""

import typing
import unittest

import numpy
import numpy.testing
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing

import sigma._extension
import sigma._node
import sigma._partition
import sigma._tree
import sigma._tree_classification
import sigma._tree_regression
import sigma._tree_survival
import sigma._tree_text
import sigma._types

try:
    import lifelines  # noqa: F401

    _HAS_LIFELINES = True
except ImportError:
    _HAS_LIFELINES = False


def _build_survival_dataset(n=200, seed=0):
    """Return (X, time, event) for a two-arm right-censored survival dataset."""
    rng = numpy.random.RandomState(seed)
    arm = (numpy.arange(n) % 2).astype(float)
    scales = numpy.where(arm == 0, 10.0, 2.0)
    survival = rng.exponential(scale=scales)
    time = numpy.minimum(survival, 8.0)
    event = (survival <= 8.0).astype(float)
    X = numpy.column_stack([arm, rng.randn(n)])
    return X, time, event


_NAN = float("nan")

# Median, risk score, survival at 3 and RMST at 4 for every node of the
# TestSurvivalMetricReferenceValues tree, in pre-order.
_REFERENCE_METRIC_VALUES = (
    2.5118615259316757,
    78.32067613395219,
    0.4749999999999997,
    2.5013268981208094,
    _NAN,
    30.84618766047522,
    0.7499999999999998,
    3.3360664285643655,
    1.3484891658185798,
    183.12638529115924,
    0.1999999999999998,
    1.6665873676772545,
)
_REFERENCE_METRIC_CI_LOW = (
    2.060320490552345,
    _NAN,
    0.40438874078061354,
    2.294843557653899,
    5.755191991686398,
    _NAN,
    0.6529041352837425,
    3.1091818096970787,
    1.0100375021560113,
    _NAN,
    0.12831194858520037,
    1.4105970533681886,
)
_REFERENCE_METRIC_CI_HIGH = (
    3.767857799758713,
    _NAN,
    0.5422054247160537,
    2.70781023858772,
    _NAN,
    _NAN,
    0.8235537160360589,
    3.5629510474316524,
    1.7874581482546053,
    _NAN,
    0.28322004379451643,
    1.9225776819863203,
)


def _survival_leaf(values):
    """Build a survival leaf carrying the given per-metric values."""
    undefined = numpy.full(len(values), numpy.nan, dtype=float)
    curve_times = numpy.array([1.0])
    curve_surv = numpy.array([1.0])
    metric_values = numpy.array(values, dtype=float)
    leaf = sigma._node.SurvivalNode(
        depth=1,
        n_samples=10,
        predicted_survival=(curve_times, curve_surv),
        survival_log_variance=numpy.zeros(1, dtype=float),
        predicted_metrics=metric_values,
        ci_low=undefined,
        ci_high=undefined.copy(),
    )
    return leaf


def _flatten_metric_field(nodes, field):
    """Return one per-metric array of every node, concatenated in node order."""
    flat = []
    for node in nodes:
        entries = getattr(node, field)
        flat.extend(float(entry) for entry in entries)
    return flat


class TestSurvivalTreeFit(unittest.TestCase):
    """Tests for the fit method of SurvivalTree."""

    __slots__ = ()

    def test_step_function_splits_correctly(self):
        """Splits on the binary covariate that flips the hazard."""
        rng = numpy.random.RandomState(0)
        n = 200
        arm = (numpy.arange(n) % 2).astype(float)
        scales = numpy.where(arm == 0, 10.0, 2.0)
        survival = rng.exponential(scale=scales)
        time = numpy.minimum(survival, 8.0)
        event = (survival <= 8.0).astype(float)
        X = numpy.column_stack([arm, rng.randn(n)])
        y = numpy.column_stack([time, event])
        estimator = sigma._tree_survival.SurvivalTree(
            min_splits=10, min_buckets=5, max_depth=3
        )
        estimator.fit(X, y)
        partition = estimator.content_.extension
        assert isinstance(partition, sigma._partition.Partition)
        self.assertEqual(partition.feature_index, 0)
        self.assertEqual(len(estimator.leaves_), 2)

    def test_alpha_zero_produces_single_leaf(self):
        """Returns a single leaf when alpha=0.0 rejects nothing."""
        rng = numpy.random.RandomState(0)
        n = 50
        time = rng.exponential(size=n)
        event = numpy.ones(n)
        X = rng.randn(n, 2)
        y = numpy.column_stack([time, event])
        estimator = sigma._tree_survival.SurvivalTree(alpha=0.0)
        estimator.fit(X, y)
        self.assertIsInstance(
            estimator.content_.extension, sigma._extension.Leaf
        )

    def test_constant_response_returns_leaf(self):
        """Returns a leaf when every subject is censored at the same time."""
        n = 30
        time = numpy.full(n, 5.0)
        event = numpy.zeros(n)
        X = numpy.arange(n, dtype=float).reshape(-1, 1)
        y = numpy.column_stack([time, event])
        estimator = sigma._tree_survival.SurvivalTree(
            min_splits=2, min_buckets=1
        )
        estimator.fit(X, y)
        self.assertIsInstance(
            estimator.content_.extension, sigma._extension.Leaf
        )

    def test_predict_returns_median_per_leaf(self):
        """predict returns the leaf median for each input sample."""
        rng = numpy.random.RandomState(0)
        n = 200
        arm = (numpy.arange(n) % 2).astype(float)
        scales = numpy.where(arm == 0, 10.0, 2.0)
        survival = rng.exponential(scale=scales)
        time = numpy.minimum(survival, 8.0)
        event = (survival <= 8.0).astype(float)
        X = arm.reshape(-1, 1)
        y = numpy.column_stack([time, event])
        estimator = sigma._tree_survival.SurvivalTree(
            min_splits=10, min_buckets=5, max_depth=2
        )
        estimator.fit(X, y)
        predictions = estimator.predict(X[:6])
        leaf_medians = [leaf.predicted_metrics[0] for leaf in estimator.leaves_]
        any_leaf_nan = any(numpy.isnan(median) for median in leaf_medians)
        for prediction in predictions:
            if numpy.isnan(prediction):
                self.assertTrue(any_leaf_nan)
            else:
                self.assertIn(prediction, leaf_medians)

    def test_predict_survival_shape_and_monotonicity(self):
        """predict_survival returns non-increasing rows in [0, 1]."""
        rng = numpy.random.RandomState(0)
        n = 200
        arm = (numpy.arange(n) % 2).astype(float)
        scales = numpy.where(arm == 0, 10.0, 2.0)
        survival = rng.exponential(scale=scales)
        time = numpy.minimum(survival, 8.0)
        event = (survival <= 8.0).astype(float)
        X = arm.reshape(-1, 1)
        y = numpy.column_stack([time, event])
        estimator = sigma._tree_survival.SurvivalTree(
            min_splits=10, min_buckets=5, max_depth=2
        )
        estimator.fit(X, y)
        times = numpy.array([0.0, 1.0, 2.0, 5.0, 10.0])
        surv = estimator.predict_survival(X[:5], times)
        self.assertEqual(surv.shape, (5, 5))
        self.assertTrue(numpy.all(surv >= 0.0))
        self.assertTrue(numpy.all(surv <= 1.0))
        differences = numpy.diff(surv, axis=1)
        self.assertTrue(numpy.all(differences <= 1e-12))

    def test_2d_y_validation_rejects_one_column(self):
        """Rejects y with a single column."""
        X = numpy.arange(20, dtype=float).reshape(-1, 1)
        y = numpy.ones((20, 1))
        estimator = sigma._tree_survival.SurvivalTree()
        with self.assertRaisesRegex(ValueError, "two columns"):
            estimator.fit(X, y)

    def test_validation_rejects_negative_time(self):
        """Rejects y with negative observed times."""
        X = numpy.arange(20, dtype=float).reshape(-1, 1)
        time = numpy.linspace(-1.0, 5.0, 20)
        event = numpy.ones(20)
        y = numpy.column_stack([time, event])
        estimator = sigma._tree_survival.SurvivalTree()
        with self.assertRaisesRegex(ValueError, "non-negative"):
            estimator.fit(X, y)

    def test_validation_rejects_event_not_binary(self):
        """Rejects y with event indicators outside {0, 1}."""
        X = numpy.arange(20, dtype=float).reshape(-1, 1)
        time = numpy.linspace(1.0, 20.0, 20)
        event = numpy.full(20, 2.0)
        y = numpy.column_stack([time, event])
        estimator = sigma._tree_survival.SurvivalTree()
        with self.assertRaisesRegex(ValueError, "0 and 1"):
            estimator.fit(X, y)

    def test_leaf_carries_predicted_survival(self):
        """Each survival leaf stores a (times, surv) tuple, not None."""
        rng = numpy.random.RandomState(0)
        n = 60
        time = rng.exponential(size=n)
        event = (rng.uniform(size=n) < 0.7).astype(float)
        X = rng.randn(n, 2)
        y = numpy.column_stack([time, event])
        estimator = sigma._tree_survival.SurvivalTree(
            min_splits=10, min_buckets=5
        )
        estimator.fit(X, y)
        for leaf in estimator.leaves_:
            assert leaf.predicted_survival is not None
            leaf_times, leaf_surv = leaf.predicted_survival
            self.assertEqual(len(leaf_times), len(leaf_surv))
            if len(leaf_times) > 1:
                self.assertTrue(numpy.all(numpy.diff(leaf_surv) <= 1e-12))


class TestSurvivalTreeYEncodings(unittest.TestCase):
    """Tests for the three accepted y encodings on fit and score."""

    __slots__ = ()

    def _build_dataset(self, n=200, seed=0):
        """Return (X, time, event) for a two-arm right-censored survival dataset."""
        dataset = _build_survival_dataset(n=n, seed=seed)
        return dataset

    def test_fit_accepts_structured_y(self):
        """Fits on the scikit-survival structured (time, event) array."""
        X, time, event = self._build_dataset()
        y_struct = numpy.array(
            list(zip(time, event.astype(bool))),
            dtype=[("time", float), ("event", bool)],
        )
        baseline = sigma._tree_survival.SurvivalTree(
            min_splits=10, min_buckets=5, max_depth=3
        )
        baseline.fit(X, numpy.column_stack([time, event]))
        coerced = sigma._tree_survival.SurvivalTree(
            min_splits=10, min_buckets=5, max_depth=3
        )
        coerced.fit(X, y_struct)
        self.assertEqual(len(coerced.leaves_), len(baseline.leaves_))
        baseline_partition = baseline.content_.extension
        coerced_partition = coerced.content_.extension
        assert isinstance(baseline_partition, sigma._partition.Partition)
        assert isinstance(coerced_partition, sigma._partition.Partition)
        self.assertEqual(
            coerced_partition.feature_index, baseline_partition.feature_index
        )

    def test_fit_accepts_1d_signed_age_y(self):
        """Fits on the 1D age-encoded y where sign carries the event indicator."""
        X, time, event = self._build_dataset()
        y_signed = numpy.where(event == 1.0, time, -time)
        baseline = sigma._tree_survival.SurvivalTree(
            min_splits=10, min_buckets=5, max_depth=3
        )
        baseline.fit(X, numpy.column_stack([time, event]))
        coerced = sigma._tree_survival.SurvivalTree(
            min_splits=10, min_buckets=5, max_depth=3
        )
        coerced.fit(X, y_signed)
        self.assertEqual(len(coerced.leaves_), len(baseline.leaves_))
        baseline_partition = baseline.content_.extension
        coerced_partition = coerced.content_.extension
        assert isinstance(baseline_partition, sigma._partition.Partition)
        assert isinstance(coerced_partition, sigma._partition.Partition)
        self.assertEqual(
            coerced_partition.feature_index, baseline_partition.feature_index
        )

    def test_score_accepts_all_three_y_encodings(self):
        """Returns the same concordance for the 2D, structured, and 1D forms."""
        X, time, event = self._build_dataset()
        y_2d = numpy.column_stack([time, event])
        y_struct = numpy.array(
            list(zip(time, event.astype(bool))),
            dtype=[("time", float), ("event", bool)],
        )
        y_signed = numpy.where(event == 1.0, time, -time)
        estimator = sigma._tree_survival.SurvivalTree(
            min_splits=10, min_buckets=5, max_depth=2
        )
        estimator.fit(X, y_2d)
        score_2d = estimator.score(X, y_2d)
        score_struct = estimator.score(X, y_struct)
        score_signed = estimator.score(X, y_signed)
        self.assertEqual(score_struct, score_2d)
        self.assertEqual(score_signed, score_2d)


class TestSurvivalTreeMetrics(unittest.TestCase):
    """Tests for the configurable per-node metrics on SurvivalTree."""

    __slots__ = ()

    def _build_dataset(self):
        """Return (X, y) for a simple two-arm survival dataset."""
        X, time, event = _build_survival_dataset()
        y = numpy.column_stack([time, event])
        return X, y

    def test_default_metrics_match_legacy_behavior(self):
        """Default metrics=('median',) preserves the legacy median display."""
        X, y = self._build_dataset()
        estimator = sigma._tree_survival.SurvivalTree(
            min_splits=10, min_buckets=5, max_depth=2
        )
        estimator.fit(X, y)
        labels = [metric.label for metric in estimator.metrics_]
        self.assertEqual(labels, ["Median survival"])
        for leaf in estimator.leaves_:
            n_metrics = len(leaf.predicted_metrics)
            self.assertEqual(n_metrics, 1)

    def test_multiple_metrics_render_each_on_own_line(self):
        """Multi-metric trees expose one descriptor per configured metric."""
        X, y = self._build_dataset()
        estimator = sigma._tree_survival.SurvivalTree(
            min_splits=10,
            min_buckets=5,
            max_depth=2,
            metrics=(
                "median",
                ("survival", 5.0, "units"),
                "risk_score",
            ),
        )
        estimator.fit(X, y)
        labels = [metric.label for metric in estimator.metrics_]
        self.assertEqual(
            labels, ["Median survival", "Survival at 5 units", "Risk score"]
        )
        for leaf in estimator.leaves_:
            n_metrics = len(leaf.predicted_metrics)
            self.assertEqual(n_metrics, 3)

    def test_first_metric_drives_predict(self):
        """predict returns the reached node's first metric, not its median."""
        X, y = self._build_dataset()
        estimator = sigma._tree_survival.SurvivalTree(
            min_splits=10,
            min_buckets=5,
            max_depth=2,
            metrics=("risk_score", "median"),
        )
        estimator.fit(X, y)
        indices = estimator.predict_index(X[:5])
        predictions = estimator.predict(X[:5])
        for prediction, index in zip(predictions, indices):
            node = estimator.nodes_[index]
            self.assertEqual(prediction, node.predicted_metrics[0])
            self.assertTrue(numpy.isfinite(prediction))

    def test_survival_value_matches_step_curve(self):
        """('survival', t, ...) value equals predict_survival at t."""
        X, y = self._build_dataset()
        estimator = sigma._tree_survival.SurvivalTree(
            min_splits=10,
            min_buckets=5,
            max_depth=2,
            metrics=(("survival", 3.0, "units"),),
        )
        estimator.fit(X, y)
        from_metric = estimator.predict(X[:6])
        from_curve = estimator.predict_survival(X[:6], numpy.array([3.0]))[:, 0]
        numpy.testing.assert_allclose(from_metric, from_curve)

    def test_text_render_emits_each_metric_label(self):
        """The text rendering emits one line per metric."""
        X, y = self._build_dataset()
        estimator = sigma._tree_survival.SurvivalTree(
            min_splits=10,
            min_buckets=5,
            max_depth=2,
            metrics=("median", ("survival", 5.0, "years")),
        )
        estimator.fit(X, y)
        output = estimator.to_text(feature_names=["arm", "noise"])
        self.assertIn("Median survival", output)
        self.assertIn("Survival at 5 years", output)

    def test_text_render_capitalizes_lowercase_response_name(self):
        """Lowercase response_name renders 'Median {Capitalized}' in survival."""
        X, y = self._build_dataset()
        estimator = sigma._tree_survival.SurvivalTree(
            min_splits=10,
            min_buckets=5,
            max_depth=2,
        )
        estimator.fit(X, y)
        output = estimator.to_text(
            feature_names=["arm", "noise"], response_name="time"
        )
        self.assertIn("Median Time", output)
        self.assertNotIn("Median time", output)

    def test_rejects_unknown_metric_kind(self):
        """Fit rejects unknown metric kinds."""
        X, y = self._build_dataset()
        estimator = sigma._tree_survival.SurvivalTree(metrics=("not_a_metric",))
        with self.assertRaisesRegex(ValueError, "unknown metric kind"):
            estimator.fit(X, y)

    def test_rejects_parametrized_metric_without_tuple(self):
        """Fit rejects ('survival',) without the value/unit pair."""
        X, y = self._build_dataset()
        estimator = sigma._tree_survival.SurvivalTree(metrics=("survival",))
        with self.assertRaisesRegex(
            ValueError, "requires a \\(kind, value, unit\\) tuple"
        ):
            estimator.fit(X, y)

    def test_rejects_parameter_free_metric_with_tuple(self):
        """Fit rejects ('median', 5, 'years') because median is parameter-free."""
        X, y = self._build_dataset()
        estimator = sigma._tree_survival.SurvivalTree(
            metrics=(("median", 5.0, "years"),)
        )
        with self.assertRaisesRegex(ValueError, "does not take parameters"):
            estimator.fit(X, y)

    def test_rejects_string_argument(self):
        """Fit rejects a bare string for metrics with a helpful TypeError."""
        X, y = self._build_dataset()
        estimator = sigma._tree_survival.SurvivalTree(metrics="median")
        with self.assertRaisesRegex(
            TypeError, "metrics must be a sequence of metric specs"
        ):
            estimator.fit(X, y)

    def test_rejects_empty_metrics(self):
        """Fit rejects an empty metrics sequence with ValueError."""
        X, y = self._build_dataset()
        estimator = sigma._tree_survival.SurvivalTree(metrics=())
        with self.assertRaisesRegex(
            ValueError, "must contain at least one entry"
        ):
            estimator.fit(X, y)

    def test_risk_score_first_orders_highest_hazard_first(self):
        """metrics=('risk_score', ...) sorts the highest-hazard leaf first."""
        X, y = self._build_dataset()
        estimator = sigma._tree_survival.SurvivalTree(
            min_splits=10,
            min_buckets=5,
            max_depth=2,
            metrics=("risk_score",),
        )
        estimator.fit(X, y)
        assert len(estimator.leaves_) >= 2
        risk_scores = [leaf.predicted_metrics[0] for leaf in estimator.leaves_]
        for k in range(len(risk_scores) - 1):
            self.assertGreaterEqual(risk_scores[k], risk_scores[k + 1])

    def test_median_first_orders_lowest_median_first(self):
        """metrics=('median',) sorts the lowest-median leaf first."""
        X, y = self._build_dataset()
        estimator = sigma._tree_survival.SurvivalTree(
            min_splits=10, min_buckets=5, max_depth=2
        )
        estimator.fit(X, y)
        assert len(estimator.leaves_) >= 2
        medians = [leaf.predicted_metrics[0] for leaf in estimator.leaves_]
        ordered = [
            float("inf") if numpy.isnan(median) else median
            for median in medians
        ]
        for k in range(len(ordered) - 1):
            self.assertLessEqual(ordered[k], ordered[k + 1])

    def test_each_specification_form_parses_to_its_class(self):
        """Every metric spec resolves to the descriptor class of its kind."""
        X, y = self._build_dataset()
        estimator = sigma._tree_survival.SurvivalTree(
            min_splits=10,
            min_buckets=5,
            max_depth=2,
            metrics=(
                "median",
                "risk_score",
                ("survival", 3.0, "units"),
                ("rmst", 4.0, "units"),
            ),
        )
        estimator.fit(X, y)
        classes = [type(metric) for metric in estimator.metrics_]
        self.assertEqual(
            classes,
            [
                sigma.MedianSurvivalMetric,
                sigma.RiskScoreMetric,
                sigma.SurvivalAtMetric,
                sigma.RmstMetric,
            ],
        )

    def test_parametrized_metrics_carry_their_time(self):
        """The parsed reference time and horizon reach their descriptors."""
        X, y = self._build_dataset()
        estimator = sigma._tree_survival.SurvivalTree(
            min_splits=10,
            min_buckets=5,
            max_depth=2,
            metrics=(("survival", 3.0, "units"), ("rmst", 4.0, "units")),
        )
        estimator.fit(X, y)
        survival_at, rmst = estimator.metrics_
        assert isinstance(survival_at, sigma.SurvivalAtMetric)
        assert isinstance(rmst, sigma.RmstMetric)
        self.assertEqual(survival_at.time, 3.0)
        self.assertEqual(rmst.horizon, 4.0)

    def test_pickle_round_trip_preserves_descriptor_classes(self):
        """A round-tripped estimator reports the same metric classes."""
        import pickle

        X, y = self._build_dataset()
        estimator = sigma._tree_survival.SurvivalTree(
            min_splits=10,
            min_buckets=5,
            max_depth=2,
            metrics=("median", ("survival", 3.0, "units")),
        )
        estimator.fit(X, y)
        payload = pickle.dumps(estimator)
        restored = pickle.loads(payload)
        original = [type(metric) for metric in estimator.metrics_]
        observed = [type(metric) for metric in restored.metrics_]
        self.assertEqual(observed, original)
        numpy.testing.assert_array_equal(
            restored.predict(X), estimator.predict(X)
        )

    def test_dumps_copy_and_compact_leave_the_original_intact(self):
        """Serializing, copying, or compacting does not strip estimator state."""
        import copy
        import pickle

        X, y = self._build_dataset()
        estimator = sigma._tree_survival.SurvivalTree(
            min_splits=10,
            min_buckets=5,
            max_depth=2,
            metrics=("median", "risk_score"),
        )
        estimator.fit(X, y)
        expected = sorted(estimator.__dict__)
        pickle.dumps(estimator)
        copy.copy(estimator)
        estimator.compact()
        self.assertEqual(sorted(estimator.__dict__), expected)
        self.assertEqual(len(estimator.metrics_), 2)

    def test_set_params_metrics_invalidates_cached_parse(self):
        """set_params(metrics=...) followed by re-fit honors the new metric."""
        X, y = self._build_dataset()
        estimator = sigma._tree_survival.SurvivalTree(
            min_splits=10,
            min_buckets=5,
            max_depth=2,
            metrics=(("rmst", 5.0, "y"),),
        )
        estimator.fit(X, y)
        estimator.predict(X)
        estimator.set_params(metrics=("median",))
        estimator.fit(X, y)
        observed = estimator.predict(X)
        baseline = sigma._tree_survival.SurvivalTree(
            min_splits=10,
            min_buckets=5,
            max_depth=2,
        )
        baseline.fit(X, y)
        expected = baseline.predict(X)
        numpy.testing.assert_array_equal(observed, expected)

    def test_lexicographic_tie_break_uses_secondary_metric(self):
        """With two metrics, a tie on the first falls through to the second."""
        metrics = (
            sigma.MedianSurvivalMetric("Median survival", False),
            sigma.SurvivalAtMetric("Survival at 5 units", False, 5.0),
        )
        leaf_a = _survival_leaf([float("inf"), 0.7])
        leaf_b = _survival_leaf([float("inf"), 0.4])
        key_a = leaf_a.leaf_sort_key(metrics)
        key_b = leaf_b.leaf_sort_key(metrics)
        self.assertLess(key_b, key_a)


class TestSurvivalLogVariance(unittest.TestCase):
    """Tests where the Greenwood log-variance is available after fit."""

    __slots__ = ()

    def _fit(self):
        """Fit a two-arm survival tree that splits at least once."""
        X, time, event = _build_survival_dataset()
        y = numpy.column_stack([time, event])
        tree = sigma._tree_survival.SurvivalTree(random_state=0)
        tree.fit(X, y)
        return tree

    def test_leaves_carry_variance_aligned_with_their_curve(self):
        """Each leaf's log-variance has one entry per time of its survival curve."""
        tree = self._fit()
        self.assertGreater(len(tree.leaves_), 1)
        for leaf in tree.leaves_:
            times, _ = leaf.predicted_survival
            self.assertEqual(leaf.survival_log_variance.shape, times.shape)

    def test_internal_nodes_carry_no_variance(self):
        """Internal nodes carry an empty log-variance array."""
        tree = self._fit()
        leaf_ids = {id(leaf) for leaf in tree.leaves_}
        internal = [node for node in tree.nodes_ if id(node) not in leaf_ids]
        self.assertGreater(len(internal), 0)
        for node in internal:
            self.assertEqual(node.survival_log_variance.size, 0)

    def test_response_image_renders_from_leaf_variance(self):
        """The response plot still draws its Greenwood band with leaf-only variance."""
        tree = self._fit()
        payload = tree.to_image("png", kind="response")
        self.assertGreater(len(payload), 0)


class TestSurvivalMetricReferenceValues(unittest.TestCase):
    """Tests that per-node metric records reproduce known-good numbers."""

    __slots__ = ()

    def _fit(self):
        """Fit a survival tree carrying one record of every metric kind."""
        X, time, event = _build_survival_dataset()
        y = numpy.column_stack([time, event])
        estimator = sigma._tree_survival.SurvivalTree(
            min_splits=10,
            min_buckets=5,
            max_depth=2,
            metrics=(
                "median",
                "risk_score",
                ("survival", 3.0, "units"),
                ("rmst", 4.0, "units"),
            ),
        )
        estimator.fit(X, y)
        return estimator

    def test_every_node_value_matches_reference(self):
        """Every node reproduces the reference value of all four metric kinds."""
        estimator = self._fit()
        observed = _flatten_metric_field(estimator.nodes_, "predicted_metrics")
        numpy.testing.assert_array_equal(observed, _REFERENCE_METRIC_VALUES)

    def test_every_node_interval_matches_reference(self):
        """Every node reproduces the reference confidence bounds."""
        estimator = self._fit()
        observed_low = _flatten_metric_field(estimator.nodes_, "ci_low")
        observed_high = _flatten_metric_field(estimator.nodes_, "ci_high")
        numpy.testing.assert_array_equal(observed_low, _REFERENCE_METRIC_CI_LOW)
        numpy.testing.assert_array_equal(
            observed_high, _REFERENCE_METRIC_CI_HIGH
        )

    def test_risk_score_carries_no_interval(self):
        """The risk-score metric declares no confidence interval."""
        estimator = self._fit()
        descriptor = estimator.metrics_[1]
        self.assertEqual(descriptor.label, "Risk score")
        self.assertFalse(descriptor.has_ci)
        for node in estimator.nodes_:
            self.assertTrue(numpy.isnan(node.ci_low[1]))
            self.assertTrue(numpy.isnan(node.ci_high[1]))


class TestSurvivalTreeSklearnTags(unittest.TestCase):
    """Tests for the sklearn tag overrides on SurvivalTree."""

    __slots__ = ()

    def test_target_tags_reflect_required_single_target_y(self):
        """target_tags advertises that y is required and carries one target."""
        estimator = sigma._tree_survival.SurvivalTree()
        tags = estimator.__sklearn_tags__()
        self.assertTrue(tags.target_tags.required)
        self.assertFalse(tags.target_tags.multi_output)
        self.assertTrue(tags.target_tags.single_output)

    def test_no_legacy_estimator_type_attribute(self):
        """SurvivalTree carries no pre-1.6 _estimator_type class attribute."""
        estimator = sigma._tree_survival.SurvivalTree()
        self.assertFalse(hasattr(estimator, "_estimator_type"))


class TestSurvivalTreeScore(unittest.TestCase):
    """Tests for the Harrell concordance score method on SurvivalTree."""

    __slots__ = ()

    def _build_dataset(self, n=200, seed=0):
        """Return (X, y) for a two-arm right-censored survival dataset."""
        X, time, event = _build_survival_dataset(n=n, seed=seed)
        y = numpy.column_stack([time, event])
        return X, y

    def test_score_returns_finite_float_in_unit_interval(self):
        """Returns a finite concordance value in [0, 1] for typical input."""
        X, y = self._build_dataset()
        estimator = sigma._tree_survival.SurvivalTree(
            min_splits=10, min_buckets=5, max_depth=2
        )
        estimator.fit(X, y)
        score = estimator.score(X, y)
        self.assertIsInstance(score, float)
        self.assertTrue(numpy.isfinite(score))
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_score_close_to_one_for_strongly_ordered_data(self):
        """Returns near-perfect concordance when X strictly orders the times."""
        n = 60
        time = numpy.linspace(1.0, 60.0, n)
        event = numpy.ones(n)
        X = (-time).reshape(-1, 1)
        y = numpy.column_stack([time, event])
        estimator = sigma._tree_survival.SurvivalTree(
            alpha=1.0, min_splits=2, min_buckets=1
        )
        estimator.fit(X, y)
        score = estimator.score(X, y)
        self.assertGreaterEqual(score, 0.95)

    def test_score_all_censored_returns_one_half(self):
        """Returns 0.5 when no comparable pairs exist (all censored)."""
        X, y = self._build_dataset()
        y[:, 1] = 0.0
        estimator = sigma._tree_survival.SurvivalTree(
            min_splits=10, min_buckets=5, max_depth=2, alpha=1.0
        )
        estimator.fit(X, y)
        score = estimator.score(X, y)
        self.assertEqual(score, 0.5)

    def test_score_accepts_sample_weight(self):
        """Accepts sample_weight; uniform weights match the unweighted call."""
        X, y = self._build_dataset()
        estimator = sigma._tree_survival.SurvivalTree(
            min_splits=10, min_buckets=5, max_depth=2
        )
        estimator.fit(X, y)
        unweighted = estimator.score(X, y)
        uniform = estimator.score(X, y, sample_weight=numpy.ones(len(y)))
        self.assertEqual(unweighted, uniform)
        nonuniform = numpy.linspace(0.5, 1.5, len(y))
        weighted = estimator.score(X, y, sample_weight=nonuniform)
        self.assertTrue(numpy.isfinite(weighted))
        self.assertGreaterEqual(weighted, 0.0)
        self.assertLessEqual(weighted, 1.0)

    def test_pipeline_score_works(self):
        """Pipeline.score(X, y) delegates to SurvivalTree.score without raising."""
        X, y = self._build_dataset()
        pipeline = sklearn.pipeline.Pipeline(
            [
                ("scaler", sklearn.preprocessing.StandardScaler()),
                (
                    "tree",
                    sigma._tree_survival.SurvivalTree(
                        min_splits=10, min_buckets=5, max_depth=2
                    ),
                ),
            ]
        )
        pipeline.fit(X, y)
        score = pipeline.score(X, y)
        self.assertIsInstance(score, float)
        self.assertTrue(numpy.isfinite(score))
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_cross_val_score_with_kfold(self):
        """cross_val_score(SurvivalTree, X, y, cv=KFold(3)) returns finite scores."""
        X, y = self._build_dataset(n=240)
        estimator = sigma._tree_survival.SurvivalTree(
            min_splits=10, min_buckets=5, max_depth=2
        )
        cv = sklearn.model_selection.KFold(n_splits=3)
        scores = sklearn.model_selection.cross_val_score(estimator, X, y, cv=cv)
        self.assertEqual(len(scores), 3)
        self.assertTrue(numpy.all(numpy.isfinite(scores)))
        self.assertTrue(numpy.all(scores >= 0.0))
        self.assertTrue(numpy.all(scores <= 1.0))

    def test_score_subsamples_above_cap(self):
        """Caps concordance computation at 10,000 rows with a deterministic seed."""
        X, y = self._build_dataset(n=10_500, seed=1)
        estimator = sigma._tree_survival.SurvivalTree(
            min_buckets=10, max_depth=3, random_state=123
        )
        estimator.fit(X, y)
        first = estimator.score(X, y)
        second = estimator.score(X, y)
        self.assertEqual(first, second)
        self.assertTrue(numpy.isfinite(first))
        self.assertGreaterEqual(first, 0.0)
        self.assertLessEqual(first, 1.0)


@unittest.skipUnless(_HAS_LIFELINES, "lifelines is required for the GBSG2 test")
class TestSurvivalTreeLiteratureCrosscheck(unittest.TestCase):
    """Reproduce the published GBSG2 conditional inference survival tree."""

    __slots__ = ()

    def test_gbsg2_matches_partykit_reference(self):
        """Reproduces tree structure and leaf medians from Hothorn et al. (2006)."""
        from lifelines.datasets import load_gbsg2

        frame = load_gbsg2()
        frame = frame.copy()
        frame["horTh_num"] = (frame["horTh"] == "yes").astype(int)
        frame["menostat_num"] = (frame["menostat"] == "Post").astype(int)
        tgrade_map = {"I": 0, "II": 1, "III": 2}
        frame["tgrade_num"] = frame["tgrade"].map(tgrade_map)
        feature_columns = [
            "horTh_num",
            "age",
            "menostat_num",
            "tsize",
            "tgrade_num",
            "pnodes",
            "progrec",
            "estrec",
        ]
        X = frame[feature_columns].to_numpy(dtype=float)
        y = frame[["time", "cens"]].to_numpy(dtype=float)
        estimator = sigma._tree_survival.SurvivalTree()
        estimator.fit(X, y)
        root = estimator.content_
        root_extension = root.extension
        assert isinstance(root_extension, sigma._partition.NumericalPartition)
        root_partition = typing.cast(
            sigma._partition.NumericalPartition[sigma._node.Node],
            root_extension,
        )
        self.assertEqual(
            feature_columns[root_partition.feature_index], "pnodes"
        )
        self.assertEqual(root_partition.thresholds[0], 3)
        left = root_partition.children[0]
        right = root_partition.children[1]
        left_partition = left.extension
        right_partition = right.extension
        assert isinstance(left_partition, sigma._partition.NumericalPartition)
        assert isinstance(right_partition, sigma._partition.NumericalPartition)
        self.assertEqual(
            feature_columns[left_partition.feature_index], "horTh_num"
        )
        self.assertEqual(
            feature_columns[right_partition.feature_index], "progrec"
        )
        self.assertEqual(right_partition.thresholds[0], 20)
        leaves_by_n = {leaf.n_samples: leaf for leaf in estimator.leaves_}
        self.assertEqual(leaves_by_n[248].predicted_metrics[0], 2093.0)
        self.assertTrue(numpy.isnan(leaves_by_n[128].predicted_metrics[0]))
        self.assertEqual(leaves_by_n[144].predicted_metrics[0], 624.0)
        self.assertEqual(leaves_by_n[166].predicted_metrics[0], 1701.0)
