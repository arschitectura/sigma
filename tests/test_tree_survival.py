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
        leaf_medians = [leaf.prediction for leaf in estimator.leaves_]
        for prediction in predictions:
            self.assertIn(prediction, leaf_medians)

    def test_predict_survival_function_shape_and_monotonicity(self):
        """predict_survival_function returns non-increasing rows in [0, 1]."""
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
        surv = estimator.predict_survival_function(X[:5], times)
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

    def test_leaf_carries_survival_function(self):
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
            assert leaf.survival_function is not None
            leaf_times, leaf_surv = leaf.survival_function
            self.assertEqual(len(leaf_times), len(leaf_surv))
            if len(leaf_times) > 1:
                self.assertTrue(numpy.all(numpy.diff(leaf_surv) <= 1e-12))


class TestSurvivalTreeYEncodings(unittest.TestCase):
    """Tests for the three accepted y encodings on fit and score."""

    __slots__ = ()

    def _build_dataset(self, n=200, seed=0):
        """Return (X, time, event) for a two-arm right-censored survival dataset."""
        rng = numpy.random.RandomState(seed)
        arm = (numpy.arange(n) % 2).astype(float)
        scales = numpy.where(arm == 0, 10.0, 2.0)
        survival = rng.exponential(scale=scales)
        time = numpy.minimum(survival, 8.0)
        event = (survival <= 8.0).astype(float)
        X = numpy.column_stack([arm, rng.randn(n)])
        return X, time, event

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
    """Tests for the configurable per-leaf metrics on SurvivalTree."""

    __slots__ = ()

    def _build_dataset(self):
        """Return (X, y) for a simple two-arm survival dataset."""
        rng = numpy.random.RandomState(0)
        n = 200
        arm = (numpy.arange(n) % 2).astype(float)
        scales = numpy.where(arm == 0, 10.0, 2.0)
        survival = rng.exponential(scale=scales)
        time = numpy.minimum(survival, 8.0)
        event = (survival <= 8.0).astype(float)
        X = numpy.column_stack([arm, rng.randn(n)])
        y = numpy.column_stack([time, event])
        return X, y

    def test_default_metrics_match_legacy_behavior(self):
        """Default metrics=('median',) preserves the legacy median display."""
        X, y = self._build_dataset()
        estimator = sigma._tree_survival.SurvivalTree(
            min_splits=10, min_buckets=5, max_depth=2
        )
        estimator.fit(X, y)
        for leaf in estimator.leaves_:
            assert leaf.metrics is not None
            self.assertEqual(len(leaf.metrics), 1)
            self.assertEqual(leaf.metrics[0].label, "Median survival")
            self.assertEqual(leaf.metrics[0].value, leaf.prediction)

    def test_multiple_metrics_render_each_on_own_line(self):
        """Multi-metric leaves expose one record per configured metric."""
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
        for leaf in estimator.leaves_:
            assert leaf.metrics is not None
            self.assertEqual(len(leaf.metrics), 3)
            labels = [m.label for m in leaf.metrics]
            self.assertEqual(labels[0], "Median survival")
            self.assertEqual(labels[1], "Survival at 5 units")
            self.assertEqual(labels[2], "Risk score")

    def test_first_metric_drives_prediction_slot(self):
        """The first metric's value mirrors Node.prediction."""
        X, y = self._build_dataset()
        estimator = sigma._tree_survival.SurvivalTree(
            min_splits=10,
            min_buckets=5,
            max_depth=2,
            metrics=("risk_score", "median"),
        )
        estimator.fit(X, y)
        for leaf in estimator.leaves_:
            assert leaf.metrics is not None
            self.assertEqual(leaf.prediction, leaf.metrics[0].value)
        predictions = estimator.predict(X[:5])
        for prediction in predictions:
            self.assertTrue(numpy.isfinite(prediction))

    def test_survival_value_matches_step_curve(self):
        """('survival', t, ...) value equals predict_survival_function at t."""
        X, y = self._build_dataset()
        estimator = sigma._tree_survival.SurvivalTree(
            min_splits=10,
            min_buckets=5,
            max_depth=2,
            metrics=(("survival", 3.0, "units"),),
        )
        estimator.fit(X, y)
        from_metric = estimator.predict(X[:6])
        from_curve = estimator.predict_survival_function(
            X[:6], numpy.array([3.0])
        )[:, 0]
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
            metrics=("median",),
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
        risk_scores = [leaf.prediction for leaf in estimator.leaves_]
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
        medians = [leaf.prediction for leaf in estimator.leaves_]
        for k in range(len(medians) - 1):
            self.assertLessEqual(medians[k], medians[k + 1])

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
            metrics=("median",),
        )
        baseline.fit(X, y)
        expected = baseline.predict(X)
        numpy.testing.assert_array_equal(observed, expected)

    def test_lexicographic_tie_break_uses_secondary_metric(self):
        """With two metrics, a tie on the first falls through to the second."""
        median_metric = sigma._node.SurvivalMetric(
            label="Median survival",
            value=float("inf"),
            ci_low=None,
            ci_high=None,
            style="value",
            better_is="higher",
        )
        leaf_a = sigma._node.SurvivalNode(
            depth=1,
            n_samples=10,
            share=0.0,
            decoration=None,
            extension=sigma._extension.Leaf(),
            survival_function=(numpy.array([1.0]), numpy.array([1.0])),
            survival_log_variance=numpy.zeros(1, dtype=float),
            metrics=[
                median_metric,
                sigma._node.SurvivalMetric(
                    label="Survival at 5 units",
                    value=0.7,
                    ci_low=None,
                    ci_high=None,
                    style="probability",
                    better_is="higher",
                ),
            ],
        )
        leaf_b = sigma._node.SurvivalNode(
            depth=1,
            n_samples=10,
            share=0.0,
            decoration=None,
            extension=sigma._extension.Leaf(),
            survival_function=(numpy.array([1.0]), numpy.array([1.0])),
            survival_log_variance=numpy.zeros(1, dtype=float),
            metrics=[
                median_metric,
                sigma._node.SurvivalMetric(
                    label="Survival at 5 units",
                    value=0.4,
                    ci_low=None,
                    ci_high=None,
                    style="probability",
                    better_is="higher",
                ),
            ],
        )
        key_a = leaf_a.leaf_sort_key()
        key_b = leaf_b.leaf_sort_key()
        self.assertLess(key_b, key_a)


class TestSurvivalTreeSklearnTags(unittest.TestCase):
    """Tests for the sklearn tag overrides on SurvivalTree."""

    __slots__ = ()

    def test_target_tags_reflect_required_two_dimensional_y(self):
        """target_tags advertises that y is required and multi-output."""
        estimator = sigma._tree_survival.SurvivalTree()
        tags = estimator.__sklearn_tags__()
        self.assertTrue(tags.target_tags.required)
        self.assertTrue(tags.target_tags.multi_output)


class TestSurvivalTreeScore(unittest.TestCase):
    """Tests for the Harrell concordance score method on SurvivalTree."""

    __slots__ = ()

    def _build_dataset(self, n=200, seed=0):
        """Return (X, y) for a two-arm right-censored survival dataset."""
        rng = numpy.random.RandomState(seed)
        arm = (numpy.arange(n) % 2).astype(float)
        scales = numpy.where(arm == 0, 10.0, 2.0)
        survival = rng.exponential(scale=scales)
        time = numpy.minimum(survival, 8.0)
        event = (survival <= 8.0).astype(float)
        X = numpy.column_stack([arm, rng.randn(n)])
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
            alpha=1.0, min_splits=2, min_buckets=1, max_depth=None
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
            min_splits=20, min_buckets=10, max_depth=3, random_state=42
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
        self.assertEqual(root_partition.threshold, 3)
        left = root_partition.left
        right = root_partition.right
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
        self.assertEqual(right_partition.threshold, 20)
        leaves_by_n = {leaf.n_samples: leaf for leaf in estimator.leaves_}
        self.assertEqual(leaves_by_n[248].prediction, 2093.0)
        self.assertEqual(leaves_by_n[128].prediction, float("inf"))
        self.assertEqual(leaves_by_n[144].prediction, 624.0)
        self.assertEqual(leaves_by_n[166].prediction, 1701.0)
