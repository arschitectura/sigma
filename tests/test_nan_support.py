"""Tests for NaN (missing value) support in the four tree estimators."""

import math
import unittest

import numpy
import pandas

import sigma
import sigma._partition
import sigma._splitting
import sigma._types


def _numeric_missingness_design(seed=0, n=200, missing_rate=0.3):
    """Build a numeric X whose response is driven by the missingness of x0."""
    rng = numpy.random.default_rng(seed)
    x0 = rng.normal(0.0, 1.0, n)
    missing = rng.random(n) < missing_rate
    x0[missing] = numpy.nan
    x1 = rng.normal(0.0, 1.0, n)
    X = numpy.column_stack([x0, x1])
    y = missing * 3.0 + numpy.nan_to_num(x0) + rng.normal(0.0, 0.3, n)
    return X, y, missing


class TestNumericPartitionRouting(unittest.TestCase):
    """Routing of NaN through a numeric partition."""

    __slots__ = ()

    def _leaf(self, value):
        """Build a minimal regression leaf carrying a prediction."""
        return sigma._node.RegressionNode(
            0,
            1,
            1.0,
            None,
            sigma._extension.Leaf(),
            value,
            None,
            None,
            numpy.array([], dtype=float),
        )

    def test_nan_routes_to_nan_child(self):
        """A NaN value routes to children[nan_child] when one was learned."""
        left, right = self._leaf(1.0), self._leaf(2.0)
        partition = sigma._partition.NumericalPartition(
            0, None, None, (left, right), (5.0,), nan_child=1
        )
        self.assertIs(partition.route(3.0), left)
        self.assertIs(partition.route(9.0), right)
        self.assertIs(partition.route(float("nan")), right)

    def test_nan_unroutable_returns_none(self):
        """A NaN value is unroutable when no missing rule was learned."""
        left, right = self._leaf(1.0), self._leaf(2.0)
        partition = sigma._partition.NumericalPartition(
            0, None, None, (left, right), (5.0,), nan_child=None
        )
        self.assertIsNone(partition.route(float("nan")))

    def test_pure_missingness_observed_routes_to_observed_child(self):
        """With no threshold, observed values reach the observed child, NaN the other."""
        observed, nan_child = self._leaf(1.0), self._leaf(9.0)
        partition = sigma._partition.NumericalPartition(
            0, None, None, (observed, nan_child), (), nan_child=1
        )
        self.assertIs(partition.route(0.0), observed)
        self.assertIs(partition.route(1e9), observed)
        self.assertIs(partition.route(float("nan")), nan_child)

    def test_dedicated_missing_child_unreached_by_observed(self):
        """An observed value above all thresholds reaches the last interval child, not a trailing missing child."""
        c0, c1, missing = self._leaf(1.0), self._leaf(2.0), self._leaf(9.0)
        partition = sigma._partition.NumericalPartition(
            0, None, None, (c0, c1, missing), (5.0,), nan_child=2
        )
        self.assertIs(partition.route(9.0), c1)
        self.assertIs(partition.route(float("nan")), missing)

    def test_branch_conditions_emit_missing_value(self):
        """branch_conditions appends a MissingValue for a dedicated missing child."""
        c0, c1, missing = self._leaf(1.0), self._leaf(2.0), self._leaf(9.0)
        partition = sigma._partition.NumericalPartition(
            0, None, None, (c0, c1, missing), (5.0,), nan_child=2
        )
        conditions = partition.branch_conditions
        self.assertEqual(len(conditions), 3)
        self.assertIsInstance(conditions[-1], sigma._partition.MissingValue)


class TestBooleanPartitionRouting(unittest.TestCase):
    """Routing of NaN through a boolean partition."""

    __slots__ = ()

    def test_nan_returns_none(self):
        """A clean boolean partition returns None for a NaN value."""
        left = sigma._node.RegressionNode(
            0,
            1,
            1.0,
            None,
            sigma._extension.Leaf(),
            1.0,
            None,
            None,
            numpy.array([], dtype=float),
        )
        right = sigma._node.RegressionNode(
            0,
            1,
            1.0,
            None,
            sigma._extension.Leaf(),
            2.0,
            None,
            None,
            numpy.array([], dtype=float),
        )
        partition = sigma._partition.BooleanPartition(
            0, None, None, (left, right)
        )
        self.assertIsNone(partition.route(float("nan")))
        self.assertIs(partition.route(0.0), left)
        self.assertIs(partition.route(1.0), right)


class TestNumericSplitAlternatives(unittest.TestCase):
    """The three numeric split alternatives under missingness."""

    __slots__ = ()

    def _find(self, X_j, h):
        """Run the numeric split search with normal correlation."""
        result = sigma._splitting.find_best_split_numeric(
            X_j,
            h.reshape(-1, 1),
            numpy.ones(len(X_j)),
            sigma._types.TestStat.QUADRATIC,
            min_buckets=3,
            is_integer=True,
            correlation=sigma._types.Correlation.NORMAL,
        )
        return result

    def test_pure_missingness_alternative_chosen(self):
        """A pure-missingness signal yields a threshold-free split."""
        X_j = numpy.arange(1, 41, dtype=float)
        X_j[:10] = numpy.nan
        h = numpy.isnan(X_j).astype(float)
        split, _ = self._find(X_j, h)
        self.assertIsNone(split.threshold)
        self.assertEqual(split.nan_child, 1)

    def test_ride_along_alternative_chosen(self):
        """An observed-value signal yields a threshold with NaN riding a side."""
        X_j = numpy.arange(1, 41, dtype=float)
        X_j[:10] = numpy.nan
        h = (X_j > 25).astype(float)
        split, _ = self._find(X_j, h)
        self.assertIsNotNone(split.threshold)
        self.assertIn(split.nan_child, (0, 1))

    def test_complete_data_has_no_missing_child(self):
        """Complete data produces a plain threshold with nan_child None."""
        X_j = numpy.arange(1, 41, dtype=float)
        h = (X_j > 20).astype(float)
        split, _ = self._find(X_j, h)
        self.assertEqual(split.threshold, 20.0)
        self.assertIsNone(split.nan_child)


class TestEndToEndNumeric(unittest.TestCase):
    """End-to-end numeric NaN fit, predict, and compaction."""

    __slots__ = ()

    def test_fit_predict_finite_and_deterministic(self):
        """Fitting and predicting with NaN yields finite, deterministic output."""
        X, y, _ = _numeric_missingness_design()
        tree = sigma.RegressionTree(min_splits=10, min_buckets=5)
        tree.fit(X, y)
        probe = numpy.array(
            [[numpy.nan, 0.5], [0.2, numpy.nan], [numpy.nan, numpy.nan]]
        )
        first = tree.predict(probe)
        second = tree.predict(probe)
        self.assertTrue(numpy.all(numpy.isfinite(tree.predict(X))))
        numpy.testing.assert_array_equal(first, second)

    def test_inf_rejected_at_fit_and_predict(self):
        """Infinity is rejected at both fit and predict even though NaN is allowed."""
        X, y, _ = _numeric_missingness_design()
        tree = sigma.RegressionTree(min_splits=10, min_buckets=5)
        tree.fit(X, y)
        X_inf = X.copy()
        X_inf[0, 1] = numpy.inf
        with self.assertRaises(ValueError):
            sigma.RegressionTree().fit(X_inf, y)
        with self.assertRaises(ValueError):
            tree.predict(numpy.array([[numpy.inf, 0.0]]))

    def test_compact_preserves_nan_routing(self):
        """compact() predicts identically on observed and NaN rows."""
        X, y, _ = _numeric_missingness_design()
        tree = sigma.RegressionTree(min_splits=10, min_buckets=5)
        tree.fit(X, y)
        compacted = tree.compact()
        probe = numpy.array(
            [[numpy.nan, 0.5], [0.2, numpy.nan], [1.0, 1.0], [numpy.nan, -2.0]]
        )
        numpy.testing.assert_allclose(tree.predict(X), compacted.predict(X))
        numpy.testing.assert_allclose(
            tree.predict(probe), compacted.predict(probe)
        )

    def test_predict_nan_on_clean_fit_column_is_node_average(self):
        """Predicting NaN on a column complete at fit yields a finite node prediction."""
        rng = numpy.random.default_rng(1)
        X = rng.normal(0.0, 1.0, (120, 2))
        y = (X[:, 0] > 0).astype(float) + rng.normal(0.0, 0.2, 120)
        tree = sigma.RegressionTree(min_splits=10, min_buckets=5)
        tree.fit(X, y)
        prediction = tree.predict(numpy.array([[numpy.nan, numpy.nan]]))
        self.assertTrue(math.isfinite(prediction[0]))


class TestEndToEndDataFrame(unittest.TestCase):
    """End-to-end NaN handling for categorical and boolean DataFrame columns."""

    __slots__ = ()

    def _frame(self, seed=0, n=80):
        """Build a DataFrame with a NaN-bearing categorical and boolean column."""
        rng = numpy.random.default_rng(seed)
        cat = numpy.where(rng.random(n) < 0.5, "a", "b").astype(object)
        cat[rng.random(n) < 0.25] = None
        flag = pandas.array(rng.random(n) < 0.5, dtype="boolean")
        flag[rng.random(n) < 0.25] = pandas.NA
        frame = pandas.DataFrame(
            {
                "cat": pandas.Series(cat, dtype="category"),
                "flag": flag,
                "num": rng.normal(0.0, 1.0, n),
            }
        )
        y = pandas.isna(pandas.Series(cat)).to_numpy() * 2.0 + rng.normal(
            0.0, 0.3, n
        )
        return frame, y

    def test_categorical_gets_na_level(self):
        """A NaN-bearing categorical learns an N/A level and predicts finitely."""
        frame, y = self._frame()
        tree = sigma.RegressionTree(min_splits=8, min_buckets=4)
        tree.fit(frame, y)
        self.assertEqual(tree.na_codes_in_, {0: 2.0, 1: 2.0})
        self.assertTrue(numpy.all(numpy.isfinite(tree.predict(frame))))

    def test_na_label_collision_increments(self):
        """The N/A label avoids collision with a real category named "N/A"."""
        n = 60
        rng = numpy.random.default_rng(2)
        values = numpy.where(rng.random(n) < 0.5, "N/A", "b").astype(object)
        values[rng.random(n) < 0.3] = None
        frame = pandas.DataFrame(
            {"cat": pandas.Series(values, dtype="category")}
        )
        y = pandas.isna(pandas.Series(values)).to_numpy() * 2.0
        tree = sigma.RegressionTree(min_splits=4, min_buckets=2)
        tree.fit(frame, y)
        self.assertEqual(tree.na_codes_in_, {0: 2.0})
        self.assertEqual(
            tree.category_labels_in_,
            {0: {0.0: "N/A", 1.0: "b", 2.0: "N/A 2"}},
        )

    def test_boolean_with_na_promoted(self):
        """A NaN-bearing boolean is promoted to a categorical with pinned codes."""
        frame, y = self._frame()
        tree = sigma.RegressionTree(min_splits=8, min_buckets=4)
        tree.fit(frame, y)
        self.assertEqual(tree.promoted_boolean_features_in_, frozenset({1}))
        self.assertEqual(
            tree.category_labels_in_,
            {
                0: {0.0: "a", 1.0: "b", 2.0: "N/A"},
                1: {0.0: "False", 1.0: "True", 2.0: "N/A"},
            },
        )

    def test_unseen_category_routes_to_node_average(self):
        """An unseen category at predict yields a finite prediction, not a raise."""
        frame, y = self._frame()
        tree = sigma.RegressionTree(min_splits=8, min_buckets=4)
        tree.fit(frame, y)
        probe = pandas.DataFrame(
            {
                "cat": pandas.Series(["zzz"], dtype="category"),
                "flag": pandas.array([pandas.NA], dtype="boolean"),
                "num": [0.0],
            }
        )
        prediction = tree.predict(probe)
        self.assertTrue(math.isfinite(prediction[0]))


class TestTags(unittest.TestCase):
    """allow_nan tag on every estimator."""

    __slots__ = ()

    def test_all_estimators_allow_nan(self):
        """All four tree estimators report input_tags.allow_nan True."""
        estimators = [
            sigma.RegressionTree(),
            sigma.ClassificationTree(),
            sigma.SurvivalTree(),
            sigma.RankingTree(),
        ]
        for estimator in estimators:
            with self.subTest(estimator=type(estimator).__name__):
                tags = estimator.__sklearn_tags__()
                self.assertTrue(tags.input_tags.allow_nan)


class TestExports(unittest.TestCase):
    """SQL and text rendering of NaN-bearing trees."""

    __slots__ = ()

    def test_sql_numeric_missingness_routes_null(self):
        """Numeric missingness emits an IS NULL branch and an ELSE prediction."""
        X, y, _ = _numeric_missingness_design()
        tree = sigma.RegressionTree(min_splits=10, min_buckets=5)
        tree.fit(X, y)
        sql = tree.to_sql()
        self.assertIn("IS NULL", sql)
        self.assertNotIn("ELSE NULL", sql)

    def test_sql_categorical_na_emits_is_null_not_label(self):
        """A categorical N/A group renders as IS NULL, never the N/A literal."""
        rng = numpy.random.default_rng(4)
        n = 100
        cat = numpy.where(rng.random(n) < 0.5, "a", "b").astype(object)
        cat[rng.random(n) < 0.3] = None
        frame = pandas.DataFrame({"cat": pandas.Series(cat, dtype="category")})
        y = pandas.isna(pandas.Series(cat)).to_numpy() * 2.0
        tree = sigma.RegressionTree(min_splits=4, min_buckets=2)
        tree.fit(frame, y)
        sql = tree.to_sql()
        self.assertIn("IS NULL", sql)
        self.assertNotIn("'N/A'", sql)

    def test_sql_promoted_boolean_uses_boolean_literals(self):
        """A promoted boolean renders boolean literals, not string membership."""
        rng = numpy.random.default_rng(5)
        n = 120
        flag = pandas.array(rng.random(n) < 0.5, dtype="boolean")
        flag[rng.random(n) < 0.3] = pandas.NA
        frame = pandas.DataFrame({"flag": flag})
        values = numpy.where(pandas.isna(flag), 1.0, 0.0)
        values = numpy.where(flag.fillna(False).to_numpy(), 5.0, values)
        y = values + rng.normal(0.0, 0.2, n)
        tree = sigma.RegressionTree(min_splits=4, min_buckets=2)
        tree.fit(frame, y)
        sql = tree.to_sql()
        self.assertNotIn("'True'", sql)
        self.assertNotIn("'False'", sql)
        self.assertTrue("TRUE" in sql or "FALSE" in sql or "IS NULL" in sql)

    def test_text_renders_missing_label(self):
        """to_text labels the numeric missing branch and does not crash."""
        X, y, _ = _numeric_missingness_design()
        tree = sigma.RegressionTree(min_splits=10, min_buckets=5)
        tree.fit(X, y)
        text = tree.to_text()
        self.assertIn("is missing", text)


if __name__ == "__main__":
    unittest.main()
