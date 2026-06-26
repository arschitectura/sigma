"""Tests for NaN (missing value) support in the four tree estimators."""

import copy
import math
import sqlite3
import unittest

import numpy
import pandas

import sigma
import sigma._partition
import sigma._splitting
import sigma._types


def _eval_numeric_sql(sql, column_name, value):
    """Evaluate a single-column numeric SQL CASE expression at one value."""
    connection = sqlite3.connect(":memory:")
    try:
        cursor = connection.cursor()
        cursor.execute(f'CREATE TABLE probe ("{column_name}" REAL)')
        cursor.execute("INSERT INTO probe VALUES (?)", (value,))
        cursor.execute(f"SELECT {sql} FROM probe")
        row = cursor.fetchone()
        result = row[0]
    finally:
        connection.close()
    return result


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

    def _rebuild_left_mask(self, X_j, split):
        """Rebuild the scored left mask implied by a chosen numeric split."""
        isnan = numpy.isnan(X_j)
        if split.threshold is None:
            return ~isnan
        at_or_below = X_j <= split.threshold
        if split.nan_child == 0:
            return at_or_below | isnan
        return at_or_below

    def _designs(self):
        """Return (label, X_j, h, threshold_is_none, nan_child) per alternative."""
        pure = numpy.arange(1, 41, dtype=float)
        pure[:10] = numpy.nan
        ride_left = numpy.arange(1, 41, dtype=float)
        ride_left[:10] = numpy.nan
        ride_right = numpy.arange(1, 41, dtype=float)
        ride_right[:10] = numpy.nan
        ride_right_h = numpy.where(
            numpy.isnan(ride_right),
            1.0,
            (numpy.arange(1, 41) > 30).astype(float),
        )
        plain = numpy.arange(1, 41, dtype=float)
        return [
            (
                "pure-missingness",
                pure,
                numpy.isnan(pure).astype(float),
                True,
                1,
            ),
            (
                "ride-along-left",
                ride_left,
                (ride_left > 25).astype(float),
                False,
                0,
            ),
            ("ride-along-right", ride_right, ride_right_h, False, 1),
            ("plain", plain, (plain > 20).astype(float), False, None),
        ]

    def test_rebuilt_mask_reproduces_returned_statistic(self):
        """Each alternative's rebuilt left mask rescores to the returned statistic."""
        for label, X_j, h, threshold_is_none, nan_child in self._designs():
            with self.subTest(design=label):
                split, statistic = self._find(X_j, h)
                self.assertEqual(split.threshold is None, threshold_is_none)
                self.assertEqual(split.nan_child, nan_child)
                left_mask = self._rebuild_left_mask(X_j, split)
                weights = numpy.ones(len(X_j))
                ranked_h = sigma._splitting._maybe_rank_h(
                    h.reshape(-1, 1), weights, sigma._types.Correlation.NORMAL
                )
                rescored = sigma._splitting._score_split(
                    left_mask,
                    ranked_h,
                    weights,
                    sigma._types.TestStat.QUADRATIC,
                )
                self.assertAlmostEqual(rescored, statistic, places=9)


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

    def _ride_along_tree(self):
        """Fit a depth-1 tree whose numeric split routes NaN along one side."""
        rng = numpy.random.default_rng(7)
        n = 400
        x = rng.uniform(0.0, 1.0, n)
        missing = rng.random(n) < 0.25
        y = (x > 0.6).astype(float) * 5.0 + rng.normal(0.0, 0.2, n)
        y[missing] = 5.0 + rng.normal(0.0, 0.2, missing.sum())
        x[missing] = numpy.nan
        tree = sigma.RegressionTree(random_state=1, max_depth=1, min_buckets=5)
        tree.fit(x.reshape(-1, 1), y)
        return tree

    def _promoted_boolean_tree(self):
        """Fit a tree on a NaN-bearing boolean promoted to three levels."""
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
        return tree

    def test_ride_along_folds_missing_into_text_and_graphviz(self):
        """A numeric ride-along branch shows the missing fold in text and graphviz."""
        tree = self._ride_along_tree()
        text = tree.to_text()
        ride_lines = [line for line in text.splitlines() if " or " in line]
        self.assertTrue(
            any(
                "is missing" in line and ("<=" in line or ">" in line)
                for line in ride_lines
            )
        )
        self.assertIn("is missing", sigma.export_graphviz(tree))

    def test_promoted_boolean_renders_boolean_words(self):
        """A promoted boolean renders is true/false/missing in text and graphviz."""
        tree = self._promoted_boolean_tree()
        text = tree.to_text()
        self.assertIn("is true", text)
        self.assertIn("is false", text)
        self.assertIn("is missing", text)
        self.assertNotIn('"True"', text)
        self.assertNotIn('"N/A"', text)
        dot = sigma.export_graphviz(tree)
        self.assertIn("is true", dot)
        self.assertIn("is missing", dot)

    def test_numpy_categorical_missing_renders_missing_word(self):
        """A numpy categorical missing branch renders the word missing, not a code."""
        rng = numpy.random.default_rng(4)
        n = 400
        codes = rng.integers(0, 3, n).astype(float)
        missing = rng.random(n) < 0.25
        codes[missing] = numpy.nan
        y = numpy.where(missing, 10.0, 0.0) + rng.normal(0.0, 0.2, n)
        tree = sigma.RegressionTree(
            random_state=1, max_depth=1, categorical_features=[0]
        )
        tree.fit(codes.reshape(-1, 1), y)
        self.assertIn("is missing", tree.to_text())


class TestNumpyCategoricalMissing(unittest.TestCase):
    """numpy categorical_features columns carrying NaN at fit time."""

    __slots__ = ()

    def _fit(self, seed=4, n=400):
        """Fit a depth-1 regression tree whose missingness drives the response."""
        rng = numpy.random.default_rng(seed)
        codes = rng.integers(0, 3, n).astype(float)
        missing = rng.random(n) < 0.25
        codes[missing] = numpy.nan
        y = numpy.where(missing, 10.0, 0.0) + rng.normal(0.0, 0.2, n)
        tree = sigma.RegressionTree(
            random_state=1, max_depth=1, categorical_features=[0]
        )
        tree.fit(codes.reshape(-1, 1), y)
        return tree

    def test_literal_na_code_routes_to_node_average(self):
        """A literal value equal to the learned N/A code routes to the node average like any unseen category, not to the missing child."""
        tree = self._fit()
        self.assertEqual(tree.na_codes_in_, {0: 3.0})
        node_average = tree.predict(numpy.array([[99.0]]))[0]
        literal_k = tree.predict(numpy.array([[3.0]]))[0]
        missing = tree.predict(numpy.array([[numpy.nan]]))[0]
        self.assertAlmostEqual(literal_k, node_average, places=6)
        self.assertGreater(missing, 5.0)
        self.assertFalse(math.isclose(literal_k, missing, rel_tol=1e-2))

    def test_sql_matches_predict_on_literal_and_missing(self):
        """The exported SQL agrees with predict on the N/A code, an unseen code, and a missing row."""
        tree = self._fit()
        sql = tree.to_sql()
        cases = [(3.0, "literal-na-code"), (99.0, "unseen"), (None, "missing")]
        for value, label in cases:
            with self.subTest(value=label):
                probe = numpy.nan if value is None else value
                predicted = tree.predict(numpy.array([[probe]]))[0]
                evaluated = _eval_numeric_sql(sql, "X[0]", value)
                self.assertAlmostEqual(predicted, evaluated, places=6)


class TestResponseValidation(unittest.TestCase):
    """NaN in the response is rejected except for ranking targets."""

    __slots__ = ()

    def test_regression_nan_in_y_raises(self):
        """A NaN response value raises for the regression tree."""
        rng = numpy.random.default_rng(0)
        X = rng.normal(0.0, 1.0, (60, 2))
        y = rng.normal(0.0, 1.0, 60)
        y[3] = numpy.nan
        with self.assertRaises(ValueError):
            sigma.RegressionTree().fit(X, y)

    def test_classification_nan_in_y_raises(self):
        """A NaN response value raises for the classification tree."""
        rng = numpy.random.default_rng(0)
        X = rng.normal(0.0, 1.0, (60, 2))
        y = (X[:, 0] > 0).astype(float)
        y[3] = numpy.nan
        with self.assertRaises(ValueError):
            sigma.ClassificationTree().fit(X, y)

    def test_survival_nan_in_y_raises(self):
        """A NaN response value raises for the survival tree."""
        rng = numpy.random.default_rng(0)
        X = rng.normal(0.0, 1.0, (60, 2))
        time = rng.exponential(5.0, 60)
        event = (rng.random(60) < 0.7).astype(float)
        y = numpy.column_stack([time, event])
        y[3, 0] = numpy.nan
        with self.assertRaises(ValueError):
            sigma.SurvivalTree().fit(X, y)

    def test_ranking_nan_in_y_is_allowed(self):
        """A partial ranking carrying NaN fits without raising."""
        rng = numpy.random.default_rng(0)
        n = 80
        X = rng.normal(0.0, 1.0, (n, 2))
        y = numpy.array([rng.permutation(3) + 1.0 for _ in range(n)])
        y[:20, 2] = numpy.nan
        tree = sigma.RankingTree(random_state=0)
        tree.fit(X, y)
        predictions = tree.predict(X)
        self.assertEqual(predictions.shape[0], n)


class TestEndToEndTasks(unittest.TestCase):
    """NaN in X is handled end to end for every task."""

    __slots__ = ()

    def test_classification_predicts_known_labels(self):
        """A classification tree fit with NaN predicts labels from its class set."""
        X, y_signal, _ = _numeric_missingness_design(seed=1)
        y = (y_signal > numpy.median(y_signal)).astype(int)
        tree = sigma.ClassificationTree(min_splits=10, min_buckets=5)
        tree.fit(X, y)
        predictions = tree.predict(X)
        self.assertTrue(
            set(predictions.tolist()).issubset(set(tree.classes_.tolist()))
        )

    def test_survival_predicts_finite(self):
        """A survival tree fit with NaN in X predicts finite risk scores."""
        rng = numpy.random.default_rng(2)
        n = 200
        X, _, missing = _numeric_missingness_design(seed=2, n=n)
        time = numpy.where(missing, 2.0, 8.0) + rng.exponential(1.0, n)
        event = (rng.random(n) < 0.7).astype(float)
        y = numpy.column_stack([time, event])
        tree = sigma.SurvivalTree(min_splits=10, min_buckets=5)
        tree.fit(X, y)
        self.assertTrue(numpy.all(numpy.isfinite(tree.predict(X))))

    def test_ranking_predicts_finite(self):
        """A ranking tree fit with NaN in X predicts finite mean ranks."""
        rng = numpy.random.default_rng(3)
        n = 160
        X, _, missing = _numeric_missingness_design(seed=3, n=n)
        base = numpy.where(missing[:, None], [1.0, 2.0, 3.0], [3.0, 2.0, 1.0])
        y = base + rng.normal(0.0, 0.1, (n, 3))
        tree = sigma.RankingTree(random_state=0, min_splits=10, min_buckets=5)
        tree.fit(X, y)
        self.assertTrue(numpy.all(numpy.isfinite(tree.predict(X))))


class TestDegenerateColumns(unittest.TestCase):
    """Degenerate NaN-bearing columns are handled without crashing."""

    __slots__ = ()

    def test_all_nan_and_constant_columns_fit_finitely(self):
        """An all-NaN column and a constant-observed column fit and predict finitely."""
        rng = numpy.random.default_rng(0)
        n = 150
        all_nan = numpy.full(n, numpy.nan)
        constant = numpy.full(n, 4.0)
        constant[rng.random(n) < 0.3] = numpy.nan
        signal = rng.normal(0.0, 1.0, n)
        X = numpy.column_stack([all_nan, constant, signal])
        y = (signal > 0).astype(float) + rng.normal(0.0, 0.2, n)
        tree = sigma.RegressionTree(min_splits=10, min_buckets=5)
        tree.fit(X, y)
        self.assertTrue(numpy.all(numpy.isfinite(tree.predict(X))))


class TestPartitionState(unittest.TestCase):
    """Numeric partition state survives shallow copying."""

    __slots__ = ()

    def test_copy_preserves_nan_child(self):
        """copy.copy keeps the nan_child slot on a numeric partition."""
        left = sigma._node.RegressionNode(
            1,
            1,
            1.0,
            None,
            sigma._extension.Leaf(),
            0.0,
            None,
            None,
            numpy.array([], dtype=float),
        )
        right = sigma._node.RegressionNode(
            1,
            1,
            1.0,
            None,
            sigma._extension.Leaf(),
            1.0,
            None,
            None,
            numpy.array([], dtype=float),
        )
        partition = sigma._partition.NumericalPartition(
            0, "x", None, (left, right), (5.0,), nan_child=1
        )
        clone = copy.copy(partition)
        self.assertEqual(clone.nan_child, 1)


if __name__ == "__main__":
    unittest.main()
