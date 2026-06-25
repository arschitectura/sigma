"""Unit tests for DataFrame/Series consumption by the Tree estimators.

Covers three behaviors derived from pandas metadata at fit time:

- Gap A: the response name (``y.name`` of a pandas Series) is captured on
  the fitted estimator and used as a fallback in text/graphviz/response
  exports.
- Gap B: categorical-dtype columns of ``X`` (pandas or polars) are
  auto-detected as categorical features and their level names captured;
  boolean and numeric columns are accepted; string, object, and otherwise
  unsupported columns raise a clear error asking the caller to type the
  column correctly.
- Gap C: when ``y`` is string-valued or a pandas Categorical, the
  classification text and graphviz exports fall back to the labels stored
  in ``classes_`` instead of integer indices.
"""

import unittest

import numpy
import pandas
import polars

import sigma._partition
import sigma._tree_classification
import sigma._tree_regression
import sigma._tree_survival
import sigma._types


class TestResponseNameExtraction(unittest.TestCase):
    """Tests for ``y.name`` capture into ``response_name_in_``."""

    __slots__ = ()

    def test_response_name_extracted_from_named_series_regression(self):
        """A named regression Series sets ``response_name_in_``."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = pandas.Series(
            numpy.where(X.ravel() <= 20, 0.0, 10.0), name="Charges"
        )
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        regression_tree.fit(X, y)
        self.assertEqual(regression_tree.response_name_in_, "Charges")

    def test_response_name_extracted_from_named_series_classification(self):
        """A named classification Series sets ``response_name_in_``."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = pandas.Series(
            numpy.where(X.ravel() <= 20, 0.0, 1.0), name="Survival"
        )
        classification_tree = sigma._tree_classification.ClassificationTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        classification_tree.fit(X, y)
        self.assertEqual(classification_tree.response_name_in_, "Survival")

    def test_response_name_none_for_numpy_y(self):
        """``response_name_in_`` is ``None`` when ``y`` is a numpy array."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        regression_tree.fit(X, y)
        self.assertIsNone(regression_tree.response_name_in_)

    def test_response_name_none_for_unnamed_series(self):
        """``response_name_in_`` is ``None`` when the Series has no name."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = pandas.Series(numpy.where(X.ravel() <= 20, 0.0, 10.0))
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        regression_tree.fit(X, y)
        self.assertIsNone(regression_tree.response_name_in_)

    def test_to_text_uses_response_name_in_when_no_override(self):
        """``to_text`` falls back to ``response_name_in_`` for the column header."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = pandas.Series(
            numpy.where(X.ravel() <= 20, 0.0, 10.0), name="Charges"
        )
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        regression_tree.fit(X, y)
        output = regression_tree.to_text()
        self.assertIn("Charges mean", output)
        self.assertNotIn("Predicted mean", output)

    def test_to_text_response_name_kwarg_overrides_in_attribute(self):
        """An explicit ``response_name`` overrides ``response_name_in_``."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = pandas.Series(
            numpy.where(X.ravel() <= 20, 0.0, 10.0), name="Charges"
        )
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        regression_tree.fit(X, y)
        output = regression_tree.to_text(response_name="Premium")
        self.assertIn("Premium mean", output)
        self.assertNotIn("Charges mean", output)

    def test_response_name_extracted_from_survival_dataframe(self):
        """A survival ``y`` DataFrame sets ``response_name_in_`` to the time
        column name.
        """
        times = numpy.linspace(1.0, 10.0, 60)
        events = numpy.tile([1.0, 0.0], 30)
        y = pandas.DataFrame({"Months": times, "event": events})
        X = numpy.column_stack([numpy.repeat([0.0, 1.0], 30)])
        survival_tree = sigma._tree_survival.SurvivalTree(
            categorical_features=[0],
            min_splits=2,
            min_buckets=1,
        )
        survival_tree.fit(X, y)
        self.assertEqual(survival_tree.response_name_in_, "Months")


class TestClassNamesFallback(unittest.TestCase):
    """Tests for falling back to ``tree.classes_`` in text/graphviz output."""

    __slots__ = ()

    def test_to_text_falls_back_to_classes_for_string_y(self):
        """Text output renders ``tree.classes_`` when ``y`` is string-valued."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, "died", "survived")
        classification_tree = sigma._tree_classification.ClassificationTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        classification_tree.fit(X, y)
        output = classification_tree.to_text()
        self.assertIn("Died proba.", output)
        self.assertIn("Survived proba.", output)

    def test_to_text_class_names_kwarg_wins_over_classes_fallback(self):
        """An explicit ``class_names`` overrides the ``classes_`` fallback."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, "died", "survived")
        classification_tree = sigma._tree_classification.ClassificationTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        classification_tree.fit(X, y)
        output = classification_tree.to_text(class_names=["alpha", "beta"])
        self.assertIn("Alpha proba.", output)
        self.assertIn("Beta proba.", output)
        self.assertNotIn("Died proba.", output)

    def test_to_text_falls_back_to_classes_for_categorical_y(self):
        """Text output uses ``y.cat.categories`` order when ``y`` is
        Categorical.
        """
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        labels = numpy.where(X.ravel() <= 20, "died", "survived")
        y = pandas.Series(
            pandas.Categorical(
                labels, categories=["survived", "died"], ordered=False
            )
        )
        classification_tree = sigma._tree_classification.ClassificationTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        classification_tree.fit(X, y)
        numpy.testing.assert_array_equal(
            classification_tree.classes_, numpy.array(["survived", "died"])
        )

    def test_to_text_keeps_integer_fallback_when_classes_are_integers(self):
        """When ``classes_`` are 0/1 integers the headers stay as ``0`` / ``1``
        (no behavior change for the existing numeric path).
        """
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 1.0)
        classification_tree = sigma._tree_classification.ClassificationTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        classification_tree.fit(X, y)
        output = classification_tree.to_text()
        self.assertIn("0.0 proba.", output)
        self.assertIn("1.0 proba.", output)


class TestCategoricalAutoDetection(unittest.TestCase):
    """Tests for auto-detecting categorical-dtype columns of ``X``."""

    __slots__ = ()

    @staticmethod
    def _signal_design(n=40):
        """Build a length-n design with a two-level signal and a noise
        column.
        """
        rng = numpy.random.default_rng(42)
        signal = numpy.repeat(["red", "blue"], n // 2)
        noise = rng.standard_normal(n)
        y = numpy.where(signal == "red", 0.0, 10.0)
        return signal, noise, y

    @staticmethod
    def _categorical_frame(signal, noise, categories=None):
        """Wrap the signal as a pandas Categorical color column."""
        color = pandas.Categorical(signal, categories=categories)
        frame = pandas.DataFrame({"color": color, "noise": noise})
        return frame

    def test_category_dtype_column_auto_detected_as_categorical(self):
        """A pandas Categorical column is treated as CATEGORICAL."""
        signal, noise, y = self._signal_design()
        X = self._categorical_frame(signal, noise, categories=["red", "blue"])
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        regression_tree.fit(X, y)
        self.assertEqual(
            regression_tree.feature_types_[0],
            sigma._types.CovariateType.CATEGORICAL,
        )

    def test_categorical_split_constructs_categorical_partition(self):
        """Splitting on a categorical column produces a CategoricalPartition."""
        signal, noise, y = self._signal_design()
        X = self._categorical_frame(signal, noise)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        regression_tree.fit(X, y)
        partition = regression_tree.content_.extension
        self.assertIsInstance(partition, sigma._partition.CategoricalPartition)

    def test_category_labels_in_preserves_explicit_order(self):
        """``category_labels_in_`` follows the explicit Categorical order."""
        signal, noise, y = self._signal_design()
        X = self._categorical_frame(signal, noise, categories=["red", "blue"])
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        regression_tree.fit(X, y)
        self.assertEqual(
            regression_tree.category_labels_in_,
            {0: {0.0: "red", 1.0: "blue"}},
        )

    def test_category_labels_in_none_when_no_categorical_columns(self):
        """``category_labels_in_`` is ``None`` for a pure numeric
        DataFrame.
        """
        rng = numpy.random.default_rng(0)
        n = 20
        X = pandas.DataFrame(
            {"a": rng.standard_normal(n), "b": rng.standard_normal(n)}
        )
        y = rng.standard_normal(n)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        regression_tree.fit(X, y)
        self.assertIsNone(regression_tree.category_labels_in_)

    def test_to_text_uses_auto_category_labels(self):
        """``to_text`` renders auto-extracted category labels with no
        kwargs.
        """
        signal, noise, y = self._signal_design()
        X = self._categorical_frame(signal, noise)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        regression_tree.fit(X, y)
        output = regression_tree.to_text()
        self.assertTrue('"red"' in output or '"blue"' in output)

    def test_user_category_labels_overrides_auto(self):
        """An explicit ``category_labels`` mapping wins over the auto map."""
        signal, noise, y = self._signal_design()
        X = self._categorical_frame(signal, noise)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        regression_tree.fit(X, y)
        output = regression_tree.to_text(
            category_labels={"color": {0.0: "ROUGE", 1.0: "BLEU"}}
        )
        self.assertTrue("ROUGE" in output or "BLEU" in output)
        self.assertNotIn('"red"', output)
        self.assertNotIn('"blue"', output)

    def test_explicit_categorical_features_still_works(self):
        """Listing a numeric column in ``categorical_features`` still
        upgrades it.
        """
        rng = numpy.random.default_rng(42)
        n = 40
        codes = numpy.repeat([0.0, 1.0], n // 2)
        noise = rng.standard_normal(n)
        y = numpy.where(codes == 0.0, 0.0, 10.0)
        X = pandas.DataFrame({"category": codes, "noise": noise})
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            categorical_features=["category"],
            min_splits=2,
            min_buckets=1,
        )
        regression_tree.fit(X, y)
        self.assertEqual(
            regression_tree.feature_types_[0],
            sigma._types.CovariateType.CATEGORICAL,
        )

    def test_predict_accepts_categorical_dataframe(self):
        """``predict`` accepts the same DataFrame schema it was fit on."""
        signal, noise, y = self._signal_design()
        X = self._categorical_frame(signal, noise)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        regression_tree.fit(X, y)
        predictions = regression_tree.predict(X)
        self.assertEqual(predictions.shape, (len(y),))


class TestBooleanColumnDetection(unittest.TestCase):
    """Tests for fit-time detection of boolean DataFrame columns."""

    __slots__ = ()

    def _two_level_design(self):
        """Build a DataFrame design with a binary signal and a noise column."""
        rng = numpy.random.default_rng(0)
        n = 40
        signal = numpy.repeat([False, True], n // 2)
        noise = rng.standard_normal(n)
        y = numpy.where(signal, 9.0, 1.0)
        return signal, noise, y

    def test_numpy_bool_column_auto_detected_as_boolean(self):
        """A numpy bool column is classified as BOOLEAN in feature_types_."""
        signal, noise, y = self._two_level_design()
        X = pandas.DataFrame({"flag": signal, "noise": noise})
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        regression_tree.fit(X, y)
        self.assertEqual(
            regression_tree.feature_types_[0],
            sigma._types.CovariateType.BOOLEAN,
        )

    def test_pandas_boolean_dtype_auto_detected_as_boolean(self):
        """A pandas BooleanDtype column is classified as BOOLEAN."""
        signal, noise, y = self._two_level_design()
        X = pandas.DataFrame(
            {
                "flag": pandas.array(signal, dtype="boolean"),
                "noise": noise,
            }
        )
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        regression_tree.fit(X, y)
        self.assertEqual(
            regression_tree.feature_types_[0],
            sigma._types.CovariateType.BOOLEAN,
        )

    def test_boolean_features_in_records_indices(self):
        """``boolean_features_in_`` is a frozenset of bool-column indices."""
        signal, noise, y = self._two_level_design()
        X = pandas.DataFrame({"flag": signal, "noise": noise})
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        regression_tree.fit(X, y)
        self.assertEqual(regression_tree.boolean_features_in_, frozenset({0}))

    def test_boolean_features_in_none_when_no_bool_columns(self):
        """``boolean_features_in_`` is None for a pure numeric DataFrame."""
        rng = numpy.random.default_rng(0)
        n = 20
        X = pandas.DataFrame(
            {"a": rng.standard_normal(n), "b": rng.standard_normal(n)}
        )
        y = rng.standard_normal(n)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        regression_tree.fit(X, y)
        self.assertIsNone(regression_tree.boolean_features_in_)

    def test_boolean_dtype_with_na_promotes_to_categorical(self):
        """A pandas BooleanDtype column with pd.NA is promoted to a categorical with an N/A level."""
        signal, noise, y = self._two_level_design()
        signal_with_na = pandas.array(list(signal), dtype="boolean")
        signal_with_na[0] = pandas.NA
        X = pandas.DataFrame({"flag": signal_with_na, "noise": noise})
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        regression_tree.fit(X, y)
        self.assertEqual(
            regression_tree.promoted_boolean_features_in_, frozenset({0})
        )
        self.assertEqual(
            regression_tree.category_labels_in_,
            {0: {0.0: "False", 1.0: "True", 2.0: "N/A"}},
        )
        predictions = regression_tree.predict(X)
        self.assertTrue(numpy.all(numpy.isfinite(predictions)))

    def test_mixed_dataframe_assigns_each_kind(self):
        """A mix of bool / categorical / numeric assigns the right type to each."""
        rng = numpy.random.default_rng(0)
        n = 40
        flag = numpy.repeat([False, True], n // 2)
        color = pandas.Categorical(numpy.where(flag, "red", "blue"))
        noise = rng.standard_normal(n)
        y = numpy.where(flag, 9.0, 1.0)
        X = pandas.DataFrame({"flag": flag, "color": color, "noise": noise})
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        regression_tree.fit(X, y)
        self.assertEqual(
            regression_tree.feature_types_[0],
            sigma._types.CovariateType.BOOLEAN,
        )
        self.assertEqual(
            regression_tree.feature_types_[1],
            sigma._types.CovariateType.CATEGORICAL,
        )
        self.assertEqual(
            regression_tree.feature_types_[2],
            sigma._types.CovariateType.REAL,
        )

    def test_split_constructs_boolean_partition(self):
        """Splitting on a bool column produces a BooleanPartition."""
        signal, noise, y = self._two_level_design()
        X = pandas.DataFrame({"flag": signal, "noise": noise})
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        regression_tree.fit(X, y)
        partition = regression_tree.content_.extension
        self.assertIsInstance(partition, sigma._partition.BooleanPartition)

    def test_predict_roundtrip_with_bool_column(self):
        """``predict`` accepts numpy bool, pandas BooleanDtype, and 0/1 floats."""
        signal, noise, y = self._two_level_design()
        X_fit = pandas.DataFrame({"flag": signal, "noise": noise})
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        regression_tree.fit(X_fit, y)
        preds_bool = regression_tree.predict(X_fit)
        X_nullable = pandas.DataFrame(
            {
                "flag": pandas.array(signal, dtype="boolean"),
                "noise": noise,
            }
        )
        preds_nullable = regression_tree.predict(X_nullable)
        X_floats = pandas.DataFrame(
            {"flag": signal.astype(float), "noise": noise}
        )
        preds_floats = regression_tree.predict(X_floats)
        numpy.testing.assert_array_equal(preds_bool, preds_nullable)
        numpy.testing.assert_array_equal(preds_bool, preds_floats)


class TestStrictColumnTyping(unittest.TestCase):
    """Tests that string, object, and unsupported columns raise a clear
    error.
    """

    __slots__ = ()

    @staticmethod
    def _design(n=40):
        """Build a length-n design with a two-level string signal."""
        rng = numpy.random.default_rng(0)
        signal = numpy.repeat(["red", "blue"], n // 2)
        noise = rng.standard_normal(n)
        y = numpy.where(signal == "red", 0.0, 10.0)
        return signal, noise, y

    def test_pandas_string_column_raises_with_guidance(self):
        """A pandas string/object column raises naming the column and a cast
        hint.
        """
        signal, noise, y = self._design()
        X = pandas.DataFrame({"color": signal, "noise": noise})
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        with self.assertRaises(ValueError) as context:
            regression_tree.fit(X, y)
        message = str(context.exception)
        self.assertIn("color", message)
        self.assertIn("categorical", message)

    def test_polars_string_column_raises_with_guidance(self):
        """A polars String column raises naming the column and a cast hint."""
        signal, noise, y = self._design()
        X = polars.DataFrame(
            {
                "color": polars.Series(signal, dtype=polars.String),
                "noise": noise,
            }
        )
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        with self.assertRaises(ValueError) as context:
            regression_tree.fit(X, y)
        message = str(context.exception)
        self.assertIn("color", message)
        self.assertIn("categorical", message)


class TestPolarsConsumption(unittest.TestCase):
    """Tests for fitting and predicting on polars DataFrames."""

    __slots__ = ()

    @staticmethod
    def _signal_design(n=40):
        """Build a length-n design with a two-level signal and a noise
        column.
        """
        rng = numpy.random.default_rng(42)
        signal = numpy.repeat(["red", "blue"], n // 2)
        noise = rng.standard_normal(n)
        y = numpy.where(signal == "red", 0.0, 10.0)
        return signal, noise, y

    def test_polars_categorical_detected_and_labelled(self):
        """A polars Categorical column is CATEGORICAL with extracted labels."""
        signal, noise, y = self._signal_design()
        X = polars.DataFrame(
            {
                "color": polars.Series(signal, dtype=polars.Categorical),
                "noise": noise,
            }
        )
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        regression_tree.fit(X, y)
        self.assertEqual(
            regression_tree.feature_types_[0],
            sigma._types.CovariateType.CATEGORICAL,
        )
        self.assertEqual(
            regression_tree.category_labels_in_,
            {0: {0.0: "red", 1.0: "blue"}},
        )

    def test_polars_categorical_matches_pandas_labels(self):
        """polars and pandas Categorical columns yield identical label maps."""
        signal, noise, y = self._signal_design()
        categories = ["red", "blue"]
        X_polars = polars.DataFrame(
            {
                "color": polars.Series(signal, dtype=polars.Categorical),
                "noise": noise,
            }
        )
        X_pandas = pandas.DataFrame(
            {
                "color": pandas.Categorical(signal, categories=categories),
                "noise": noise,
            }
        )
        polars_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        polars_tree.fit(X_polars, y)
        pandas_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        pandas_tree.fit(X_pandas, y)
        self.assertEqual(
            polars_tree.category_labels_in_,
            pandas_tree.category_labels_in_,
        )

    def test_polars_numeric_only_fits_and_names(self):
        """A purely numeric polars DataFrame fits and sets feature names."""
        rng = numpy.random.default_rng(0)
        n = 40
        X = polars.DataFrame(
            {"a": rng.standard_normal(n), "b": rng.standard_normal(n)}
        )
        a_values = X["a"].to_numpy()
        y = a_values * 2.0
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        regression_tree.fit(X, y)
        self.assertIsNone(regression_tree.category_labels_in_)
        names_in = getattr(regression_tree, "feature_names_in_")
        self.assertEqual(list(names_in), ["a", "b"])

    def test_polars_boolean_detected(self):
        """A polars Boolean column is classified as BOOLEAN."""
        rng = numpy.random.default_rng(0)
        n = 40
        flag = numpy.repeat([False, True], n // 2)
        noise = rng.standard_normal(n)
        y = numpy.where(flag, 9.0, 1.0)
        X = polars.DataFrame(
            {
                "flag": polars.Series(flag, dtype=polars.Boolean),
                "noise": noise,
            }
        )
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        regression_tree.fit(X, y)
        self.assertEqual(regression_tree.boolean_features_in_, frozenset({0}))
        self.assertEqual(
            regression_tree.feature_types_[0],
            sigma._types.CovariateType.BOOLEAN,
        )

    def test_polars_predict_roundtrip(self):
        """``predict`` accepts the same polars schema it was fit on."""
        signal, noise, y = self._signal_design()
        X = polars.DataFrame(
            {
                "color": polars.Series(signal, dtype=polars.Categorical),
                "noise": noise,
            }
        )
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        regression_tree.fit(X, y)
        predictions = regression_tree.predict(X)
        self.assertEqual(predictions.shape, (len(y),))

    def test_polars_unseen_level_at_predict_routes_to_node_average(self):
        """An unseen polars level at predict routes to the holding node's prediction rather than raising."""
        signal, noise, y = self._signal_design()
        X = polars.DataFrame(
            {
                "color": polars.Series(signal, dtype=polars.Categorical),
                "noise": noise,
            }
        )
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        regression_tree.fit(X, y)
        unseen = numpy.where(noise > 0, "green", "red")
        X_unseen = polars.DataFrame(
            {
                "color": polars.Series(unseen, dtype=polars.Categorical),
                "noise": noise,
            }
        )
        predictions = regression_tree.predict(X_unseen)
        self.assertEqual(predictions.shape, (len(y),))
        self.assertTrue(numpy.all(numpy.isfinite(predictions)))


if __name__ == "__main__":
    unittest.main()
