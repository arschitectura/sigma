"""Unit tests for fitting and text/format output of Tree estimators."""

import typing
import unittest
import warnings

import _helpers
import numpy
import numpy.testing
import pandas

import sigma._extension
import sigma._node
import sigma._partition
import sigma._tree
import sigma._tree_classification
import sigma._tree_regression
import sigma._tree_survival
import sigma._tree_text
import sigma._types


class TestRegressionTreeFit(unittest.TestCase):
    """Tests for the fit method of RegressionTree."""

    __slots__ = ()

    def test_step_function_splits_correctly(self):
        """Finds the correct split on a clear step function."""
        regression_tree = _helpers._fit_step_regression_tree()
        partition = regression_tree.content_.extension
        assert isinstance(partition, sigma._partition.NumericalPartition)
        self.assertEqual(partition.feature_index, 0)
        self.assertEqual(partition.thresholds[0], 20.0)
        left = typing.cast(sigma._node.RegressionNode, partition.children[0])
        right = typing.cast(sigma._node.RegressionNode, partition.children[1])
        numpy.testing.assert_allclose(left.prediction, 0.0)
        numpy.testing.assert_allclose(right.prediction, 10.0)

    def test_stump_max_depth_one(self):
        """Produces a stump with exactly two leaf children."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            max_depth=1,
            min_splits=2,
            min_buckets=1,
        )
        regression_tree.fit(X, y)
        extension = regression_tree.content_.extension
        assert isinstance(extension, sigma._partition.Partition)
        root_extension = typing.cast(
            sigma._partition.Partition[sigma._node.Node], extension
        )
        left_child, right_child = root_extension.children
        assert isinstance(left_child.extension, sigma._extension.Leaf)
        assert isinstance(right_child.extension, sigma._extension.Leaf)

    def test_alpha_zero_produces_single_leaf(self):
        """Returns a single leaf when alpha=0.0 rejects nothing."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", alpha=0.0
        )
        regression_tree.fit(X, y)
        assert isinstance(
            regression_tree.content_.extension, sigma._extension.Leaf
        )
        expected_mean = y.mean()
        numpy.testing.assert_allclose(
            regression_tree.content_.prediction, expected_mean
        )

    def test_constant_response_returns_leaf(self):
        """Returns a leaf when the response is constant."""
        X = numpy.arange(20, dtype=float).reshape(-1, 1)
        y = numpy.full(20, 5.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        regression_tree.fit(X, y)
        assert isinstance(
            regression_tree.content_.extension, sigma._extension.Leaf
        )
        numpy.testing.assert_allclose(regression_tree.content_.prediction, 5.0)

    def test_min_splits_prevents_split(self):
        """Returns a leaf when min_splits exceeds sample count."""
        X = numpy.arange(1, 11, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 5, 0.0, 10.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_buckets=1
        )
        regression_tree.fit(X, y)
        assert isinstance(
            regression_tree.content_.extension, sigma._extension.Leaf
        )

    def test_mixed_features(self):
        """Splits on the categorical feature when it carries the signal."""
        rng = numpy.random.default_rng(42)
        n = 40
        categorical_column = numpy.repeat([0.0, 1.0], n // 2)
        noise = rng.standard_normal(n)
        X = numpy.column_stack([categorical_column, noise])
        y = numpy.where(categorical_column == 0.0, 0.0, 10.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            categorical_features=[0],
            min_splits=2,
            min_buckets=1,
        )
        regression_tree.fit(X, y)
        partition = regression_tree.content_.extension
        assert isinstance(partition, sigma._partition.CategoricalPartition)
        self.assertEqual(partition.feature_index, 0)

    def test_maximum_test_stat_with_categorical(self):
        """Fits with maximum-type statistic on a multi-level categorical."""
        n = 60
        categorical_column = numpy.repeat([0.0, 1.0, 2.0], n // 3)
        X = categorical_column.reshape(-1, 1)
        y = numpy.where(
            categorical_column == 0.0,
            0.0,
            numpy.where(categorical_column == 1.0, 10.0, 20.0),
        )
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            test_stat="maximum",
            categorical_features=[0],
            min_splits=2,
            min_buckets=1,
        )
        regression_tree.fit(X, y)
        partition = regression_tree.content_.extension
        assert isinstance(partition, sigma._partition.CategoricalPartition)
        self.assertEqual(partition.feature_index, 0)

    def test_categorical_features_by_label(self):
        """Resolves a string entry against the fit DataFrame columns."""
        rng = numpy.random.default_rng(42)
        n = 40
        categorical_column = numpy.repeat([0.0, 1.0], n // 2)
        noise = rng.standard_normal(n)
        X = pandas.DataFrame({"category": categorical_column, "noise": noise})
        y = numpy.where(categorical_column == 0.0, 0.0, 10.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            categorical_features=["category"],
            min_splits=2,
            min_buckets=1,
        )
        regression_tree.fit(X, y)
        self.assertIsInstance(
            regression_tree.features_[0], sigma.CategoricalFeature
        )
        self.assertIsInstance(
            regression_tree.features_[1], sigma.NumericFeature
        )
        partition = regression_tree.content_.extension
        assert isinstance(partition, sigma._partition.CategoricalPartition)
        self.assertEqual(partition.feature_index, 0)

    def test_categorical_features_mixed_int_and_label(self):
        """Accepts a mix of integer indices and string labels."""
        rng = numpy.random.default_rng(0)
        n = 10
        X = pandas.DataFrame(
            {
                "a": rng.standard_normal(n),
                "b": rng.standard_normal(n),
                "c": rng.standard_normal(n),
            }
        )
        y = rng.standard_normal(n)
        regression_tree = sigma._tree_regression.RegressionTree(
            categorical_features=[0, "c"]
        )
        regression_tree.fit(X, y)
        self.assertIsInstance(
            regression_tree.features_[0], sigma.CategoricalFeature
        )
        self.assertIsInstance(
            regression_tree.features_[1], sigma.NumericFeature
        )
        self.assertIsInstance(
            regression_tree.features_[2], sigma.CategoricalFeature
        )

    def test_categorical_features_label_without_name_source(self):
        """Raises when a string label is used without any name source."""
        X = numpy.arange(20, dtype=float).reshape(-1, 2)
        y = numpy.arange(10, dtype=float)
        regression_tree = sigma._tree_regression.RegressionTree(
            categorical_features=["a"]
        )
        with self.assertRaises(ValueError) as context:
            regression_tree.fit(X, y)
        message = str(context.exception)
        assert "no feature names available" in message

    def test_categorical_features_unknown_label(self):
        """Raises when a label is not among the available feature names."""
        rng = numpy.random.default_rng(0)
        n = 10
        X = pandas.DataFrame(
            {"a": rng.standard_normal(n), "b": rng.standard_normal(n)}
        )
        y = rng.standard_normal(n)
        regression_tree = sigma._tree_regression.RegressionTree(
            categorical_features=["missing"]
        )
        with self.assertRaises(ValueError) as context:
            regression_tree.fit(X, y)
        message = str(context.exception)
        assert "missing" in message
        assert "a" in message
        assert "b" in message


class TestFeatureNames(unittest.TestCase):
    """Tests for fit-time feature_names_in_ and display-time feature_names."""

    __slots__ = ()

    def test_split_feature_name_from_dataframe_columns(self):
        """DataFrame columns populate the split feature name when fitting."""
        regression_tree = _helpers._fit_categorical_regression_tree(
            X_is_dataframe=True
        )
        extension = regression_tree.content_.extension
        assert isinstance(extension, sigma._partition.Partition)
        assert extension.feature.name == "category"

    def test_split_feature_name_none_without_name_source(self):
        """The split feature name is None when fit on a numpy array."""
        regression_tree = _helpers._fit_step_regression_tree()
        extension = regression_tree.content_.extension
        assert isinstance(extension, sigma._partition.Partition)
        assert extension.feature.name is None
        assert extension.feature_index == 0

    def test_to_text_uses_index_fallback_without_names(self):
        """to_text renders X[<index>] when no name source is available."""
        regression_tree = _helpers._fit_step_regression_tree()
        output = regression_tree.to_text()
        self.assertIn("X[0]", output)

    def test_to_text_feature_names_overrides_fit_names(self):
        """Display-time feature_names overrides any fit-time names in output."""
        regression_tree = _helpers._fit_categorical_regression_tree(
            X_is_dataframe=True
        )
        output = regression_tree.to_text(feature_names=["alt0", "alt1"])
        self.assertIn("alt0", output)
        self.assertNotIn("category", output)

    def test_to_text_feature_names_for_numpy_fit(self):
        """Display-time feature_names supplies names for numpy-fit estimators."""
        regression_tree = _helpers._fit_step_regression_tree()
        output = regression_tree.to_text(feature_names=["spread"])
        self.assertIn("spread", output)
        self.assertNotIn("X[0]", output)

    def test_predict_dataframe_column_mismatch_after_fit(self):
        """sklearn validate_data catches DataFrame-column drift at predict
        time.
        """
        rng = numpy.random.default_rng(0)
        n = 12
        X_fit = pandas.DataFrame(
            {"a": rng.standard_normal(n), "b": rng.standard_normal(n)}
        )
        y = rng.standard_normal(n)
        regression_tree = sigma._tree_regression.RegressionTree()
        regression_tree.fit(X_fit, y)
        X_predict = pandas.DataFrame(
            {"a": rng.standard_normal(3), "c": rng.standard_normal(3)}
        )
        with warnings.catch_warnings():
            warnings.simplefilter("error", UserWarning)
            with self.assertRaises((ValueError, UserWarning)):
                regression_tree.predict(X_predict)


class TestCategoryLabels(unittest.TestCase):
    """Tests for display-time category_labels resolution and rendering."""

    __slots__ = ()

    def test_category_labels_string_key_via_dataframe(self):
        """String key in category_labels resolves via fit-time DataFrame
        columns.
        """
        regression_tree = _helpers._fit_categorical_regression_tree(
            X_is_dataframe=True
        )
        output = regression_tree.to_text(
            category_labels={"category": {0.0: "red", 1.0: "blue"}}
        )
        has_label = "red" in output or "blue" in output
        self.assertTrue(has_label)

    def test_category_labels_string_key_via_display_feature_names(self):
        """String key resolves against display-time feature_names override."""
        regression_tree = _helpers._fit_categorical_regression_tree(
            X_is_dataframe=False
        )
        output = regression_tree.to_text(
            feature_names=["category", "noise"],
            category_labels={"category": {0.0: "red", 1.0: "blue"}},
        )
        has_label = "red" in output or "blue" in output
        self.assertTrue(has_label)

    def test_category_labels_unknown_string_key(self):
        """Raises when a category_labels string key is not a known column."""
        regression_tree = _helpers._fit_categorical_regression_tree(
            X_is_dataframe=True
        )
        with self.assertRaises(ValueError) as context:
            regression_tree.to_text(category_labels={"missing": {0.0: "x"}})
        message = str(context.exception)
        assert "missing" in message
        assert "category" in message
        assert "noise" in message


class TestToTextAndPlotContentSignatures(unittest.TestCase):
    """Tests confirming to_text / plot_content accept name kwargs."""

    __slots__ = ()

    def test_to_text_accepts_feature_names(self):
        """to_text accepts feature_names and renders the supplied name."""
        regression_tree = _helpers._fit_step_regression_tree()
        output = regression_tree.to_text(feature_names=["spread"])
        self.assertIn("spread", output)

    def test_to_text_accepts_response_name(self):
        """to_text accepts response_name and renders the supplied
        label.
        """
        regression_tree = _helpers._fit_step_regression_tree()
        output = regression_tree.to_text(response_name="Price")
        self.assertIn("Price mean", output)

    def test_to_text_accepts_category_labels(self):
        """to_text accepts category_labels with integer keys."""
        regression_tree = _helpers._fit_categorical_regression_tree(
            X_is_dataframe=False
        )
        output = regression_tree.to_text(
            category_labels={0: {0.0: "red", 1.0: "blue"}}
        )
        has_label = "red" in output or "blue" in output
        self.assertTrue(has_label)


class TestToTextMaxDepth(unittest.TestCase):
    """Tests for the max_depth display knob on to_text."""

    __slots__ = ()

    def _capture(self, regression_tree, **kwargs):
        """Run to_text with kwargs and return the resulting text."""
        output = regression_tree.to_text(**kwargs)
        return output

    def test_max_depth_zero_truncates_below_root(self):
        """Renders the root line and a single ... marker, with no branches."""
        regression_tree = _helpers._fit_three_step_regression_tree()
        output = self._capture(regression_tree, max_depth=0)
        self.assertIn("All records", output)
        self.assertEqual(output.count("..."), 1)
        self.assertNotIn("<=", output)

    def test_max_depth_one_renders_root_and_one_level(self):
        """Renders both depth-1 branch lines, with ... below non-leaf children."""
        regression_tree = _helpers._fit_three_step_regression_tree()
        output = self._capture(regression_tree, max_depth=1)
        self.assertIn("<=", output)
        root_extension = regression_tree.content_.extension
        assert isinstance(root_extension, sigma._partition.Partition)
        non_leaf_depth_one = sum(
            1
            for child in root_extension.children
            if isinstance(child.extension, sigma._partition.Partition)
        )
        self.assertEqual(output.count("..."), non_leaf_depth_one)

    def test_max_depth_exceeds_tree_depth_matches_full_output(self):
        """A max_depth above the actual tree depth equals the no-arg output."""
        regression_tree = _helpers._fit_three_step_regression_tree()
        full = self._capture(regression_tree)
        large = self._capture(regression_tree, max_depth=99)
        self.assertEqual(large, full)

    def test_max_depth_negative_raises_value_error(self):
        """A negative max_depth raises ValueError."""
        regression_tree = _helpers._fit_three_step_regression_tree()
        with self.assertRaises(ValueError):
            regression_tree.to_text(max_depth=-1)


class TestToTextPrecision(unittest.TestCase):
    """Tests for the precision display knob on to_text."""

    __slots__ = ()

    def _capture(self, estimator, **kwargs):
        """Run to_text with kwargs and return the resulting text."""
        output = estimator.to_text(**kwargs)
        return output

    def _fit_noisy_classification_tree(self):
        """Fit a binary classification tree with leaves that are not pure."""
        rng = numpy.random.default_rng(42)
        X = numpy.arange(1, 81, dtype=float).reshape(-1, 1)
        base = (X.ravel() > 40).astype(float)
        flip = rng.random(80) < 0.2
        y = numpy.where(flip, 1.0 - base, base)
        classification_tree = sigma._tree_classification.ClassificationTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        classification_tree.fit(X, y)
        return classification_tree

    def test_precision_zero_renders_threshold_with_no_decimals(self):
        """precision=0 strips digits after the decimal in split thresholds."""
        regression_tree = _helpers._fit_step_regression_tree()
        output = self._capture(regression_tree, precision=0)
        self.assertNotIn("20.5", output)

    def test_precision_five_renders_threshold_with_five_decimals(self):
        """precision=5 widens the threshold formatter to five fractional digits."""
        X = (numpy.arange(1, 41, dtype=float) + 0.25).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20.5, 0.0, 10.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        regression_tree.fit(X, y)
        output = self._capture(regression_tree, precision=5)
        self.assertIn("20.75000", output)

    def test_precision_does_not_change_scientific_notation_boundary(self):
        """The 1e-3 / 1e6 sci-notation switch is independent of precision."""
        X = (numpy.arange(1, 41, dtype=float) * 1e7 + 0.5).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20.5 * 1e7, 0.0, 10.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        regression_tree.fit(X, y)
        import re

        output_p1 = self._capture(regression_tree, precision=1)
        output_p5 = self._capture(regression_tree, precision=5)
        self.assertIsNotNone(re.search(r"\de[+-]\d+", output_p1))
        self.assertIsNotNone(re.search(r"\d\.\d{5}e[+-]\d+", output_p5))

    def test_precision_threads_into_classification_probability(self):
        """precision=3 produces 'NN.NNN%' classification probability cells."""
        classification_tree = self._fit_noisy_classification_tree()
        output = self._capture(classification_tree, precision=3)
        import re

        match = re.search(r"\d+\.\d{3}%", output)
        self.assertIsNotNone(match)

    def test_precision_one_matches_legacy_probability_format(self):
        """precision=1 reproduces the pre-Step-0.2 single-decimal format."""
        classification_tree = self._fit_noisy_classification_tree()
        output = self._capture(classification_tree, precision=1)
        import re

        self.assertIsNotNone(re.search(r"\d+\.\d%", output))
        self.assertIsNone(re.search(r"\d+\.\d{2,}%", output))

    def test_precision_preserves_zero_and_one_probability_short_circuits(self):
        """Probabilities exactly 0.0 and 1.0 still render as '0%' and '100%'."""
        classification_tree = _helpers._fit_step_classification_tree()
        output = self._capture(classification_tree, precision=5)
        self.assertNotIn(" 0.00000%", output)
        self.assertNotIn(" 100.00000%", output)
        self.assertIn(" 0%", output)
        self.assertIn(" 100%", output)

    def test_precision_skipped_when_prediction_formatter_supplied(self):
        """A user prediction_formatter overrides the default and ignores precision."""
        regression_tree = _helpers._fit_step_regression_tree()
        output = self._capture(
            regression_tree, precision=5, prediction_formatter=lambda v: "[fmt]"
        )
        self.assertIn("[fmt]", output)
        self.assertNotIn("= 0.00000", output)
        self.assertNotIn("= 10.00000", output)

    def test_precision_negative_raises_value_error(self):
        """A negative precision raises ValueError."""
        regression_tree = _helpers._fit_step_regression_tree()
        with self.assertRaises(ValueError):
            regression_tree.to_text(precision=-1)


class TestClassificationTreeFit(unittest.TestCase):
    """Tests for the fit method of ClassificationTree."""

    __slots__ = ()

    def test_binary_step_function_splits_correctly(self):
        """Finds the correct split on a two-class step function."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 1.0)
        classification_tree = sigma._tree_classification.ClassificationTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        classification_tree.fit(X, y)
        partition = classification_tree.content_.extension
        assert isinstance(partition, sigma._partition.NumericalPartition)
        self.assertEqual(partition.feature_index, 0)
        self.assertEqual(partition.thresholds[0], 20.0)

    def test_multiclass_splits_correctly(self):
        """Splits three-class data into meaningful groups."""
        n = 60
        X = numpy.arange(n, dtype=float).reshape(-1, 1)
        y = numpy.repeat([0.0, 1.0, 2.0], n // 3)
        classification_tree = sigma._tree_classification.ClassificationTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        classification_tree.fit(X, y)
        assert isinstance(
            classification_tree.content_.extension, sigma._partition.Partition
        )

    def test_constant_class_returns_leaf(self):
        """Returns a leaf when only one class is present."""
        X = numpy.arange(20, dtype=float).reshape(-1, 1)
        y = numpy.ones(20)
        classification_tree = sigma._tree_classification.ClassificationTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        classification_tree.fit(X, y)
        assert isinstance(
            classification_tree.content_.extension, sigma._extension.Leaf
        )
        preds = classification_tree.predict(X)
        numpy.testing.assert_allclose(preds, 1.0)

    def test_alpha_zero_produces_single_leaf(self):
        """Returns a single leaf when alpha=0.0 rejects nothing."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 1.0)
        classification_tree = sigma._tree_classification.ClassificationTree(
            correlation="normal", alpha=0.0
        )
        classification_tree.fit(X, y)
        assert isinstance(
            classification_tree.content_.extension, sigma._extension.Leaf
        )

    def test_classes_attribute_set_after_fit(self):
        """Sets classes_ to the sorted unique class labels."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 1.0)
        classification_tree = sigma._tree_classification.ClassificationTree(
            correlation="normal"
        )
        classification_tree.fit(X, y)
        numpy.testing.assert_allclose(
            classification_tree.classes_, numpy.array([0.0, 1.0])
        )
        assert classification_tree.n_classes_ == 2


class TestTreeCategoricalFormat(unittest.TestCase):
    """Tests for categorical split formatting."""

    __slots__ = ()

    def test_to_text_shows_categorical_branch_labels(self):
        """Shows category lists as branch labels for categorical splits."""
        rng = numpy.random.default_rng(42)
        n = 60
        categorical_column = numpy.repeat([0.0, 1.0, 2.0], n // 3)
        X = numpy.column_stack([categorical_column, rng.standard_normal(n)])
        y = numpy.where(categorical_column == 2.0, 10.0, 0.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            categorical_features=[0],
            min_splits=2,
            min_buckets=1,
        )
        regression_tree.fit(X, y)
        output = regression_tree.to_text()
        self.assertIn("All records", output)
        self.assertIn("Split p-value", output)
        lines = output.strip().split("\n")
        branch_lines = [
            line for line in lines if "├──" in line or "└──" in line
        ]
        self.assertTrue(len(branch_lines) >= 2)
        for branch_line in branch_lines:
            self.assertIn(" is ", branch_line)
        self.assertNotIn("Yes:", output)
        self.assertNotIn("No:", output)
