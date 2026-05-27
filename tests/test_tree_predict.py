"""Unit tests for prediction and traversal across Tree estimators."""

import typing
import unittest
import warnings

import numpy
import numpy.testing
import scipy.sparse
import sklearn.datasets
import sklearn.exceptions
import sklearn.model_selection

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


class TestRegressionTreePredict(unittest.TestCase):
    """Tests for the predict method of RegressionTree."""

    __slots__ = ()

    def test_predict_output_shape(self):
        """Returns an array with shape (n_samples,)."""
        X_train = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y_train = numpy.where(X_train.ravel() <= 20, 0.0, 10.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1, alpha=0.05
        )
        regression_tree.fit(X_train, y_train)
        X_test = numpy.arange(1, 11, dtype=float).reshape(-1, 1)
        preds = regression_tree.predict(X_test)
        assert preds.shape == (10,)

    def test_predict_values_match_leaves(self):
        """Predicts leaf means for samples in each partition."""
        X_train = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y_train = numpy.where(X_train.ravel() <= 20, 0.0, 10.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1, alpha=0.05
        )
        regression_tree.fit(X_train, y_train)
        X_left = numpy.array([[5.0], [10.0]])
        X_right = numpy.array([[25.0], [30.0]])
        preds_left = regression_tree.predict(X_left)
        preds_right = regression_tree.predict(X_right)
        numpy.testing.assert_allclose(preds_left, 0.0)
        numpy.testing.assert_allclose(preds_right, 10.0)

    def test_predict_single_leaf_returns_mean(self):
        """Returns the training mean for all inputs when tree is a leaf."""
        X_train = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y_train = numpy.where(X_train.ravel() <= 20, 0.0, 10.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", alpha=0.0
        )
        regression_tree.fit(X_train, y_train)
        X_test = numpy.array([[1.0], [20.0], [40.0]])
        preds = regression_tree.predict(X_test)
        expected = y_train.mean()
        numpy.testing.assert_allclose(preds, expected)


class TestRegressionTreeValidation(unittest.TestCase):
    """Tests for input validation in RegressionTree."""

    __slots__ = ()

    def test_predict_before_fit_raises(self):
        """Raises sklearn.exceptions.NotFittedError when predict is called
        before fit.
        """
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal"
        )
        X = numpy.array([[1.0], [2.0]])
        with self.assertRaises(sklearn.exceptions.NotFittedError):
            regression_tree.predict(X)

    def test_fit_stores_n_features_in(self):
        """Sets n_features_in_ to the number of columns after fit."""
        X = numpy.arange(60, dtype=float).reshape(20, 3)
        y = numpy.ones(20)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal"
        )
        regression_tree.fit(X, y)
        assert regression_tree.n_features_in_ == 3


class TestRegressionTreeDiabetes(unittest.TestCase):
    """Integration tests on the sklearn diabetes dataset."""

    __slots__ = ()

    def test_r2_above_threshold(self):
        """Achieves R² > 0.30 on diabetes via 5-fold cross-validation."""
        X, y = sklearn.datasets.load_diabetes(return_X_y=True)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=20, min_buckets=7, alpha=0.05
        )
        # CV fold subsets of the diabetes dataset have >50% unique
        # integer targets, triggering a spurious sklearn warning.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="The number of unique classes",
                category=UserWarning,
            )
            scores = sklearn.model_selection.cross_val_score(
                regression_tree, X, y, cv=5, scoring="r2"
            )
        mean_r2 = scores.mean()
        self.assertGreater(
            mean_r2,
            0.30,
            f"Mean R² = {mean_r2:.3f}, expected > 0.30",
        )

    def test_predictions_correlate_with_target(self):
        """Predictions correlate with the target (Pearson r > 0.55)."""
        X, y = sklearn.datasets.load_diabetes(return_X_y=True)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=20, min_buckets=7, alpha=0.05
        )
        regression_tree.fit(X, y)
        preds = regression_tree.predict(X)
        correlation = numpy.corrcoef(y, preds)[0, 1]
        self.assertGreater(
            correlation,
            0.55,
            f"Pearson r = {correlation:.3f}, expected > 0.55",
        )

    def test_tree_is_not_degenerate(self):
        """Produces a non-trivial tree (not a single leaf)."""
        X, y = sklearn.datasets.load_diabetes(return_X_y=True)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=20, min_buckets=7, alpha=0.05
        )
        regression_tree.fit(X, y)
        self.assertIsInstance(
            regression_tree.content_.extension,
            sigma._partition.Partition,
            "Tree should split on diabetes data, not return a single leaf",
        )


class TestClassificationTreePredict(unittest.TestCase):
    """Tests for the predict and predict_proba methods."""

    __slots__ = ()

    def test_predict_output_shape(self):
        """Returns an array with shape (n_samples,)."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 1.0)
        classification_tree = sigma._tree_classification.ClassificationTree(
            correlation="normal", min_splits=2, min_buckets=1, alpha=0.05
        )
        classification_tree.fit(X, y)
        preds = classification_tree.predict(X)
        assert preds.shape == (40,)

    def test_predict_returns_class_labels(self):
        """Predictions are drawn from classes_."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 1.0)
        classification_tree = sigma._tree_classification.ClassificationTree(
            correlation="normal", min_splits=2, min_buckets=1, alpha=0.05
        )
        classification_tree.fit(X, y)
        preds = classification_tree.predict(X)
        unique_preds = numpy.unique(preds)
        for p in unique_preds:
            assert p in classification_tree.classes_

    def test_predict_proba_shape(self):
        """Returns shape (n_samples, n_classes)."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 1.0)
        classification_tree = sigma._tree_classification.ClassificationTree(
            correlation="normal", min_splits=2, min_buckets=1, alpha=0.05
        )
        classification_tree.fit(X, y)
        proba = classification_tree.predict_proba(X)
        assert proba.shape == (40, 2)

    def test_predict_proba_sums_to_one(self):
        """Each row of predict_proba sums to 1."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 1.0)
        classification_tree = sigma._tree_classification.ClassificationTree(
            correlation="normal", min_splits=2, min_buckets=1, alpha=0.05
        )
        classification_tree.fit(X, y)
        proba = classification_tree.predict_proba(X)
        row_sums = proba.sum(axis=1)
        numpy.testing.assert_allclose(row_sums, 1.0)

    def test_predict_before_fit_raises(self):
        """Raises sklearn.exceptions.NotFittedError when predict is called
        before fit.
        """
        classification_tree = sigma._tree_classification.ClassificationTree(
            correlation="normal"
        )
        X = numpy.array([[1.0], [2.0]])
        with self.assertRaises(sklearn.exceptions.NotFittedError):
            classification_tree.predict(X)

    def test_predict_proba_before_fit_raises(self):
        """Raises sklearn.exceptions.NotFittedError when predict_proba is
        called before fit.
        """
        classification_tree = sigma._tree_classification.ClassificationTree(
            correlation="normal"
        )
        X = numpy.array([[1.0], [2.0]])
        with self.assertRaises(sklearn.exceptions.NotFittedError):
            classification_tree.predict_proba(X)


class TestClassificationTreeIris(unittest.TestCase):
    """Integration tests on the sklearn iris dataset."""

    __slots__ = ()

    def test_accuracy_above_threshold(self):
        """Achieves accuracy > 0.85 on iris via 5-fold CV."""
        X, y = sklearn.datasets.load_iris(return_X_y=True)
        classification_tree = sigma._tree_classification.ClassificationTree(
            correlation="normal",
            min_splits=20,
            min_buckets=7,
            alpha=0.05,
        )
        scores = sklearn.model_selection.cross_val_score(
            classification_tree,
            X,
            y,
            cv=5,
            scoring="accuracy",
        )
        mean_acc = scores.mean()
        self.assertGreater(
            mean_acc,
            0.85,
            f"Mean accuracy = {mean_acc:.3f}, expected > 0.85",
        )

    def test_roc_auc_above_threshold(self):
        """Achieves ROC AUC (OVR, weighted) > 0.90 on iris via 5-fold CV."""
        X, y = sklearn.datasets.load_iris(return_X_y=True)
        classification_tree = sigma._tree_classification.ClassificationTree(
            correlation="normal",
            min_splits=20,
            min_buckets=7,
            alpha=0.05,
        )
        scores = sklearn.model_selection.cross_val_score(
            classification_tree,
            X,
            y,
            cv=5,
            scoring="roc_auc_ovr_weighted",
        )
        mean_auc = scores.mean()
        self.assertGreater(
            mean_auc,
            0.90,
            f"Mean ROC AUC = {mean_auc:.3f}, expected > 0.90",
        )

    def test_tree_is_not_degenerate(self):
        """Produces a non-trivial tree (not a single leaf)."""
        X, y = sklearn.datasets.load_iris(return_X_y=True)
        classification_tree = sigma._tree_classification.ClassificationTree(
            correlation="normal",
            min_splits=20,
            min_buckets=7,
            alpha=0.05,
        )
        classification_tree.fit(X, y)
        self.assertIsInstance(
            classification_tree.content_.extension,
            sigma._partition.Partition,
            "Tree should split on iris data, not return a single leaf",
        )


class TestConstructorValidation(unittest.TestCase):
    """Constructor-time validation of all Literal parameters."""

    __slots__ = ()

    def _assert_regression_tree_kwargs_raise(
        self, kwargs: dict[str, object]
    ) -> None:
        """Assert the regression tree constructor raises ValueError for kwargs."""
        factory = typing.cast(
            typing.Callable[..., object], sigma._tree_regression.RegressionTree
        )
        with self.assertRaises(ValueError):
            factory(**kwargs)

    def _assert_classification_tree_kwargs_raise(
        self, kwargs: dict[str, object]
    ) -> None:
        """Assert the classification tree constructor raises ValueError for kwargs."""
        factory = typing.cast(
            typing.Callable[..., object],
            sigma._tree_classification.ClassificationTree,
        )
        with self.assertRaises(ValueError):
            factory(**kwargs)

    def test_regression_tree_invalid_correlation_raises(self):
        """Invalid correlation raises ValueError in RegressionTree.__init__."""
        self._assert_regression_tree_kwargs_raise({"correlation": "bogus"})

    def test_regression_tree_invalid_test_stat_raises(self):
        """Invalid test_stat raises ValueError in RegressionTree.__init__."""
        self._assert_regression_tree_kwargs_raise({"test_stat": "bogus"})

    def test_regression_tree_invalid_test_type_raises(self):
        """Invalid test_type raises ValueError in RegressionTree.__init__."""
        self._assert_regression_tree_kwargs_raise({"test_type": "bogus"})

    def test_regression_tree_invalid_ci_method_raises(self):
        """Invalid ci_method raises ValueError in RegressionTree.__init__."""
        self._assert_regression_tree_kwargs_raise({"ci_method": "bogus"})

    def test_regression_tree_classification_tree_only_ci_method_raises(self):
        """ClassificationTree-only ci_method on regression tree raises in __init__."""
        for value in _helpers._CLASSIFICATION_TREE_CI_METHODS:
            self._assert_regression_tree_kwargs_raise({"ci_method": value})

    def test_classification_tree_invalid_correlation_raises(self):
        """Invalid correlation raises ValueError in ClassificationTree.__init__."""
        self._assert_classification_tree_kwargs_raise({"correlation": "bogus"})

    def test_classification_tree_invalid_test_stat_raises(self):
        """Invalid test_stat raises ValueError in ClassificationTree.__init__."""
        self._assert_classification_tree_kwargs_raise({"test_stat": "bogus"})

    def test_classification_tree_invalid_test_type_raises(self):
        """Invalid test_type raises ValueError in ClassificationTree.__init__."""
        self._assert_classification_tree_kwargs_raise({"test_type": "bogus"})

    def test_classification_tree_invalid_ci_method_raises(self):
        """Invalid ci_method raises ValueError in ClassificationTree.__init__."""
        self._assert_classification_tree_kwargs_raise({"ci_method": "bogus"})

    def test_classification_tree_regression_tree_only_ci_method_raises(self):
        """RegressionTree-only ci_method on classification tree raises in __init__."""
        for value in _helpers._REGRESSION_TREE_CI_METHODS:
            self._assert_classification_tree_kwargs_raise({"ci_method": value})


class TestClassificationTreePredictionOrder(unittest.TestCase):
    """Tests for classification tree prediction display order."""

    __slots__ = ()

    def test_to_text_lists_classes_in_natural_order(self):
        """Classes appear in label order, not sorted by probability."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 1.0)
        classification_tree = sigma._tree_classification.ClassificationTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            alpha=0.05,
        )
        classification_tree.fit(X, y)
        import re

        output = classification_tree.to_text(class_names=["cat", "dog"])
        cat_lines = [m.start() for m in re.finditer(r"Cat proba\.", output)]
        dog_lines = [m.start() for m in re.finditer(r"Dog proba\.", output)]
        self.assertEqual(len(cat_lines), len(dog_lines))
        for cat_pos, dog_pos in zip(cat_lines, dog_lines):
            self.assertLess(cat_pos, dog_pos)


class TestDisplayChildOrder(unittest.TestCase):
    """Tests for child display ordering by best leaf first."""

    __slots__ = ()

    def test_regression_first_branch_has_larger_prediction(self):
        """The first displayed branch has the larger prediction value."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1, alpha=0.05
        )
        regression_tree.fit(X, y)
        import re

        output = regression_tree.to_text()
        lines = output.splitlines()
        header_line = lines[0]
        prediction_start = header_line.index("Predicted mean")
        count_start = header_line.index("Obs. count")
        branch_lines = [
            line for line in lines if "├──" in line or "└──" in line
        ]
        predictions = []
        for line in branch_lines:
            cell = line[prediction_start:count_start].strip()
            match = re.search(r"([\d.]+)", cell)
            if match:
                predictions.append(float(match.group(1)))
        self.assertEqual(len(predictions), 2)
        self.assertGreaterEqual(predictions[0], predictions[1])

    def test_classification_first_branch_has_smaller_distribution(self):
        """The first displayed branch has the lexicographically smaller
        distribution.
        """
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 1.0)
        classification_tree = sigma._tree_classification.ClassificationTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            alpha=0.05,
        )
        classification_tree.fit(X, y)
        import re

        output = classification_tree.to_text(class_names=["a", "b"])
        lines = output.splitlines()
        header_line = lines[0]
        column_starts = [
            header_line.index("A proba."),
            header_line.index("B proba."),
            header_line.index("Obs. count"),
        ]
        branch_lines = [
            line for line in lines if "├──" in line or "└──" in line
        ]
        distributions = []
        for line in branch_lines:
            cells = []
            for i in range(len(column_starts) - 1):
                cell = line[column_starts[i] : column_starts[i + 1]].strip()
                match = re.search(r"([\d.]+)%", cell)
                if match:
                    cells.append(float(match.group(1)))
            if cells:
                distributions.append(tuple(cells))
        self.assertEqual(len(distributions), 2)
        self.assertLessEqual(distributions[0], distributions[1])


class TestRegressionTreePredictIndex(unittest.TestCase):
    """Tests for predict_index on RegressionTree."""

    __slots__ = ()

    def test_output_shape_and_dtype(self):
        """Returns an integer array with shape (n_samples,)."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1, alpha=0.05
        )
        regression_tree.fit(X, y)
        indices = regression_tree.predict_index(X)
        self.assertEqual(indices.shape, (40,))
        self.assertTrue(numpy.issubdtype(indices.dtype, numpy.integer))

    def test_values_in_range(self):
        """All indices fall in [0, len(leaves_))."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1, alpha=0.05
        )
        regression_tree.fit(X, y)
        indices = regression_tree.predict_index(X)
        n_leaves = len(regression_tree.leaves_)
        self.assertTrue(numpy.all(indices >= 0))
        self.assertTrue(numpy.all(indices < n_leaves))

    def test_consistent_with_predict(self):
        """Leaf predictions via indices match predict output."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1, alpha=0.05
        )
        regression_tree.fit(X, y)
        indices = regression_tree.predict_index(X)
        from_index = numpy.array(
            [regression_tree.leaves_[i].prediction for i in indices]
        )
        from_predict = regression_tree.predict(X)
        numpy.testing.assert_allclose(from_index, from_predict)

    def test_before_fit_raises(self):
        """Raises sklearn.exceptions.NotFittedError when called before fit."""
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal"
        )
        X = numpy.array([[1.0], [2.0]])
        with self.assertRaises(sklearn.exceptions.NotFittedError):
            regression_tree.predict_index(X)

    def test_single_leaf_returns_zero(self):
        """Returns index 0 for all samples when tree is a single leaf."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", alpha=0.0
        )
        regression_tree.fit(X, y)
        indices = regression_tree.predict_index(X)
        numpy.testing.assert_allclose(indices, 0)


class TestClassificationTreePredictIndex(unittest.TestCase):
    """Tests for predict_index on ClassificationTree."""

    __slots__ = ()

    def test_output_shape_and_dtype(self):
        """Returns an integer array with shape (n_samples,)."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 1.0)
        classification_tree = sigma._tree_classification.ClassificationTree(
            correlation="normal", min_splits=2, min_buckets=1, alpha=0.05
        )
        classification_tree.fit(X, y)
        indices = classification_tree.predict_index(X)
        self.assertEqual(indices.shape, (40,))
        self.assertTrue(numpy.issubdtype(indices.dtype, numpy.integer))

    def test_consistent_with_predict(self):
        """Class labels via indices match predict output."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 1.0)
        classification_tree = sigma._tree_classification.ClassificationTree(
            correlation="normal", min_splits=2, min_buckets=1, alpha=0.05
        )
        classification_tree.fit(X, y)
        indices = classification_tree.predict_index(X)
        class_indices = numpy.array(
            [int(classification_tree.leaves_[i].prediction) for i in indices]
        )
        from_index = classification_tree.classes_[class_indices]
        from_predict = classification_tree.predict(X)
        numpy.testing.assert_allclose(from_index, from_predict)

    def test_consistent_with_predict_proba(self):
        """Class distributions via indices match predict_proba output."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 1.0)
        classification_tree = sigma._tree_classification.ClassificationTree(
            correlation="normal", min_splits=2, min_buckets=1, alpha=0.05
        )
        classification_tree.fit(X, y)
        indices = classification_tree.predict_index(X)
        from_index = numpy.array(
            [classification_tree.leaves_[i].class_distribution for i in indices]
        )
        from_proba = classification_tree.predict_proba(X)
        numpy.testing.assert_allclose(from_index, from_proba)


class TestLeafId(unittest.TestCase):
    """Tests for the Leaf.leaf_id field after fit."""

    __slots__ = ()

    def _walk_internal_nodes(
        self, root: sigma._node.Node
    ) -> list[sigma._node.Node]:
        """Collect every node in the subtree that has a partition."""
        collected: list[sigma._node.Node] = []
        stack: list[sigma._node.Node] = [root]
        while stack:
            node = stack.pop()
            if isinstance(node.extension, sigma._partition.Partition):
                partition = typing.cast(
                    sigma._partition.Partition[sigma._node.Node], node.extension
                )
                collected.append(node)
                stack.append(partition.left)
                stack.append(partition.right)
        return collected

    def test_internal_nodes_have_no_leaf_id(self):
        """Every internal node carries a Partition extension with no leaf_id field."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1, alpha=0.05
        )
        regression_tree.fit(X, y)
        internal_nodes = self._walk_internal_nodes(regression_tree.content_)
        self.assertGreater(len(internal_nodes), 0)
        for node in internal_nodes:
            self.assertIsInstance(node.extension, sigma._partition.Partition)
            self.assertFalse(hasattr(node.extension, "leaf_id"))

    def test_leaf_id_consecutive_for_leaves(self):
        """Leaves carry leaf_id 0..N-1 in Tree.leaves_ order."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1, alpha=0.05
        )
        regression_tree.fit(X, y)
        observed = []
        for leaf in regression_tree.leaves_:
            extension = leaf.extension
            assert isinstance(extension, sigma._extension.Leaf)
            observed.append(extension.leaf_id)
        expected = list(range(len(regression_tree.leaves_)))
        self.assertEqual(observed, expected)

    def test_leaf_id_matches_predict_index(self):
        """leaf_id of each routed leaf equals predict_index for that sample."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1, alpha=0.05
        )
        regression_tree.fit(X, y)
        indices = regression_tree.predict_index(X)
        for i in range(X.shape[0]):
            leaf = regression_tree.content_.traverse(X[i])
            extension = leaf.extension
            assert isinstance(extension, sigma._extension.Leaf)
            self.assertEqual(extension.leaf_id, indices[i])

    def test_leaf_id_zero_for_single_leaf_root(self):
        """A fitted tree with no splits has content_.extension.leaf_id equal to 0."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", alpha=0.0
        )
        regression_tree.fit(X, y)
        extension = regression_tree.content_.extension
        assert isinstance(extension, sigma._extension.Leaf)
        self.assertEqual(extension.leaf_id, 0)

    def test_leaf_id_assignment_reverses_under_reverse_order(self):
        """A reverse-ordered tree assigns leaf_ids on a reversed leaves list."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        natural_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1, alpha=0.05
        )
        natural_tree.fit(X, y)
        reversed_tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            alpha=0.05,
            reverse_order=True,
        )
        reversed_tree.fit(X, y)
        natural_predictions = [leaf.prediction for leaf in natural_tree.leaves_]
        reversed_predictions = [
            leaf.prediction for leaf in reversed_tree.leaves_
        ]
        self.assertEqual(
            reversed_predictions, list(reversed(natural_predictions))
        )


class TestNodeId(unittest.TestCase):
    """Tests for the Node.node_id field and Tree.nodes_ attribute."""

    __slots__ = ()

    def _fit_three_step_tree(self) -> sigma._tree_regression.RegressionTree:
        """Fit a regression tree on a three-step response with splits."""
        X = numpy.arange(1, 81, dtype=float).reshape(-1, 1)
        y = numpy.where(
            X.ravel() < 20, 0.0, numpy.where(X.ravel() < 60, 5.0, 10.0)
        )
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1, alpha=0.05
        )
        regression_tree.fit(X, y)
        return regression_tree

    def test_root_has_node_id_zero(self):
        """The fitted root has node_id 0 and is the first element of nodes_."""
        regression_tree = self._fit_three_step_tree()
        self.assertEqual(regression_tree.content_.node_id, 0)
        self.assertIs(regression_tree.nodes_[0], regression_tree.content_)

    def test_nodes_round_trip(self):
        """nodes_[x].node_id == x for every x in [0, len(nodes_))."""
        regression_tree = self._fit_three_step_tree()
        for x in range(len(regression_tree.nodes_)):
            self.assertEqual(regression_tree.nodes_[x].node_id, x)

    def test_node_ids_are_unique(self):
        """Every node carries a distinct node_id."""
        regression_tree = self._fit_three_step_tree()
        ids = [node.node_id for node in regression_tree.nodes_]
        self.assertEqual(len(set(ids)), len(ids))
        self.assertEqual(len(ids), len(regression_tree.nodes_))

    def test_node_ids_cover_full_range(self):
        """Node ids form a contiguous 0..M-1 range."""
        regression_tree = self._fit_three_step_tree()
        ids = sorted(node.node_id for node in regression_tree.nodes_)
        self.assertEqual(ids, list(range(len(regression_tree.nodes_))))

    def test_node_ids_are_pre_order(self):
        """Each parent's node_id is smaller than either child's."""
        regression_tree = self._fit_three_step_tree()
        for node in regression_tree.nodes_:
            extension = node.extension
            if not isinstance(extension, sigma._partition.Partition):
                continue
            partition = typing.cast(
                sigma._partition.Partition[sigma._node.Node], extension
            )
            assert node.node_id is not None
            assert partition.left.node_id is not None
            assert partition.right.node_id is not None
            self.assertLess(node.node_id, partition.left.node_id)
            self.assertLess(node.node_id, partition.right.node_id)

    def test_node_ids_unchanged_by_reverse_order(self):
        """node_ids are assigned identically regardless of reverse_order."""
        natural_tree = self._fit_three_step_tree()
        X = numpy.arange(1, 81, dtype=float).reshape(-1, 1)
        y = numpy.where(
            X.ravel() < 20, 0.0, numpy.where(X.ravel() < 60, 5.0, 10.0)
        )
        reversed_tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            alpha=0.05,
            reverse_order=True,
        )
        reversed_tree.fit(X, y)
        natural_ids = [node.node_id for node in natural_tree.nodes_]
        reversed_ids = [node.node_id for node in reversed_tree.nodes_]
        self.assertEqual(natural_ids, reversed_ids)

    def test_internal_nodes_have_node_id(self):
        """Internal (non-leaf) nodes have a non-None node_id after fit."""
        regression_tree = self._fit_three_step_tree()
        for node in regression_tree.nodes_:
            if isinstance(node.extension, sigma._partition.Partition):
                self.assertIsNotNone(node.node_id)


class TestApply(unittest.TestCase):
    """Tests for the apply method on Tree."""

    __slots__ = ()

    def _fit_step_regression_tree(
        self,
    ) -> sigma._tree_regression.RegressionTree:
        """Fit a regression tree on a step function response."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1, alpha=0.05
        )
        regression_tree.fit(X, y)
        return regression_tree

    def test_output_shape_and_dtype(self):
        """apply returns an integer array with shape (n_samples,)."""
        regression_tree = self._fit_step_regression_tree()
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        ids = regression_tree.apply(X)
        self.assertEqual(ids.shape, (40,))
        self.assertTrue(numpy.issubdtype(ids.dtype, numpy.integer))

    def test_apply_returns_leaf_node_ids(self):
        """Every returned id points to a leaf in nodes_."""
        regression_tree = self._fit_step_regression_tree()
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        ids = regression_tree.apply(X)
        for value in ids:
            node = regression_tree.nodes_[int(value)]
            self.assertIsInstance(node.extension, sigma._extension.Leaf)

    def test_apply_consistent_with_manual_traversal(self):
        """apply(X)[i] equals content_.traverse(X[i]).node_id for every i."""
        regression_tree = self._fit_step_regression_tree()
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        ids = regression_tree.apply(X)
        for i in range(X.shape[0]):
            leaf = regression_tree.content_.traverse(X[i])
            self.assertEqual(int(ids[i]), leaf.node_id)

    def test_apply_before_fit_raises(self):
        """apply called before fit raises NotFittedError."""
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal"
        )
        X = numpy.array([[1.0], [2.0]])
        with self.assertRaises(sklearn.exceptions.NotFittedError):
            regression_tree.apply(X)

    def test_apply_single_leaf_returns_zero(self):
        """A fitted tree with no splits returns 0 for every sample."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", alpha=0.0
        )
        regression_tree.fit(X, y)
        ids = regression_tree.apply(X)
        numpy.testing.assert_array_equal(ids, 0)


class TestDecisionPath(unittest.TestCase):
    """Tests for the decision_path method on Tree."""

    __slots__ = ()

    def _fit_three_step_tree(self) -> sigma._tree_regression.RegressionTree:
        """Fit a regression tree with several splits."""
        X = numpy.arange(1, 81, dtype=float).reshape(-1, 1)
        y = numpy.where(
            X.ravel() < 20, 0.0, numpy.where(X.ravel() < 60, 5.0, 10.0)
        )
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1, alpha=0.05
        )
        regression_tree.fit(X, y)
        return regression_tree

    def test_output_type_and_shape(self):
        """decision_path returns CSR of shape (n_samples, len(nodes_))."""
        regression_tree = self._fit_three_step_tree()
        X = numpy.arange(1, 81, dtype=float).reshape(-1, 1)
        path = regression_tree.decision_path(X)
        self.assertIsInstance(path, scipy.sparse.csr_matrix)
        self.assertEqual(path.shape, (80, len(regression_tree.nodes_)))
        self.assertTrue(numpy.issubdtype(path.dtype, numpy.integer))

    def test_root_column_always_one(self):
        """Every sample's path contains the root node (column 0)."""
        regression_tree = self._fit_three_step_tree()
        X = numpy.arange(1, 81, dtype=float).reshape(-1, 1)
        path = regression_tree.decision_path(X)
        root_column = path[:, 0].toarray().ravel()
        numpy.testing.assert_array_equal(root_column, 1)

    def test_last_visited_equals_apply(self):
        """The highest-numbered visited node in each row matches apply(X)."""
        regression_tree = self._fit_three_step_tree()
        X = numpy.arange(1, 81, dtype=float).reshape(-1, 1)
        path = regression_tree.decision_path(X).toarray()
        ids = regression_tree.apply(X)
        for i in range(X.shape[0]):
            visited = numpy.flatnonzero(path[i])
            self.assertEqual(int(visited.max()), int(ids[i]))

    def test_row_sums_equal_path_length(self):
        """Each row sum equals the depth of the routed leaf plus 1."""
        regression_tree = self._fit_three_step_tree()
        X = numpy.arange(1, 81, dtype=float).reshape(-1, 1)
        path = regression_tree.decision_path(X)
        row_sums = path.sum(axis=1).A1
        for i in range(X.shape[0]):
            leaf = regression_tree.content_.traverse(X[i])
            self.assertEqual(int(row_sums[i]), leaf.depth + 1)

    def test_decision_path_before_fit_raises(self):
        """decision_path called before fit raises NotFittedError."""
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal"
        )
        X = numpy.array([[1.0], [2.0]])
        with self.assertRaises(sklearn.exceptions.NotFittedError):
            regression_tree.decision_path(X)


class TestRegressionTreeLeaves(unittest.TestCase):
    """Tests for the leaves_ attribute on RegressionTree."""

    __slots__ = ()

    def test_length_matches_leaf_count(self):
        """Number of leaves_ matches the count of leaf nodes in the tree."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1, alpha=0.05
        )
        regression_tree.fit(X, y)
        tree_leaves = [
            n
            for n in _helpers._collect_nodes(regression_tree.content_)
            if isinstance(n.extension, sigma._extension.Leaf)
        ]
        self.assertEqual(len(regression_tree.leaves_), len(tree_leaves))

    def test_shares_sum_to_one(self):
        """Sum of leaf shares equals 1.0."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1, alpha=0.05
        )
        regression_tree.fit(X, y)
        total_share = sum(leaf.share for leaf in regression_tree.leaves_)
        numpy.testing.assert_allclose(total_share, 1.0)

    def test_ordered_by_prediction(self):
        """Leaves are sorted by ascending prediction value."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1, alpha=0.05
        )
        regression_tree.fit(X, y)
        predictions = [leaf.prediction for leaf in regression_tree.leaves_]
        self.assertEqual(predictions, sorted(predictions))

    def test_n_samples_sum_equals_total(self):
        """Sum of leaf n_samples equals root n_samples."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1, alpha=0.05
        )
        regression_tree.fit(X, y)
        total = sum(leaf.n_samples for leaf in regression_tree.leaves_)
        self.assertEqual(total, regression_tree.content_.n_samples)

    def test_leaves_have_ci(self):
        """Leaves include CI bounds when ci_coverage is set."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1, alpha=0.05
        )
        regression_tree.fit(X, y)
        for leaf in regression_tree.leaves_:
            self.assertIsNotNone(leaf.ci_low)
            self.assertIsNotNone(leaf.ci_high)

    def test_leaves_no_ci_when_disabled(self):
        """Leaves have None CI when ci_coverage is None."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            alpha=0.05,
            ci_coverage=None,
        )
        regression_tree.fit(X, y)
        for leaf in regression_tree.leaves_:
            self.assertIsNone(leaf.ci_low)
            self.assertIsNone(leaf.ci_high)

    def test_regression_leaves_are_regression_nodes(self):
        """Each leaf of a RegressionTree is an instance of RegressionNode."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1, alpha=0.05
        )
        regression_tree.fit(X, y)
        for leaf in regression_tree.leaves_:
            self.assertIsInstance(leaf, sigma._node.RegressionNode)
            self.assertFalse(hasattr(leaf, "class_distribution"))


class TestClassificationTreeLeaves(unittest.TestCase):
    """Tests for the leaves_ attribute on ClassificationTree."""

    __slots__ = ()

    def test_length_matches_leaf_count(self):
        """Number of leaves_ matches the count of leaf nodes in the tree."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 1.0)
        classification_tree = sigma._tree_classification.ClassificationTree(
            correlation="normal", min_splits=2, min_buckets=1, alpha=0.05
        )
        classification_tree.fit(X, y)
        tree_leaves = [
            n
            for n in _helpers._collect_nodes(classification_tree.content_)
            if isinstance(n.extension, sigma._extension.Leaf)
        ]
        self.assertEqual(len(classification_tree.leaves_), len(tree_leaves))

    def test_shares_sum_to_one(self):
        """Sum of leaf shares equals 1.0."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 1.0)
        classification_tree = sigma._tree_classification.ClassificationTree(
            correlation="normal", min_splits=2, min_buckets=1, alpha=0.05
        )
        classification_tree.fit(X, y)
        total_share = sum(leaf.share for leaf in classification_tree.leaves_)
        numpy.testing.assert_allclose(total_share, 1.0)

    def test_leaves_have_class_distribution(self):
        """Classification leaves include a class distribution array."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 1.0)
        classification_tree = sigma._tree_classification.ClassificationTree(
            correlation="normal", min_splits=2, min_buckets=1, alpha=0.05
        )
        classification_tree.fit(X, y)
        for leaf in classification_tree.leaves_:
            self.assertIsNotNone(leaf.class_distribution)
            distribution = leaf.class_distribution
            if distribution is None:
                continue
            dist_sum = distribution.sum()
            numpy.testing.assert_allclose(dist_sum, 1.0)

    def test_leaves_no_ci_when_disabled(self):
        """Classification leaves have None per-class CI when ci_coverage is None."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 1.0)
        classification_tree = sigma._tree_classification.ClassificationTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            alpha=0.05,
            ci_coverage=None,
        )
        classification_tree.fit(X, y)
        for leaf in classification_tree.leaves_:
            self.assertIsNone(leaf.ci_low)
            self.assertIsNone(leaf.ci_high)
