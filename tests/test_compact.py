"""Unit tests for Tree.compact() binary-to-N-ary compaction."""

import unittest

import numpy
import numpy.testing

import sigma
import sigma._partition
import sigma._tree_classification
import sigma._tree_ranking
import sigma._tree_regression
import sigma._tree_survival


def _fit_nested_numeric_regression():
    """Fit a regression tree splitting one numeric feature into four levels."""
    X = numpy.arange(1, 81, dtype=float).reshape(-1, 1)
    values = X.ravel()
    y = numpy.select(
        [values <= 20, values <= 40, values <= 60],
        [0.0, 10.0, 20.0],
        default=30.0,
    )
    tree = sigma._tree_regression.RegressionTree(
        correlation="normal", min_splits=2, min_buckets=1, ci_coverage=None
    )
    tree.fit(X, y)
    return tree, X


def _fit_nested_categorical_regression():
    """Fit a regression tree splitting one categorical feature per category."""
    rng = numpy.random.default_rng(0)
    categories = numpy.repeat(numpy.array([0.0, 1.0, 2.0, 3.0]), 60)
    y = categories * 100.0 + rng.standard_normal(categories.size) * 0.1
    X = categories.reshape(-1, 1)
    tree = sigma._tree_regression.RegressionTree(
        correlation="normal",
        categorical_features=[0],
        min_splits=2,
        min_buckets=1,
        ci_coverage=None,
        random_state=0,
    )
    tree.fit(X, y)
    return tree, X


def _fit_step_regression():
    """Fit a regression tree on a single step, giving one split and two leaves."""
    X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
    y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
    tree = sigma._tree_regression.RegressionTree(
        correlation="normal",
        min_splits=2,
        min_buckets=1,
        ci_coverage=None,
    )
    tree.fit(X, y)
    return tree, X


def _fit_nested_numeric_classification():
    """Fit a classification tree splitting one numeric feature into four levels."""
    X = numpy.arange(1, 81, dtype=float).reshape(-1, 1)
    values = X.ravel()
    y = numpy.select(
        [values <= 20, values <= 40, values <= 60],
        [0, 1, 2],
        default=3,
    )
    tree = sigma._tree_classification.ClassificationTree(
        correlation="normal", min_splits=2, min_buckets=1, ci_coverage=None
    )
    tree.fit(X, y)
    return tree, X


def _fit_numeric_survival():
    """Fit a survival tree on a stepped numeric feature."""
    X = numpy.arange(1, 81, dtype=float).reshape(-1, 1)
    values = X.ravel()
    time = numpy.select(
        [values <= 20, values <= 40, values <= 60],
        [1.0, 5.0, 9.0],
        default=13.0,
    )
    event = numpy.tile([1.0, 1.0, 1.0, 0.0], 20)
    y = numpy.column_stack([time, event])
    tree = sigma._tree_survival.SurvivalTree(min_splits=2, min_buckets=1)
    tree.fit(X, y)
    return tree, X


def _fit_numeric_ranking():
    """Fit a ranking tree on a stepped numeric feature."""
    X = numpy.arange(1, 81, dtype=float).reshape(-1, 1)
    values = X.ravel()
    orders = numpy.array(
        [
            [1.0, 2.0, 3.0],
            [2.0, 3.0, 1.0],
            [3.0, 1.0, 2.0],
            [1.0, 3.0, 2.0],
        ]
    )
    group = numpy.select(
        [values <= 20, values <= 40, values <= 60], [0, 1, 2], default=3
    )
    y = orders[group]
    tree = sigma._tree_ranking.RankingTree(
        pca_components=2,
        min_splits=2,
        min_buckets=1,
        ci_replicates=5,
        random_state=0,
    )
    tree.fit(X, y)
    return tree, X


def _collect_numeric_thresholds(tree) -> list[float]:
    """Collect every numeric split threshold across a fitted tree."""
    thresholds: list[float] = []
    for node in tree.nodes_:
        extension = node.extension
        if isinstance(extension, sigma._partition.NumericalPartition):
            for threshold in extension.thresholds:
                thresholds.append(float(threshold))
    return thresholds


class TestNumericCompaction(unittest.TestCase):
    """Tests for compacting recursive numeric splits."""

    __slots__ = ()

    def test_chain_collapses_to_single_multiway_node(self):
        """A recursive single-feature numeric tree collapses to one node."""
        tree, _ = _fit_nested_numeric_regression()
        compacted = tree.compact()
        root = compacted.content_.extension
        self.assertIsInstance(root, sigma._partition.NumericalPartition)
        self.assertEqual(len(root.children), len(tree.leaves_))

    def test_merged_thresholds_are_sorted_union_of_originals(self):
        """The merged node's thresholds are the sorted original cut points."""
        tree, _ = _fit_nested_numeric_regression()
        compacted = tree.compact()
        root = compacted.content_.extension
        assert isinstance(root, sigma._partition.NumericalPartition)
        expected = sorted(set(_collect_numeric_thresholds(tree)))
        actual = [float(threshold) for threshold in root.thresholds]
        self.assertEqual(actual, expected)

    def test_merged_node_has_no_statistics(self):
        """A merged node exposes no split statistics."""
        tree, _ = _fit_nested_numeric_regression()
        compacted = tree.compact()
        self.assertIsNone(compacted.content_.extension.statistics)

    def test_predict_matches_original_on_training_data(self):
        """Predictions are identical before and after compaction."""
        tree, X = _fit_nested_numeric_regression()
        compacted = tree.compact()
        numpy.testing.assert_array_equal(tree.predict(X), compacted.predict(X))

    def test_predict_matches_original_outside_training_range(self):
        """Predictions match for values below and above every threshold."""
        tree, _ = _fit_nested_numeric_regression()
        compacted = tree.compact()
        probe = numpy.array([[-100.0], [0.5], [1000.0]])
        numpy.testing.assert_array_equal(
            tree.predict(probe), compacted.predict(probe)
        )


class TestCategoricalCompaction(unittest.TestCase):
    """Tests for compacting recursive categorical splits."""

    __slots__ = ()

    def test_chain_collapses_to_one_group_per_category(self):
        """A recursive categorical tree collapses to one group per category."""
        tree, _ = _fit_nested_categorical_regression()
        compacted = tree.compact()
        root = compacted.content_.extension
        self.assertIsInstance(root, sigma._partition.CategoricalPartition)
        self.assertEqual(len(root.children), len(tree.leaves_))

    def test_observed_categories_preserved(self):
        """The merged node observes the same categories as the source tree."""
        tree, _ = _fit_nested_categorical_regression()
        compacted = tree.compact()
        root = compacted.content_.extension
        assert isinstance(root, sigma._partition.CategoricalPartition)
        self.assertEqual(
            root.observed_categories, frozenset({0.0, 1.0, 2.0, 3.0})
        )

    def test_predict_matches_original_including_unseen_category(self):
        """Seen and unseen categories predict identically after compaction."""
        tree, X = _fit_nested_categorical_regression()
        compacted = tree.compact()
        probe = numpy.array([[0.0], [1.0], [2.0], [3.0], [99.0]])
        numpy.testing.assert_array_equal(
            tree.predict(probe), compacted.predict(probe)
        )
        numpy.testing.assert_array_equal(tree.predict(X), compacted.predict(X))


class TestClassificationCompaction(unittest.TestCase):
    """Tests that classification probabilities survive compaction."""

    __slots__ = ()

    def test_predict_proba_matches_original(self):
        """Class probabilities are identical before and after compaction."""
        tree, X = _fit_nested_numeric_classification()
        compacted = tree.compact()
        numpy.testing.assert_allclose(
            tree.predict_proba(X), compacted.predict_proba(X)
        )


class TestNonMergingCompaction(unittest.TestCase):
    """Tests that nodes with no same-feature chain are left intact."""

    __slots__ = ()

    def test_single_split_keeps_statistics_and_two_children(self):
        """A stump has no chain to merge, so it keeps its split statistics."""
        tree, X = _fit_step_regression()
        compacted = tree.compact()
        root = compacted.content_.extension
        assert isinstance(root, sigma._partition.Partition)
        self.assertEqual(len(root.children), 2)
        self.assertIsNotNone(root.statistics)
        numpy.testing.assert_array_equal(tree.predict(X), compacted.predict(X))

    def test_single_leaf_tree_compacts_to_equivalent_leaf(self):
        """A tree that never splits compacts to an equivalent single leaf."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.zeros(X.shape[0], dtype=float)
        tree = sigma._tree_regression.RegressionTree(
            correlation="normal", ci_coverage=None
        )
        tree.fit(X, y)
        compacted = tree.compact()
        self.assertIsInstance(
            compacted.content_.extension, sigma._partition.Leaf
        )
        numpy.testing.assert_array_equal(tree.predict(X), compacted.predict(X))


class TestCompactionLeavesOriginalIntact(unittest.TestCase):
    """Tests that compaction does not mutate the source tree."""

    __slots__ = ()

    def test_original_structure_unchanged(self):
        """The source tree keeps its node count, ids and binary root."""
        tree, _ = _fit_nested_numeric_regression()
        original_node_count = len(tree.nodes_)
        original_ids = [node.node_id for node in tree.nodes_]
        original_leaf_ids = [leaf.extension.leaf_id for leaf in tree.leaves_]
        tree.compact()
        self.assertEqual(len(tree.nodes_), original_node_count)
        self.assertEqual([node.node_id for node in tree.nodes_], original_ids)
        self.assertEqual(
            [leaf.extension.leaf_id for leaf in tree.leaves_],
            original_leaf_ids,
        )
        self.assertEqual(len(tree.content_.extension.children), 2)
        self.assertIsNotNone(tree.content_.extension.statistics)


class TestCompactedTreeOwnsItsValueArrays(unittest.TestCase):
    """Tests that a compacted tree shares no value array with its source."""

    __slots__ = ()

    def test_regression_response_samples_are_copied(self):
        """Compacted regression nodes hold their own response_samples array."""
        tree, _ = _fit_nested_numeric_regression()
        compacted = tree.compact()
        source_arrays = [node.response_samples for node in tree.nodes_]
        for node in compacted.nodes_:
            self._assert_not_shared(node.response_samples, source_arrays)

    def test_classification_proba_edit_leaves_source_unchanged(self):
        """Editing a compacted leaf's predicted_proba leaves the source
        predictions untouched."""
        tree, X = _fit_nested_numeric_classification()
        compacted = tree.compact()
        expected = tree.predict_proba(X)
        compacted.leaves_[0].predicted_proba[:] = 0.25
        numpy.testing.assert_array_equal(tree.predict_proba(X), expected)

    def test_survival_curve_and_metric_arrays_are_copied(self):
        """Compacted survival nodes hold their own curve, variance and metric
        arrays."""
        tree, _ = _fit_numeric_survival()
        compacted = tree.compact()
        source_arrays = []
        for node in tree.nodes_:
            times, surv = node.predicted_survival
            source_arrays.append(times)
            source_arrays.append(surv)
            source_arrays.append(node.survival_log_variance)
            source_arrays.append(node.predicted_metrics)
            source_arrays.append(node.ci_low)
            source_arrays.append(node.ci_high)
        for node in compacted.nodes_:
            times, surv = node.predicted_survival
            self._assert_not_shared(times, source_arrays)
            self._assert_not_shared(surv, source_arrays)
            self._assert_not_shared(node.survival_log_variance, source_arrays)
            self._assert_not_shared(node.predicted_metrics, source_arrays)
            self._assert_not_shared(node.ci_low, source_arrays)
            self._assert_not_shared(node.ci_high, source_arrays)

    def test_ranking_expected_rank_arrays_are_copied(self):
        """Compacted ranking nodes hold their own expected-rank and bound
        arrays."""
        tree, _ = _fit_numeric_ranking()
        compacted = tree.compact()
        source_arrays = []
        for node in tree.nodes_:
            source_arrays.append(node.predicted_ranks)
            source_arrays.append(node.ci_low)
            source_arrays.append(node.ci_high)
        for node in compacted.nodes_:
            self._assert_not_shared(node.predicted_ranks, source_arrays)
            self._assert_not_shared(node.ci_low, source_arrays)
            self._assert_not_shared(node.ci_high, source_arrays)

    def test_kept_split_statistics_are_copied(self):
        """A partition kept whole reports the same p-value through its own
        statistics record."""
        tree, _ = _fit_step_regression()
        compacted = tree.compact()
        source = tree.content_.extension.statistics
        statistics = compacted.content_.extension.statistics
        self.assertIsNot(statistics, source)
        self.assertEqual(statistics.p_value, source.p_value)

    def _assert_not_shared(self, array, source_arrays) -> None:
        """Fail when array is one of the source tree's array objects."""
        for source in source_arrays:
            self.assertIsNot(array, source)


class TestCompactedTreeInvariants(unittest.TestCase):
    """Tests that a compacted tree satisfies the fitted-tree invariants."""

    __slots__ = ()

    def test_node_ids_are_contiguous_preorder(self):
        """nodes_[k].node_id == k holds on the compacted tree."""
        tree, _ = _fit_nested_numeric_regression()
        compacted = tree.compact()
        for index, node in enumerate(compacted.nodes_):
            self.assertEqual(node.node_id, index)

    def test_leaf_ids_match_position(self):
        """leaves_[k].extension.leaf_id == k holds on the compacted tree."""
        tree, _ = _fit_nested_numeric_regression()
        compacted = tree.compact()
        for index, leaf in enumerate(compacted.leaves_):
            self.assertEqual(leaf.extension.leaf_id, index)

    def test_depth_is_recomputed(self):
        """Merged children sit one level below the merged root."""
        tree, _ = _fit_nested_numeric_regression()
        compacted = tree.compact()
        root = compacted.content_
        self.assertEqual(root.depth, 0)
        for child in root.extension.children:
            self.assertEqual(child.depth, 1)


class TestCompactedRendering(unittest.TestCase):
    """Tests that the renderers handle the N-ary compacted tree."""

    __slots__ = ()

    def test_text_lists_every_branch(self):
        """Text export emits one branch line per merged child."""
        tree, _ = _fit_nested_numeric_regression()
        compacted = tree.compact()
        text = compacted.to_text()
        branch_lines = text.count("├──") + text.count("└──")
        self.assertEqual(
            branch_lines, len(compacted.content_.extension.children)
        )

    def test_sql_emits_one_when_per_branch(self):
        """SQL export emits one WHEN clause per merged child."""
        tree, _ = _fit_nested_numeric_regression()
        compacted = tree.compact()
        sql = compacted.to_sql()
        self.assertEqual(
            sql.count("WHEN"), len(compacted.content_.extension.children)
        )

    def test_svg_render_succeeds(self):
        """Image export of a compacted tree produces SVG output."""
        tree, _ = _fit_nested_numeric_regression()
        compacted = tree.compact()
        svg = compacted.to_image("svg")
        self.assertIn(b"svg", svg)


if __name__ == "__main__":
    unittest.main()
