"""Unit tests for the Node hierarchy, the Partition hierarchy, and traversal."""

import unittest
import weakref

import numpy

import sigma
import sigma._extension
import sigma._node
import sigma._partition


def _leaf_regression(prediction: float) -> sigma._node.RegressionNode:
    """Build a regression leaf with the given prediction."""
    leaf = sigma._node.RegressionNode(
        depth=1,
        n_samples=10,
        share=0.0,
        decoration=None,
        prediction=prediction,
        ci_low=None,
        ci_high=None,
        response_samples=numpy.empty(0, dtype=float),
    )
    return leaf


def _survival_node(
    values: list[float],
    ci_low: None | list[float] = None,
    ci_high: None | list[float] = None,
) -> sigma._node.SurvivalNode:
    """Build a survival leaf carrying the given per-metric values and bounds."""
    undefined = [numpy.nan] * len(values)
    node = sigma._node.SurvivalNode(
        depth=0,
        n_samples=10,
        share=1.0,
        decoration=None,
        survival_function=(numpy.array([1.0]), numpy.array([1.0])),
        survival_log_variance=numpy.zeros(1, dtype=float),
        values=numpy.array(values, dtype=float),
        ci_low=numpy.array(ci_low or undefined, dtype=float),
        ci_high=numpy.array(ci_high or undefined, dtype=float),
    )
    return node


def _split_statistics(p_value) -> sigma._partition.SplitStatistics:
    """Build a SplitStatistics with the given p-value."""
    statistics = sigma._partition.SplitStatistics(p_value=p_value)
    return statistics


def _numeric_partition(left, right) -> sigma._partition.NumericalPartition:
    """Build a NumericalPartition on x[0] <= 5.0 around the given children."""
    partition = sigma._partition.NumericalPartition(
        feature=sigma.NumericFeature(0, "x"),
        statistics=_split_statistics(0.01),
        children=(left, right),
        thresholds=(5.0,),
    )
    return partition


def _regression_root(
    extension, prediction, n_samples, share=1.0
) -> sigma._node.RegressionNode:
    """Build a depth-0 regression root carrying the given extension."""
    root = sigma._node.RegressionNode(
        depth=0,
        n_samples=n_samples,
        share=share,
        decoration=None,
        prediction=prediction,
        ci_low=None,
        ci_high=None,
        response_samples=numpy.empty(0, dtype=float),
    )
    root.extension = extension
    return root


class TestRegressionNode(unittest.TestCase):
    """Tests for the RegressionNode subclass."""

    __slots__ = ()

    def test_leaf_has_leaf_extension(self):
        """A leaf RegressionNode carries a Leaf extension."""
        leaf = _leaf_regression(3.5)
        self.assertIsInstance(leaf.extension, sigma._extension.Leaf)

    def test_stores_prediction_and_ci(self):
        """RegressionNode stores prediction and the scalar CI bounds."""
        leaf = sigma._node.RegressionNode(
            depth=2,
            n_samples=42,
            share=0.42,
            decoration=None,
            prediction=7.0,
            ci_low=6.0,
            ci_high=8.0,
            response_samples=numpy.empty(0, dtype=float),
        )
        self.assertEqual(leaf.prediction, 7.0)
        self.assertEqual(leaf.ci_low, 6.0)
        self.assertEqual(leaf.ci_high, 8.0)
        self.assertEqual(leaf.n_samples, 42)
        self.assertAlmostEqual(leaf.share, 0.42)

    def test_leaf_sort_key_orders_ascending(self):
        """RegressionNode.leaf_sort_key returns (prediction,) for ascending sort."""
        low = _leaf_regression(1.0)
        high = _leaf_regression(5.0)
        self.assertLess(low.leaf_sort_key(()), high.leaf_sort_key(()))


class TestClassificationNode(unittest.TestCase):
    """Tests for the ClassificationNode subclass."""

    __slots__ = ()

    def test_stores_class_distribution_and_per_class_ci(self):
        """ClassificationNode stores class_distribution and per-class CI arrays."""
        distribution = numpy.array([0.3, 0.7])
        ci_low = numpy.array([0.1, 0.5])
        ci_high = numpy.array([0.5, 0.9])
        leaf = sigma._node.ClassificationNode(
            depth=0,
            n_samples=20,
            share=1.0,
            decoration=None,
            prediction=1,
            class_distribution=distribution,
            ci_low=ci_low,
            ci_high=ci_high,
            mean_offset_proba=None,
        )
        numpy.testing.assert_array_equal(leaf.class_distribution, distribution)
        leaf_ci_low = leaf.ci_low
        leaf_ci_high = leaf.ci_high
        assert leaf_ci_low is not None
        assert leaf_ci_high is not None
        numpy.testing.assert_array_equal(leaf_ci_low, ci_low)
        numpy.testing.assert_array_equal(leaf_ci_high, ci_high)
        self.assertEqual(leaf.prediction, 1)
        self.assertIsNone(leaf.mean_offset_proba)

    def test_leaf_sort_key_orders_descending_distribution(self):
        """ClassificationNode.leaf_sort_key negates probabilities for descending sort."""
        majority_zero = sigma._node.ClassificationNode(
            depth=0,
            n_samples=10,
            share=1.0,
            decoration=None,
            prediction=0,
            class_distribution=numpy.array([0.9, 0.1]),
            ci_low=None,
            ci_high=None,
            mean_offset_proba=None,
        )
        majority_one = sigma._node.ClassificationNode(
            depth=0,
            n_samples=10,
            share=1.0,
            decoration=None,
            prediction=1,
            class_distribution=numpy.array([0.4, 0.6]),
            ci_low=None,
            ci_high=None,
            mean_offset_proba=None,
        )
        self.assertLess(
            majority_zero.leaf_sort_key(()), majority_one.leaf_sort_key(())
        )


class TestSurvivalNode(unittest.TestCase):
    """Tests for the SurvivalNode subclass."""

    __slots__ = ()

    def test_prediction_property_reads_first_value(self):
        """SurvivalNode.prediction returns the first entry of values."""
        leaf = _survival_node([2.5])
        self.assertEqual(leaf.prediction, 2.5)

    def test_stores_per_metric_ci_arrays(self):
        """SurvivalNode stores one CI bound pair per metric."""
        leaf = _survival_node([2.5], ci_low=[2.0], ci_high=[3.0])
        numpy.testing.assert_array_equal(leaf.ci_low, numpy.array([2.0]))
        numpy.testing.assert_array_equal(leaf.ci_high, numpy.array([3.0]))

    def test_leaf_sort_key_negates_lower_is_better(self):
        """SurvivalNode.leaf_sort_key flips the sign for better_is='lower' metrics."""
        metrics = (
            sigma.MedianSurvivalMetric("Median survival", False),
            sigma.RiskScoreMetric("Risk score"),
        )
        leaf = _survival_node([5.0, 2.0])
        key = leaf.leaf_sort_key(metrics)
        self.assertEqual(key, (5.0, -2.0))


class TestTraverse(unittest.TestCase):
    """Tests for Node.traverse routing through partitions."""

    __slots__ = ()

    def _build_numeric_tree(self) -> sigma._node.RegressionNode:
        """Build a tree: root partitions on x[0] <= 5, two regression leaves."""
        left = _leaf_regression(1.0)
        right = _leaf_regression(9.0)
        partition = _numeric_partition(left, right)
        root = _regression_root(partition, 5.0, 20)
        return root

    def test_leaf_returns_itself(self):
        """A leaf's traverse returns the leaf itself."""
        leaf = _leaf_regression(3.0)
        result = leaf.traverse(numpy.array([99.0]))
        self.assertIs(result, leaf)

    def test_numeric_routes_left_when_below_threshold(self):
        """Values <= threshold route to the left child."""
        root = self._build_numeric_tree()
        result = root.traverse(numpy.array([2.0]))
        self.assertEqual(result.prediction, 1.0)

    def test_numeric_routes_right_when_above_threshold(self):
        """Values > threshold route to the right child."""
        root = self._build_numeric_tree()
        result = root.traverse(numpy.array([7.0]))
        self.assertEqual(result.prediction, 9.0)

    def test_categorical_routes_by_membership(self):
        """Categorical traversal routes by membership in left_categories / right_categories."""
        left = _leaf_regression(1.0)
        right = _leaf_regression(9.0)
        partition = sigma._partition.CategoricalPartition(
            feature=sigma.CategoricalFeature(0, "cat"),
            statistics=_split_statistics(0.05),
            children=(left, right),
            category_groups=(frozenset({0.0, 1.0}), frozenset({2.0})),
        )
        root = _regression_root(partition, 5.0, 10)
        self.assertIs(root.traverse(numpy.array([0.0])), left)
        self.assertIs(root.traverse(numpy.array([1.0])), left)
        self.assertIs(root.traverse(numpy.array([2.0])), right)

    def test_categorical_unknown_value_stops_at_holding_node(self):
        """An unseen category stops traversal at the holding node."""
        left = _leaf_regression(1.0)
        right = _leaf_regression(9.0)
        partition = sigma._partition.CategoricalPartition(
            feature=sigma.CategoricalFeature(0, "cat"),
            statistics=_split_statistics(0.05),
            children=(left, right),
            category_groups=(frozenset({0.0}), frozenset({1.0})),
        )
        root = _regression_root(partition, 5.0, 10)
        result = root.traverse(numpy.array([2.0]))
        self.assertIs(result, root)


class TestLeavesAndShare(unittest.TestCase):
    """Tests for Node.leaves and the _populate_share post-pass."""

    __slots__ = ()

    def _build_tree(self) -> sigma._node.RegressionNode:
        """Build a small tree: root with two regression leaves."""
        left = sigma._node.RegressionNode(
            depth=1,
            n_samples=15,
            share=0.0,
            decoration=None,
            prediction=1.0,
            ci_low=None,
            ci_high=None,
            response_samples=numpy.empty(0, dtype=float),
        )
        right = sigma._node.RegressionNode(
            depth=1,
            n_samples=5,
            share=0.0,
            decoration=None,
            prediction=9.0,
            ci_low=None,
            ci_high=None,
            response_samples=numpy.empty(0, dtype=float),
        )
        partition = _numeric_partition(left, right)
        root = _regression_root(partition, 3.0, 20, share=0.0)
        return root

    def test_leaves_returns_left_then_right(self):
        """Node.leaves returns leaves in left-to-right order."""
        root = self._build_tree()
        partition = root.extension
        assert isinstance(partition, sigma._partition.Partition)
        leaves = root.leaves()
        self.assertEqual(leaves, [partition.children[0], partition.children[1]])

    def test_populate_share_sets_n_samples_fraction(self):
        """_populate_share sets share = n_samples / root.n_samples on every node."""
        root = self._build_tree()
        sigma._node._populate_share(root)
        self.assertAlmostEqual(root.share, 1.0)
        match root.extension:
            case sigma._partition.Partition() as partition:
                self.assertAlmostEqual(partition.children[0].share, 0.75)  # ty: ignore[unresolved-attribute]
                self.assertAlmostEqual(partition.children[1].share, 0.25)  # ty: ignore[unresolved-attribute]
            case _:
                self.fail("root extension should be a Partition")


class TestPartitionTypes(unittest.TestCase):
    """Tests for the NumericalPartition and CategoricalPartition concrete classes."""

    __slots__ = ()

    def test_numerical_route_matches_threshold(self):
        """NumericalPartition.route returns left for value <= threshold, right otherwise."""
        left = _leaf_regression(1.0)
        right = _leaf_regression(9.0)
        partition = _numeric_partition(left, right)
        self.assertIs(partition.route(5.0), left)
        self.assertIs(partition.route(5.0001), right)

    def test_categorical_observed_categories_is_union(self):
        """CategoricalPartition.observed_categories is left_categories | right_categories."""
        left = _leaf_regression(1.0)
        right = _leaf_regression(9.0)
        partition = sigma._partition.CategoricalPartition(
            feature=sigma.CategoricalFeature(0, "cat"),
            statistics=_split_statistics(0.05),
            children=(left, right),
            category_groups=(frozenset({"a", "b"}), frozenset({"c"})),
        )
        self.assertEqual(
            partition.observed_categories, frozenset({"a", "b", "c"})
        )

    def test_boolean_route_directs_false_left_and_true_right(self):
        """BooleanPartition.route returns left for False / 0.0 and right for True / 1.0."""
        left = _leaf_regression(1.0)
        right = _leaf_regression(9.0)
        partition = sigma._partition.BooleanPartition(
            feature=sigma.BooleanFeature(0, "flag"),
            statistics=_split_statistics(0.01),
            children=(left, right),
        )
        self.assertIs(partition.route(False), left)
        self.assertIs(partition.route(0.0), left)
        self.assertIs(partition.route(True), right)
        self.assertIs(partition.route(1.0), right)

    def test_boolean_route_rejects_non_boolean_value(self):
        """BooleanPartition.route raises ValueError on values that are neither 0 nor 1."""
        left = _leaf_regression(1.0)
        right = _leaf_regression(9.0)
        partition = sigma._partition.BooleanPartition(
            feature=sigma.BooleanFeature(0, "flag"),
            statistics=_split_statistics(0.01),
            children=(left, right),
        )
        with self.assertRaises(ValueError):
            partition.route(0.5)

    def test_boolean_partition_has_empty_slots(self):
        """BooleanPartition adds no instance state beyond the Partition base."""
        self.assertEqual(sigma._partition.BooleanPartition.__slots__, ())


class TestLeafIdDefault(unittest.TestCase):
    """Tests for the Leaf.leaf_id field at construction time."""

    __slots__ = ()

    def test_freshly_constructed_leaf_has_sentinel_leaf_id(self):
        """A newly built leaf carries the 0 sentinel leaf_id."""
        leaf = _leaf_regression(3.0)
        extension = leaf.extension
        assert isinstance(extension, sigma._extension.Leaf)
        self.assertEqual(extension.leaf_id, 0)

    def test_freshly_constructed_internal_node_has_no_leaf_id(self):
        """A freshly built internal node carries a Partition without a leaf_id field."""
        left = _leaf_regression(1.0)
        right = _leaf_regression(9.0)
        partition = _numeric_partition(left, right)
        root = _regression_root(partition, 5.0, 20)
        extension = root.extension
        self.assertIsInstance(extension, sigma._partition.Partition)
        self.assertFalse(hasattr(extension, "leaf_id"))


class TestNodeIdDefault(unittest.TestCase):
    """Tests for the Node.node_id field at construction time."""

    __slots__ = ()

    def test_freshly_constructed_leaf_has_zero_node_id(self):
        """A newly built leaf has node_id equal to the sentinel 0."""
        leaf = _leaf_regression(3.0)
        self.assertEqual(leaf.node_id, 0)

    def test_freshly_constructed_internal_node_has_zero_node_id(self):
        """A newly built internal node has node_id equal to the sentinel 0."""
        left = _leaf_regression(1.0)
        right = _leaf_regression(9.0)
        partition = _numeric_partition(left, right)
        root = _regression_root(partition, 5.0, 20)
        self.assertEqual(root.node_id, 0)


class TestWeakReferenceable(unittest.TestCase):
    """Tests that every public class can be the target of a weak reference."""

    __slots__ = ()

    def _public_instances(self) -> list[object]:
        """Build one instance of each public class for weakref checks."""
        leaf = _leaf_regression(1.0)
        right_leaf = _leaf_regression(9.0)
        numerical = _numeric_partition(leaf, right_leaf)
        boolean = sigma._partition.BooleanPartition(
            feature=sigma.BooleanFeature(0, "flag"),
            statistics=_split_statistics(0.01),
            children=(leaf, right_leaf),
        )
        categorical = sigma._partition.CategoricalPartition(
            feature=sigma.CategoricalFeature(0, "cat"),
            statistics=_split_statistics(0.05),
            children=(leaf, right_leaf),
            category_groups=(frozenset({0.0}), frozenset({1.0})),
        )
        metrics = [
            sigma.MedianSurvivalMetric("Median survival", False),
            sigma.RiskScoreMetric("Risk score"),
            sigma.SurvivalAtMetric("Survival at 5 years", False, 5.0),
            sigma.RmstMetric("RMST at 2 years", False, 2.0),
            sigma.ExpectedRankMetric("Ebi rank", False),
        ]
        classification = sigma._node.ClassificationNode(
            depth=0,
            n_samples=10,
            share=1.0,
            decoration=None,
            prediction=0,
            class_distribution=numpy.array([0.6, 0.4]),
            ci_low=None,
            ci_high=None,
            mean_offset_proba=None,
        )
        survival = _survival_node([2.5])
        ranking = sigma._node.RankingNode(
            depth=0,
            n_samples=10,
            share=1.0,
            decoration=None,
            values=numpy.array([1.0]),
            ci_low=numpy.array([numpy.nan]),
            ci_high=numpy.array([numpy.nan]),
        )
        instances: list[object] = [
            leaf.extension,
            leaf,
            numerical,
            boolean,
            categorical,
            classification,
            survival,
            ranking,
            *metrics,
            _split_statistics(0.01),
            sigma._partition.NumericInterval(None, 5.0),
            sigma._partition.CategorySubset(frozenset({0.0})),
            sigma._partition.BooleanValue(True),
            sigma.ClassificationTree(),
            sigma.RegressionTree(),
            sigma.SurvivalTree(),
            sigma.RankingTree(),
        ]
        return instances

    def test_public_instances_support_weakref(self):
        """weakref.ref succeeds and resolves back to each public instance."""
        instances = self._public_instances()
        for instance in instances:
            instance_type = type(instance)
            with self.subTest(cls=instance_type.__name__):
                reference = weakref.ref(instance)
                resolved = reference()
                self.assertIs(resolved, instance)


if __name__ == "__main__":
    unittest.main()
