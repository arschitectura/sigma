"""Unit tests for Node.polars_expression and the node parent links."""

import pickle
import unittest

import numpy
import polars

import sigma
import sigma._node
import sigma._partition


def _leaf(predicted_mean: float) -> sigma._node.RegressionNode:
    """Build a regression leaf with the given predicted mean."""
    leaf = sigma._node.RegressionNode(
        depth=1,
        n_samples=10,
        predicted_mean=predicted_mean,
        ci_low=None,
        ci_high=None,
        response_samples=numpy.empty(0, dtype=float),
    )
    return leaf


def _root(extension) -> sigma._node.RegressionNode:
    """Build a depth-0 regression root carrying the given extension."""
    root = sigma._node.RegressionNode(
        depth=0,
        n_samples=20,
        predicted_mean=0.0,
        ci_low=None,
        ci_high=None,
        response_samples=numpy.empty(0, dtype=float),
    )
    root.extension = extension
    return root


def _link(root, partition) -> None:
    """Stamp root as the parent of every child of partition."""
    for child in partition.children:
        child.parent = root


class TestPolarsExpressionStructure(unittest.TestCase):
    """Tests for the recursive expression structure on manual nodes."""

    __slots__ = ()

    def test_root_expression_is_literal_true(self):
        """A node without a parent returns a literal true expression."""
        root = _root(sigma._partition.Leaf())
        expression = root.polars_expression()
        expected = polars.lit(True)
        equal = expression.meta.eq(expected)
        self.assertTrue(equal)

    def test_child_ands_parent_with_branch_condition(self):
        """A child combines its parent's expression with its own branch condition using AND."""
        left, right = _leaf(1.0), _leaf(9.0)
        partition = sigma._partition.NumericalPartition(
            feature=sigma.NumericFeature(0, "x"),
            statistics=None,
            children=(left, right),
            thresholds=(5.0,),
        )
        root = _root(partition)
        _link(root, partition)
        base = polars.lit(True)
        column = polars.col("x")
        left_expected = base & (column <= 5.0)
        right_expected = base & (column > 5.0)
        left_expression = left.polars_expression()
        right_expression = right.polars_expression()
        left_equal = left_expression.meta.eq(left_expected)
        right_equal = right_expression.meta.eq(right_expected)
        self.assertTrue(left_equal)
        self.assertTrue(right_equal)

    def test_grandchild_chains_all_conditions_from_root(self):
        """A depth-2 node AND-chains the root and intermediate branch conditions."""
        deep_left, deep_right = _leaf(1.0), _leaf(2.0)
        inner_partition = sigma._partition.NumericalPartition(
            feature=sigma.NumericFeature(1, "z"),
            statistics=None,
            children=(deep_left, deep_right),
            thresholds=(2.0,),
        )
        inner = _root(inner_partition)
        _link(inner, inner_partition)
        outer_right = _leaf(9.0)
        outer_partition = sigma._partition.NumericalPartition(
            feature=sigma.NumericFeature(0, "x"),
            statistics=None,
            children=(inner, outer_right),
            thresholds=(5.0,),
        )
        root = _root(outer_partition)
        _link(root, outer_partition)
        base = polars.lit(True)
        x_column = polars.col("x")
        z_column = polars.col("z")
        expected = (base & (x_column <= 5.0)) & (z_column <= 2.0)
        expression = deep_left.polars_expression()
        equal = expression.meta.eq(expected)
        self.assertTrue(equal)

    def test_missing_feature_name_falls_back_to_index(self):
        """A partition without a feature name references the X[i] column."""
        left, right = _leaf(1.0), _leaf(9.0)
        partition = sigma._partition.NumericalPartition(
            feature=sigma.NumericFeature(3, None),
            statistics=None,
            children=(left, right),
            thresholds=(5.0,),
        )
        root = _root(partition)
        _link(root, partition)
        base = polars.lit(True)
        column = polars.col("X[3]")
        expected = base & (column <= 5.0)
        expression = left.polars_expression()
        equal = expression.meta.eq(expected)
        self.assertTrue(equal)


class TestPolarsExpressionConditions(unittest.TestCase):
    """Tests for the per-partition branch conditions in the expression."""

    __slots__ = ()

    def test_numeric_dedicated_missing_child_is_null(self):
        """A dedicated missing child of a numeric partition tests for null."""
        c0, c1, missing = _leaf(1.0), _leaf(2.0), _leaf(9.0)
        partition = sigma._partition.NumericalPartition(
            feature=sigma.NumericFeature(0, "x"),
            statistics=None,
            children=(c0, c1, missing),
            thresholds=(5.0,),
            nan_child=2,
        )
        root = _root(partition)
        _link(root, partition)
        base = polars.lit(True)
        column = polars.col("x")
        expected = base & column.is_null()
        expression = missing.polars_expression()
        equal = expression.meta.eq(expected)
        self.assertTrue(equal)

    def test_numeric_ride_along_missing_ors_is_null(self):
        """An interval branch that also admits missing values ORs in a null test."""
        left, right = _leaf(1.0), _leaf(9.0)
        partition = sigma._partition.NumericalPartition(
            feature=sigma.NumericFeature(0, "x"),
            statistics=None,
            children=(left, right),
            thresholds=(5.0,),
            nan_child=1,
        )
        root = _root(partition)
        _link(root, partition)
        base = polars.lit(True)
        column = polars.col("x")
        left_expected = base & (column <= 5.0)
        right_expected = base & ((column > 5.0) | column.is_null())
        left_expression = left.polars_expression()
        right_expression = right.polars_expression()
        left_equal = left_expression.meta.eq(left_expected)
        right_equal = right_expression.meta.eq(right_expected)
        self.assertTrue(left_equal)
        self.assertTrue(right_equal)

    def test_numeric_inner_interval_bounds_both_sides(self):
        """An interior interval branch bounds the column on both sides."""
        c0, c1, c2 = _leaf(1.0), _leaf(2.0), _leaf(3.0)
        partition = sigma._partition.NumericalPartition(
            feature=sigma.NumericFeature(0, "x"),
            statistics=None,
            children=(c0, c1, c2),
            thresholds=(2.0, 5.0),
        )
        root = _root(partition)
        _link(root, partition)
        base = polars.lit(True)
        column = polars.col("x")
        expected = base & ((column > 2.0) & (column <= 5.0))
        expression = c1.polars_expression()
        equal = expression.meta.eq(expected)
        self.assertTrue(equal)

    def test_boolean_children_negate_and_pass_the_column(self):
        """A boolean partition emits the negated column and the bare column."""
        false_leaf, true_leaf = _leaf(1.0), _leaf(9.0)
        partition = sigma._partition.BooleanPartition(
            feature=sigma.BooleanFeature(0, "flag"),
            statistics=None,
            children=(false_leaf, true_leaf),
        )
        root = _root(partition)
        _link(root, partition)
        base = polars.lit(True)
        column = polars.col("flag")
        false_expected = base & ~column
        true_expected = base & column
        false_expression = false_leaf.polars_expression()
        true_expression = true_leaf.polars_expression()
        false_equal = false_expression.meta.eq(false_expected)
        true_equal = true_expression.meta.eq(true_expected)
        self.assertTrue(false_equal)
        self.assertTrue(true_equal)

    def test_categorical_codes_without_labels(self):
        """A categorical partition without labels compares the raw codes."""
        left, right = _leaf(1.0), _leaf(9.0)
        partition = sigma._partition.CategoricalPartition(
            feature=sigma.CategoricalFeature(0, "group"),
            statistics=None,
            children=(left, right),
            category_groups=(frozenset({2.0}), frozenset({7.0, 5.0})),
        )
        root = _root(partition)
        _link(root, partition)
        base = polars.lit(True)
        column = polars.col("group")
        left_expected = base & (column == 2.0)
        right_expected = base & column.is_in([5.0, 7.0])
        left_expression = left.polars_expression()
        right_expression = right.polars_expression()
        left_equal = left_expression.meta.eq(left_expected)
        right_equal = right_expression.meta.eq(right_expected)
        self.assertTrue(left_equal)
        self.assertTrue(right_equal)

    def test_categorical_labels_replace_codes(self):
        """A categorical partition with labels compares the label strings."""
        left, right = _leaf(1.0), _leaf(9.0)
        partition = sigma._partition.CategoricalPartition(
            feature=sigma.CategoricalFeature(
                0,
                "color",
                category_labels={0.0: "red", 1.0: "blue", 2.0: "green"},
            ),
            statistics=None,
            children=(left, right),
            category_groups=(frozenset({0.0, 2.0}), frozenset({1.0})),
        )
        root = _root(partition)
        _link(root, partition)
        base = polars.lit(True)
        column = polars.col("color")
        left_expected = base & column.is_in(["red", "green"])
        right_expected = base & (column == "blue")
        left_expression = left.polars_expression()
        right_expression = right.polars_expression()
        left_equal = left_expression.meta.eq(left_expected)
        right_equal = right_expression.meta.eq(right_expected)
        self.assertTrue(left_equal)
        self.assertTrue(right_equal)

    def test_categorical_na_code_emits_is_null(self):
        """A category group holding the N/A code tests for null instead of the code."""
        left, right = _leaf(1.0), _leaf(9.0)
        partition = sigma._partition.CategoricalPartition(
            feature=sigma.CategoricalFeature(
                0,
                "color",
                category_labels={0.0: "red", 1.0: "blue", 2.0: "N/A"},
                na_code=2.0,
            ),
            statistics=None,
            children=(left, right),
            category_groups=(frozenset({0.0}), frozenset({1.0, 2.0})),
        )
        root = _root(partition)
        _link(root, partition)
        base = polars.lit(True)
        column = polars.col("color")
        right_expected = base & ((column == "blue") | column.is_null())
        right_expression = right.polars_expression()
        right_equal = right_expression.meta.eq(right_expected)
        self.assertTrue(right_equal)

    def test_promoted_boolean_uses_boolean_column_tests(self):
        """A promoted boolean partition tests the column truth values and null."""
        true_leaf, rest_leaf = _leaf(9.0), _leaf(1.0)
        partition = sigma._partition.CategoricalPartition(
            feature=sigma.PromotedBooleanFeature(
                0,
                "flag",
                category_labels={0.0: "False", 1.0: "True", 2.0: "N/A"},
                na_code=2.0,
            ),
            statistics=None,
            children=(true_leaf, rest_leaf),
            category_groups=(frozenset({1.0}), frozenset({0.0, 2.0})),
        )
        root = _root(partition)
        _link(root, partition)
        base = polars.lit(True)
        column = polars.col("flag")
        true_expected = base & column
        rest_expected = base & (~column | column.is_null())
        true_expression = true_leaf.polars_expression()
        rest_expression = rest_leaf.polars_expression()
        true_equal = true_expression.meta.eq(true_expected)
        rest_equal = rest_expression.meta.eq(rest_expected)
        self.assertTrue(true_equal)
        self.assertTrue(rest_equal)


class TestParentLinks(unittest.TestCase):
    """Tests for the parent attribute stamped by fit and compact."""

    __slots__ = ()

    def test_unfitted_node_defaults_to_no_parent(self):
        """A freshly constructed node carries parent None."""
        leaf = _leaf(1.0)
        self.assertIsNone(leaf.parent)

    def test_fit_stamps_parent_links(self):
        """After fit, the root has no parent and every child points to its holder."""
        tree = _numeric_tree()
        self._assert_parent_links(tree)

    def test_compact_restamps_parent_links(self):
        """A compacted tree carries consistent parent links on its own nodes."""
        tree = _numeric_tree()
        compacted = tree.compact()
        self._assert_parent_links(compacted)
        self._assert_parent_links(tree)

    def _assert_parent_links(self, tree) -> None:
        """Assert the root has no parent and children point to their holder."""
        self.assertIsNone(tree.content_.parent)
        for node in tree.nodes_:
            extension = node.extension
            if isinstance(extension, sigma._partition.Partition):
                for child in extension.children:
                    self.assertIs(child.parent, node)


class TestPolarsExpressionOnFittedTrees(unittest.TestCase):
    """Tests matching expressions against decision_path on fitted trees."""

    __slots__ = ()

    def test_numeric_tree_matches_decision_path(self):
        """Numeric split expressions select exactly the decision-path rows."""
        tree = _numeric_tree()
        self._assert_expressions_match_paths(tree, _NUMERIC_FRAME)

    def test_categorical_tree_matches_decision_path(self):
        """Categorical label expressions select exactly the decision-path rows."""
        tree, frame = _categorical_tree()
        self._assert_expressions_match_paths(tree, frame)

    def test_boolean_tree_matches_decision_path(self):
        """Boolean split expressions select exactly the decision-path rows."""
        tree, frame = _boolean_tree()
        self._assert_expressions_match_paths(tree, frame)

    def test_numeric_missing_tree_matches_decision_path(self):
        """A learned numeric missing rule selects the null rows of its branch."""
        tree, frame = _numeric_missing_tree()
        extension = tree.content_.extension
        match extension:
            case sigma._partition.NumericalPartition() as partition:
                self.assertIsNotNone(partition.nan_child)
            case _:
                self.fail("expected a numeric split at the root")
        self._assert_expressions_match_paths(tree, frame)

    def test_promoted_boolean_tree_matches_decision_path(self):
        """A promoted boolean split selects its true, false, and null rows."""
        tree, frame = _promoted_boolean_tree()
        self.assertIsInstance(tree.features_[0], sigma.PromotedBooleanFeature)
        self._assert_expressions_match_paths(tree, frame)

    def test_compacted_tree_matches_decision_path(self):
        """Merged N-ary partitions keep expressions aligned with decision_path."""
        tree = _numeric_tree()
        compacted = tree.compact()
        widths = []
        for node in compacted.nodes_:
            extension = node.extension
            if isinstance(extension, sigma._partition.Partition):
                widths.append(len(extension.children))
        self.assertTrue(any(width > 2 for width in widths))
        self._assert_expressions_match_paths(compacted, _NUMERIC_FRAME)

    def test_classification_tree_matches_decision_path(self):
        """ClassificationTree node expressions select the decision-path rows."""
        tree, frame = _classification_tree()
        self._assert_expressions_match_paths(tree, frame)

    def test_category_labels_agree_with_to_text(self):
        """Categorical conditions carry the labels to_text renders, and both
        follow a display-time category_labels override the same way."""
        tree, _frame = _categorical_tree()
        feature = tree.features_[0]
        assert isinstance(feature, sigma.CategoricalFeature)
        labels = feature.category_labels
        assert labels is not None
        rendered = tree.to_text()
        for label in labels.values():
            self.assertIn(label, rendered)
        expression = str(tree.leaves_[0].polars_expression())
        used = [label for label in labels.values() if label in expression]
        self.assertTrue(used)
        override = {code: f"{label}!" for code, label in labels.items()}
        overridden = tree.to_text(category_labels={0: override})
        for label in override.values():
            self.assertIn(label, overridden)

    def test_pickled_tree_expressions_survive(self):
        """A pickle round-trip preserves parent links and expression parity."""
        tree = _numeric_tree()
        payload = pickle.dumps(tree)
        loaded = pickle.loads(payload)
        self.assertIsNone(loaded.content_.parent)
        self._assert_expressions_match_paths(loaded, _NUMERIC_FRAME)

    def _assert_expressions_match_paths(self, tree, frame) -> None:
        """Assert every node's expression selects its decision-path rows."""
        self.assertGreater(len(tree.nodes_), 1)
        path_matrix = tree.decision_path(frame)
        dense = path_matrix.toarray()
        indexed = frame.with_row_index("row_index")
        for node in tree.nodes_:
            expression = node.polars_expression()
            filtered = indexed.filter(expression)
            selected = filtered["row_index"].to_list()
            column = dense[:, node.node_id]
            flat = numpy.flatnonzero(column)
            expected = [int(index) for index in flat]
            self.assertEqual(selected, expected)


def _numeric_frame() -> tuple[polars.DataFrame, numpy.ndarray]:
    """Build a two-column numeric polars frame with a stepwise response."""
    rng = numpy.random.default_rng(3)
    n = 300
    x = rng.uniform(0.0, 8.0, n)
    noise = rng.standard_normal(n)
    y = numpy.floor(x / 2.0) * 5.0 + rng.normal(0.0, 0.3, n)
    frame = polars.DataFrame({"x": x, "noise": noise})
    return frame, y


def _numeric_tree() -> sigma.RegressionTree:
    """Fit a regression tree on the shared numeric frame."""
    tree = sigma.RegressionTree(min_splits=8, min_buckets=4, random_state=0)
    tree.fit(_NUMERIC_FRAME, _NUMERIC_Y)
    return tree


_NUMERIC_FRAME, _NUMERIC_Y = _numeric_frame()


def _categorical_tree() -> tuple[sigma.RegressionTree, polars.DataFrame]:
    """Fit a regression tree on a polars Categorical column."""
    rng = numpy.random.default_rng(4)
    n = 200
    levels = numpy.array(["red", "blue", "green", "gold"])
    codes = rng.integers(0, 4, n)
    color = levels[codes]
    y = codes * 3.0 + rng.normal(0.0, 0.3, n)
    series = polars.Series(color, dtype=polars.Categorical)
    frame = polars.DataFrame({"color": series})
    tree = sigma.RegressionTree(min_splits=8, min_buckets=4, random_state=0)
    tree.fit(frame, y)
    return tree, frame


def _boolean_tree() -> tuple[sigma.RegressionTree, polars.DataFrame]:
    """Fit a regression tree on a complete polars Boolean column."""
    rng = numpy.random.default_rng(5)
    n = 200
    flag = rng.random(n) < 0.5
    y = numpy.where(flag, 4.0, 0.0) + rng.normal(0.0, 0.2, n)
    series = polars.Series(flag, dtype=polars.Boolean)
    frame = polars.DataFrame({"flag": series})
    tree = sigma.RegressionTree(min_splits=8, min_buckets=4, random_state=0)
    tree.fit(frame, y)
    return tree, frame


def _numeric_missing_tree() -> tuple[sigma.RegressionTree, polars.DataFrame]:
    """Fit a depth-1 tree whose numeric split routes nulls along one side."""
    rng = numpy.random.default_rng(7)
    n = 400
    x = rng.uniform(0.0, 1.0, n)
    missing = rng.random(n) < 0.25
    y = (x > 0.6).astype(float) * 5.0 + rng.normal(0.0, 0.2, n)
    y[missing] = 5.0 + rng.normal(0.0, 0.2, int(missing.sum()))
    values = [None if gone else float(value) for gone, value in zip(missing, x)]
    series = polars.Series(values, dtype=polars.Float64)
    frame = polars.DataFrame({"x": series})
    tree = sigma.RegressionTree(random_state=1, max_depth=1, min_buckets=5)
    tree.fit(frame, y)
    return tree, frame


def _promoted_boolean_tree() -> tuple[sigma.RegressionTree, polars.DataFrame]:
    """Fit a tree on a null-bearing polars Boolean promoted to three levels."""
    rng = numpy.random.default_rng(5)
    n = 240
    flag = rng.random(n) < 0.5
    missing = rng.random(n) < 0.3
    y = numpy.where(missing, 1.0, numpy.where(flag, 5.0, 0.0))
    y = y + rng.normal(0.0, 0.2, n)
    values = [
        None if gone else bool(value) for gone, value in zip(missing, flag)
    ]
    series = polars.Series(values, dtype=polars.Boolean)
    frame = polars.DataFrame({"flag": series})
    tree = sigma.RegressionTree(min_splits=4, min_buckets=2, random_state=0)
    tree.fit(frame, y)
    return tree, frame


def _classification_tree() -> tuple[sigma.ClassificationTree, polars.DataFrame]:
    """Fit a classification tree on a numeric polars frame."""
    rng = numpy.random.default_rng(9)
    n = 200
    x = rng.uniform(0.0, 8.0, n)
    jitter = rng.normal(0.0, 0.4, n)
    y = ((x + jitter) > 4.0).astype(int)
    frame = polars.DataFrame({"x": x})
    tree = sigma.ClassificationTree(min_splits=8, min_buckets=4, random_state=0)
    tree.fit(frame, y)
    return tree, frame


if __name__ == "__main__":
    unittest.main()
