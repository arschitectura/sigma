"""Unit tests for the graphviz tree visualization."""

import importlib.util
import unittest

import numpy

import sigma._extension
import sigma._partition
import sigma._tree_classification
import sigma._tree_regression
import sigma._tree_survival

_HAS_GRAPHVIZ = importlib.util.find_spec("graphviz") is not None


def _make_regression_tree(reverse_order: bool = False):
    """Fit a simple regression tree on a step function."""
    X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
    y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
    regression_tree = sigma._tree_regression.RegressionTree(
        correlation="normal",
        min_splits=2,
        min_buckets=1,
        reverse_order=reverse_order,
    )
    regression_tree.fit(X, y)
    return regression_tree, X, y


def _make_three_step_regression_tree():
    """Fit a regression tree on a 3-step response, yielding a depth >= 2 tree."""
    X = numpy.arange(1, 81, dtype=float).reshape(-1, 1)
    y = numpy.where(X.ravel() < 20, 0.0, numpy.where(X.ravel() < 60, 5.0, 10.0))
    regression_tree = sigma._tree_regression.RegressionTree(
        correlation="normal",
        min_splits=2,
        min_buckets=1,
    )
    regression_tree.fit(X, y)
    return regression_tree


def _make_classification_tree():
    """Fit a simple binary classification tree."""
    rng = numpy.random.default_rng(42)
    X = rng.standard_normal((60, 2))
    y = (X[:, 0] > 0).astype(float)
    classification_tree = sigma._tree_classification.ClassificationTree(
        correlation="normal",
        min_splits=2,
        min_buckets=1,
    )
    classification_tree.fit(X, y)
    return classification_tree, X, y


def _make_categorical_regression_tree():
    """Fit a regression tree with a categorical split."""
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
    return regression_tree, X, y


def _make_survival_tree():
    """Fit a simple survival tree with two leaves and two metrics."""
    rng = numpy.random.RandomState(0)
    n = 200
    arm = (numpy.arange(n) % 2).astype(float)
    scales = numpy.where(arm == 0, 10.0, 2.0)
    survival = rng.exponential(scale=scales)
    time = numpy.minimum(survival, 8.0)
    event = (survival <= 8.0).astype(float)
    X = numpy.column_stack([arm, rng.randn(n)])
    y = numpy.column_stack([time, event])
    survival_tree = sigma._tree_survival.SurvivalTree(
        min_splits=10,
        min_buckets=5,
        max_depth=2,
        metrics=("median", ("survival", 5.0, "years")),
    )
    survival_tree.fit(X, y)
    return survival_tree


@unittest.skipUnless(_HAS_GRAPHVIZ, "graphviz not installed")
class TestBuildDigraph(unittest.TestCase):
    """Tests for the _build_digraph helper function."""

    __slots__ = ()

    def setUp(self):
        """Set up fitted models for digraph tests."""
        from sigma import _graphviz

        self._graphviz = _graphviz

        def build_digraph(
            root,
            labels,
            class_names=None,
            feature_names=None,
            orientation="top-down",
            reverse_order=False,
        ):
            """Invoke _build_digraph with default colors."""
            digraph = _graphviz._build_digraph(
                root,
                labels,
                class_names,
                _graphviz._DEFAULT_ROOT_COLORS,
                _graphviz._DEFAULT_SPLIT_COLORS,
                _graphviz._DEFAULT_LEAF_PALETTE,
                "black",
                feature_names=feature_names,
                orientation=orientation,
                reverse_order=reverse_order,
            )
            return digraph

        self._build_digraph = build_digraph
        self.regression_tree, _, _ = _make_regression_tree()
        self.classification_tree, _, _ = _make_classification_tree()
        self.categorical_regression_tree, _, _ = (
            _make_categorical_regression_tree()
        )

    def test_node_labels_contain_prediction_and_count(self):
        """All node labels contain prediction and count."""
        dot = self._build_digraph(self.regression_tree.content_, None)
        source = dot.source
        self.assertIn("Predicted mean = ", source)
        self.assertIn("Obs. count = ", source)

    def test_node_labels_have_no_split_info(self):
        """Node labels do not contain split information."""
        dot = self._build_digraph(self.regression_tree.content_, None)
        source = dot.source
        self.assertNotIn("split below", source)

    def test_split_node_labels_contain_p_value(self):
        """Non-leaf node labels report a Split p-value line."""
        dot = self._build_digraph(self.regression_tree.content_, None)
        source = dot.source
        self.assertIn("Split p-value = ", source)

    def test_edge_labels_numeric_split(self):
        """Numeric split edges contain <= and > conditions."""
        regression_tree, _, _ = _make_regression_tree()
        dot = self._build_digraph(
            regression_tree.content_, None, feature_names=numpy.asarray(["x0"])
        )
        source = dot.source
        self.assertIn("<=", source)
        self.assertIn(">", source)

    def test_edge_labels_categorical_split(self):
        """Categorical split edges contain 'is' conditions."""
        categorical_regression_tree, _, _ = _make_categorical_regression_tree()
        dot = self._build_digraph(
            categorical_regression_tree.content_,
            None,
            feature_names=numpy.asarray(["category", "noise"]),
        )
        source = dot.source
        self.assertIn("category is", source)

    def test_feature_names_appear_in_edges(self):
        """Display-time feature_names appear in edge labels."""
        regression_tree, _, _ = _make_regression_tree()
        dot = self._build_digraph(
            regression_tree.content_,
            None,
            feature_names=numpy.asarray(["my_feature"]),
        )
        source = dot.source
        self.assertIn("my_feature", source)

    def test_category_labels_appear_in_edges(self):
        """Display-time category_labels appear in edge labels."""
        categorical_regression_tree, _, _ = _make_categorical_regression_tree()
        labels = {0: {0.0: "red", 1.0: "blue"}}
        dot = self._build_digraph(
            categorical_regression_tree.content_,
            labels,
            feature_names=numpy.asarray(["color", "noise"]),
        )
        source = dot.source
        has_label = "red" in source or "blue" in source
        self.assertTrue(has_label)

    def test_node_count_matches_tree(self):
        """Digraph has one node per tree node."""
        dot = self._build_digraph(self.regression_tree.content_, None)
        source = dot.source
        label_count = source.count("label=")
        edge_count = source.count("->")
        node_count = label_count - edge_count
        self.assertGreaterEqual(node_count, 3)

    def test_default_source_has_white_graph_bgcolor(self):
        """Default digraph source sets the graph bgcolor to white."""
        dot = self._build_digraph(self.regression_tree.content_, None)
        self.assertIn("bgcolor=white", dot.source)

    def test_background_color_sets_graph_bgcolor(self):
        """A background_color sets the graph-level bgcolor attribute."""
        dot = self._graphviz._build_digraph(
            self.regression_tree.content_,
            None,
            None,
            self._graphviz._DEFAULT_ROOT_COLORS,
            self._graphviz._DEFAULT_SPLIT_COLORS,
            self._graphviz._DEFAULT_LEAF_PALETTE,
            "black",
            background_color="transparent",
        )
        self.assertIn("bgcolor=transparent", dot.source)

    def test_edges_take_foreground_color(self):
        """Non-truncation edges carry foreground_color as color and fontcolor."""
        dot = self._graphviz._build_digraph(
            self.regression_tree.content_,
            None,
            None,
            self._graphviz._DEFAULT_ROOT_COLORS,
            ("magenta", "lightgray", "darkblue"),
            self._graphviz._DEFAULT_LEAF_PALETTE,
            "deeppink",
        )
        edge_lines = [
            line
            for line in dot.source.splitlines()
            if " -> " in line and "trunc_" not in line
        ]
        self.assertGreater(len(edge_lines), 0)
        for line in edge_lines:
            self.assertIn("color=deeppink", line)
            self.assertIn("fontcolor=deeppink", line)
            self.assertNotIn("color=magenta", line)

    def test_leaf_border_takes_foreground_color(self):
        """Leaf-node borders take foreground_color, not split_colors[0]."""
        import re

        dot = self._graphviz._build_digraph(
            self.regression_tree.content_,
            None,
            None,
            self._graphviz._DEFAULT_ROOT_COLORS,
            ("magenta", "lightgray", "darkblue"),
            self._graphviz._DEFAULT_LEAF_PALETTE,
            "deeppink",
        )
        leaf_node_lines = re.findall(
            r"^\t\d+ \[label=<<table.*$", dot.source, flags=re.MULTILINE
        )
        self.assertGreater(len(leaf_node_lines), 0)
        for line in leaf_node_lines:
            self.assertIn("color=deeppink", line)
            self.assertNotIn("color=magenta", line)

    def test_truncation_node_keeps_split_colors(self):
        """Truncation '...' nodes still take font/fill/border from split_colors."""
        dot = self._graphviz._build_digraph(
            self.regression_tree.content_,
            None,
            None,
            self._graphviz._DEFAULT_ROOT_COLORS,
            ("navy", "lavender", "darkblue"),
            self._graphviz._DEFAULT_LEAF_PALETTE,
            "deeppink",
            max_depth=0,
        )
        trunc_node_lines = [
            line
            for line in dot.source.splitlines()
            if "trunc_" in line and "label" in line and " -> " not in line
        ]
        self.assertGreater(len(trunc_node_lines), 0)
        for line in trunc_node_lines:
            self.assertIn("fontcolor=navy", line)
            self.assertIn("fillcolor=lavender", line)
            self.assertIn("color=darkblue", line)

    def test_truncation_edges_take_foreground_color(self):
        """Truncation edges take foreground_color, not split_colors[0]."""
        dot = self._graphviz._build_digraph(
            self.regression_tree.content_,
            None,
            None,
            self._graphviz._DEFAULT_ROOT_COLORS,
            ("navy", "lavender", "darkblue"),
            self._graphviz._DEFAULT_LEAF_PALETTE,
            "deeppink",
            max_depth=0,
        )
        trunc_edge_lines = [
            line for line in dot.source.splitlines() if " -> trunc_" in line
        ]
        self.assertGreater(len(trunc_edge_lines), 0)
        for line in trunc_edge_lines:
            self.assertIn("color=deeppink", line)
            self.assertNotIn("color=navy", line)

    def test_default_orientation_is_top_down(self):
        """Default digraph source sets graphviz rankdir to TB (top-down)."""
        dot = self._build_digraph(self.regression_tree.content_, None)
        self.assertIn("rankdir=TB", dot.source)

    def test_left_to_right_orientation_sets_rankdir_lr(self):
        """orientation='left-to-right' sets graphviz rankdir to LR."""
        dot = self._build_digraph(
            self.regression_tree.content_, None, orientation="left-to-right"
        )
        self.assertIn("rankdir=LR", dot.source)

    def test_left_to_right_inverts_child_emission_order(self):
        """LR mode emits the higher-prediction child first so green sits on top."""
        root = self.regression_tree.content_
        partition = root.extension
        assert isinstance(partition, sigma._partition.Partition)
        if partition.left.prediction <= partition.right.prediction:
            smaller_child, larger_child = partition.left, partition.right
        else:
            smaller_child, larger_child = partition.right, partition.left
        root_id = str(id(root))
        smaller_edge = f"{root_id} -> {id(smaller_child)}"
        larger_edge = f"{root_id} -> {id(larger_child)}"
        dot_top_down = self._build_digraph(root, None).source
        dot_left_to_right = self._build_digraph(
            root, None, orientation="left-to-right"
        ).source
        self.assertLess(
            dot_top_down.index(smaller_edge), dot_top_down.index(larger_edge)
        )
        self.assertLess(
            dot_left_to_right.index(larger_edge),
            dot_left_to_right.index(smaller_edge),
        )

    def test_reverse_order_xor_truth_table(self):
        """The four (orientation, reverse_order) combinations follow XOR."""
        root = self.regression_tree.content_
        partition = root.extension
        assert isinstance(partition, sigma._partition.Partition)
        if partition.left.prediction <= partition.right.prediction:
            smaller_child, larger_child = partition.left, partition.right
        else:
            smaller_child, larger_child = partition.right, partition.left
        root_id = str(id(root))
        smaller_edge = f"{root_id} -> {id(smaller_child)}"
        larger_edge = f"{root_id} -> {id(larger_child)}"
        sources = {}
        for orientation in ("top-down", "left-to-right"):
            for reverse_order in (False, True):
                source = self._build_digraph(
                    root,
                    None,
                    orientation=orientation,
                    reverse_order=reverse_order,
                ).source
                smaller_first = source.index(smaller_edge) < source.index(
                    larger_edge
                )
                sources[(orientation, reverse_order)] = smaller_first
        self.assertEqual(
            sources[("top-down", False)], sources[("left-to-right", True)]
        )
        self.assertEqual(
            sources[("top-down", True)], sources[("left-to-right", False)]
        )
        self.assertNotEqual(
            sources[("top-down", False)], sources[("top-down", True)]
        )

    def test_reverse_order_default_false_unchanged(self):
        """reverse_order=False gives the same DOT source as omitting it."""
        for orientation in ("top-down", "left-to-right"):
            with self.subTest(orientation=orientation):
                without = self._build_digraph(
                    self.regression_tree.content_, None, orientation=orientation
                ).source
                with_explicit_false = self._build_digraph(
                    self.regression_tree.content_,
                    None,
                    orientation=orientation,
                    reverse_order=False,
                ).source
                self.assertEqual(without, with_explicit_false)

    def test_reverse_order_swaps_leaf_fill_colors(self):
        """A reverse-ordered tree swaps the badge fill colors of low/high leaves."""
        import re

        natural_tree, _, _ = _make_regression_tree()
        reversed_tree, _, _ = _make_regression_tree(reverse_order=True)
        natural_leaves = natural_tree.content_.leaves()
        reversed_leaves = reversed_tree.content_.leaves()
        natural_lowest = min(natural_leaves, key=lambda node: node.prediction)
        natural_highest = max(natural_leaves, key=lambda node: node.prediction)
        reversed_lowest = min(reversed_leaves, key=lambda node: node.prediction)
        reversed_highest = max(
            reversed_leaves, key=lambda node: node.prediction
        )
        natural_source = self._build_digraph(natural_tree.content_, None).source
        reversed_source = self._build_digraph(
            reversed_tree.content_, None
        ).source

        def find_fillcolor(source, node_id):
            """Return the leaf fillcolor declared on a given graphviz node id."""
            pattern = rf"^\t{node_id} \[.*?fillcolor=(\"#[0-9A-F]+\"|\w+)"
            match = re.search(pattern, source, flags=re.MULTILINE)
            if match is None:
                raise AssertionError(f"fillcolor for node {node_id} not found")
            return match.group(1)

        natural_low = find_fillcolor(natural_source, str(id(natural_lowest)))
        natural_high = find_fillcolor(natural_source, str(id(natural_highest)))
        reversed_low = find_fillcolor(reversed_source, str(id(reversed_lowest)))
        reversed_high = find_fillcolor(
            reversed_source, str(id(reversed_highest))
        )
        self.assertNotEqual(natural_low, natural_high)
        self.assertEqual(natural_low, reversed_high)
        self.assertEqual(natural_high, reversed_low)


@unittest.skipUnless(_HAS_GRAPHVIZ, "graphviz not installed")
class TestToImage(unittest.TestCase):
    """Tests for the Tree.to_image public method."""

    __slots__ = ()

    def test_regression_tree_svg_returns_bytes(self):
        """RegressionTree to_image returns valid SVG bytes."""
        regression_tree, _, _ = _make_regression_tree()
        result = regression_tree.to_image("svg")
        self.assertIsInstance(result, bytes)
        self.assertTrue(
            result.startswith(b"<?xml") or result.startswith(b"<svg")
        )

    def test_regression_tree_png_returns_bytes(self):
        """RegressionTree to_image returns valid PNG bytes."""
        regression_tree, _, _ = _make_regression_tree()
        result = regression_tree.to_image("png")
        self.assertIsInstance(result, bytes)
        self.assertTrue(result.startswith(b"\x89PNG"))

    def test_regression_tree_pdf_returns_bytes(self):
        """RegressionTree to_image returns valid PDF bytes."""
        regression_tree, _, _ = _make_regression_tree()
        result = regression_tree.to_image("pdf")
        self.assertIsInstance(result, bytes)
        self.assertTrue(result.startswith(b"%PDF-"))

    def test_classification_tree_svg_returns_bytes(self):
        """ClassificationTree to_image returns valid SVG bytes."""
        classification_tree, _, _ = _make_classification_tree()
        result = classification_tree.to_image("svg")
        self.assertIsInstance(result, bytes)
        self.assertTrue(
            result.startswith(b"<?xml") or result.startswith(b"<svg")
        )

    def test_classification_tree_png_returns_bytes(self):
        """ClassificationTree to_image returns valid PNG bytes."""
        classification_tree, _, _ = _make_classification_tree()
        result = classification_tree.to_image("png")
        self.assertIsInstance(result, bytes)
        self.assertTrue(result.startswith(b"\x89PNG"))

    def test_classification_tree_pdf_returns_bytes(self):
        """ClassificationTree to_image returns valid PDF bytes."""
        classification_tree, _, _ = _make_classification_tree()
        result = classification_tree.to_image("pdf")
        self.assertIsInstance(result, bytes)
        self.assertTrue(result.startswith(b"%PDF-"))

    def test_regression_tree_gif_returns_bytes(self):
        """RegressionTree to_image returns valid GIF bytes."""
        regression_tree, _, _ = _make_regression_tree()
        result = regression_tree.to_image("gif")
        self.assertIsInstance(result, bytes)
        self.assertTrue(result.startswith(b"GIF8"))

    def test_classification_tree_gif_returns_bytes(self):
        """ClassificationTree to_image returns valid GIF bytes."""
        classification_tree, _, _ = _make_classification_tree()
        result = classification_tree.to_image("gif")
        self.assertIsInstance(result, bytes)
        self.assertTrue(result.startswith(b"GIF8"))

    def test_higher_dpi_produces_larger_png(self):
        """Higher dpi produces a larger PNG output."""
        regression_tree, _, _ = _make_regression_tree()
        low_dpi = regression_tree.to_image("png", dpi=72)
        high_dpi = regression_tree.to_image("png", dpi=216)
        self.assertGreater(len(high_dpi), len(low_dpi))

    def test_higher_dpi_produces_larger_gif(self):
        """Higher dpi produces a larger GIF output."""
        regression_tree, _, _ = _make_regression_tree()
        low_dpi = regression_tree.to_image("gif", dpi=72)
        high_dpi = regression_tree.to_image("gif", dpi=216)
        self.assertGreater(len(high_dpi), len(low_dpi))

    def test_raises_not_fitted_error(self):
        """to_image raises NotFittedError before fit."""
        import sklearn.exceptions

        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal"
        )
        with self.assertRaises(sklearn.exceptions.NotFittedError):
            regression_tree.to_image("svg")

    def test_feature_names_in_svg(self):
        """Display-time feature_names appear in the SVG output."""
        regression_tree, _, _ = _make_regression_tree()
        result = regression_tree.to_image("svg", feature_names=["spread"])
        self.assertIn(b"spread", result)

    def test_category_labels_in_svg(self):
        """Display-time category_labels appear in the SVG output."""
        regression_tree, _, _ = _make_categorical_regression_tree()
        result = regression_tree.to_image(
            "svg",
            feature_names=["color", "noise"],
            category_labels={0: {0.0: "red", 1.0: "blue"}},
        )
        has_label = b"red" in result or b"blue" in result
        self.assertTrue(has_label)

    def test_background_color_svg_returns_bytes(self):
        """to_image with a background_color returns valid SVG bytes."""
        regression_tree, _, _ = _make_regression_tree()
        result = regression_tree.to_image("svg", background_color="transparent")
        self.assertIsInstance(result, bytes)
        self.assertTrue(
            result.startswith(b"<?xml") or result.startswith(b"<svg")
        )

    def test_background_color_png_returns_bytes(self):
        """to_image with a background_color returns valid PNG bytes."""
        regression_tree, _, _ = _make_regression_tree()
        result = regression_tree.to_image("png", background_color="transparent")
        self.assertIsInstance(result, bytes)
        self.assertTrue(result.startswith(b"\x89PNG"))

    def test_background_color_pdf_returns_bytes(self):
        """to_image with a background_color returns valid PDF bytes."""
        regression_tree, _, _ = _make_regression_tree()
        result = regression_tree.to_image("pdf", background_color="transparent")
        self.assertIsInstance(result, bytes)
        self.assertTrue(result.startswith(b"%PDF-"))

    def test_default_orientation_svg_has_rankdir_tb(self):
        """Default to_image('svg') output reflects rankdir=TB layout."""
        regression_tree, _, _ = _make_regression_tree()
        dot_source = sigma.export_graphviz(regression_tree)
        self.assertIn("rankdir=TB", dot_source)
        result = regression_tree.to_image("svg")
        self.assertIsInstance(result, bytes)

    def test_left_to_right_orientation_emits_rankdir_lr_in_dot(self):
        """to_image with orientation='left-to-right' emits rankdir=LR in DOT."""
        regression_tree, _, _ = _make_regression_tree()
        dot_source = sigma.export_graphviz(
            regression_tree, orientation="left-to-right"
        )
        self.assertIn("rankdir=LR", dot_source)
        result = regression_tree.to_image("svg", orientation="left-to-right")
        self.assertIsInstance(result, bytes)


@unittest.skipUnless(_HAS_GRAPHVIZ, "graphviz not installed")
class TestLeafBadges(unittest.TestCase):
    """Tests for numbered circle badges on leaf nodes."""

    __slots__ = ()

    def setUp(self):
        """Set up fitted models for badge tests."""
        from sigma import _graphviz
        from sigma import _node

        self._graphviz = _graphviz
        self._node = _node
        self.regression_tree, _, _ = _make_regression_tree()
        self.classification_tree, _, _ = _make_classification_tree()

    def test_leaf_badges_present_regression(self):
        """Regression leaves have numbered badges in SVG output."""
        result = self.regression_tree.to_image("svg")
        source = result.decode()
        leaves = self.regression_tree.content_.leaves()
        for i in range(1, len(leaves) + 1):
            self.assertIn(f">{i}<", source)

    def test_leaf_badges_present_classification(self):
        """Classification leaves have numbered badges in SVG output."""
        result = self.classification_tree.to_image("svg")
        source = result.decode()
        leaves = self.classification_tree.content_.leaves()
        for i in range(1, len(leaves) + 1):
            self.assertIn(f">{i}<", source)

    def test_badge_order_regression(self):
        """Tree.leaves_ orders leaves by ascending prediction with leaf_id 0..N-1."""
        leaves = self.regression_tree.leaves_
        ranked = sorted(leaves, key=lambda node: node.prediction)
        for expected_index, node in enumerate(ranked):
            extension = node.extension
            assert isinstance(extension, sigma._extension.Leaf)
            self.assertEqual(extension.leaf_id, expected_index)
            self.assertIs(self.regression_tree.leaves_[expected_index], node)

    def test_badge_order_classification(self):
        """Tree.leaves_ orders classification leaves by descending distribution tuple."""
        leaves = self.classification_tree.leaves_
        ranked = sorted(
            leaves,
            key=lambda node: tuple(node.class_distribution),
            reverse=True,
        )
        for expected_index, node in enumerate(ranked):
            extension = node.extension
            assert isinstance(extension, sigma._extension.Leaf)
            self.assertEqual(extension.leaf_id, expected_index)
            self.assertIs(
                self.classification_tree.leaves_[expected_index], node
            )

    def test_single_leaf_tree_has_badge(self):
        """A single-node tree gets leaf_id 0 and displays badge 1."""
        X = numpy.ones((10, 1))
        y = numpy.ones(10)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        regression_tree.fit(X, y)
        extension = regression_tree.content_.extension
        assert isinstance(extension, sigma._extension.Leaf)
        self.assertEqual(extension.leaf_id, 0)
        result = regression_tree.to_image("svg")
        source = result.decode()
        self.assertIn(">1<", source)

    def test_badge_order_regression_reversed(self):
        """A reverse-ordered regression tree displays the flipped badge per leaf."""
        natural_tree, _, _ = _make_regression_tree()
        reversed_tree, _, _ = _make_regression_tree(reverse_order=True)
        natural_leaves = natural_tree.leaves_
        reversed_leaves = reversed_tree.leaves_
        natural_predictions = [leaf.prediction for leaf in natural_leaves]
        reversed_predictions = [leaf.prediction for leaf in reversed_leaves]
        self.assertEqual(
            reversed_predictions, list(reversed(natural_predictions))
        )
        dot_source = self._graphviz._build_digraph(
            reversed_tree.content_,
            None,
            None,
            self._graphviz._DEFAULT_ROOT_COLORS,
            self._graphviz._DEFAULT_SPLIT_COLORS,
            self._graphviz._DEFAULT_LEAF_PALETTE,
            foreground_color="black",
        ).source
        for leaf in reversed_leaves:
            leaf_extension = leaf.extension
            assert isinstance(leaf_extension, sigma._extension.Leaf)
            expected_badge = leaf_extension.leaf_id + 1
            label_token = f"<b>{expected_badge}</b>"
            self.assertIn(label_token, dot_source)

    def test_badge_order_classification_reversed(self):
        """A reverse-ordered classification tree displays the flipped badge per leaf."""
        from sigma._tree_classification import ClassificationTree

        rng = numpy.random.default_rng(42)
        X = rng.standard_normal((60, 2))
        y = (X[:, 0] > 0).astype(float)
        reversed_tree = ClassificationTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            reverse_order=True,
        )
        reversed_tree.fit(X, y)
        dot_source = self._graphviz._build_digraph(
            reversed_tree.content_,
            None,
            None,
            self._graphviz._DEFAULT_ROOT_COLORS,
            self._graphviz._DEFAULT_SPLIT_COLORS,
            self._graphviz._DEFAULT_LEAF_PALETTE,
            foreground_color="black",
        ).source
        for leaf in reversed_tree.leaves_:
            leaf_extension = leaf.extension
            assert isinstance(leaf_extension, sigma._extension.Leaf)
            expected_badge = leaf_extension.leaf_id + 1
            label_token = f"<b>{expected_badge}</b>"
            self.assertIn(label_token, dot_source)


@unittest.skipUnless(_HAS_GRAPHVIZ, "graphviz not installed")
class TestPlotTreeDecoration(unittest.TestCase):
    """Tests for decoration rendering in to_image output."""

    __slots__ = ()

    def test_decoration_appears_in_svg(self):
        """Node decorations appear in the SVG output."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)

        def decorator(
            X_active, y_active, w_active, offset_active, side_data_active
        ):
            """Return a distinctive tag for every node."""
            return "DECO_TAG"

        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            decorator=decorator,
        )
        regression_tree.fit(X, y)
        result = regression_tree.to_image("svg")
        self.assertIn(b"DECO_TAG", result)

    def test_absent_decoration_leaves_svg_clean(self):
        """Without a decorator, no decoration marker leaks into the SVG."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        regression_tree.fit(X, y)
        result = regression_tree.to_image("svg")
        self.assertNotIn(b"None", result)


@unittest.skipUnless(_HAS_GRAPHVIZ, "graphviz not installed")
class TestBuildDigraphMaxDepth(unittest.TestCase):
    """Tests for the max_depth display knob on to_image / _build_digraph."""

    __slots__ = ()

    def setUp(self):
        """Set up a fitted three-step regression tree and a build helper."""
        from sigma import _graphviz
        from sigma import _node

        self._graphviz = _graphviz
        self._node = _node
        self.regression_tree = _make_three_step_regression_tree()

    def _build(self, max_depth):
        """Invoke _build_digraph with a max_depth and default colors."""
        digraph = self._graphviz._build_digraph(
            self.regression_tree.content_,
            None,
            None,
            self._graphviz._DEFAULT_ROOT_COLORS,
            self._graphviz._DEFAULT_SPLIT_COLORS,
            self._graphviz._DEFAULT_LEAF_PALETTE,
            foreground_color="black",
            max_depth=max_depth,
        )
        return digraph

    def test_max_depth_none_emits_no_placeholders(self):
        """With the default max_depth, no ... placeholders are emitted."""
        source = self._build(None).source
        count = source.count('label="..."')
        self.assertEqual(count, 0)

    def test_max_depth_zero_emits_one_truncation_placeholder(self):
        """max_depth=0 emits exactly one ... placeholder below the root."""
        source = self._build(0).source
        count = source.count('label="..."')
        self.assertEqual(count, 1)
        root_id = str(id(self.regression_tree.content_))
        placeholder_id = f"trunc_{root_id}"
        self.assertIn(f"{root_id} -> {placeholder_id}", source)

    def test_max_depth_one_emits_placeholder_per_truncated_or_leaf_node(self):
        """At max_depth=1, every depth-1 child (truncated or leaf) gets a ..."""
        source = self._build(1).source
        count = source.count('label="..."')
        partition = self.regression_tree.content_.extension
        assert isinstance(partition, sigma._partition.Partition)
        depth_one_children = [partition.left, partition.right]
        self.assertEqual(count, len(depth_one_children))

    def test_max_depth_exceeds_tree_depth_still_marks_leaves(self):
        """A max_depth above the tree depth still emits ... below every leaf."""
        source = self._build(99).source
        count = source.count('label="..."')
        leaves = self.regression_tree.content_.leaves()
        self.assertEqual(count, len(leaves))

    def test_max_depth_negative_raises_value_error(self):
        """to_image with a negative max_depth raises ValueError."""
        with self.assertRaises(ValueError):
            self.regression_tree.to_image("svg", max_depth=-1)


@unittest.skipUnless(_HAS_GRAPHVIZ, "graphviz not installed")
class TestBuildDigraphPrecision(unittest.TestCase):
    """Tests for the precision display knob on to_image / _build_digraph."""

    __slots__ = ()

    def setUp(self):
        """Set up a fitted step regression tree and a build helper."""
        from sigma import _graphviz

        self._graphviz = _graphviz
        self.regression_tree, _, _ = _make_regression_tree()

    def _build(self, precision):
        """Invoke _build_digraph with a precision and default colors."""
        digraph = self._graphviz._build_digraph(
            self.regression_tree.content_,
            None,
            None,
            self._graphviz._DEFAULT_ROOT_COLORS,
            self._graphviz._DEFAULT_SPLIT_COLORS,
            self._graphviz._DEFAULT_LEAF_PALETTE,
            foreground_color="black",
            precision=precision,
        )
        return digraph

    def test_precision_threads_into_node_labels(self):
        """precision is forwarded into prediction labels in the dot output."""
        source = self._build(5).source
        self.assertIn("0.00000", source)
        self.assertIn("10.00000", source)

    def test_precision_threads_into_edge_labels(self):
        """precision is forwarded into split-threshold edge labels."""
        X = (numpy.arange(1, 41, dtype=float) + 0.25).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20.5, 0.0, 10.0)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        regression_tree.fit(X, y)
        digraph = self._graphviz._build_digraph(
            regression_tree.content_,
            None,
            None,
            self._graphviz._DEFAULT_ROOT_COLORS,
            self._graphviz._DEFAULT_SPLIT_COLORS,
            self._graphviz._DEFAULT_LEAF_PALETTE,
            foreground_color="black",
            precision=5,
        )
        self.assertIn("20.75000", digraph.source)

    def test_precision_negative_raises_value_error(self):
        """to_image with a negative precision raises ValueError."""
        with self.assertRaises(ValueError):
            self.regression_tree.to_image("svg", precision=-1)


class TestEllipsize(unittest.TestCase):
    """Tests for the _ellipsize string truncation helper."""

    __slots__ = ()

    def test_returns_string_unchanged_when_shorter_than_limit(self):
        """A string shorter than max_length is returned unchanged."""
        from sigma._tree_text import _ellipsize

        self.assertEqual(_ellipsize("Bla", 10), "Bla")

    def test_returns_string_unchanged_at_exact_max_length(self):
        """A string of length equal to max_length is returned unchanged."""
        from sigma._tree_text import _ellipsize

        self.assertEqual(_ellipsize("Hello", 5), "Hello")

    def test_truncates_longer_string_with_trailing_ellipsis(self):
        """A longer string is truncated to max_length characters ending in '...'."""
        from sigma._tree_text import _ellipsize

        result = _ellipsize("Bla bla bla bla", 6)
        self.assertEqual(result, "Bla...")
        self.assertEqual(len(result), 6)

    def test_max_length_below_three_returns_truncated_ellipsis_only(self):
        """When max_length is smaller than the ellipsis itself, only ellipsis chars are kept."""
        from sigma._tree_text import _ellipsize

        self.assertEqual(_ellipsize("Hello", 2), "..")
        self.assertEqual(_ellipsize("Hello", 1), ".")
        self.assertEqual(_ellipsize("Hello", 0), "")


@unittest.skipUnless(_HAS_GRAPHVIZ, "graphviz not installed")
class TestBuildDigraphMaxBranchLength(unittest.TestCase):
    """Tests for the max_branch_length display knob on to_image / _build_digraph."""

    __slots__ = ()

    def setUp(self):
        """Set up a fitted regression tree and a deliberately long feature name."""
        from sigma import _graphviz

        self._graphviz = _graphviz
        self.regression_tree, _, _ = _make_regression_tree()
        self.long_feature_name = (
            "AVeryLongFeatureNameThatExceedsTheDefaultBranchLengthOfFifty"
        )

    def _build(self, max_branch_length):
        """Invoke _build_digraph with the long feature name and a max_branch_length."""
        digraph = self._graphviz._build_digraph(
            self.regression_tree.content_,
            None,
            None,
            self._graphviz._DEFAULT_ROOT_COLORS,
            self._graphviz._DEFAULT_SPLIT_COLORS,
            self._graphviz._DEFAULT_LEAF_PALETTE,
            foreground_color="black",
            feature_names=numpy.array([self.long_feature_name]),
            max_branch_length=max_branch_length,
        )
        return digraph

    def test_default_truncates_long_branch_labels_with_ellipsis(self):
        """At the default max_branch_length, a long branch label ends in '...'."""
        digraph = self._graphviz._build_digraph(
            self.regression_tree.content_,
            None,
            None,
            self._graphviz._DEFAULT_ROOT_COLORS,
            self._graphviz._DEFAULT_SPLIT_COLORS,
            self._graphviz._DEFAULT_LEAF_PALETTE,
            foreground_color="black",
            feature_names=numpy.array([self.long_feature_name]),
        )
        self.assertIn("...", digraph.source)
        self.assertNotIn(self.long_feature_name, digraph.source)

    def test_larger_max_branch_length_keeps_full_label(self):
        """A max_branch_length above the natural label length leaves it untouched."""
        digraph = self._build(max_branch_length=1000)
        self.assertIn(self.long_feature_name, digraph.source)
        self.assertNotIn("...", digraph.source)

    def test_small_max_branch_length_still_renders(self):
        """A very small max_branch_length still produces a valid DOT source."""
        digraph = self._build(max_branch_length=4)
        self.assertTrue(digraph.source)
        self.assertIn("...", digraph.source)

    def test_max_branch_length_zero_raises_value_error(self):
        """to_image with max_branch_length=0 raises ValueError."""
        with self.assertRaises(ValueError):
            self.regression_tree.to_image("svg", max_branch_length=0)

    def test_max_branch_length_negative_raises_value_error(self):
        """to_image with a negative max_branch_length raises ValueError."""
        with self.assertRaises(ValueError):
            self.regression_tree.to_image("svg", max_branch_length=-1)

    def test_max_branch_length_non_int_raises_value_error(self):
        """to_image with a non-integer max_branch_length raises ValueError."""
        with self.assertRaises(ValueError):
            self.regression_tree.to_image("svg", max_branch_length=50.0)


def _content_node_widths(plain_output, exclude_prefix="trunc_"):
    """Return per-node widths (in inches) parsed from graphviz plain output."""
    widths = {}
    for line in plain_output.splitlines():
        if not line.startswith("node "):
            continue
        tokens = line.split(None, 5)
        if len(tokens) < 5:
            continue
        name = tokens[1]
        if exclude_prefix and name.startswith(exclude_prefix):
            continue
        widths[name] = float(tokens[4])
    return widths


@unittest.skipUnless(_HAS_GRAPHVIZ, "graphviz not installed")
class TestUniformNodeWidths(unittest.TestCase):
    """Tests for the uniform-width behavior across node boxes."""

    __slots__ = ()

    def setUp(self):
        """Set up fitted models with deliberately varied label lengths."""
        from sigma import _graphviz

        self._graphviz = _graphviz
        self.three_step = _make_three_step_regression_tree()
        self.classification_tree, _, _ = _make_classification_tree()

    def _build(self, root, **kwargs):
        """Invoke _build_digraph with default colors."""
        digraph = self._graphviz._build_digraph(
            root,
            None,
            None,
            self._graphviz._DEFAULT_ROOT_COLORS,
            self._graphviz._DEFAULT_SPLIT_COLORS,
            self._graphviz._DEFAULT_LEAF_PALETTE,
            foreground_color="black",
            **kwargs,
        )
        return digraph

    def test_dot_source_sets_uniform_width(self):
        """The DOT source carries a node-level width attribute."""
        dot = self._build(self.three_step.content_)
        source = dot.source
        self.assertIn("width=", source)
        self.assertNotIn("fixedsize=true", source)
        self.assertNotIn("fixedsize=shape", source)

    def test_leaf_nodes_carry_uniform_width_attribute(self):
        """Leaf node lines in the DOT source carry a node-level width."""
        dot = self._build(self.three_step.content_)
        source = dot.source
        leaf_lines = [line for line in source.splitlines() if "<<table" in line]
        self.assertGreater(len(leaf_lines), 0)
        for line in leaf_lines:
            self.assertRegex(line, r"\bwidth=\d")

    def test_leaf_html_table_carries_inner_width(self):
        """Leaf HTML labels carry a table-level width so badges left-align."""
        dot = self._build(self.three_step.content_)
        source = dot.source
        leaf_lines = [line for line in source.splitlines() if "<<table" in line]
        self.assertGreater(len(leaf_lines), 0)
        for line in leaf_lines:
            self.assertRegex(line, r'<<table[^>]*\bwidth="\d+"')

    def test_all_content_nodes_have_equal_rendered_width(self):
        """All non-truncation node boxes render at the same width in plain."""
        dot = self._build(self.three_step.content_)
        plain = dot.pipe(format="plain").decode("utf-8")
        widths = _content_node_widths(plain)
        self.assertGreater(len(widths), 1)
        rounded = {round(w, 3) for w in widths.values()}
        self.assertEqual(len(rounded), 1)

    def test_classification_tree_content_nodes_have_equal_width(self):
        """ClassificationTree node boxes also render at a single uniform width."""
        dot = self._build(self.classification_tree.content_)
        plain = dot.pipe(format="plain").decode("utf-8")
        widths = _content_node_widths(plain)
        self.assertGreater(len(widths), 1)
        rounded = {round(w, 3) for w in widths.values()}
        self.assertEqual(len(rounded), 1)

    def test_uniform_width_at_least_widest_natural_width(self):
        """The forced uniform width fits the widest natural label."""
        dot_uniform = self._build(self.three_step.content_)
        plain_uniform = dot_uniform.pipe(format="plain").decode("utf-8")
        uniform_widths = _content_node_widths(plain_uniform)
        forced = next(iter(uniform_widths.values()))
        natural_dot = self._graphviz._emit_digraph(
            self.three_step.content_,
            None,
            None,
            self._graphviz._DEFAULT_ROOT_COLORS,
            self._graphviz._DEFAULT_SPLIT_COLORS,
            self._graphviz._DEFAULT_LEAF_PALETTE,
            "black",
        )
        plain_natural = natural_dot.pipe(format="plain").decode("utf-8")
        natural_widths = _content_node_widths(plain_natural)
        self.assertAlmostEqual(forced, max(natural_widths.values()), places=3)

    def test_truncation_placeholders_remain_auto_sized(self):
        """trunc_* placeholders are not forced to the uniform width."""
        dot = self._build(self.three_step.content_, max_depth=1)
        plain = dot.pipe(format="plain").decode("utf-8")
        content_widths = _content_node_widths(plain)
        placeholder_widths = _content_node_widths(plain, exclude_prefix=None)
        placeholder_only = {
            name: width
            for name, width in placeholder_widths.items()
            if name.startswith("trunc_")
        }
        self.assertGreater(len(placeholder_only), 0)
        forced = next(iter(content_widths.values()))
        for width in placeholder_only.values():
            self.assertLess(width, forced)

    def test_single_node_tree_does_not_force_width(self):
        """A single-leaf tree skips uniform sizing and renders without error."""
        X = numpy.ones((10, 1))
        y = numpy.ones(10)
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        regression_tree.fit(X, y)
        dot = self._build(regression_tree.content_)
        source = dot.source
        self.assertNotRegex(source, r"\bwidth=\d")


@unittest.skipUnless(_HAS_GRAPHVIZ, "graphviz not installed")
class TestBoldPredictionValues(unittest.TestCase):
    """Tests for the predicted-value bolding in plotted node labels."""

    __slots__ = ()

    def _build(self, root):
        """Invoke _build_digraph with default colors."""
        from sigma import _graphviz

        digraph = _graphviz._build_digraph(
            root,
            None,
            None,
            _graphviz._DEFAULT_ROOT_COLORS,
            _graphviz._DEFAULT_SPLIT_COLORS,
            _graphviz._DEFAULT_LEAF_PALETTE,
            "black",
        )
        return digraph

    def test_regression_value_is_wrapped_in_bold(self):
        """The regression prediction value appears wrapped in <b>...</b>."""
        regression_tree, _, _ = _make_regression_tree()
        source = self._build(regression_tree.content_).source
        self.assertIn("Predicted mean = <b>", source)
        self.assertIn("</b> (", source)

    def test_regression_ci_bounds_are_not_bold(self):
        """CI bounds in the regression label are not wrapped in <b>...</b>."""
        regression_tree, _, _ = _make_regression_tree()
        source = self._build(regression_tree.content_).source
        self.assertNotIn(" to <b>", source)

    def test_classification_each_class_value_is_bold(self):
        """Every class probability value is wrapped in <b>...</b>."""
        classification_tree, _, _ = _make_classification_tree()
        source = self._build(classification_tree.content_).source
        opens = source.count("proba. = <b>")
        closes = source.count("</b>")
        self.assertGreater(opens, 0)
        self.assertGreaterEqual(closes, opens)

    def test_survival_each_metric_value_is_bold(self):
        """Each survival metric value is wrapped in <b>...</b>."""
        survival_tree = _make_survival_tree()
        source = self._build(survival_tree.content_).source
        self.assertIn("Median survival = <b>", source)
        self.assertIn("Survival at 5 years = <b>", source)

    def test_split_p_value_is_not_bold(self):
        """The split p-value text is not wrapped in <b>...</b>."""
        regression_tree, _, _ = _make_regression_tree()
        source = self._build(regression_tree.content_).source
        self.assertNotIn("Split p-value = <b>", source)

    def test_obs_count_is_not_bold(self):
        """The observation count line is not wrapped in <b>...</b>."""
        regression_tree, _, _ = _make_regression_tree()
        source = self._build(regression_tree.content_).source
        self.assertNotIn("Obs. count = <b>", source)

    def test_export_text_has_no_bold_markers(self):
        """Plain-text export contains neither HTML tags nor sentinel chars."""
        import sigma
        from sigma import _tree_text

        regression_tree, _, _ = _make_regression_tree()
        text = sigma.export_text(regression_tree)
        self.assertNotIn("<b>", text)
        self.assertNotIn("</b>", text)
        self.assertNotIn(_tree_text._BOLD_OPEN, text)
        self.assertNotIn(_tree_text._BOLD_CLOSE, text)
