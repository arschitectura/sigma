"""Unit tests for the module-level export_text, export_graphviz, and
export_image functions, plus the Tree.to_text and Tree.to_image shortcuts."""

import importlib.util
import inspect
import io
import os
import tempfile
import typing
import unittest

import numpy
import sklearn.exceptions

import sigma
import sigma._tree
import sigma._tree_classification
import sigma._tree_regression
import sigma._tree_survival

_HAS_GRAPHVIZ = importlib.util.find_spec("graphviz") is not None
_HAS_MATPLOTLIB = importlib.util.find_spec("matplotlib") is not None


def _fit_step_regression_tree(reverse_order: bool = False):
    """Fit a simple step-function regression tree used across export tests."""
    X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
    y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
    regression_tree = sigma._tree_regression.RegressionTree(
        correlation="normal",
        min_splits=2,
        min_buckets=1,
        reverse_order=reverse_order,
    )
    regression_tree.fit(X, y)
    return regression_tree


def _fit_three_step_regression_tree():
    """Fit a regression tree on a 3-step response, yielding a depth >= 2 tree."""
    X = numpy.arange(1, 81, dtype=float).reshape(-1, 1)
    y = numpy.where(X.ravel() < 20, 0.0, numpy.where(X.ravel() < 60, 5.0, 10.0))
    regression_tree = sigma._tree_regression.RegressionTree(
        correlation="normal", min_splits=2, min_buckets=1
    )
    regression_tree.fit(X, y)
    return regression_tree


def _fit_categorical_regression_tree():
    """Fit a regression tree with a binary categorical signal."""
    rng = numpy.random.default_rng(42)
    n = 40
    categorical_column = numpy.repeat([0.0, 1.0], n // 2)
    noise = rng.standard_normal(n)
    y = numpy.where(categorical_column == 0.0, 0.0, 10.0)
    X = numpy.column_stack([categorical_column, noise])
    regression_tree = sigma._tree_regression.RegressionTree(
        correlation="normal",
        categorical_features=[0],
        min_splits=2,
        min_buckets=1,
    )
    regression_tree.fit(X, y)
    return regression_tree


def _fit_multi_value_categorical_regression_tree():
    """Fit a regression tree with a 4-level numeric categorical signal."""
    rng = numpy.random.default_rng(42)
    n_per = 30
    cat = numpy.tile([0.0, 1.0, 2.0, 3.0], n_per)
    noise = rng.standard_normal(n_per * 4)
    y = numpy.where(cat < 2.0, 0.0, 10.0) + 0.01 * noise
    X = numpy.column_stack([cat, noise])
    regression_tree = sigma._tree_regression.RegressionTree(
        correlation="normal",
        categorical_features=[0],
        min_splits=4,
        min_buckets=2,
    )
    regression_tree.fit(X, y)
    return regression_tree


def _fit_mixed_cardinality_categorical_regression_tree():
    """Fit a regression tree with a 4-level categorical signal split 1-vs-3."""
    rng = numpy.random.default_rng(42)
    n_per = 30
    cat = numpy.tile([0.0, 1.0, 2.0, 3.0], n_per)
    noise = rng.standard_normal(n_per * 4)
    y = numpy.where(cat == 0.0, 0.0, 10.0) + 0.01 * noise
    X = numpy.column_stack([cat, noise])
    regression_tree = sigma._tree_regression.RegressionTree(
        correlation="normal",
        categorical_features=[0],
        min_splits=4,
        min_buckets=2,
    )
    regression_tree.fit(X, y)
    return regression_tree


def _fit_boolean_regression_tree():
    """Fit a regression tree with a bool-dtype DataFrame column."""
    import pandas

    rng = numpy.random.default_rng(42)
    n = 40
    flag = numpy.repeat([False, True], n // 2)
    noise = rng.standard_normal(n)
    y = numpy.where(flag, 10.0, 0.0)
    X = pandas.DataFrame({"flag": flag, "noise": noise})
    regression_tree = sigma._tree_regression.RegressionTree(
        correlation="normal",
        min_splits=2,
        min_buckets=1,
    )
    regression_tree.fit(X, y)
    return regression_tree


class TestExportText(unittest.TestCase):
    """Tests for the module-level export_text free function."""

    __slots__ = ()

    def test_export_text_returns_string(self):
        """Returns a non-empty Python str matching the to_text output."""
        regression_tree = _fit_step_regression_tree()
        result = sigma.export_text(regression_tree)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_export_text_matches_to_text_output(self):
        """Returned string equals the result of Tree.to_text."""
        regression_tree = _fit_step_regression_tree()
        from_method = regression_tree.to_text()
        result = sigma.export_text(regression_tree)
        self.assertEqual(result, from_method)

    def test_export_text_forwards_feature_names(self):
        """Display-time feature_names are forwarded to to_text."""
        regression_tree = _fit_step_regression_tree()
        result = sigma.export_text(regression_tree, feature_names=["spread"])
        self.assertIn("spread", result)

    def test_export_text_forwards_response_name(self):
        """response_name keyword is forwarded and rendered in the output."""
        regression_tree = _fit_step_regression_tree()
        result = sigma.export_text(regression_tree, response_name="Price")
        self.assertIn("Price mean", result)
        self.assertNotIn("Predicted mean", result)

    def test_export_text_forwards_max_depth(self):
        """max_depth=0 truncates everything below the root with a marker."""
        regression_tree = _fit_three_step_regression_tree()
        result = sigma.export_text(regression_tree, max_depth=0)
        self.assertIn("All records", result)
        self.assertIn("└── ...", result)
        self.assertNotIn("├──", result)

    def test_export_text_forwards_precision(self):
        """precision=0 suppresses the decimal point in formatted predictions."""
        regression_tree = _fit_step_regression_tree()
        result = sigma.export_text(regression_tree, precision=0)
        self.assertNotIn("0.000", result)
        self.assertNotIn("10.000", result)

    def test_export_text_raises_not_fitted(self):
        """Raises NotFittedError when called on an unfit estimator."""
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal"
        )
        with self.assertRaises(sklearn.exceptions.NotFittedError):
            sigma.export_text(regression_tree)

    def test_export_text_reverse_order_inverts_branch_order(self):
        """A reverse-ordered tree swaps the two branch lines under the root."""
        natural_tree = _fit_step_regression_tree()
        reversed_tree = _fit_step_regression_tree(reverse_order=True)
        natural = sigma.export_text(natural_tree)
        reversed_text = sigma.export_text(reversed_tree)
        natural_lines = [
            line
            for line in natural.splitlines()
            if "├──" in line or "└──" in line
        ]
        reversed_lines = [
            line
            for line in reversed_text.splitlines()
            if "├──" in line or "└──" in line
        ]
        self.assertEqual(len(natural_lines), 2)
        self.assertEqual(len(reversed_lines), 2)
        natural_first_label = natural_lines[0].split("├──", 1)[1].strip()
        reversed_first_label = reversed_lines[0].split("├──", 1)[1].strip()
        self.assertNotEqual(natural_first_label, reversed_first_label)

    def test_export_text_renders_best_leaf_before_worst_by_default(self):
        """Without reverse_order, the higher-prediction branch appears first."""
        natural_tree = _fit_step_regression_tree()
        text = sigma.export_text(natural_tree)
        branch_lines = [
            line for line in text.splitlines() if "├──" in line or "└──" in line
        ]
        self.assertEqual(len(branch_lines), 2)
        first_branch = branch_lines[0]
        second_branch = branch_lines[1]
        self.assertIn(" > ", first_branch)
        self.assertIn(" <= ", second_branch)

    def test_export_text_renders_worst_leaf_before_best_when_reversed(self):
        """With reverse_order=True, the lower-prediction branch appears first."""
        reversed_tree = _fit_step_regression_tree(reverse_order=True)
        text = sigma.export_text(reversed_tree)
        branch_lines = [
            line for line in text.splitlines() if "├──" in line or "└──" in line
        ]
        self.assertEqual(len(branch_lines), 2)
        first_branch = branch_lines[0]
        second_branch = branch_lines[1]
        self.assertIn(" <= ", first_branch)
        self.assertIn(" > ", second_branch)

    def test_to_text_matches_export_text_for_reversed_tree(self):
        """Tree.to_text() on a reverse-ordered tree matches export_text output."""
        regression_tree = _fit_step_regression_tree(reverse_order=True)
        from_method = regression_tree.to_text()
        result = sigma.export_text(regression_tree)
        self.assertEqual(result, from_method)

    def test_export_text_table_header_column_order(self):
        """Header row lists prediction, count, share, p-value, leaf-index in order."""
        regression_tree = _fit_step_regression_tree()
        result = sigma.export_text(regression_tree)
        header_line = result.splitlines()[0]
        predicted_index = header_line.index("Predicted mean")
        count_index = header_line.index("Obs. count")
        share_index = header_line.index("Obs. share")
        p_value_index = header_line.index("Split p-value")
        leaf_index_index = header_line.index("Leaf index")
        self.assertLess(predicted_index, count_index)
        self.assertLess(count_index, share_index)
        self.assertLess(share_index, p_value_index)
        self.assertLess(p_value_index, leaf_index_index)

    def test_export_text_leaf_index_cell_matches_chart_one_based_value(
        self,
    ) -> None:
        """Each leaf row shows leaf_id + 1 in the Leaf index column."""
        regression_tree = _fit_three_step_regression_tree()
        result = sigma.export_text(regression_tree)
        lines = result.splitlines()
        header_line = lines[0]
        column_start = header_line.index("Leaf index")
        column_width = len("Leaf index")
        expected_values = {
            leaf.extension.leaf_id + 1 for leaf in regression_tree.leaves_
        }
        observed_values: set[int] = set()
        for line in lines[2:]:
            cell = line[column_start : column_start + column_width].strip()
            if not cell:
                continue
            observed_values.add(int(cell))
        self.assertEqual(observed_values, expected_values)

    def test_export_text_split_rows_have_empty_leaf_index_cell(self):
        """Root and split-node rows leave the Leaf index column whitespace."""
        regression_tree = _fit_step_regression_tree()
        result = sigma.export_text(regression_tree)
        lines = result.splitlines()
        header_line = lines[0]
        column_start = header_line.index("Leaf index")
        column_width = len("Leaf index")
        root_line = next(line for line in lines if "All records" in line)
        root_cell = root_line[column_start : column_start + column_width]
        self.assertEqual(root_cell.strip(), "")

    def test_export_text_truncated_row_has_empty_leaf_index_cell(self):
        """A max_depth-truncated '...' row leaves the Leaf index column blank."""
        regression_tree = _fit_three_step_regression_tree()
        result = sigma.export_text(regression_tree, max_depth=0)
        lines = result.splitlines()
        header_line = lines[0]
        column_start = header_line.index("Leaf index")
        column_width = len("Leaf index")
        truncated_line = next(line for line in lines if "└── ..." in line)
        cell = truncated_line[column_start : column_start + column_width]
        self.assertEqual(cell.strip(), "")

    def test_export_text_dashed_separator_line(self):
        """The second output line carries dashed underlines for named headers."""
        regression_tree = _fit_step_regression_tree()
        result = sigma.export_text(regression_tree)
        lines = result.splitlines()
        header_line = lines[0]
        separator_line = lines[1]
        predicted_start = header_line.index("Predicted mean")
        predicted_width = len("Predicted mean")
        underline = separator_line[
            predicted_start : predicted_start + predicted_width
        ]
        self.assertEqual(underline, "-" * predicted_width)

    def test_export_text_column_alignment_between_header_and_data(self):
        """Header column starts and corresponding data cells share a column."""
        regression_tree = _fit_step_regression_tree()
        result = sigma.export_text(regression_tree)
        lines = result.splitlines()
        header_line = lines[0]
        share_start = header_line.index("Obs. share")
        data_lines = [line for line in lines[2:] if line.strip()]
        self.assertGreater(len(data_lines), 0)
        for data_line in data_lines:
            cell = data_line[share_start : share_start + len("Obs. share")]
            self.assertTrue(
                cell.startswith(" ") or cell[0].isdigit(),
                f"unexpected first character in share cell: {cell!r}",
            )

    def test_export_text_leaf_has_empty_split_p_value_cell(self):
        """Leaf rows have whitespace at the Split p-value column position."""
        regression_tree = _fit_step_regression_tree()
        result = sigma.export_text(regression_tree)
        lines = result.splitlines()
        header_line = lines[0]
        p_value_start = header_line.index("Split p-value")
        leaf_lines = [line for line in lines if "└──" in line or "├──" in line]
        self.assertGreater(len(leaf_lines), 0)
        for leaf_line in leaf_lines:
            cell = leaf_line[
                p_value_start : p_value_start + len("Split p-value")
            ]
            self.assertEqual(cell.strip(), "")

    def test_export_text_decoration_appears_between_p_value_and_leaf_index(
        self,
    ):
        """A node decoration is rendered as an unnamed column left of the leaf index."""
        rng = numpy.random.RandomState(0)
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 10.0) + 0.0 * rng.randn(40)
        decorated_tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            decorator=lambda *args: "stratum=A",
        )
        decorated_tree.fit(X, y)
        result = sigma.export_text(decorated_tree)
        lines = result.splitlines()
        header_line = lines[0]
        self.assertNotIn("stratum=A", header_line)
        self.assertNotIn("Decoration", header_line)
        p_value_end = header_line.index("Split p-value") + len("Split p-value")
        leaf_index_start = header_line.index("Leaf index")
        data_lines = [line for line in lines[2:] if line.strip()]
        self.assertGreater(len(data_lines), 0)
        for data_line in data_lines:
            decoration_slice = data_line[p_value_end:leaf_index_start]
            self.assertEqual(decoration_slice.strip(), "stratum=A")


class TestExportTextBoolean(unittest.TestCase):
    """Tests for the rendering of BooleanPartition splits in to_text."""

    __slots__ = ()

    def test_to_text_renders_boolean_split_as_is_true_is_false(self):
        """A BOOLEAN split appears as '<feature> is true' / '<feature> is false'."""
        regression_tree = _fit_boolean_regression_tree()
        output = regression_tree.to_text()
        self.assertIn("flag is true", output)
        self.assertIn("flag is false", output)

    def test_to_text_boolean_label_carries_no_quotes(self):
        """The boolean rendering does not wrap true/false in double quotes."""
        regression_tree = _fit_boolean_regression_tree()
        output = regression_tree.to_text()
        self.assertNotIn('"true"', output)
        self.assertNotIn('"false"', output)


@unittest.skipUnless(_HAS_GRAPHVIZ, "graphviz not installed")
class TestExportGraphvizBoolean(unittest.TestCase):
    """Tests for the rendering of BooleanPartition splits in export_graphviz."""

    __slots__ = ()

    def test_export_graphviz_renders_boolean_split_labels(self):
        """The DOT source contains the boolean edge labels."""
        regression_tree = _fit_boolean_regression_tree()
        result = sigma.export_graphviz(regression_tree)
        self.assertIn("flag is true", result)
        self.assertIn("flag is false", result)


@unittest.skipUnless(_HAS_GRAPHVIZ, "graphviz not installed")
class TestExportGraphviz(unittest.TestCase):
    """Tests for the module-level export_graphviz free function."""

    __slots__ = ()

    def test_export_graphviz_returns_dot_string_when_out_file_none(self):
        """Returns the DOT source as a Python str when out_file is None."""
        regression_tree = _fit_step_regression_tree()
        result = sigma.export_graphviz(regression_tree)
        self.assertIsInstance(result, str)
        self.assertTrue(result.startswith("digraph"))

    def test_export_graphviz_writes_to_file_path(self):
        """Writes the DOT source to disk when out_file is a string path."""
        regression_tree = _fit_step_regression_tree()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "tree.dot")
            result = sigma.export_graphviz(regression_tree, path)
            self.assertIsNone(result)
            with open(path, encoding="utf-8") as file_handle:
                contents = file_handle.read()
        self.assertTrue(contents.startswith("digraph"))

    def test_export_graphviz_writes_to_file_handle(self):
        """Writes the DOT source to a file-like out_file and returns None."""
        regression_tree = _fit_step_regression_tree()
        buffer = io.StringIO()
        result = sigma.export_graphviz(regression_tree, buffer)
        self.assertIsNone(result)
        self.assertTrue(buffer.getvalue().startswith("digraph"))

    def test_export_graphviz_forwards_feature_names(self):
        """Display-time feature_names appear in the DOT source."""
        regression_tree = _fit_step_regression_tree()
        result = sigma.export_graphviz(
            regression_tree, feature_names=["spread"]
        )
        self.assertIn("spread", result)

    def test_export_graphviz_forwards_category_labels(self):
        """Display-time category_labels are forwarded into the DOT source."""
        regression_tree = _fit_categorical_regression_tree()
        result = sigma.export_graphviz(
            regression_tree,
            feature_names=["color", "noise"],
            category_labels={0: {0.0: "red", 1.0: "blue"}},
        )
        has_label = "red" in result or "blue" in result
        self.assertTrue(has_label)

    def test_export_graphviz_forwards_max_depth(self):
        """A truncation placeholder appears in the DOT source under max_depth."""
        regression_tree = _fit_three_step_regression_tree()
        result = sigma.export_graphviz(regression_tree, max_depth=0)
        self.assertIn("...", result)

    def test_export_graphviz_forwards_precision(self):
        """precision=0 suppresses the decimal point in the formatted threshold."""
        regression_tree = _fit_step_regression_tree()
        result = sigma.export_graphviz(regression_tree, precision=0)
        self.assertIn("<= 20", result)
        self.assertNotIn("<= 20.5", result)

    def test_export_graphviz_raises_on_invalid_max_depth(self):
        """Raises ValueError for negative or non-integer max_depth."""
        regression_tree = _fit_step_regression_tree()
        with self.assertRaises(ValueError):
            sigma.export_graphviz(regression_tree, max_depth=-1)

    def test_export_graphviz_raises_on_invalid_precision(self):
        """Raises ValueError for negative or non-integer precision."""
        regression_tree = _fit_step_regression_tree()
        with self.assertRaises(ValueError):
            sigma.export_graphviz(regression_tree, precision=-1)

    def test_export_graphviz_raises_on_invalid_out_file_type(self):
        """Raises TypeError for an out_file that is neither None, str, nor file-like."""
        regression_tree = _fit_step_regression_tree()
        bad_out_file = typing.cast(typing.IO[str], 123)
        with self.assertRaises(TypeError):
            sigma.export_graphviz(regression_tree, out_file=bad_out_file)

    def test_export_graphviz_raises_not_fitted(self):
        """Raises NotFittedError when called on an unfit estimator."""
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal"
        )
        with self.assertRaises(sklearn.exceptions.NotFittedError):
            sigma.export_graphviz(regression_tree)

    def test_export_graphviz_default_orientation_is_top_down(self):
        """Default DOT source sets rankdir=TB (top-down layout)."""
        regression_tree = _fit_step_regression_tree()
        result = sigma.export_graphviz(regression_tree)
        self.assertIn("rankdir=TB", result)

    def test_export_graphviz_orientation_left_to_right(self):
        """orientation='left-to-right' sets rankdir=LR in the DOT source."""
        regression_tree = _fit_step_regression_tree()
        result = sigma.export_graphviz(
            regression_tree, orientation="left-to-right"
        )
        self.assertIn("rankdir=LR", result)

    def test_export_graphviz_reverse_order_inverts_child_emission(self):
        """A reverse-ordered tree flips the order of the two child edges."""
        natural_tree = _fit_step_regression_tree()
        reversed_tree = _fit_step_regression_tree(reverse_order=True)
        natural_root = natural_tree.content_
        natural_partition = natural_root.extension
        assert isinstance(natural_partition, sigma.Partition)
        natural_left = natural_partition.left
        natural_right = natural_partition.right
        if natural_left.prediction <= natural_right.prediction:
            natural_smaller_child = natural_left
            natural_larger_child = natural_right
        else:
            natural_smaller_child = natural_right
            natural_larger_child = natural_left
        natural_root_id = str(id(natural_root))
        natural_smaller_edge = (
            f"{natural_root_id} -> {id(natural_smaller_child)}"
        )
        natural_larger_edge = f"{natural_root_id} -> {id(natural_larger_child)}"
        reversed_root = reversed_tree.content_
        reversed_partition = reversed_root.extension
        assert isinstance(reversed_partition, sigma.Partition)
        reversed_left = reversed_partition.left
        reversed_right = reversed_partition.right
        if reversed_left.prediction <= reversed_right.prediction:
            reversed_smaller_child = reversed_left
            reversed_larger_child = reversed_right
        else:
            reversed_smaller_child = reversed_right
            reversed_larger_child = reversed_left
        reversed_root_id = str(id(reversed_root))
        reversed_smaller_edge = (
            f"{reversed_root_id} -> {id(reversed_smaller_child)}"
        )
        reversed_larger_edge = (
            f"{reversed_root_id} -> {id(reversed_larger_child)}"
        )
        natural = sigma.export_graphviz(natural_tree)
        reversed_source = sigma.export_graphviz(reversed_tree)
        natural_smaller_first = natural.index(
            natural_smaller_edge
        ) < natural.index(natural_larger_edge)
        reversed_smaller_first = reversed_source.index(
            reversed_smaller_edge
        ) < reversed_source.index(reversed_larger_edge)
        self.assertNotEqual(natural_smaller_first, reversed_smaller_first)

    def test_export_graphviz_default_dpi_appears_in_dot_graph_attr(self):
        """The default dpi=192 is baked into the DOT graph attribute."""
        regression_tree = _fit_step_regression_tree()
        result = sigma.export_graphviz(regression_tree)
        self.assertIn("dpi=192", result)

    def test_export_graphviz_custom_dpi_appears_in_dot_graph_attr(self):
        """A custom dpi value is baked into the returned DOT graph attribute."""
        regression_tree = _fit_step_regression_tree()
        result = sigma.export_graphviz(regression_tree, dpi=72)
        self.assertIn("dpi=72", result)

    def test_export_graphviz_raises_on_invalid_dpi(self):
        """Raises ValueError when dpi is zero, negative, or not an integer."""
        regression_tree = _fit_step_regression_tree()
        with self.assertRaises(ValueError):
            sigma.export_graphviz(regression_tree, dpi=0)
        with self.assertRaises(ValueError):
            sigma.export_graphviz(regression_tree, dpi=-50)


def _fit_step_classification_tree(reverse_order: bool = False):
    """Fit a binary classification tree on a perfectly separable step function."""
    X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
    y = numpy.where(X.ravel() <= 20, 0.0, 1.0)
    classification_tree = sigma._tree_classification.ClassificationTree(
        correlation="normal",
        min_splits=2,
        min_buckets=1,
        reverse_order=reverse_order,
    )
    classification_tree.fit(X, y)
    return classification_tree


def _fit_step_survival_tree(reverse_order: bool = False):
    """Fit a survival tree on a binary categorical signal with median metric."""
    times = numpy.linspace(1.0, 10.0, 60)
    events = numpy.tile([1.0, 0.0], 30)
    y = numpy.column_stack([times, events])
    X = numpy.column_stack([numpy.repeat([0.0, 1.0], 30)])
    survival_tree = sigma._tree_survival.SurvivalTree(
        categorical_features=[0],
        min_splits=2,
        min_buckets=1,
        reverse_order=reverse_order,
    )
    survival_tree.fit(X, y)
    return survival_tree


def _fit_regression_tree_no_ci():
    """Fit a regression tree with confidence intervals disabled."""
    X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
    y = numpy.where(X.ravel() <= 20, 0.0, 10.0)
    regression_tree = sigma._tree_regression.RegressionTree(
        correlation="normal",
        min_splits=2,
        min_buckets=1,
        ci_coverage=None,
    )
    regression_tree.fit(X, y)
    return regression_tree


def _svg_root_width(svg_bytes: bytes) -> float:
    """Parse the SVG width attribute (in points) from the root <svg> element."""
    import re

    decoded = svg_bytes.decode("utf-8")
    match = re.search(r'<svg[^>]*\swidth="([0-9.]+)pt"', decoded)
    if match is None:
        raise AssertionError(
            f"could not find a points-valued width on the SVG root: "
            f"{decoded[:300]}"
        )
    width = float(match.group(1))
    return width


def _select_single_point_scatters(axes: typing.Any) -> list[typing.Any]:
    """Return PathCollection scatters with a single offset (the mean dots)."""
    import matplotlib.collections

    selected: list[typing.Any] = []
    for collection in axes.collections:
        if not isinstance(collection, matplotlib.collections.PathCollection):
            continue
        offsets = numpy.asarray(collection.get_offsets(), dtype=float)
        if offsets.shape[0] == 1:
            selected.append(collection)
    return selected


def _select_rectangle_bars(axes: typing.Any) -> list[typing.Any]:
    """Return Rectangle patches positioned within the axes' data range."""
    import matplotlib.patches

    bars: list[typing.Any] = []
    x_min, x_max = axes.get_xlim()
    for patch in axes.patches:
        if not isinstance(patch, matplotlib.patches.Rectangle):
            continue
        center = float(patch.get_x()) + float(patch.get_width()) / 2.0
        if x_min <= center <= x_max:
            bars.append(patch)
    return bars


@unittest.skipUnless(_HAS_GRAPHVIZ, "graphviz not installed")
class TestExportImage(unittest.TestCase):
    """Tests for the module-level export_image free function."""

    __slots__ = ()

    def test_export_image_png_returns_bytes(self):
        """PNG format returns bytes that begin with the PNG magic prefix."""
        regression_tree = _fit_step_regression_tree()
        result = sigma.export_image(regression_tree, "png")
        self.assertIsInstance(result, bytes)
        self.assertTrue(result.startswith(b"\x89PNG"))

    def test_export_image_svg_returns_bytes(self):
        """SVG format returns bytes that begin with the SVG/XML header."""
        regression_tree = _fit_step_regression_tree()
        result = sigma.export_image(regression_tree, "svg")
        self.assertIsInstance(result, bytes)
        self.assertTrue(
            result.startswith(b"<?xml") or result.startswith(b"<svg")
        )

    def test_export_image_pdf_returns_bytes(self):
        """PDF format returns bytes that begin with the PDF magic prefix."""
        regression_tree = _fit_step_regression_tree()
        result = sigma.export_image(regression_tree, "pdf")
        self.assertIsInstance(result, bytes)
        self.assertTrue(result.startswith(b"%PDF-"))

    def test_export_image_gif_returns_bytes(self):
        """GIF format returns bytes that begin with the GIF magic prefix."""
        regression_tree = _fit_step_regression_tree()
        result = sigma.export_image(regression_tree, "gif")
        self.assertIsInstance(result, bytes)
        self.assertTrue(result.startswith(b"GIF8"))

    def test_export_image_custom_dpi_propagates_to_png(self):
        """A higher dpi produces a strictly larger PNG byte payload."""
        regression_tree = _fit_step_regression_tree()
        low = sigma.export_image(regression_tree, "png", dpi=72)
        high = sigma.export_image(regression_tree, "png", dpi=600)
        self.assertGreater(len(high), len(low))

    def test_export_image_custom_dpi_propagates_to_gif(self):
        """A higher dpi produces a strictly larger GIF byte payload."""
        regression_tree = _fit_step_regression_tree()
        low = sigma.export_image(regression_tree, "gif", dpi=72)
        high = sigma.export_image(regression_tree, "gif", dpi=600)
        self.assertGreater(len(high), len(low))

    def test_export_image_custom_dpi_propagates_to_svg(self):
        """A higher dpi proportionally enlarges the SVG width attribute."""
        regression_tree = _fit_step_regression_tree()
        low = sigma.export_image(regression_tree, "svg", dpi=72)
        high = sigma.export_image(regression_tree, "svg", dpi=600)
        self.assertGreater(_svg_root_width(high), _svg_root_width(low))

    def test_export_image_custom_dpi_propagates_to_pdf(self):
        """A higher dpi produces a strictly larger PDF byte payload."""
        regression_tree = _fit_step_regression_tree()
        low = sigma.export_image(regression_tree, "pdf", dpi=72)
        high = sigma.export_image(regression_tree, "pdf", dpi=600)
        self.assertGreater(len(high), len(low))

    def test_export_image_writes_to_file_path(self):
        """A string out_file writes the bytes to disk and returns None."""
        regression_tree = _fit_step_regression_tree()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "tree.png")
            result = sigma.export_image(regression_tree, "png", path)
            self.assertIsNone(result)
            with open(path, "rb") as file_handle:
                contents = file_handle.read()
        self.assertTrue(contents.startswith(b"\x89PNG"))

    def test_export_image_writes_to_file_handle(self):
        """A binary file-like out_file is written to and returns None."""
        regression_tree = _fit_step_regression_tree()
        buffer = io.BytesIO()
        result = sigma.export_image(regression_tree, "png", buffer)
        self.assertIsNone(result)
        self.assertTrue(buffer.getvalue().startswith(b"\x89PNG"))

    def test_export_image_raises_on_invalid_format(self):
        """Raises ValueError for an unsupported format string."""
        regression_tree = _fit_step_regression_tree()
        bad_format = typing.cast(typing.Any, "jpeg")
        with self.assertRaises(ValueError):
            sigma.export_image(regression_tree, bad_format)

    def test_export_image_raises_on_invalid_dpi(self):
        """Raises ValueError when dpi is zero, negative, or not an integer."""
        regression_tree = _fit_step_regression_tree()
        with self.assertRaises(ValueError):
            sigma.export_image(regression_tree, "png", dpi=0)
        with self.assertRaises(ValueError):
            sigma.export_image(regression_tree, "png", dpi=-50)

    def test_export_image_raises_on_invalid_out_file_type(self):
        """Raises TypeError for an out_file that is neither None, str, nor file-like."""
        regression_tree = _fit_step_regression_tree()
        bad_out_file = typing.cast(typing.IO[bytes], 123)
        with self.assertRaises(TypeError):
            sigma.export_image(regression_tree, "png", bad_out_file)

    def test_export_image_raises_not_fitted(self):
        """Raises NotFittedError when called on an unfit estimator."""
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal"
        )
        with self.assertRaises(sklearn.exceptions.NotFittedError):
            sigma.export_image(regression_tree, "svg")

    def test_export_image_forwards_feature_names(self):
        """Display-time feature_names appear inside the rendered SVG."""
        regression_tree = _fit_step_regression_tree()
        result = sigma.export_image(
            regression_tree, "svg", feature_names=["spread"]
        )
        self.assertIn(b"spread", result)

    def test_export_image_forwards_category_labels(self):
        """Display-time category_labels appear inside the rendered SVG."""
        regression_tree = _fit_categorical_regression_tree()
        result = sigma.export_image(
            regression_tree,
            "svg",
            feature_names=["color", "noise"],
            category_labels={0: {0.0: "red", 1.0: "blue"}},
        )
        has_label = b"red" in result or b"blue" in result
        self.assertTrue(has_label)

    def test_export_image_forwards_max_depth_and_precision(self):
        """max_depth and precision threading affect the SVG output."""
        regression_tree = _fit_three_step_regression_tree()
        truncated = sigma.export_image(regression_tree, "svg", max_depth=0)
        full = sigma.export_image(regression_tree, "svg")
        self.assertNotEqual(truncated, full)
        coarse = sigma.export_image(regression_tree, "svg", precision=0)
        precise = sigma.export_image(regression_tree, "svg", precision=5)
        self.assertNotEqual(coarse, precise)

    def test_export_image_reverse_order(self):
        """A reverse-ordered tree changes the rendered SVG output."""
        natural_tree = _fit_step_regression_tree()
        reversed_tree = _fit_step_regression_tree(reverse_order=True)
        natural = sigma.export_image(natural_tree, "svg")
        reversed_svg = sigma.export_image(reversed_tree, "svg")
        self.assertNotEqual(natural, reversed_svg)


@unittest.skipUnless(_HAS_GRAPHVIZ, "graphviz not installed")
class TestToText(unittest.TestCase):
    """Tests for the Tree.to_text out_file dispatch."""

    __slots__ = ()

    def test_to_text_returns_string_by_default(self):
        """With no out_file, returns the rendered text as a Python str."""
        regression_tree = _fit_step_regression_tree()
        result = regression_tree.to_text()
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_to_text_writes_to_file_path(self):
        """A string out_file writes the text to disk and returns None."""
        regression_tree = _fit_step_regression_tree()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "tree.txt")
            result = regression_tree.to_text(path)
            self.assertIsNone(result)
            with open(path, encoding="utf-8") as file_handle:
                contents = file_handle.read()
        expected = sigma.export_text(regression_tree)
        self.assertEqual(contents, expected)

    def test_to_text_writes_to_file_handle(self):
        """A text file-like out_file is written to and returns None."""
        regression_tree = _fit_step_regression_tree()
        buffer = io.StringIO()
        result = regression_tree.to_text(buffer)
        self.assertIsNone(result)
        expected = sigma.export_text(regression_tree)
        self.assertEqual(buffer.getvalue(), expected)

    def test_to_text_raises_on_invalid_out_file_type(self):
        """Raises TypeError for an out_file that is neither None, str, nor file-like."""
        regression_tree = _fit_step_regression_tree()
        bad_out_file = typing.cast(typing.IO[str], 123)
        with self.assertRaises(TypeError):
            regression_tree.to_text(bad_out_file)


@unittest.skipUnless(_HAS_GRAPHVIZ, "graphviz not installed")
class TestToImage(unittest.TestCase):
    """Tests for the Tree.to_image out_file dispatch and dpi propagation."""

    __slots__ = ()

    def test_to_image_returns_bytes_by_default(self):
        """With no out_file, returns the rendered bytes."""
        regression_tree = _fit_step_regression_tree()
        result = regression_tree.to_image("png")
        self.assertIsInstance(result, bytes)
        self.assertTrue(result.startswith(b"\x89PNG"))

    def test_to_image_writes_png_to_file_path(self):
        """A string out_file writes the PNG to disk and returns None."""
        regression_tree = _fit_step_regression_tree()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "tree.png")
            result = regression_tree.to_image("png", path)
            self.assertIsNone(result)
            with open(path, "rb") as file_handle:
                contents = file_handle.read()
        self.assertTrue(contents.startswith(b"\x89PNG"))

    def test_to_image_writes_png_to_file_handle(self):
        """A binary file-like out_file is written to and returns None."""
        regression_tree = _fit_step_regression_tree()
        buffer = io.BytesIO()
        result = regression_tree.to_image("png", buffer)
        self.assertIsNone(result)
        self.assertTrue(buffer.getvalue().startswith(b"\x89PNG"))

    def test_to_image_custom_dpi_propagates_to_png(self):
        """A higher dpi via the shortcut method produces a larger PNG."""
        regression_tree = _fit_step_regression_tree()
        low = regression_tree.to_image("png", dpi=72)
        high = regression_tree.to_image("png", dpi=600)
        self.assertGreater(len(high), len(low))

    def test_to_image_custom_dpi_propagates_to_gif(self):
        """A higher dpi via the shortcut method produces a larger GIF."""
        regression_tree = _fit_step_regression_tree()
        low = regression_tree.to_image("gif", dpi=72)
        high = regression_tree.to_image("gif", dpi=600)
        self.assertGreater(len(high), len(low))

    def test_to_image_custom_dpi_propagates_to_svg(self):
        """A higher dpi via the shortcut method enlarges the SVG width."""
        regression_tree = _fit_step_regression_tree()
        low = regression_tree.to_image("svg", dpi=72)
        high = regression_tree.to_image("svg", dpi=600)
        self.assertGreater(_svg_root_width(high), _svg_root_width(low))

    def test_to_image_custom_dpi_propagates_to_pdf(self):
        """A higher dpi via the shortcut method produces a larger PDF."""
        regression_tree = _fit_step_regression_tree()
        low = regression_tree.to_image("pdf", dpi=72)
        high = regression_tree.to_image("pdf", dpi=600)
        self.assertGreater(len(high), len(low))

    def test_to_image_raises_on_invalid_format(self):
        """Raises ValueError for an unsupported format string (including 'dot')."""
        regression_tree = _fit_step_regression_tree()
        bad_format = typing.cast(typing.Any, "dot")
        with self.assertRaises(ValueError):
            regression_tree.to_image(bad_format)

    def test_to_image_raises_on_invalid_out_file_type(self):
        """Raises TypeError for an out_file that is neither None, str, nor file-like."""
        regression_tree = _fit_step_regression_tree()
        bad_out_file = typing.cast(typing.IO[bytes], 123)
        with self.assertRaises(TypeError):
            regression_tree.to_image("png", bad_out_file)


class TestReprMimebundle(unittest.TestCase):
    """Tests for the IPython rich-display protocol on Tree."""

    __slots__ = ()

    def test_fitted_tree_returns_svg_in_bundle(self):
        """A fitted tree's mime bundle exposes image/svg+xml as a string."""
        regression_tree = _fit_step_regression_tree()
        bundle = regression_tree._repr_mimebundle_()
        self.assertIn("image/svg+xml", bundle)
        self.assertIsInstance(bundle["image/svg+xml"], str)

    def test_fitted_tree_svg_is_well_formed(self):
        """The bundled SVG payload contains an <svg ...> root element."""
        regression_tree = _fit_step_regression_tree()
        bundle = regression_tree._repr_mimebundle_()
        svg = bundle["image/svg+xml"]
        self.assertIn("<svg", svg)

    def test_fitted_tree_bundle_includes_text_plain(self):
        """The fitted-tree bundle still carries a text/plain fallback."""
        regression_tree = _fit_step_regression_tree()
        bundle = regression_tree._repr_mimebundle_()
        self.assertIn("text/plain", bundle)
        self.assertIsInstance(bundle["text/plain"], str)

    def test_fitted_tree_bundle_excludes_estimator_widget(self):
        """The fitted-tree bundle does not carry sklearn's text/html widget."""
        regression_tree = _fit_step_regression_tree()
        bundle = regression_tree._repr_mimebundle_()
        self.assertNotIn("text/html", bundle)

    def test_unfitted_tree_falls_back_to_sklearn(self):
        """An unfitted tree falls back to sklearn's hyperparameter widget."""
        regression_tree = sigma._tree_regression.RegressionTree()
        bundle = regression_tree._repr_mimebundle_()
        self.assertNotIn("image/svg+xml", bundle)
        self.assertIn("text/plain", bundle)


@unittest.skipUnless(_HAS_MATPLOTLIB, "matplotlib not installed")
class TestExportImageResponse(unittest.TestCase):
    """Tests for export_image and Tree.to_image with kind='response'."""

    __slots__ = ()

    def test_default_kind_is_tree_when_omitted(self):
        """Omitting kind reproduces the same bytes as kind='tree'."""
        regression_tree = _fit_step_regression_tree()
        if not _HAS_GRAPHVIZ:
            self.skipTest("graphviz not installed")
        default_bytes = sigma.export_image(regression_tree, "svg")
        explicit_bytes = sigma.export_image(regression_tree, "svg", kind="tree")
        self.assertEqual(default_bytes, explicit_bytes)

    def test_response_regression_png_returns_bytes(self):
        """Regression response PNG begins with the PNG magic prefix."""
        regression_tree = _fit_step_regression_tree()
        result = sigma.export_image(
            regression_tree, "png", kind="response", response_name="Y"
        )
        self.assertIsInstance(result, bytes)
        self.assertTrue(result.startswith(b"\x89PNG"))

    def test_response_regression_svg_carries_response_name(self):
        """Regression response SVG embeds response_name in its y-axis label."""
        regression_tree = _fit_step_regression_tree()
        result = sigma.export_image(
            regression_tree,
            "svg",
            kind="response",
            response_name="MyResponseY",
        )
        self.assertIn(b"MyResponseY", result)

    def test_response_regression_svg_includes_leaf_axis_label(self):
        """Regression response SVG carries the 'Leaf number' x-axis label."""
        regression_tree = _fit_step_regression_tree()
        result = sigma.export_image(regression_tree, "svg", kind="response")
        self.assertIn(b"Leaf number", result)

    def test_response_classification_svg_contains_class_names(self):
        """Classification response SVG renders class names with leading caps."""
        classification_tree = _fit_step_classification_tree()
        result = sigma.export_image(
            classification_tree,
            "svg",
            kind="response",
            class_names=["died", "survived"],
        )
        self.assertIn(b"Died", result)
        self.assertIn(b"Survived", result)

    def test_response_survival_svg_contains_legend_and_axis(self):
        """Survival response SVG carries the 'Leaf number' legend and time x-axis."""
        survival_tree = _fit_step_survival_tree()
        result = sigma.export_image(
            survival_tree, "svg", kind="response", response_name="time"
        )
        self.assertIn(b"Leaf number", result)
        self.assertIn(b"Time", result)
        self.assertIn(b"Survival probability", result)

    def test_response_survival_curves_start_at_origin(self):
        """Each leaf survival curve begins at time 0 with survival 100%."""
        if not _HAS_MATPLOTLIB:
            self.skipTest("matplotlib not installed")
        import matplotlib.figure

        from sigma import _response_plot

        survival_tree = _fit_step_survival_tree()
        figure = matplotlib.figure.Figure()
        axes = figure.add_subplot(111)
        _response_plot._plot_survival(
            axes,
            survival_tree,
            response_name="time",
        )
        lines = axes.get_lines()
        self.assertEqual(len(lines), len(survival_tree.leaves_))
        for line in lines:
            xs = numpy.asarray(line.get_xdata(), dtype=float)
            ys = numpy.asarray(line.get_ydata(), dtype=float)
            self.assertEqual(float(xs[0]), 0.0)
            self.assertEqual(float(ys[0]), 100.0)

    def test_response_survival_curves_use_badge_palette(self):
        """Each leaf survival curve takes the badge color of its leaf number."""
        if not _HAS_MATPLOTLIB:
            self.skipTest("matplotlib not installed")
        import matplotlib.colors
        import matplotlib.figure

        from sigma import _palette
        from sigma import _response_plot

        survival_tree = _fit_step_survival_tree()
        figure = matplotlib.figure.Figure()
        axes = figure.add_subplot(111)
        _response_plot._plot_survival(
            axes,
            survival_tree,
            response_name="time",
        )
        lines = axes.get_lines()
        n_leaves = len(survival_tree.leaves_)
        for index, line in enumerate(lines):
            expected = matplotlib.colors.to_hex(
                _palette._leaf_color(index, n_leaves)
            )
            actual = matplotlib.colors.to_hex(line.get_color())
            self.assertEqual(actual.lower(), expected.lower())

    def test_response_survival_curves_have_endpoint_dots(self):
        """Each survival curve has start and end marker dots aligned with the curve."""
        if not _HAS_MATPLOTLIB:
            self.skipTest("matplotlib not installed")
        import matplotlib.collections
        import matplotlib.figure

        from sigma import _response_plot

        survival_tree = _fit_step_survival_tree()
        figure = matplotlib.figure.Figure()
        axes = figure.add_subplot(111)
        _response_plot._plot_survival(
            axes,
            survival_tree,
            response_name="time",
        )
        lines = axes.get_lines()
        scatter_collections = [
            collection
            for collection in axes.collections
            if isinstance(collection, matplotlib.collections.PathCollection)
        ]
        n_leaves = len(survival_tree.leaves_)
        self.assertEqual(len(scatter_collections), n_leaves)
        for line, collection in zip(lines, scatter_collections):
            offsets = numpy.asarray(collection.get_offsets(), dtype=float)
            line_xs = numpy.asarray(line.get_xdata(), dtype=float)
            line_ys = numpy.asarray(line.get_ydata(), dtype=float)
            self.assertEqual(offsets.shape, (2, 2))
            self.assertEqual(float(offsets[0][0]), float(line_xs[0]))
            self.assertEqual(float(offsets[0][1]), float(line_ys[0]))
            self.assertEqual(float(offsets[1][0]), float(line_xs[-1]))
            self.assertEqual(float(offsets[1][1]), float(line_ys[-1]))

    def test_response_survival_band_drawn_when_ci_coverage_set(self):
        """One CI band PolyCollection is drawn per leaf when ci_coverage is set."""
        if not _HAS_MATPLOTLIB:
            self.skipTest("matplotlib not installed")
        import matplotlib.collections
        import matplotlib.figure

        from sigma import _response_plot

        survival_tree = _fit_step_survival_tree()
        figure = matplotlib.figure.Figure()
        axes = figure.add_subplot(111)
        _response_plot._plot_survival(
            axes,
            survival_tree,
            response_name="time",
        )
        bands = [
            collection
            for collection in axes.collections
            if isinstance(collection, matplotlib.collections.PolyCollection)
        ]
        self.assertEqual(len(bands), len(survival_tree.leaves_))

    def test_response_survival_band_skipped_when_ci_coverage_none(self):
        """No band is drawn when the tree was fitted with ci_coverage=None."""
        if not _HAS_MATPLOTLIB:
            self.skipTest("matplotlib not installed")
        import matplotlib.collections
        import matplotlib.figure

        from sigma import _response_plot

        times = numpy.linspace(1.0, 10.0, 60)
        events = numpy.tile([1.0, 0.0], 30)
        y = numpy.column_stack([times, events])
        X = numpy.column_stack([numpy.repeat([0.0, 1.0], 30)])
        survival_tree = sigma._tree_survival.SurvivalTree(
            categorical_features=[0],
            min_splits=2,
            min_buckets=1,
            ci_coverage=None,
        )
        survival_tree.fit(X, y)
        figure = matplotlib.figure.Figure()
        axes = figure.add_subplot(111)
        _response_plot._plot_survival(
            axes,
            survival_tree,
            response_name="time",
        )
        bands = [
            collection
            for collection in axes.collections
            if isinstance(collection, matplotlib.collections.PolyCollection)
        ]
        self.assertEqual(bands, [])

    def test_response_survival_band_color_matches_curve(self):
        """Each leaf's CI band uses the same hue as its curve, at alpha=0.15."""
        if not _HAS_MATPLOTLIB:
            self.skipTest("matplotlib not installed")
        import matplotlib.collections
        import matplotlib.colors
        import matplotlib.figure

        from sigma import _palette
        from sigma import _response_plot

        survival_tree = _fit_step_survival_tree()
        figure = matplotlib.figure.Figure()
        axes = figure.add_subplot(111)
        _response_plot._plot_survival(
            axes,
            survival_tree,
            response_name="time",
        )
        bands = [
            collection
            for collection in axes.collections
            if isinstance(collection, matplotlib.collections.PolyCollection)
        ]
        n_leaves = len(survival_tree.leaves_)
        for index, band in enumerate(bands):
            facecolor = numpy.asarray(band.get_facecolor(), dtype=float)[0]
            expected_rgb = matplotlib.colors.to_rgb(
                _palette._leaf_color(index, n_leaves)
            )
            for axis in range(3):
                self.assertAlmostEqual(
                    float(facecolor[axis]),
                    float(expected_rgb[axis]),
                    places=6,
                )
            self.assertAlmostEqual(float(facecolor[3]), 0.15, places=6)

    def test_response_survival_legend_lists_leaves_in_reverse(self):
        """Survival legend lists leaf numbers from N down to 1, top to bottom."""
        if not _HAS_MATPLOTLIB:
            self.skipTest("matplotlib not installed")
        import matplotlib.figure

        from sigma import _response_plot

        survival_tree = _fit_step_survival_tree()
        figure = matplotlib.figure.Figure()
        axes = figure.add_subplot(111)
        _response_plot._plot_survival(
            axes,
            survival_tree,
            response_name="time",
        )
        legend = axes.get_legend()
        if legend is None:
            raise AssertionError(
                "expected a legend on the survival response plot"
            )
        labels = [text.get_text() for text in legend.get_texts()]
        n_leaves = len(survival_tree.leaves_)
        expected = [str(number) for number in range(n_leaves, 0, -1)]
        self.assertEqual(labels, expected)

    def test_response_classification_legend_capitalizes_class_names(self):
        """Classification legend renders user-supplied class names capitalized."""
        if not _HAS_MATPLOTLIB:
            self.skipTest("matplotlib not installed")
        import matplotlib.figure

        from sigma import _response_plot

        classification_tree = _fit_step_classification_tree()
        figure = matplotlib.figure.Figure()
        axes = figure.add_subplot(111)
        _response_plot._plot_classification(
            axes,
            classification_tree,
            class_names=["died", "survived"],
        )
        legend = axes.get_legend()
        if legend is None:
            raise AssertionError(
                "expected a legend on the classification response plot"
            )
        labels = [text.get_text() for text in legend.get_texts()]
        self.assertIn("Died", labels)
        self.assertIn("Survived", labels)
        self.assertNotIn("died", labels)
        self.assertNotIn("survived", labels)

    def test_response_regression_y_label_uses_response_name(self):
        """Regression y-axis label reads '<Response_name> mean'."""
        if not _HAS_MATPLOTLIB:
            self.skipTest("matplotlib not installed")
        import matplotlib.figure

        from sigma import _response_plot

        regression_tree = _fit_step_regression_tree()
        figure = matplotlib.figure.Figure()
        axes = figure.add_subplot(111)
        _response_plot._plot_regression(
            axes,
            regression_tree,
            response_name="my_response",
        )
        self.assertEqual(axes.get_ylabel(), "My_response mean")

    def test_response_survival_x_label_capitalizes_response_name(self):
        """Survival x-axis label upper-cases the user-supplied response_name."""
        if not _HAS_MATPLOTLIB:
            self.skipTest("matplotlib not installed")
        import matplotlib.figure

        from sigma import _response_plot

        survival_tree = _fit_step_survival_tree()
        figure = matplotlib.figure.Figure()
        axes = figure.add_subplot(111)
        _response_plot._plot_survival(
            axes,
            survival_tree,
            response_name="time",
        )
        self.assertEqual(axes.get_xlabel(), "Time")

    def test_response_regression_uses_leaf_palette_colors(self):
        """Each regression mean dot inherits its leaf's badge palette color."""
        if not _HAS_MATPLOTLIB:
            self.skipTest("matplotlib not installed")
        import matplotlib.colors
        import matplotlib.figure

        from sigma import _palette
        from sigma import _response_plot

        regression_tree = _fit_step_regression_tree()
        figure = matplotlib.figure.Figure()
        axes = figure.add_subplot(111)
        _response_plot._plot_regression(
            axes,
            regression_tree,
            response_name=None,
        )
        leaves = regression_tree.leaves_
        n_leaves = len(leaves)
        mean_dot_collections = _select_single_point_scatters(axes)
        self.assertEqual(len(mean_dot_collections), n_leaves)
        for index, collection in enumerate(mean_dot_collections):
            expected = matplotlib.colors.to_hex(
                _palette._leaf_color(index, n_leaves)
            )
            face = numpy.asarray(collection.get_facecolor(), dtype=float)
            actual = matplotlib.colors.to_hex(tuple(face[0].tolist()))
            self.assertEqual(actual.lower(), expected.lower())

    def test_response_regression_reverse_flips_predictions_per_x_position(self):
        """A reverse-ordered tree mirrors the predictions plotted at each x."""
        if not _HAS_MATPLOTLIB:
            self.skipTest("matplotlib not installed")
        import matplotlib.figure

        from sigma import _response_plot

        natural_tree = _fit_step_regression_tree()
        reversed_tree = _fit_step_regression_tree(reverse_order=True)
        natural_figure = matplotlib.figure.Figure()
        natural_axes = natural_figure.add_subplot(111)
        _response_plot._plot_regression(
            natural_axes, natural_tree, response_name=None
        )
        reversed_figure = matplotlib.figure.Figure()
        reversed_axes = reversed_figure.add_subplot(111)
        _response_plot._plot_regression(
            reversed_axes, reversed_tree, response_name=None
        )
        natural_dots = _select_single_point_scatters(natural_axes)
        reversed_dots = _select_single_point_scatters(reversed_axes)
        natural_predictions = [
            float(numpy.asarray(dot.get_offsets(), dtype=float)[0][1])
            for dot in natural_dots
        ]
        reversed_predictions = [
            float(numpy.asarray(dot.get_offsets(), dtype=float)[0][1])
            for dot in reversed_dots
        ]
        self.assertEqual(
            reversed_predictions, list(reversed(natural_predictions))
        )

    def test_response_classification_reverse_flips_legend_order(self):
        """A reverse-ordered tree reverses the classification legend label list."""
        if not _HAS_MATPLOTLIB:
            self.skipTest("matplotlib not installed")
        import matplotlib.figure

        from sigma import _response_plot

        natural_tree = _fit_step_classification_tree()
        reversed_tree = _fit_step_classification_tree(reverse_order=True)
        figure_default = matplotlib.figure.Figure()
        axes_default = figure_default.add_subplot(111)
        _response_plot._plot_classification(
            axes_default,
            natural_tree,
            class_names=["died", "survived"],
        )
        figure_reversed = matplotlib.figure.Figure()
        axes_reversed = figure_reversed.add_subplot(111)
        _response_plot._plot_classification(
            axes_reversed,
            reversed_tree,
            class_names=["died", "survived"],
        )
        default_legend = axes_default.get_legend()
        reversed_legend = axes_reversed.get_legend()
        if default_legend is None or reversed_legend is None:
            raise AssertionError("expected a legend on both response plots")
        default_labels = [
            text.get_text() for text in default_legend.get_texts()
        ]
        reversed_labels = [
            text.get_text() for text in reversed_legend.get_texts()
        ]
        self.assertEqual(reversed_labels, list(reversed(default_labels)))
        self.assertNotEqual(default_labels[0], default_labels[-1])

    def test_response_classification_uses_class_palette_colors(self):
        """Each classification class shares one palette color across its (leaf, class) CI boxes."""
        if not _HAS_MATPLOTLIB:
            self.skipTest("matplotlib not installed")
        import matplotlib.colors
        import matplotlib.figure

        from sigma import _palette
        from sigma import _response_plot

        classification_tree = _fit_step_classification_tree()
        figure = matplotlib.figure.Figure()
        axes = figure.add_subplot(111)
        _response_plot._plot_classification(
            axes,
            classification_tree,
            class_names=["died", "survived"],
        )
        n_leaves = len(classification_tree.leaves_)
        n_classes = int(classification_tree.n_classes_)
        bars = _select_rectangle_bars(axes)
        self.assertEqual(len(bars), n_leaves * n_classes)
        for slot_idx in range(n_classes):
            class_bars = bars[slot_idx * n_leaves : (slot_idx + 1) * n_leaves]
            expected = matplotlib.colors.to_hex(
                _palette._leaf_color(slot_idx, n_classes)
            )
            for bar in class_bars:
                actual = matplotlib.colors.to_hex(bar.get_facecolor())
                self.assertEqual(actual.lower(), expected.lower())

    def test_response_classification_renders_per_class_connector_lines(self):
        """A per-class XY line connects the proportion dots of that class across leaves."""
        if not _HAS_MATPLOTLIB:
            self.skipTest("matplotlib not installed")
        import matplotlib.colors
        import matplotlib.figure

        from sigma import _palette
        from sigma import _response_plot

        classification_tree = _fit_step_classification_tree()
        figure = matplotlib.figure.Figure()
        axes = figure.add_subplot(111)
        _response_plot._plot_classification(
            axes,
            classification_tree,
            class_names=["died", "survived"],
        )
        n_classes = int(classification_tree.n_classes_)
        n_leaves = len(classification_tree.leaves_)
        lines = axes.get_lines()
        self.assertEqual(len(lines), n_classes)
        for slot_idx, line in enumerate(lines):
            expected = matplotlib.colors.to_hex(
                _palette._leaf_color(slot_idx, n_classes)
            )
            actual = matplotlib.colors.to_hex(line.get_color())
            self.assertEqual(actual.lower(), expected.lower())
            self.assertEqual(numpy.asarray(line.get_xdata()).size, n_leaves)

    def test_response_classification_renders_per_class_ci_boxes(self):
        """Classification renders one CI box per (leaf, class), or none when ci_coverage is disabled."""
        if not _HAS_MATPLOTLIB:
            self.skipTest("matplotlib not installed")
        import matplotlib.figure

        from sigma import _response_plot

        classification_tree = _fit_step_classification_tree()
        figure_with_ci = matplotlib.figure.Figure()
        axes_with_ci = figure_with_ci.add_subplot(111)
        _response_plot._plot_classification(
            axes_with_ci,
            classification_tree,
            class_names=["died", "survived"],
        )
        n_leaves = len(classification_tree.leaves_)
        n_classes = int(classification_tree.n_classes_)
        bars_with_ci = _select_rectangle_bars(axes_with_ci)
        self.assertEqual(len(bars_with_ci), n_leaves * n_classes)
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, 0.0, 1.0)
        no_ci_tree = sigma._tree_classification.ClassificationTree(
            correlation="normal",
            min_splits=2,
            min_buckets=1,
            ci_coverage=None,
        )
        no_ci_tree.fit(X, y)
        figure_no_ci = matplotlib.figure.Figure()
        axes_no_ci = figure_no_ci.add_subplot(111)
        _response_plot._plot_classification(
            axes_no_ci,
            no_ci_tree,
            class_names=["died", "survived"],
        )
        bars_no_ci = _select_rectangle_bars(axes_no_ci)
        self.assertEqual(len(bars_no_ci), 0)

    def test_response_survival_reverse_flips_curve_data_per_position(self):
        """A reverse-ordered tree mirrors the survival curve data per x-position."""
        if not _HAS_MATPLOTLIB:
            self.skipTest("matplotlib not installed")
        import matplotlib.colors
        import matplotlib.figure

        from sigma import _response_plot

        natural_tree = _fit_step_survival_tree()
        reversed_tree = _fit_step_survival_tree(reverse_order=True)
        natural_figure = matplotlib.figure.Figure()
        natural_axes = natural_figure.add_subplot(111)
        _response_plot._plot_survival(
            natural_axes, natural_tree, response_name=None
        )
        reversed_figure = matplotlib.figure.Figure()
        reversed_axes = reversed_figure.add_subplot(111)
        _response_plot._plot_survival(
            reversed_axes, reversed_tree, response_name=None
        )
        natural_lines = natural_axes.get_lines()
        reversed_lines = reversed_axes.get_lines()
        natural_endpoints = [
            float(numpy.asarray(line.get_ydata(), dtype=float)[-1])
            for line in natural_lines
        ]
        reversed_endpoints = [
            float(numpy.asarray(line.get_ydata(), dtype=float)[-1])
            for line in reversed_lines
        ]
        self.assertEqual(reversed_endpoints, list(reversed(natural_endpoints)))
        for natural_line, reversed_line in zip(natural_lines, reversed_lines):
            natural_color = matplotlib.colors.to_hex(
                natural_line.get_color()
            ).lower()
            reversed_color = matplotlib.colors.to_hex(
                reversed_line.get_color()
            ).lower()
            self.assertEqual(natural_color, reversed_color)

    def test_response_no_ci_renders_without_error(self):
        """A regression tree fitted without CI still renders a response plot."""
        regression_tree = _fit_regression_tree_no_ci()
        result = sigma.export_image(regression_tree, "png", kind="response")
        self.assertIsInstance(result, bytes)
        self.assertTrue(result.startswith(b"\x89PNG"))

    def test_response_gif_returns_gif_bytes(self):
        """Response GIF output begins with the GIF magic prefix."""
        regression_tree = _fit_step_regression_tree()
        result = sigma.export_image(regression_tree, "gif", kind="response")
        self.assertIsInstance(result, bytes)
        self.assertTrue(result.startswith(b"GIF8"))

    def test_response_pdf_returns_pdf_bytes(self):
        """Response PDF output begins with the PDF magic prefix."""
        regression_tree = _fit_step_regression_tree()
        result = sigma.export_image(regression_tree, "pdf", kind="response")
        self.assertIsInstance(result, bytes)
        self.assertTrue(result.startswith(b"%PDF-"))

    def test_response_writes_to_file_path(self):
        """A string out_file writes the response bytes to disk and returns None."""
        regression_tree = _fit_step_regression_tree()
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "tree_response.png")
            result = sigma.export_image(
                regression_tree, "png", path, kind="response"
            )
            self.assertIsNone(result)
            with open(path, "rb") as file_handle:
                contents = file_handle.read()
        self.assertTrue(contents.startswith(b"\x89PNG"))

    def test_response_writes_to_file_handle(self):
        """A binary file-like out_file is written to and returns None."""
        regression_tree = _fit_step_regression_tree()
        buffer = io.BytesIO()
        result = sigma.export_image(
            regression_tree, "png", buffer, kind="response"
        )
        self.assertIsNone(result)
        self.assertTrue(buffer.getvalue().startswith(b"\x89PNG"))

    def test_response_invalid_kind_raises(self):
        """An unsupported kind value raises ValueError."""
        regression_tree = _fit_step_regression_tree()
        bad_kind = typing.cast(typing.Any, "anything")
        with self.assertRaises(ValueError):
            sigma.export_image(regression_tree, "png", kind=bad_kind)

    def test_to_image_response_matches_export_image_response(self):
        """Tree.to_image(kind='response') equals sigma.export_image(kind='response')."""
        regression_tree = _fit_step_regression_tree()
        from_method = regression_tree.to_image(
            "png", kind="response", response_name="Y"
        )
        from_module = sigma.export_image(
            regression_tree, "png", kind="response", response_name="Y"
        )
        self.assertEqual(from_method, from_module)

    def test_response_custom_dpi_propagates_to_regression_svg(self):
        """A higher dpi proportionally enlarges the regression response SVG width."""
        regression_tree = _fit_step_regression_tree()
        low = sigma.export_image(
            regression_tree, "svg", kind="response", dpi=72
        )
        high = sigma.export_image(
            regression_tree, "svg", kind="response", dpi=600
        )
        self.assertGreater(_svg_root_width(high), _svg_root_width(low))

    def test_response_custom_dpi_propagates_to_classification_svg(self):
        """A higher dpi proportionally enlarges the classification response SVG width."""
        classification_tree = _fit_step_classification_tree()
        low = sigma.export_image(
            classification_tree, "svg", kind="response", dpi=72
        )
        high = sigma.export_image(
            classification_tree, "svg", kind="response", dpi=600
        )
        self.assertGreater(_svg_root_width(high), _svg_root_width(low))

    def test_response_custom_dpi_propagates_to_survival_svg(self):
        """A higher dpi proportionally enlarges the survival response SVG width."""
        survival_tree = _fit_step_survival_tree()
        low = sigma.export_image(survival_tree, "svg", kind="response", dpi=72)
        high = sigma.export_image(
            survival_tree, "svg", kind="response", dpi=600
        )
        self.assertGreater(_svg_root_width(high), _svg_root_width(low))

    def test_response_svg_dpi_scaling_matches_tree_svg(self):
        """Tree and response SVG widths scale by the same dpi/72 factor."""
        if not _HAS_GRAPHVIZ:
            self.skipTest("graphviz not installed")
        regression_tree = _fit_step_regression_tree()
        tree_low = sigma.export_image(regression_tree, "svg", dpi=72)
        tree_high = sigma.export_image(regression_tree, "svg", dpi=600)
        response_low = sigma.export_image(
            regression_tree, "svg", kind="response", dpi=72
        )
        response_high = sigma.export_image(
            regression_tree, "svg", kind="response", dpi=600
        )
        tree_ratio = _svg_root_width(tree_high) / _svg_root_width(tree_low)
        response_ratio = _svg_root_width(response_high) / _svg_root_width(
            response_low
        )
        self.assertAlmostEqual(tree_ratio, response_ratio, delta=0.5)


class TestPublicApiArgumentOrdering(unittest.TestCase):
    """Lock the argument order of every public render entrypoint."""

    def _params(self, function: typing.Callable) -> list[str]:
        """Return the parameter names of function in declaration order."""
        names = list(inspect.signature(function).parameters.keys())
        return names

    def test_export_text_argument_order(self) -> None:
        """sigma.export_text follows the canonical text-render order."""
        expected = [
            "tree",
            "out_file",
            "feature_names",
            "class_names",
            "response_name",
            "category_labels",
            "prediction_formatter",
            "max_depth",
            "precision",
        ]
        self.assertEqual(self._params(sigma.export_text), expected)

    def test_to_text_argument_order(self) -> None:
        """Tree.to_text mirrors export_text minus the leading tree argument."""
        expected = [
            "self",
            "out_file",
            "feature_names",
            "class_names",
            "response_name",
            "category_labels",
            "prediction_formatter",
            "max_depth",
            "precision",
        ]
        self.assertEqual(self._params(sigma._tree.Tree.to_text), expected)

    def test_export_graphviz_argument_order(self) -> None:
        """sigma.export_graphviz follows the canonical graphical-render order."""
        expected = [
            "tree",
            "out_file",
            "feature_names",
            "class_names",
            "response_name",
            "category_labels",
            "prediction_formatter",
            "max_depth",
            "precision",
            "orientation",
            "dpi",
            "root_colors",
            "split_colors",
            "leaf_colors",
            "background_color",
        ]
        self.assertEqual(self._params(sigma.export_graphviz), expected)

    def test_export_image_argument_order(self) -> None:
        """sigma.export_image follows the canonical graphical-render order."""
        expected = [
            "tree",
            "format",
            "out_file",
            "kind",
            "feature_names",
            "class_names",
            "response_name",
            "category_labels",
            "prediction_formatter",
            "max_depth",
            "precision",
            "orientation",
            "dpi",
            "root_colors",
            "split_colors",
            "leaf_colors",
            "background_color",
        ]
        self.assertEqual(self._params(sigma.export_image), expected)

    def test_to_image_argument_order(self) -> None:
        """Tree.to_image mirrors export_image minus the leading tree argument."""
        expected = [
            "self",
            "format",
            "out_file",
            "kind",
            "feature_names",
            "class_names",
            "response_name",
            "category_labels",
            "prediction_formatter",
            "max_depth",
            "precision",
            "orientation",
            "dpi",
            "root_colors",
            "split_colors",
            "leaf_colors",
            "background_color",
        ]
        self.assertEqual(self._params(sigma._tree.Tree.to_image), expected)

    def test_export_sql_argument_order(self) -> None:
        """sigma.export_sql follows the canonical SQL-render order."""
        expected = [
            "tree",
            "out_file",
            "target_class",
            "feature_names",
            "category_labels",
            "max_depth",
        ]
        self.assertEqual(self._params(sigma.export_sql), expected)

    def test_to_sql_argument_order(self) -> None:
        """Tree.to_sql mirrors export_sql minus the leading tree argument."""
        expected = [
            "self",
            "out_file",
            "target_class",
            "feature_names",
            "category_labels",
            "max_depth",
        ]
        self.assertEqual(self._params(sigma._tree.Tree.to_sql), expected)


class TestExportSql(unittest.TestCase):
    """Tests for the module-level export_sql free function and Tree.to_sql."""

    __slots__ = ()

    def test_export_sql_returns_string(self):
        """Returns a non-empty Python str matching the to_sql output."""
        regression_tree = _fit_step_regression_tree()
        result = sigma.export_sql(regression_tree)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_export_sql_matches_to_sql_output(self):
        """Returned string equals the result of Tree.to_sql."""
        regression_tree = _fit_step_regression_tree()
        from_method = regression_tree.to_sql()
        result = sigma.export_sql(regression_tree)
        self.assertEqual(result, from_method)

    def test_export_sql_writes_to_string_path(self):
        """Writing to a filesystem path returns None and produces the file."""
        regression_tree = _fit_step_regression_tree()
        expected = sigma.export_sql(regression_tree)
        with tempfile.TemporaryDirectory() as directory:
            file_path = os.path.join(directory, "tree.sql")
            result = sigma.export_sql(regression_tree, file_path)
            self.assertIsNone(result)
            with open(file_path, "r", encoding="utf-8") as file_handle:
                written = file_handle.read()
            self.assertEqual(written, expected)

    def test_export_sql_writes_to_file_like(self):
        """Writing to an io.StringIO returns None and writes the same content."""
        regression_tree = _fit_step_regression_tree()
        expected = sigma.export_sql(regression_tree)
        buffer = io.StringIO()
        result = sigma.export_sql(regression_tree, buffer)
        self.assertIsNone(result)
        self.assertEqual(buffer.getvalue(), expected)

    def test_export_sql_emits_case_not_if(self):
        """The emitted expression uses CASE/WHEN/END and never the IF function."""
        regression_tree = _fit_three_step_regression_tree()
        result = sigma.export_sql(regression_tree)
        self.assertIn("CASE", result)
        self.assertIn("WHEN ", result)
        self.assertIn("END", result)
        self.assertNotRegex(result, r"\bIF\s*\(")
        self.assertNotRegex(result, r"\bIIF\s*\(")

    def test_export_sql_numerical_tree_every_level_has_else_null(self):
        """A numerical-only tree carries ELSE NULL on every internal CASE."""
        regression_tree = _fit_three_step_regression_tree()
        result = sigma.export_sql(regression_tree)
        case_count = result.count("CASE")
        else_null_count = result.count("ELSE NULL")
        self.assertGreater(case_count, 0)
        self.assertEqual(else_null_count, case_count)

    def test_export_sql_categorical_split_else_carries_internal_node_prediction(
        self,
    ):
        """A categorical split's ELSE emits the internal node's prediction."""
        regression_tree = _fit_categorical_regression_tree()
        result = sigma.export_sql(regression_tree)
        root_prediction = regression_tree.content_.prediction
        expected_literal = repr(float(root_prediction))
        self.assertIn(f"ELSE {expected_literal}", result)
        self.assertNotIn("ELSE NULL", result)

    def test_export_sql_numerical_split_emits_both_branches(self):
        """A numerical split renders both <= and > as explicit WHEN clauses."""
        regression_tree = _fit_step_regression_tree()
        result = sigma.export_sql(regression_tree, feature_names=["spread"])
        self.assertIn('WHEN "spread" <= 20 THEN', result)
        self.assertIn('WHEN "spread" > 20 THEN', result)

    def test_export_sql_integer_threshold_has_no_trailing_dot_zero(self):
        """Integer-valued numerical thresholds render as bare ints."""
        regression_tree = _fit_step_regression_tree()
        result = sigma.export_sql(regression_tree)
        self.assertNotIn(" <= 20.0 THEN", result)
        self.assertNotIn(" > 20.0 THEN", result)
        self.assertIn(" <= 20 THEN", result)
        self.assertIn(" > 20 THEN", result)

    def test_export_sql_boolean_split_uses_not_feature(self):
        """Boolean splits emit NOT <feat> / <feat>, never = TRUE / = FALSE."""
        regression_tree = _fit_boolean_regression_tree()
        result = sigma.export_sql(regression_tree)
        self.assertIn('WHEN "flag" THEN', result)
        self.assertIn('WHEN NOT "flag" THEN', result)
        self.assertNotIn("= TRUE", result)
        self.assertNotIn("= FALSE", result)

    def test_export_sql_single_value_categorical_split_emits_equality(self):
        """A categorical split routing one value per side emits = v, not IN (v)."""
        regression_tree = _fit_categorical_regression_tree()
        result = sigma.export_sql(regression_tree)
        self.assertNotIn(" IN (", result)
        self.assertRegex(result, r'WHEN "X\[0\]" = 0\.0 THEN')
        self.assertRegex(result, r'WHEN "X\[0\]" = 1\.0 THEN')

    def test_export_sql_multi_value_categorical_split_keeps_in_clause(self):
        """A categorical split routing multiple values to a side keeps IN (...)."""
        regression_tree = _fit_multi_value_categorical_regression_tree()
        result = sigma.export_sql(regression_tree)
        self.assertIn(" IN (", result)
        self.assertRegex(result, r'WHEN "X\[0\]" IN \([^)]+,[^)]+\) THEN')

    def test_export_sql_mixed_cardinality_split_uses_equality_and_in(self):
        """A 1-vs-N categorical split uses = on the singleton and IN on the N side."""
        regression_tree = _fit_mixed_cardinality_categorical_regression_tree()
        result = sigma.export_sql(regression_tree)
        self.assertRegex(result, r'WHEN "X\[0\]" = [0-9.]+ THEN')
        self.assertRegex(result, r'WHEN "X\[0\]" IN \([^,)]+,[^)]+\) THEN')

    def test_export_sql_feature_names_double_quoted(self):
        """User-supplied feature_names are emitted as double-quoted identifiers."""
        regression_tree = _fit_step_regression_tree()
        result = sigma.export_sql(regression_tree, feature_names=["spread"])
        self.assertIn('"spread"', result)
        self.assertNotIn("'spread'", result)

    def test_export_sql_feature_name_with_spaces_quoted(self):
        """A feature name with a space survives via double-quoting."""
        regression_tree = _fit_step_regression_tree()
        result = sigma.export_sql(
            regression_tree, feature_names=["Spread (bps)"]
        )
        self.assertIn('"Spread (bps)"', result)

    def test_export_sql_reserved_keyword_feature_name(self):
        """A reserved SQL keyword as a feature name is safe via quoting."""
        regression_tree = _fit_step_regression_tree()
        result = sigma.export_sql(regression_tree, feature_names=["select"])
        self.assertIn('"select"', result)

    def test_export_sql_feature_name_with_embedded_double_quote(self):
        """An embedded double quote in a feature name is doubled per SQL-92."""
        regression_tree = _fit_step_regression_tree()
        result = sigma.export_sql(regression_tree, feature_names=['a"b'])
        self.assertIn('"a""b"', result)

    def test_export_sql_categorical_label_single_quoted_with_escape(self):
        """String category labels render single-quoted with ' doubled."""
        regression_tree = _fit_categorical_regression_tree()
        category_labels = {0: {0.0: "O'Brien", 1.0: "Smith"}}
        result = sigma.export_sql(
            regression_tree, category_labels=category_labels
        )
        self.assertIn("'O''Brien'", result)
        self.assertIn("'Smith'", result)

    def test_export_sql_category_labels_render_as_equality(self):
        """When category_labels are provided, singleton sides render = 'label'."""
        regression_tree = _fit_categorical_regression_tree()
        category_labels = {0: {0.0: "low", 1.0: "high"}}
        result = sigma.export_sql(
            regression_tree, category_labels=category_labels
        )
        self.assertIn("= 'low'", result)
        self.assertIn("= 'high'", result)
        self.assertNotIn(" IN (", result)

    def test_export_sql_leaf_comments_numbered_one_indexed(self):
        """Each leaf carries a 1-indexed -- Leaf N comment."""
        regression_tree = _fit_three_step_regression_tree()
        result = sigma.export_sql(regression_tree)
        leaf_count = len(regression_tree.leaves_)
        for leaf_index in range(leaf_count):
            self.assertIn(f"-- Leaf {leaf_index + 1}", result)
        self.assertNotIn(f"-- Leaf {leaf_count + 1}", result)
        self.assertNotIn("-- Leaf 0", result)

    def test_export_sql_each_leaf_on_distinct_line(self):
        """One -- Leaf N comment per non-blank leaf-rendering line."""
        regression_tree = _fit_three_step_regression_tree()
        result = sigma.export_sql(regression_tree)
        leaf_lines = [
            line for line in result.splitlines() if "-- Leaf " in line
        ]
        self.assertEqual(len(leaf_lines), len(regression_tree.leaves_))

    def test_export_sql_branch_order_matches_text_export_default(self):
        """With reverse_order=False, the first WHEN matches the higher branch."""
        regression_tree = _fit_step_regression_tree(reverse_order=False)
        sql = sigma.export_sql(regression_tree)
        case_body = sql.split("\n")
        when_lines = [
            line for line in case_body if line.lstrip().startswith("WHEN ")
        ]
        self.assertGreaterEqual(len(when_lines), 2)
        self.assertIn(" > 20 THEN", when_lines[0])
        self.assertIn(" <= 20 THEN", when_lines[1])

    def test_export_sql_branch_order_inverts_when_reverse_order_true(self):
        """A reverse-ordered tree flips the SQL branch ordering."""
        regression_tree = _fit_step_regression_tree(reverse_order=True)
        sql = sigma.export_sql(regression_tree)
        case_body = sql.split("\n")
        when_lines = [
            line for line in case_body if line.lstrip().startswith("WHEN ")
        ]
        self.assertGreaterEqual(len(when_lines), 2)
        self.assertIn(" <= 20 THEN", when_lines[0])
        self.assertIn(" > 20 THEN", when_lines[1])

    def test_export_sql_max_depth_zero_truncates_to_root(self):
        """max_depth=0 collapses the entire tree to a single root leaf line."""
        regression_tree = _fit_three_step_regression_tree()
        result = sigma.export_sql(regression_tree, max_depth=0)
        self.assertNotIn("CASE", result)
        self.assertNotIn("WHEN ", result)
        self.assertIn("-- Truncated at depth 0", result)

    def test_export_sql_max_depth_one_collapses_descendants(self):
        """max_depth=1 keeps the root split and collapses each child to a leaf."""
        regression_tree = _fit_three_step_regression_tree()
        result = sigma.export_sql(regression_tree, max_depth=1)
        self.assertEqual(result.count("CASE"), 1)
        self.assertIn("-- Truncated at depth 1", result)

    def test_export_sql_raises_not_fitted(self):
        """Raises NotFittedError when called on an unfit estimator."""
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal"
        )
        with self.assertRaises(sklearn.exceptions.NotFittedError):
            sigma.export_sql(regression_tree)

    def test_export_sql_rejects_negative_max_depth(self):
        """Negative max_depth raises ValueError, matching export_text."""
        regression_tree = _fit_step_regression_tree()
        with self.assertRaises(ValueError):
            sigma.export_sql(regression_tree, max_depth=-1)

    def test_export_sql_rejects_invalid_out_file(self):
        """A non-path non-file-like out_file raises TypeError."""
        regression_tree = _fit_step_regression_tree()
        bad_out_file = typing.cast(typing.IO[str], 123)
        with self.assertRaises(TypeError):
            sigma.export_sql(regression_tree, bad_out_file)

    def test_export_sql_classification_default_target_class_last(self):
        """Default target_class=None emits class_distribution[-1] per leaf."""
        classification_tree = _fit_step_classification_tree()
        result = classification_tree.to_sql()
        last_class_index = len(classification_tree.classes_) - 1
        for leaf in classification_tree.leaves_:
            expected = repr(float(leaf.class_distribution[last_class_index]))
            self.assertIn(expected, result)

    def test_export_sql_classification_explicit_target_class(self):
        """target_class=<first class> emits class_distribution[0] per leaf."""
        classification_tree = _fit_step_classification_tree()
        first_class = classification_tree.classes_[0]
        result = classification_tree.to_sql(target_class=first_class)
        for leaf in classification_tree.leaves_:
            expected = repr(float(leaf.class_distribution[0]))
            self.assertIn(expected, result)

    def test_export_sql_classification_string_target_class_resolves(self):
        """A string target_class resolves to its index when classes_ are strings."""
        X = numpy.arange(1, 41, dtype=float).reshape(-1, 1)
        y = numpy.where(X.ravel() <= 20, "low", "high")
        classification_tree = sigma._tree_classification.ClassificationTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        classification_tree.fit(X, y)
        low_index = list(classification_tree.classes_).index("low")
        result = classification_tree.to_sql(target_class="low")
        for leaf in classification_tree.leaves_:
            expected = repr(float(leaf.class_distribution[low_index]))
            self.assertIn(expected, result)

    def test_export_sql_target_class_for_regression_raises(self):
        """target_class on a regression tree raises ValueError."""
        regression_tree = _fit_step_regression_tree()
        with self.assertRaises(ValueError):
            regression_tree.to_sql(target_class="anything")

    def test_export_sql_target_class_for_survival_raises(self):
        """target_class on a survival tree raises ValueError."""
        survival_tree = _fit_step_survival_tree()
        with self.assertRaises(ValueError):
            survival_tree.to_sql(target_class="anything")

    def test_export_sql_target_class_not_in_classes_raises(self):
        """target_class not in tree.classes_ raises ValueError."""
        classification_tree = _fit_step_classification_tree()
        with self.assertRaises(ValueError):
            classification_tree.to_sql(target_class=42.0)

    def test_export_sql_single_leaf_tree_has_no_case(self):
        """A tree with a single leaf renders as a single leaf-value line."""
        constant_x = numpy.zeros((10, 1), dtype=float)
        constant_y = numpy.ones(10, dtype=float)
        single_leaf_tree = sigma._tree_regression.RegressionTree(
            correlation="normal", min_splits=2, min_buckets=1
        )
        single_leaf_tree.fit(constant_x, constant_y)
        result = single_leaf_tree.to_sql()
        self.assertNotIn("CASE", result)
        self.assertNotIn("WHEN ", result)
        self.assertIn("-- Leaf 1", result)

    def test_export_sql_survival_renders_first_metric_value(self):
        """A SurvivalTree emits node.prediction (first metric) at each leaf."""
        survival_tree = _fit_step_survival_tree()
        result = survival_tree.to_sql()
        for leaf in survival_tree.leaves_:
            expected = _format_sql_numeric(leaf.prediction)
            self.assertIn(expected, result)

    def test_export_sql_no_trailing_newline(self):
        """The emitted SQL does not end with a trailing newline."""
        regression_tree = _fit_three_step_regression_tree()
        result = sigma.export_sql(regression_tree)
        self.assertFalse(result.endswith("\n"))

    def test_export_sql_non_finite_leaf_emits_null(self):
        """A non-finite leaf prediction renders as the bare NULL keyword."""
        regression_tree = _fit_step_regression_tree()
        first_leaf = regression_tree.leaves_[0]
        first_leaf.prediction = float("nan")
        result = regression_tree.to_sql()
        self.assertIn("NULL -- Leaf 1", result)

    def test_export_sql_predict_equivalence_via_sqlite(self):
        """Generated SQL evaluated in SQLite bit-exactly matches tree.predict."""
        import sqlite3

        rng = numpy.random.default_rng(0)
        sample_count = 200
        spread = rng.uniform(0.0, 100.0, sample_count)
        flag = rng.choice([False, True], sample_count)
        region = rng.choice(["north", "south", "east", "west"], sample_count)
        y = (
            2.0 * spread
            + 5.0 * flag.astype(float)
            + numpy.where(region == "north", 10.0, 0.0)
        )
        import pandas

        X = pandas.DataFrame({"spread": spread, "flag": flag, "region": region})
        regression_tree = sigma._tree_regression.RegressionTree(
            correlation="normal",
            categorical_features=["region"],
            min_splits=10,
            min_buckets=5,
        )
        regression_tree.fit(X, y)
        expected_predictions = regression_tree.predict(X)
        sql_expression = regression_tree.to_sql()
        connection = sqlite3.connect(":memory:")
        try:
            cursor = connection.cursor()
            cursor.execute(
                "CREATE TABLE points (spread REAL, flag INTEGER, region TEXT)"
            )
            for index in range(sample_count):
                cursor.execute(
                    "INSERT INTO points VALUES (?, ?, ?)",
                    (
                        float(spread[index]),
                        int(flag[index]),
                        str(region[index]),
                    ),
                )
            connection.commit()
            cursor.execute(f"SELECT {sql_expression} FROM points")
            sql_predictions = numpy.array([row[0] for row in cursor.fetchall()])
        finally:
            connection.close()
        self.assertEqual(sql_predictions.shape, expected_predictions.shape)
        match_count = int(numpy.sum(sql_predictions == expected_predictions))
        self.assertEqual(match_count, sample_count)

    def test_export_sql_predict_equivalence_with_unobserved_category_via_sqlite(
        self,
    ):
        """An unobserved categorical code yields the internal node's prediction in SQL."""
        import sqlite3

        regression_tree = _fit_multi_value_categorical_regression_tree()
        unobserved_X = numpy.array([[99.0, 0.0]])
        python_prediction = regression_tree.predict(unobserved_X)[0]
        sql_expression = regression_tree.to_sql(feature_names=["cat", "noise"])
        connection = sqlite3.connect(":memory:")
        try:
            cursor = connection.cursor()
            cursor.execute('CREATE TABLE points ("cat" REAL, "noise" REAL)')
            cursor.execute("INSERT INTO points VALUES (?, ?)", (99.0, 0.0))
            connection.commit()
            cursor.execute(f"SELECT {sql_expression} FROM points")
            sql_prediction = cursor.fetchone()[0]
        finally:
            connection.close()
        self.assertIsNotNone(sql_prediction)
        self.assertEqual(sql_prediction, python_prediction)


def _format_sql_numeric(value: float) -> str:
    """Mirror sigma._tree_sql._format_sql_numeric_literal for assertion lookups."""
    import math

    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, int):
        return repr(value)
    float_value = float(value)
    if not math.isfinite(float_value):
        return "NULL"
    return repr(float_value)


if __name__ == "__main__":
    unittest.main()
