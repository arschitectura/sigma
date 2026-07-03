"""Unit tests for the fit/predict column-type consistency contract.

Every predict input must present each column with the same type kind it had
at fit; only numeric width (int, float32, float64, nullable, null-carrying)
may differ. Plain arrays and lists carry no column types and are therefore
numbers-only, at fit and at predict. The SQL export states the expected
column types in a leading comment.
"""

import sqlite3
import unittest
import warnings

import numpy
import pandas
import polars

import sigma._tree_regression


_N = 20


def _fit(X, y, categorical_features=None):
    """Fit a small deterministic RegressionTree on X and y."""
    tree = sigma._tree_regression.RegressionTree(
        min_splits=2,
        min_buckets=3,
        categorical_features=categorical_features,
        random_state=0,
    )
    tree.fit(X, y)
    return tree


def _predict_quietly(tree, X):
    """Predict while silencing sklearn feature-name warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        predictions = tree.predict(X)
    return predictions


def _numeric_missing_tree():
    """Numeric numpy fit with a missing branch: leaves -10, +10, +5."""
    X = numpy.array([[0.0]] * _N + [[10.0]] * _N + [[numpy.nan]] * _N)
    y = numpy.array([-10.0] * _N + [10.0] * _N + [5.0] * _N)
    tree = _fit(X, y)
    return tree


def _declared_categorical_tree():
    """Numpy fit, codes 10/20/30 plus NaN declared categorical by position."""
    X = numpy.array(
        [[10.0]] * _N + [[20.0]] * _N + [[30.0]] * _N + [[numpy.nan]] * _N
    )
    y = numpy.array([0.0] * _N + [-10.0] * _N + [10.0] * _N + [5.0] * _N)
    tree = _fit(X, y, categorical_features=[0])
    return tree


def _boolean_tree():
    """Pandas plain-bool fit without missing: leaves -10 (False), +10 (True)."""
    X = pandas.DataFrame({"flag": [False] * _N + [True] * _N})
    y = numpy.array([-10.0] * _N + [10.0] * _N)
    tree = _fit(X, y)
    return tree


def _promoted_boolean_tree():
    """Pandas nullable-boolean fit with missing: leaves 0, -10, +10 (missing)."""
    flags = pandas.array(
        [True] * _N + [False] * _N + [None] * _N, dtype="boolean"
    )
    X = pandas.DataFrame({"flag": flags})
    y = numpy.array([0.0] * _N + [-10.0] * _N + [10.0] * _N)
    tree = _fit(X, y)
    return tree


def _polars_promoted_boolean_tree():
    """Polars Boolean fit with nulls: leaves 0, -10, +10 (missing)."""
    values = [True] * _N + [False] * _N + [None] * _N
    series = polars.Series(values, dtype=polars.Boolean)
    X = polars.DataFrame({"flag": series})
    y = numpy.array([0.0] * _N + [-10.0] * _N + [10.0] * _N)
    tree = _fit(X, y)
    return tree


def _categorical_tree():
    """Pandas categorical fit with missing: leaves 0, -10, +10, +5 (missing)."""
    values = ["a"] * _N + ["b"] * _N + ["c"] * _N + [None] * _N
    categories = pandas.Categorical(values, categories=["a", "b", "c"])
    X = pandas.DataFrame({"cat": categories})
    y = numpy.array([0.0] * _N + [-10.0] * _N + [10.0] * _N + [5.0] * _N)
    tree = _fit(X, y)
    return tree


def _polars_categorical_tree():
    """Polars Categorical fit with nulls: leaves 0, -10, +10, +5 (missing)."""
    values = ["a"] * _N + ["b"] * _N + ["c"] * _N + [None] * _N
    series = polars.Series(values, dtype=polars.Categorical)
    X = polars.DataFrame({"cat": series})
    y = numpy.array([0.0] * _N + [-10.0] * _N + [10.0] * _N + [5.0] * _N)
    tree = _fit(X, y)
    return tree


def _two_feature_tree():
    """Two nested numeric splits: root on column 0, children on column 1."""
    x0 = numpy.repeat([0.0, 10.0], 2 * _N)
    inner = numpy.repeat([0.0, 10.0], _N)
    x1 = numpy.tile(inner, 2)
    X = numpy.column_stack([x0, x1])
    y = 20.0 * (x0 > 5.0) + 5.0 * (x1 > 5.0)
    tree = _fit(X, y)
    return tree


class TestNumericWidthEquivalence(unittest.TestCase):
    """Numeric predict input is accepted in any width and container."""

    __slots__ = ()

    def test_list_with_none_predicts_missing_branch(self):
        """A plain list with None routes the missing row to the missing leaf."""
        tree = _numeric_missing_tree()
        predictions = tree.predict([[0.0], [10.0], [None]])
        expected = numpy.array([-10.0, 10.0, 5.0])
        numpy.testing.assert_array_equal(predictions, expected)

    def test_numpy_integer_and_float32_predict_equally(self):
        """Integer and float32 arrays predict the same leaves as float64."""
        tree = _numeric_missing_tree()
        X_integer = numpy.array([[0], [10]])
        X_float32 = numpy.array([[0.0], [10.0]], dtype=numpy.float32)
        predictions_integer = tree.predict(X_integer)
        predictions_float32 = tree.predict(X_float32)
        expected = numpy.array([-10.0, 10.0])
        numpy.testing.assert_array_equal(predictions_integer, expected)
        numpy.testing.assert_array_equal(predictions_float32, expected)

    def test_pandas_nullable_numeric_predicts_missing_branch(self):
        """Pandas nullable Int64 with NA routes NA to the missing leaf."""
        tree = _numeric_missing_tree()
        column = pandas.array([0, 10, None], dtype="Int64")
        X = pandas.DataFrame({"x": column})
        predictions = _predict_quietly(tree, X)
        expected = numpy.array([-10.0, 10.0, 5.0])
        numpy.testing.assert_array_equal(predictions, expected)

    def test_polars_integer_with_null_predicts_missing_branch(self):
        """Polars Int64 with null routes null to the missing leaf."""
        tree = _numeric_missing_tree()
        series = polars.Series([0, 10, None], dtype=polars.Int64)
        X = polars.DataFrame({"x": series})
        predictions = _predict_quietly(tree, X)
        expected = numpy.array([-10.0, 10.0, 5.0])
        numpy.testing.assert_array_equal(predictions, expected)

    def test_declared_categorical_accepts_codes_across_containers(self):
        """Numeric codes in a list or a polars column route to their leaves."""
        tree = _declared_categorical_tree()
        predictions_list = tree.predict([[10.0], [20.0], [30.0], [None]])
        series = polars.Series([10.0, 20.0, 30.0, None], dtype=polars.Float64)
        X_polars = polars.DataFrame({"x": series})
        predictions_polars = _predict_quietly(tree, X_polars)
        expected = numpy.array([0.0, -10.0, 10.0, 5.0])
        numpy.testing.assert_array_equal(predictions_list, expected)
        numpy.testing.assert_array_equal(predictions_polars, expected)

    def test_declared_categorical_unseen_code_falls_back(self):
        """Unseen codes, including the internal missing code, fall back to the node average."""
        tree = _declared_categorical_tree()
        predictions = tree.predict(numpy.array([[99.0], [31.0]]))
        expected = numpy.array([1.25, 1.25])
        numpy.testing.assert_array_equal(predictions, expected)


class TestBareContainerRejection(unittest.TestCase):
    """Plain arrays and lists are numbers-only, at fit and at predict."""

    __slots__ = ()

    def test_fit_rejects_list_of_booleans(self):
        """Fitting on a list of booleans raises with typed-column guidance."""
        X = [[False]] * _N + [[True]] * _N
        y = numpy.array([-10.0] * _N + [10.0] * _N)
        with self.assertRaisesRegex(
            ValueError, "boolean values in a plain array or list"
        ):
            _fit(X, y)

    def test_fit_rejects_numpy_bool_array(self):
        """Fitting on a bool-dtype numpy array raises."""
        X = numpy.array([[False]] * _N + [[True]] * _N)
        y = numpy.array([-10.0] * _N + [10.0] * _N)
        with self.assertRaisesRegex(
            ValueError, "boolean values in a plain array or list"
        ):
            _fit(X, y)

    def test_fit_rejects_object_array_with_booleans(self):
        """Fitting on an object array mixing booleans and None raises."""
        X = [[True], [None]]
        y = numpy.array([0.0, 1.0])
        with self.assertRaisesRegex(
            ValueError, "boolean values in a plain array or list"
        ):
            _fit(X, y)

    def test_numeric_fit_rejects_bool_array_at_predict(self):
        """A bool-dtype numpy array is rejected against a numeric fit."""
        tree = _numeric_missing_tree()
        X = numpy.array([[False], [True]])
        with self.assertRaisesRegex(
            ValueError, "boolean values in a plain array or list"
        ):
            tree.predict(X)

    def test_numeric_fit_rejects_list_of_booleans_at_predict(self):
        """A list of booleans is rejected against a numeric fit."""
        tree = _numeric_missing_tree()
        with self.assertRaisesRegex(
            ValueError, "boolean values in a plain array or list"
        ):
            tree.predict([[False], [True]])

    def test_boolean_fit_rejects_plain_arrays_at_predict(self):
        """A numeric array is rejected when the model has a boolean column."""
        tree = _boolean_tree()
        X = numpy.array([[0.0], [1.0]])
        with self.assertRaisesRegex(
            ValueError, "plain array or list without column types"
        ):
            tree.predict(X)

    def test_promoted_boolean_fit_rejects_plain_arrays_at_predict(self):
        """A numeric array is rejected when the boolean column had missing values at fit."""
        tree = _promoted_boolean_tree()
        X = numpy.array([[0.0], [1.0], [2.0], [numpy.nan]])
        with self.assertRaisesRegex(
            ValueError, "plain array or list without column types"
        ):
            tree.predict(X)

    def test_categorical_fit_rejects_plain_arrays_at_predict(self):
        """A numeric array of internal codes is rejected against a categorical fit."""
        tree = _categorical_tree()
        X = numpy.array([[0.0], [1.0], [2.0], [3.0]])
        with self.assertRaisesRegex(
            ValueError, "plain array or list without column types"
        ):
            tree.predict(X)

    def test_categorical_fit_rejects_bare_strings_at_predict(self):
        """A list of strings is rejected with the plain-array error, not a cast failure."""
        tree = _categorical_tree()
        with self.assertRaisesRegex(
            ValueError, "plain array or list without column types"
        ):
            tree.predict([["a"], ["b"]])

    def test_apply_and_decision_path_reject_plain_arrays(self):
        """apply and decision_path validate exactly like predict."""
        tree = _promoted_boolean_tree()
        X = numpy.array([[0.0], [1.0]])
        with self.assertRaisesRegex(
            ValueError, "plain array or list without column types"
        ):
            tree.apply(X)
        with self.assertRaisesRegex(
            ValueError, "plain array or list without column types"
        ):
            tree.decision_path(X)


class TestColumnTypeMismatch(unittest.TestCase):
    """A DataFrame predict column must keep its fit-time type kind."""

    __slots__ = ()

    def test_numeric_fit_rejects_boolean_column(self):
        """Boolean pandas and polars columns are rejected against a numeric fit."""
        tree = _numeric_missing_tree()
        column = pandas.array([False, True], dtype="boolean")
        X_pandas = pandas.DataFrame({"x": column})
        X_polars = polars.DataFrame({"x": [False, True]})
        with self.assertRaisesRegex(
            ValueError, "was fit as numeric but supplied as boolean"
        ):
            _predict_quietly(tree, X_pandas)
        with self.assertRaisesRegex(
            ValueError, "was fit as numeric but supplied as boolean"
        ):
            _predict_quietly(tree, X_polars)

    def test_declared_categorical_rejects_categorical_column(self):
        """A number-valued Categorical column is rejected against numeric codes."""
        tree = _declared_categorical_tree()
        categories = pandas.Categorical([10.0, 20.0, 30.0])
        X = pandas.DataFrame({"x": categories})
        with self.assertRaisesRegex(
            ValueError, "was fit as numeric but supplied as categorical"
        ):
            _predict_quietly(tree, X)

    def test_boolean_fit_rejects_numeric_column(self):
        """A float column is rejected against a boolean fit."""
        tree = _boolean_tree()
        X = pandas.DataFrame({"flag": [0.0, 1.0]})
        with self.assertRaisesRegex(
            ValueError, "was fit as boolean but supplied as numeric"
        ):
            tree.predict(X)

    def test_boolean_fit_rejects_object_column(self):
        """An object column of booleans is rejected against a boolean fit."""
        tree = _boolean_tree()
        column = pandas.Series([False, True], dtype=object)
        X = pandas.DataFrame({"flag": column})
        with self.assertRaisesRegex(
            ValueError, "was fit as boolean but supplied as unsupported"
        ):
            tree.predict(X)

    def test_promoted_boolean_fit_rejects_numeric_column(self):
        """A float column is rejected when the boolean column had missing values at fit."""
        tree = _promoted_boolean_tree()
        X = pandas.DataFrame({"flag": [0.0, 1.0, 2.0, numpy.nan]})
        with self.assertRaisesRegex(
            ValueError, "was fit as boolean but supplied as numeric"
        ):
            tree.predict(X)

    def test_categorical_fit_rejects_numeric_column(self):
        """Float pandas and polars columns are rejected against a categorical fit."""
        tree = _categorical_tree()
        X_pandas = pandas.DataFrame({"cat": [0.0, 1.0, 2.0, 3.0]})
        series = polars.Series([0.0, 1.0, 2.0, 3.0], dtype=polars.Float64)
        X_polars = polars.DataFrame({"cat": series})
        with self.assertRaisesRegex(
            ValueError, "was fit as categorical but supplied as numeric"
        ):
            tree.predict(X_pandas)
        with self.assertRaisesRegex(
            ValueError, "was fit as categorical but supplied as numeric"
        ):
            tree.predict(X_polars)

    def test_categorical_fit_rejects_object_string_column(self):
        """A plain string column is rejected against a categorical fit."""
        tree = _categorical_tree()
        column = pandas.Series(["a", "b"], dtype=object)
        X = pandas.DataFrame({"cat": column})
        with self.assertRaisesRegex(
            ValueError, "was fit as categorical but supplied as unsupported"
        ):
            tree.predict(X)


class TestMatchingTypesPredictCorrectly(unittest.TestCase):
    """Same-kind predict input keeps working across libraries."""

    __slots__ = ()

    def test_boolean_fit_accepts_boolean_columns_cross_library(self):
        """Plain, nullable, and polars boolean columns predict identically."""
        tree = _boolean_tree()
        nullable = pandas.array([False, True], dtype="boolean")
        X_nullable = pandas.DataFrame({"flag": nullable})
        X_polars = polars.DataFrame({"flag": [False, True]})
        predictions_nullable = tree.predict(X_nullable)
        predictions_polars = tree.predict(X_polars)
        expected = numpy.array([-10.0, 10.0])
        numpy.testing.assert_array_equal(predictions_nullable, expected)
        numpy.testing.assert_array_equal(predictions_polars, expected)

    def test_promoted_boolean_cross_library(self):
        """Boolean columns with missing predict identically across libraries."""
        pandas_tree = _promoted_boolean_tree()
        polars_tree = _polars_promoted_boolean_tree()
        series = polars.Series([True, False, None], dtype=polars.Boolean)
        X_polars = polars.DataFrame({"flag": series})
        nullable = pandas.array([True, False, None], dtype="boolean")
        X_pandas = pandas.DataFrame({"flag": nullable})
        predictions_polars = pandas_tree.predict(X_polars)
        predictions_pandas = polars_tree.predict(X_pandas)
        expected = numpy.array([0.0, -10.0, 10.0])
        numpy.testing.assert_array_equal(predictions_polars, expected)
        numpy.testing.assert_array_equal(predictions_pandas, expected)

    def test_categorical_cross_library_and_category_order(self):
        """Polars Categorical, polars Enum, and reordered pandas categories all route by label."""
        tree = _categorical_tree()
        series = polars.Series(["a", "b", "c", None], dtype=polars.Categorical)
        X_polars = polars.DataFrame({"cat": series})
        enum_dtype = polars.Enum(["a", "b", "c"])
        enum_series = polars.Series(["a", "b", "c", None], dtype=enum_dtype)
        X_enum = polars.DataFrame({"cat": enum_series})
        reordered = pandas.Categorical(
            ["a", "b", "c", None], categories=["c", "b", "a"]
        )
        X_reordered = pandas.DataFrame({"cat": reordered})
        expected = numpy.array([0.0, -10.0, 10.0, 5.0])
        predictions_polars = tree.predict(X_polars)
        predictions_enum = tree.predict(X_enum)
        predictions_reordered = tree.predict(X_reordered)
        numpy.testing.assert_array_equal(predictions_polars, expected)
        numpy.testing.assert_array_equal(predictions_enum, expected)
        numpy.testing.assert_array_equal(predictions_reordered, expected)

    def test_polars_categorical_fit_accepts_pandas_categorical(self):
        """A pandas Categorical predicts correctly against a polars categorical fit."""
        tree = _polars_categorical_tree()
        categories = pandas.Categorical(["a", "b", "c", None])
        X = pandas.DataFrame({"cat": categories})
        predictions = tree.predict(X)
        expected = numpy.array([0.0, -10.0, 10.0, 5.0])
        numpy.testing.assert_array_equal(predictions, expected)

    def test_categorical_unseen_level_falls_back_to_holding_node(self):
        """An unknown category label falls back to the holding node's average."""
        tree = _categorical_tree()
        unseen = pandas.Categorical(["z"], categories=["z"])
        X = pandas.DataFrame({"cat": unseen})
        predictions = tree.predict(X)
        expected = numpy.array([1.25])
        numpy.testing.assert_array_equal(predictions, expected)

    def test_mixed_frame_with_nullable_numeric_roundtrip(self):
        """A frame mixing nullable numeric and boolean-with-missing predicts its own rows."""
        x_column = pandas.array(
            [0] * _N + [10] * _N + [None] * _N, dtype="Int64"
        )
        flag_column = pandas.array(
            [True] * _N + [False] * _N + [None] * _N, dtype="boolean"
        )
        X = pandas.DataFrame({"x": x_column, "flag": flag_column})
        y = numpy.array([0.0] * _N + [-10.0] * _N + [10.0] * _N)
        tree = _fit(X, y)
        head = X.head(3)
        tail = X.tail(3)
        predictions_head = tree.predict(head)
        predictions_tail = tree.predict(tail)
        expected_head = numpy.array([0.0, 0.0, 0.0])
        expected_tail = numpy.array([10.0, 10.0, 10.0])
        numpy.testing.assert_array_equal(predictions_head, expected_head)
        numpy.testing.assert_array_equal(predictions_tail, expected_tail)


class TestSqlExpectationsHeader(unittest.TestCase):
    """to_sql states the expected column types in a leading comment."""

    __slots__ = ()

    def test_header_numeric(self):
        """A numeric split is announced as a numeric column."""
        tree = _numeric_missing_tree()
        sql = tree.to_sql()
        first_line = sql.splitlines()[0]
        self.assertEqual(first_line, '-- Expects: "X[0]" numeric')

    def test_header_boolean(self):
        """A boolean split is announced as a boolean column."""
        tree = _boolean_tree()
        sql = tree.to_sql()
        first_line = sql.splitlines()[0]
        self.assertEqual(first_line, '-- Expects: "flag" boolean')

    def test_header_promoted_boolean(self):
        """A boolean column with missing values is still announced as boolean."""
        tree = _promoted_boolean_tree()
        sql = tree.to_sql()
        first_line = sql.splitlines()[0]
        self.assertEqual(first_line, '-- Expects: "flag" boolean')

    def test_header_categorical_text(self):
        """A label categorical split is announced as a text column."""
        tree = _categorical_tree()
        sql = tree.to_sql()
        first_line = sql.splitlines()[0]
        self.assertEqual(first_line, '-- Expects: "cat" text')

    def test_header_declared_categorical_numeric(self):
        """A code categorical declared on numpy input is announced as numeric."""
        tree = _declared_categorical_tree()
        sql = tree.to_sql()
        first_line = sql.splitlines()[0]
        self.assertEqual(first_line, '-- Expects: "X[0]" numeric')

    def test_header_absent_without_splits(self):
        """A single-leaf tree emits no expectations comment."""
        X = numpy.arange(20, dtype=float).reshape(-1, 1)
        y = numpy.zeros(20)
        tree = _fit(X, y)
        sql = tree.to_sql()
        self.assertNotIn("-- Expects:", sql)
        self.assertIn("-- Leaf 1", sql)

    def test_header_respects_max_depth(self):
        """Columns referenced only below max_depth are not announced."""
        tree = _two_feature_tree()
        sql_full = tree.to_sql()
        sql_truncated = tree.to_sql(max_depth=1)
        full_first_line = sql_full.splitlines()[0]
        truncated_first_line = sql_truncated.splitlines()[0]
        self.assertEqual(
            full_first_line, '-- Expects: "X[0]" numeric, "X[1]" numeric'
        )
        self.assertEqual(truncated_first_line, '-- Expects: "X[0]" numeric')

    def test_sql_with_header_executes_in_sqlite(self):
        """The emitted expression, header included, evaluates in SQLite and matches predict."""
        tree = _categorical_tree()
        sql = tree.to_sql()
        connection = sqlite3.connect(":memory:")
        connection.execute('CREATE TABLE t ("cat" TEXT)')
        rows = [("a",), ("b",), ("c",), (None,)]
        connection.executemany("INSERT INTO t VALUES (?)", rows)
        cursor = connection.execute(f"SELECT {sql} FROM t")
        fetched = cursor.fetchall()
        sql_values = [row[0] for row in fetched]
        categories = pandas.Categorical(
            ["a", "b", "c", None], categories=["a", "b", "c"]
        )
        X = pandas.DataFrame({"cat": categories})
        predictions = tree.predict(X)
        prediction_list = list(predictions)
        self.assertEqual(sql_values, prediction_list)


if __name__ == "__main__":
    unittest.main()
