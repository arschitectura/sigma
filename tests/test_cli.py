"""Unit tests for the sigma command line interface."""

import decimal
import importlib.util
import inspect
import io
import json
import os
import pickle
import re
import subprocess
import sys
import tempfile
import unittest
import unittest.mock
import warnings

import numpy
import numpy.testing
import pandas
import pyarrow
import pyarrow.csv
import pyarrow.ipc
import pyarrow.parquet

import sigma
import sigma._types
import sigma.cli

_HAS_GRAPHVIZ = importlib.util.find_spec("graphviz") is not None
"""Whether the optional graphviz dependency is installed."""

_TABLE = pyarrow.table(
    {
        "feature": [0.5, 1.25, 2.75, 3.125],
        "target": ["a", "b", "a", "b"],
    }
)
"""Reference table whose values survive every supported file format."""

_INPUT_EXTENSIONS = (
    ".csv",
    ".tsv",
    ".parquet",
    ".arrow",
    ".feather",
    ".orc",
    ".jsonl",
    ".ndjson",
)
"""File extensions the command line interface reads."""


def _fit_mixed_tree():
    """Fit a tree on a categorical, a boolean, and a numeric column."""
    levels = pandas.Series(numpy.tile(["a", "b"], 30))
    frame = pandas.DataFrame(
        {
            "g": levels.astype("category"),
            "b": numpy.tile([True, False], 30),
            "x": numpy.arange(60.0),
        }
    )
    response = numpy.tile([0.0, 10.0], 30)
    tree = sigma.RegressionTree(ci_coverage=None).fit(frame, response)
    return tree


_DATA_DIRECTORY = os.path.join(os.path.dirname(__file__), "data")
"""Directory holding the committed test data files."""

_AIRQUALITY = os.path.join(_DATA_DIRECTORY, "airquality.csv")
"""Committed numeric data file used for regression fits."""

_GLAUCOMA = os.path.join(_DATA_DIRECTORY, "glaucoma_m.csv")
"""Committed data file with a text target used for classification fits."""


def _fit_from_file(data_file, task, target, flags=()):
    """Run a fit through the command line and load the resulting tree."""
    with tempfile.TemporaryDirectory() as directory:
        model_path = os.path.join(directory, "model.pkl")
        argv = ["--log", "none", "fit", data_file, task, target, model_path]
        argv.extend(flags)
        status = sigma.cli.run(argv)
        if status != 0:
            raise AssertionError(f"fit failed with status {status}")
        with open(model_path, "rb") as file_handle:
            tree = pickle.load(file_handle)
    return tree


def _fit_exit_status(task, flags):
    """Run a fit that is expected to fail and return its exit status."""
    argv = ["--log", "none", "fit", _AIRQUALITY, task, "Temp", "m.pkl"]
    argv.extend(flags)
    with unittest.mock.patch("sys.stderr", new_callable=io.StringIO):
        try:
            status = sigma.cli.run(argv)
        except SystemExit as system_exit:
            status = system_exit.code
    return status


class _Recorder:
    """Estimator stand-in recording the constructor keywords it receives."""

    def __init__(self, **kwargs):
        _CAPTURED_KWARGS.clear()
        _CAPTURED_KWARGS.update(kwargs)

    def fit(self, X, y, sample_weight=None):
        """Stand in for a fitted tree without fitting anything."""
        return self


_CAPTURED_KWARGS: dict[str, object] = {}
"""Constructor keywords recorded by the most recent stand-in estimator."""


def _capture_estimator_kwargs(task, flags):
    """Collect the constructor keywords a set of fit flags produces."""
    _CAPTURED_KWARGS.clear()
    replacement = {**sigma.cli._ESTIMATORS, task: _Recorder}
    target = _target_for(task)
    with tempfile.TemporaryDirectory() as directory:
        path = os.path.join(directory, "data.csv")
        _write_task_csv(task, path)
        model_path = os.path.join(directory, "model.pkl")
        argv = ["--log", "none", "fit", path, task, target, model_path]
        argv.extend(flags)
        with unittest.mock.patch.dict(sigma.cli._ESTIMATORS, replacement):
            status = sigma.cli.run(argv)
    if status != 0:
        raise AssertionError(f"fit failed with status {status}")
    captured = dict(_CAPTURED_KWARGS)
    return captured


def _target_for(task):
    """Return the target column specification each task fixture uses."""
    match task:
        case "survival":
            return "time,event"
        case "ranking":
            return "i1,i2,i3"
        case _:
            return "y"


def _write_task_csv(task, path):
    """Write the smallest data file the given task accepts."""
    match task:
        case "survival":
            _write_survival_csv(path)
        case "ranking":
            _write_ranking_csv(path)
        case _:
            table = pyarrow.table(
                {"x": numpy.arange(40.0), "y": numpy.tile([0.0, 1.0], 20)}
            )
            sigma.cli._save_data(table, path)


def _write_survival_csv(path, blank_event=False):
    """Write a small right-censored survival data file."""
    times = numpy.linspace(1.0, 10.0, 60)
    events = numpy.tile([1.0, 0.0], 30)
    group = numpy.repeat([0.0, 1.0], 30)
    rendered = [f"{value:g}" for value in events]
    if blank_event:
        rendered[0] = ""
    lines = ["time,event,group"]
    for index in range(60):
        lines.append(f"{times[index]:g},{rendered[index]},{group[index]:g}")
    with open(path, "w", encoding="utf-8") as file_handle:
        file_handle.write("\n".join(lines) + "\n")


def _write_ranking_csv(path, blank_rank=False):
    """Write a small ranking data file with three ranked items."""
    generator = numpy.random.default_rng(0)
    a = numpy.repeat([0.0, 1.0], 20)
    b = generator.standard_normal(40)
    lines = ["a,b,i1,i2,i3"]
    for index in range(40):
        if a[index] == 0.0:
            ranks = ["1", "2", "3"]
        else:
            ranks = ["3", "2", "1"]
        if blank_rank and index == 0:
            ranks[2] = ""
        joined = ",".join(ranks)
        lines.append(f"{a[index]:g},{b[index]:.6f},{joined}")
    with open(path, "w", encoding="utf-8") as file_handle:
        file_handle.write("\n".join(lines) + "\n")


def _survival_frame():
    """Build the covariate frame matching the survival data file."""
    frame = pandas.DataFrame({"group": numpy.repeat([0.0, 1.0], 30)})
    return frame


def _anonymize_node_names(source):
    """Replace the identity-derived graphviz node names with a placeholder."""
    anonymized = re.sub(r"\b\d{6,}\b", "N", source)
    return anonymized


def _run_and_capture(argv):
    """Run a command, returning its status, its stdout bytes, and its stderr."""
    errors = io.StringIO()
    with unittest.mock.patch("sys.stdout") as mock_stdout:
        mock_stdout.buffer = io.BytesIO()
        with unittest.mock.patch("sys.stderr", errors):
            try:
                status = sigma.cli.run(argv)
            except SystemExit as system_exit:
                status = system_exit.code
        payload = mock_stdout.buffer.getvalue()
    return status, payload, errors.getvalue()


def _write_pickle(tree, path):
    """Write a fitted tree to a pickle file."""
    with open(path, "wb") as file_handle:
        pickle.dump(tree, file_handle)


def _predict_table(argv):
    """Run a predict command and read the table it wrote to standard output."""
    status, payload, errors = _run_and_capture(argv)
    if status != 0:
        raise AssertionError(f"predict failed with status {status}: {errors}")
    reader = pyarrow.csv.read_csv(io.BytesIO(payload))
    return reader


class TestParseArgs(unittest.TestCase):
    """Command line parsing and rejection."""

    __slots__ = ()

    def test_fit_requires_four_positional_arguments(self):
        """Omitting any fit positional exits through argparse."""
        with (
            unittest.mock.patch("sys.stderr", new_callable=io.StringIO),
            self.assertRaises(SystemExit),
        ):
            sigma.cli._parse_args(["fit", "d.csv", "regression", "y"])

    def test_predict_requires_two_positional_arguments(self):
        """Omitting the model file of predict exits through argparse."""
        with (
            unittest.mock.patch("sys.stderr", new_callable=io.StringIO),
            self.assertRaises(SystemExit),
        ):
            sigma.cli._parse_args(["predict", "d.csv"])

    def test_export_requires_one_positional_argument(self):
        """Omitting the model file of export exits through argparse."""
        with (
            unittest.mock.patch("sys.stderr", new_callable=io.StringIO),
            self.assertRaises(SystemExit),
        ):
            sigma.cli._parse_args(["export"])

    def test_a_missing_subcommand_is_rejected(self):
        """A bare invocation with no subcommand exits through argparse."""
        with (
            unittest.mock.patch("sys.stderr", new_callable=io.StringIO),
            self.assertRaises(SystemExit),
        ):
            sigma.cli._parse_args([])

    def test_an_unknown_subcommand_is_rejected(self):
        """A subcommand outside fit, predict and export exits."""
        with (
            unittest.mock.patch("sys.stderr", new_callable=io.StringIO),
            self.assertRaises(SystemExit),
        ):
            sigma.cli._parse_args(["train", "d.csv"])

    def test_an_abbreviated_flag_is_rejected(self):
        """Flag prefixes are not accepted, since allow_abbrev is off."""
        argv = ["fit", "d.csv", "regression", "y", "m.pkl", "--max-d", "3"]
        with (
            unittest.mock.patch("sys.stderr", new_callable=io.StringIO),
            self.assertRaises(SystemExit),
        ):
            sigma.cli._parse_args(argv)

    def test_an_unknown_flag_is_rejected(self):
        """A flag the parser does not define exits through argparse."""
        argv = ["fit", "d.csv", "regression", "y", "m.pkl", "--depth", "3"]
        with (
            unittest.mock.patch("sys.stderr", new_callable=io.StringIO),
            self.assertRaises(SystemExit),
        ):
            sigma.cli._parse_args(argv)

    def test_the_task_argument_is_lower_cased(self):
        """A task given in upper case is stored in lower case."""
        args = sigma.cli._parse_args(
            ["fit", "d.csv", "REGRESSION", "y", "m.pkl"]
        )
        self.assertEqual(args.task, "regression")

    def test_the_task_argument_rejects_an_unknown_family(self):
        """A task outside the four estimator families exits."""
        with (
            unittest.mock.patch("sys.stderr", new_callable=io.StringIO),
            self.assertRaises(SystemExit),
        ):
            sigma.cli._parse_args(["fit", "d.csv", "forecasting", "y", "m.pkl"])

    def test_target_columns_accepts_a_single_name(self):
        """One target column parses to a single-entry list."""
        args = sigma.cli._parse_args(
            ["fit", "d.csv", "regression", "y", "m.pkl"]
        )
        self.assertEqual(args.target_columns, ["y"])

    def test_target_columns_splits_a_comma_separated_list(self):
        """A comma-separated target parses to one entry per name."""
        args = sigma.cli._parse_args(
            ["fit", "d.csv", "survival", "time,event", "m.pkl"]
        )
        self.assertEqual(args.target_columns, ["time", "event"])

    def test_target_columns_preserves_the_given_order(self):
        """The target order is kept, which is what selects time before event."""
        args = sigma.cli._parse_args(
            ["fit", "d.csv", "survival", "event,time", "m.pkl"]
        )
        self.assertEqual(args.target_columns, ["event", "time"])

    def test_target_columns_rejects_an_empty_entry(self):
        """A trailing or doubled comma in the target list exits."""
        with (
            unittest.mock.patch("sys.stderr", new_callable=io.StringIO),
            self.assertRaises(SystemExit),
        ):
            sigma.cli._parse_args(
                ["fit", "d.csv", "survival", "time,", "m.pkl"]
            )

    def test_an_omitted_fit_flag_is_absent_from_the_namespace(self):
        """Flags default to SUPPRESS so the estimator default wins."""
        args = sigma.cli._parse_args(
            ["fit", "d.csv", "regression", "y", "m.pkl"]
        )
        self.assertNotIn("alpha", vars(args))

    def test_log_defaults_to_info(self):
        """Omitting the log level selects info."""
        args = sigma.cli._parse_args(
            ["fit", "d.csv", "regression", "y", "m.pkl"]
        )
        self.assertEqual(args.log, "info")

    def test_log_is_read_before_the_subcommand(self):
        """The log level belongs to the root parser, as in Tau."""
        args = sigma.cli._parse_args(
            ["--log", "DEBUG", "fit", "d.csv", "regression", "y", "m.pkl"]
        )
        self.assertEqual(args.log, "debug")

    def test_log_after_the_subcommand_is_rejected(self):
        """Writing the log level after the subcommand exits."""
        argv = ["fit", "d.csv", "regression", "y", "m.pkl", "--log", "debug"]
        with (
            unittest.mock.patch("sys.stderr", new_callable=io.StringIO),
            self.assertRaises(SystemExit),
        ):
            sigma.cli._parse_args(argv)

    def test_version_exits_successfully(self):
        """The version flag prints and exits with status zero."""
        with (
            unittest.mock.patch("sys.stdout", new_callable=io.StringIO),
            self.assertRaises(SystemExit) as raised,
        ):
            sigma.cli._parse_args(["--version"])
        self.assertEqual(raised.exception.code, 0)

    def test_predict_output_defaults_to_none(self):
        """Omitting the output path selects standard output."""
        args = sigma.cli._parse_args(["predict", "d.csv", "m.pkl"])
        self.assertIsNone(args.output)

    def test_predict_output_stores_the_given_path(self):
        """An output path is stored verbatim."""
        args = sigma.cli._parse_args(
            ["predict", "d.csv", "m.pkl", "--output", "out.csv"]
        )
        self.assertEqual(args.output, "out.csv")

    def test_predict_output_format_is_lower_cased(self):
        """An output format given in upper case is stored in lower case."""
        args = sigma.cli._parse_args(
            ["predict", "d.csv", "m.pkl", "--output-format", "PARQUET"]
        )
        self.assertEqual(args.output_format, "parquet")

    def test_predict_output_format_rejects_an_unlisted_value(self):
        """An output format outside the supported list exits."""
        argv = ["predict", "d.csv", "m.pkl", "--output-format", "xyz"]
        with (
            unittest.mock.patch("sys.stderr", new_callable=io.StringIO),
            self.assertRaises(SystemExit),
        ):
            sigma.cli._parse_args(argv)

    def test_every_predict_flag_is_off_by_default(self):
        """The optional prediction outputs are all disabled by default."""
        args = sigma.cli._parse_args(["predict", "d.csv", "m.pkl"])
        self.assertFalse(args.proba)
        self.assertFalse(args.rank)
        self.assertFalse(args.node)
        self.assertFalse(args.with_input)
        self.assertIsNone(args.times)

    def test_times_parses_a_comma_separated_float_list(self):
        """A time list parses to floats in the given order."""
        args = sigma.cli._parse_args(
            ["predict", "d.csv", "m.pkl", "--times", "1,2.5,10"]
        )
        self.assertEqual(args.times, [1.0, 2.5, 10.0])

    def test_times_rejects_a_non_numeric_entry(self):
        """A time list holding a non-number exits."""
        argv = ["predict", "d.csv", "m.pkl", "--times", "1,later"]
        with (
            unittest.mock.patch("sys.stderr", new_callable=io.StringIO),
            self.assertRaises(SystemExit),
        ):
            sigma.cli._parse_args(argv)

    def test_times_rejects_a_decreasing_list(self):
        """A time list that is not non-decreasing exits."""
        argv = ["predict", "d.csv", "m.pkl", "--times", "5,1"]
        with (
            unittest.mock.patch("sys.stderr", new_callable=io.StringIO),
            self.assertRaises(SystemExit),
        ):
            sigma.cli._parse_args(argv)

    def test_export_format_defaults_to_none(self):
        """Omitting the export format leaves it to be inferred."""
        args = sigma.cli._parse_args(["export", "m.pkl"])
        self.assertIsNone(args.format)

    def test_export_format_rejects_an_unlisted_value(self):
        """An export format outside the supported list exits."""
        with (
            unittest.mock.patch("sys.stderr", new_callable=io.StringIO),
            self.assertRaises(SystemExit),
        ):
            sigma.cli._parse_args(["export", "m.pkl", "--format", "jpeg"])

    def test_a_negative_max_depth_is_rejected(self):
        """A negative depth exits instead of fitting a silent stump."""
        argv = ["fit", "d.csv", "regression", "y", "m.pkl", "--max-depth", "-1"]
        with (
            unittest.mock.patch("sys.stderr", new_callable=io.StringIO),
            self.assertRaises(SystemExit),
        ):
            sigma.cli._parse_args(argv)

    def test_an_out_of_range_alpha_is_rejected(self):
        """A significance level outside (0, 1] exits."""
        argv = ["fit", "d.csv", "regression", "y", "m.pkl", "--alpha", "-0.5"]
        with (
            unittest.mock.patch("sys.stderr", new_callable=io.StringIO),
            self.assertRaises(SystemExit),
        ):
            sigma.cli._parse_args(argv)

    def test_max_depth_accepts_the_literal_none(self):
        """The literal none forwards the unlimited-depth default."""
        args = sigma.cli._parse_args(
            ["fit", "d.csv", "regression", "y", "m.pkl", "--max-depth", "none"]
        )
        self.assertIsNone(args.max_depth)

    def test_ci_coverage_accepts_the_literal_none(self):
        """The literal none disables confidence intervals."""
        args = sigma.cli._parse_args(
            [
                "fit",
                "d.csv",
                "regression",
                "y",
                "m.pkl",
                "--ci-coverage",
                "none",
            ]
        )
        self.assertIsNone(args.ci_coverage)


class TestFileFormats(unittest.TestCase):
    """Reading and writing every supported tabular file format."""

    __slots__ = ()

    def test_every_input_format_round_trips_the_reference_table(self):
        """Each readable extension restores the schema and the values."""
        with tempfile.TemporaryDirectory() as directory:
            for extension in _INPUT_EXTENSIONS:
                with self.subTest(extension=extension):
                    path = os.path.join(directory, f"data{extension}")
                    sigma.cli._save_data(_TABLE, path)
                    restored = sigma.cli._load_data(path)
                    self.assertEqual(restored.to_pydict(), _TABLE.to_pydict())

    def test_an_uppercase_extension_selects_the_same_reader(self):
        """Extension matching ignores case."""
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "DATA.CSV")
            sigma.cli._save_data(_TABLE, path)
            restored = sigma.cli._load_data(path)
        self.assertEqual(restored.column_names, ["feature", "target"])

    def test_markdown_is_rejected_as_an_input_format(self):
        """Markdown is an output-only format."""
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "data.md")
            sigma.cli._save_data(_TABLE, path)
            with self.assertRaisesRegex(ValueError, "unsupported input file"):
                sigma.cli._load_data(path)

    def test_an_unlisted_input_extension_is_rejected(self):
        """An unknown extension names the supported input formats."""
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "data.xyz")
            with open(path, "wb") as file_handle:
                file_handle.write(b"anything")
            with self.assertRaisesRegex(ValueError, "unsupported input file"):
                sigma.cli._load_data(path)

    def test_a_missing_input_file_is_rejected(self):
        """A path that does not exist raises before any dispatch."""
        with self.assertRaises(FileNotFoundError):
            sigma.cli._load_data("absent.csv")

    def test_a_file_with_duplicate_column_names_is_rejected(self):
        """Duplicate column names cannot be addressed and are refused."""
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "data.csv")
            with open(path, "w", encoding="utf-8") as file_handle:
                file_handle.write("a,a\n1,2\n")
            with self.assertRaisesRegex(ValueError, "duplicate column names"):
                sigma.cli._load_data(path)

    def test_every_output_format_writes_a_non_empty_file(self):
        """Each writable format produces bytes on disk."""
        with tempfile.TemporaryDirectory() as directory:
            for output_format in sigma.cli._OUTPUT_FORMATS:
                with self.subTest(output_format=output_format):
                    path = os.path.join(directory, f"out.{output_format}")
                    sigma.cli._save_data(_TABLE, path, output_format)
                    self.assertGreater(os.path.getsize(path), 0)

    def test_every_output_format_reaches_standard_output(self):
        """Each writable format can be streamed to standard output."""
        for output_format in sigma.cli._OUTPUT_FORMATS:
            with self.subTest(output_format=output_format):
                with unittest.mock.patch("sys.stdout") as mock_stdout:
                    mock_stdout.buffer = io.BytesIO()
                    sigma.cli._save_data(_TABLE, None, output_format)
                    payload = mock_stdout.buffer.getvalue()
                self.assertGreater(len(payload), 0)

    def test_standard_output_without_a_forced_format_is_csv(self):
        """Standard output defaults to comma separated values."""
        with unittest.mock.patch("sys.stdout") as mock_stdout:
            mock_stdout.buffer = io.BytesIO()
            sigma.cli._save_data(_TABLE, None)
            payload = mock_stdout.buffer.getvalue()
        self.assertTrue(payload.startswith(b'"feature","target"'))

    def test_a_forced_format_overrides_the_file_extension(self):
        """The explicit format wins over the extension of the output path."""
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "out.csv")
            sigma.cli._save_data(_TABLE, path, "parquet")
            restored = pyarrow.parquet.read_table(path)
        self.assertEqual(restored.to_pydict(), _TABLE.to_pydict())

    def test_an_unlisted_output_format_is_rejected(self):
        """An unknown output format names the supported output formats."""
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "out.xyz")
            with self.assertRaisesRegex(ValueError, "unsupported output file"):
                sigma.cli._save_data(_TABLE, path)

    def test_arrow_output_is_written_without_a_deprecation_warning(self):
        """Arrow files go through the ipc writer, not the deprecated feather one."""
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "out.arrow")
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                sigma.cli._save_data(_TABLE, path)
                sigma.cli._load_data(path)
            with open(path, "rb") as file_handle:
                magic = file_handle.read(6)
        self.assertEqual(magic, b"ARROW1")
        deprecations = [
            entry
            for entry in caught
            if issubclass(entry.category, (DeprecationWarning, FutureWarning))
        ]
        self.assertEqual(deprecations, [])

    def test_feather_and_arrow_extensions_produce_identical_bytes(self):
        """The two arrow extensions select the same writer."""
        with tempfile.TemporaryDirectory() as directory:
            arrow_path = os.path.join(directory, "out.arrow")
            feather_path = os.path.join(directory, "out.feather")
            sigma.cli._save_data(_TABLE, arrow_path)
            sigma.cli._save_data(_TABLE, feather_path)
            with open(arrow_path, "rb") as file_handle:
                arrow_bytes = file_handle.read()
            with open(feather_path, "rb") as file_handle:
                feather_bytes = file_handle.read()
        self.assertEqual(arrow_bytes, feather_bytes)


class TestJsonLinesWriter(unittest.TestCase):
    """The JSON Lines rendering of a table."""

    __slots__ = ()

    def test_json_lines_output_holds_one_object_per_row(self):
        """Each line parses back to the matching row mapping."""
        target = io.BytesIO()
        sigma.cli._save_jsonl(_TABLE, target)
        text = target.getvalue().decode()
        rows = [json.loads(line) for line in text.splitlines()]
        self.assertEqual(rows, _TABLE.to_pylist())

    def test_json_lines_output_ends_with_a_newline(self):
        """The document is newline terminated, including the last row."""
        target = io.BytesIO()
        sigma.cli._save_jsonl(_TABLE, target)
        self.assertTrue(target.getvalue().endswith(b"\n"))

    def test_json_lines_renders_a_missing_value_as_json_null(self):
        """A null cell becomes a JSON null."""
        table = pyarrow.table({"a": [1.0, None]})
        target = io.BytesIO()
        sigma.cli._save_jsonl(table, target)
        rows = [
            json.loads(line) for line in target.getvalue().decode().splitlines()
        ]
        self.assertIsNone(rows[1]["a"])


class TestMarkdownWriter(unittest.TestCase):
    """The Markdown rendering of a table."""

    __slots__ = ()

    def test_markdown_output_starts_with_a_header_row_and_a_rule(self):
        """The first line names the columns and the second is a rule."""
        target = io.BytesIO()
        sigma.cli._save_markdown_table(_TABLE, target)
        lines = target.getvalue().decode().splitlines()
        self.assertEqual(lines[0], "| feature | target |")
        self.assertTrue(lines[1].startswith("|---"))
        self.assertEqual(len(lines), 2 + _TABLE.num_rows)

    def test_markdown_cells_are_padded_to_the_column_width(self):
        """Every cell is left justified to the widest value of its column."""
        table = pyarrow.table({"a": ["x", "yyy"], "bb": [1.0, 2.0]})
        target = io.BytesIO()
        sigma.cli._save_markdown_table(table, target)
        lines = target.getvalue().decode().splitlines()
        self.assertEqual(lines[0], "| a   | bb  |")
        self.assertEqual(lines[1], "|-----|-----|")
        self.assertEqual(lines[2], "| x   | 1.0 |")
        self.assertEqual(lines[3], "| yyy | 2.0 |")

    def test_markdown_renders_a_missing_value_as_null(self):
        """A null cell renders as the text null."""
        table = pyarrow.table({"a": [None]})
        target = io.BytesIO()
        sigma.cli._save_markdown_table(table, target)
        self.assertIn("null", target.getvalue().decode())

    def test_markdown_renders_booleans_in_lower_case(self):
        """Truth values render as true and false."""
        table = pyarrow.table({"a": [True, False]})
        target = io.BytesIO()
        sigma.cli._save_markdown_table(table, target)
        text = target.getvalue().decode()
        self.assertIn("true", text)
        self.assertIn("false", text)

    def test_markdown_limits_floats_to_six_significant_digits(self):
        """A repeating fraction is cut to six significant digits."""
        table = pyarrow.table({"a": [1.0 / 3.0]})
        target = io.BytesIO()
        sigma.cli._save_markdown_table(table, target)
        self.assertIn("0.333333", target.getvalue().decode())

    def test_markdown_keeps_a_whole_number_float_decimal(self):
        """A float that renders as an integer keeps a decimal part."""
        table = pyarrow.table({"a": [2.0]})
        target = io.BytesIO()
        sigma.cli._save_markdown_table(table, target)
        self.assertIn("2.0", target.getvalue().decode())

    def test_markdown_folds_a_multi_line_value_onto_one_row(self):
        """Line feeds inside a value are replaced by spaces."""
        table = pyarrow.table({"a": ["multi\n\npara"]})
        target = io.BytesIO()
        sigma.cli._save_markdown_table(table, target)
        lines = target.getvalue().decode().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertIn("multi  para", lines[2])

    def test_markdown_output_of_an_empty_table_has_no_rows(self):
        """A table with no rows renders as a header and a rule only."""
        table = pyarrow.table({"a": pyarrow.array([], type=pyarrow.float64())})
        target = io.BytesIO()
        sigma.cli._save_markdown_table(table, target)
        lines = target.getvalue().decode().splitlines()
        self.assertEqual(lines, ["| a |", "|---|"])


class TestColumnNormalization(unittest.TestCase):
    """Arrow-side typing and the conversion to pandas."""

    __slots__ = ()

    def test_a_string_column_becomes_a_pandas_categorical(self):
        """Text is turned into a categorical, which is what sigma accepts."""
        table = pyarrow.table({"g": ["a", "b", "a"]})
        normalized = sigma.cli._normalize_fit_columns(table)
        frame = sigma.cli._to_pandas(normalized)
        self.assertIsInstance(frame["g"].dtype, pandas.CategoricalDtype)

    def test_a_dictionary_column_becomes_a_pandas_categorical(self):
        """A dictionary encoded column normalizes like plain text."""
        values = pyarrow.array(["a", "b", "a"]).dictionary_encode()
        table = pyarrow.table({"g": values})
        normalized = sigma.cli._normalize_fit_columns(table)
        frame = sigma.cli._to_pandas(normalized)
        self.assertIsInstance(frame["g"].dtype, pandas.CategoricalDtype)

    def test_category_labels_do_not_depend_on_the_null_count(self):
        """The same levels render identically with and without a null."""
        complete = pyarrow.array([1, 2], type=pyarrow.int32())
        with_null = pyarrow.array([1, 2, None], type=pyarrow.int32())
        first = pyarrow.table({"g": complete.dictionary_encode()})
        second = pyarrow.table({"g": with_null.dictionary_encode()})
        first_normalized = sigma.cli._normalize_fit_columns(first)
        second_normalized = sigma.cli._normalize_fit_columns(second)
        first_frame = sigma.cli._to_pandas(first_normalized)
        second_frame = sigma.cli._to_pandas(second_normalized)
        first_levels = first_frame["g"].cat.categories.tolist()
        second_levels = second_frame["g"].cat.categories.tolist()
        self.assertEqual(first_levels, second_levels)

    def test_a_null_free_boolean_column_stays_boolean(self):
        """Truth values keep a boolean dtype sigma recognizes."""
        table = pyarrow.table({"b": [True, False]})
        normalized = sigma.cli._normalize_fit_columns(table)
        frame = sigma.cli._to_pandas(normalized)
        self.assertTrue(pandas.api.types.is_bool_dtype(frame["b"].dtype))

    def test_a_boolean_column_with_a_null_stays_boolean(self):
        """A missing truth value does not demote the column to an object."""
        table = pyarrow.table({"b": [True, None, False]})
        normalized = sigma.cli._normalize_fit_columns(table)
        frame = sigma.cli._to_pandas(normalized)
        self.assertTrue(pandas.api.types.is_bool_dtype(frame["b"].dtype))

    def test_a_decimal_column_becomes_numeric(self):
        """Decimals widen to floats rather than reaching sigma as objects."""
        values = pyarrow.array(
            [decimal.Decimal("1.5"), decimal.Decimal("2.5")],
            type=pyarrow.decimal128(12, 3),
        )
        table = pyarrow.table({"d": values})
        normalized = sigma.cli._normalize_fit_columns(table)
        frame = sigma.cli._to_pandas(normalized)
        self.assertTrue(pandas.api.types.is_numeric_dtype(frame["d"].dtype))

    def test_a_string_view_column_becomes_a_pandas_categorical(self):
        """The string view type is text too, and arrow files may carry it."""
        values = pyarrow.array(["a", "b"], type=pyarrow.string_view())
        table = pyarrow.table({"g": values})
        normalized = sigma.cli._normalize_fit_columns(table)
        frame = sigma.cli._to_pandas(normalized)
        self.assertIsInstance(frame["g"].dtype, pandas.CategoricalDtype)

    def test_a_null_free_integer_column_stays_integer(self):
        """Whole numbers keep their integer dtype."""
        table = pyarrow.table({"n": [1, 2, 3]})
        normalized = sigma.cli._normalize_fit_columns(table)
        frame = sigma.cli._to_pandas(normalized)
        self.assertTrue(pandas.api.types.is_integer_dtype(frame["n"].dtype))

    def test_a_timestamp_column_is_refused_by_name_and_type(self):
        """An unusable column type names the column and the arrow type."""
        values = pyarrow.array([0, 1], type=pyarrow.timestamp("s"))
        table = pyarrow.table({"t": values})
        with self.assertRaisesRegex(ValueError, "cannot use as a feature"):
            sigma.cli._normalize_fit_columns(table)

    def test_an_empty_csv_field_is_read_as_a_missing_value(self):
        """A blank cell is a null rather than an empty category level."""
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "data.csv")
            with open(path, "w", encoding="utf-8") as file_handle:
                file_handle.write("g,x\na,1\n,2\nb,3\n")
            table = sigma.cli._load_data(path)
        self.assertEqual(table.column("g").null_count, 1)


class TestPredictSchema(unittest.TestCase):
    """Rebuilding the fit-time column types from a fitted tree."""

    __slots__ = ()

    def test_the_reader_types_mirror_the_fitted_column_kinds(self):
        """Each feature column is read back as the type it was fitted with."""
        tree = _fit_mixed_tree()
        column_types = sigma.cli._predict_column_types(tree)
        self.assertEqual(column_types["g"], pyarrow.string())
        self.assertEqual(column_types["b"], pyarrow.bool_())
        self.assertEqual(column_types["x"], pyarrow.float64())

    def test_a_declared_categorical_numeric_column_stays_numeric(self):
        """A numeric column named in categorical_features keeps its width."""
        frame = pandas.DataFrame(
            {"c": numpy.tile([0.0, 1.0, 2.0], 20), "x": numpy.arange(60.0)}
        )
        response = numpy.tile([0.0, 5.0, 10.0], 20)
        estimator = sigma.RegressionTree(
            categorical_features=["c"], ci_coverage=None
        )
        tree = estimator.fit(frame, response)
        column_types = sigma.cli._predict_column_types(tree)
        self.assertEqual(column_types["c"], pyarrow.float64())

    def test_prediction_columns_are_projected_in_the_fitted_order(self):
        """Extra columns are dropped and the fitted order is restored."""
        tree = _fit_mixed_tree()
        table = pyarrow.table(
            {
                "extra": [1.0, 2.0],
                "x": [0.0, 1.0],
                "b": [True, False],
                "g": ["a", "b"],
            }
        )
        projected = sigma.cli._normalize_predict_columns(table, tree)
        self.assertEqual(projected.column_names, ["g", "b", "x"])

    def test_a_missing_feature_column_names_every_absentee(self):
        """A data file short of fitted features lists what is missing."""
        tree = _fit_mixed_tree()
        table = pyarrow.table({"g": ["a", "b"]})
        with self.assertRaisesRegex(ValueError, "missing feature columns"):
            sigma.cli._normalize_predict_columns(table, tree)

    def test_a_boolean_feature_supplied_as_text_is_refused(self):
        """Text in a boolean column fails loudly instead of reading as true."""
        tree = _fit_mixed_tree()
        table = pyarrow.table(
            {"g": ["a", "b"], "b": ["yes", "no"], "x": [0.0, 1.0]}
        )
        with self.assertRaisesRegex(ValueError, "column 'b'"):
            sigma.cli._normalize_predict_columns(table, tree)

    def test_a_numeric_feature_supplied_as_a_timestamp_is_refused(self):
        """Dates in a numeric column fail instead of becoming epoch counts."""
        tree = _fit_mixed_tree()
        table = pyarrow.table(
            {
                "g": ["a", "b"],
                "b": [True, False],
                "x": pyarrow.array([0, 1], type=pyarrow.timestamp("s")),
            }
        )
        with self.assertRaisesRegex(ValueError, "column 'x'"):
            sigma.cli._normalize_predict_columns(table, tree)

    def test_a_numeric_looking_category_survives_a_csv_round_trip(self):
        """Levels such as 01 are not re-inferred as integers at predict."""
        levels = numpy.tile(["01", "02"], 30)
        series = pandas.Series(levels).astype("category")
        frame = pandas.DataFrame({"g": series})
        response = numpy.tile([0.0, 10.0], 30)
        tree = sigma.RegressionTree(ci_coverage=None).fit(frame, response)
        expected = tree.predict(frame)
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "data.csv")
            table = pyarrow.table({"g": levels.tolist()})
            sigma.cli._save_data(table, path)
            column_types = sigma.cli._predict_column_types(tree)
            reloaded = sigma.cli._load_data(path, column_types)
            projected = sigma.cli._normalize_predict_columns(reloaded, tree)
            restored = sigma.cli._to_pandas(projected)
        predictions = tree.predict(restored)
        numpy.testing.assert_array_equal(predictions, expected)


class TestRunFitRegression(unittest.TestCase):
    """Fitting a regression tree from a data file."""

    __slots__ = ()

    def test_fitting_a_csv_writes_a_loadable_regression_tree(self):
        """The fitted pickle restores as a RegressionTree."""
        with tempfile.TemporaryDirectory() as directory:
            model_path = os.path.join(directory, "model.pkl")
            status = sigma.cli.run(
                [
                    "--log",
                    "none",
                    "fit",
                    _AIRQUALITY,
                    "regression",
                    "Temp",
                    model_path,
                ]
            )
            with open(model_path, "rb") as file_handle:
                tree = pickle.load(file_handle)
        self.assertEqual(status, 0)
        self.assertIsInstance(tree, sigma.RegressionTree)

    def test_every_non_target_column_becomes_a_feature(self):
        """The feature set is the data file minus the target column."""
        tree = _fit_from_file(_AIRQUALITY, "regression", "Temp")
        expected = ["rownames", "Ozone", "Solar.R", "Wind", "Month", "Day"]
        self.assertEqual(list(tree.feature_names_in_), expected)

    def test_the_fitted_tree_carries_the_running_sigma_version(self):
        """The pickle written by the command line is version stamped."""
        tree = _fit_from_file(_AIRQUALITY, "regression", "Temp")
        state = tree.__getstate__()
        self.assertEqual(state["_sigma_version"], sigma.__version__)

    def test_a_target_column_holding_missing_values_is_refused(self):
        """A target with nulls names the column instead of failing in sklearn."""
        with tempfile.TemporaryDirectory() as directory:
            model_path = os.path.join(directory, "model.pkl")
            with self.assertRaisesRegex(ValueError, "missing values"):
                sigma.cli.run_fit(
                    _AIRQUALITY, "regression", ["Ozone"], model_path
                )

    def test_a_missing_target_column_lists_the_available_columns(self):
        """An unknown target names the columns the file does carry."""
        with tempfile.TemporaryDirectory() as directory:
            model_path = os.path.join(directory, "model.pkl")
            with self.assertRaisesRegex(ValueError, "not found in data"):
                sigma.cli.run_fit(
                    _AIRQUALITY, "regression", ["absent"], model_path
                )

    def test_more_than_one_target_column_is_refused(self):
        """Regression takes exactly one target column."""
        with tempfile.TemporaryDirectory() as directory:
            model_path = os.path.join(directory, "model.pkl")
            with self.assertRaisesRegex(ValueError, "one target column"):
                sigma.cli.run_fit(
                    _AIRQUALITY, "regression", ["Temp", "Wind"], model_path
                )

    def test_the_sample_weight_column_is_not_used_as_a_feature(self):
        """A weight column is consumed as weights and dropped from X."""
        with tempfile.TemporaryDirectory() as directory:
            model_path = os.path.join(directory, "model.pkl")
            sigma.cli.run_fit(
                _AIRQUALITY,
                "regression",
                ["Temp"],
                model_path,
                sample_weight="Day",
            )
            with open(model_path, "rb") as file_handle:
                tree = pickle.load(file_handle)
        self.assertNotIn("Day", list(tree.feature_names_in_))


class TestRunFitClassification(unittest.TestCase):
    """Fitting a classification tree from a data file."""

    __slots__ = ()

    def test_a_text_target_recovers_the_class_labels(self):
        """Text classes are read from the file without a manual cast."""
        tree = _fit_from_file(_GLAUCOMA, "classification", "Class")
        self.assertEqual(list(tree.classes_), ["glaucoma", "normal"])

    def test_the_written_model_is_a_classification_tree(self):
        """The task name selects the ClassificationTree estimator."""
        tree = _fit_from_file(_GLAUCOMA, "classification", "Class")
        self.assertIsInstance(tree, sigma.ClassificationTree)


class TestRunFitSurvival(unittest.TestCase):
    """Fitting a survival tree from a data file."""

    __slots__ = ()

    def test_two_target_columns_are_read_as_time_and_event(self):
        """A time and an event column fit a SurvivalTree."""
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "survival.csv")
            _write_survival_csv(path)
            tree = _fit_from_file(path, "survival", "time,event")
        self.assertIsInstance(tree, sigma.SurvivalTree)

    def test_the_target_order_selects_time_before_event(self):
        """Naming the event column first is refused, so the order is what counts."""
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "survival.csv")
            _write_survival_csv(path)
            model_path = os.path.join(directory, "model.pkl")
            with self.assertRaisesRegex(ValueError, "only 0 and 1"):
                sigma.cli.run_fit(
                    path, "survival", ["event", "time"], model_path
                )

    def test_three_target_columns_are_refused(self):
        """Survival takes one or two target columns."""
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "survival.csv")
            _write_survival_csv(path)
            model_path = os.path.join(directory, "model.pkl")
            with self.assertRaisesRegex(ValueError, "target column"):
                sigma.cli.run_fit(
                    path, "survival", ["time", "event", "group"], model_path
                )

    def test_an_event_column_holding_missing_values_is_refused(self):
        """A survival target with nulls names the column."""
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "survival.csv")
            _write_survival_csv(path, blank_event=True)
            model_path = os.path.join(directory, "model.pkl")
            with self.assertRaisesRegex(ValueError, "missing values"):
                sigma.cli.run_fit(
                    path, "survival", ["time", "event"], model_path
                )


class TestRunFitRanking(unittest.TestCase):
    """Fitting a ranking tree from a data file."""

    __slots__ = ()

    def test_the_rank_columns_become_the_item_names(self):
        """The target column names are carried into item_names_."""
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ranking.csv")
            _write_ranking_csv(path)
            tree = _fit_from_file(
                path, "ranking", "i1,i2,i3", ["--ci-coverage", "none"]
            )
        self.assertEqual(list(tree.item_names_), ["i1", "i2", "i3"])

    def test_a_single_target_column_is_refused(self):
        """Ranking needs at least two item columns."""
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ranking.csv")
            _write_ranking_csv(path)
            model_path = os.path.join(directory, "model.pkl")
            with self.assertRaisesRegex(ValueError, "target column"):
                sigma.cli.run_fit(path, "ranking", ["i1"], model_path)

    def test_an_unranked_item_stays_a_missing_value(self):
        """A blank rank cell is legitimate and does not refuse the fit."""
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ranking.csv")
            _write_ranking_csv(path, blank_rank=True)
            tree = _fit_from_file(
                path, "ranking", "i1,i2,i3", ["--ci-coverage", "none"]
            )
        self.assertIsInstance(tree, sigma.RankingTree)


class TestHyperparameterForwarding(unittest.TestCase):
    """Passing fit flags through to the estimator constructor."""

    __slots__ = ()

    def test_every_shared_flag_reaches_the_constructor(self):
        """Each shared flag arrives as the constructor keyword it names."""
        cases = (
            (["--correlation", "normal"], "correlation", "normal"),
            (["--test-stat", "maximum"], "test_stat", "maximum"),
            (["--test-type", "bonferroni"], "test_type", "bonferroni"),
            (["--alpha", "0.1"], "alpha", 0.1),
            (["--min-splits", "5"], "min_splits", 5),
            (["--min-buckets", "3"], "min_buckets", 3),
            (["--max-depth", "2"], "max_depth", 2),
            (["--max-depth", "none"], "max_depth", None),
            (
                ["--categorical-features", "a,b"],
                "categorical_features",
                ["a", "b"],
            ),
            (["--ci-method", "normal"], "ci_method", "normal"),
            (["--ci-coverage", "0.9"], "ci_coverage", 0.9),
            (["--ci-coverage", "none"], "ci_coverage", None),
            (["--resamples", "50"], "resamples", 50),
            (["--random-state", "7"], "random_state", 7),
            (["--reverse-order"], "reverse_order", True),
        )
        for flags, keyword, expected in cases:
            with self.subTest(flags=flags):
                captured = _capture_estimator_kwargs("regression", flags)
                self.assertEqual(captured[keyword], expected)

    def test_an_omitted_flag_is_not_passed_to_the_constructor(self):
        """Omitted flags leave the estimator default in charge."""
        captured = _capture_estimator_kwargs("regression", [])
        self.assertEqual(captured, {})

    def test_the_regression_only_flag_reaches_the_constructor(self):
        """The response sample size is forwarded for regression."""
        captured = _capture_estimator_kwargs(
            "regression", ["--response-sample-size", "0"]
        )
        self.assertEqual(captured["response_sample_size"], 0)

    def test_a_parametrized_metric_is_forwarded_as_a_tuple(self):
        """A kind:value:unit metric becomes a three-element tuple."""
        captured = _capture_estimator_kwargs(
            "survival", ["--metrics", "median,survival:5:years"]
        )
        self.assertEqual(
            captured["metrics"], ("median", ("survival", 5.0, "years"))
        )

    def test_the_ranking_only_flags_reach_the_constructor(self):
        """Every ranking knob is forwarded under its constructor name."""
        flags = [
            "--pca-components",
            "3",
            "--npseudo",
            "0.25",
            "--pl-max-iter",
            "10",
            "--pl-tolerance",
            "0.001",
            "--ci-replicates",
            "5",
        ]
        captured = _capture_estimator_kwargs("ranking", flags)
        self.assertEqual(captured["pca_components"], 3)
        self.assertEqual(captured["npseudo"], 0.25)
        self.assertEqual(captured["pl_max_iter"], 10)
        self.assertEqual(captured["pl_tolerance"], 0.001)
        self.assertEqual(captured["ci_replicates"], 5)

    def test_a_monte_carlo_fit_runs_end_to_end(self):
        """The test type and its resample count work together on real data."""
        tree = _fit_from_file(
            _AIRQUALITY,
            "regression",
            "Temp",
            [
                "--test-type",
                "monte_carlo",
                "--resamples",
                "50",
                "--random-state",
                "0",
            ],
        )
        self.assertEqual(
            tree.test_type_enum_, sigma._types.TestType.MONTE_CARLO
        )


class TestPerTaskFlagRejection(unittest.TestCase):
    """Refusing a fit flag that the chosen task does not accept."""

    __slots__ = ()

    def test_a_ranking_only_flag_is_refused_for_the_other_tasks(self):
        """Ranking knobs exit with a usage error on the other three tasks."""
        flags = (
            ["--pca-components", "3"],
            ["--npseudo", "0.5"],
            ["--pl-max-iter", "10"],
            ["--pl-tolerance", "0.001"],
            ["--ci-replicates", "5"],
        )
        for task in ("regression", "classification", "survival"):
            for flag in flags:
                with self.subTest(task=task, flag=flag[0]):
                    self.assertEqual(_fit_exit_status(task, flag), 2)

    def test_the_survival_only_flag_is_refused_for_the_other_tasks(self):
        """The metric list exits with a usage error outside survival."""
        for task in ("regression", "classification", "ranking"):
            with self.subTest(task=task):
                status = _fit_exit_status(task, ["--metrics", "median"])
                self.assertEqual(status, 2)

    def test_the_regression_only_flag_is_refused_for_the_other_tasks(self):
        """The response sample size exits with a usage error elsewhere."""
        for task in ("classification", "survival", "ranking"):
            with self.subTest(task=task):
                status = _fit_exit_status(
                    task, ["--response-sample-size", "10"]
                )
                self.assertEqual(status, 2)

    def test_a_confidence_interval_method_of_another_task_is_refused(self):
        """A valid method belonging to another task exits with a usage error."""
        cases = (
            ("regression", "jeffreys"),
            ("classification", "bca"),
            ("survival", "wilson"),
            ("ranking", "student_t"),
        )
        for task, method in cases:
            with self.subTest(task=task, method=method):
                status = _fit_exit_status(task, ["--ci-method", method])
                self.assertEqual(status, 2)

    def test_the_rejection_names_the_flag_and_the_task(self):
        """The usage error quotes both the offending flag and the task."""
        captured = io.StringIO()
        with (
            unittest.mock.patch("sys.stderr", captured),
            self.assertRaises(SystemExit),
        ):
            sigma.cli._parse_args(
                [
                    "fit",
                    "d.csv",
                    "regression",
                    "y",
                    "m.pkl",
                    "--npseudo",
                    "0.5",
                ]
            )
        message = captured.getvalue()
        self.assertIn("--npseudo", message)
        self.assertIn("'ranking'", message)


class TestRunPredict(unittest.TestCase):
    """Predicting with a fitted tree from a data file."""

    __slots__ = ()

    model_bytes = b""

    @classmethod
    def setUpClass(cls):
        """Fit the shared regression tree once for the whole class."""
        tree = _fit_from_file(_AIRQUALITY, "regression", "Temp")
        cls.model_bytes = pickle.dumps(tree)

    def test_predict_writes_one_prediction_per_input_row(self):
        """The output holds exactly as many rows as the data file."""
        with tempfile.TemporaryDirectory() as directory:
            model_path = os.path.join(directory, "model.pkl")
            with open(model_path, "wb") as file_handle:
                file_handle.write(self.model_bytes)
            table = _predict_table(
                ["--log", "none", "predict", _AIRQUALITY, model_path]
            )
        self.assertEqual(table.column_names, ["prediction"])
        self.assertEqual(table.num_rows, 153)

    def test_predict_writes_the_requested_output_file(self):
        """An output path receives the predictions instead of stdout."""
        with tempfile.TemporaryDirectory() as directory:
            model_path = os.path.join(directory, "model.pkl")
            with open(model_path, "wb") as file_handle:
                file_handle.write(self.model_bytes)
            out_path = os.path.join(directory, "out.parquet")
            status = sigma.cli.run(
                [
                    "--log",
                    "none",
                    "predict",
                    _AIRQUALITY,
                    model_path,
                    "--output",
                    out_path,
                ]
            )
            restored = pyarrow.parquet.read_table(out_path)
        self.assertEqual(status, 0)
        self.assertEqual(restored.num_rows, 153)

    def test_predict_reads_every_input_format(self):
        """Predictions are identical whichever readable format carries the data."""
        with tempfile.TemporaryDirectory() as directory:
            model_path = os.path.join(directory, "model.pkl")
            with open(model_path, "wb") as file_handle:
                file_handle.write(self.model_bytes)
            source_table = sigma.cli._load_data(_AIRQUALITY)
            expected = None
            for extension in _INPUT_EXTENSIONS:
                with self.subTest(extension=extension):
                    path = os.path.join(directory, f"data{extension}")
                    sigma.cli._save_data(source_table, path)
                    table = _predict_table(
                        ["--log", "none", "predict", path, model_path]
                    )
                    values = table.column("prediction").to_pylist()
                    if expected is None:
                        expected = values
                    self.assertEqual(values, expected)

    def test_a_data_file_missing_a_feature_column_is_refused(self):
        """A file short of a fitted feature exits with a named error."""
        with tempfile.TemporaryDirectory() as directory:
            model_path = os.path.join(directory, "model.pkl")
            with open(model_path, "wb") as file_handle:
                file_handle.write(self.model_bytes)
            path = os.path.join(directory, "short.csv")
            table = pyarrow.table({"Wind": [1.0, 2.0]})
            sigma.cli._save_data(table, path)
            status, _, errors = _run_and_capture(
                ["--log", "none", "predict", path, model_path]
            )
        self.assertEqual(status, 1)
        self.assertIn("missing feature columns", errors)

    def test_a_data_file_still_carrying_the_target_is_accepted(self):
        """The target column left in the file is ignored rather than refused."""
        with tempfile.TemporaryDirectory() as directory:
            model_path = os.path.join(directory, "model.pkl")
            with open(model_path, "wb") as file_handle:
                file_handle.write(self.model_bytes)
            table = _predict_table(
                ["--log", "none", "predict", _AIRQUALITY, model_path]
            )
        self.assertEqual(table.num_rows, 153)


class TestPredictFlags(unittest.TestCase):
    """The optional prediction outputs and how they compose."""

    __slots__ = ()

    def test_node_adds_the_node_identifier_column(self):
        """The node flag reports which node each row lands in."""
        with tempfile.TemporaryDirectory() as directory:
            tree = _fit_from_file(_AIRQUALITY, "regression", "Temp")
            model_path = os.path.join(directory, "model.pkl")
            _write_pickle(tree, model_path)
            table = _predict_table(
                ["--log", "none", "predict", _AIRQUALITY, model_path, "--node"]
            )
            reloaded = sigma.cli._load_data(_AIRQUALITY)
            projected = sigma.cli._normalize_predict_columns(reloaded, tree)
            frame = sigma.cli._to_pandas(projected)
            expected = tree.apply(frame)
        self.assertEqual(table.column_names, ["prediction", "node_id"])
        numpy.testing.assert_array_equal(
            numpy.asarray(table.column("node_id").to_pylist()), expected
        )

    def test_with_input_prepends_every_input_column(self):
        """The input columns are echoed ahead of the prediction."""
        with tempfile.TemporaryDirectory() as directory:
            tree = _fit_from_file(_AIRQUALITY, "regression", "Temp")
            model_path = os.path.join(directory, "model.pkl")
            _write_pickle(tree, model_path)
            table = _predict_table(
                [
                    "--log",
                    "none",
                    "predict",
                    _AIRQUALITY,
                    model_path,
                    "--with-input",
                ]
            )
        expected = [
            "rownames",
            "Ozone",
            "Solar.R",
            "Wind",
            "Temp",
            "Month",
            "Day",
            "prediction",
        ]
        self.assertEqual(table.column_names, expected)

    def test_with_input_refuses_a_colliding_column_name(self):
        """An input column named like an output column is refused."""
        with tempfile.TemporaryDirectory() as directory:
            frame = pandas.DataFrame(
                {"x": numpy.arange(40.0), "prediction": numpy.arange(40.0)}
            )
            response = numpy.tile([0.0, 1.0], 20)
            tree = sigma.RegressionTree(ci_coverage=None).fit(frame, response)
            model_path = os.path.join(directory, "model.pkl")
            _write_pickle(tree, model_path)
            path = os.path.join(directory, "data.csv")
            table = pyarrow.table(
                {"x": numpy.arange(40.0), "prediction": numpy.arange(40.0)}
            )
            sigma.cli._save_data(table, path)
            status, _, errors = _run_and_capture(
                ["--log", "none", "predict", path, model_path, "--with-input"]
            )
        self.assertEqual(status, 1)
        self.assertIn("prediction", errors)

    def test_proba_writes_one_column_per_class(self):
        """Class probabilities are named after the fitted class labels."""
        with tempfile.TemporaryDirectory() as directory:
            tree = _fit_from_file(_GLAUCOMA, "classification", "Class")
            model_path = os.path.join(directory, "model.pkl")
            _write_pickle(tree, model_path)
            table = _predict_table(
                ["--log", "none", "predict", _GLAUCOMA, model_path, "--proba"]
            )
        expected = ["prediction", "glaucoma", "normal"]
        self.assertEqual(table.column_names, expected)
        first = table.column("glaucoma").to_pylist()
        second = table.column("normal").to_pylist()
        for left, right in zip(first, second, strict=True):
            self.assertAlmostEqual(left + right, 1.0)

    def test_proba_on_a_non_classification_model_is_refused(self):
        """The probability flag names the model type it does not apply to."""
        with tempfile.TemporaryDirectory() as directory:
            tree = _fit_from_file(_AIRQUALITY, "regression", "Temp")
            model_path = os.path.join(directory, "model.pkl")
            _write_pickle(tree, model_path)
            status, _, errors = _run_and_capture(
                ["--log", "none", "predict", _AIRQUALITY, model_path, "--proba"]
            )
        self.assertEqual(status, 1)
        self.assertIn("--proba", errors)

    def test_rank_writes_one_column_per_item(self):
        """Expected ranks are named after the fitted item names."""
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ranking.csv")
            _write_ranking_csv(path)
            tree = _fit_from_file(
                path, "ranking", "i1,i2,i3", ["--ci-coverage", "none"]
            )
            model_path = os.path.join(directory, "model.pkl")
            _write_pickle(tree, model_path)
            table = _predict_table(
                ["--log", "none", "predict", path, model_path, "--rank"]
            )
        self.assertEqual(table.column_names, ["prediction", "i1", "i2", "i3"])

    def test_rank_on_a_non_ranking_model_is_refused(self):
        """The rank flag names the model type it does not apply to."""
        with tempfile.TemporaryDirectory() as directory:
            tree = _fit_from_file(_AIRQUALITY, "regression", "Temp")
            model_path = os.path.join(directory, "model.pkl")
            _write_pickle(tree, model_path)
            status, _, errors = _run_and_capture(
                ["--log", "none", "predict", _AIRQUALITY, model_path, "--rank"]
            )
        self.assertEqual(status, 1)
        self.assertIn("--rank", errors)

    def test_times_writes_one_survival_column_per_time(self):
        """Survival probabilities are named after the requested times."""
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "survival.csv")
            _write_survival_csv(path)
            tree = _fit_from_file(path, "survival", "time,event")
            model_path = os.path.join(directory, "model.pkl")
            _write_pickle(tree, model_path)
            table = _predict_table(
                ["--log", "none", "predict", path, model_path, "--times", "2,5"]
            )
        expected = ["prediction", "survival_2", "survival_5"]
        self.assertEqual(table.column_names, expected)
        first = table.column("survival_2").to_pylist()
        second = table.column("survival_5").to_pylist()
        for left, right in zip(first, second, strict=True):
            self.assertGreaterEqual(left, right)

    def test_times_on_a_non_survival_model_is_refused(self):
        """The times flag names the model type it does not apply to."""
        with tempfile.TemporaryDirectory() as directory:
            tree = _fit_from_file(_AIRQUALITY, "regression", "Temp")
            model_path = os.path.join(directory, "model.pkl")
            _write_pickle(tree, model_path)
            status, _, errors = _run_and_capture(
                [
                    "--log",
                    "none",
                    "predict",
                    _AIRQUALITY,
                    model_path,
                    "--times",
                    "1,2",
                ]
            )
        self.assertEqual(status, 1)
        self.assertIn("--times", errors)

    def test_node_and_with_input_compose(self):
        """The input echo, the prediction, and the node identifier line up."""
        with tempfile.TemporaryDirectory() as directory:
            tree = _fit_from_file(_AIRQUALITY, "regression", "Temp")
            model_path = os.path.join(directory, "model.pkl")
            _write_pickle(tree, model_path)
            table = _predict_table(
                [
                    "--log",
                    "none",
                    "predict",
                    _AIRQUALITY,
                    model_path,
                    "--with-input",
                    "--node",
                ]
            )
        self.assertEqual(table.column_names[0], "rownames")
        self.assertEqual(table.column_names[-2:], ["prediction", "node_id"])


class TestPredictionFidelity(unittest.TestCase):
    """Predictions taken through a file match an in-process predict."""

    __slots__ = ()

    def test_regression_predictions_match_an_in_process_predict(self):
        """A regression round trip reproduces the library result exactly."""
        self._assert_round_trip(_AIRQUALITY, "regression", "Temp", ())

    def test_classification_predictions_match_an_in_process_predict(self):
        """A classification round trip reproduces the library result exactly."""
        self._assert_round_trip(_GLAUCOMA, "classification", "Class", ())

    def test_a_categorical_feature_survives_a_csv_round_trip(self):
        """Text features predict identically after a file round trip."""
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "data.csv")
            table = pyarrow.table(
                {
                    "g": numpy.tile(["a", "b"], 30).tolist(),
                    "x": numpy.arange(60.0),
                    "y": numpy.tile([0.0, 10.0], 30),
                }
            )
            sigma.cli._save_data(table, path)
            self._assert_round_trip(path, "regression", "y", ())

    def _assert_round_trip(self, data_file, task, target, flags):
        """Compare command line predictions against a direct predict call."""
        with tempfile.TemporaryDirectory() as directory:
            tree = _fit_from_file(data_file, task, target, flags)
            model_path = os.path.join(directory, "model.pkl")
            _write_pickle(tree, model_path)
            table = _predict_table(
                ["--log", "none", "predict", data_file, model_path]
            )
            column_types = sigma.cli._predict_column_types(tree)
            reloaded = sigma.cli._load_data(data_file, column_types)
            projected = sigma.cli._normalize_predict_columns(reloaded, tree)
            frame = sigma.cli._to_pandas(projected)
            expected = tree.predict(frame)
        produced = table.column("prediction").to_pylist()
        numpy.testing.assert_array_equal(numpy.asarray(produced), expected)


class TestRunExport(unittest.TestCase):
    """Rendering a fitted tree through the export subcommand."""

    __slots__ = ()

    def test_export_defaults_to_text_on_standard_output(self):
        """With no flags the rendering matches export_text."""
        with tempfile.TemporaryDirectory() as directory:
            tree = _fit_from_file(_AIRQUALITY, "regression", "Temp")
            model_path = os.path.join(directory, "model.pkl")
            _write_pickle(tree, model_path)
            status, payload, _ = _run_and_capture(
                ["--log", "none", "export", model_path]
            )
            expected = sigma.export_text(tree)
        self.assertEqual(status, 0)
        self.assertEqual(payload.decode(), expected)

    def test_export_writes_sql_when_the_format_says_so(self):
        """The sql format matches export_sql."""
        with tempfile.TemporaryDirectory() as directory:
            tree = _fit_from_file(_AIRQUALITY, "regression", "Temp")
            model_path = os.path.join(directory, "model.pkl")
            _write_pickle(tree, model_path)
            status, payload, _ = _run_and_capture(
                ["--log", "none", "export", model_path, "--format", "sql"]
            )
            expected = sigma.export_sql(tree)
        self.assertEqual(status, 0)
        self.assertEqual(payload.decode(), expected)

    def test_the_output_extension_selects_the_rendering(self):
        """A .sql output path produces SQL without an explicit format."""
        with tempfile.TemporaryDirectory() as directory:
            tree = _fit_from_file(_AIRQUALITY, "regression", "Temp")
            model_path = os.path.join(directory, "model.pkl")
            _write_pickle(tree, model_path)
            out_path = os.path.join(directory, "tree.sql")
            status = sigma.cli.run(
                ["--log", "none", "export", model_path, "--output", out_path]
            )
            with open(out_path, encoding="utf-8") as file_handle:
                contents = file_handle.read()
            expected = sigma.export_sql(tree)
        self.assertEqual(status, 0)
        self.assertEqual(contents, expected)

    def test_the_explicit_format_overrides_the_output_extension(self):
        """An explicit format wins over the extension of the output path."""
        with tempfile.TemporaryDirectory() as directory:
            tree = _fit_from_file(_AIRQUALITY, "regression", "Temp")
            model_path = os.path.join(directory, "model.pkl")
            _write_pickle(tree, model_path)
            out_path = os.path.join(directory, "tree.sql")
            sigma.cli.run(
                [
                    "--log",
                    "none",
                    "export",
                    model_path,
                    "--output",
                    out_path,
                    "--format",
                    "text",
                ]
            )
            with open(out_path, encoding="utf-8") as file_handle:
                contents = file_handle.read()
            expected = sigma.export_text(tree)
        self.assertEqual(contents, expected)

    def test_max_depth_and_precision_reach_the_renderer(self):
        """The depth and precision flags match the export_text arguments."""
        with tempfile.TemporaryDirectory() as directory:
            tree = _fit_from_file(_AIRQUALITY, "regression", "Temp")
            model_path = os.path.join(directory, "model.pkl")
            _write_pickle(tree, model_path)
            status, payload, _ = _run_and_capture(
                [
                    "--log",
                    "none",
                    "export",
                    model_path,
                    "--max-depth",
                    "1",
                    "--precision",
                    "1",
                ]
            )
            expected = sigma.export_text(tree, max_depth=1, precision=1)
        self.assertEqual(status, 0)
        self.assertEqual(payload.decode(), expected)

    def test_the_target_class_reaches_the_sql_renderer(self):
        """The target class flag selects which class the SQL returns."""
        with tempfile.TemporaryDirectory() as directory:
            tree = _fit_from_file(_GLAUCOMA, "classification", "Class")
            model_path = os.path.join(directory, "model.pkl")
            _write_pickle(tree, model_path)
            status, payload, _ = _run_and_capture(
                [
                    "--log",
                    "none",
                    "export",
                    model_path,
                    "--format",
                    "sql",
                    "--target-class",
                    "glaucoma",
                ]
            )
            expected = sigma.export_sql(tree, target_class="glaucoma")
        self.assertEqual(status, 0)
        self.assertEqual(payload.decode(), expected)

    def test_an_unknown_target_class_lists_the_fitted_classes(self):
        """A class the model never saw names the classes it did."""
        with tempfile.TemporaryDirectory() as directory:
            tree = _fit_from_file(_GLAUCOMA, "classification", "Class")
            model_path = os.path.join(directory, "model.pkl")
            _write_pickle(tree, model_path)
            status, _, errors = _run_and_capture(
                [
                    "--log",
                    "none",
                    "export",
                    model_path,
                    "--format",
                    "sql",
                    "--target-class",
                    "absent",
                ]
            )
        self.assertEqual(status, 1)
        self.assertIn("available classes", errors)

    def test_the_target_class_on_a_regression_model_is_refused(self):
        """The target class flag names the model type it does not apply to."""
        with tempfile.TemporaryDirectory() as directory:
            tree = _fit_from_file(_AIRQUALITY, "regression", "Temp")
            model_path = os.path.join(directory, "model.pkl")
            _write_pickle(tree, model_path)
            status, _, errors = _run_and_capture(
                [
                    "--log",
                    "none",
                    "export",
                    model_path,
                    "--format",
                    "sql",
                    "--target-class",
                    "a",
                ]
            )
        self.assertEqual(status, 1)
        self.assertIn("--target-class", errors)

    def test_sql_export_of_a_ranking_tree_is_refused(self):
        """Ranking trees have no SQL rendering, and say so on one line."""
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "ranking.csv")
            _write_ranking_csv(path)
            tree = _fit_from_file(
                path, "ranking", "i1,i2,i3", ["--ci-coverage", "none"]
            )
            model_path = os.path.join(directory, "model.pkl")
            _write_pickle(tree, model_path)
            status, _, errors = _run_and_capture(
                ["--log", "none", "export", model_path, "--format", "sql"]
            )
        self.assertEqual(status, 1)
        self.assertNotIn("Traceback", errors)

    def test_an_image_only_flag_is_refused_for_a_text_export(self):
        """A flag belonging to another rendering exits with a usage error."""
        argv = ["export", "m.pkl", "--format", "text", "--dpi", "96"]
        with (
            unittest.mock.patch("sys.stderr", new_callable=io.StringIO),
            self.assertRaises(SystemExit) as raised,
        ):
            sigma.cli._parse_args(argv)
        self.assertEqual(raised.exception.code, 2)

    def test_the_kind_flag_is_refused_for_a_text_export(self):
        """The image contents flag exits with a usage error on text."""
        argv = ["export", "m.pkl", "--format", "text", "--kind", "response"]
        with (
            unittest.mock.patch("sys.stderr", new_callable=io.StringIO),
            self.assertRaises(SystemExit) as raised,
        ):
            sigma.cli._parse_args(argv)
        self.assertEqual(raised.exception.code, 2)

    def test_precision_is_refused_for_a_sql_export(self):
        """SQL is emitted at full precision, so the flag exits."""
        argv = ["export", "m.pkl", "--format", "sql", "--precision", "2"]
        with (
            unittest.mock.patch("sys.stderr", new_callable=io.StringIO),
            self.assertRaises(SystemExit) as raised,
        ):
            sigma.cli._parse_args(argv)
        self.assertEqual(raised.exception.code, 2)


@unittest.skipUnless(_HAS_GRAPHVIZ, "graphviz not installed")
class TestRunExportGraphviz(unittest.TestCase):
    """Renderings that need the graphviz optional dependency."""

    __slots__ = ()

    def test_the_dot_format_matches_export_graphviz(self):
        """The dot rendering is byte identical to export_graphviz."""
        with tempfile.TemporaryDirectory() as directory:
            tree = _fit_from_file(_AIRQUALITY, "regression", "Temp")
            model_path = os.path.join(directory, "model.pkl")
            _write_pickle(tree, model_path)
            status, payload, _ = _run_and_capture(
                ["--log", "none", "export", model_path, "--format", "dot"]
            )
            expected = sigma.export_graphviz(tree)
        self.assertEqual(status, 0)
        produced = _anonymize_node_names(payload.decode())
        self.assertEqual(produced, _anonymize_node_names(expected))

    def test_a_png_output_path_writes_image_bytes(self):
        """A .png output path produces a portable network graphic."""
        with tempfile.TemporaryDirectory() as directory:
            tree = _fit_from_file(_AIRQUALITY, "regression", "Temp")
            model_path = os.path.join(directory, "model.pkl")
            _write_pickle(tree, model_path)
            out_path = os.path.join(directory, "tree.png")
            status = sigma.cli.run(
                ["--log", "none", "export", model_path, "--output", out_path]
            )
            with open(out_path, "rb") as file_handle:
                magic = file_handle.read(8)
        self.assertEqual(status, 0)
        self.assertEqual(magic, b"\x89PNG\r\n\x1a\n")


class TestErrorReporting(unittest.TestCase):
    """One-line error reporting and the exit status of each failure class."""

    __slots__ = ()

    def test_a_missing_data_file_names_the_path(self):
        """A data file that does not exist is reported on one line."""
        status, _, errors = _run_and_capture(
            ["--log", "none", "fit", "absent.csv", "regression", "y", "m.pkl"]
        )
        self.assertEqual(status, 1)
        self.assertIn("data file not found: absent.csv", errors)

    def test_a_missing_model_file_names_the_path(self):
        """A model file that does not exist is reported on one line."""
        status, _, errors = _run_and_capture(
            ["--log", "none", "predict", _AIRQUALITY, "absent.pkl"]
        )
        self.assertEqual(status, 1)
        self.assertIn("model file not found: absent.pkl", errors)

    def test_a_file_that_is_not_a_sigma_model_is_refused(self):
        """A pickle holding something else names what it holds."""
        with tempfile.TemporaryDirectory() as directory:
            model_path = os.path.join(directory, "model.pkl")
            with open(model_path, "wb") as file_handle:
                pickle.dump({"not": "a tree"}, file_handle)
            status, _, errors = _run_and_capture(
                ["--log", "none", "predict", _AIRQUALITY, model_path]
            )
        self.assertEqual(status, 1)
        self.assertIn("is not a sigma model file", errors)

    def test_loading_a_file_that_is_not_a_model_raises_a_type_error(self):
        """The loader reports a pickle of the wrong type as a TypeError."""
        with tempfile.TemporaryDirectory() as directory:
            model_path = os.path.join(directory, "model.pkl")
            with open(model_path, "wb") as file_handle:
                pickle.dump({"not": "a tree"}, file_handle)
            with self.assertRaises(TypeError):
                sigma.cli._load_model(model_path)

    def test_a_corrupt_model_file_is_refused_without_a_traceback(self):
        """Unpicklable bytes produce one line, not a stack trace."""
        with tempfile.TemporaryDirectory() as directory:
            model_path = os.path.join(directory, "model.pkl")
            with open(model_path, "wb") as file_handle:
                file_handle.write(b"not a pickle at all")
            status, _, errors = _run_and_capture(
                ["--log", "none", "predict", _AIRQUALITY, model_path]
            )
        self.assertEqual(status, 1)
        self.assertNotIn("Traceback", errors)

    def test_every_error_line_carries_the_error_prefix(self):
        """User errors are prefixed with error and nothing else."""
        status, _, errors = _run_and_capture(
            ["--log", "none", "fit", "absent.csv", "regression", "y", "m.pkl"]
        )
        lines = errors.splitlines()
        self.assertEqual(status, 1)
        self.assertEqual(len(lines), 1)
        self.assertTrue(lines[0].startswith("error: "))

    def test_a_user_error_prints_no_traceback(self):
        """The stack trace is withheld for an expected failure."""
        _, _, errors = _run_and_capture(
            ["--log", "none", "fit", "absent.csv", "regression", "y", "m.pkl"]
        )
        self.assertNotIn("Traceback", errors)

    def test_the_debug_log_level_restores_the_traceback(self):
        """Asking for debug output brings the stack trace back."""
        _, _, errors = _run_and_capture(
            ["--log", "debug", "fit", "absent.csv", "regression", "y", "m.pkl"]
        )
        self.assertIn("Traceback", errors)

    def test_an_unexpected_failure_is_reported_as_an_internal_error(self):
        """A failure outside the expected set keeps its stack trace."""
        with unittest.mock.patch(
            "sigma.cli.run_fit", side_effect=RuntimeError("boom")
        ):
            status, _, errors = _run_and_capture(
                [
                    "--log",
                    "none",
                    "fit",
                    _AIRQUALITY,
                    "regression",
                    "Temp",
                    "m.pkl",
                ]
            )
        self.assertEqual(status, 1)
        self.assertIn("internal error:", errors)
        self.assertIn("Traceback", errors)

    def test_a_successful_command_returns_zero(self):
        """A completed fit reports success."""
        with tempfile.TemporaryDirectory() as directory:
            model_path = os.path.join(directory, "model.pkl")
            status = sigma.cli.run(
                [
                    "--log",
                    "none",
                    "fit",
                    _AIRQUALITY,
                    "regression",
                    "Temp",
                    model_path,
                ]
            )
        self.assertEqual(status, 0)

    def test_a_bad_command_line_exits_with_status_two(self):
        """Argparse usage failures keep their conventional status."""
        with (
            unittest.mock.patch("sys.stderr", new_callable=io.StringIO),
            self.assertRaises(SystemExit) as raised,
        ):
            sigma.cli.run(["fit", "only-one-argument"])
        self.assertEqual(raised.exception.code, 2)

    def test_an_interrupted_command_returns_one_hundred_thirty(self):
        """A keyboard interrupt reports the conventional status."""
        with unittest.mock.patch(
            "sigma.cli.run_fit", side_effect=KeyboardInterrupt
        ):
            status, _, errors = _run_and_capture(
                [
                    "--log",
                    "none",
                    "fit",
                    _AIRQUALITY,
                    "regression",
                    "Temp",
                    "m.pkl",
                ]
            )
        self.assertEqual(status, 130)
        self.assertNotIn("Traceback", errors)

    def test_a_closed_output_pipe_returns_one_hundred_forty_one(self):
        """A reader that goes away leaves no error text behind."""
        with (
            unittest.mock.patch(
                "sigma.cli.run_predict", side_effect=BrokenPipeError
            ),
            unittest.mock.patch("sigma.cli._silence_stdout"),
        ):
            status, _, errors = _run_and_capture(
                ["--log", "none", "predict", _AIRQUALITY, "m.pkl"]
            )
        self.assertEqual(status, 141)
        self.assertEqual(errors, "")

    def test_a_cross_version_model_warns_on_one_line(self):
        """A model saved by another version reports a warning, not a trace."""
        with tempfile.TemporaryDirectory() as directory:
            tree = _fit_from_file(_AIRQUALITY, "regression", "Temp")
            model_path = os.path.join(directory, "model.pkl")
            with unittest.mock.patch("sigma._tree._SIGMA_VERSION", "0.0.0"):
                _write_pickle(tree, model_path)
            status, _, errors = _run_and_capture(
                ["--log", "none", "predict", _AIRQUALITY, model_path]
            )
        self.assertEqual(status, 0)
        self.assertIn("warning: ", errors)
        self.assertNotIn("Traceback", errors)


class TestModuleEntryPoint(unittest.TestCase):
    """Running the package as a module."""

    __slots__ = ()

    def test_the_module_reports_the_command_status(self):
        """python -m sigma propagates the status and the one-line error."""
        completed = subprocess.run(
            [
                sys.executable,
                "-m",
                "sigma",
                "--log",
                "none",
                "predict",
                "absent.csv",
                "absent.pkl",
            ],
            capture_output=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 1)
        self.assertNotIn(b"Traceback", completed.stderr)
        self.assertTrue(completed.stderr.startswith(b"error: "))


class TestHelpTextDefaults(unittest.TestCase):
    """The documented flag defaults cannot drift from the constructors."""

    __slots__ = ()

    def test_every_flag_names_a_parameter_of_every_task_it_serves(self):
        """Each flag maps onto a real constructor parameter."""
        for hyperparameter in sigma.cli._HYPERPARAMETERS:
            for task in hyperparameter.tasks:
                with self.subTest(flag=hyperparameter.flag, task=task):
                    parameters = _constructor_parameters(task)
                    self.assertIn(hyperparameter.parameter, parameters)

    def test_every_constructor_parameter_is_exposed_or_excluded(self):
        """No estimator parameter is silently absent from the interface."""
        exposed = {
            hyperparameter.parameter
            for hyperparameter in sigma.cli._HYPERPARAMETERS
        }
        declared = exposed | set(sigma.cli._UNEXPOSED)
        for task in sigma.cli._TASKS:
            with self.subTest(task=task):
                parameters = set(_constructor_parameters(task))
                self.assertEqual(parameters - declared, set())

    def test_every_help_string_states_a_default(self):
        """No flag can dodge the default-drift check."""
        for hyperparameter in sigma.cli._HYPERPARAMETERS:
            with self.subTest(flag=hyperparameter.flag):
                self.assertRegex(hyperparameter.help, r"\(default: [^)]+\)")

    def test_each_stated_default_matches_the_constructor(self):
        """The documented default is the one the estimator actually applies."""
        for hyperparameter in sigma.cli._HYPERPARAMETERS:
            if hyperparameter.parameter in sigma.cli._PER_TASK_DEFAULTS:
                continue
            with self.subTest(flag=hyperparameter.flag):
                defaults = {
                    _constructor_parameters(task)[hyperparameter.parameter]
                    for task in hyperparameter.tasks
                }
                self.assertEqual(len(defaults), 1)
                value = defaults.pop()
                rendered = sigma.cli._format_default(value)
                self.assertIn(f"(default: {rendered})", hyperparameter.help)

    def test_a_per_task_default_is_named_for_every_task_it_differs_on(self):
        """A parameter with several defaults states each of them."""
        for parameter in sigma.cli._PER_TASK_DEFAULTS:
            with self.subTest(parameter=parameter):
                matches = [
                    hyperparameter
                    for hyperparameter in sigma.cli._HYPERPARAMETERS
                    if hyperparameter.parameter == parameter
                ]
                hyperparameter = matches[0]
                for task in hyperparameter.tasks:
                    value = _constructor_parameters(task)[parameter]
                    rendered = sigma.cli._format_default(value)
                    self.assertIn(rendered, hyperparameter.help)


def _constructor_parameters(task):
    """Map each constructor parameter of a task to its default value."""
    estimator_class = sigma.cli._ESTIMATORS[task]
    signature = inspect.signature(estimator_class.__init__)
    defaults = {}
    for name, parameter in signature.parameters.items():
        if name != "self":
            defaults[name] = parameter.default
    return defaults


if __name__ == "__main__":
    unittest.main()
