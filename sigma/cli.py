"""Command line interface for sigma.

Three subcommands operate on tabular files and pickled trees:

- fit: read a data file, fit one of the four estimators, and write the
  fitted tree as a pickle.
- predict: read a data file and a fitted tree, and write the predictions
  as a table.
- export: read a fitted tree and render it as text, SQL, Graphviz DOT, or
  an image.

pyarrow and pandas are required, and are installed by the cli extra.
"""

from __future__ import annotations

import argparse
import collections.abc
import enum
import json
import logging
import math
import os
import os.path
import pickle
import sys
import traceback
import typing
import warnings

import numpy
import numpy.typing

from . import __version__ as version
from . import (
    _export,
    _tree,
    _tree_classification,
    _tree_ranking,
    _tree_regression,
    _tree_survival,
    _types,
)

if typing.TYPE_CHECKING:
    import pandas
    import pyarrow

logger = logging.getLogger(__name__)
"""Module-level logger for command line progress messages."""

_OUTPUT_FORMATS = (
    "csv",
    "tsv",
    "parquet",
    "arrow",
    "feather",
    "orc",
    "jsonl",
    "ndjson",
    "md",
)
"""Table formats the interface writes."""

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
"""File extensions the interface reads."""

_TASKS = ("regression", "classification", "survival", "ranking")
"""Estimator families the fit subcommand accepts."""

_EXPORT_FORMATS = ("text", "sql", "dot", "png", "svg", "pdf", "gif")
"""Renderings the export subcommand produces."""

_EXPORT_EXTENSIONS = {
    ".txt": "text",
    ".sql": "sql",
    ".dot": "dot",
    ".gv": "dot",
    ".png": "png",
    ".svg": "svg",
    ".pdf": "pdf",
    ".gif": "gif",
}
"""Export format selected by each output file extension."""

_IMAGE_FORMATS = ("png", "svg", "pdf", "gif")
"""Export formats rendered as bytes rather than text."""

_TextRenderer = collections.abc.Callable[..., str]
"""Signature of the export functions returning a rendering as text."""

_ImageRenderer = collections.abc.Callable[..., bytes]
"""Signature of the export function returning a rendering as bytes."""

_ESTIMATORS = {
    "regression": _tree_regression.RegressionTree,
    "classification": _tree_classification.ClassificationTree,
    "survival": _tree_survival.SurvivalTree,
    "ranking": _tree_ranking.RankingTree,
}
"""Estimator class fitted by each task name."""

_Estimator = (
    _tree_regression.RegressionTree
    | _tree_classification.ClassificationTree
    | _tree_survival.SurvivalTree
    | _tree_ranking.RankingTree
)
"""Fitted tree the interface writes, loads, predicts with, and renders."""

_CI_METHOD_ENUMS = {
    "regression": _types.CiMethodRegressionTree,
    "classification": _types.CiMethodClassificationTree,
    "survival": _types.CiMethodSurvival,
    "ranking": _types.CiMethodRankingTree,
}
"""Confidence interval vocabulary accepted by each task."""


def run(argv: None | collections.abc.Sequence[str] = None) -> int:
    """Run the sigma command line interface.

    Args:
        argv: Command-line arguments; None reads sys.argv.

    Returns:
        Exit code, where 0 signals success, 1 a failed command, 130 an
        interrupted command, and 141 an output pipe closed by its reader.

    Raises:
        SystemExit: If the command line itself is malformed, with code 2.
    """
    args = _parse_args(argv)
    _configure_logging(args.log)
    logger.info("Sigma v%s", version)
    expected = _expected_errors()
    try:
        status = _dispatch(args)
        return status
    except BrokenPipeError:
        _silence_stdout()
        return 141
    except KeyboardInterrupt:
        logger.info("Interrupted.")
        return 130
    except expected as exception:
        message = str(exception)
        print(f"error: {message}", file=sys.stderr)
        if args.log == "debug":
            traceback.print_exc(file=sys.stderr)
        return 1
    except Exception as exception:
        name = type(exception).__name__
        message = str(exception)
        logger.exception("Failed.")
        print(f"internal error: {name}: {message}", file=sys.stderr)
        if args.log == "none":
            traceback.print_exc(file=sys.stderr)
        return 1


def _parse_args(
    args: None | collections.abc.Sequence[str] = None,
) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="sigma",
        description="Sigma: conditional inference trees for tabular data",
        allow_abbrev=False,
    )
    parser.add_argument(
        "--log",
        choices=["none", "debug", "info", "warning", "error"],
        default="info",
        type=str.lower,
        help="Set logging level (default: info)",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"sigma {version}",
        help="Print the installed sigma version and exit",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    fit_parser = subparsers.add_parser(
        "fit",
        help="Fit a tree on training data",
        allow_abbrev=False,
    )
    _add_fit_arguments(fit_parser)
    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict with a fitted tree",
        allow_abbrev=False,
    )
    _add_predict_arguments(predict_parser)
    export_parser = subparsers.add_parser(
        "export",
        help="Render a fitted tree",
        allow_abbrev=False,
    )
    _add_export_arguments(export_parser)
    parsed_args = parser.parse_args(args)
    command = typing.cast(
        typing.Literal["fit", "predict", "export"], parsed_args.command
    )
    match command:
        case "fit":
            _validate_fit_flags(fit_parser, parsed_args)
        case "export":
            _validate_export_flags(export_parser, parsed_args)
        case "predict":
            pass
        case _:
            typing.assert_never(command)
    return parsed_args


def _add_fit_arguments(parser: argparse.ArgumentParser) -> None:
    """Declare the positional arguments and flags of the fit subcommand."""
    parser.add_argument(
        "data_file",
        type=str,
        help="Path to the training data file",
    )
    parser.add_argument(
        "task",
        type=str.lower,
        choices=_TASKS,
        help="Estimator family to fit",
    )
    parser.add_argument(
        "target_columns",
        type=_column_list,
        help="Comma-separated target column names: one for regression and"
        " classification, time,event for survival, one per item for ranking",
    )
    parser.add_argument(
        "model_file",
        type=str,
        help="Path where the fitted tree will be saved",
    )
    parser.add_argument(
        "--sample-weight",
        type=str,
        default=argparse.SUPPRESS,
        help="Column holding the case weights; it is not used as a feature",
    )
    for hyperparameter in _HYPERPARAMETERS:
        parser.add_argument(
            hyperparameter.flag,
            dest=hyperparameter.parameter,
            default=argparse.SUPPRESS,
            help=hyperparameter.help,
            **hyperparameter.options,
        )


def _add_predict_arguments(parser: argparse.ArgumentParser) -> None:
    """Declare the positional arguments and flags of the predict subcommand."""
    parser.add_argument(
        "data_file",
        type=str,
        help="Path to the data file to predict on",
    )
    parser.add_argument(
        "model_file",
        type=str,
        help="Path to the fitted tree file",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path where predictions will be saved (default: stdout)",
    )
    parser.add_argument(
        "--output-format",
        type=str.lower,
        default=None,
        choices=_OUTPUT_FORMATS,
        help="Force the output format (default: inferred from the file"
        " extension, or csv for stdout)",
    )
    parser.add_argument(
        "--proba",
        action="store_true",
        help="Add one class probability column per class (classification only)",
    )
    parser.add_argument(
        "--rank",
        action="store_true",
        help="Add one expected rank column per item (ranking only)",
    )
    parser.add_argument(
        "--times",
        type=_time_list,
        default=None,
        help="Comma-separated non-decreasing times; adds one survival"
        " probability column per time (survival only)",
    )
    parser.add_argument(
        "--node",
        action="store_true",
        help="Add the identifier of the node each row lands in",
    )
    parser.add_argument(
        "--with-input",
        action="store_true",
        help="Prepend the columns of the input file",
    )


def _add_export_arguments(parser: argparse.ArgumentParser) -> None:
    """Declare the positional argument and flags of the export subcommand."""
    parser.add_argument(
        "model_file",
        type=str,
        help="Path to the fitted tree file",
    )
    parser.add_argument(
        "--format",
        type=str.lower,
        default=None,
        choices=_EXPORT_FORMATS,
        help="Rendering to produce (default: inferred from the output file"
        " extension, or text for stdout)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path where the rendering will be saved (default: stdout)",
    )
    parser.add_argument(
        "--kind",
        type=str.lower,
        choices=["tree", "response"],
        default=argparse.SUPPRESS,
        help="Image contents: the tree or the leaf response distributions"
        " (default: tree)",
    )
    parser.add_argument(
        "--max-depth",
        type=_optional_non_negative_integer,
        default=argparse.SUPPRESS,
        help="Deepest level to render, or none for the whole tree"
        " (default: none)",
    )
    parser.add_argument(
        "--precision",
        type=_non_negative_integer,
        default=argparse.SUPPRESS,
        help="Number of decimals shown for predicted values (default: 3)",
    )
    parser.add_argument(
        "--top-displayed-items",
        type=_positive_integer,
        default=argparse.SUPPRESS,
        help="Number of ranked items shown per leaf (default: all)",
    )
    parser.add_argument(
        "--target-class",
        type=str,
        default=argparse.SUPPRESS,
        help="Class whose probability the SQL expression returns"
        " (default: the last class)",
    )
    parser.add_argument(
        "--orientation",
        type=str.lower,
        choices=["top-down", "left-to-right"],
        default=argparse.SUPPRESS,
        help="Direction the tree grows in (default: top-down)",
    )
    parser.add_argument(
        "--dpi",
        type=_positive_integer,
        default=argparse.SUPPRESS,
        help="Image resolution in dots per inch (default: 192)",
    )
    parser.add_argument(
        "--max-branch-length",
        type=_positive_integer,
        default=argparse.SUPPRESS,
        help="Characters kept on a branch label before wrapping (default: 60)",
    )


def _validate_fit_flags(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Reject a fit flag that the chosen task does not accept."""
    supplied = vars(args)
    for hyperparameter in _HYPERPARAMETERS:
        applies = args.task in hyperparameter.tasks
        if hyperparameter.parameter in supplied and not applies:
            owner = hyperparameter.tasks[0]
            parser.error(
                f"{hyperparameter.flag} applies to task {owner!r} only;"
                f" got task {args.task!r}"
            )
    if "ci_method" in supplied:
        enum_class = _CI_METHOD_ENUMS[args.task]
        values = _enum_values(enum_class)
        if args.ci_method not in values:
            listing = ", ".join(values)
            parser.error(
                f"--ci-method {args.ci_method!r} does not apply to task"
                f" {args.task!r}; its methods are {listing}"
            )


def _validate_export_flags(
    parser: argparse.ArgumentParser, args: argparse.Namespace
) -> None:
    """Reject an export flag that the chosen rendering does not accept."""
    try:
        chosen = _export_format(args.format, args.output)
    except ValueError as value_error:
        parser.error(str(value_error))
    supplied = vars(args)
    for parameter, formats in _EXPORT_FLAG_FORMATS.items():
        if parameter in supplied and chosen not in formats:
            dashed = parameter.replace("_", "-")
            listing = ", ".join(formats)
            parser.error(
                f"--{dashed} applies to the {listing} formats only;"
                f" got {chosen!r}"
            )


def _configure_logging(level: str) -> None:
    """Install a stderr log handler, or silence logging altogether."""
    if level == "none":
        logging.disable(logging.CRITICAL)
        return
    logging.disable(logging.NOTSET)
    upper_level = level.upper()
    numeric_level = getattr(logging, upper_level)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        force=True,
    )


def _expected_errors() -> tuple[type[BaseException], ...]:
    """List the exception types reported as a single error line."""
    errors: list[type[BaseException]] = [
        ValueError,
        TypeError,
        AttributeError,
        LookupError,
        OSError,
        ImportError,
        NotImplementedError,
        pickle.PickleError,
        EOFError,
    ]
    pyarrow = sys.modules.get("pyarrow")
    if pyarrow is not None:
        errors.append(pyarrow.ArrowException)
    return tuple(errors)


def _dispatch(args: argparse.Namespace) -> int:
    """Run the subcommand the parsed arguments name."""
    command = typing.cast(
        typing.Literal["fit", "predict", "export"], args.command
    )
    match command:
        case "fit":
            parameters = _estimator_kwargs(args)
            weight = getattr(args, "sample_weight", None)
            status = run_fit(
                args.data_file,
                args.task,
                args.target_columns,
                args.model_file,
                weight,
                parameters,
            )
        case "predict":
            status = run_predict(
                args.data_file,
                args.model_file,
                args.output,
                args.output_format,
                args.proba,
                args.rank,
                args.times,
                args.node,
                args.with_input,
            )
        case "export":
            parameters = _export_kwargs(args)
            status = run_export(
                args.model_file, args.format, args.output, parameters
            )
        case _:
            typing.assert_never(command)
    return status


def _estimator_kwargs(args: argparse.Namespace) -> dict[str, object]:
    """Collect the constructor keywords the command line supplied."""
    supplied = vars(args)
    keywords: dict[str, object] = {}
    for hyperparameter in _HYPERPARAMETERS:
        if hyperparameter.parameter in supplied:
            keywords[hyperparameter.parameter] = supplied[
                hyperparameter.parameter
            ]
    return keywords


def _export_kwargs(args: argparse.Namespace) -> dict[str, object]:
    """Collect the rendering keywords the command line supplied."""
    supplied = vars(args)
    keywords: dict[str, object] = {}
    for parameter in _EXPORT_FLAG_FORMATS:
        if parameter in supplied:
            keywords[parameter] = supplied[parameter]
    return keywords


def _silence_stdout() -> None:
    """Point standard output at the null device after its reader went away."""
    devnull = os.open(os.devnull, os.O_WRONLY)
    descriptor = sys.stdout.fileno()
    os.dup2(devnull, descriptor)


def run_fit(
    data_file: str,
    task: str,
    target_columns: collections.abc.Sequence[str],
    model_file: str,
    sample_weight: None | str = None,
    parameters: None | dict[str, object] = None,
) -> int:
    """Fit a tree on a data file and save it as a pickle.

    Args:
        data_file: Path to the training data file.
        task: Estimator family, one of regression, classification,
            survival, and ranking.
        target_columns: Response column names, in the order the estimator
            reads them.
        model_file: Path where the fitted tree is written.
        sample_weight: Column holding the case weights, or None. The
            column is not used as a feature.
        parameters: Estimator constructor keywords, or None to accept
            every default.

    Returns:
        Exit code, where 0 signals success.

    Raises:
        FileNotFoundError: If the data file does not exist.
        ValueError: If a named column is absent, if the number of target
            columns does not suit the task, if a target column holds
            missing values, or if a feature column has a type sigma
            cannot use.
    """
    keywords = {} if parameters is None else parameters
    names = list(target_columns)
    logger.info("Loading fitting data from %s", data_file)
    table = _load_data(data_file)
    _check_target_arity(task, names)
    _check_columns_present(table, names, "target")
    if sample_weight is not None:
        _check_columns_present(table, [sample_weight], "sample weight")
    excluded = set(names)
    if sample_weight is not None:
        excluded.add(sample_weight)
    feature_names = [
        name for name in table.column_names if name not in excluded
    ]
    if not feature_names:
        raise ValueError(
            "the data file holds no feature column once the target columns"
            " are removed"
        )
    features = table.select(feature_names)
    normalized = _normalize_fit_columns(features)
    X = _to_pandas(normalized)
    y = _build_response(table, task, names)
    weights = _build_sample_weight(table, sample_weight)
    estimator_class = _ESTIMATORS[task]
    # The parser validated each keyword against the constructor it feeds.
    factory = typing.cast(
        collections.abc.Callable[..., _Estimator], estimator_class
    )
    estimator = factory(**keywords)
    logger.info(
        "Fitting a %s tree on %s rows with %s features",
        task,
        len(X),
        len(feature_names),
    )
    tree = estimator.fit(X, y, sample_weight=weights)
    logger.info("Saving the fitted tree to %s", model_file)
    with open(model_file, "wb") as file_handle:
        pickle.dump(tree, file_handle)
    logger.info("Fitting completed successfully")
    return 0


def run_predict(
    data_file: str,
    model_file: str,
    output_file: None | str = None,
    output_format: None | str = None,
    proba: bool = False,
    rank: bool = False,
    times: None | collections.abc.Sequence[float] = None,
    node: bool = False,
    with_input: bool = False,
) -> int:
    """Predict with a saved tree and write the results as a table.

    Args:
        data_file: Path to the data file to predict on. Columns the model
            was not fitted on are ignored.
        model_file: Path to the saved tree.
        output_file: Path where the table is written, or None for
            standard output.
        output_format: Table format to force, or None to read it from the
            output file extension.
        proba: Whether to add one class probability column per class.
        rank: Whether to add one expected rank column per item.
        times: Times at which to add a survival probability column, or
            None.
        node: Whether to add the identifier of the node each row reaches.
        with_input: Whether to prepend the columns of the data file.

    Returns:
        Exit code, where 0 signals success.

    Raises:
        FileNotFoundError: If the data file or the model file does not
            exist.
        TypeError: If the model file holds something other than a fitted
            tree.
        ValueError: If the data file lacks a fitted feature column, if a
            requested output does not suit the model, or if two output
            columns would carry the same name.
    """
    tree = _load_model(model_file)
    classification = _require_classification(tree) if proba else None
    ranking = _require_ranking(tree) if rank else None
    survival = None
    survival_times = numpy.empty(0, dtype=float)
    if times is not None:
        survival = _require_survival(tree)
        survival_times = numpy.asarray(times, dtype=float)
    column_types = _predict_column_types(tree)
    logger.info("Loading prediction data from %s", data_file)
    table = _load_data(data_file, column_types)
    projected = _normalize_predict_columns(table, tree)
    frame = _to_pandas(projected)
    logger.info("Predicting on %s rows", len(frame))
    columns: list[object] = []
    names: list[str] = []
    if with_input:
        for name in table.column_names:
            columns.append(table.column(name))
            names.append(name)
    predictions = tree.predict(frame)
    columns.append(_arrow_column(predictions))
    names.append("prediction")
    if classification is not None:
        matrix = classification.predict_proba(frame)
        labels = [str(label) for label in classification.classes_]
        _extend_matrix(columns, names, matrix, labels)
    if ranking is not None:
        matrix = ranking.predict_rank(frame)
        labels = [str(label) for label in ranking.item_names_]
        _extend_matrix(columns, names, matrix, labels)
    if survival is not None:
        matrix = survival.predict_survival(frame, survival_times)
        labels = [f"survival_{time:g}" for time in survival_times]
        _extend_matrix(columns, names, matrix, labels)
    if node:
        indices = tree.predict_index(frame)
        columns.append(_arrow_column(indices))
        names.append("node_id")
    _reject_duplicate_names(names)
    import pyarrow

    output_table = pyarrow.table(columns, names=names)
    target_name = "stdout" if output_file is None else output_file
    logger.info("Saving predictions to %s", target_name)
    _save_data(output_table, output_file, output_format)
    logger.info("Prediction completed successfully")
    return 0


def run_export(
    model_file: str,
    export_format: None | str = None,
    output_file: None | str = None,
    parameters: None | dict[str, object] = None,
) -> int:
    """Render a saved tree as text, SQL, Graphviz DOT, or an image.

    Args:
        model_file: Path to the saved tree.
        export_format: Rendering to produce, or None to read it from the
            output file extension, defaulting to text.
        output_file: Path where the rendering is written, or None for
            standard output.
        parameters: Rendering keywords, or None to accept every default.

    Returns:
        Exit code, where 0 signals success.

    Raises:
        FileNotFoundError: If the model file does not exist.
        ImportError: If the requested rendering needs graphviz or
            matplotlib and it is not installed.
        NotImplementedError: If a SQL rendering is asked of a ranking
            tree.
        TypeError: If the model file holds something other than a fitted
            tree.
        ValueError: If the format cannot be read from the output path, or
            if a named target class is not one the model was fitted on.
    """
    keywords = dict({} if parameters is None else parameters)
    tree = _load_model(model_file)
    chosen = _export_format(export_format, output_file)
    if "target_class" in keywords:
        keywords["target_class"] = _resolve_target_class(
            tree, keywords["target_class"]
        )
    logger.info("Rendering the tree as %s", chosen)
    # The parser validated each keyword against the renderer it feeds.
    match chosen:
        case "text":
            renderer = typing.cast(_TextRenderer, _export.export_text)
            content = renderer(tree, **keywords)
            _write_output_text(content, output_file)
        case "sql":
            renderer = typing.cast(_TextRenderer, _export.export_sql)
            content = renderer(tree, **keywords)
            _write_output_text(content, output_file)
        case "dot":
            renderer = typing.cast(_TextRenderer, _export.export_graphviz)
            content = renderer(tree, **keywords)
            _write_output_text(content, output_file)
        case _:
            image = typing.cast(_ImageRenderer, _export.export_image)
            payload = image(tree, chosen, **keywords)
            _write_output_bytes(payload, output_file)
    logger.info("Export completed successfully")
    return 0


class _FlagOptions(typing.TypedDict, total=False):
    """Argparse keywords a fit flag adds to the ones every flag shares."""

    action: str
    choices: list[str]
    type: collections.abc.Callable[[str], object]


class _Hyperparameter:
    """One fit flag and the estimator constructor parameter it sets."""

    __slots__ = ("flag", "help", "options", "parameter", "tasks")

    def __init__(
        self,
        flag: str,
        parameter: str,
        tasks: tuple[str, ...],
        help: str,
        options: _FlagOptions,
    ) -> None:
        self.flag = flag
        self.parameter = parameter
        self.tasks = tasks
        self.help = help
        self.options = options


def _enum_values(enum_class: type[enum.Enum]) -> list[str]:
    """List the string values of an enumeration, in declaration order."""
    values = [member.value for member in enum_class]
    return values


def _ci_method_values() -> list[str]:
    """List every confidence interval method name, across the four tasks."""
    values: list[str] = []
    enum_classes = (
        _types.CiMethodRegressionTree,
        _types.CiMethodClassificationTree,
        _types.CiMethodSurvival,
        _types.CiMethodRankingTree,
    )
    for enum_class in enum_classes:
        for value in _enum_values(enum_class):
            if value not in values:
                values.append(value)
    return values


def _column_list(value: str) -> list[str]:
    """Argparse type that splits a comma-separated list of column names."""
    names = value.split(",")
    for name in names:
        if not name:
            raise argparse.ArgumentTypeError(
                f"expected a comma-separated list of column names;"
                f" got {value!r}"
            )
    return names


def _time_list(value: str) -> list[float]:
    """Argparse type that splits a comma-separated non-decreasing time list."""
    times: list[float] = []
    for entry in value.split(","):
        try:
            time = float(entry)
        except ValueError as value_error:
            raise argparse.ArgumentTypeError(
                f"expected a comma-separated list of times; got {entry!r}"
            ) from value_error
        times.append(time)
    previous = times[0]
    for time in times[1:]:
        if time < previous:
            raise argparse.ArgumentTypeError(
                f"expected non-decreasing times; got {value!r}"
            )
        previous = time
    return times


def _positive_integer(value: str) -> int:
    """Argparse type that accepts only positive integers."""
    parsed = _parse_integer(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer; got {parsed}"
        )
    return parsed


def _non_negative_integer(value: str) -> int:
    """Argparse type that accepts only non-negative integers."""
    parsed = _parse_integer(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(
            f"expected a non-negative integer; got {parsed}"
        )
    return parsed


def _optional_non_negative_integer(value: str) -> None | int:
    """Argparse type that accepts a non-negative integer or the word none."""
    lowered = value.lower()
    if lowered == "none":
        return None
    parsed = _non_negative_integer(value)
    return parsed


def _parse_integer(value: str) -> int:
    """Convert an argument to an integer, or raise an argparse error."""
    try:
        parsed = int(value)
    except ValueError as value_error:
        raise argparse.ArgumentTypeError(
            f"expected an integer; got {value!r}"
        ) from value_error
    return parsed


def _positive_float(value: str) -> float:
    """Argparse type that accepts only positive finite floats."""
    parsed = _parse_float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError(
            f"expected a positive number; got {parsed}"
        )
    return parsed


def _significance_level(value: str) -> float:
    """Argparse type that accepts a significance level in (0, 1]."""
    parsed = _parse_float(value)
    if parsed <= 0.0 or parsed > 1.0:
        raise argparse.ArgumentTypeError(
            f"expected a number in (0, 1]; got {parsed}"
        )
    return parsed


def _optional_coverage(value: str) -> None | float:
    """Argparse type that accepts a coverage in (0, 1) or the word none."""
    lowered = value.lower()
    if lowered == "none":
        return None
    parsed = _parse_float(value)
    if parsed <= 0.0 or parsed >= 1.0:
        raise argparse.ArgumentTypeError(
            f"expected a number in (0, 1); got {parsed}"
        )
    return parsed


def _parse_float(value: str) -> float:
    """Convert an argument to a finite float, or raise an argparse error."""
    try:
        parsed = float(value)
    except ValueError as value_error:
        raise argparse.ArgumentTypeError(
            f"expected a number; got {value!r}"
        ) from value_error
    is_finite = math.isfinite(parsed)
    if not is_finite:
        raise argparse.ArgumentTypeError(
            f"expected a finite number; got {value!r}"
        )
    return parsed


def _metric_list(value: str) -> tuple[object, ...]:
    """Argparse type that splits a comma-separated survival metric list."""
    specifications: list[object] = []
    for entry in value.split(","):
        parts = entry.split(":")
        match parts:
            case [kind] if kind:
                specifications.append(kind)
            case [kind, amount, unit] if kind and unit:
                parsed = _parse_float(amount)
                specifications.append((kind, parsed, unit))
            case _:
                raise argparse.ArgumentTypeError(
                    f"expected a metric kind or kind:value:unit; got {entry!r}"
                )
    return tuple(specifications)


_HYPERPARAMETERS = (
    _Hyperparameter(
        "--correlation",
        "correlation",
        _TASKS,
        "Rank-transform the inputs or use their raw values (default: rank,"
        " and normal for survival)",
        {"choices": _enum_values(_types.Correlation), "type": str.lower},
    ),
    _Hyperparameter(
        "--test-stat",
        "test_stat",
        _TASKS,
        "How the multivariate score is reduced to a scalar"
        " (default: quadratic)",
        {"choices": _enum_values(_types.TestStat), "type": str.lower},
    ),
    _Hyperparameter(
        "--test-type",
        "test_type",
        _TASKS,
        "Multiplicity adjustment applied across covariates (default: sidak)",
        {"choices": _enum_values(_types.TestType), "type": str.lower},
    ),
    _Hyperparameter(
        "--alpha",
        "alpha",
        _TASKS,
        "Significance level of the stopping rule (default: 0.05)",
        {"type": _significance_level},
    ),
    _Hyperparameter(
        "--min-splits",
        "min_splits",
        _TASKS,
        "Smallest sum of weights a node may have and still be split"
        " (default: 20)",
        {"type": _positive_integer},
    ),
    _Hyperparameter(
        "--min-buckets",
        "min_buckets",
        _TASKS,
        "Smallest sum of weights each child may have (default: 7)",
        {"type": _positive_integer},
    ),
    _Hyperparameter(
        "--max-depth",
        "max_depth",
        _TASKS,
        "Deepest level the tree may reach, or none for no limit"
        " (default: none)",
        {"type": _optional_non_negative_integer},
    ),
    _Hyperparameter(
        "--categorical-features",
        "categorical_features",
        _TASKS,
        "Comma-separated numeric columns to treat as categorical"
        " (default: none)",
        {"type": _column_list},
    ),
    _Hyperparameter(
        "--ci-method",
        "ci_method",
        _TASKS,
        "Confidence interval method for the node predictions (default:"
        " bayesian_bootstrap for regression and ranking, jeffreys for"
        " classification, brookmeyer_crowley for survival)",
        {"choices": _ci_method_values(), "type": str.lower},
    ),
    _Hyperparameter(
        "--ci-coverage",
        "ci_coverage",
        _TASKS,
        "Coverage of the node confidence intervals, or none to skip them"
        " (default: 0.95)",
        {"type": _optional_coverage},
    ),
    _Hyperparameter(
        "--resamples",
        "resamples",
        _TASKS,
        "Permutations drawn when the test type is monte_carlo (default: none)",
        {"type": _positive_integer},
    ),
    _Hyperparameter(
        "--random-state",
        "random_state",
        _TASKS,
        "Seed for resampling, bootstrap intervals, and plot jitter"
        " (default: none)",
        {"type": _non_negative_integer},
    ),
    _Hyperparameter(
        "--reverse-order",
        "reverse_order",
        _TASKS,
        "Reverse the leaf order used by every rendering (default: False)",
        {"action": "store_true"},
    ),
    _Hyperparameter(
        "--response-sample-size",
        "response_sample_size",
        ("regression",),
        "Responses kept per leaf for the response plot (default: 1000)",
        {"type": _non_negative_integer},
    ),
    _Hyperparameter(
        "--metrics",
        "metrics",
        ("survival",),
        "Comma-separated metrics, each a kind or kind:value:unit; the first"
        " one drives predict (default: median)",
        {"type": _metric_list},
    ),
    _Hyperparameter(
        "--pca-components",
        "pca_components",
        ("ranking",),
        "Principal components of the per-node log-rank matrix (default: 10)",
        {"type": _positive_integer},
    ),
    _Hyperparameter(
        "--npseudo",
        "npseudo",
        ("ranking",),
        "Weight of the Turner ghost-item pseudo-comparisons (default: 0.5)",
        {"type": _positive_float},
    ),
    _Hyperparameter(
        "--pl-max-iter",
        "pl_max_iter",
        ("ranking",),
        "Hunter iterations allowed per Plackett-Luce fit (default: 100)",
        {"type": _positive_integer},
    ),
    _Hyperparameter(
        "--pl-tolerance",
        "pl_tolerance",
        ("ranking",),
        "Convergence tolerance on the log-worth (default: 1e-06)",
        {"type": _positive_float},
    ),
    _Hyperparameter(
        "--ci-replicates",
        "ci_replicates",
        ("ranking",),
        "Bootstrap replicates for the resampling interval methods"
        " (default: 200)",
        {"type": _positive_integer},
    ),
)
"""Every fit flag, the parameter it sets, and the tasks that accept it."""

_UNEXPOSED = frozenset({"transmuter", "decorator", "item_names"})
"""Constructor parameters deliberately left off the command line."""

_PER_TASK_DEFAULTS = frozenset({"correlation", "ci_method"})
"""Parameters whose default differs from one task to the next."""


def _load_data(
    file_path: str,
    column_types: None | dict[str, object] = None,
) -> pyarrow.Table:
    """Read a table from a file, forcing the given column types when able."""
    _require_io()
    import pyarrow.csv
    import pyarrow.ipc
    import pyarrow.json
    import pyarrow.orc
    import pyarrow.parquet

    exists = os.path.exists(file_path)
    if not exists:
        raise FileNotFoundError(f"data file not found: {file_path}")
    _, suffix = os.path.splitext(file_path)
    suffix = suffix.lower()
    convert_options = pyarrow.csv.ConvertOptions(
        column_types=column_types, strings_can_be_null=True
    )
    match suffix:
        case ".csv":
            table = pyarrow.csv.read_csv(
                file_path, convert_options=convert_options
            )
        case ".tsv":
            parse_options = pyarrow.csv.ParseOptions(delimiter="\t")
            table = pyarrow.csv.read_csv(
                file_path,
                parse_options=parse_options,
                convert_options=convert_options,
            )
        case ".parquet":
            table = pyarrow.parquet.read_table(file_path)
        case ".arrow" | ".feather":
            with pyarrow.ipc.open_file(file_path) as reader:
                table = reader.read_all()
        case ".orc":
            table = pyarrow.orc.read_table(file_path)
        case ".jsonl" | ".ndjson":
            table = pyarrow.json.read_json(file_path)
        case _:
            supported = ", ".join(_INPUT_EXTENSIONS)
            raise ValueError(
                f"unsupported input file format: {suffix!r}, supported"
                f" formats are {supported}"
            )
    _reject_duplicate_columns(table)
    return table


def _reject_duplicate_columns(table: pyarrow.Table) -> None:
    """Raise when a table carries the same column name more than once."""
    seen: set[str] = set()
    duplicates: list[str] = []
    for name in table.column_names:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
    if duplicates:
        listing = ", ".join(duplicates)
        raise ValueError(f"data file has duplicate column names: {listing}")


def _save_data(
    table: pyarrow.Table,
    file_path: None | str,
    output_format: None | str = None,
) -> None:
    """Write a table to a file, or to standard output when no path is given."""
    _require_io()
    import pyarrow.csv
    import pyarrow.ipc
    import pyarrow.orc
    import pyarrow.parquet

    if file_path is None:
        target: str | typing.BinaryIO = sys.stdout.buffer
        inferred_format = "csv"
    else:
        target = file_path
        _, suffix = os.path.splitext(file_path)
        lowered = suffix.lower()
        inferred_format = lowered.lstrip(".")
    if output_format is None:
        chosen = inferred_format
    else:
        chosen = output_format
    match chosen:
        case "csv":
            pyarrow.csv.write_csv(table, target)
        case "tsv":
            write_options = pyarrow.csv.WriteOptions(delimiter="\t")
            pyarrow.csv.write_csv(table, target, write_options=write_options)
        case "parquet":
            pyarrow.parquet.write_table(table, target)
        case "arrow" | "feather":
            with pyarrow.ipc.new_file(target, table.schema) as writer:
                writer.write_table(table)
        case "orc":
            pyarrow.orc.write_table(table, target)
        case "jsonl" | "ndjson":
            _save_jsonl(table, target)
        case "md":
            _save_markdown_table(table, target)
        case _:
            supported = ", ".join(_OUTPUT_FORMATS)
            raise ValueError(
                f"unsupported output file format: {chosen!r}, supported"
                f" formats are {supported}"
            )


def _save_jsonl(table: pyarrow.Table, target: str | typing.BinaryIO) -> None:
    """Write a table as one JSON object per line."""
    rows = table.to_pylist()
    lines = []
    for row in rows:
        rendered = json.dumps(row)
        lines.append(rendered + "\n")
    content = "".join(lines)
    _write_text(content, target)


def _save_markdown_table(
    table: pyarrow.Table, target: str | typing.BinaryIO
) -> None:
    """Write a table as a Markdown table with padded cells."""
    names = table.column_names
    columns = [column.to_pylist() for column in table.columns]
    rows = [names]
    for index in range(table.num_rows):
        cells = [_format_markdown_cell(column[index]) for column in columns]
        rows.append(cells)
    widths = []
    for index in range(len(names)):
        lengths = [len(row[index]) for row in rows]
        widths.append(max(lengths))
    lines = []
    for row in rows:
        pairs = zip(row, widths, strict=True)
        padded = [cell.ljust(width) for cell, width in pairs]
        joined = " | ".join(padded)
        lines.append(f"| {joined} |")
    separators = ["-" * (width + 2) for width in widths]
    rule = "|".join(separators)
    lines.insert(1, f"|{rule}|")
    content = "\n".join(lines) + "\n"
    _write_text(content, target)


def _format_markdown_cell(value: object) -> str:
    """Render one table entry as the text of a Markdown table cell."""
    match value:
        case None:
            text = "null"
        case bool():
            text = "true" if value else "false"
        case float():
            text = f"{value:.6g}"
            stripped = text.lstrip("-")
            if stripped.isdigit():
                text = text + ".0"
        case _:
            rendered = str(value)
            pieces = rendered.splitlines()
            text = " ".join(pieces)
    return text


def _write_text(content: str, target: str | typing.BinaryIO) -> None:
    """Write text to a file path or to an already-open binary stream."""
    data = content.encode()
    if isinstance(target, str):
        with open(target, "wb") as file_handle:
            file_handle.write(data)
    else:
        target.write(data)
        target.flush()


def _require_io() -> None:
    """Raise a pointed ImportError when pyarrow or pandas is missing."""
    try:
        import pandas  # noqa: F401
        import pyarrow  # noqa: F401
    except ImportError as import_error:
        raise ImportError(
            "pyarrow and pandas are required by the sigma command. "
            "Install them with: pip install ars-sigma[cli]"
        ) from import_error


def _normalize_fit_columns(table: pyarrow.Table) -> pyarrow.Table:
    """Cast every column of a fit table to a type sigma's estimators accept."""
    import pyarrow

    columns = []
    for field in table.schema:
        column = table.column(field.name)
        target = _fit_arrow_type(field.name, field.type)
        if target == field.type:
            columns.append(column)
        else:
            columns.append(_cast_column(column, target, field.name))
    rebuilt = pyarrow.table(columns, names=table.column_names)
    return rebuilt


def _fit_arrow_type(name: str, arrow_type: object) -> object:
    """Return the canonical arrow type a fit column is normalized to."""
    import pyarrow
    import pyarrow.types

    if pyarrow.types.is_boolean(arrow_type):
        return arrow_type
    if pyarrow.types.is_integer(arrow_type):
        return arrow_type
    if pyarrow.types.is_floating(arrow_type):
        return arrow_type
    if pyarrow.types.is_decimal(arrow_type):
        return pyarrow.float64()
    if pyarrow.types.is_null(arrow_type):
        return pyarrow.float64()
    if pyarrow.types.is_dictionary(arrow_type):
        return pyarrow.string()
    is_text = (
        pyarrow.types.is_string(arrow_type)
        or pyarrow.types.is_large_string(arrow_type)
        or pyarrow.types.is_string_view(arrow_type)
    )
    if is_text:
        return pyarrow.string()
    raise ValueError(
        f"column {name!r} has type {arrow_type} which sigma cannot use as a"
        f" feature; supply numeric, boolean, or text columns"
    )


def _cast_column(
    column: pyarrow.ChunkedArray, target: object, name: str
) -> pyarrow.ChunkedArray:
    """Cast one column, naming the column when the cast is not possible."""
    import pyarrow
    import pyarrow.compute

    try:
        cast = pyarrow.compute.cast(column, target)
    except pyarrow.ArrowException as arrow_error:
        raise ValueError(
            f"column {name!r} holds {column.type} values that cannot be read"
            f" as {target}: {arrow_error}"
        ) from arrow_error
    return cast


def _to_pandas(table: pyarrow.Table) -> pandas.DataFrame:
    """Convert a normalized arrow table to the pandas dtypes sigma expects."""
    import pyarrow.types

    frame = table.to_pandas()
    for field in table.schema:
        column = table.column(field.name)
        if pyarrow.types.is_string(field.type):
            frame[field.name] = frame[field.name].astype("category")
        elif pyarrow.types.is_boolean(field.type) and column.null_count > 0:
            frame[field.name] = frame[field.name].astype("boolean")
    return frame


def _predict_column_types(tree: _tree.Tree) -> dict[str, object]:
    """Map each fitted feature name to the arrow type it was fitted with."""
    import pyarrow

    names = _feature_names(tree)
    column_types: dict[str, object] = {}
    for index, name in enumerate(names):
        kind = tree._fit_column_kind(index)
        match kind:
            case "categorical":
                column_types[name] = pyarrow.string()
            case "boolean":
                column_types[name] = pyarrow.bool_()
            case _:
                column_types[name] = pyarrow.float64()
    return column_types


def _normalize_predict_columns(
    table: pyarrow.Table, tree: _tree.Tree
) -> pyarrow.Table:
    """Select the fitted features in fit order and restore their types."""
    import pyarrow

    names = _feature_names(tree)
    available = set(table.column_names)
    missing = [name for name in names if name not in available]
    if missing:
        listing = ", ".join(missing)
        raise ValueError(
            f"data file is missing feature columns required by the model:"
            f" {listing}"
        )
    column_types = _predict_column_types(tree)
    columns = []
    for name in names:
        column = table.column(name)
        target = column_types[name]
        if column.type == target:
            columns.append(column)
        else:
            columns.append(_cast_column(column, target, name))
    projected = pyarrow.table(columns, names=list(names))
    return projected


def _feature_names(tree: _tree.Tree) -> list[str]:
    """List the fitted feature names, refusing a model fitted without them."""
    names = getattr(tree, "feature_names_in_", None)
    if names is None:
        raise ValueError(
            "the model was fitted without column names and cannot be used"
            " from the command line; refit it on a data file"
        )
    listing = [str(name) for name in names]
    return listing


def _export_format(export_format: None | str, output_file: None | str) -> str:
    """Resolve the rendering from the explicit format or the output path."""
    if export_format is not None:
        return export_format
    if output_file is None:
        return "text"
    _, suffix = os.path.splitext(output_file)
    lowered = suffix.lower()
    chosen = _EXPORT_EXTENSIONS.get(lowered)
    if chosen is None:
        supported = ", ".join(_EXPORT_EXTENSIONS)
        raise ValueError(
            f"unsupported export format: {lowered!r}, supported formats"
            f" are {supported}"
        )
    return chosen


def _check_target_arity(
    task: str, names: collections.abc.Sequence[str]
) -> None:
    """Raise when the number of target columns does not suit the task."""
    count = len(names)
    match task:
        case "survival":
            if count not in (1, 2):
                raise ValueError(
                    f"task 'survival' takes one or two target columns,"
                    f" time,event or one age-encoded column; got {count}"
                )
        case "ranking":
            if count < 2:
                raise ValueError(
                    f"task 'ranking' takes at least two target columns, one"
                    f" per item; got {count}"
                )
        case _:
            if count != 1:
                raise ValueError(
                    f"task {task!r} takes one target column; got {count}"
                )


def _check_columns_present(
    table: pyarrow.Table,
    names: collections.abc.Sequence[str],
    role: str,
) -> None:
    """Raise when a named column is absent from the data file."""
    available = table.column_names
    for name in names:
        if name in available:
            continue
        quoted = [repr(entry) for entry in available]
        listing = ", ".join(quoted)
        message = (
            f"{role} column {name!r} not found in data; available columns:"
            f" {listing}"
        )
        commas = [entry for entry in available if "," in entry]
        if commas:
            message = message + (
                " note: a column name containing a comma cannot be"
                " selected, because the target list is comma-separated"
            )
        raise ValueError(message)


def _build_response(
    table: pyarrow.Table,
    task: str,
    names: collections.abc.Sequence[str],
) -> pandas.Series | pandas.DataFrame:
    """Assemble the response the estimator of the given task expects."""
    selected = table.select(list(names))
    match task:
        case "classification":
            column = selected.column(names[0])
            _reject_missing_target(column, names[0])
            normalized = _normalize_fit_columns(selected)
            frame = _to_pandas(normalized)
            return frame[names[0]]
        case "ranking":
            floats = _float_columns(selected, names, False)
            frame = _to_pandas(floats)
            return frame
        case _:
            floats = _float_columns(selected, names, True)
            frame = _to_pandas(floats)
            if len(names) == 1:
                return frame[names[0]]
            return frame


def _float_columns(
    table: pyarrow.Table,
    names: collections.abc.Sequence[str],
    reject_missing: bool,
) -> pyarrow.Table:
    """Cast the named columns to floats, optionally refusing missing values."""
    import pyarrow

    columns = []
    for name in names:
        column = table.column(name)
        if reject_missing:
            _reject_missing_target(column, name)
        target = pyarrow.float64()
        cast = _cast_column(column, target, name)
        columns.append(cast)
    rebuilt = pyarrow.table(columns, names=list(names))
    return rebuilt


def _reject_missing_target(column: pyarrow.ChunkedArray, name: str) -> None:
    """Raise when a response column carries missing values."""
    count = column.null_count
    if count > 0:
        raise ValueError(
            f"target column {name!r} has {count} missing values; remove"
            f" those rows or choose another target"
        )


def _build_sample_weight(
    table: pyarrow.Table, name: None | str
) -> None | numpy.typing.NDArray[numpy.floating]:
    """Read the case weights from the named column, or return None."""
    if name is None:
        return None
    import pyarrow

    column = table.column(name)
    target = pyarrow.float64()
    cast = _cast_column(column, target, name)
    weights = cast.to_numpy(zero_copy_only=False)
    return weights


def _load_model(model_file: str) -> _Estimator:
    """Read a pickled tree, reporting a version mismatch as a warning line."""
    exists = os.path.exists(model_file)
    if not exists:
        raise FileNotFoundError(f"model file not found: {model_file}")
    logger.info("Loading the tree from %s", model_file)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with open(model_file, "rb") as file_handle:
            loaded = pickle.load(file_handle)
    for entry in caught:
        print(f"warning: {entry.message}", file=sys.stderr)
    estimator_classes = tuple(_ESTIMATORS.values())
    if not isinstance(loaded, estimator_classes):
        name = type(loaded).__name__
        raise TypeError(
            f"{model_file} is not a sigma model file; it holds {name}"
        )
    return loaded


def _require_classification(
    tree: _Estimator,
) -> _tree_classification.ClassificationTree:
    """Return the model as the classification tree --proba needs."""
    is_classification = isinstance(
        tree, _tree_classification.ClassificationTree
    )
    if is_classification:
        return tree
    name = type(tree).__name__
    raise ValueError(
        f"--proba applies to a classification model only; got {name}"
    )


def _require_ranking(tree: _Estimator) -> _tree_ranking.RankingTree:
    """Return the model as the ranking tree --rank needs."""
    is_ranking = isinstance(tree, _tree_ranking.RankingTree)
    if is_ranking:
        return tree
    name = type(tree).__name__
    raise ValueError(f"--rank applies to a ranking model only; got {name}")


def _require_survival(tree: _Estimator) -> _tree_survival.SurvivalTree:
    """Return the model as the survival tree --times needs."""
    is_survival = isinstance(tree, _tree_survival.SurvivalTree)
    if is_survival:
        return tree
    name = type(tree).__name__
    raise ValueError(f"--times applies to a survival model only; got {name}")


def _arrow_column(values: numpy.typing.NDArray) -> pyarrow.Array:
    """Convert a prediction array to an arrow column."""
    import pyarrow

    listing = values.tolist()
    column = pyarrow.array(listing)
    return column


def _extend_matrix(
    columns: list[object],
    names: list[str],
    matrix: numpy.typing.NDArray,
    labels: collections.abc.Sequence[str],
) -> None:
    """Append one output column per label of a per-row prediction matrix."""
    for index, label in enumerate(labels):
        column = _arrow_column(matrix[:, index])
        columns.append(column)
        names.append(label)


def _reject_duplicate_names(names: collections.abc.Sequence[str]) -> None:
    """Raise when two output columns would carry the same name."""
    seen: set[str] = set()
    for name in names:
        if name in seen:
            raise ValueError(
                f"--with-input would produce two output columns named"
                f" {name!r}; rename the input column or drop --with-input"
            )
        seen.add(name)


def _resolve_target_class(tree: _tree.Tree, label: object) -> object:
    """Match a target class name against the classes the model was fitted on."""
    classes = getattr(tree, "classes_", None)
    if classes is None:
        name = type(tree).__name__
        raise ValueError(
            f"--target-class applies to a classification model only; got {name}"
        )
    for candidate in classes:
        rendered = str(candidate)
        if rendered == label:
            return candidate
    quoted = [repr(str(entry)) for entry in classes]
    listing = ", ".join(quoted)
    raise ValueError(
        f"target class {label!r} not found in the model; available classes:"
        f" {listing}"
    )


def _write_output_text(content: str, output_file: None | str) -> None:
    """Write a rendering to a file path or to standard output."""
    if output_file is None:
        target: str | typing.BinaryIO = sys.stdout.buffer
    else:
        target = output_file
    _write_text(content, target)


def _write_output_bytes(payload: bytes, output_file: None | str) -> None:
    """Write image bytes to a file path or to standard output."""
    if output_file is None:
        sys.stdout.buffer.write(payload)
        sys.stdout.buffer.flush()
    else:
        with open(output_file, "wb") as file_handle:
            file_handle.write(payload)


def _format_default(value: object) -> str:
    """Render a constructor default the way the flag help states it."""
    match value:
        case None:
            return "none"
        case bool():
            return str(value)
        case tuple():
            rendered = [str(entry) for entry in value]
            joined = ", ".join(rendered)
            return joined
        case float():
            return repr(value)
        case _:
            return str(value)


_EXPORT_FLAG_FORMATS = {
    "kind": _IMAGE_FORMATS,
    "max_depth": _EXPORT_FORMATS,
    "precision": ("text", "dot") + _IMAGE_FORMATS,
    "top_displayed_items": ("text",) + _IMAGE_FORMATS,
    "target_class": ("sql",),
    "orientation": ("dot",) + _IMAGE_FORMATS,
    "dpi": ("dot",) + _IMAGE_FORMATS,
    "max_branch_length": ("dot",) + _IMAGE_FORMATS,
}
"""Renderings each export flag applies to."""
