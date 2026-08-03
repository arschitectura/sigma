"""Abstract Tree base class for conditional inference tree estimators.

Implements the shared recursive partitioning algorithm from Hothorn,
Hornik, and Zeileis (2006). The concrete ClassificationTree,
RegressionTree, SurvivalTree, and RankingTree subclasses live in their
own sibling modules.
"""

from __future__ import annotations

import abc
import collections.abc
import copy
import sys
import typing

import numpy
import numpy.typing
import scipy.sparse
import sklearn.base
import sklearn.utils.validation
import typing_extensions

from . import (
    _extension,
    _feature,
    _metric,
    _node,
    _partition,
    _splitting,
    _statistics,
    _tree_text,
    _types,
)

if typing.TYPE_CHECKING:
    import matplotlib.axes
    import pandas
    import polars


_CategoryLabels: typing.TypeAlias = (
    dict[int, dict[float, str]]
    | dict[str, dict[float, str]]
    | dict[int | str, dict[float, str]]
)

_OFFSET_EPS = 1e-15


N = typing.TypeVar("N", bound=_node.Node)


class Tree(
    sklearn.base.BaseEstimator,
    abc.ABC,
    typing.Generic[N],
):
    """Abstract base class for conditional inference trees.

    Implements the shared recursive partitioning algorithm from Hothorn, Hornik,
    and Zeileis (2006), "Unbiased Recursive Partitioning: A Conditional
    Inference Framework," *Journal of Computational and Graphical Statistics*,
    15(3), 651-674.

    Args:
        correlation: Correlation type: "normal" or "rank" (default).
        test_stat: Test statistic form: "maximum" or "quadratic".
        test_type: Multiplicity adjustment method: "bonferroni",
            "monte_carlo", or "sidak".
        alpha: Significance level for the stopping rule.
        min_splits: Minimum sum of weights required to attempt a split.
        min_buckets: Minimum sum of weights in each child node.
        max_depth: Maximum tree depth. None means no limit.
        categorical_features: Categorical features. Entries may be column-name
            strings or integer column indices. String entries are resolved
            against the DataFrame columns at fit time (i.e., they require X to
            be a pandas DataFrame). None means all numeric.
        ci_coverage: Coverage level for node-prediction confidence intervals.
            Defaults to 0.95. Set to None to disable CI computation.
        transmuter: Optional callable applied to node data before computing
            predictions and confidence intervals. Signature: (X, y,
            sample_weight, offset, side_data) -> (y', sample_weight',
            offset'). X has shape (n_active, n_features), y has shape
            (n_active,), sample_weight is None or has shape (n_active,),
            offset is None (when fit was called without offset) or has shape
            matching the per-task fit-time offset (n_active for regression
            and survival, (n_active, n_classes) for classification), and
            side_data is None (when fit was called without side_data) or has
            shape (n_active, ...). The returned y' and sample_weight' may
            have a different length than the inputs (n'); offset' must have
            row count n' aligned with y' and the same per-task shape and
            domain as the input offset (None if and only if the input offset
            was None). The returned tuple is validated; mismatched shapes or
            None-vs-non-None inconsistency on offset' raises ValueError.
        resamples: Number of permutations for min-P resampling when
            test_type="monte_carlo". Must be a positive integer. Ignored for
            other test_type values.
        decorator: Optional callable invoked once per node after the tree is
            built. Signature: (X_active, y_active, w_active, offset_active,
            side_data_active) -> decoration. The five arguments are the
            per-node subsets of the raw fit inputs (same length, active
            samples only). offset_active is None when fit was called without
            offset; side_data_active is None when fit was called without
            side_data. The returned value (any object, or None) is stored on
            the node as node.decoration.
        random_state: Seed for stochastic operations. Pass an integer for
            reproducibility; None uses an unpredictable seed. Controls
            min-P permutation resampling under test_type="monte_carlo",
            the bootstrap-family CI methods of RegressionTree
            ("bayesian_bootstrap", "bca", "log_normal_gci"), and the
            jitter of to_image(kind="response") raincloud plots.

    Attributes:
        content_: Root node of the fitted tree structure.
        leaves_: List of leaf nodes, in the leaf order of the estimator's
            task, as documented by each estimator.
        nodes_: List of all nodes in pre-order DFS, ordered by node_id
            (so nodes_[k].node_id == k). Indices match the output of
            predict_index.
        n_features_in_: Number of features seen during fit.
        feature_names_in_: Column names seen during fit, set when X is a
            pandas or polars object that carries them.
        features_: One Feature per column of X, in column order. Each entry
            is a NumericFeature, BooleanFeature, CategoricalFeature, or
            PromotedBooleanFeature and carries the display name, category
            labels, and missing-value code learned for that column.
        metrics_: One Metric per entry of the value array each node
            carries, in the same order, describing how that value is
            labeled, formatted, and ordered. Empty for the estimators
            whose nodes report their values through dedicated attributes
            (RegressionTree and ClassificationTree).
        response_name_in_: Display name captured from a named pandas Series y
            (or, for survival, the first column name of a DataFrame y) at fit
            time. None when y carries no usable name.
    """

    # sklearn.base.BaseEstimator forbids __slots__ on subclasses (its
    # __getstate__ raises TypeError if any subclass declares __slots__).
    # Instances inherit __dict__ and __weakref__ from BaseEstimator.

    content_: N
    leaves_: list[N]
    nodes_: list[N]
    n_features_in_: int
    feature_names_in_: numpy.typing.NDArray
    features_: tuple[_feature.Feature, ...]
    metrics_: tuple[_metric.Metric, ...]
    response_name_in_: None | str
    correlation_enum_: _types.Correlation
    test_stat_enum_: _types.TestStat
    test_type_enum_: _types.TestType
    _fit_with_offset: bool
    _rng_: numpy.random.Generator
    _rng_ci_: numpy.random.Generator

    def __init__(
        self,
        correlation: typing.Literal["normal", "rank"] = "rank",
        test_stat: typing.Literal["maximum", "quadratic"] = "quadratic",
        test_type: typing.Literal[
            "bonferroni", "monte_carlo", "sidak"
        ] = "sidak",
        alpha: float = 0.05,
        min_splits: int = 20,
        min_buckets: int = 7,
        max_depth: None | int = None,
        categorical_features: None | collections.abc.Sequence[str | int] = None,
        ci_coverage: None | float = 0.95,
        transmuter: None | typing.Callable = None,
        resamples: None | int = None,
        decorator: None | typing.Callable = None,
        random_state: None | int = None,
        reverse_order: bool = False,
    ) -> None:
        _types._validate_literal_param(
            correlation, _types.Correlation, "correlation"
        )
        _types._validate_literal_param(test_stat, _types.TestStat, "test_stat")
        _types._validate_literal_param(test_type, _types.TestType, "test_type")
        self.correlation = correlation
        self.test_stat = test_stat
        self.test_type = test_type
        self.alpha = alpha
        self.min_splits = min_splits
        self.min_buckets = min_buckets
        self.max_depth = max_depth
        self.categorical_features = categorical_features
        self.ci_coverage = ci_coverage
        self.transmuter = transmuter
        self.resamples = resamples
        self.decorator = decorator
        self.random_state = random_state
        self.reverse_order = reverse_order
        self._fit_with_offset = False

    def __sklearn_tags__(self):
        """Declare that NaN is accepted in the covariate matrix X."""
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        return tags

    def fit(
        self,
        X: numpy.typing.NDArray[numpy.floating]
        | pandas.DataFrame
        | polars.DataFrame,
        y: (
            numpy.typing.NDArray[numpy.floating]
            | pandas.Series
            | pandas.DataFrame
            | polars.DataFrame
        ),
        sample_weight: None | numpy.typing.NDArray[numpy.floating] = None,
        offset: None | numpy.typing.NDArray[numpy.floating] = None,
        side_data: None | numpy.typing.NDArray[numpy.floating] = None,
    ) -> Tree:
        """Fit the conditional inference tree.

        Args:
            X: Training covariate matrix, shape (n_samples, n_features).
            y: Training target vector, shape (n_samples,).
            sample_weight: Per-sample weights, shape (n_samples,). If None, all
                samples are weighted equally. Values must be non-negative and
                finite. A weight of k is equivalent to observing that sample k
                times.
            offset: Optional per-sample baseline expressed in the user-facing
                response space. Shape and scale depend on the estimator:
                regression expects (n_samples,) on the response scale,
                classification expects (n_samples, n_classes) of probabilities
                summing to 1 along the class axis, and survival expects
                (n_samples,) of survival probabilities S(T_i) at the record's
                observation time. None disables the offset path.
            side_data: Optional auxiliary per-sample data, shape (n_samples,
                ...). Must have the same number of rows as X. When a transmuter
                is set, the active subset is passed to it as a 5th positional
                argument. When a decorator is set, the active subset is passed
                to it as a 5th positional argument.

        Returns:
            The fitted estimator.

        Raises:
            ValueError: If side_data has a different number of rows than X; if
                sample_weight is not 1D, has the wrong length, contains
                non-finite or negative values, or is all zero; if resamples is
                not a positive integer; if test_type="monte_carlo" and
                resamples is None; if categorical_features contains a string
                label without a DataFrame name source, or a string label that
                is not among the DataFrame columns; if X is a plain array or
                list carrying boolean values; if any response value
                violates the domain required by the chosen ci_method (y > 0
                for "log_normal", y >= 0 for "gamma" / "poisson" /
                "exponential", y in [0, 1] for "beta"); or if offset has the
                wrong shape, contains non-finite values, or violates the
                per-task domain.
        """
        with numpy.errstate(divide="ignore", invalid="ignore", over="ignore"):
            self.response_name_in_ = _extract_response_name(y)
            X, typed_features = _preprocess_dataframe_X(X)
            input_columns = getattr(X, "columns", None)
            if input_columns is None:
                _reject_boolean_bare_values(X)
            X, y = self._validate_fit_params(X, y)
            n_rows = X.shape[0]
            if side_data is not None:
                side_rows = side_data.shape[0]
                if side_rows != n_rows:
                    raise ValueError(
                        f"side_data has {side_rows} rows, expected {n_rows}"
                    )
            offset_array = self._validate_offset(offset, n_rows)
            weights = self._validate_sample_weight(sample_weight, n_rows)
            names = self._effective_feature_names()
            self.features_ = self._build_features(
                X, weights, names, typed_features
            )
            X = self._encode_categorical_missing(X, reset=True)
            self.correlation_enum_ = _types.Correlation(self.correlation)
            self.test_stat_enum_ = _types.TestStat(self.test_stat)
            test_type_enum = _types.TestType(self.test_type)
            self.test_type_enum_ = test_type_enum
            resamples = self.resamples
            if resamples is not None and resamples < 1:
                raise ValueError(
                    f"resamples must be a positive integer, got {resamples}"
                )
            if (
                test_type_enum == _types.TestType.MONTE_CARLO
                and resamples is None
            ):
                raise ValueError(
                    "resamples must be set when test_type='monte_carlo'"
                )
            self.metrics_ = self._build_metrics()
            seed_seq = numpy.random.SeedSequence(self.random_state)
            ss_fit, ss_ci = seed_seq.spawn(2)
            self._rng_ = numpy.random.default_rng(ss_fit)
            self._rng_ci_ = numpy.random.default_rng(ss_ci)
            self._fit_with_offset = offset_array is not None
            h = self._compute_influence(y, offset_array)
            y_transmuted, w_transmuted, offset_transmuted = (
                self._apply_transmuter(
                    X, y, weights, sample_weight, side_data, offset_array
                )
            )
            self.content_ = self._build_tree(
                X,
                y,
                h,
                weights,
                depth=0,
                y_transmuted=y_transmuted,
                w_transmuted=w_transmuted,
                sample_weight=sample_weight,
                side_data=side_data,
                offset=offset_array,
                offset_transmuted=offset_transmuted,
            )
            _node._populate_share(self.content_)
            # Final numbering pass: tree shape is frozen above this line.
            # Insert any future tree-transformation step (e.g. pruning) ABOVE
            # this comment, NEVER below it. Both leaf_id (via _build_leaves)
            # and node_id (via _assign_node_ids) are derived from the final
            # tree shape and must be the last things written before fit
            # returns.
            self._assign_node_ids()
            self._build_leaves()
            return self

    def _build_features(
        self,
        X: numpy.typing.NDArray[numpy.floating],
        weights: numpy.typing.NDArray[numpy.floating],
        names: None | numpy.typing.NDArray,
        typed_features: dict[int, _feature.Feature],
    ) -> tuple[_feature.Feature, ...]:
        """Build one Feature per column, completing the DataFrame-typed columns with the numeric and declared-categorical ones."""
        n_features = X.shape[1]
        active = weights > 0
        features: list[_feature.Feature] = []
        for j in range(n_features):
            typed = typed_features.get(j)
            if typed is not None:
                features.append(typed)
                continue
            column = X[active, j]
            observed = column[~numpy.isnan(column)]
            integer = observed.size > 0 and numpy.all(
                numpy.mod(observed, 1) == 0
            )
            features.append(_feature.NumericFeature(j, integer=bool(integer)))
        categorical_features = self.categorical_features
        if categorical_features is not None:
            for entry in categorical_features:
                index = self._resolve_categorical_entry(entry, names)
                # A boolean or already categorical column keeps the type its
                # dtype gave it; only a numeric one is promoted here.
                if isinstance(features[index], _feature.NumericFeature):
                    features[index] = _feature.CategoricalFeature(index)
        if names is not None:
            for feature in features:
                feature.name = str(names[feature.index])
        result = tuple(features)
        return result

    def _build_metrics(self) -> tuple[_metric.Metric, ...]:
        """Build the descriptors of the values each node reports; none by default."""
        return ()

    def _encode_categorical_missing(
        self, X: numpy.typing.NDArray[numpy.floating], reset: bool
    ) -> numpy.typing.NDArray[numpy.floating]:
        """Map NaN in numeric-coded categorical columns to an N/A code and send
        unseen values to an unroutable sentinel. At fit the N/A code is one past
        the largest observed code, and the observed codes are recorded. At
        predict a missing value reuses the recorded N/A code (or an unroutable
        sentinel when the column was complete at fit), and any value not observed
        at fit becomes the unroutable sentinel so it falls back to the holding
        node's prediction. DataFrame categoricals are already coded by
        _preprocess_dataframe_X, so only numpy categorical_features columns are
        affected here.
        """
        categorical = [
            feature
            for feature in self.features_
            if isinstance(feature, _feature.CategoricalFeature)
        ]
        if reset:
            columns = [
                feature
                for feature in categorical
                if numpy.isnan(X[:, feature.index]).any()
            ]
        else:
            columns = [
                feature
                for feature in categorical
                if numpy.isnan(X[:, feature.index]).any()
                or feature.observed_codes is not None
            ]
        if not columns:
            return X
        X = numpy.array(X, copy=True)
        for feature in columns:
            j = feature.index
            missing = numpy.isnan(X[:, j])
            if reset:
                observed = X[~missing, j]
                code = float(observed.max()) + 1.0 if observed.size else 0.0
                feature.na_code = code
                feature.observed_codes = frozenset(
                    float(value) for value in numpy.unique(observed)
                )
            else:
                code = -1.0 if feature.na_code is None else feature.na_code
                known = feature.observed_codes
                if known is not None:
                    known_array = numpy.fromiter(known, dtype=float)
                    seen = numpy.isin(X[:, j], known_array)
                    unroutable = ~missing & ~seen
                    X[unroutable, j] = -1.0
            X[missing, j] = code
        return X

    @staticmethod
    def _resolve_categorical_entry(
        entry: str | int,
        feature_names: None | numpy.typing.NDArray,
    ) -> int:
        """Resolve a column identifier (str or int) to a column index."""
        match entry:
            case str():
                if feature_names is None:
                    raise ValueError(
                        f"column label {entry!r} cannot be resolved:"
                        f" no feature names available (fit with a pandas"
                        f" DataFrame, or pass feature_names to the display"
                        f" method)"
                    )
                matches = numpy.flatnonzero(feature_names == entry)
                if matches.size == 0:
                    raise ValueError(
                        f"column label {entry!r} is not among the"
                        f" available feature names"
                        f" {list(feature_names)}"
                    )
                index = int(matches[0])
            case int():
                index = entry
        return index

    def _effective_feature_names(
        self,
        feature_names: None | list[str] = None,
    ) -> None | numpy.typing.NDArray:
        """Return the active column-name source, or None if unset."""
        if feature_names is not None:
            names = numpy.asarray(feature_names, dtype=object)
            return names
        names_in = getattr(self, "feature_names_in_", None)
        return names_in

    def _effective_response_name(
        self,
        response_name: None | str = None,
    ) -> None | str:
        """Return the active response-name source capitalized, or None if unset."""
        if response_name is None:
            name = getattr(self, "response_name_in_", None)
        else:
            name = response_name
        if name is None:
            return None
        capitalized = _tree_text._capitalize_first_letter(name)
        return capitalized

    def _effective_class_names(
        self,
        class_names: None | list[str] = None,
    ) -> None | list[str]:
        """Return the active class-name source, or None if unset."""
        return class_names

    def _resolve_top_displayed_items(
        self,
        top_displayed_items: None | int,
    ) -> None | int:
        """Reject a displayed-item count, which only a RankingTree reports."""
        if top_displayed_items is not None:
            raise ValueError(
                "top_displayed_items is only supported for RankingTree"
                " estimators"
            )
        return None

    def _resolve_displayed_indices(
        self,
        top_displayed_items: None | int,
    ) -> list[int]:
        """Return the value positions to display, none outside RankingTree."""
        self._resolve_top_displayed_items(top_displayed_items)
        return []

    def _resolve_target_class_index(
        self,
        target_class: None | object,
    ) -> None | int:
        """Reject a target class, which only a ClassificationTree reports."""
        if target_class is not None:
            raise ValueError(
                "target_class is only valid for ClassificationTree;"
                f" got {type(self).__name__}"
            )
        return None

    def _resolve_category_labels(
        self,
        category_labels: None | _CategoryLabels,
        names: None | numpy.typing.NDArray,
    ) -> None | dict[int, dict[float, str]]:
        """Resolve the string keys of a display-time category_labels override to integer column indices."""
        if category_labels is None:
            return None
        resolved: dict[int, dict[float, str]] = {}
        for key, mapping in category_labels.items():
            index = Tree._resolve_categorical_entry(key, names)
            resolved[index] = dict(mapping)
        return resolved

    @staticmethod
    def _validate_sample_weight(
        sample_weight: None | numpy.typing.NDArray[numpy.floating],
        n: int,
    ) -> numpy.typing.NDArray[numpy.floating]:
        """Validate and return the sample weight array."""
        if sample_weight is None:
            weights = numpy.ones(n, dtype=float)
            return weights
        sample_weight = numpy.asarray(sample_weight, dtype=float)
        sample_weight_shape = sample_weight.shape
        if sample_weight.ndim != 1 or sample_weight_shape[0] != n:
            raise ValueError(
                f"sample_weight must be 1D with length {n},"
                f" got shape {sample_weight_shape}"
            )
        if not numpy.all(numpy.isfinite(sample_weight)):
            raise ValueError("sample_weight values must be finite")
        if numpy.any(sample_weight < 0):
            raise ValueError("sample_weight values must be non-negative")
        if not numpy.any(sample_weight > 0):
            raise ValueError("sample_weight values must not be all zero")
        weights = sample_weight.copy()
        return weights

    def predict_index(
        self,
        X: numpy.typing.NDArray[numpy.floating]
        | pandas.DataFrame
        | polars.DataFrame,
    ) -> numpy.typing.NDArray[numpy.intp]:
        """Predict node indices for the given samples.

        Args:
            X: Samples to predict, shape (n_samples, n_features).

        Returns:
            Node indices into self.nodes_, shape (n_samples,). When a
            sample's categorical value is not routable at an internal
            node, the index is that of the holding node rather than a
            descendant leaf.

        Raises:
            ValueError: If a column's type kind differs from its fit-time
                kind, if X is a plain array or list while the model was
                fit with boolean or categorical columns, or if a plain
                array or list carries boolean values.
        """
        sklearn.utils.validation.check_is_fitted(self, "content_")
        self._validate_predict_column_types(X)
        X = _apply_categorical_encoding(X, self.features_)
        X_validated = sklearn.utils.validation.validate_data(
            self, X, reset=False, dtype="float64", ensure_all_finite="allow-nan"
        )
        X_array = typing.cast(numpy.typing.NDArray[numpy.floating], X_validated)
        X_array = self._encode_categorical_missing(X_array, reset=False)
        n_samples = X_array.shape[0]
        indices = numpy.empty(n_samples, dtype=numpy.intp)
        for i in range(n_samples):
            node = self.content_.traverse(X_array[i])
            indices[i] = node.node_id
        return indices

    def apply(
        self,
        X: numpy.typing.NDArray[numpy.floating]
        | pandas.DataFrame
        | polars.DataFrame,
    ) -> numpy.typing.NDArray[numpy.intp]:
        """Return the node_id of the node each sample is routed to.

        Args:
            X: Samples to predict, shape (n_samples, n_features).

        Returns:
            Node ids, shape (n_samples,). For samples that reach a leaf
            the id is that leaf's; for samples whose categorical value
            is not routable at an internal node, the id is that of the
            holding node.

        Raises:
            ValueError: If a column's type kind differs from its fit-time
                kind, or if X is a plain array or list that carries
                boolean values or predicts into a model fit with boolean
                or categorical columns.
        """
        ids = self.predict_index(X)
        return ids

    def decision_path(
        self,
        X: numpy.typing.NDArray[numpy.floating]
        | pandas.DataFrame
        | polars.DataFrame,
    ) -> scipy.sparse.csr_matrix:
        """Return the decision path indicator matrix for the given samples.

        Args:
            X: Samples to predict, shape (n_samples, n_features).

        Returns:
            A scipy.sparse.csr_matrix of shape (n_samples, len(nodes_)),
            dtype numpy.intp, with 1s on visited nodes and 0s elsewhere.
            For samples whose categorical value is not routable at an
            internal node, the path ends at that holding node.

        Raises:
            ValueError: If a column's type kind differs from its fit-time
                kind, or if X is a plain array or list that carries
                boolean values or predicts into a model fit with boolean
                or categorical columns.
        """
        sklearn.utils.validation.check_is_fitted(self, "content_")
        self._validate_predict_column_types(X)
        X = _apply_categorical_encoding(X, self.features_)
        X_validated = sklearn.utils.validation.validate_data(
            self, X, reset=False, dtype="float64", ensure_all_finite="allow-nan"
        )
        X_array = typing.cast(numpy.typing.NDArray[numpy.floating], X_validated)
        X_array = self._encode_categorical_missing(X_array, reset=False)
        n_samples = X_array.shape[0]
        indptr = numpy.empty(n_samples + 1, dtype=numpy.intp)
        indptr[0] = 0
        indices: list[int] = []
        for i in range(n_samples):
            visited = self.content_.traverse_path(X_array[i])
            for node in visited:
                indices.append(node.node_id)
            indptr[i + 1] = len(indices)
        indices_array = numpy.asarray(indices, dtype=numpy.intp)
        data = numpy.ones(indices_array.shape[0], dtype=numpy.intp)
        path = scipy.sparse.csr_matrix(
            (data, indices_array, indptr),
            shape=(n_samples, len(self.nodes_)),
            dtype=numpy.intp,
        )
        return path

    def _validate_predict_column_types(
        self,
        X: numpy.typing.NDArray[numpy.floating]
        | pandas.DataFrame
        | polars.DataFrame,
    ) -> None:
        """Reject predict input whose column types differ from the fit-time column types."""
        columns = getattr(X, "columns", None)
        if columns is None:
            listing = self._typed_column_listing()
            if listing is not None:
                raise ValueError(
                    f"predict input is a plain array or list without column"
                    f" types, but the model was fit with typed columns"
                    f" ({listing}); supply a pandas or polars DataFrame with"
                    f" the fit-time column types"
                )
            _reject_boolean_bare_values(X)
            return
        sklearn.utils.validation.validate_data(
            self, X, reset=False, skip_check_array=True
        )
        for index, name in enumerate(columns):
            expected = self._fit_column_kind(index)
            column = X[name]
            actual = _column_kind(column)
            if actual == expected:
                continue
            dtype = column.dtype
            raise ValueError(
                f"column {name!r} was fit as {expected} but supplied as"
                f" {actual} (dtype {dtype}) at predict; supply each column"
                f" with its fit-time type"
            )

    def _fit_column_kind(self, index: int) -> str:
        """Fit-time type kind of a column: boolean, categorical, or numeric."""
        match self.features_[index]:
            case _feature.BooleanFeature() | _feature.PromotedBooleanFeature():
                return "boolean"
            case _feature.CategoricalFeature() as categorical:
                # A column declared through categorical_features carries no
                # labels and keeps the numeric dtype it was supplied with.
                if categorical.category_labels is None:
                    return "numeric"
                return "categorical"
            case _:
                return "numeric"

    def _typed_column_listing(self) -> None | str:
        """Comma-separated name (kind) listing of the boolean and categorical fit columns, or None."""
        parts: list[str] = []
        for feature in self.features_:
            kind = self._fit_column_kind(feature.index)
            if kind == "numeric":
                continue
            if feature.name is None:
                label = f"X[{feature.index}]"
            else:
                label = feature.name
            parts.append(f"{label!r} ({kind})")
        if not parts:
            return None
        listing = ", ".join(parts)
        return listing

    def compact(self) -> typing_extensions.Self:
        """Return a new tree with recursive same-feature splits merged.

        Wherever a node splits on a feature and one of its children splits on
        the same feature, the chain collapses into a single node whose
        branches carry numeric intervals or category subsets. The returned
        tree predicts identically to this one for routable samples; the tree
        this is called on is left unchanged.

        Merged nodes carry no split statistics. Because merging removes
        nodes, apply and decision_path report different node ids on the
        returned tree than on this one.

        Returns:
            A new fitted tree of the same type whose internal nodes are
            N-ary wherever a same-feature chain was collapsed.
        """
        sklearn.utils.validation.check_is_fitted(self, "content_")
        compacted = copy.copy(self)
        root = _compact_node(self.content_, 0)
        compacted.content_ = typing.cast(N, root)
        _node._populate_share(compacted.content_)
        compacted._assign_node_ids()
        compacted._build_leaves()
        return compacted

    def _build_leaves(self) -> None:
        """Populate leaves_ and assign each leaf its leaf_id."""
        raw_leaves = self.content_.leaves()
        metrics = self.metrics_
        sorted_leaves = sorted(
            raw_leaves, key=lambda n: n.leaf_sort_key(metrics)
        )
        if self.reverse_order:
            sorted_leaves = list(reversed(sorted_leaves))
        self.leaves_ = sorted_leaves
        for index, leaf in enumerate(sorted_leaves):
            leaf_extension = typing.cast(_extension.Leaf, leaf.extension)
            leaf_extension.leaf_id = index

    def _assign_node_ids(self) -> None:
        """Pre-order DFS: assign node_id and parent to every node and populate nodes_."""
        collected: list[_node.Node] = []
        root: _node.Node = self.content_
        root.parent = None
        stack: list[_node.Node] = [root]
        while stack:
            node = stack.pop()
            node.node_id = len(collected)
            collected.append(node)
            match node.extension:
                case _partition.Partition() as partition:
                    for child in reversed(partition.children):
                        child.parent = node  # ty: ignore[unresolved-attribute]
                        stack.append(child)  # ty: ignore[invalid-argument-type]
        self.nodes_ = typing.cast(list[N], collected)

    def _build_tree(
        self,
        X: numpy.typing.NDArray[numpy.floating],
        y: numpy.typing.NDArray[numpy.floating],
        h: numpy.typing.NDArray[numpy.floating],
        weights: numpy.typing.NDArray[numpy.floating],
        depth: int,
        y_transmuted: numpy.typing.NDArray[numpy.floating],
        w_transmuted: numpy.typing.NDArray[numpy.floating],
        sample_weight: None | numpy.typing.NDArray[numpy.floating],
        side_data: None | numpy.typing.NDArray,
        offset: None | numpy.typing.NDArray[numpy.floating],
        offset_transmuted: None | numpy.typing.NDArray[numpy.floating],
    ) -> N:
        """Recursively build the conditional inference tree.

        Args:
            X: Covariate matrix, shape (n_samples, n_features).
            y: Encoded target vector, shape (n_samples,).
            h: Influence function values, shape (n_samples, q).
            weights: Case weights encoding node membership, shape (n_samples,).
            depth: Current depth in the tree.
            y_transmuted: Transmuted target vector for this node, used for
                prediction and CI computation. When no transmuter is set, this
                is the same object as y.
            w_transmuted: Transmuted weights for this node. When no transmuter
                is set, this is the same object as weights.
            sample_weight: Caller-provided per-sample weights from fit, or None.
                Forwarded unchanged through recursion; consumed by
                _apply_transmuter and _validate_split_transmuted.
            side_data: Caller-provided per-sample auxiliary data from fit, or
                None. Forwarded unchanged through recursion; consumed by
                _apply_transmuter, _validate_split_transmuted, and
                _apply_decorator.
            offset: Validated per-sample fit-time offset, or None. Forwarded
                unchanged through recursion; the per-side fit-time slice is
                derived inside _validate_split_transmuted to feed the
                transmuter at each split.
            offset_transmuted: Post-transmutation offset for the current node,
                aligned to (y_transmuted, w_transmuted). Equal to the
                fit-time offset when no transmuter is set. Consumed by every
                node-level computation that depends on the offset.

        Returns:
            Root of the constructed subtree.
        """
        w_sum = float(w_transmuted.sum())
        n_samples = int(numpy.count_nonzero(w_transmuted))
        is_constant = self._is_constant_response(
            y_transmuted, w_transmuted, offset_transmuted
        )
        if (
            w_sum < self.min_splits
            or (self.max_depth is not None and depth >= self.max_depth)
            or is_constant
        ):
            leaf = self._create_node(
                depth,
                n_samples,
                y_transmuted,
                w_transmuted,
                offset_transmuted,
                is_leaf=True,
            )
            self._apply_decorator(leaf, X, y, weights, side_data, offset)
            return leaf
        selection = _statistics.select_variable(
            X,
            h,
            weights,
            self.features_,
            self.test_stat_enum_,
            self.test_type_enum_,
            self.alpha,
            self.correlation_enum_,
            resamples=self.resamples,
            rng=self._rng_,
        )
        if selection is None:
            leaf = self._create_node(
                depth,
                n_samples,
                y_transmuted,
                w_transmuted,
                offset_transmuted,
                is_leaf=True,
            )
            self._apply_decorator(leaf, X, y, weights, side_data, offset)
            return leaf
        feature_index = selection.feature_index
        p_value = selection.p_value
        split_result = _splitting.find_best_split(
            X,
            h,
            weights,
            feature_index,
            self.features_,
            self.test_stat_enum_,
            self.min_buckets,
            self.correlation_enum_,
        )
        if split_result is None:
            leaf = self._create_node(
                depth,
                n_samples,
                y_transmuted,
                w_transmuted,
                offset_transmuted,
                is_leaf=True,
            )
            self._apply_decorator(leaf, X, y, weights, side_data, offset)
            return leaf
        split_criterion, _test_statistic = split_result
        numeric_thresholds: None | tuple[int | float, ...] = None
        numeric_nan_child: None | int = None
        left_categories: None | frozenset = None
        right_categories: None | frozenset = None
        split_feature = self.features_[feature_index]
        match split_feature:
            case _feature.BooleanFeature():
                left_mask = X[:, feature_index] <= 0.5
            case _feature.CategoricalFeature():
                categorical_criterion = typing.cast(frozenset, split_criterion)
                left_mask = numpy.isin(
                    X[:, feature_index], list(categorical_criterion)
                )
                left_categories = categorical_criterion
                active = weights > 0
                observed = frozenset(
                    numpy.unique(X[active, feature_index]).tolist()
                )
                right_categories = observed - left_categories
            case _feature.NumericFeature() as numeric_feature:
                numeric_split = typing.cast(
                    _splitting._NumericSplit, split_criterion
                )
                column = X[:, feature_index]
                isnan_column = numpy.isnan(column)
                numeric_nan_child = numeric_split.nan_child
                if numeric_split.threshold is None:
                    left_mask = ~isnan_column
                    numeric_thresholds = ()
                else:
                    threshold = numeric_split.threshold
                    if numeric_nan_child == 0:
                        left_mask = (column <= threshold) | isnan_column
                    else:
                        left_mask = column <= threshold
                    typed: int | float
                    if numeric_feature.integer:
                        typed = int(threshold)
                    else:
                        typed = float(threshold)
                    numeric_thresholds = (typed,)
        left_weights = weights * left_mask.astype(float)
        right_weights = weights * (~left_mask).astype(float)
        if self.transmuter is None:
            y_left_transmuted = y
            w_left_transmuted = left_weights
            offset_left_transmuted = offset
            y_right_transmuted = y
            w_right_transmuted = right_weights
            offset_right_transmuted = offset
        else:
            validation = self._validate_split_transmuted(
                X,
                y,
                weights,
                left_mask,
                sample_weight,
                side_data,
                offset,
            )
            (
                transmuted_p,
                y_left_transmuted,
                w_left_transmuted,
                offset_left_transmuted,
                y_right_transmuted,
                w_right_transmuted,
                offset_right_transmuted,
            ) = validation
            if transmuted_p > self.alpha:
                leaf = self._create_node(
                    depth,
                    n_samples,
                    y_transmuted,
                    w_transmuted,
                    offset_transmuted,
                    is_leaf=True,
                )
                self._apply_decorator(leaf, X, y, weights, side_data, offset)
                return leaf
            p_value = max(p_value, transmuted_p)
        node = self._create_node(
            depth,
            n_samples,
            y_transmuted,
            w_transmuted,
            offset_transmuted,
            is_leaf=False,
        )
        left_child = self._build_tree(
            X,
            y,
            h,
            left_weights,
            depth + 1,
            y_transmuted=y_left_transmuted,
            w_transmuted=w_left_transmuted,
            sample_weight=sample_weight,
            side_data=side_data,
            offset=offset,
            offset_transmuted=offset_left_transmuted,
        )
        right_child = self._build_tree(
            X,
            y,
            h,
            right_weights,
            depth + 1,
            y_transmuted=y_right_transmuted,
            w_transmuted=w_right_transmuted,
            sample_weight=sample_weight,
            side_data=side_data,
            offset=offset,
            offset_transmuted=offset_right_transmuted,
        )
        split_statistics = _partition.SplitStatistics(p_value=p_value)
        children = (left_child, right_child)
        partition: _partition.Partition[N]
        match split_feature:
            case _feature.BooleanFeature():
                partition = _partition.BooleanPartition(
                    feature=split_feature,
                    statistics=split_statistics,
                    children=children,
                )
            case _feature.CategoricalFeature():
                category_groups = (
                    typing.cast(frozenset, left_categories),
                    typing.cast(frozenset, right_categories),
                )
                partition = _partition.CategoricalPartition(
                    feature=split_feature,
                    statistics=split_statistics,
                    children=children,
                    category_groups=category_groups,
                )
            case _:
                partition = _partition.NumericalPartition(
                    feature=split_feature,
                    statistics=split_statistics,
                    children=children,
                    thresholds=typing.cast(
                        "tuple[int | float, ...]", numeric_thresholds
                    ),
                    nan_child=numeric_nan_child,
                )
        node.extension = partition
        self._apply_decorator(node, X, y, weights, side_data, offset)
        return node

    @abc.abstractmethod
    def _create_node(
        self,
        depth: int,
        n_samples: int,
        y_transmuted: numpy.typing.NDArray[numpy.floating],
        w_transmuted: numpy.typing.NDArray[numpy.floating],
        offset_transmuted: None | numpy.typing.NDArray[numpy.floating],
        is_leaf: bool,
    ) -> N:
        """Build the task's Node from its own active samples."""

    @abc.abstractmethod
    def _validate_fit_params(
        self,
        X: numpy.typing.NDArray[numpy.floating]
        | pandas.DataFrame
        | polars.DataFrame,
        y: (
            numpy.typing.NDArray[numpy.floating]
            | pandas.Series
            | pandas.DataFrame
            | polars.DataFrame
        ),
    ) -> tuple[
        numpy.typing.NDArray[numpy.floating],
        numpy.typing.NDArray[numpy.floating],
    ]:
        """Validate inputs and encode the target."""

    @abc.abstractmethod
    def _validate_offset(
        self,
        offset: None | numpy.typing.NDArray[numpy.floating],
        n_samples: int,
    ) -> None | numpy.typing.NDArray[numpy.floating]:
        """Validate and coerce the fit-time offset to its canonical shape."""

    def _validate_offset_shape_finite(
        self,
        offset: numpy.typing.NDArray[numpy.floating],
        expected_shape: tuple[int, ...],
    ) -> numpy.typing.NDArray[numpy.floating]:
        """Coerce offset to float and require expected_shape with finite values."""
        offset_array = numpy.asarray(offset, dtype=float)
        if offset_array.shape != expected_shape:
            raise ValueError(
                f"offset must have shape {expected_shape},"
                f" got shape {offset_array.shape}"
            )
        if not numpy.all(numpy.isfinite(offset_array)):
            raise ValueError("offset values must be finite")
        return offset_array

    @abc.abstractmethod
    def _compute_influence(
        self,
        y: numpy.typing.NDArray[numpy.floating],
        offset: None | numpy.typing.NDArray[numpy.floating],
    ) -> numpy.typing.NDArray[numpy.floating]:
        """Compute the influence function h from the target."""

    @abc.abstractmethod
    def _is_constant_response(
        self,
        y: numpy.typing.NDArray[numpy.floating],
        weights: numpy.typing.NDArray[numpy.floating],
        offset: None | numpy.typing.NDArray[numpy.floating],
    ) -> bool:
        """Check whether the response is constant in this node."""

    def _ci_alpha(self) -> None | float:
        """Half the non-coverage tail mass, or None when CI is disabled."""
        ci_coverage = self.ci_coverage
        if ci_coverage is None:
            return None
        alpha = (1.0 - ci_coverage) / 2.0
        return alpha

    @staticmethod
    def _offset_active_slice(
        offset: None | numpy.typing.NDArray[numpy.floating],
        active: numpy.typing.NDArray[numpy.bool_],
    ) -> None | numpy.typing.NDArray[numpy.floating]:
        """Return offset[active] or None when no fit-time offset."""
        if offset is None:
            return None
        slice_ = offset[active]
        return slice_

    @abc.abstractmethod
    def _plot_response(
        self,
        axes: matplotlib.axes.Axes,
        response_name: None | str,
        class_names: None | list[str],
        displayed_indices: list[int],
        leaf_palette: tuple[str, str, str],
        background_color: str,
    ) -> None:
        """Draw the task's per-leaf response summary on axes."""

    @abc.abstractmethod
    def _validate_transmuted_y_shape(
        self,
        y_out: numpy.typing.NDArray[numpy.floating],
    ) -> int:
        """Validate y' shape returned by the transmuter and return its row count."""

    def _validate_transmuter_return(
        self,
        result: object,
        offset_input_was_none: bool,
    ) -> tuple[
        numpy.typing.NDArray[numpy.floating],
        numpy.typing.NDArray[numpy.floating],
        None | numpy.typing.NDArray[numpy.floating],
    ]:
        """Validate the (y', w', offset') tuple returned by the transmuter.

        Coerces y' and w' (when not None) to float NDArrays. Checks tuple
        arity, per-task y' shape, w' alignment with y', and offset' being
        None if and only if the input offset was None. When non-None, the
        offset' is validated for shape and per-task domain via
        self._validate_offset.

        Args:
            result: Object returned by the user's transmuter callable.
            offset_input_was_none: Whether the offset passed into the
                transmuter was None.

        Returns:
            (y_out, w_out, offset_out). w_out defaults to all-ones when the
            transmuter returned None for w'.

        Raises:
            ValueError: When result is not a 3-tuple, when y'/w'/offset'
                shapes are inconsistent, or when offset' violates the
                None-iff-None invariant or the per-task domain.
        """
        if not (isinstance(result, tuple) and len(result) == 3):
            raise ValueError(
                "transmuter must return a 3-tuple (y, sample_weight, offset)"
            )
        y_raw, w_raw, offset_raw = result
        y_out = numpy.asarray(y_raw, dtype=float)
        n_t = self._validate_transmuted_y_shape(y_out)
        if w_raw is None:
            w_out = numpy.ones(n_t, dtype=float)
        else:
            w_out = numpy.asarray(w_raw, dtype=float)
            w_out_shape = w_out.shape
            if w_out.ndim != 1 or w_out_shape[0] != n_t:
                raise ValueError(
                    f"transmuter sample_weight must be 1D with length"
                    f" {n_t}, got shape {w_out_shape}"
                )
        if offset_input_was_none:
            if offset_raw is not None:
                raise ValueError(
                    "transmuter must return offset=None when the input"
                    " offset was None"
                )
            offset_out: None | numpy.typing.NDArray[numpy.floating] = None
        else:
            if offset_raw is None:
                raise ValueError(
                    "transmuter must return a non-None offset when the"
                    " input offset was non-None"
                )
            offset_array = typing.cast(
                numpy.typing.NDArray[numpy.floating], offset_raw
            )
            offset_out = self._validate_offset(offset_array, n_t)
        return y_out, w_out, offset_out

    def _apply_transmuter(
        self,
        X: numpy.typing.NDArray[numpy.floating],
        y: numpy.typing.NDArray[numpy.floating],
        weights: numpy.typing.NDArray[numpy.floating],
        sample_weight: None | numpy.typing.NDArray[numpy.floating],
        side_data: None | numpy.typing.NDArray,
        offset: None | numpy.typing.NDArray[numpy.floating],
    ) -> tuple[
        numpy.typing.NDArray[numpy.floating],
        numpy.typing.NDArray[numpy.floating],
        None | numpy.typing.NDArray[numpy.floating],
    ]:
        """Apply the transmuter to active samples.

        When the transmuter is None, returns (y, weights, offset) unchanged
        (same objects, no copy). Otherwise, extracts active samples and
        calls the transmuter with the active subsets of side_data and
        offset. side_data and offset are passed as None when fit was called
        without the corresponding argument.

        Args:
            X: Covariate matrix, shape (n_samples, n_features).
            y: Target vector, shape (n_samples,).
            weights: Case weights, shape (n_samples,).
            sample_weight: Caller-provided per-sample weights from fit, or
                None.
            side_data: Caller-provided per-sample auxiliary data from fit, or
                None.
            offset: Validated per-sample fit-time offset, or None.

        Returns:
            (y_out, weights_out, offset_out) with transmuted data. The
            three arrays share the same row count.

        Raises:
            ValueError: When the transmuter does not return a 3-tuple, when
                the shapes of the returned arrays are inconsistent, or
                when offset' is None but the input offset was not (or
                vice versa).
        """
        transmuter = self.transmuter
        if transmuter is None:
            return y, weights, offset
        active = weights > 0
        X_active = X[active]
        y_active = y[active]
        w_active = None if sample_weight is None else sample_weight[active]
        side_data_active = None if side_data is None else side_data[active]
        offset_active = self._offset_active_slice(offset, active)
        result = transmuter(
            X_active, y_active, w_active, offset_active, side_data_active
        )
        y_out, w_out, offset_out = self._validate_transmuter_return(
            result, offset_active is None
        )
        return y_out, w_out, offset_out

    def _validate_split_transmuted(
        self,
        X: numpy.typing.NDArray[numpy.floating],
        y: numpy.typing.NDArray[numpy.floating],
        weights: numpy.typing.NDArray[numpy.floating],
        left_mask: numpy.typing.NDArray[numpy.bool_],
        sample_weight: None | numpy.typing.NDArray[numpy.floating],
        side_data: None | numpy.typing.NDArray,
        offset: None | numpy.typing.NDArray[numpy.floating],
    ) -> tuple[
        float,
        numpy.typing.NDArray[numpy.floating],
        numpy.typing.NDArray[numpy.floating],
        None | numpy.typing.NDArray[numpy.floating],
        numpy.typing.NDArray[numpy.floating],
        numpy.typing.NDArray[numpy.floating],
        None | numpy.typing.NDArray[numpy.floating],
    ]:
        """Test whether a split is significant on transmuted data.

        Transmutes left and right subsets independently (passing the
        corresponding subset of side_data to each call), then computes the
        test statistic and p-value on the combined transmuted data using the
        existing conditional inference framework. The raw p-value is adjusted
        for multiplicity using the same correction as variable selection,
        treating this as one of m tests where m is the number of covariates.
        When test_type is MONTE_CARLO, Sidak correction is used instead because
        the Monte Carlo adjustment requires the full resampling loop over all
        covariates which is not available in this single-test context. Sidak is
        conservative under covariate dependence.

        Args:
            X: Covariate matrix, shape (n_samples, n_features).
            y: Target vector, shape (n_samples,).
            weights: Case weights, shape (n_samples,).
            left_mask: Boolean mask for left child, shape (n_samples,).
            sample_weight: Caller-provided per-sample weights from fit, or
                None.
            side_data: Caller-provided per-sample auxiliary data from fit, or
                None.
            offset: Validated per-sample fit-time offset, or None.

        Returns:
            (adjusted_p_value, y_left_transmuted, w_left_transmuted,
            offset_left_transmuted, y_right_transmuted, w_right_transmuted,
            offset_right_transmuted). The post-transmutation arrays for each
            side share their row count, and the offsets are None when the
            input offset was None.

        Raises:
            ValueError: When either transmuter call returns a malformed
                3-tuple (see _validate_transmuter_return).
        """
        transmuter = typing.cast(typing.Callable, self.transmuter)
        offset_active_slice = self._offset_active_slice
        validate_transmuter_return = self._validate_transmuter_return
        active = weights > 0
        left_active = left_mask & active
        right_active = ~left_mask & active
        X_left = X[left_active]
        y_left = y[left_active]
        w_left = None if sample_weight is None else sample_weight[left_active]
        side_data_left = None if side_data is None else side_data[left_active]
        offset_left = offset_active_slice(offset, left_active)
        left_result = transmuter(
            X_left, y_left, w_left, offset_left, side_data_left
        )
        (
            y_left_transmuted,
            w_left_transmuted,
            offset_left_transmuted,
        ) = validate_transmuter_return(left_result, offset_left is None)
        X_right = X[right_active]
        y_right = y[right_active]
        w_right = None if sample_weight is None else sample_weight[right_active]
        side_data_right = None if side_data is None else side_data[right_active]
        offset_right = offset_active_slice(offset, right_active)
        right_result = transmuter(
            X_right, y_right, w_right, offset_right, side_data_right
        )
        (
            y_right_transmuted,
            w_right_transmuted,
            offset_right_transmuted,
        ) = validate_transmuter_return(right_result, offset_right is None)
        n_left = len(y_left_transmuted)
        n_right = len(y_right_transmuted)
        y_combined = numpy.concatenate([y_left_transmuted, y_right_transmuted])
        w_combined = numpy.concatenate([w_left_transmuted, w_right_transmuted])
        g_j = numpy.concatenate(
            [
                numpy.ones((n_left, 1)),
                numpy.zeros((n_right, 1)),
            ]
        )
        # Offset is intentionally not propagated to the post-transmutation
        # significance test: y_combined has length n_left + n_right which does
        # not align with the fit-time offset, and the transmuter may have
        # changed the row count. The check is run on the unshifted influence
        # function.
        h_combined = self._compute_influence(y_combined, None)
        T = _statistics.compute_linear_statistic(g_j, h_combined, w_combined)
        mu = _statistics.compute_conditional_expectation(
            g_j, h_combined, w_combined
        )
        Sigma = _statistics.compute_conditional_covariance(
            g_j, h_combined, w_combined
        )
        test_stat_enum = self.test_stat_enum_
        statistic = _statistics.compute_test_statistic(
            T, mu, Sigma, test_stat_enum
        )
        raw_p_value = _statistics.compute_p_value(
            statistic, Sigma, test_stat_enum
        )
        m = X.shape[1]
        p_values = numpy.full(m, raw_p_value)
        test_type = self.test_type_enum_
        if test_type == _types.TestType.MONTE_CARLO:
            test_type = _types.TestType.SIDAK
        adjusted_p_values = _statistics._adjust_p_values(p_values, test_type)
        p_value = float(adjusted_p_values[0])
        return (
            p_value,
            y_left_transmuted,
            w_left_transmuted,
            offset_left_transmuted,
            y_right_transmuted,
            w_right_transmuted,
            offset_right_transmuted,
        )

    def _apply_decorator(
        self,
        node: _node.Node,
        X: numpy.typing.NDArray[numpy.floating],
        y: numpy.typing.NDArray[numpy.floating],
        weights: numpy.typing.NDArray[numpy.floating],
        side_data: None | numpy.typing.NDArray,
        offset: None | numpy.typing.NDArray[numpy.floating],
    ) -> None:
        """Invoke the decorator on the node's active subset and store the
        result.
        """
        decorator = self.decorator
        if decorator is None:
            return
        active = weights > 0
        X_active = X[active]
        y_active = y[active]
        w_active = weights[active]
        side_data_active = None if side_data is None else side_data[active]
        offset_active = self._offset_active_slice(offset, active)
        node.decoration = decorator(
            X_active, y_active, w_active, offset_active, side_data_active
        )

    @typing.overload
    def to_text(
        self,
        out_file: None = None,
        feature_names: None | list[str] = None,
        class_names: None | list[str] = None,
        response_name: None | str = None,
        category_labels: None | _CategoryLabels = None,
        prediction_formatter: None | typing.Callable[[float], str] = None,
        max_depth: None | int = None,
        precision: int = 3,
        top_displayed_items: None | int = None,
    ) -> str: ...

    @typing.overload
    def to_text(
        self,
        out_file: str | typing.IO[str],
        feature_names: None | list[str] = None,
        class_names: None | list[str] = None,
        response_name: None | str = None,
        category_labels: None | _CategoryLabels = None,
        prediction_formatter: None | typing.Callable[[float], str] = None,
        max_depth: None | int = None,
        precision: int = 3,
        top_displayed_items: None | int = None,
    ) -> None: ...

    def to_text(
        self,
        out_file: None | str | typing.IO[str] = None,
        feature_names: None | list[str] = None,
        class_names: None | list[str] = None,
        response_name: None | str = None,
        category_labels: None | _CategoryLabels = None,
        prediction_formatter: None | typing.Callable[[float], str] = None,
        max_depth: None | int = None,
        precision: int = 3,
        top_displayed_items: None | int = None,
    ) -> None | str:
        """Serialize the fitted tree structure to text.

        When out_file is None the text is returned as a string; otherwise
        it is written to the destination and the function returns None.

        Args:
            out_file: Where to write the text. When None (the default), the
                text is returned. When a string, it is treated as a
                filesystem path; the file is opened with UTF-8 encoding,
                written, closed, and the function returns None. When a
                file-like object (anything with a write method), the text
                is written to it and the function returns None.
            top_displayed_items: RankingTree only. The displayed item
                columns are the union of each leaf's top items by lowest
                expected rank; this argument sets how many per leaf.
                When None and the tree is a RankingTree, defaults to 3.
                Must be None for non-ranking trees.

        Returns:
            The text as a string when out_file is None; otherwise None.

        Raises:
            ValueError: If max_depth is a negative integer or not an integer.
            ValueError: If precision is not a non-negative integer.
            ValueError: If top_displayed_items is supplied for a non-ranking
                tree, is not a positive integer, or is less than 1.
            TypeError: If out_file is neither None, a string, nor a
                file-like object with a write method.
        """
        from . import _export

        result = _export.export_text(
            self,
            out_file,
            feature_names=feature_names,
            class_names=class_names,
            response_name=response_name,
            category_labels=category_labels,
            prediction_formatter=prediction_formatter,
            max_depth=max_depth,
            precision=precision,
            top_displayed_items=top_displayed_items,
        )
        return result

    @typing.overload
    def to_sql(
        self,
        out_file: None = None,
        target_class: None | object = None,
        feature_names: None | list[str] = None,
        category_labels: None | _CategoryLabels = None,
        max_depth: None | int = None,
    ) -> str: ...

    @typing.overload
    def to_sql(
        self,
        out_file: str | typing.IO[str],
        target_class: None | object = None,
        feature_names: None | list[str] = None,
        category_labels: None | _CategoryLabels = None,
        max_depth: None | int = None,
    ) -> None: ...

    def to_sql(
        self,
        out_file: None | str | typing.IO[str] = None,
        target_class: None | object = None,
        feature_names: None | list[str] = None,
        category_labels: None | _CategoryLabels = None,
        max_depth: None | int = None,
    ) -> None | str:
        """Serialize the fitted tree as a SQL CASE expression.

        When out_file is None the SQL is returned as a string; otherwise
        it is written to the destination and the function returns None.
        The emitted expression routes each input row to its leaf and
        returns the leaf's numeric prediction (regression mean, or
        target_class probability for classification, or the first
        survival metric value). Any value a node cannot route - an unseen
        category, or a missing value (NULL) at a node that learned no
        missing rule - evaluates to that node's own prediction, mirroring
        predict; a learned missing rule emits an explicit IS NULL branch.
        Branch ordering follows tree.reverse_order exactly like to_text
        and to_image. The first line is a SQL comment naming each
        referenced column and the column type the expression expects
        (numeric, boolean, or text); the expression assumes every column
        keeps its fit-time type.

        Args:
            out_file: Where to write the SQL. When None (the default), the
                SQL is returned. When a string, it is treated as a
                filesystem path; the file is opened with UTF-8 encoding,
                written, closed, and the function returns None. When a
                file-like object (anything with a write method), the SQL
                is written to it and the function returns None.
            target_class: Classification-only. Selects which class's
                probability is emitted at each leaf. Must equal one of
                the values in tree.classes_. When None (the default), the
                last class (tree.classes_[-1]) is used. Passing a
                non-None value for a non-ClassificationTree raises
                ValueError.
            feature_names: Optional display names for the covariates, one
                per column of X. Used as the double-quoted SQL column
                identifiers in the emitted conditions. See
                sigma.export_sql for full resolution rules.
            category_labels: Optional mapping from a categorical feature
                (column-name string or integer index) to a dict of
                {code: label}. When provided (or captured at fit time from
                a pandas categorical column), categorical comparisons emit
                label strings; otherwise they emit the numeric codes
                stored at fit time.
            max_depth: Maximum depth to render, with the root counted as
                depth 0. When None (the default), the full tree is
                rendered. When a non-negative integer, subtrees rooted
                below this depth are collapsed to a single leaf line
                carrying the truncated node's own prediction and a
                "-- Truncated at depth N" comment.

        Returns:
            The SQL CASE expression as a string when out_file is None;
            otherwise None.

        Raises:
            ValueError: If max_depth is a negative integer or not an
                integer.
            ValueError: If target_class is provided for a non-classification
                tree, or is not a value in tree.classes_.
            TypeError: If out_file is neither None, a string, nor a
                file-like object with a write method.
        """
        from . import _export

        result = _export.export_sql(
            self,
            out_file,
            target_class=target_class,
            feature_names=feature_names,
            category_labels=category_labels,
            max_depth=max_depth,
        )
        return result

    @typing.overload
    def to_image(
        self,
        format: typing.Literal["gif", "pdf", "png", "svg"],
        out_file: None = None,
        kind: typing.Literal["tree", "response"] = "tree",
        feature_names: None | list[str] = None,
        class_names: None | list[str] = None,
        response_name: None | str = None,
        category_labels: None | _CategoryLabels = None,
        prediction_formatter: None | typing.Callable[[float], str] = None,
        max_depth: None | int = None,
        precision: int = 3,
        max_branch_length: int = 60,
        orientation: typing.Literal["top-down", "left-to-right"] = "top-down",
        dpi: int = 192,
        root_colors: None | tuple[str, str, str] = None,
        split_colors: None | tuple[str, str, str] = None,
        leaf_palette: None | tuple[str, str, str] = None,
        background_color: None | str = None,
        foreground_color: None | str = None,
        top_displayed_items: None | int = None,
    ) -> bytes: ...

    @typing.overload
    def to_image(
        self,
        format: typing.Literal["gif", "pdf", "png", "svg"],
        out_file: str | typing.IO[bytes],
        kind: typing.Literal["tree", "response"] = "tree",
        feature_names: None | list[str] = None,
        class_names: None | list[str] = None,
        response_name: None | str = None,
        category_labels: None | _CategoryLabels = None,
        prediction_formatter: None | typing.Callable[[float], str] = None,
        max_depth: None | int = None,
        precision: int = 3,
        max_branch_length: int = 60,
        orientation: typing.Literal["top-down", "left-to-right"] = "top-down",
        dpi: int = 192,
        root_colors: None | tuple[str, str, str] = None,
        split_colors: None | tuple[str, str, str] = None,
        leaf_palette: None | tuple[str, str, str] = None,
        background_color: None | str = None,
        foreground_color: None | str = None,
        top_displayed_items: None | int = None,
    ) -> None: ...

    def to_image(
        self,
        format: typing.Literal["gif", "pdf", "png", "svg"],
        out_file: None | str | typing.IO[bytes] = None,
        kind: typing.Literal["tree", "response"] = "tree",
        feature_names: None | list[str] = None,
        class_names: None | list[str] = None,
        response_name: None | str = None,
        category_labels: None | _CategoryLabels = None,
        prediction_formatter: None | typing.Callable[[float], str] = None,
        max_depth: None | int = None,
        precision: int = 3,
        max_branch_length: int = 60,
        orientation: typing.Literal["top-down", "left-to-right"] = "top-down",
        dpi: int = 192,
        root_colors: None | tuple[str, str, str] = None,
        split_colors: None | tuple[str, str, str] = None,
        leaf_palette: None | tuple[str, str, str] = None,
        background_color: None | str = None,
        foreground_color: None | str = None,
        top_displayed_items: None | int = None,
    ) -> None | bytes:
        """Render the fitted tree as a GIF, PDF, PNG, or SVG image.

        When out_file is None the rendered bytes are returned; otherwise
        they are written to the destination and the function returns
        None. The kind parameter selects between the default decision
        tree diagram ("tree") and a per-leaf response summary
        ("response").

        Args:
            format: Output format, one of "gif", "pdf", "png", or "svg".
            out_file: Where to write the rendered bytes. When None (the
                default), the bytes are returned. When a string, it is
                treated as a filesystem path; the file is opened in binary
                write mode, written, closed, and the function returns None.
                When a file-like object (anything with a write method), the
                bytes are written to it and the function returns None.
            kind: Which visualization to render, one of "tree" (default,
                the decision tree diagram) or "response" (a per-leaf
                response summary chart). See sigma.export_image for the
                per-task layout and the parameters that apply to each
                kind.
            max_branch_length: Maximum number of characters per branch
                label describing a split condition. Longer labels are
                truncated with a trailing "..." in the rendered tree.
                Defaults to 60. Ignored when kind is "response".
            top_displayed_items: RankingTree only. The displayed item
                columns are the union of each leaf's top items by lowest
                expected rank; this argument sets how many per leaf.
                When None and the tree is a RankingTree, defaults to 3.
                Must be None for non-ranking trees.

        Returns:
            The rendered image bytes when out_file is None; otherwise None.

        Raises:
            ValueError: If format is not one of the supported formats.
            ValueError: If kind is not one of the supported kinds.
            ValueError: If max_depth is a negative integer or not an integer.
            ValueError: If precision is not a non-negative integer.
            ValueError: If dpi is not a positive integer.
            ValueError: If max_branch_length is not a positive integer.
            ValueError: If top_displayed_items is supplied for a non-ranking
                tree, is not a positive integer, or is less than 1.
            TypeError: If out_file is neither None, a string, nor a
                file-like object with a write method.
            ImportError: If graphviz is not installed when kind is
                "tree", or if matplotlib is not installed when kind is
                "response".
        """
        from . import _export

        result = _export.export_image(
            self,
            format,
            out_file,
            kind=kind,
            feature_names=feature_names,
            class_names=class_names,
            response_name=response_name,
            category_labels=category_labels,
            prediction_formatter=prediction_formatter,
            max_depth=max_depth,
            precision=precision,
            orientation=orientation,
            dpi=dpi,
            root_colors=root_colors,
            split_colors=split_colors,
            leaf_palette=leaf_palette,
            background_color=background_color,
            foreground_color=foreground_color,
            top_displayed_items=top_displayed_items,
            max_branch_length=max_branch_length,
        )
        return result

    def _repr_mimebundle_(self, **kwargs: typing.Any) -> dict[str, typing.Any]:
        """IPython rich-display: SVG for fitted trees, sklearn fallback otherwise."""
        if not hasattr(self, "content_"):
            parent = typing.cast(typing.Any, super())
            bundle = parent._repr_mimebundle_(**kwargs)
            return bundle
        svg_bytes = self.to_image("svg")
        bundle = {
            "text/plain": repr(self),
            "image/svg+xml": svg_bytes.decode("utf-8"),
        }
        return bundle


def _compact_node(node: _node.Node, depth: int) -> _node.Node:
    """Return the compacted copy of node placed at the given depth."""
    match node.extension:
        case _partition.Partition() as partition:
            result = _compact_internal(node, partition, depth)
        case _:
            result = node._copy()
            result.extension = _extension.Leaf()
            result.depth = depth
    return result


def _compact_internal(
    node: _node.Node, partition: _partition.Partition, depth: int
) -> _node.Node:
    """Return the compacted copy of an internal node, merging any same-feature
    chain rooted at it into one N-ary node."""
    feature_index = partition.feature_index
    chain_partitions: list[_partition.Partition] = []
    frontier: list[_node.Node] = []
    _collect_same_feature_chain(node, feature_index, chain_partitions, frontier)
    merged_partition: None | _partition.Partition = None
    if len(chain_partitions) > 1:
        match partition:
            case _partition.NumericalPartition():
                merged_partition = _merge_numeric_chain(
                    node, partition, chain_partitions, depth
                )
            case _partition.CategoricalPartition():
                merged_partition = _merge_categorical_chain(
                    node, partition, frontier, depth
                )
    if merged_partition is None:
        new_partition = _rebuild_partition(partition, depth)
    else:
        new_partition = merged_partition
    result = node._copy()
    result.extension = new_partition
    result.depth = depth
    return result


def _rebuild_partition(
    partition: _partition.Partition, depth: int
) -> _partition.Partition:
    """Return a copy of partition with compacted children, statistics kept."""
    new_children: list[_node.Node] = []
    for child in partition.children:
        new_children.append(_compact_node(child, depth + 1))
    new_partition = copy.copy(partition)
    new_partition.children = tuple(new_children)
    statistics = partition.statistics
    if statistics is not None:
        new_partition.statistics = _partition.SplitStatistics(
            statistics.p_value
        )
    return new_partition


def _merge_numeric_chain(
    node: _node.Node,
    partition: _partition.Partition,
    chain_partitions: list[_partition.Partition],
    depth: int,
) -> _partition.NumericalPartition:
    """Build the merged numeric partition for a same-feature chain at node."""
    feature_index = partition.feature_index
    threshold_values: set[int | float] = set()
    has_nan = False
    for chain_partition in chain_partitions:
        numeric_partition = typing.cast(
            _partition.NumericalPartition, chain_partition
        )
        for threshold in numeric_partition.thresholds:
            threshold_values.add(threshold)
        if numeric_partition.nan_child is not None:
            has_nan = True
    sorted_thresholds = sorted(threshold_values)
    probes: list[int | float] = list(sorted_thresholds)
    probes.append(numpy.inf)
    probe_frontiers: list[_node.Node] = []
    children: list[_node.Node] = []
    for probe in probes:
        frontier_node = _route_same_feature(node, feature_index, probe)
        probe_frontiers.append(frontier_node)
        children.append(_compact_node(frontier_node, depth + 1))
    nan_child: None | int = None
    if has_nan:
        nan_frontier = _route_same_feature(node, feature_index, numpy.nan)
        nan_child = _index_by_identity(probe_frontiers, nan_frontier)
        if nan_child is None:
            children.append(_compact_node(nan_frontier, depth + 1))
            nan_child = len(children) - 1
    merged = _partition.NumericalPartition(
        feature=partition.feature,
        statistics=None,
        children=tuple(children),
        thresholds=tuple(sorted_thresholds),
        nan_child=nan_child,
    )
    return merged


def _index_by_identity(
    nodes: list[_node.Node], target: _node.Node
) -> None | int:
    """Return the index of target within nodes by identity, or None."""
    for index, node in enumerate(nodes):
        if node is target:
            return index
    return None


def _merge_categorical_chain(
    node: _node.Node,
    partition: _partition.Partition,
    frontier: list[_node.Node],
    depth: int,
) -> _partition.CategoricalPartition:
    """Build the merged categorical partition for a same-feature chain."""
    feature_index = partition.feature_index
    categorical = typing.cast(_partition.CategoricalPartition, partition)
    observed = categorical.observed_categories
    frontier_categories: dict[int, set] = {}
    for frontier_node in frontier:
        frontier_categories[id(frontier_node)] = set()
    for category in observed:
        reached = _route_same_feature(node, feature_index, category)
        frontier_categories[id(reached)].add(category)
    category_groups: list[frozenset] = []
    children: list[_node.Node] = []
    for frontier_node in frontier:
        categories = frontier_categories[id(frontier_node)]
        category_groups.append(frozenset(categories))
        children.append(_compact_node(frontier_node, depth + 1))
    merged = _partition.CategoricalPartition(
        feature=categorical.feature,
        statistics=None,
        children=tuple(children),
        category_groups=tuple(category_groups),
    )
    return merged


def _collect_same_feature_chain(
    node: _node.Node,
    feature_index: int,
    chain_partitions: list[_partition.Partition],
    frontier: list[_node.Node],
) -> None:
    """Gather the contiguous same-feature partitions and their frontier nodes."""
    if _splits_on_feature(node, feature_index):
        partition = typing.cast(_partition.Partition, node.extension)
        chain_partitions.append(partition)
        for child in partition.children:
            _collect_same_feature_chain(
                child, feature_index, chain_partitions, frontier
            )
    else:
        frontier.append(node)


def _route_same_feature(
    start: _node.Node, feature_index: int, value: object
) -> _node.Node:
    """Descend from start following only splits on feature_index."""
    current = start
    while _splits_on_feature(current, feature_index):
        partition = typing.cast(_partition.Partition, current.extension)
        child = partition.route(value)
        if child is None:
            break
        current = child
    return current


def _splits_on_feature(node: _node.Node, feature_index: int) -> bool:
    """Whether node is an internal node splitting on feature_index."""
    extension = node.extension
    if isinstance(extension, _partition.Partition):
        return extension.feature_index == feature_index
    return False


def _extract_response_name(y: typing.Any) -> None | str:
    """Capture y.name from a pandas Series, or first column from DataFrame."""
    name_attr = getattr(y, "name", None)
    if name_attr is not None and isinstance(name_attr, str):
        return name_attr
    columns = getattr(y, "columns", None)
    if columns is not None and len(columns) > 0:
        first = columns[0]
        if isinstance(first, str):
            return first
    return None


def _preprocess_dataframe_X(
    X: typing.Any,
) -> tuple[typing.Any, dict[int, _feature.Feature]]:
    """Encode boolean and categorical DataFrame columns of X to float codes, raising on string, object, or otherwise unsupported column dtypes. Missing values become a dedicated N/A level; a boolean column with missing values is promoted to a categorical. Returns the encoded X and the Feature of each column it typed, keyed by column index."""
    columns = getattr(X, "columns", None)
    if columns is None:
        return X, {}
    features: dict[int, _feature.Feature] = {}
    encoded_data: dict[typing.Any, typing.Any] = {}
    for index, name in enumerate(columns):
        column = X[name]
        kind = _column_kind(column)
        match kind:
            case "boolean" if _has_missing(column):
                codes, label_map = _encode_promoted_boolean(column)
                encoded_data[name] = codes
                features[index] = _feature.PromotedBooleanFeature(
                    index, category_labels=label_map, na_code=2.0
                )
            case "boolean":
                encoded_data[name] = _encode_boolean_column(column)
                features[index] = _feature.BooleanFeature(index)
            case "categorical":
                levels, codes, na_label = _categorical_levels_and_codes(column)
                encoded_data[name] = codes
                label_map = {
                    float(code): str(level) for code, level in enumerate(levels)
                }
                na_code: None | float = None
                if na_label is not None:
                    na_code = float(len(levels))
                    label_map[na_code] = na_label
                features[index] = _feature.CategoricalFeature(
                    index, category_labels=label_map, na_code=na_code
                )
            case "numeric":
                encoded_data[name] = numpy.asarray(column)
            case _:
                message = _unsupported_column_message(name, column)
                raise ValueError(message)
    if not features:
        return X, {}
    new_X = _rebuild_dataframe(X, encoded_data)
    return new_X, features


def _apply_categorical_encoding(
    X: typing.Any,
    features: tuple[_feature.Feature, ...],
) -> typing.Any:
    """Re-encode predict-time categorical columns to float codes using the fit-time category labels. A missing value maps to the column's N/A code when one was learned; any other unseen value maps to an unroutable sentinel code so it falls through to the holding node's prediction."""
    columns = getattr(X, "columns", None)
    if columns is None:
        return X
    encoded_data: dict[typing.Any, typing.Any] = {}
    changed = False
    feature_count = len(features)
    for index, name in enumerate(columns):
        column = X[name]
        labels: None | dict[float, str] = None
        na_code: None | float = None
        if index < feature_count:
            feature = features[index]
            if isinstance(feature, _feature.CategoricalFeature):
                labels = feature.category_labels
                na_code = feature.na_code
        if labels is None:
            encoded_data[name] = numpy.asarray(column)
            continue
        changed = True
        level_to_code = {
            label: code for code, label in labels.items() if code != na_code
        }
        missing = _missing_mask(column)
        values = _column_values(column)
        codes = numpy.empty(len(values), dtype=numpy.float64)
        for position, value in enumerate(values):
            if missing[position]:
                codes[position] = -1.0 if na_code is None else na_code
            else:
                codes[position] = level_to_code.get(str(value), -1.0)
        encoded_data[name] = codes
    if not changed:
        return X
    new_X = _rebuild_dataframe(X, encoded_data)
    return new_X


def _column_kind(column: typing.Any) -> str:
    """Classify a DataFrame column as boolean, categorical, numeric, or unsupported."""
    pandas = sys.modules.get("pandas")
    if pandas is not None and isinstance(column, pandas.Series):
        dtype = column.dtype
        if pandas.api.types.is_bool_dtype(dtype):
            return "boolean"
        if isinstance(dtype, pandas.CategoricalDtype):
            return "categorical"
        if pandas.api.types.is_numeric_dtype(dtype):
            return "numeric"
        return "unsupported"
    polars = sys.modules.get("polars")
    if polars is not None and isinstance(column, polars.Series):
        dtype = column.dtype
        if dtype == polars.Boolean:
            return "boolean"
        if dtype in (polars.Categorical, polars.Enum):
            return "categorical"
        if dtype.is_numeric():
            return "numeric"
        return "unsupported"
    return "unsupported"


def _reject_boolean_bare_values(X: typing.Any) -> None:
    """Raise when a plain array or list of X carries boolean values."""
    array = numpy.asarray(X)
    has_boolean = array.dtype.kind == "b"
    if not has_boolean and array.dtype.kind == "O":
        for value in array.flat:
            if isinstance(value, (bool, numpy.bool_)):
                has_boolean = True
                break
    if has_boolean:
        raise ValueError(
            "X contains boolean values in a plain array or list; supply"
            " numbers only, or use a pandas or polars DataFrame with typed"
            " boolean columns"
        )


def _encode_boolean_column(
    column: typing.Any,
) -> numpy.typing.NDArray[numpy.float64]:
    """Cast a boolean column with no missing values to float 0.0 / 1.0."""
    array = numpy.asarray(column)
    boolean_array = array.astype(bool)
    float_array = boolean_array.astype(numpy.float64)
    return float_array


def _encode_promoted_boolean(
    column: typing.Any,
) -> tuple[numpy.typing.NDArray[numpy.float64], dict[float, str]]:
    """Encode a boolean column with missing values as categorical float codes (False=0, True=1, missing=2) with a collision-free N/A label."""
    missing = _missing_mask(column)
    values = _column_values(column)
    codes = numpy.empty(len(values), dtype=numpy.float64)
    for position, value in enumerate(values):
        if missing[position]:
            codes[position] = 2.0
        elif bool(value):
            codes[position] = 1.0
        else:
            codes[position] = 0.0
    na_label = _free_na_label(["False", "True"])
    label_map = {0.0: "False", 1.0: "True", 2.0: na_label}
    return codes, label_map


def _categorical_levels_and_codes(
    column: typing.Any,
) -> tuple[list[typing.Any], numpy.typing.NDArray[numpy.float64], None | str]:
    """Return a categorical column's real levels, its float codes (missing rows mapped to a new trailing N/A code), and the N/A label (None when the column has no missing values)."""
    pandas = sys.modules.get("pandas")
    polars = sys.modules.get("polars")
    if pandas is not None and isinstance(column, pandas.Series):
        categorical = column.cat
        levels = categorical.categories.tolist()
        codes = numpy.asarray(categorical.codes, dtype=numpy.float64)
    elif polars is not None and column.dtype == polars.Enum:
        levels = column.dtype.categories.to_list()
        codes = numpy.asarray(column.to_physical(), dtype=numpy.float64)
    else:
        levels = column.unique(maintain_order=True).drop_nulls().to_list()
        code_of = {level: float(index) for index, level in enumerate(levels)}
        values = _column_values(column)
        codes = numpy.empty(len(values), dtype=numpy.float64)
        for position, value in enumerate(values):
            codes[position] = code_of.get(value, numpy.nan)
    missing = _missing_mask(column)
    na_label = None
    if missing.any():
        codes = numpy.where(missing, float(len(levels)), codes)
        na_label = _free_na_label([str(level) for level in levels])
    return levels, codes, na_label


def _free_na_label(used: list[str]) -> str:
    """Return "N/A", or "N/A 2", "N/A 3", ... - the first not already used."""
    used_labels = set(used)
    if "N/A" not in used_labels:
        return "N/A"
    suffix = 2
    while f"N/A {suffix}" in used_labels:
        suffix += 1
    label = f"N/A {suffix}"
    return label


def _has_missing(column: typing.Any) -> bool:
    """Whether a pandas or polars column contains any missing value."""
    mask = _missing_mask(column)
    return bool(mask.any())


def _missing_mask(column: typing.Any) -> numpy.typing.NDArray[numpy.bool_]:
    """Per-element missing-value mask for a pandas or polars column."""
    if hasattr(column, "isna"):
        mask = numpy.asarray(column.isna(), dtype=bool)
        return mask
    if hasattr(column, "is_null"):
        mask = numpy.asarray(column.is_null(), dtype=bool)
        return mask
    array = numpy.asarray(column)
    mask = numpy.not_equal(array, array)
    return mask


def _column_values(column: typing.Any) -> list[typing.Any]:
    """Return a DataFrame column's values as a Python list."""
    if hasattr(column, "to_list"):
        values = column.to_list()
        return values
    if hasattr(column, "tolist"):
        values = column.tolist()
        return values
    array = numpy.asarray(column)
    values = array.tolist()
    return values


def _rebuild_dataframe(
    X: typing.Any, encoded_data: dict[typing.Any, typing.Any]
) -> typing.Any:
    """Build a same-typed DataFrame from the encoded column arrays, preserving column order."""
    pandas = sys.modules.get("pandas")
    if pandas is not None and isinstance(X, pandas.DataFrame):
        rebuilt = pandas.DataFrame(encoded_data, index=X.index)
        return rebuilt
    constructor = type(X)
    rebuilt = constructor(encoded_data)
    return rebuilt


def _unsupported_column_message(name: typing.Any, column: typing.Any) -> str:
    """Build the error message for a column whose dtype sigma cannot encode."""
    dtype = getattr(column, "dtype", None)
    message = (
        f"column {name!r} has unsupported dtype {dtype!r}; sigma's Tree"
        f" estimators accept only numeric, boolean, and categorical"
        f" columns. Cast string or object columns to a categorical dtype"
        f" before fitting (pandas: df[{name!r}] ="
        f" df[{name!r}].astype('category'); polars: df ="
        f" df.with_columns(polars.col({name!r}).cast(polars.Categorical)))."
    )
    return message
