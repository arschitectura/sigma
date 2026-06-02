"""Abstract Tree base class for conditional inference tree estimators.

Implements the shared recursive partitioning algorithm from Hothorn,
Hornik, and Zeileis (2006). The concrete ClassificationTree,
RegressionTree, SurvivalTree, and RankingTree subclasses live in their
own sibling modules.
"""

from __future__ import annotations

import abc
import collections.abc
import typing

import numpy
import numpy.typing
import scipy.sparse
import sklearn.base
import sklearn.utils.validation

from . import _extension
from . import _node
from . import _partition
from . import _splitting
from . import _statistics
from . import _types

if typing.TYPE_CHECKING:
    import pandas

# TODO XXX review all shared data structures


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
        leaves_: List of leaf nodes, ordered by ascending prediction value.
        nodes_: List of all nodes in pre-order DFS, ordered by node_id
            (so nodes_[k].node_id == k). Indices match the output of
            predict_index.
        n_features_in_: Number of features seen during fit.
        feature_types_: Per-feature CovariateType, shape (n_features,).
        response_name_in_: Display name captured from a named pandas Series y
            (or, for survival, the first column name of a DataFrame y) at fit
            time. None when y carries no usable name.
        category_labels_in_: Auto-extracted category-label maps, keyed by
            column index, captured from pandas categorical-dtype and
            object-dtype columns of X at fit time. None when X is a numpy
            array or contains no categorical / object columns.
        boolean_features_in_: Frozenset of column indices flagged as boolean
            (pandas BooleanDtype or numpy bool) at fit time. None when X is a
            numpy array or contains no boolean columns.
    """

    # sklearn.base.BaseEstimator forbids __slots__ on subclasses (its
    # __getstate__ raises TypeError if any subclass declares __slots__).
    # Instances inherit __dict__ and __weakref__ from BaseEstimator.

    content_: N
    leaves_: list[N]
    nodes_: list[N]
    n_features_in_: int
    feature_types_: numpy.typing.NDArray
    response_name_in_: None | str
    category_labels_in_: None | dict[int, dict[float, str]]
    boolean_features_in_: None | frozenset[int]
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

    def fit(
        self,
        X: numpy.typing.NDArray[numpy.floating] | pandas.DataFrame,
        y: (
            numpy.typing.NDArray[numpy.floating]
            | pandas.Series
            | pandas.DataFrame
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
                non-finite or negative values; if resamples is not a positive
                integer; if test_type="monte_carlo" and resamples is None; if
                categorical_features contains a string label without a
                DataFrame name source, or a string label that is not among the
                DataFrame columns; if any response value violates the domain
                required by the chosen ci_method (y > 0 for "log_normal", y >= 0
                for "gamma" / "poisson" / "exponential", y in [0, 1] for
                "beta"); or if offset has the wrong shape, contains non-finite
                values, or violates the per-task domain.
        """
        with numpy.errstate(divide="ignore", invalid="ignore", over="ignore"):
            self.response_name_in_ = _extract_response_name(y)
            (
                X,
                self.category_labels_in_,
                self.boolean_features_in_,
            ) = _preprocess_dataframe_X(X)
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
            self.feature_types_ = self._build_feature_types(X, weights, names)
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
                names=names,
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

    def _build_feature_types(
        self,
        X: numpy.typing.NDArray[numpy.floating],
        weights: numpy.typing.NDArray[numpy.floating],
        names: None | numpy.typing.NDArray,
    ) -> numpy.typing.NDArray:
        """Classify each column as BOOLEAN, CATEGORICAL, INTEGER, or REAL."""
        n_features = X.shape[1]
        active = weights > 0
        types = numpy.empty(n_features, dtype=object)
        for j in range(n_features):
            column = X[active, j]
            if column.size > 0 and numpy.all(numpy.mod(column, 1) == 0):
                types[j] = _types.CovariateType.INTEGER
            else:
                types[j] = _types.CovariateType.REAL
        category_labels_in = self.category_labels_in_
        if category_labels_in is not None:
            for index in category_labels_in:
                types[index] = _types.CovariateType.CATEGORICAL
        categorical_features = self.categorical_features
        if categorical_features is not None:
            for entry in categorical_features:
                index = self._resolve_categorical_entry(entry, names)
                types[index] = _types.CovariateType.CATEGORICAL
        boolean_features_in = self.boolean_features_in_
        if boolean_features_in is not None:
            for index in boolean_features_in:
                types[index] = _types.CovariateType.BOOLEAN
        return types

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
        """Return the active response-name source, or None if unset."""
        if response_name is not None:
            return response_name
        name_in = getattr(self, "response_name_in_", None)
        return name_in

    def _effective_class_names(
        self,
        class_names: None | list[str] = None,
    ) -> None | list[str]:
        """Return the active class-name source, or None if unset."""
        return class_names

    def _resolve_category_labels(
        self,
        category_labels: None | _CategoryLabels,
        names: None | numpy.typing.NDArray,
    ) -> None | dict[int, dict[float, str]]:
        """Resolve string keys in category_labels to integer indices, merging
        with the auto-extracted category_labels_in_ map.
        """
        auto = getattr(self, "category_labels_in_", None)
        if category_labels is None and auto is None:
            return None
        merged: dict[int, dict[float, str]] = {}
        if auto is not None:
            for index, mapping in auto.items():
                merged[index] = dict(mapping)
        if category_labels is not None:
            for key, mapping in category_labels.items():
                index = Tree._resolve_categorical_entry(key, names)
                merged[index] = dict(mapping)
        return merged

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
        weights = sample_weight.copy()
        return weights

    def predict_index(
        self,
        X: numpy.typing.NDArray[numpy.floating] | pandas.DataFrame,
    ) -> numpy.typing.NDArray[numpy.intp]:
        """Predict node indices for the given samples.

        Args:
            X: Samples to predict, shape (n_samples, n_features).

        Returns:
            Node indices into self.nodes_, shape (n_samples,). When a
            sample's categorical value is not routable at an internal
            node, the index is that of the holding node rather than a
            descendant leaf.
        """
        sklearn.utils.validation.check_is_fitted(self, "content_")
        X = _apply_categorical_encoding(X, self.category_labels_in_)
        X_validated = sklearn.utils.validation.validate_data(
            self, X, reset=False, dtype="float64"
        )
        X_array = typing.cast(numpy.typing.NDArray[numpy.floating], X_validated)
        n_samples = X_array.shape[0]
        indices = numpy.empty(n_samples, dtype=numpy.intp)
        for i in range(n_samples):
            node = self.content_.traverse(X_array[i])
            indices[i] = node.node_id
        return indices

    def apply(
        self,
        X: numpy.typing.NDArray[numpy.floating] | pandas.DataFrame,
    ) -> numpy.typing.NDArray[numpy.intp]:
        """Return the node_id of the node each sample is routed to.

        Args:
            X: Samples to predict, shape (n_samples, n_features).

        Returns:
            Node ids, shape (n_samples,). For samples that reach a leaf
            the id is that leaf's; for samples whose categorical value
            is not routable at an internal node, the id is that of the
            holding node.
        """
        sklearn.utils.validation.check_is_fitted(self, "content_")
        X = _apply_categorical_encoding(X, self.category_labels_in_)
        X_validated = sklearn.utils.validation.validate_data(
            self, X, reset=False, dtype="float64"
        )
        X_array = typing.cast(numpy.typing.NDArray[numpy.floating], X_validated)
        n_samples = X_array.shape[0]
        ids = numpy.empty(n_samples, dtype=numpy.intp)
        for i in range(n_samples):
            node = self.content_.traverse(X_array[i])
            ids[i] = node.node_id
        return ids

    def decision_path(
        self,
        X: numpy.typing.NDArray[numpy.floating] | pandas.DataFrame,
    ) -> scipy.sparse.csr_matrix:
        """Return the decision path indicator matrix for the given samples.

        Args:
            X: Samples to predict, shape (n_samples, n_features).

        Returns:
            A scipy.sparse.csr_matrix of shape (n_samples, len(nodes_)),
            dtype numpy.intp, with 1s on visited nodes and 0s elsewhere.
            For samples whose categorical value is not routable at an
            internal node, the path ends at that holding node.
        """
        sklearn.utils.validation.check_is_fitted(self, "content_")
        X = _apply_categorical_encoding(X, self.category_labels_in_)
        X_validated = sklearn.utils.validation.validate_data(
            self, X, reset=False, dtype="float64"
        )
        X_array = typing.cast(numpy.typing.NDArray[numpy.floating], X_validated)
        n_samples = X_array.shape[0]
        indptr = numpy.empty(n_samples + 1, dtype=numpy.intp)
        indptr[0] = 0
        indices: list[int] = []
        for i in range(n_samples):
            x = X_array[i]
            node: _node.Node = self.content_
            while True:
                indices.append(node.node_id)
                match node.extension:
                    case _partition.Partition() as partition:
                        value = x[partition.feature_index]
                        # TODO: better distinguish this from new, unseen, categorical levels
                        child = partition.route(value)
                        if child is None:
                            break
                        node = child
                    case _:
                        break
            indptr[i + 1] = len(indices)
        indices_array = numpy.asarray(indices, dtype=numpy.intp)
        data = numpy.ones(indices_array.shape[0], dtype=numpy.intp)
        path = scipy.sparse.csr_matrix(
            (data, indices_array, indptr),
            shape=(n_samples, len(self.nodes_)),
            dtype=numpy.intp,
        )
        return path

    def _build_leaves(self) -> None:
        """Populate leaves_ and assign each leaf its leaf_id."""
        raw_leaves = self.content_.leaves()
        sorted_leaves = sorted(raw_leaves, key=lambda n: n.leaf_sort_key())
        if self.reverse_order:
            sorted_leaves = list(reversed(sorted_leaves))
        self.leaves_ = sorted_leaves
        for index, leaf in enumerate(sorted_leaves):
            leaf_extension = typing.cast(_extension.Leaf, leaf.extension)
            leaf_extension.leaf_id = index

    def _assign_node_ids(self) -> None:
        """Pre-order DFS: assign node_id to every node and populate nodes_."""
        collected: list[_node.Node] = []
        stack: list[_node.Node] = [self.content_]
        while stack:
            node = stack.pop()
            node.node_id = len(collected)
            collected.append(node)
            match node.extension:
                case _partition.Partition() as partition:
                    stack.append(partition.right)
                    stack.append(partition.left)
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
        names: None | numpy.typing.NDArray,
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
            names: Effective feature names for this fit, or None when no name
                source is available. Used to populate split_feature_name on each
                internal node.
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
        prediction = self._compute_prediction(
            y_transmuted, w_transmuted, offset_transmuted
        )
        ci_low, ci_high = self._compute_ci(
            y_transmuted, w_transmuted, offset_transmuted
        )
        ci_low_per_class, ci_high_per_class = self._compute_per_class_ci(
            y_transmuted, w_transmuted
        )
        n_samples = int(numpy.count_nonzero(w_transmuted))
        is_constant = self._is_constant_response(
            y_transmuted, w_transmuted, offset_transmuted
        )
        if (
            w_sum < self.min_splits
            or (self.max_depth is not None and depth >= self.max_depth)
            or is_constant
        ):
            leaf = self._create_leaf(
                depth,
                y_transmuted,
                w_transmuted,
                weights,
                offset_transmuted,
            )
            self._apply_decorator(leaf, X, y, weights, side_data, offset)
            return leaf
        selection = _statistics.select_variable(
            X,
            h,
            weights,
            self.feature_types_,
            self.test_stat_enum_,
            self.test_type_enum_,
            self.alpha,
            self.correlation_enum_,
            resamples=self.resamples,
            rng=self._rng_,
        )
        if selection is None:
            leaf = self._create_leaf(
                depth,
                y_transmuted,
                w_transmuted,
                weights,
                offset_transmuted,
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
            self.feature_types_,
            self.test_stat_enum_,
            self.min_buckets,
            self.correlation_enum_,
        )
        if split_result is None:
            leaf = self._create_leaf(
                depth,
                y_transmuted,
                w_transmuted,
                weights,
                offset_transmuted,
            )
            self._apply_decorator(leaf, X, y, weights, side_data, offset)
            return leaf
        split_criterion, _test_statistic = split_result
        split_threshold: None | int | float = None
        left_categories: None | frozenset = None
        right_categories: None | frozenset = None
        match self.feature_types_[feature_index]:
            case _types.CovariateType.BOOLEAN:
                left_mask = X[:, feature_index] <= 0.5
            case _types.CovariateType.CATEGORICAL:
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
            case _types.CovariateType.INTEGER:
                integer_criterion = typing.cast(float, split_criterion)
                left_mask = X[:, feature_index] <= integer_criterion
                split_threshold = int(integer_criterion)
            case _types.CovariateType.REAL:
                real_criterion = typing.cast(float, split_criterion)
                left_mask = X[:, feature_index] <= real_criterion
                split_threshold = float(real_criterion)
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
                leaf = self._create_leaf(
                    depth,
                    y_transmuted,
                    w_transmuted,
                    weights,
                    offset_transmuted,
                )
                self._apply_decorator(leaf, X, y, weights, side_data, offset)
                return leaf
            p_value = max(p_value, transmuted_p)
        left_child = self._build_tree(
            X,
            y,
            h,
            left_weights,
            depth + 1,
            y_transmuted=y_left_transmuted,
            w_transmuted=w_left_transmuted,
            names=names,
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
            names=names,
            sample_weight=sample_weight,
            side_data=side_data,
            offset=offset,
            offset_transmuted=offset_right_transmuted,
        )
        class_distribution = self._compute_class_distribution(
            y_transmuted, w_transmuted
        )
        survival_function = self._compute_survival_function(
            y_transmuted, w_transmuted
        )
        survival_log_variance = self._compute_survival_log_variance(
            y_transmuted, w_transmuted
        )
        survival_metrics = self._compute_survival_metrics(
            y_transmuted, w_transmuted
        )
        ranking_metrics = self._compute_ranking_metrics(
            y_transmuted, w_transmuted
        )
        mean_offset_proba = self._compute_mean_offset_proba(
            w_transmuted, offset_transmuted
        )
        split_name = None if names is None else str(names[feature_index])
        T = selection.T
        mu = selection.mu
        Sigma = selection.Sigma
        partition: _partition.Partition[_node.Node]
        match self.feature_types_[feature_index]:
            case _types.CovariateType.BOOLEAN:
                partition = _partition.BooleanPartition(
                    feature_index=feature_index,
                    feature_name=split_name,
                    p_value=p_value,
                    T=T,
                    mu=mu,
                    Sigma=Sigma,
                    left=left_child,
                    right=right_child,
                )
            case _types.CovariateType.CATEGORICAL:
                partition = _partition.CategoricalPartition(
                    feature_index=feature_index,
                    feature_name=split_name,
                    p_value=p_value,
                    T=T,
                    mu=mu,
                    Sigma=Sigma,
                    left=left_child,
                    right=right_child,
                    left_categories=typing.cast(frozenset, left_categories),
                    right_categories=typing.cast(frozenset, right_categories),
                )
            case _types.CovariateType.INTEGER | _types.CovariateType.REAL:
                partition = _partition.NumericalPartition(
                    feature_index=feature_index,
                    feature_name=split_name,
                    p_value=p_value,
                    T=T,
                    mu=mu,
                    Sigma=Sigma,
                    left=left_child,
                    right=right_child,
                    threshold=typing.cast(int | float, split_threshold),
                )
        node = self._make_node(
            depth=depth,
            n_samples=n_samples,
            extension=partition,
            prediction=prediction,
            ci_low=ci_low,
            ci_high=ci_high,
            ci_low_per_class=ci_low_per_class,
            ci_high_per_class=ci_high_per_class,
            class_distribution=class_distribution,
            survival_function=survival_function,
            survival_log_variance=survival_log_variance,
            survival_metrics=survival_metrics,
            ranking_metrics=ranking_metrics,
            mean_offset_proba=mean_offset_proba,
            response_samples=None,
        )
        self._apply_decorator(node, X, y, weights, side_data, offset)
        return node

    @abc.abstractmethod
    def _make_node(
        self,
        depth: int,
        n_samples: int,
        extension: _extension.Extension,
        prediction: float,
        ci_low: None | float,
        ci_high: None | float,
        ci_low_per_class: None | numpy.typing.NDArray[numpy.floating],
        ci_high_per_class: None | numpy.typing.NDArray[numpy.floating],
        class_distribution: None | numpy.typing.NDArray[numpy.floating],
        survival_function: None
        | tuple[
            numpy.typing.NDArray[numpy.floating],
            numpy.typing.NDArray[numpy.floating],
        ],
        survival_log_variance: None | numpy.typing.NDArray[numpy.floating],
        survival_metrics: None | list[_node.SurvivalMetric],
        ranking_metrics: None | list[_node.RankingMetric],
        mean_offset_proba: None | numpy.typing.NDArray[numpy.floating],
        response_samples: None | numpy.typing.NDArray[numpy.floating],
    ) -> N:
        """Construct a task-specific Node from the per-task computed payloads."""

    def _compute_response_samples_for_leaf(
        self,
        y_transmuted: numpy.typing.NDArray[numpy.floating],
        w_transmuted: numpy.typing.NDArray[numpy.floating],
        offset_transmuted: None | numpy.typing.NDArray[numpy.floating],
    ) -> None | numpy.typing.NDArray[numpy.floating]:
        """Compute the per-leaf response sample array."""
        return None

    @abc.abstractmethod
    def _validate_fit_params(
        self,
        X: numpy.typing.NDArray[numpy.floating] | pandas.DataFrame,
        y: (
            numpy.typing.NDArray[numpy.floating]
            | pandas.Series
            | pandas.DataFrame
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

    @abc.abstractmethod
    def _compute_influence(
        self,
        y: numpy.typing.NDArray[numpy.floating],
        offset: None | numpy.typing.NDArray[numpy.floating],
    ) -> numpy.typing.NDArray[numpy.floating]:
        """Compute the influence function h from the target."""

    @abc.abstractmethod
    def _compute_prediction(
        self,
        y: numpy.typing.NDArray[numpy.floating],
        weights: numpy.typing.NDArray[numpy.floating],
        offset: None | numpy.typing.NDArray[numpy.floating],
    ) -> float:
        """Compute the node prediction value."""

    def _compute_mean_offset_proba(
        self,
        weights: numpy.typing.NDArray[numpy.floating],
        offset: None | numpy.typing.NDArray[numpy.floating],
    ) -> None | numpy.typing.NDArray[numpy.floating]:
        """Compute the per-leaf weighted mean of offset probability rows.

        Default implementation returns None; ClassificationTree overrides it.
        """
        return None

    @abc.abstractmethod
    def _is_constant_response(
        self,
        y: numpy.typing.NDArray[numpy.floating],
        weights: numpy.typing.NDArray[numpy.floating],
        offset: None | numpy.typing.NDArray[numpy.floating],
    ) -> bool:
        """Check whether the response is constant in this node."""

    @abc.abstractmethod
    def _compute_ci(
        self,
        y: numpy.typing.NDArray[numpy.floating],
        weights: numpy.typing.NDArray[numpy.floating],
        offset: None | numpy.typing.NDArray[numpy.floating],
    ) -> tuple[None | float, None | float]:
        """Compute the confidence interval for the node prediction."""

    @abc.abstractmethod
    def _compute_per_class_ci(
        self,
        y: numpy.typing.NDArray[numpy.floating],
        weights: numpy.typing.NDArray[numpy.floating],
    ) -> tuple[
        None | numpy.typing.NDArray[numpy.floating],
        None | numpy.typing.NDArray[numpy.floating],
    ]:
        """Compute per-class CI bounds (classification only)."""

    @abc.abstractmethod
    def _compute_class_distribution(
        self,
        y: numpy.typing.NDArray[numpy.floating],
        weights: numpy.typing.NDArray[numpy.floating],
    ) -> None | numpy.typing.NDArray[numpy.floating]:
        """Compute the class distribution for a node."""

    @abc.abstractmethod
    def _compute_survival_function(
        self,
        y: numpy.typing.NDArray[numpy.floating],
        weights: numpy.typing.NDArray[numpy.floating],
    ) -> (
        None
        | tuple[
            numpy.typing.NDArray[numpy.floating],
            numpy.typing.NDArray[numpy.floating],
        ]
    ):
        """Compute the survival function for a node."""

    @abc.abstractmethod
    def _compute_survival_log_variance(
        self,
        y: numpy.typing.NDArray[numpy.floating],
        weights: numpy.typing.NDArray[numpy.floating],
    ) -> None | numpy.typing.NDArray[numpy.floating]:
        """Compute the Greenwood variance of log S(t) for a node."""

    @abc.abstractmethod
    def _compute_survival_metrics(
        self,
        y: numpy.typing.NDArray[numpy.floating],
        weights: numpy.typing.NDArray[numpy.floating],
    ) -> None | list[_node.SurvivalMetric]:
        """Compute the per-node summary metrics for the display stack."""

    @abc.abstractmethod
    def _compute_ranking_metrics(
        self,
        y: numpy.typing.NDArray[numpy.floating],
        weights: numpy.typing.NDArray[numpy.floating],
    ) -> None | list[_node.RankingMetric]:
        """Compute the per-item ranking metrics for the display stack."""

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

    def _create_leaf(
        self,
        depth: int,
        y_transmuted: numpy.typing.NDArray[numpy.floating],
        w_transmuted: numpy.typing.NDArray[numpy.floating],
        weights: numpy.typing.NDArray[numpy.floating],
        offset_transmuted: None | numpy.typing.NDArray[numpy.floating],
    ) -> N:
        """Build a leaf node from transmuted data."""
        prediction = self._compute_prediction(
            y_transmuted, w_transmuted, offset_transmuted
        )
        n_samples = int(numpy.count_nonzero(w_transmuted))
        ci_low, ci_high = self._compute_ci(
            y_transmuted, w_transmuted, offset_transmuted
        )
        ci_low_per_class, ci_high_per_class = self._compute_per_class_ci(
            y_transmuted, w_transmuted
        )
        class_distribution = self._compute_class_distribution(
            y_transmuted, w_transmuted
        )
        survival_function = self._compute_survival_function(
            y_transmuted, w_transmuted
        )
        survival_log_variance = self._compute_survival_log_variance(
            y_transmuted, w_transmuted
        )
        survival_metrics = self._compute_survival_metrics(
            y_transmuted, w_transmuted
        )
        ranking_metrics = self._compute_ranking_metrics(
            y_transmuted, w_transmuted
        )
        mean_offset_proba = self._compute_mean_offset_proba(
            w_transmuted, offset_transmuted
        )
        response_samples = self._compute_response_samples_for_leaf(
            y_transmuted, w_transmuted, offset_transmuted
        )
        leaf = self._make_node(
            depth=depth,
            n_samples=n_samples,
            extension=_extension.Leaf(),
            prediction=prediction,
            ci_low=ci_low,
            ci_high=ci_high,
            ci_low_per_class=ci_low_per_class,
            ci_high_per_class=ci_high_per_class,
            class_distribution=class_distribution,
            survival_function=survival_function,
            survival_log_variance=survival_log_variance,
            survival_metrics=survival_metrics,
            ranking_metrics=ranking_metrics,
            mean_offset_proba=mean_offset_proba,
            response_samples=response_samples,
        )
        return leaf

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
        survival metric value). Categorical values not seen during
        training evaluate to the holding node's prediction, mirroring
        predict. Other unmatched inputs (e.g. NULL at a numerical or
        boolean split) fall through to ELSE NULL. Branch ordering
        follows tree.reverse_order exactly like to_text and to_image.

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
                {code: label}. When provided (or available via
                category_labels_in_), categorical comparisons emit
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
        max_branch_length: int = 50,
        orientation: typing.Literal["top-down", "left-to-right"] = "top-down",
        dpi: int = 192,
        root_colors: None | tuple[str, str, str] = None,
        split_colors: None | tuple[str, str, str] = None,
        leaf_colors: None | tuple[str, str, str] = None,
        background_color: None | str = None,
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
        max_branch_length: int = 50,
        orientation: typing.Literal["top-down", "left-to-right"] = "top-down",
        dpi: int = 192,
        root_colors: None | tuple[str, str, str] = None,
        split_colors: None | tuple[str, str, str] = None,
        leaf_colors: None | tuple[str, str, str] = None,
        background_color: None | str = None,
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
        max_branch_length: int = 50,
        orientation: typing.Literal["top-down", "left-to-right"] = "top-down",
        dpi: int = 192,
        root_colors: None | tuple[str, str, str] = None,
        split_colors: None | tuple[str, str, str] = None,
        leaf_colors: None | tuple[str, str, str] = None,
        background_color: None | str = None,
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
                Defaults to 50. Ignored when kind is "response".
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
                "tree", or if cairosvg is not installed when requesting
                PDF or PNG output with kind="tree" or GIF output with
                kind="response", or if matplotlib is not installed when
                kind is "response".
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
            leaf_colors=leaf_colors,
            background_color=background_color,
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
) -> tuple[
    typing.Any,
    None | dict[int, dict[float, str]],
    None | frozenset[int],
]:
    """Encode categorical, string, object, and boolean DataFrame columns to
    float codes.

    Returns the (possibly rewritten) X, the per-column categorical label
    map keyed by column index, and a frozenset of column indices flagged
    as boolean (pandas BooleanDtype or numpy bool). For non-DataFrame X or
    DataFrames with no categorical / string / object / boolean columns, X
    is returned unchanged and both auxiliary outputs are None. The new
    DataFrame keeps the original column names and order so sklearn's
    feature_names_in_ extraction is unaffected.
    """
    columns_attr = getattr(X, "columns", None)
    if columns_attr is None:
        return X, None, None
    import pandas

    if not isinstance(X, pandas.DataFrame):
        return X, None, None
    labels: dict[int, dict[float, str]] = {}
    boolean_indices: set[int] = set()
    encoded_data: dict[str, typing.Any] = {}
    changed = False
    for index, column_name in enumerate(X.columns):
        column = X[column_name]
        column_dtype = column.dtype
        if pandas.api.types.is_bool_dtype(column_dtype):
            if (
                isinstance(column_dtype, pandas.BooleanDtype)
                and column.isna().any()
            ):
                raise ValueError(
                    f"column {column_name!r} contains missing values;"
                    f" sigma's Tree estimators do not support NaN in"
                    f" boolean columns"
                )
            encoded_data[column_name] = numpy.asarray(
                column.astype(bool), dtype=numpy.float64
            )
            boolean_indices.add(index)
            changed = True
            continue
        if isinstance(column_dtype, pandas.CategoricalDtype):
            categorical = column
        elif isinstance(
            column_dtype, pandas.StringDtype
        ) or pandas.api.types.is_object_dtype(column_dtype):
            categorical = column.astype("category")
        else:
            encoded_data[column_name] = column
            continue
        categorical_cat = categorical.cat
        codes = numpy.asarray(categorical_cat.codes, dtype=numpy.int64)
        if numpy.any(codes < 0):
            raise ValueError(
                f"column {column_name!r} contains missing values; sigma's"
                f" Tree estimators do not support NaN in categorical,"
                f" string, or object-dtype columns"
            )
        encoded_data[column_name] = codes.astype(numpy.float64)
        labels[index] = {
            float(code): str(level)
            for code, level in enumerate(categorical_cat.categories)
        }
        changed = True
    if not changed:
        return X, None, None
    new_X = pandas.DataFrame(encoded_data, index=X.index)
    labels_out = labels if labels else None
    booleans_out = frozenset(boolean_indices) if boolean_indices else None
    return new_X, labels_out, booleans_out


def _apply_categorical_encoding(
    X: typing.Any,
    category_labels_in: None | dict[int, dict[float, str]],
) -> typing.Any:
    """Re-encode at predict time using the fit-time category label map."""
    if category_labels_in is None:
        return X
    columns_attr = getattr(X, "columns", None)
    if columns_attr is None:
        return X
    import pandas

    if not isinstance(X, pandas.DataFrame):
        return X
    encoded_data: dict[str, typing.Any] = {}
    for index, column_name in enumerate(X.columns):
        column = X[column_name]
        labels = category_labels_in.get(index)
        if labels is None:
            encoded_data[column_name] = column
            continue
        ordered_levels = [labels[float(code)] for code in range(len(labels))]
        recoded = pandas.Categorical(column, categories=ordered_levels)
        codes = numpy.asarray(recoded.codes, dtype=numpy.int64)
        if numpy.any(codes < 0):
            unknown_mask = codes < 0
            unknown_values = sorted(
                str(value) for value in column[unknown_mask].unique()
            )
            raise ValueError(
                f"column {column_name!r} contains values not seen at fit"
                f" time: {unknown_values}"
            )
        encoded_data[column_name] = codes.astype(numpy.float64)
    new_X = pandas.DataFrame(encoded_data, index=X.index)
    return new_X
