"""RankingTree estimator and its per-item mean-rank confidence intervals."""

from __future__ import annotations

import collections.abc
import typing

import numpy
import numpy.linalg
import numpy.typing
import sklearn.base
import sklearn.utils.extmath
import sklearn.utils.validation

from . import _node
from . import _ranking
from . import _tree
from . import _tree_regression
from . import _types

if typing.TYPE_CHECKING:
    import pandas


class RankingTree(_tree.Tree[_node.RankingNode]):
    """Conditional inference tree for full and partial rankings of items.

    Uses permutation-based conditional inference for unbiased variable
    selection and recursive binary partitioning, as described in Hothorn,
    Hornik, and Zeileis (2006), "Unbiased Recursive Partitioning: A
    Conditional Inference Framework," *Journal of Computational and
    Graphical Statistics*, 15(3), 651-674.

    The response y is supplied as a ranks-in-cell matrix of shape
    (n_obs, n_items) where each row carries the per-item rank position
    of one observation. Unranked items are NaN. Tied items share the
    same rank value; the actual rank values do not need to follow a
    specific convention (min, dense, average) because only the relative
    ordering and tie-equality are consulted. When y is a pandas
    DataFrame, its column names are used as item_names_; otherwise the
    constructor item_names argument (or integer indices 0..n_items - 1)
    apply.

    The statistical-test response is not item-aligned. The full-catalogue
    Y is imputed with the per-row tail mean for NaN cells, log-transformed
    via log(1 + rank), column-centered, and projected onto the top
    n_top_items right singular vectors of the resulting matrix. The
    R-dimensional projection serves as the influence function for the
    stat tests, following Leydesdorff (2006), "Classification and
    Powerlaws: The Logarithmic Transformation," *Journal of the
    American Society for Information Science and Technology*, 57(11),
    1470-1486, who prescribes log-transformation of power-law-distributed
    rank data before multivariate factor analysis / PCA. Per-node
    mean-rank statistics and confidence intervals are computed over the
    full catalogue. The n_top_items items with the largest aggregate
    loading magnitude across the R principal components are flagged for
    display so that tree text and graphviz renderings remain readable.

    Args:
        n_top_items: Number of principal components computed from the
            log-rank matrix, and number of items shown per leaf in
            tree text / graphviz renderings (the items with the largest
            aggregate loading magnitude across those components). Must
            be at least 1. When n_top_items is at least the catalogue
            size, the projection covers the full response space.
            Default 10.
        item_names: Optional display labels, one per item. When None and
            y is a numpy array, falls back to integer indices 0..n_items
            - 1. When y is a pandas DataFrame, its column names are used
            in lieu of this argument.
        correlation: Correlation type: "normal" or "rank" (default).
        test_stat: Test statistic form: "maximum" or "quadratic".
        test_type: Multiplicity adjustment method: "bonferroni",
            "monte_carlo", or "sidak".
        alpha: Significance level for the stopping rule.
        min_splits: Minimum sum of weights required to attempt a split.
        min_buckets: Minimum sum of weights in each child node.
        max_depth: Maximum tree depth. None means no limit.
        categorical_features: Categorical features. Entries may be
            column-name strings or integer column indices. String entries
            are resolved against the DataFrame columns at fit time (i.e.,
            they require X to be a pandas DataFrame). None means all
            numeric.
        ci_method: Per-item confidence interval method on the leaf mean
            rank of each item. Reuses the RegressionTree CI dispatcher
            applied per item. Only the four distribution-free options
            apply because the distribution-specific methods of
            RegressionTree rely on parametric assumptions incompatible
            with rank data. "bayesian_bootstrap" (default): Bayesian
            bootstrap interval. "bca": bias-corrected and accelerated
            bootstrap interval. "normal": normal-approximation interval
            on the weighted mean rank. "student_t": Student-t interval
            on the weighted mean rank.
        ci_coverage: Coverage level for per-item mean-rank confidence
            intervals. Defaults to 0.95. Set to None to disable CI
            computation.
        transmuter: Optional callable applied to node data before
            computing predictions and confidence intervals, with post-hoc
            split validation. See Tree for full signature and behavior.
        resamples: Number of permutations for min-P resampling when
            test_type="monte_carlo". Must be a positive integer. Ignored
            for other test_type values.
        decorator: Optional callable producing a per-node decoration
            stored on the node and rendered by to_text and to_image. See
            Tree for full signature and behavior.
        random_state: Seed for stochastic operations. Pass an integer for
            reproducibility; None uses an unpredictable seed. Controls
            min-P permutation resampling under test_type="monte_carlo"
            and the bootstrap-family CI methods ("bayesian_bootstrap",
            "bca") applied per item per leaf.

    Attributes:
        content_: Root node of the fitted tree structure.
        leaves_: List of leaf nodes, ordered by ascending mean rank of
            the top-ranked item.
        nodes_: List of all nodes in pre-order DFS, ordered by node_id.
            Indices match the output of predict_index.
        n_items_: Total number of items K in the fit-time catalogue.
        item_names_: Per-item labels, shape (n_items,). Set from the
            y DataFrame columns, else the constructor item_names, else
            falling back to integer indices 0..n_items - 1.
        n_features_in_: Number of features seen during fit.
        feature_types_: Per-feature CovariateType, shape (n_features,).
    """

    n_items_: int
    item_names_: numpy.typing.NDArray
    _y_full_: numpy.typing.NDArray[numpy.floating]
    _top_indices_: numpy.typing.NDArray[numpy.integer]
    _pc_loadings_: numpy.typing.NDArray[numpy.floating]

    def __init__(
        self,
        n_top_items: int = 10,
        item_names: None | collections.abc.Sequence[str] = None,
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
        ci_method: typing.Literal[
            "bayesian_bootstrap",
            "bca",
            "normal",
            "student_t",
        ] = "bayesian_bootstrap",
        ci_coverage: None | float = 0.95,
        transmuter: None | typing.Callable = None,
        resamples: None | int = None,
        decorator: None | typing.Callable = None,
        random_state: None | int = None,
        reverse_order: bool = False,
    ) -> None:
        if not isinstance(n_top_items, int) or isinstance(n_top_items, bool):
            raise TypeError(
                f"n_top_items must be an integer,"
                f" got {type(n_top_items).__name__}"
            )
        if n_top_items < 1:
            raise ValueError(
                f"n_top_items must be at least 1, got {n_top_items}"
            )
        if transmuter is not None:
            raise ValueError(
                "RankingTree does not support a transmuter; the leaf-level"
                " full-catalogue statistics require Y rows to align with"
                " the fit-time Y, which a row-modifying transmuter would"
                " break"
            )
        _types._validate_literal_param(
            ci_method, _types.CiMethodRankingTree, "ci_method"
        )
        self.n_top_items = n_top_items
        self.item_names = item_names
        self.ci_method = ci_method
        super().__init__(
            correlation=correlation,
            test_stat=test_stat,
            test_type=test_type,
            alpha=alpha,
            min_splits=min_splits,
            min_buckets=min_buckets,
            max_depth=max_depth,
            categorical_features=categorical_features,
            ci_coverage=ci_coverage,
            transmuter=transmuter,
            resamples=resamples,
            decorator=decorator,
            random_state=random_state,
            reverse_order=reverse_order,
        )

    def __sklearn_tags__(self):
        """Mark y as required and 2D for sklearn meta-estimator routing."""
        tags = super().__sklearn_tags__()
        target_tags = tags.target_tags
        target_tags.required = True
        target_tags.multi_output = True
        return tags

    def predict(
        self,
        X: numpy.typing.NDArray[numpy.floating] | pandas.DataFrame,
        offset: None | numpy.typing.NDArray[numpy.floating] = None,
    ) -> numpy.typing.NDArray:
        """Predict the favorite item label for each sample.

        Args:
            X: Samples to predict, shape (n_samples, n_features).
            offset: Ignored; RankingTree does not support offsets.

        Returns:
            Per-sample label of the item with the lowest mean rank at
            the sample's leaf, shape (n_samples,). The dtype matches
            item_names_: integer indices when no names were provided
            at fit, else the supplied labels.
        """
        indices = self.predict_index(X)
        node_predictions = numpy.array(
            [int(node.prediction) for node in self.nodes_]
        )
        favorite_indices = node_predictions[indices]
        predictions = self.item_names_[favorite_indices]
        return predictions

    def predict_rank(
        self,
        X: numpy.typing.NDArray[numpy.floating] | pandas.DataFrame,
    ) -> numpy.typing.NDArray[numpy.floating]:
        """Predict the per-item mean rank vector for each sample.

        Args:
            X: Samples to predict, shape (n_samples, n_features).

        Returns:
            Per-sample mean-rank matrix, shape (n_samples, n_items).
            Items with no observations in the predicted leaf are
            reported as NaN.
        """
        indices = self.predict_index(X)
        node_mean_ranks = numpy.array(
            [[metric.value for metric in node.metrics] for node in self.nodes_]
        )
        predictions = node_mean_ranks[indices]
        return predictions

    def _capture_y_column_names(self, y) -> None | list[str]:
        """Return DataFrame column names from y if present, else None."""
        try:
            import pandas
        except ImportError:
            return None
        if isinstance(y, pandas.DataFrame):
            names = [str(column) for column in y.columns]
            return names
        return None

    def _impute_tail_mean(
        self,
        y: numpy.typing.NDArray[numpy.floating],
    ) -> numpy.typing.NDArray[numpy.floating]:
        """Fill NaN cells with the per-row PL tail mean (d_i+1+n_items)/2."""
        valid_mask = ~numpy.isnan(y)
        d = valid_mask.sum(axis=1).astype(float)
        n_items = y.shape[1]
        tail_mean = (d + 1.0 + n_items) / 2.0
        y_out = numpy.where(numpy.isnan(y), tail_mean.reshape(-1, 1), y)
        return y_out

    def _truncated_svd_v(
        self,
        matrix: numpy.typing.NDArray[numpy.floating],
        n_components: int,
    ) -> numpy.typing.NDArray[numpy.floating]:
        """Top right singular vectors of matrix, as columns of V."""
        n_rows, n_cols = matrix.shape
        r = min(n_components, min(n_rows, n_cols))
        if r < 1:
            empty = numpy.zeros((n_cols, 0), dtype=float)
            return empty
        if n_cols > 1000:
            _, _, Vt = sklearn.utils.extmath.randomized_svd(
                matrix, n_components=r, random_state=0
            )
        else:
            _, _, Vt_full = numpy.linalg.svd(matrix, full_matrices=False)
            Vt = Vt_full[:r]
        V = Vt.T
        return V

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
        """Validate inputs, run the log-rank PCA, and emit Z as the response."""
        if y is None:
            raise ValueError(
                "RankingTree requires y to be passed, but the target y is None"
            )
        column_names = self._capture_y_column_names(y)
        X_validated = sklearn.utils.validation.validate_data(
            self, X, y="no_validation", dtype=numpy.float64
        )
        X_array = typing.cast(
            numpy.typing.NDArray[numpy.floating],
            numpy.asarray(X_validated, dtype=numpy.float64),
        )
        y_array = numpy.asarray(y)
        if y_array.dtype.kind == "c":
            raise ValueError(
                "Complex data not supported: y must be real-valued for"
                " RankingTree"
            )
        y_full = numpy.asarray(y_array, dtype=numpy.float64)
        if y_full.ndim != 2:
            raise ValueError(
                f"y must be 2D for ranking; got shape {y_full.shape}"
            )
        if y_full.shape[0] != X_array.shape[0]:
            raise ValueError(
                f"X has {X_array.shape[0]} rows but y has {y_full.shape[0]}"
            )
        n_items = y_full.shape[1]
        if n_items < 2:
            raise ValueError(
                f"y must have at least 2 columns; got shape {y_full.shape}"
            )
        valid_per_row = (~numpy.isnan(y_full)).sum(axis=1)
        if numpy.any(valid_per_row < 2):
            raise ValueError(
                "each row must rank at least 2 items (have at least 2"
                " non-NaN cells)"
            )
        self._y_full_ = y_full
        self.n_items_ = n_items
        if column_names is not None:
            self.item_names_ = numpy.asarray(column_names)
        elif self.item_names is not None:
            names_list = list(self.item_names)
            if len(names_list) != n_items:
                raise ValueError(
                    f"item_names has length {len(names_list)}"
                    f" but y has {n_items} columns"
                )
            self.item_names_ = numpy.asarray(names_list)
        else:
            self.item_names_ = numpy.arange(n_items)
        y_imputed = self._impute_tail_mean(y_full)
        y_log = numpy.log1p(y_imputed)
        y_log_centered = y_log - y_log.mean(axis=0)
        n_components = min(self.n_top_items, n_items)
        V = self._truncated_svd_v(y_log_centered, n_components)
        self._pc_loadings_ = V
        Z = y_log_centered @ V
        importance = numpy.sqrt((V**2).sum(axis=1))
        n_display = min(self.n_top_items, n_items)
        top = numpy.argsort(-importance, kind="stable")[:n_display]
        self._top_indices_ = numpy.sort(top).astype(numpy.int64)
        return X_array, Z

    def _validate_offset(
        self,
        offset: None | numpy.typing.NDArray[numpy.floating],
        n_samples: int,
    ) -> None | numpy.typing.NDArray[numpy.floating]:
        """Reject offsets - not supported in v1 of RankingTree."""
        if offset is None:
            return None
        raise ValueError("RankingTree does not support fit-time offsets")

    def _validate_transmuted_y_shape(
        self,
        y_out: numpy.typing.NDArray[numpy.floating],
    ) -> int:
        """Validate transmuted y shape: 2D with at least 2 columns."""
        y_out_shape = y_out.shape
        if y_out.ndim != 2 or y_out_shape[1] < 2:
            raise ValueError(
                f"transmuter y must be 2D with at least 2 columns"
                f" for ranking, got shape {y_out_shape}"
            )
        return y_out_shape[0]

    def _make_node(
        self,
        depth,
        n_samples,
        extension,
        prediction,
        ci_low,
        ci_high,
        ci_low_per_class,
        ci_high_per_class,
        class_distribution,
        survival_function,
        survival_log_variance,
        survival_metrics,
        ranking_metrics,
        mean_offset_proba,
        response_samples,
    ):
        """Construct a RankingNode with the per-item metric payload."""
        node = _node.RankingNode(
            depth=depth,
            n_samples=n_samples,
            share=0.0,
            decoration=None,
            extension=extension,
            metrics=typing.cast(list[_node.RankingMetric], ranking_metrics),
        )
        return node

    def _compute_influence(
        self,
        y: numpy.typing.NDArray[numpy.floating],
        offset: None | numpy.typing.NDArray[numpy.floating],
    ) -> numpy.typing.NDArray[numpy.floating]:
        """Return y unchanged; the PCA-projected response is the influence."""
        return y

    def _compute_prediction(
        self,
        y: numpy.typing.NDArray[numpy.floating],
        weights: numpy.typing.NDArray[numpy.floating],
        offset: None | numpy.typing.NDArray[numpy.floating],
    ) -> float:
        """Return the catalogue index of the favorite item as a float."""
        active = weights > 0
        if not numpy.any(active):
            return 0.0
        y_full_active = self._y_full_[active]
        w_active = weights[active]
        mean_rank = _ranking.compute_mean_rank_vector(y_full_active, w_active)
        favorite_index = _argmin_with_nan(mean_rank)
        prediction = float(favorite_index)
        return prediction

    def _is_constant_response(
        self,
        y: numpy.typing.NDArray[numpy.floating],
        weights: numpy.typing.NDArray[numpy.floating],
        offset: None | numpy.typing.NDArray[numpy.floating],
    ) -> bool:
        """Check whether the test response carries no signal in this node."""
        active = weights > 0
        if not numpy.any(active):
            return True
        y_active = y[active]
        n_active = y_active.shape[0]
        if n_active <= 1:
            return True
        first_row = y_active[0]
        nan_mask = numpy.isnan(first_row)
        first_finite = first_row[~nan_mask]
        for row in y_active[1:]:
            row_nan_mask = numpy.isnan(row)
            if not numpy.array_equal(nan_mask, row_nan_mask):
                return False
            row_finite = row[~row_nan_mask]
            if not numpy.allclose(row_finite, first_finite, equal_nan=False):
                return False
        return True

    def _compute_ci(
        self,
        y: numpy.typing.NDArray[numpy.floating],
        weights: numpy.typing.NDArray[numpy.floating],
        offset: None | numpy.typing.NDArray[numpy.floating],
    ) -> tuple[None | float, None | float]:
        """Return (None, None) - the favorite-item index has no CI."""
        return None, None

    def _compute_per_class_ci(
        self,
        y: numpy.typing.NDArray[numpy.floating],
        weights: numpy.typing.NDArray[numpy.floating],
    ) -> tuple[
        None | numpy.typing.NDArray[numpy.floating],
        None | numpy.typing.NDArray[numpy.floating],
    ]:
        """Return (None, None) - ranking CIs flow through ranking_metrics."""
        return None, None

    def _compute_class_distribution(
        self,
        y: numpy.typing.NDArray[numpy.floating],
        weights: numpy.typing.NDArray[numpy.floating],
    ) -> None | numpy.typing.NDArray[numpy.floating]:
        """Return None - ranking has no class distribution."""
        return None

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
        """Return None - ranking has no survival function."""
        return None

    def _compute_survival_log_variance(
        self,
        y: numpy.typing.NDArray[numpy.floating],
        weights: numpy.typing.NDArray[numpy.floating],
    ) -> None | numpy.typing.NDArray[numpy.floating]:
        """Return None - ranking has no survival log-variance."""
        return None

    def _compute_survival_metrics(
        self,
        y: numpy.typing.NDArray[numpy.floating],
        weights: numpy.typing.NDArray[numpy.floating],
    ) -> None | list[_node.SurvivalMetric]:
        """Return None - ranking has no survival metrics."""
        return None

    def _compute_ranking_metrics(
        self,
        y: numpy.typing.NDArray[numpy.floating],
        weights: numpy.typing.NDArray[numpy.floating],
    ) -> None | list[_node.RankingMetric]:
        """Compute the per-item RankingMetric list over the full catalogue."""
        active = weights > 0
        y_full_active = self._y_full_[active]
        w_active = weights[active]
        mean_rank = _ranking.compute_mean_rank_vector(y_full_active, w_active)
        per_item_ci_low, per_item_ci_high = self._compute_per_item_ci(
            y_full_active, w_active
        )
        displayed_set = set(int(idx) for idx in self._top_indices_.tolist())
        metrics: list[_node.RankingMetric] = []
        for k in range(self.n_items_):
            label = str(self.item_names_[k])
            value = float(mean_rank[k])
            low = None if per_item_ci_low is None else float(per_item_ci_low[k])
            high = (
                None if per_item_ci_high is None else float(per_item_ci_high[k])
            )
            metrics.append(
                _node.RankingMetric(
                    label=label,
                    value=value,
                    ci_low=low,
                    ci_high=high,
                    is_displayed=k in displayed_set,
                )
            )
        return metrics

    def _compute_per_item_ci(
        self,
        y_full_active: numpy.typing.NDArray[numpy.floating],
        weights_active: numpy.typing.NDArray[numpy.floating],
    ) -> tuple[
        None | numpy.typing.NDArray[numpy.floating],
        None | numpy.typing.NDArray[numpy.floating],
    ]:
        """Compute the per-item leaf CI on the full-catalogue active subset.

        Only preselected items receive a computed CI; non-preselected
        items report NaN bounds because the bootstrap cost scales with
        the catalogue size and the renderer hides their values anyway.
        """
        ci_coverage = self.ci_coverage
        if ci_coverage is None:
            return None, None
        n_active = y_full_active.shape[0]
        n_items = self.n_items_
        if n_active == 0:
            empty = numpy.full(n_items, numpy.nan, dtype=float)
            return empty, empty
        alpha = (1.0 - ci_coverage) / 2.0
        ci_method_enum = _types.CiMethodRankingTree(self.ci_method)
        ci_low_vec = numpy.full(n_items, numpy.nan, dtype=float)
        ci_high_vec = numpy.full(n_items, numpy.nan, dtype=float)
        displayed_set = set(int(idx) for idx in self._top_indices_.tolist())
        for k in range(n_items):
            if k not in displayed_set:
                continue
            column = y_full_active[:, k]
            observed_mask = ~numpy.isnan(column)
            y_k = column[observed_mask]
            w_k = weights_active[observed_mask]
            n_observed = y_k.size
            if n_observed == 0:
                continue
            if n_observed == 1:
                ci_low_vec[k] = float(y_k[0])
                ci_high_vec[k] = float(y_k[0])
                continue
            match ci_method_enum:
                case _types.CiMethodRankingTree.BAYESIAN_BOOTSTRAP:
                    low, high = (
                        _tree_regression.RegressionTree._compute_ci_bayesian_bootstrap(
                            self._rng_ci_, y_k, w_k, alpha
                        )
                    )
                case _types.CiMethodRankingTree.BCA:
                    low, high = _tree_regression.RegressionTree._compute_ci_bca(
                        self._rng_ci_, y_k, w_k, alpha
                    )
                case _types.CiMethodRankingTree.NORMAL:
                    low, high = (
                        _tree_regression.RegressionTree._compute_ci_normal(
                            y_k, w_k, alpha
                        )
                    )
                case _types.CiMethodRankingTree.STUDENT_T:
                    low, high = (
                        _tree_regression.RegressionTree._compute_ci_student_t(
                            y_k, w_k, alpha
                        )
                    )
            ci_low_vec[k] = low
            ci_high_vec[k] = high
        return ci_low_vec, ci_high_vec


def _argmin_with_nan(values: numpy.typing.NDArray[numpy.floating]) -> int:
    """Return argmin ignoring NaN; if all NaN, return 0."""
    nan_mask = numpy.isnan(values)
    if numpy.all(nan_mask):
        return 0
    safe = numpy.where(nan_mask, numpy.inf, values)
    index = int(numpy.argmin(safe))
    return index
