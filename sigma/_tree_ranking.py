"""RankingTree estimator and its per-item expected-rank confidence intervals."""

from __future__ import annotations

import collections.abc
import typing
import warnings

import numpy
import numpy.linalg
import numpy.typing
import scipy.linalg
import scipy.stats
import sklearn.base
import sklearn.utils.extmath
import sklearn.utils.validation

from . import _node, _ranking, _tree, _types

if typing.TYPE_CHECKING:
    import pandas
    import polars


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
    pca_components right singular vectors of the resulting matrix. The
    R-dimensional projection serves as the influence function for the
    stat tests, following Leydesdorff (2006), "Classification and
    Powerlaws: The Logarithmic Transformation," *Journal of the
    American Society for Information Science and Technology*, 57(11),
    1470-1486, who prescribes log-transformation of power-law-distributed
    rank data before multivariate factor analysis / PCA. Each node fits a
    Plackett-Luce maximum-likelihood worth vector on its weighted
    partial rankings by Hunter (2004), "MM Algorithms for Generalized
    Bradley-Terry Models," *Annals of Statistics*, 32(1), 384-406,
    regularised by ghost-item pseudo-rankings introduced in Turner et
    al. (2020), "Modelling Rankings in R: The PlackettLuce Package,"
    *Computational Statistics*, 35(3), 1027-1057. The reported per-item
    statistic is the implied Plackett-Luce expected rank, lying in
    [1, n_items] with lower values denoting preference. Tree text and
    response plots restrict the displayed items to the union of each
    leaf's top-N items by lowest expected rank; N is the
    top_displayed_items argument of to_text and to_image (default 3).

    Args:
        pca_components: Number of principal components computed from the
            log-rank matrix. Must be at least 1. When pca_components is at
            least the catalogue size, the projection covers the full
            response space. Default 10.
        item_names: Optional display labels, one per item. When None and
            y is a numpy array, falls back to integer indices 0..n_items
            - 1. When y is a pandas DataFrame, its column names are used
            in lieu of this argument.
        correlation: Correlation type: "normal" or "rank" (default).
        test_stat: Test statistic form: "maximum" or "quadratic".
            "maximum" attains higher power against low-dimensional
            ranking alternatives; the "quadratic" default keeps behavior
            consistent with the other tree types.
        test_type: Multiplicity adjustment method: "bonferroni",
            "monte_carlo", or "sidak". Pairing test_stat="maximum" with
            "monte_carlo" is much slower, as it evaluates a
            multivariate-normal CDF once per permutation per feature.
        alpha: Significance level for the stopping rule.
        min_splits: Minimum sum of weights required to attempt a split.
        min_buckets: Minimum sum of weights in each child node.
        max_depth: Maximum tree depth. None means no limit.
        categorical_features: Categorical features. Entries may be
            column-name strings or integer column indices. String entries
            are resolved against the DataFrame columns at fit time (i.e.,
            they require X to be a pandas DataFrame). None means all
            numeric.
        npseudo: Weight of the Turner ghost-item pseudo-comparisons added
            to each real item during the per-node Plackett-Luce fit.
            Must be strictly positive. Defaults to 0.5 (Turner et al.
            2020 default).
        pl_max_iter: Maximum number of Hunter MM iterations per
            Plackett-Luce fit. Must be a positive integer. Defaults to
            100.
        pl_tolerance: Convergence tolerance on the maximum absolute
            change in log-worth between successive MM iterations. Must
            be strictly positive. Defaults to 1e-6.
        ci_method: Per-item confidence interval method on the expected
            rank of each item.
            "bayesian_bootstrap" (default): Dirichlet-weighted refit
            interval. "bca": bias-corrected and accelerated row-resample
            refit interval. "wald": closed-form asymptotic interval
            built from the observed Plackett-Luce Fisher information
            with a delta-method propagation onto each per-item expected
            rank. "gaussian_multiplier": multiplier-CLT bootstrap that
            forms percentile intervals on Gaussian-weighted score
            perturbations linearised through the Fisher information.
        ci_replicates: Number of bootstrap replicates used by the
            "bayesian_bootstrap", "bca" and "gaussian_multiplier" CI
            methods. Must be a positive integer. Ignored when
            ci_method="wald". Defaults to 200.
        ci_coverage: Coverage level for per-item expected-rank
            confidence intervals. Defaults to 0.95. Set to None to
            disable CI computation. Each interval is marginal at this
            level for its own item; the probability that all M displayed
            items are covered simultaneously is lower, bounded above by
            roughly ci_coverage ** M.
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
            and every resampling CI method
            ("bayesian_bootstrap", "bca", "gaussian_multiplier")
            applied per item per node.

    Attributes:
        content_: Root node of the fitted tree structure.
        leaves_: List of leaf nodes, ordered by ascending expected rank
            of the top-ranked item.
        nodes_: List of all nodes in pre-order DFS, ordered by node_id.
            Indices match the output of predict_index.
        n_items_: Total number of items K in the fit-time catalogue.
        item_names_: Per-item labels, shape (n_items,). Set from the
            y DataFrame columns, else the constructor item_names, else
            falling back to integer indices 0..n_items - 1.
        n_features_in_: Number of features seen during fit.
        features_: One Feature per column of X, in column order.
    """

    n_items_: int
    item_names_: numpy.typing.NDArray
    _y_full: numpy.typing.NDArray[numpy.floating]

    def __init__(
        self,
        pca_components: int = 10,
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
        npseudo: float = 0.5,
        pl_max_iter: int = 100,
        pl_tolerance: float = 1.0e-6,
        ci_method: typing.Literal[
            "bayesian_bootstrap",
            "bca",
            "wald",
            "gaussian_multiplier",
        ] = "bayesian_bootstrap",
        ci_replicates: int = 200,
        ci_coverage: None | float = 0.95,
        transmuter: None | typing.Callable = None,
        resamples: None | int = None,
        decorator: None | typing.Callable = None,
        random_state: None | int = None,
        reverse_order: bool = False,
    ) -> None:
        if not isinstance(pca_components, int) or isinstance(
            pca_components, bool
        ):
            raise TypeError(
                f"pca_components must be an integer,"
                f" got {type(pca_components).__name__}"
            )
        if pca_components < 1:
            raise ValueError(
                f"pca_components must be at least 1, got {pca_components}"
            )
        if isinstance(npseudo, bool) or not isinstance(npseudo, (int, float)):
            raise TypeError(
                f"npseudo must be a real number, got {type(npseudo).__name__}"
            )
        if not numpy.isfinite(npseudo) or npseudo <= 0.0:
            raise ValueError(
                f"npseudo must be a positive finite float, got {npseudo}"
            )
        if not isinstance(pl_max_iter, int) or isinstance(pl_max_iter, bool):
            raise TypeError(
                f"pl_max_iter must be an integer,"
                f" got {type(pl_max_iter).__name__}"
            )
        if pl_max_iter < 1:
            raise ValueError(
                f"pl_max_iter must be at least 1, got {pl_max_iter}"
            )
        if isinstance(pl_tolerance, bool) or not isinstance(
            pl_tolerance, (int, float)
        ):
            raise TypeError(
                f"pl_tolerance must be a real number,"
                f" got {type(pl_tolerance).__name__}"
            )
        if not numpy.isfinite(pl_tolerance) or pl_tolerance <= 0.0:
            raise ValueError(
                f"pl_tolerance must be a positive finite float,"
                f" got {pl_tolerance}"
            )
        if transmuter is not None:
            raise ValueError(
                "RankingTree does not support a transmuter; the per-node"
                " full-catalogue statistics require Y rows to align with"
                " the fit-time Y, which a row-modifying transmuter would"
                " break"
            )
        _types._validate_literal_param(
            ci_method, _types.CiMethodRankingTree, "ci_method"
        )
        if not isinstance(ci_replicates, int) or isinstance(
            ci_replicates, bool
        ):
            raise TypeError(
                f"ci_replicates must be an integer,"
                f" got {type(ci_replicates).__name__}"
            )
        if ci_replicates < 1:
            raise ValueError(
                f"ci_replicates must be at least 1, got {ci_replicates}"
            )
        self.pca_components = pca_components
        self.item_names = item_names
        self.npseudo = float(npseudo)
        self.pl_max_iter = int(pl_max_iter)
        self.pl_tolerance = float(pl_tolerance)
        self.ci_method = ci_method
        self.ci_replicates = int(ci_replicates)
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
        """Declare that y is required and accepted only as multi-output."""
        tags = super().__sklearn_tags__()
        target_tags = tags.target_tags
        target_tags.required = True
        target_tags.multi_output = True
        target_tags.single_output = False
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
    ) -> _tree.Tree:
        """Fit the conditional inference ranking tree.

        Args:
            X: Training covariate matrix, shape (n_samples, n_features).
            y: Ranks-in-cell matrix, shape (n_samples, n_items), carrying the
                rank position each observation gave each item. Unranked items
                are NaN and every row must rank at least two items. A pandas
                DataFrame supplies item_names_ through its column names.
            sample_weight: Per-sample weights, shape (n_samples,). If None,
                all samples are weighted equally.
            offset: Must be None; RankingTree does not support offsets.
            side_data: Optional auxiliary per-sample data, shape (n_samples,
                ...). Must have the same number of rows as X, and its active
                subset reaches the decorator as a 5th positional argument.

        Returns:
            The fitted estimator. The ranking matrix passed as y is not
            retained on it.

        Raises:
            ValueError: If y is None, is not 2D, has fewer than two columns,
                has a row ranking fewer than two items, or has a row count
                differing from X; if offset is not None; or under any of the
                conditions listed by Tree.fit.
        """
        try:
            fitted = super().fit(X, y, sample_weight, offset, side_data)
        finally:
            self.__dict__.pop("_y_full", None)
        return fitted

    def predict(
        self,
        X: numpy.typing.NDArray[numpy.floating]
        | pandas.DataFrame
        | polars.DataFrame,
        offset: None | numpy.typing.NDArray[numpy.floating] = None,
    ) -> numpy.typing.NDArray:
        """Predict the favorite item label for each sample.

        Args:
            X: Samples to predict, shape (n_samples, n_features).
            offset: Ignored; RankingTree does not support offsets.

        Returns:
            Per-sample label of the item with the lowest Plackett-Luce
            expected rank at the node each sample reaches, shape
            (n_samples,). The dtype matches item_names_: integer indices
            when no names were provided at fit, else the supplied labels.
            A sample reaching a node whose items all have an undefined
            expected rank receives the first item (index 0).

        Raises:
            ValueError: If a column's type kind differs from its fit-time
                kind, or if X is a plain array or list that carries
                boolean values or predicts into a model fit with boolean
                or categorical columns.
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
        X: numpy.typing.NDArray[numpy.floating]
        | pandas.DataFrame
        | polars.DataFrame,
    ) -> numpy.typing.NDArray[numpy.floating]:
        """Predict the per-item Plackett-Luce expected rank for each sample.

        Args:
            X: Samples to predict, shape (n_samples, n_features).

        Returns:
            Per-sample expected-rank matrix, shape (n_samples, n_items).
            Each finite entry lies in [1, n_items_]; nodes with no active
            rows report NaN for every item.

        Raises:
            ValueError: If a column's type kind differs from its fit-time
                kind, or if X is a plain array or list that carries
                boolean values or predicts into a model fit with boolean
                or categorical columns.
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
        """Validate inputs, run the log-rank PCA, and emit Z as the response."""
        if y is None:
            raise ValueError(
                "RankingTree requires y to be passed, but the target y is None"
            )
        column_names = self._capture_y_column_names(y)
        X_validated = sklearn.utils.validation.validate_data(
            self,
            X,
            y="no_validation",
            dtype=numpy.float64,
            ensure_all_finite="allow-nan",
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
        self._y_full = y_full
        self.n_items_ = n_items
        if column_names is None:
            if self.item_names is None:
                self.item_names_ = numpy.arange(n_items)
            else:
                names_list = list(self.item_names)
                if len(names_list) != n_items:
                    raise ValueError(
                        f"item_names has length {len(names_list)}"
                        f" but y has {n_items} columns"
                    )
                self.item_names_ = numpy.asarray(names_list)
        else:
            self.item_names_ = numpy.asarray(column_names)
        y_imputed = self._impute_tail_mean(y_full)
        y_log = numpy.log1p(y_imputed)
        y_log_centered = y_log - y_log.mean(axis=0)
        n_components = min(self.pca_components, n_items)
        if n_components >= n_items:
            Z = y_log_centered
        else:
            V = self._truncated_svd_v(y_log_centered, n_components)
            Z = y_log_centered @ V
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

    def _create_node(
        self,
        depth: int,
        n_samples: int,
        y_transmuted: numpy.typing.NDArray[numpy.floating],
        w_transmuted: numpy.typing.NDArray[numpy.floating],
        offset_transmuted: None | numpy.typing.NDArray[numpy.floating],
        is_leaf: bool,
    ) -> _node.RankingNode:
        """Build a RankingNode with its per-item expected-rank metrics."""
        metrics = self._compute_ranking_metrics(w_transmuted)
        node = _node.RankingNode(
            depth=depth,
            n_samples=n_samples,
            share=0.0,
            decoration=None,
            metrics=metrics,
        )
        return node

    def _compute_influence(
        self,
        y: numpy.typing.NDArray[numpy.floating],
        offset: None | numpy.typing.NDArray[numpy.floating],
    ) -> numpy.typing.NDArray[numpy.floating]:
        """Return y unchanged; the PCA-projected response is the influence."""
        return y

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

    def _compute_ranking_metrics(
        self,
        weights: numpy.typing.NDArray[numpy.floating],
    ) -> list[_node.RankingMetric]:
        """Compute the per-item RankingMetric list over the full catalogue."""
        active = weights > 0
        y_full_active = self._y_full[active]
        w_active = weights[active]
        alpha = _ranking.compute_pl_mle(
            y_full_active,
            w_active,
            npseudo=self.npseudo,
            tolerance=self.pl_tolerance,
            max_iter=self.pl_max_iter,
        )
        expected_rank = _ranking.pl_expected_rank(alpha)
        per_item_ci_low, per_item_ci_high = self._compute_per_item_ci(
            y_full_active, w_active
        )
        metrics: list[_node.RankingMetric] = []
        for k in range(self.n_items_):
            label = str(self.item_names_[k])
            value = float(expected_rank[k])
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
                )
            )
        return metrics

    def _compute_displayed_indices(self, top_displayed_items: int) -> list[int]:
        """Return the union of each leaf's top items by lowest expected rank."""
        union: set[int] = set()
        for leaf in self.leaves_:
            values = numpy.array(
                [metric.value for metric in leaf.metrics], dtype=float
            )
            valid = ~numpy.isnan(values)
            valid_indices = numpy.flatnonzero(valid)
            if valid_indices.size == 0:
                continue
            valid_values = values[valid_indices]
            take = min(top_displayed_items, valid_indices.size)
            order = numpy.argsort(valid_values, kind="stable")[:take]
            for idx in valid_indices[order]:
                union.add(int(idx))
        return sorted(union)

    def _compute_per_item_ci(
        self,
        y_full_active: numpy.typing.NDArray[numpy.floating],
        weights_active: numpy.typing.NDArray[numpy.floating],
    ) -> tuple[
        None | numpy.typing.NDArray[numpy.floating],
        None | numpy.typing.NDArray[numpy.floating],
    ]:
        """Bootstrap a per-item expected-rank CI by refitting the PL MLE."""
        ci_coverage = self.ci_coverage
        if ci_coverage is None:
            return None, None
        n_active = int(y_full_active.shape[0])
        n_items = self.n_items_
        if n_active == 0:
            empty = numpy.full(n_items, numpy.nan, dtype=float)
            return empty, empty
        tail_alpha = (1.0 - ci_coverage) / 2.0
        ci_method_enum = _types.CiMethodRankingTree(self.ci_method)
        replicate_ranks = numpy.empty(
            (self.ci_replicates, n_items), dtype=float
        )
        weights_total = float(weights_active.sum())
        cache = _ranking._extract_orderings_cache(y_full_active, weights_active)
        match ci_method_enum:
            case _types.CiMethodRankingTree.BAYESIAN_BOOTSTRAP:
                dirichlet_alphas = weights_active.astype(float, copy=False)
                for b in range(self.ci_replicates):
                    bootstrap_weights = (
                        self._rng_ci_.dirichlet(dirichlet_alphas)
                        * weights_total
                    )
                    replicate_alpha = _ranking._compute_pl_mle_from_cache(
                        cache,
                        bootstrap_weights[cache.row_indices_in_y],
                        n_items,
                        npseudo=self.npseudo,
                        tolerance=self.pl_tolerance,
                        max_iter=self.pl_max_iter,
                    )
                    replicate_ranks[b] = _ranking.pl_expected_rank(
                        replicate_alpha
                    )
                with (
                    numpy.errstate(invalid="ignore"),
                    warnings.catch_warnings(),
                ):
                    warnings.filterwarnings(
                        "ignore",
                        message="All-NaN slice encountered",
                        category=RuntimeWarning,
                    )
                    quantiles = numpy.nanquantile(
                        replicate_ranks,
                        [tail_alpha, 1.0 - tail_alpha],
                        axis=0,
                    )
                ci_low_vec = quantiles[0]
                ci_high_vec = quantiles[1]
                all_nan_columns = numpy.all(
                    numpy.isnan(replicate_ranks), axis=0
                )
                ci_low_vec = numpy.where(all_nan_columns, numpy.nan, ci_low_vec)
                ci_high_vec = numpy.where(
                    all_nan_columns, numpy.nan, ci_high_vec
                )
            case _types.CiMethodRankingTree.BCA:
                point_alpha = _ranking._compute_pl_mle_from_cache(
                    cache,
                    cache.ordering_weights,
                    n_items,
                    npseudo=self.npseudo,
                    tolerance=self.pl_tolerance,
                    max_iter=self.pl_max_iter,
                )
                point_rank = _ranking.pl_expected_rank(point_alpha)
                for b in range(self.ci_replicates):
                    indices = self._rng_ci_.integers(0, n_active, size=n_active)
                    subset = _ranking._subset_cache(cache, indices)
                    replicate_alpha = _ranking._compute_pl_mle_from_cache(
                        subset,
                        subset.ordering_weights,
                        n_items,
                        npseudo=self.npseudo,
                        tolerance=self.pl_tolerance,
                        max_iter=self.pl_max_iter,
                    )
                    replicate_ranks[b] = _ranking.pl_expected_rank(
                        replicate_alpha
                    )
                ci_low_vec, ci_high_vec = _bca_per_item_quantiles(
                    point_rank,
                    replicate_ranks,
                    cache,
                    tail_alpha,
                    self.npseudo,
                    self.pl_tolerance,
                    self.pl_max_iter,
                )
            case _types.CiMethodRankingTree.WALD:
                ci_low_vec, ci_high_vec = _wald_per_item_ci(
                    cache,
                    n_items,
                    ci_coverage,
                    self.npseudo,
                    self.pl_tolerance,
                    self.pl_max_iter,
                )
            case _types.CiMethodRankingTree.GAUSSIAN_MULTIPLIER:
                ci_low_vec, ci_high_vec = _gaussian_multiplier_per_item_ci(
                    cache,
                    n_items,
                    ci_coverage,
                    self._rng_ci_,
                    self.ci_replicates,
                    self.npseudo,
                    self.pl_tolerance,
                    self.pl_max_iter,
                )
        return ci_low_vec, ci_high_vec


def _bca_per_item_quantiles(
    point_rank: numpy.typing.NDArray[numpy.floating],
    replicate_ranks: numpy.typing.NDArray[numpy.floating],
    cache: _ranking._OrderingsCache,
    tail_alpha: float,
    npseudo: float,
    pl_tolerance: float,
    pl_max_iter: int,
) -> tuple[
    numpy.typing.NDArray[numpy.floating],
    numpy.typing.NDArray[numpy.floating],
]:
    """Apply BCa quantile adjustment per item to PL bootstrap replicates."""
    n_replicates, n_items = replicate_ranks.shape
    n_active = int(cache.row_sizes.size)
    jackknife_ranks = numpy.empty((n_active, n_items), dtype=float)
    full_indices = numpy.arange(n_active, dtype=numpy.intp)
    for i in range(n_active):
        keep_indices = numpy.concatenate(
            (full_indices[:i], full_indices[i + 1 :])
        )
        subset = _ranking._subset_cache(cache, keep_indices)
        jackknife_alpha = _ranking._compute_pl_mle_from_cache(
            subset,
            subset.ordering_weights,
            n_items,
            npseudo=npseudo,
            tolerance=pl_tolerance,
            max_iter=pl_max_iter,
        )
        jackknife_ranks[i] = _ranking.pl_expected_rank(jackknife_alpha)
    proportion_below = (replicate_ranks < point_rank.reshape(1, -1)).mean(
        axis=0
    )
    proportion_clamped = numpy.clip(
        proportion_below,
        0.5 / n_replicates,
        1.0 - 0.5 / n_replicates,
    )
    z0 = scipy.stats.norm.ppf(proportion_clamped)
    jackknife_mean = jackknife_ranks.mean(axis=0)
    deviations = jackknife_mean - jackknife_ranks
    sum_cubed = (deviations**3).sum(axis=0)
    sum_squared = (deviations**2).sum(axis=0)
    acceleration = numpy.where(
        sum_squared > 0.0,
        sum_cubed / (6.0 * numpy.power(sum_squared, 1.5)),
        0.0,
    )
    z_lo = float(scipy.stats.norm.ppf(tail_alpha))
    z_hi = float(scipy.stats.norm.ppf(1.0 - tail_alpha))
    denominator_lo = 1.0 - acceleration * (z0 + z_lo)
    denominator_hi = 1.0 - acceleration * (z0 + z_hi)
    adjusted_lo = numpy.where(
        denominator_lo > 0.0,
        scipy.stats.norm.cdf(z0 + (z0 + z_lo) / denominator_lo),
        tail_alpha,
    )
    adjusted_hi = numpy.where(
        denominator_hi > 0.0,
        scipy.stats.norm.cdf(z0 + (z0 + z_hi) / denominator_hi),
        1.0 - tail_alpha,
    )
    ci_low_vec = numpy.empty(n_items, dtype=float)
    ci_high_vec = numpy.empty(n_items, dtype=float)
    for k in range(n_items):
        column = replicate_ranks[:, k]
        if numpy.all(numpy.isnan(column)):
            ci_low_vec[k] = numpy.nan
            ci_high_vec[k] = numpy.nan
            continue
        ci_low_vec[k] = float(numpy.nanquantile(column, adjusted_lo[k]))
        ci_high_vec[k] = float(numpy.nanquantile(column, adjusted_hi[k]))
    return ci_low_vec, ci_high_vec


def _pl_fisher_cho_factor(
    cache: _ranking._OrderingsCache,
    alpha: numpy.typing.NDArray[numpy.floating],
    n_items: int,
) -> tuple[numpy.typing.NDArray[numpy.floating], bool]:
    """Ridge-regularized Cholesky factor of the PL Fisher information."""
    h = _ranking._compute_pl_fisher_info(
        cache, alpha, cache.ordering_weights, n_items
    )
    trace_value = float(numpy.trace(h))
    ridge = 1.0e-9 * max(trace_value, 1.0) / n_items
    h_regularised = h + ridge * numpy.eye(n_items, dtype=float)
    cho = scipy.linalg.cho_factor(h_regularised, lower=True)
    return cho


def _wald_per_item_ci(
    cache: _ranking._OrderingsCache,
    n_items: int,
    ci_coverage: float,
    npseudo: float,
    pl_tolerance: float,
    pl_max_iter: int,
) -> tuple[
    numpy.typing.NDArray[numpy.floating],
    numpy.typing.NDArray[numpy.floating],
]:
    """Closed-form Wald CI on per-item expected ranks via PL Fisher info + delta."""
    alpha = _ranking._compute_pl_mle_from_cache(
        cache,
        cache.ordering_weights,
        n_items,
        npseudo=npseudo,
        tolerance=pl_tolerance,
        max_iter=pl_max_iter,
    )
    point_rank = _ranking.pl_expected_rank(alpha)
    with numpy.errstate(divide="ignore", over="ignore", invalid="ignore"):
        c_factor, lower = _pl_fisher_cho_factor(cache, alpha, n_items)
        g = _ranking._compute_pl_expected_rank_jacobian(alpha)
        z_matrix = scipy.linalg.cho_solve((c_factor, lower), g.T)
        var_per_item = numpy.einsum("ki,ik->k", g, z_matrix)
    se = numpy.sqrt(numpy.maximum(var_per_item, 0.0))
    z = float(scipy.stats.norm.ppf((1.0 + ci_coverage) / 2.0))
    halfwidth = z * se
    ci_low_vec = point_rank - halfwidth
    ci_high_vec = point_rank + halfwidth
    return ci_low_vec, ci_high_vec


def _gaussian_multiplier_per_item_ci(
    cache: _ranking._OrderingsCache,
    n_items: int,
    ci_coverage: float,
    rng: numpy.random.Generator,
    n_replicates: int,
    npseudo: float,
    pl_tolerance: float,
    pl_max_iter: int,
) -> tuple[
    numpy.typing.NDArray[numpy.floating],
    numpy.typing.NDArray[numpy.floating],
]:
    """Multiplier-CLT bootstrap CI: B Gaussian-weighted score-linearised draws."""
    alpha = _ranking._compute_pl_mle_from_cache(
        cache,
        cache.ordering_weights,
        n_items,
        npseudo=npseudo,
        tolerance=pl_tolerance,
        max_iter=pl_max_iter,
    )
    point_rank = _ranking.pl_expected_rank(alpha)
    with numpy.errstate(divide="ignore", over="ignore", invalid="ignore"):
        c_factor, lower = _pl_fisher_cho_factor(cache, alpha, n_items)
        score = _ranking._compute_pl_score_per_row(cache, alpha, n_items)
        weighted_score = cache.ordering_weights[:, None] * score
        n_orderings = int(cache.row_sizes.size)
        omega = rng.normal(size=(n_replicates, n_orderings))
        score_sums = omega @ weighted_score
        delta_theta = scipy.linalg.cho_solve((c_factor, lower), score_sums.T).T
        g = _ranking._compute_pl_expected_rank_jacobian(alpha)
        delta_er = delta_theta @ g.T
    tail = (1.0 - ci_coverage) / 2.0
    ci_low_vec = point_rank + numpy.quantile(delta_er, tail, axis=0)
    ci_high_vec = point_rank + numpy.quantile(delta_er, 1.0 - tail, axis=0)
    return ci_low_vec, ci_high_vec
