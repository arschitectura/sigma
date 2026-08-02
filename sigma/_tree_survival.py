"""SurvivalTree estimator and its log-rank / Kaplan-Meier helpers."""

from __future__ import annotations

import collections.abc
import typing

import numpy
import numpy.typing
import sklearn.utils
import sklearn.utils.validation

from . import _metric, _node, _survival, _tree, _types

if typing.TYPE_CHECKING:
    import pandas
    import polars


_SurvivalMetricSpec: typing.TypeAlias = str | tuple[str, float | int, str]

_METRIC_KINDS = ("median", "risk_score", "survival", "rmst")
_PARAMETRIZED_METRIC_KINDS = ("survival", "rmst")


class _NodeCurve:
    """Kaplan-Meier curve and weighted counts shared by one node's metrics."""

    __slots__ = ("d_w", "r_w", "surv", "times", "var_log_s")

    def __init__(
        self,
        times: numpy.typing.NDArray[numpy.floating],
        surv: numpy.typing.NDArray[numpy.floating],
        var_log_s: numpy.typing.NDArray[numpy.floating],
        d_w: numpy.typing.NDArray[numpy.floating],
        r_w: numpy.typing.NDArray[numpy.floating],
    ) -> None:
        self.times = times
        self.surv = surv
        self.var_log_s = var_log_s
        self.d_w = d_w
        self.r_w = r_w


class SurvivalTree(_tree.Tree[_node.SurvivalNode]):
    """Conditional inference tree for right-censored survival outcomes.

    Uses permutation-based conditional inference for unbiased variable selection
    and recursive binary partitioning, as described in Hothorn, Hornik, and
    Zeileis (2006), "Unbiased Recursive Partitioning: A Conditional Inference
    Framework," *Journal of Computational and Graphical Statistics*, 15(3),
    651-674, Section 4 ("Censored regression"). The same approach grounds
    Hothorn, Buhlmann, Dudoit, Molinaro, and Van der Laan (2006), "Survival
    Ensembles," *Biostatistics*, 7(3), 355-373.

    The metrics parameter selects one or more per-node summaries to display
    alongside the tree (median survival, S(t) at a reference time, restricted
    mean survival time, or risk score).

    The response y can be supplied in any of three equivalent encodings:

    - 2D float array, shape (n, 2), columns (time, event); event is 1 for
      an observed event and 0 for right-censored. Example:
      numpy.column_stack([time, event]).
    - Structured array, shape (n,), with fields "time" (float) and "event"
      (bool), matching scikit-survival. Example:
      numpy.array([(5.0, True), (8.0, False)],
               dtype=[("time", float), ("event", bool)]).
    - 1D float array, shape (n,), age-encoded: a value >= 0 means the
      subject died at that age, a value < 0 means the subject is still
      alive at age -value. A value of exactly 0 is read as "died at age
      0". Example: numpy.array([5.0, -8.0]) is equivalent to time=[5, 8],
      event=[1, 0].

    All three are normalized internally to the 2D form. The remainder of
    this docstring describes y in the 2D form.

    sklearn integration: the estimator works as the final stage of a
    Pipeline; Pipeline.score(X, y) returns Harrell's concordance index,
    which is also the default scorer used by cross_val_score and
    GridSearchCV when no explicit scoring= is supplied. Use KFold or a
    survival-aware splitter; StratifiedKFold is not applicable.

    Args:
        correlation: Correlation type: "normal" (default) or "rank".
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
        ci_method: Confidence interval method for the median survival time.
            "brookmeyer_crowley" (default): Brookmeyer-Crowley interval.
        ci_coverage: Coverage level for node-prediction confidence intervals.
            Defaults to 0.95. Set to None to disable CI computation. Also
            controls the pointwise log-log Greenwood band drawn behind each
            Kaplan-Meier curve in the response plot.
        transmuter: Optional callable applied to node data before computing
            predictions and confidence intervals, with post-hoc split
            validation. See Tree for full signature and behavior.
        resamples: Number of permutations for min-P resampling when
            test_type="monte_carlo". Must be a positive integer. Ignored for
            other test_type values.
        decorator: Optional callable producing a per-node decoration stored on
            the node and rendered by to_text and to_image. See Tree
            for full signature and behavior.
        random_state: Seed for the random number generator used in permutation
            resampling. Pass an integer for reproducibility. None uses an
            unpredictable seed. Ignored unless test_type="monte_carlo".
        metrics: Sequence of per-node metric specifications. Each entry is
            either a literal string ("median", "risk_score") for
            parameter-free metrics, or a tuple (kind, value, unit) for
            parametrized metrics ("survival" and "rmst"). value is in y's
            native time units. unit is the trailing token appended after
            the value in the rendered label; pass only the unit name, not a
            value+unit phrase. Examples:
                ("survival", 5.0, "years")   renders as "Survival at 5 years"
                ("rmst",     2.0, "years")   renders as "RMST at 2 years"
                ("survival", 12.0, "months") renders as "Survival at 12 months"
            Metrics render in the order given. The first metric drives the
            leaf badge ordering and the prediction returned by predict(X).
            Default is ("median",) which preserves the legacy single-line
            rendering and median-driven predict().

    Attributes:
        content_: Root node of the fitted tree structure.
        leaves_: List of leaf nodes, ordered by ascending value of the first
            metric.
        nodes_: List of all nodes in pre-order DFS, ordered by node_id.
            Indices match the output of predict_index.
        n_features_in_: Number of features seen during fit.
        features_: One Feature per column of X, in column order.
        metrics_: One Metric per configured metric specification, in the
            order given, aligned with the value arrays of every node.
    """

    event_grid_: numpy.typing.NDArray[numpy.floating]

    def __init__(
        self,
        correlation: typing.Literal["normal", "rank"] = "normal",
        test_stat: typing.Literal["maximum", "quadratic"] = "quadratic",
        test_type: typing.Literal[
            "bonferroni", "monte_carlo", "sidak"
        ] = "sidak",
        alpha: float = 0.05,
        min_splits: int = 20,
        min_buckets: int = 7,
        max_depth: None | int = None,
        categorical_features: None | collections.abc.Sequence[str | int] = None,
        ci_method: typing.Literal["brookmeyer_crowley"] = "brookmeyer_crowley",
        ci_coverage: None | float = 0.95,
        transmuter: None | typing.Callable = None,
        resamples: None | int = None,
        decorator: None | typing.Callable = None,
        random_state: None | int = None,
        metrics: collections.abc.Sequence[_SurvivalMetricSpec] = ("median",),
        reverse_order: bool = False,
    ) -> None:
        _types._validate_literal_param(
            ci_method, _types.CiMethodSurvival, "ci_method"
        )
        self.ci_method = ci_method
        self.metrics = metrics
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
        """Declare that y is required."""
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        return tags

    def predict(
        self,
        X: numpy.typing.NDArray[numpy.floating]
        | pandas.DataFrame
        | polars.DataFrame,
        offset: None | numpy.typing.NDArray[numpy.floating] = None,
    ) -> numpy.typing.NDArray[numpy.floating]:
        """Predict the value of the first configured metric.

        Args:
            X: Samples to predict, shape (n_samples, n_features).
            offset: Optional per-sample baseline survival probabilities,
                shape (n_samples, len(self.event_grid_)), non-increasing
                along the time axis with values in (0, 1].

        Returns:
            Predicted scalar metric, shape (n_samples,). An undefined median
            survival (the leaf Kaplan-Meier curve never reaching 0.5) is
            returned as NaN.

        Raises:
            ValueError: If a column's type kind differs from its fit-time
                kind, or if X is a plain array or list that carries
                boolean values or predicts into a model fit with boolean
                or categorical columns.
        """
        indices = self.predict_index(X)
        if offset is None and not self._fit_with_offset:
            predictions = self._gather_node_predictions(indices)
            return predictions
        event_grid = self.event_grid_
        n_indices = len(indices)
        n_event_grid = len(event_grid)
        if offset is None:
            offset_grid: numpy.typing.NDArray[numpy.floating] = numpy.ones(
                (n_indices, n_event_grid)
            )
        else:
            offset_grid = self._validate_predict_offset_grid(
                offset, n_indices, n_event_grid
            )
        bare = self.predict_survival(X, event_grid, offset=None)
        combined = offset_grid * bare
        first = self.metrics_[0]
        predictions = self._compute_first_metric_from_curves(
            event_grid, combined, first
        )
        return predictions

    def predict_survival(
        self,
        X: numpy.typing.NDArray[numpy.floating]
        | pandas.DataFrame
        | polars.DataFrame,
        times: numpy.typing.NDArray[numpy.floating],
        offset: None | numpy.typing.NDArray[numpy.floating] = None,
    ) -> numpy.typing.NDArray[numpy.floating]:
        """Predict survival probabilities at the requested times.

        Args:
            X: Samples to predict, shape (n_samples, n_features).
            times: Times at which to evaluate the survival function, shape
                (n_times,). Must be non-decreasing.
            offset: Optional per-sample baseline survival probabilities at
                the requested times, shape (n_samples, n_times),
                non-increasing along the time axis with values in (0, 1].

        Returns:
            Survival probabilities, shape (n_samples, n_times).

        Raises:
            ValueError: If a column's type kind differs from its fit-time
                kind, or if X is a plain array or list that carries
                boolean values or predicts into a model fit with boolean
                or categorical columns.
        """
        node_indices = self.predict_index(X)
        times_array = numpy.asarray(times, dtype=float)
        n_samples = len(node_indices)
        n_times = len(times_array)
        result = numpy.empty((n_samples, n_times), dtype=float)
        node_curves = [node.survival_function for node in self.nodes_]
        for i in range(n_samples):
            curve = node_curves[node_indices[i]]
            if curve is None:
                result[i, :] = 1.0
                continue
            node_times, node_surv = curve
            result[i, :] = _evaluate_step_curve(
                node_times, node_surv, times_array
            )
        if offset is None:
            return result
        offset_array = self._validate_predict_offset_grid(
            offset, n_samples, n_times
        )
        combined = offset_array * result
        return combined

    def score(
        self,
        X: numpy.typing.NDArray[numpy.floating]
        | pandas.DataFrame
        | polars.DataFrame,
        y: numpy.typing.NDArray[numpy.floating],
        sample_weight: None | numpy.typing.NDArray[numpy.floating] = None,
    ) -> float:
        """Return Harrell's concordance index on the predicted risk.

        Args:
            X: Samples to score, shape (n_samples, n_features).
            y: Observed outcomes; any of the three y encodings accepted
                by fit (see class docstring).
            sample_weight: Optional per-sample weights, shape
                (n_samples,).

        Returns:
            Harrell's concordance index of predicted risk against the
            observed (time, event) pairs. 1.0 indicates perfect ordering,
            0.5 indicates random, and 0.0 indicates fully inverted ordering.
            When no comparable pairs exist (e.g. all samples are censored),
            returns 0.5.
        """
        y_array = _coerce_survival_y(y)
        y_array_shape = y_array.shape
        if y_array.ndim != 2 or y_array_shape[1] != 2:
            raise ValueError(
                f"y must be a 2D array with two columns (time, event); got"
                f" shape {y_array_shape}"
            )
        weight_array = (
            None
            if sample_weight is None
            else numpy.asarray(sample_weight, dtype=float)
        )
        n_samples = len(y_array)
        X_subset: (
            numpy.typing.NDArray[numpy.floating]
            | pandas.DataFrame
            | polars.DataFrame
        ) = X
        if n_samples > 10_000:
            rng = numpy.random.default_rng(self.random_state)
            indices = rng.choice(n_samples, size=10_000, replace=False)
            X_subset = sklearn.utils._safe_indexing(X, indices)
            y_array = y_array[indices]
            if weight_array is not None:
                weight_array = weight_array[indices]
        survival = self.predict_survival(X_subset, self.event_grid_)
        clipped = numpy.clip(survival, numpy.finfo(float).tiny, 1.0)
        risk = -numpy.log(clipped).sum(axis=1)
        time_column = y_array[:, 0]
        event_column = y_array[:, 1]
        c_index = _concordance_index(
            time_column, event_column, risk, weight_array
        )
        return c_index

    def _validate_predict_offset_grid(
        self,
        offset: numpy.typing.NDArray[numpy.floating],
        n_samples: int,
        n_times: int,
    ) -> numpy.typing.NDArray[numpy.floating]:
        """Validate a survival predict-time offset on (n_samples, n_times)."""
        expected_shape = (n_samples, n_times)
        offset_array = self._validate_offset_shape_finite(
            offset, expected_shape
        )
        if numpy.any(offset_array <= 0.0) or numpy.any(offset_array > 1.0):
            raise ValueError("offset survival probabilities must lie in (0, 1]")
        if n_times >= 2:
            diffs = numpy.diff(offset_array, axis=1)
            if numpy.any(diffs > 0.0):
                raise ValueError(
                    "offset must be non-increasing along the time axis"
                )
        return offset_array

    def _compute_first_metric_from_curves(
        self,
        times: numpy.typing.NDArray[numpy.floating],
        survival_matrix: numpy.typing.NDArray[numpy.floating],
        metric: _metric.Metric,
    ) -> numpy.typing.NDArray[numpy.floating]:
        """Compute one row of the first metric per sample from given curves."""
        n_samples = survival_matrix.shape[0]
        result = numpy.empty(n_samples, dtype=float)
        for i in range(n_samples):
            surv_row = survival_matrix[i]
            match metric:
                case _metric.MedianSurvivalMetric():
                    result[i] = _survival.compute_median_survival(
                        times, surv_row
                    )
                case _metric.SurvivalAtMetric() as survival_at:
                    result[i] = _survival.compute_survival_at(
                        times, surv_row, survival_at.time
                    )
                case _metric.RmstMetric() as rmst:
                    result[i] = _survival.compute_rmst(
                        times, surv_row, rmst.horizon
                    )
                case _metric.RiskScoreMetric():
                    cum_haz_row = -numpy.log(
                        numpy.maximum(surv_row, _tree._OFFSET_EPS)
                    )
                    result[i] = float(cum_haz_row.sum())
        return result

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
        """Validate inputs for survival analysis."""
        y_coerced = y if y is None else _coerce_survival_y(y)
        X_validated, y_validated = sklearn.utils.validation.validate_data(
            self,
            X,
            y_coerced,
            dtype="float64",
            multi_output=True,
            ensure_all_finite="allow-nan",
        )
        X_array = typing.cast(numpy.typing.NDArray[numpy.floating], X_validated)
        y_array = typing.cast(numpy.typing.NDArray[numpy.floating], y_validated)
        y_array_shape = y_array.shape
        if y_array.ndim != 2 or y_array_shape[1] != 2:
            raise ValueError(
                f"y must be a 2D array with two columns (time, event); got shape"
                f" {y_array_shape}"
            )
        if not numpy.all(numpy.isfinite(y_array)):
            raise ValueError("y must contain only finite values")
        time_column = y_array[:, 0]
        event_column = y_array[:, 1]
        if numpy.any(time_column < 0):
            raise ValueError("y[:, 0] (time) must be non-negative")
        unique_events = numpy.unique(event_column)
        if not numpy.all(numpy.isin(unique_events, [0.0, 1.0])):
            raise ValueError("y[:, 1] (event) must contain only 0 and 1")
        events_mask = event_column == 1.0
        self.event_grid_ = numpy.unique(time_column[events_mask])
        return X_array, y_array

    def _validate_offset(
        self,
        offset: None | numpy.typing.NDArray[numpy.floating],
        n_samples: int,
    ) -> None | numpy.typing.NDArray[numpy.floating]:
        """Validate the survival fit-time offset (1D, length n_samples)."""
        if offset is None:
            return None
        offset_array = self._validate_offset_shape_finite(offset, (n_samples,))
        if numpy.any(offset_array <= 0.0) or numpy.any(offset_array > 1.0):
            raise ValueError("offset survival probabilities must lie in (0, 1]")
        return offset_array

    def _validate_transmuted_y_shape(
        self,
        y_out: numpy.typing.NDArray[numpy.floating],
    ) -> int:
        """Validate transmuted y shape for survival (2D, two columns)."""
        y_out_shape = y_out.shape
        if y_out.ndim != 2 or y_out_shape[1] != 2:
            raise ValueError(
                f"transmuter y must be 2D with two columns (time, event)"
                f" for survival, got shape {y_out_shape}"
            )
        return y_out_shape[0]

    def _build_metrics(self) -> tuple[_metric.Metric, ...]:
        """Parse the configured metric specifications into one descriptor each."""
        metrics = self.metrics
        if isinstance(metrics, (str, bytes)):
            raise TypeError(
                f"metrics must be a sequence of metric specs, not a string;"
                f" wrap a single entry in a tuple: ({metrics!r},)"
            )
        if not isinstance(metrics, collections.abc.Sequence):
            raise TypeError(
                f"metrics must be a sequence; got {type(metrics).__name__}"
            )
        if len(metrics) == 0:
            raise ValueError("metrics must contain at least one entry")
        has_ci = self.ci_coverage is not None
        descriptors: list[_metric.Metric] = []
        for spec in metrics:
            descriptor = _build_survival_metric(spec, has_ci)
            descriptors.append(descriptor)
        result = tuple(descriptors)
        return result

    def _create_node(
        self,
        depth: int,
        n_samples: int,
        y_transmuted: numpy.typing.NDArray[numpy.floating],
        w_transmuted: numpy.typing.NDArray[numpy.floating],
        offset_transmuted: None | numpy.typing.NDArray[numpy.floating],
        is_leaf: bool,
    ) -> _node.SurvivalNode:
        """Build a SurvivalNode with its Kaplan-Meier curve and metric values."""
        curve = self._compute_node_curve(y_transmuted, w_transmuted)
        survival_function = (curve.times, curve.surv)
        if is_leaf:
            survival_log_variance = curve.var_log_s
        else:
            survival_log_variance = numpy.empty(0, dtype=float)
        values, ci_low, ci_high = self._compute_metric_values(curve)
        node = _node.SurvivalNode(
            depth=depth,
            n_samples=n_samples,
            share=0.0,
            decoration=None,
            survival_function=survival_function,
            survival_log_variance=survival_log_variance,
            values=values,
            ci_low=ci_low,
            ci_high=ci_high,
        )
        return node

    def _compute_influence(
        self,
        y: numpy.typing.NDArray[numpy.floating],
        offset: None | numpy.typing.NDArray[numpy.floating],
    ) -> numpy.typing.NDArray[numpy.floating]:
        """Compute the per-sample influence function for survival."""
        time_column = y[:, 0]
        event_column = y[:, 1]
        if offset is None:
            scores = _survival.compute_logrank_scores(time_column, event_column)
            result = scores.reshape(-1, 1)
            return result
        clipped = numpy.clip(offset, _tree._OFFSET_EPS, 1.0)
        cum_hazard = -numpy.log(clipped)
        martingale = event_column - cum_hazard
        result = martingale.reshape(-1, 1)
        return result

    def _is_constant_response(
        self,
        y: numpy.typing.NDArray[numpy.floating],
        weights: numpy.typing.NDArray[numpy.floating],
        offset: None | numpy.typing.NDArray[numpy.floating],
    ) -> bool:
        """Check whether the response carries no signal in this node."""
        active = weights > 0
        if not numpy.any(active):
            return True
        if offset is not None:
            event_active = y[active, 1]
            offset_active = numpy.clip(offset[active], _tree._OFFSET_EPS, 1.0)
            martingale = event_active - (-numpy.log(offset_active))
            constant = bool(numpy.ptp(martingale) == 0.0)
            return constant
        time_active = y[active, 0]
        event_active = y[active, 1]
        no_events = bool(numpy.all(event_active == 0.0))
        if no_events:
            return True
        single_time = bool(numpy.ptp(time_active) == 0.0)
        return single_time

    def _compute_node_curve(
        self,
        y: numpy.typing.NDArray[numpy.floating],
        weights: numpy.typing.NDArray[numpy.floating],
    ) -> _NodeCurve:
        """Compute the weighted Kaplan-Meier curve for the active samples."""
        time_column = y[:, 0]
        event_column = y[:, 1]
        times, surv, var_log_s, d_w, r_w = (
            _survival.compute_kaplan_meier_with_variance(
                time_column, event_column, weights
            )
        )
        curve = _NodeCurve(
            times=times,
            surv=surv,
            var_log_s=var_log_s,
            d_w=d_w,
            r_w=r_w,
        )
        return curve

    def _compute_metric_values(
        self,
        curve: _NodeCurve,
    ) -> tuple[
        numpy.typing.NDArray[numpy.floating],
        numpy.typing.NDArray[numpy.floating],
        numpy.typing.NDArray[numpy.floating],
    ]:
        """Compute the value and confidence bounds of every configured metric."""
        count = len(self.metrics_)
        values = numpy.empty(count, dtype=float)
        ci_low = numpy.empty(count, dtype=float)
        ci_high = numpy.empty(count, dtype=float)
        alpha = self._ci_alpha()
        if alpha is None:
            alpha = 0.025
        for index, metric in enumerate(self.metrics_):
            match metric:
                case _metric.MedianSurvivalMetric():
                    triple = self._compute_median_value(curve, alpha)
                case _metric.RiskScoreMetric():
                    triple = self._compute_risk_score_value(curve)
                case _metric.SurvivalAtMetric() as survival_at:
                    triple = self._compute_survival_at_value(
                        curve, survival_at, alpha
                    )
                case _metric.RmstMetric() as rmst:
                    triple = self._compute_rmst_value(curve, rmst, alpha)
            values[index], ci_low[index], ci_high[index] = triple
        return values, ci_low, ci_high

    def _compute_median_value(
        self,
        curve: _NodeCurve,
        alpha: float,
    ) -> tuple[float, float, float]:
        """Median survival time with Brookmeyer-Crowley CI."""
        value = _survival.compute_median_survival(curve.times, curve.surv)
        if self.ci_coverage is None or len(curve.times) == 0:
            ci_low = float("nan")
            ci_high = float("nan")
        else:
            ci_low, ci_high = _survival.compute_brookmeyer_crowley_ci(
                curve.times, curve.surv, curve.var_log_s, alpha
            )
        return value, ci_low, ci_high

    def _compute_risk_score_value(
        self,
        curve: _NodeCurve,
    ) -> tuple[float, float, float]:
        """Sum of node cumulative hazard at training event times, without CI."""
        value = _survival.compute_risk_score(
            curve.times, curve.d_w, curve.r_w, self.event_grid_
        )
        return value, float("nan"), float("nan")

    def _compute_survival_at_value(
        self,
        curve: _NodeCurve,
        metric: _metric.SurvivalAtMetric,
        alpha: float,
    ) -> tuple[float, float, float]:
        """Kaplan-Meier S(t) at a reference time with log-log CI."""
        query = metric.time
        value = _survival.compute_survival_at(curve.times, curve.surv, query)
        if self.ci_coverage is None:
            ci_low = float("nan")
            ci_high = float("nan")
        else:
            ci_low, ci_high = _survival.compute_log_log_ci_at(
                curve.times, curve.surv, curve.var_log_s, query, alpha
            )
        return value, ci_low, ci_high

    def _compute_rmst_value(
        self,
        curve: _NodeCurve,
        metric: _metric.RmstMetric,
        alpha: float,
    ) -> tuple[float, float, float]:
        """Restricted mean survival time up to a horizon with integrated CI."""
        horizon = metric.horizon
        value = _survival.compute_rmst(curve.times, curve.surv, horizon)
        if self.ci_coverage is None:
            ci_low = float("nan")
            ci_high = float("nan")
        else:
            ci_low, ci_high = _survival.compute_rmst_ci(
                curve.times,
                curve.surv,
                curve.d_w,
                curve.r_w,
                horizon,
                alpha,
            )
        return value, ci_low, ci_high


def _coerce_survival_y(
    y: numpy.typing.ArrayLike,
) -> numpy.typing.NDArray[numpy.floating]:
    """Coerce y from any supported encoding to a 2D float (time, event) array."""
    arr = numpy.asarray(y)
    arr_dtype = arr.dtype
    dtype_names = arr_dtype.names
    if dtype_names is not None:
        names = set(dtype_names)
        if names != {"time", "event"}:
            raise ValueError(
                f"structured y must have exactly fields 'time' and 'event';"
                f" got {sorted(dtype_names)}"
            )
        time = numpy.asarray(arr["time"], dtype=float)
        event = numpy.asarray(arr["event"], dtype=float)
        coerced = numpy.column_stack([time, event])
        return coerced
    is_real_numeric = numpy.issubdtype(
        arr_dtype, numpy.floating
    ) or numpy.issubdtype(arr_dtype, numpy.integer)
    if arr.ndim == 1 and is_real_numeric:
        ages = arr.astype(float)
        time = numpy.abs(ages)
        event = (ages >= 0.0).astype(float)
        coerced = numpy.column_stack([time, event])
        return coerced
    if arr_dtype.kind == "O":
        coerced = numpy.asarray(arr, dtype=float)
        return coerced
    return arr


def _concordance_index(
    time: numpy.typing.NDArray[numpy.floating],
    event: numpy.typing.NDArray[numpy.floating],
    risk: numpy.typing.NDArray[numpy.floating],
    sample_weight: None | numpy.typing.NDArray[numpy.floating],
) -> float:
    """Harrell's concordance index of risk against (time, event)."""
    weights = (
        numpy.ones_like(time)
        if sample_weight is None
        else numpy.asarray(sample_weight, dtype=float)
    )
    event_indices = numpy.flatnonzero(event == 1.0)
    weighted_concordant = 0.0
    weighted_comparable = 0.0
    for index in event_indices:
        later_mask = time > time[index]
        if not numpy.any(later_mask):
            continue
        pair_weights = weights[index] * weights[later_mask]
        risk_others = risk[later_mask]
        contribution = (risk[index] > risk_others).astype(float) + 0.5 * (
            risk[index] == risk_others
        )
        weighted_concordant += float(numpy.sum(pair_weights * contribution))
        weighted_comparable += float(numpy.sum(pair_weights))
    if weighted_comparable == 0.0:
        return 0.5
    score = weighted_concordant / weighted_comparable
    return score


def _evaluate_step_curve(
    curve_times: numpy.typing.NDArray[numpy.floating],
    curve_values: numpy.typing.NDArray[numpy.floating],
    query_times: numpy.typing.NDArray[numpy.floating],
) -> numpy.typing.NDArray[numpy.floating]:
    """Evaluate a right-continuous step curve at query points (1 below first)."""
    n_query = len(query_times)
    if len(curve_times) == 0:
        ones = numpy.ones(n_query, dtype=float)
        return ones
    indices = numpy.searchsorted(curve_times, query_times, side="right") - 1
    result = numpy.where(
        indices < 0, 1.0, curve_values[numpy.maximum(indices, 0)]
    )
    return result


def _build_survival_metric(
    spec: _SurvivalMetricSpec, has_ci: bool
) -> _metric.Metric:
    """Build the descriptor of a single metric specification entry."""
    match spec:
        case str():
            kind = spec
            value: None | float = None
            unit: None | str = None
        case (str() as kind, float() | int() as raw_value, str() as unit):
            value = float(raw_value)
        case _:
            raise TypeError(
                f"metric spec must be a string or a (kind, value, unit)"
                f" tuple; got {spec!r}"
            )
    if kind not in _METRIC_KINDS:
        valid = list(_METRIC_KINDS)
        raise ValueError(
            f"unknown metric kind {kind!r}; valid values are {valid}"
        )
    if kind in _PARAMETRIZED_METRIC_KINDS:
        if value is None:
            raise ValueError(f"{kind!r} requires a (kind, value, unit) tuple")
    else:
        if value is not None:
            raise ValueError(f"{kind!r} does not take parameters; got {spec!r}")
    match kind:
        case "median":
            metric: _metric.Metric = _metric.MedianSurvivalMetric(
                "Median survival", has_ci
            )
        case "risk_score":
            metric = _metric.RiskScoreMetric("Risk score")
        case "survival":
            quantity = typing.cast(float, value)
            unit_name = typing.cast(str, unit)
            label = _format_metric_label("Survival at", quantity, unit_name)
            metric = _metric.SurvivalAtMetric(label, has_ci, quantity)
        # Only "rmst" is left: the kind was validated against _METRIC_KINDS.
        case _:
            quantity = typing.cast(float, value)
            unit_name = typing.cast(str, unit)
            label = _format_metric_label("RMST at", quantity, unit_name)
            metric = _metric.RmstMetric(label, has_ci, quantity)
    return metric


def _format_metric_label(prefix: str, quantity: float, unit: str) -> str:
    """Build the display label of a parametrized metric."""
    formatted = _format_metric_quantity(quantity)
    label = f"{prefix} {formatted} {unit}"
    return label


def _format_metric_quantity(value: float) -> str:
    """Format a metric parameter value, dropping trailing zeros."""
    if float(value).is_integer():
        return str(int(value))
    return str(value)
