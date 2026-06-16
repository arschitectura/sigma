"""ClassificationTree estimator and per-class confidence-interval helpers."""

from __future__ import annotations

import collections.abc
import typing

import numpy
import numpy.typing
import scipy.optimize
import scipy.stats
import sklearn.base
import sklearn.utils.multiclass
import sklearn.utils.validation

from . import _node
from . import _tree
from . import _types

if typing.TYPE_CHECKING:
    import pandas


class ClassificationTree(
    sklearn.base.ClassifierMixin, _tree.Tree[_node.ClassificationNode]
):
    """Conditional inference classification tree.

    Uses permutation-based conditional inference for unbiased variable selection
    and recursive binary partitioning, as described in Hothorn, Hornik, and
    Zeileis (2006), "Unbiased Recursive Partitioning: A Conditional Inference
    Framework," *Journal of Computational and Graphical Statistics*, 15(3),
    651-674.

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
        ci_method: Per-class confidence interval method.
            "agresti_coull": closed-form adjusted Wald interval; slightly
            wider than "wilson" at small sample sizes, convergent at large
            sample sizes.
            "clopper_pearson": exact Beta-based interval; guarantees
            coverage at least ci_coverage but conservative, with intervals
            wider than Wilson, Jeffreys, or Agresti-Coull on average.
            "jeffreys" (default): Bayesian interval using the Jeffreys
            non-informative prior; shorter than Clopper-Pearson on average
            while retaining close-to-nominal coverage.
            "mid_p_exact": mid-p variant of Clopper-Pearson; strictly
            narrower than Clopper-Pearson with average coverage close to
            nominal.
            "wilson": closed-form Wilson score interval; accurate for
            moderate sample sizes.
            "wilson_cc": continuity-corrected Wilson; slightly wider than
            Wilson, restoring lower-tail coverage at small sample sizes.
        ci_coverage: Coverage level for per-class confidence intervals on node
            class proportions. Defaults to 0.95. Set to None to disable CI
            computation.
        transmuter: Optional callable applied to node data before computing
            predictions and class distributions, with post-hoc split validation.
            See Tree for full signature and behavior.
        resamples: Number of permutations for min-P resampling when
            test_type="monte_carlo". Must be a positive integer. Ignored for
            other test_type values.
        decorator: Optional callable producing a per-node decoration stored on
            the node and rendered by to_text and to_image. See Tree
            for full signature and behavior.
        random_state: Seed for the random number generator used in permutation
            resampling. Pass an integer for reproducibility. None uses an
            unpredictable seed. Ignored unless test_type="monte_carlo".

    Attributes:
        content_: Root node of the fitted tree structure.
        leaves_: List of leaf nodes, ordered by ascending prediction value.
        nodes_: List of all nodes in pre-order DFS, ordered by node_id.
            Indices match the output of predict_index.
        classes_: Unique class labels, shape (n_classes,).
        n_classes_: Number of classes.
        n_features_in_: Number of features seen during fit.
        feature_types_: Per-feature CovariateType, shape (n_features,).
    """

    classes_: numpy.typing.NDArray
    n_classes_: int

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
        ci_method: typing.Literal[
            "agresti_coull",
            "clopper_pearson",
            "jeffreys",
            "mid_p_exact",
            "wilson",
            "wilson_cc",
        ] = "jeffreys",
        ci_coverage: None | float = 0.95,
        transmuter: None | typing.Callable = None,
        resamples: None | int = None,
        decorator: None | typing.Callable = None,
        random_state: None | int = None,
        reverse_order: bool = False,
    ) -> None:
        _types._validate_literal_param(
            ci_method, _types.CiMethodClassificationTree, "ci_method"
        )
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
        """Validate inputs and encode class labels as integers."""
        X_validated, y_validated = sklearn.utils.validation.validate_data(
            self, X, y, dtype=None
        )
        X_array = typing.cast(
            numpy.typing.NDArray[numpy.floating],
            numpy.asarray(X_validated, dtype=numpy.float64),
        )
        sklearn.utils.multiclass.check_classification_targets(y_validated)
        if hasattr(y, "cat"):
            import pandas

            if isinstance(y, pandas.Series) and isinstance(
                y.dtype, pandas.CategoricalDtype
            ):
                y_cat = y.cat
                classes = numpy.asarray(y_cat.categories, dtype=object)
                y_encoded = numpy.asarray(y_cat.codes, dtype=float)
                self.classes_ = classes
                self.n_classes_ = len(classes)
                return X_array, y_encoded
        classes = numpy.unique(y_validated)
        y_encoded = numpy.searchsorted(classes, y_validated).astype(float)
        self.classes_ = classes
        self.n_classes_ = len(classes)
        return X_array, y_encoded

    def _effective_class_names(
        self,
        class_names: None | list[str] = None,
    ) -> None | list[str]:
        """Return display class names: explicit override, else classes_."""
        if class_names is not None:
            return class_names
        classes = getattr(self, "classes_", None)
        if classes is None:
            return None
        names = [str(value) for value in classes]
        return names

    def _validate_offset(
        self,
        offset: None | numpy.typing.NDArray[numpy.floating],
        n_samples: int,
    ) -> None | numpy.typing.NDArray[numpy.floating]:
        """Validate the classification offset (n_samples, n_classes)."""
        if offset is None:
            return None
        expected_shape = (n_samples, self.n_classes_)
        offset_array = self._validate_offset_shape_finite(
            offset, expected_shape
        )
        if numpy.any(offset_array < 0.0) or numpy.any(offset_array > 1.0):
            raise ValueError("offset values must lie in [0, 1]")
        row_sums = offset_array.sum(axis=1)
        if not numpy.allclose(row_sums, 1.0, atol=1e-6):
            raise ValueError("offset rows must sum to 1 within 1e-6 tolerance")
        return offset_array

    def _validate_transmuted_y_shape(
        self,
        y_out: numpy.typing.NDArray[numpy.floating],
    ) -> int:
        """Validate transmuted y shape for classification (1D)."""
        y_out_shape = y_out.shape
        if y_out.ndim != 1:
            raise ValueError(
                f"transmuter y must be 1D for classification,"
                f" got shape {y_out_shape}"
            )
        return y_out_shape[0]

    def _make_node(
        self, payload: _tree._NodePayload
    ) -> _node.ClassificationNode:
        """Construct a ClassificationNode with the per-class CI payload."""
        node = _node.ClassificationNode(
            depth=payload.depth,
            n_samples=payload.n_samples,
            share=0.0,
            decoration=None,
            extension=payload.extension,
            prediction=int(payload.prediction),
            class_distribution=typing.cast(
                numpy.typing.NDArray[numpy.floating], payload.class_distribution
            ),
            ci_low=payload.ci_low_per_class,
            ci_high=payload.ci_high_per_class,
            mean_offset_proba=payload.mean_offset_proba,
        )
        return node

    def _compute_influence(
        self,
        y: numpy.typing.NDArray[numpy.floating],
        offset: None | numpy.typing.NDArray[numpy.floating],
    ) -> numpy.typing.NDArray[numpy.floating]:
        """Compute the per-sample influence function for classification."""
        y_int = y.astype(int)
        h = numpy.eye(self.n_classes_)[y_int]
        if offset is None:
            return h
        residual = h - offset
        return residual

    def _compute_prediction(
        self,
        y: numpy.typing.NDArray[numpy.floating],
        weights: numpy.typing.NDArray[numpy.floating],
        offset: None | numpy.typing.NDArray[numpy.floating],
    ) -> float:
        """Compute the majority class index (weighted)."""
        y_int = y.astype(int)
        counts = numpy.bincount(
            y_int, weights=weights, minlength=self.n_classes_
        )
        if offset is not None:
            w_sum = float(weights.sum())
            empirical = counts / w_sum if w_sum > 0.0 else counts
            mean_offset = self._compute_mean_offset_proba(weights, offset)
            if mean_offset is not None:
                offset_eps = _tree._OFFSET_EPS
                log_unnorm = numpy.log(
                    numpy.maximum(empirical, offset_eps)
                ) - numpy.log(numpy.maximum(mean_offset, offset_eps))
                majority_idx = int(numpy.argmax(log_unnorm))
                prediction = float(majority_idx)
                return prediction
        majority_idx = int(numpy.argmax(counts))
        prediction = float(majority_idx)
        return prediction

    def _compute_mean_offset_proba(
        self,
        weights: numpy.typing.NDArray[numpy.floating],
        offset: None | numpy.typing.NDArray[numpy.floating],
    ) -> None | numpy.typing.NDArray[numpy.floating]:
        """Weighted mean of fit-time offset rows over the active samples."""
        if offset is None:
            return None
        active = weights > 0
        w_active = weights[active]
        w_sum = float(w_active.sum())
        if w_sum <= 0.0:
            return None
        mean = (w_active[:, None] * offset[active]).sum(axis=0) / w_sum
        return mean

    def _is_constant_response(
        self,
        y: numpy.typing.NDArray[numpy.floating],
        weights: numpy.typing.NDArray[numpy.floating],
        offset: None | numpy.typing.NDArray[numpy.floating],
    ) -> bool:
        """Check if the response carries no signal in this node."""
        active = weights > 0
        n_unique = len(numpy.unique(y[active]))
        if n_unique > 1:
            return False
        if offset is not None:
            rows = offset[active]
            if rows.shape[0] <= 1:
                return True
            varying = bool(numpy.any(numpy.ptp(rows, axis=0) > 0.0))
            return not varying
        return True

    def _compute_per_class_ci(
        self,
        y: numpy.typing.NDArray[numpy.floating],
        weights: numpy.typing.NDArray[numpy.floating],
    ) -> tuple[
        None | numpy.typing.NDArray[numpy.floating],
        None | numpy.typing.NDArray[numpy.floating],
    ]:
        """Compute per-class CI using self.ci_method."""
        alpha = self._ci_alpha()
        if alpha is None:
            return None, None
        w_total = float(weights.sum())
        n_classes = self.n_classes_
        if w_total == 0.0:
            ci_low = numpy.zeros(n_classes)
            ci_high = numpy.ones(n_classes)
            return ci_low, ci_high
        ci_low = numpy.empty(n_classes)
        ci_high = numpy.empty(n_classes)
        y_int = y.astype(int)
        ci_method_enum = _types.CiMethodClassificationTree(self.ci_method)
        for k in range(n_classes):
            w_k = float(weights[y_int == k].sum())
            w_rest = w_total - w_k
            match ci_method_enum:
                case _types.CiMethodClassificationTree.AGRESTI_COULL:
                    low_k, high_k = _compute_class_ci_agresti_coull(
                        w_k, w_total, alpha
                    )
                case _types.CiMethodClassificationTree.CLOPPER_PEARSON:
                    low_k, high_k = _compute_class_ci_clopper_pearson(
                        w_k, w_rest, alpha
                    )
                case _types.CiMethodClassificationTree.JEFFREYS:
                    low_k, high_k = _compute_class_ci_jeffreys(
                        w_k, w_rest, alpha
                    )
                case _types.CiMethodClassificationTree.MID_P_EXACT:
                    low_k, high_k = _compute_class_ci_mid_p_exact(
                        w_k, w_rest, alpha
                    )
                case _types.CiMethodClassificationTree.WILSON:
                    low_k, high_k = _compute_class_ci_wilson(
                        w_k, w_total, alpha
                    )
                case _types.CiMethodClassificationTree.WILSON_CC:
                    low_k, high_k = _compute_class_ci_wilson_cc(
                        w_k, w_total, alpha
                    )
            ci_low[k] = low_k
            ci_high[k] = high_k
        return ci_low, ci_high

    def _compute_class_distribution(
        self,
        y: numpy.typing.NDArray[numpy.floating],
        weights: numpy.typing.NDArray[numpy.floating],
    ) -> None | numpy.typing.NDArray[numpy.floating]:
        """Compute the weighted class probability distribution."""
        y_int = y.astype(int)
        counts = numpy.bincount(
            y_int, weights=weights, minlength=self.n_classes_
        )
        w_sum = weights.sum()
        distribution = counts / w_sum
        return distribution

    def predict(
        self,
        X: numpy.typing.NDArray[numpy.floating] | pandas.DataFrame,
        offset: None | numpy.typing.NDArray[numpy.floating] = None,
    ) -> numpy.typing.NDArray:
        """Predict class labels for the given samples.

        Args:
            X: Samples to predict, shape (n_samples, n_features).
            offset: Optional per-sample baseline class probabilities, shape
                (n_samples, n_classes), summing to 1 along the class axis.
                When None and the model was fit with offset, defaults to
                uniform 1/n_classes.

        Returns:
            Predicted class labels, shape (n_samples,).
        """
        if offset is None and not self._fit_with_offset:
            node_indices = self.predict_index(X)
            class_indices = numpy.array(
                [int(node.prediction) for node in self.nodes_]
            )
            predictions = self.classes_[class_indices[node_indices]]
            return predictions
        proba = self.predict_proba(X, offset=offset)
        class_indices = numpy.argmax(proba, axis=1)
        predictions = self.classes_[class_indices]
        return predictions

    def predict_proba(
        self,
        X: numpy.typing.NDArray[numpy.floating] | pandas.DataFrame,
        offset: None | numpy.typing.NDArray[numpy.floating] = None,
    ) -> numpy.typing.NDArray[numpy.floating]:
        """Predict class probabilities for the given samples.

        Args:
            X: Samples to predict, shape (n_samples, n_features).
            offset: Optional per-sample baseline class probabilities, shape
                (n_samples, n_classes), summing to 1 along the class axis.
                When None and the model was fit with offset, defaults to
                uniform 1/n_classes.

        Returns:
            Predicted class probabilities, shape (n_samples, n_classes).
        """
        node_indices = self.predict_index(X)
        nodes = self.nodes_
        all_distributions = numpy.array(
            [node.class_distribution for node in nodes]
        )
        node_p = all_distributions[node_indices]
        n_pred = len(node_indices)
        fit_with_offset = self._fit_with_offset
        if offset is None and not fit_with_offset:
            return node_p
        n_classes = self.n_classes_
        offset_eps = _tree._OFFSET_EPS
        if offset is None:
            offset_new = numpy.full((n_pred, n_classes), 1.0 / n_classes)
        else:
            offset_new = self._validate_predict_offset(offset, n_pred)
        log_term = numpy.log(numpy.maximum(offset_new, offset_eps)) + numpy.log(
            numpy.maximum(node_p, offset_eps)
        )
        if fit_with_offset:
            mean_off_per_node = numpy.array(
                [
                    mean_offset_proba
                    if (mean_offset_proba := node.mean_offset_proba) is not None
                    else numpy.full(n_classes, 1.0 / n_classes)
                    for node in nodes
                ]
            )
            node_mean_off = mean_off_per_node[node_indices]
            log_term = log_term - numpy.log(
                numpy.maximum(node_mean_off, offset_eps)
            )
        log_term = log_term - log_term.max(axis=1, keepdims=True)
        unnorm = numpy.exp(log_term)
        probabilities = unnorm / unnorm.sum(axis=1, keepdims=True)
        return probabilities

    def _validate_predict_offset(
        self,
        offset: numpy.typing.NDArray[numpy.floating],
        n_samples: int,
    ) -> numpy.typing.NDArray[numpy.floating]:
        """Validate the predict-time offset for classification."""
        expected_shape = (n_samples, self.n_classes_)
        offset_array = self._validate_offset_shape_finite(
            offset, expected_shape
        )
        if numpy.any(offset_array < 0.0) or numpy.any(offset_array > 1.0):
            raise ValueError("offset values must lie in [0, 1]")
        row_sums = offset_array.sum(axis=1)
        if not numpy.allclose(row_sums, 1.0, atol=1e-6):
            raise ValueError("offset rows must sum to 1 within 1e-6 tolerance")
        return offset_array


def _compute_class_ci_agresti_coull(
    w_k: float,
    w_total: float,
    alpha: float,
) -> tuple[float, float]:
    """Agresti-Coull adjusted Wald interval for a single class proportion."""
    z = float(scipy.stats.norm.ppf(1.0 - alpha))
    z_sq = z * z
    n_tilde = w_total + z_sq
    p_tilde = (w_k + z_sq / 2.0) / n_tilde
    half = z * float(numpy.sqrt(p_tilde * (1.0 - p_tilde) / n_tilde))
    ci_low = float(max(0.0, p_tilde - half))
    ci_high = float(min(1.0, p_tilde + half))
    return ci_low, ci_high


def _compute_class_ci_clopper_pearson(
    w_k: float,
    w_rest: float,
    alpha: float,
) -> tuple[float, float]:
    """Clopper-Pearson exact Beta interval for a single class proportion."""
    if w_k == 0.0:
        ci_low = 0.0
        ci_high = float(scipy.stats.beta.ppf(1.0 - alpha, 1.0, w_rest))
        return ci_low, ci_high
    if w_rest == 0.0:
        ci_low = float(scipy.stats.beta.ppf(alpha, w_k, 1.0))
        ci_high = 1.0
        return ci_low, ci_high
    ci_low = float(scipy.stats.beta.ppf(alpha, w_k, w_rest + 1.0))
    ci_high = float(scipy.stats.beta.ppf(1.0 - alpha, w_k + 1.0, w_rest))
    return ci_low, ci_high


def _compute_class_ci_jeffreys(
    w_k: float,
    w_rest: float,
    alpha: float,
) -> tuple[float, float]:
    """Jeffreys-prior Beta credible interval for a single class proportion."""
    if w_k == 0.0:
        ci_low = 0.0
        ci_high = float(scipy.stats.beta.ppf(1.0 - alpha, 0.5, w_rest + 0.5))
        return ci_low, ci_high
    if w_rest == 0.0:
        ci_low = float(scipy.stats.beta.ppf(alpha, w_k + 0.5, 0.5))
        ci_high = 1.0
        return ci_low, ci_high
    ci_low = float(scipy.stats.beta.ppf(alpha, w_k + 0.5, w_rest + 0.5))
    ci_high = float(scipy.stats.beta.ppf(1.0 - alpha, w_k + 0.5, w_rest + 0.5))
    return ci_low, ci_high


def _compute_class_ci_mid_p_exact(
    w_k: float,
    w_rest: float,
    alpha: float,
) -> tuple[float, float]:
    """Mid-p exact interval for a single class proportion."""
    if w_k == 0.0:
        ci_low = 0.0
        ci_high = float(1.0 - (2.0 * alpha) ** (1.0 / w_rest))
        return ci_low, ci_high
    if w_rest == 0.0:
        ci_low = float((2.0 * alpha) ** (1.0 / w_k))
        ci_high = 1.0
        return ci_low, ci_high
    p_hat = w_k / (w_k + w_rest)
    cp_low = float(scipy.stats.beta.ppf(alpha, w_k, w_rest + 1.0))
    cp_high = float(scipy.stats.beta.ppf(1.0 - alpha, w_k + 1.0, w_rest))

    def f_low(p: float) -> float:
        value = (
            0.5 * scipy.stats.beta.cdf(p, w_k, w_rest + 1.0)
            + 0.5 * scipy.stats.beta.cdf(p, w_k + 1.0, w_rest)
            - alpha
        )
        return value

    def f_high(p: float) -> float:
        value = (
            0.5 * scipy.stats.beta.sf(p, w_k, w_rest + 1.0)
            + 0.5 * scipy.stats.beta.sf(p, w_k + 1.0, w_rest)
            - alpha
        )
        return value

    ci_low = float(
        scipy.optimize.brentq(
            f_low, cp_low, p_hat, xtol=1e-12, rtol=1e-12, maxiter=200
        )
    )
    ci_high = float(
        scipy.optimize.brentq(
            f_high, p_hat, cp_high, xtol=1e-12, rtol=1e-12, maxiter=200
        )
    )
    return ci_low, ci_high


def _compute_class_ci_wilson(
    w_k: float,
    w_total: float,
    alpha: float,
) -> tuple[float, float]:
    """Wilson score interval for a single class proportion."""
    z = float(scipy.stats.norm.ppf(1.0 - alpha))
    z_sq = z * z
    p_hat = w_k / w_total
    denom = w_total + z_sq
    center = (w_k + z_sq / 2.0) / denom
    half = z * numpy.sqrt(w_total * p_hat * (1.0 - p_hat) + z_sq / 4.0) / denom
    ci_low = float(max(0.0, center - half))
    ci_high = float(min(1.0, center + half))
    return ci_low, ci_high


def _compute_class_ci_wilson_cc(
    w_k: float,
    w_total: float,
    alpha: float,
) -> tuple[float, float]:
    """Continuity-corrected Wilson score interval for a single class proportion."""
    z = float(scipy.stats.norm.ppf(1.0 - alpha))
    z_sq = z * z
    p_hat = w_k / w_total
    denom = 2.0 * (w_total + z_sq)
    base = z_sq - 1.0 / w_total + 4.0 * w_k * (1.0 - p_hat)
    rad_low = max(0.0, base + (4.0 * p_hat - 2.0))
    rad_high = max(0.0, base - (4.0 * p_hat - 2.0))
    disc_low = z * float(numpy.sqrt(rad_low))
    disc_high = z * float(numpy.sqrt(rad_high))
    ci_low = float(max(0.0, (2.0 * w_k + z_sq - 1.0 - disc_low) / denom))
    ci_high = float(min(1.0, (2.0 * w_k + z_sq + 1.0 + disc_high) / denom))
    if w_k == 0.0:
        ci_low = 0.0
    if w_k == w_total:
        ci_high = 1.0
    return ci_low, ci_high
