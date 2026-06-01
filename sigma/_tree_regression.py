"""RegressionTree estimator and its weighted-mean confidence intervals."""

from __future__ import annotations

import collections.abc
import typing

import numpy
import numpy.typing
import scipy.stats
import sklearn.base
import sklearn.utils.multiclass
import sklearn.utils.validation

from . import _node
from . import _tree
from . import _tree_classification
from . import _types

if typing.TYPE_CHECKING:
    import pandas


class RegressionTree(
    sklearn.base.RegressorMixin, _tree.Tree[_node.RegressionNode]
):
    """Conditional inference regression tree.

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
        ci_method: Confidence interval method.
            "bayesian_bootstrap" (default): Bayesian bootstrap interval.
            "bca": bias-corrected and accelerated bootstrap interval;
            frequentist counterpart to "bayesian_bootstrap".
            Non-deterministic.
            "beta": Beta interval for a continuous proportional response;
            requires y in [0, 1].
            "exponential": exact chi-squared interval for an Exponential
            mean; requires y >= 0.
            "gamma": exact chi-squared interval for a Gamma mean; requires
            y >= 0.
            "log_normal": Cox's interval for the log-normal mean; requires
            y > 0.
            "log_normal_gci": generalized confidence interval for the
            log-normal mean; requires y > 0. Non-deterministic.
            "normal": normal-approximation interval on the weighted mean.
            "poisson": exact Garwood chi-squared interval for a Poisson
            rate; requires y >= 0.
            "poisson_jeffreys": equal-tailed Jeffreys interval for a
            Poisson rate; requires y >= 0. Shorter than "poisson" at
            moderate rates.
            "student_t": Student-t interval on the weighted mean; wider
            than "normal" for small effective sample sizes.
        ci_coverage: Coverage level for node-prediction confidence intervals.
            Defaults to 0.95. Set to None to disable CI computation.
        transmuter: Optional callable applied to node data before computing
            predictions and confidence intervals, with post-hoc split
            validation. See Tree for full signature and behavior.
        resamples: Number of permutations for min-P resampling when
            test_type="monte_carlo". Must be a positive integer. Ignored for
            other test_type values.
        decorator: Optional callable producing a per-node decoration stored on
            the node and rendered by to_text and to_image. See Tree
            for full signature and behavior.
        random_state: Seed for stochastic operations. Pass an integer for
            reproducibility; None uses an unpredictable seed. Controls
            min-P permutation resampling under test_type="monte_carlo",
            the bootstrap-family CI methods ("bayesian_bootstrap",
            "bca", "log_normal_gci"), and the jitter of
            to_image(kind="response") raincloud plots.
        response_sample_size: Maximum number of response samples stored on
            each leaf for the response-distribution overlay in
            to_image(kind="response"). Set to 0 to disable (each leaf
            carries an empty response_samples array). Defaults to 1000. Must
            be a non-negative integer.

    Attributes:
        content_: Root node of the fitted tree structure.
        leaves_: List of leaf nodes, ordered by ascending prediction value.
        nodes_: List of all nodes in pre-order DFS, ordered by node_id.
            Indices match the output of predict_index.
        n_features_in_: Number of features seen during fit.
        feature_types_: Per-feature CovariateType, shape (n_features,).
    """

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
            "bayesian_bootstrap",
            "bca",
            "beta",
            "exponential",
            "gamma",
            "log_normal",
            "log_normal_gci",
            "normal",
            "poisson",
            "poisson_jeffreys",
            "student_t",
        ] = "bayesian_bootstrap",
        ci_coverage: None | float = 0.95,
        transmuter: None | typing.Callable = None,
        resamples: None | int = None,
        decorator: None | typing.Callable = None,
        random_state: None | int = None,
        response_sample_size: int = 1000,
        reverse_order: bool = False,
    ) -> None:
        _types._validate_literal_param(
            ci_method, _types.CiMethodRegressionTree, "ci_method"
        )
        if (
            not isinstance(response_sample_size, int)
            or isinstance(response_sample_size, bool)
            or response_sample_size < 0
        ):
            raise ValueError(
                f"response_sample_size must be a non-negative integer,"
                f" got {response_sample_size!r}"
            )
        self.ci_method = ci_method
        self.response_sample_size = response_sample_size
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
        """Validate inputs for regression."""
        sklearn.utils.multiclass.type_of_target(y, raise_unknown=True)
        X_validated, y_validated = sklearn.utils.validation.validate_data(
            self, X, y, dtype="float64"
        )
        X_array = typing.cast(numpy.typing.NDArray[numpy.floating], X_validated)
        y_array = typing.cast(numpy.typing.NDArray[numpy.floating], y_validated)
        return X_array, y_array

    def _validate_offset(
        self,
        offset: None | numpy.typing.NDArray[numpy.floating],
        n_samples: int,
    ) -> None | numpy.typing.NDArray[numpy.floating]:
        """Validate the regression offset (1D, length n_samples, finite)."""
        if offset is None:
            return None
        offset_array = numpy.asarray(offset, dtype=float)
        offset_shape = offset_array.shape
        if offset_array.ndim != 1 or offset_shape[0] != n_samples:
            raise ValueError(
                f"offset must be 1D with length {n_samples},"
                f" got shape {offset_shape}"
            )
        if not numpy.all(numpy.isfinite(offset_array)):
            raise ValueError("offset values must be finite")
        return offset_array

    def _validate_transmuted_y_shape(
        self,
        y_out: numpy.typing.NDArray[numpy.floating],
    ) -> int:
        """Validate transmuted y shape for regression (1D)."""
        y_out_shape = y_out.shape
        if y_out.ndim != 1:
            raise ValueError(
                f"transmuter y must be 1D for regression,"
                f" got shape {y_out_shape}"
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
        """Construct a RegressionNode with the regression-relevant payload."""
        node = _node.RegressionNode(
            depth=depth,
            n_samples=n_samples,
            share=0.0,
            decoration=None,
            extension=extension,
            prediction=prediction,
            ci_low=ci_low,
            ci_high=ci_high,
            response_samples=(
                response_samples
                if response_samples is not None
                else numpy.empty(0, dtype=float)
            ),
        )
        return node

    def _compute_response_samples_for_leaf(
        self,
        y_transmuted: numpy.typing.NDArray[numpy.floating],
        w_transmuted: numpy.typing.NDArray[numpy.floating],
        offset_transmuted: None | numpy.typing.NDArray[numpy.floating],
    ) -> None | numpy.typing.NDArray[numpy.floating]:
        """Subsample post-transmutation residuals from a leaf's active rows."""
        size = self.response_sample_size
        if size == 0:
            return numpy.empty(0, dtype=float)
        active = w_transmuted > 0
        if not numpy.any(active):
            return numpy.empty(0, dtype=float)
        if offset_transmuted is not None:
            residuals = y_transmuted[active] - offset_transmuted[active]
        else:
            residuals = y_transmuted[active]
        w_active = w_transmuted[active]
        n_residuals = residuals.shape[0]
        if n_residuals <= size:
            return residuals.copy()
        probabilities = w_active / w_active.sum()
        indices = self._rng_.choice(
            n_residuals, size=size, replace=False, p=probabilities
        )
        sampled = residuals[indices]
        return sampled

    def _compute_influence(
        self,
        y: numpy.typing.NDArray[numpy.floating],
        offset: None | numpy.typing.NDArray[numpy.floating],
    ) -> numpy.typing.NDArray[numpy.floating]:
        """Return h(Y_i) = Y_i (or Y_i - offset_i when offset is given)."""
        if offset is None:
            result = y.reshape(-1, 1)
            return result
        residual = y - offset
        result = residual.reshape(-1, 1)
        return result

    def _compute_prediction(
        self,
        y: numpy.typing.NDArray[numpy.floating],
        weights: numpy.typing.NDArray[numpy.floating],
        offset: None | numpy.typing.NDArray[numpy.floating],
    ) -> float:
        """Compute the weighted mean of the response (or residual)."""
        w_sum = weights.sum()
        if offset is not None:
            residual = y - offset
            prediction = float(numpy.dot(weights, residual) / w_sum)
            return prediction
        prediction = float(numpy.dot(weights, y) / w_sum)
        return prediction

    def _is_constant_response(
        self,
        y: numpy.typing.NDArray[numpy.floating],
        weights: numpy.typing.NDArray[numpy.floating],
        offset: None | numpy.typing.NDArray[numpy.floating],
    ) -> bool:
        """Check if all active residuals are identical."""
        active = weights > 0
        if offset is not None:
            residual = y[active] - offset[active]
            is_constant = numpy.ptp(residual) == 0.0
            return is_constant
        is_constant = numpy.ptp(y[active]) == 0.0
        return is_constant

    def _compute_ci(
        self,
        y: numpy.typing.NDArray[numpy.floating],
        weights: numpy.typing.NDArray[numpy.floating],
        offset: None | numpy.typing.NDArray[numpy.floating],
    ) -> tuple[None | float, None | float]:
        """Compute a confidence interval for the weighted mean."""
        ci_coverage = self.ci_coverage
        if ci_coverage is None:
            return None, None
        active = weights > 0
        y_active = y[active]
        if offset is not None:
            y_active = y_active - offset[active]
        n_active = len(y_active)
        if n_active <= 1:
            prediction = float(y_active[0]) if n_active == 1 else 0.0
            return prediction, prediction
        w_active = weights[active]
        alpha = (1.0 - ci_coverage) / 2.0
        ci_method_enum = _types.CiMethodRegressionTree(self.ci_method)
        match ci_method_enum:
            case _types.CiMethodRegressionTree.BAYESIAN_BOOTSTRAP:
                ci_low, ci_high = self._compute_ci_bayesian_bootstrap(
                    self._rng_ci_, y_active, w_active, alpha
                )
            case _types.CiMethodRegressionTree.BCA:
                ci_low, ci_high = self._compute_ci_bca(
                    self._rng_ci_, y_active, w_active, alpha
                )
            case _types.CiMethodRegressionTree.BETA:
                ci_low, ci_high = self._compute_ci_beta(
                    y_active, w_active, alpha
                )
            case _types.CiMethodRegressionTree.EXPONENTIAL:
                ci_low, ci_high = self._compute_ci_exponential(
                    y_active, w_active, alpha
                )
            case _types.CiMethodRegressionTree.GAMMA:
                ci_low, ci_high = self._compute_ci_gamma(
                    y_active, w_active, alpha
                )
            case _types.CiMethodRegressionTree.LOG_NORMAL:
                ci_low, ci_high = self._compute_ci_log_normal(
                    y_active, w_active, alpha
                )
            case _types.CiMethodRegressionTree.LOG_NORMAL_GCI:
                ci_low, ci_high = self._compute_ci_log_normal_gci(
                    self._rng_ci_, y_active, w_active, alpha
                )
            case _types.CiMethodRegressionTree.NORMAL:
                ci_low, ci_high = self._compute_ci_normal(
                    y_active, w_active, alpha
                )
            case _types.CiMethodRegressionTree.POISSON:
                ci_low, ci_high = self._compute_ci_poisson(
                    y_active, w_active, alpha
                )
            case _types.CiMethodRegressionTree.POISSON_JEFFREYS:
                ci_low, ci_high = self._compute_ci_poisson_jeffreys(
                    y_active, w_active, alpha
                )
            case _types.CiMethodRegressionTree.STUDENT_T:
                ci_low, ci_high = self._compute_ci_student_t(
                    y_active, w_active, alpha
                )
        return ci_low, ci_high

    @staticmethod
    def _compute_ci_bayesian_bootstrap(
        rng: numpy.random.Generator,
        y_active: numpy.typing.NDArray[numpy.floating],
        w_active: numpy.typing.NDArray[numpy.floating],
        alpha: float,
    ) -> tuple[float, float]:
        """Bayesian bootstrap CI for the weighted mean via Dirichlet draws."""
        n_draws = 10_000
        dirichlet_weights = rng.dirichlet(w_active, size=n_draws)
        means = numpy.dot(dirichlet_weights, y_active)
        ci_low = float(numpy.quantile(means, alpha))
        ci_high = float(numpy.quantile(means, 1.0 - alpha))
        return ci_low, ci_high

    @staticmethod
    def _compute_ci_bca(
        rng: numpy.random.Generator,
        y_active: numpy.typing.NDArray[numpy.floating],
        w_active: numpy.typing.NDArray[numpy.floating],
        alpha: float,
    ) -> tuple[float, float]:
        """Bias-corrected and accelerated bootstrap CI for the weighted mean."""
        w_sum = float(w_active.sum())
        theta_hat = float(numpy.dot(w_active, y_active) / w_sum)
        weighted_variance = float(
            numpy.dot(w_active, (y_active - theta_hat) ** 2)
        )
        if weighted_variance == 0.0:
            return theta_hat, theta_hat
        n = len(y_active)
        n_draws = 10_000
        probabilities = w_active / w_sum
        bootstrap_indices = rng.choice(
            n, size=(n_draws, n), replace=True, p=probabilities
        )
        bootstrap_means = y_active[bootstrap_indices].mean(axis=1)
        proportion_below = float((bootstrap_means < theta_hat).mean())
        proportion_clamped = max(
            0.5 / n_draws, min(1.0 - 0.5 / n_draws, proportion_below)
        )
        z0 = float(scipy.stats.norm.ppf(proportion_clamped))
        total_wy = float(numpy.dot(w_active, y_active))
        jackknife = (total_wy - w_active * y_active) / (w_sum - w_active)
        jackknife_mean = float(jackknife.mean())
        deviations = jackknife_mean - jackknife
        sum_cubed = float(numpy.sum(deviations**3))
        sum_squared = float(numpy.sum(deviations**2))
        if sum_squared == 0.0:
            acceleration = 0.0
        else:
            acceleration = sum_cubed / (6.0 * sum_squared**1.5)
        z_lo = float(scipy.stats.norm.ppf(alpha))
        z_hi = float(scipy.stats.norm.ppf(1.0 - alpha))
        denominator_lo = 1.0 - acceleration * (z0 + z_lo)
        denominator_hi = 1.0 - acceleration * (z0 + z_hi)
        if denominator_lo <= 0.0:
            adjusted_lo = alpha
        else:
            adjusted_lo = float(
                scipy.stats.norm.cdf(z0 + (z0 + z_lo) / denominator_lo)
            )
        if denominator_hi <= 0.0:
            adjusted_hi = 1.0 - alpha
        else:
            adjusted_hi = float(
                scipy.stats.norm.cdf(z0 + (z0 + z_hi) / denominator_hi)
            )
        ci_low = float(numpy.quantile(bootstrap_means, adjusted_lo))
        ci_high = float(numpy.quantile(bootstrap_means, adjusted_hi))
        return ci_low, ci_high

    @staticmethod
    def _compute_ci_beta(
        y_active: numpy.typing.NDArray[numpy.floating],
        w_active: numpy.typing.NDArray[numpy.floating],
        alpha: float,
    ) -> tuple[float, float]:
        """Clopper-Pearson-style Beta CI for a [0, 1] proportional response."""
        if numpy.any(y_active < 0.0) or numpy.any(y_active > 1.0):
            raise ValueError("beta CI requires all response values in [0, 1]")
        w_sum = float(w_active.sum())
        weighted_successes = float(numpy.dot(w_active, y_active))
        weighted_failures = w_sum - weighted_successes
        ci_low, ci_high = (
            _tree_classification._compute_class_ci_clopper_pearson(
                weighted_successes, weighted_failures, alpha
            )
        )
        return ci_low, ci_high

    @staticmethod
    def _compute_ci_exponential(
        y_active: numpy.typing.NDArray[numpy.floating],
        w_active: numpy.typing.NDArray[numpy.floating],
        alpha: float,
    ) -> tuple[float, float]:
        """Exact chi-squared CI for the mean of an Exponential response."""
        if numpy.any(y_active < 0.0):
            raise ValueError("exponential CI requires all response values >= 0")
        w_sum = float(w_active.sum())
        p_hat = float(numpy.dot(w_active, y_active) / w_sum)
        if p_hat == 0.0:
            return 0.0, 0.0
        df = 2.0 * w_sum
        chi2_low = float(scipy.stats.chi2.ppf(alpha, df))
        chi2_high = float(scipy.stats.chi2.ppf(1.0 - alpha, df))
        ci_low = 2.0 * w_sum * p_hat / chi2_high
        ci_high = 2.0 * w_sum * p_hat / chi2_low
        return ci_low, ci_high

    @staticmethod
    def _compute_ci_gamma(
        y_active: numpy.typing.NDArray[numpy.floating],
        w_active: numpy.typing.NDArray[numpy.floating],
        alpha: float,
    ) -> tuple[float, float]:
        """Exact chi-squared CI for the mean of a Gamma response."""
        if numpy.any(y_active < 0.0):
            raise ValueError("gamma CI requires all response values >= 0")
        w_sum = w_active.sum()
        p_hat = float(numpy.dot(w_active, y_active) / w_sum)
        if p_hat == 0.0:
            return 0.0, 0.0
        variance = float(numpy.dot(w_active, (y_active - p_hat) ** 2) / w_sum)
        if variance == 0.0:
            return p_hat, p_hat
        n_eff = float(w_sum**2 / numpy.dot(w_active, w_active))
        alpha_shape = p_hat**2 / variance
        df = 2.0 * n_eff * alpha_shape
        chi2_low = float(scipy.stats.chi2.ppf(alpha, df))
        chi2_high = float(scipy.stats.chi2.ppf(1.0 - alpha, df))
        ci_low = df * p_hat / chi2_high
        ci_high = df * p_hat / chi2_low
        return ci_low, ci_high

    @staticmethod
    def _compute_ci_log_normal(
        y_active: numpy.typing.NDArray[numpy.floating],
        w_active: numpy.typing.NDArray[numpy.floating],
        alpha: float,
    ) -> tuple[float, float]:
        """Cox's method CI for the arithmetic mean of a log-normal response."""
        if numpy.any(y_active <= 0.0):
            raise ValueError("log_normal CI requires all response values > 0")
        log_y = numpy.log(y_active)
        w_sum = w_active.sum()
        mu_log = float(numpy.dot(w_active, log_y) / w_sum)
        sigma_sq = float(numpy.dot(w_active, (log_y - mu_log) ** 2) / w_sum)
        if sigma_sq == 0.0:
            point = float(numpy.exp(mu_log))
            return point, point
        n_eff = float(w_sum**2 / numpy.dot(w_active, w_active))
        log_mean = mu_log + sigma_sq / 2.0
        if n_eff <= 1.0:
            point = float(numpy.exp(log_mean))
            return point, point
        se_sq = sigma_sq / n_eff + (sigma_sq**2) / (2.0 * (n_eff - 1.0))
        se = float(numpy.sqrt(se_sq))
        t = float(scipy.stats.t.ppf(1.0 - alpha, df=n_eff - 1.0))
        ci_low = float(numpy.exp(log_mean - t * se))
        ci_high = float(numpy.exp(log_mean + t * se))
        return ci_low, ci_high

    @staticmethod
    def _compute_ci_log_normal_gci(
        rng: numpy.random.Generator,
        y_active: numpy.typing.NDArray[numpy.floating],
        w_active: numpy.typing.NDArray[numpy.floating],
        alpha: float,
    ) -> tuple[float, float]:
        """Generalized CI for the arithmetic mean of a log-normal response."""
        if numpy.any(y_active <= 0.0):
            raise ValueError(
                "log_normal_gci CI requires all response values > 0"
            )
        log_y = numpy.log(y_active)
        w_sum = w_active.sum()
        mu_log = float(numpy.dot(w_active, log_y) / w_sum)
        sigma_sq = float(numpy.dot(w_active, (log_y - mu_log) ** 2) / w_sum)
        if sigma_sq == 0.0:
            point = float(numpy.exp(mu_log))
            return point, point
        n_eff = float(w_sum**2 / numpy.dot(w_active, w_active))
        if n_eff <= 1.0:
            point = float(numpy.exp(mu_log + sigma_sq / 2.0))
            return point, point
        n_draws = 10_000
        z = rng.standard_normal(n_draws)
        u = rng.chisquare(df=n_eff - 1.0, size=n_draws)
        r = mu_log - z * numpy.sqrt(sigma_sq / u) + n_eff * sigma_sq / (2.0 * u)
        ci_low = float(numpy.exp(numpy.quantile(r, alpha)))
        ci_high = float(numpy.exp(numpy.quantile(r, 1.0 - alpha)))
        return ci_low, ci_high

    @staticmethod
    def _compute_ci_normal(
        y_active: numpy.typing.NDArray[numpy.floating],
        w_active: numpy.typing.NDArray[numpy.floating],
        alpha: float,
    ) -> tuple[float, float]:
        """Normal-approximation CI with Kish effective sample size."""
        w_sum = w_active.sum()
        p_hat = float(numpy.dot(w_active, y_active) / w_sum)
        variance = float(numpy.dot(w_active, (y_active - p_hat) ** 2) / w_sum)
        if variance == 0.0:
            return p_hat, p_hat
        n_eff = float(w_sum**2 / numpy.dot(w_active, w_active))
        se = float(numpy.sqrt(variance / n_eff))
        z = float(scipy.stats.norm.ppf(1.0 - alpha))
        ci_low = p_hat - z * se
        ci_high = p_hat + z * se
        return ci_low, ci_high

    @staticmethod
    def _compute_ci_poisson(
        y_active: numpy.typing.NDArray[numpy.floating],
        w_active: numpy.typing.NDArray[numpy.floating],
        alpha: float,
    ) -> tuple[float, float]:
        """Exact Garwood chi-squared CI for a Poisson mean rate."""
        if numpy.any(y_active < 0.0):
            raise ValueError("poisson CI requires all response values >= 0")
        w_sum = float(w_active.sum())
        s = float(numpy.dot(w_active, y_active))
        chi2_high = float(scipy.stats.chi2.ppf(1.0 - alpha, 2.0 * (s + 1.0)))
        ci_high = 0.5 * chi2_high / w_sum
        if s == 0.0:
            return 0.0, ci_high
        chi2_low = float(scipy.stats.chi2.ppf(alpha, 2.0 * s))
        ci_low = 0.5 * chi2_low / w_sum
        return ci_low, ci_high

    @staticmethod
    def _compute_ci_poisson_jeffreys(
        y_active: numpy.typing.NDArray[numpy.floating],
        w_active: numpy.typing.NDArray[numpy.floating],
        alpha: float,
    ) -> tuple[float, float]:
        """Equal-tailed Jeffreys CI for a Poisson mean rate."""
        if numpy.any(y_active < 0.0):
            raise ValueError(
                "poisson_jeffreys CI requires all response values >= 0"
            )
        w_sum = float(w_active.sum())
        s = float(numpy.dot(w_active, y_active))
        shape = s + 0.5
        scale = 1.0 / w_sum
        ci_low = float(scipy.stats.gamma.ppf(alpha, shape, scale=scale))
        ci_high = float(scipy.stats.gamma.ppf(1.0 - alpha, shape, scale=scale))
        return ci_low, ci_high

    @staticmethod
    def _compute_ci_student_t(
        y_active: numpy.typing.NDArray[numpy.floating],
        w_active: numpy.typing.NDArray[numpy.floating],
        alpha: float,
    ) -> tuple[float, float]:
        """Student-t CI with Kish effective sample size, df = n_eff - 1."""
        w_sum = w_active.sum()
        p_hat = float(numpy.dot(w_active, y_active) / w_sum)
        variance = float(numpy.dot(w_active, (y_active - p_hat) ** 2) / w_sum)
        if variance == 0.0:
            return p_hat, p_hat
        n_eff = float(w_sum**2 / numpy.dot(w_active, w_active))
        df = n_eff - 1.0
        if df <= 0.0:
            return p_hat, p_hat
        se = float(numpy.sqrt(variance / n_eff))
        t = float(scipy.stats.t.ppf(1.0 - alpha, df=df))
        ci_low = p_hat - t * se
        ci_high = p_hat + t * se
        return ci_low, ci_high

    def _compute_per_class_ci(
        self,
        y: numpy.typing.NDArray[numpy.floating],
        weights: numpy.typing.NDArray[numpy.floating],
    ) -> tuple[
        None | numpy.typing.NDArray[numpy.floating],
        None | numpy.typing.NDArray[numpy.floating],
    ]:
        """Return (None, None) - regression has no per-class CI."""
        return None, None

    def _compute_class_distribution(
        self,
        y: numpy.typing.NDArray[numpy.floating],
        weights: numpy.typing.NDArray[numpy.floating],
    ) -> None | numpy.typing.NDArray[numpy.floating]:
        """Return None - regression has no class distribution."""
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
        """Return None - regression has no survival function."""
        return None

    def _compute_survival_log_variance(
        self,
        y: numpy.typing.NDArray[numpy.floating],
        weights: numpy.typing.NDArray[numpy.floating],
    ) -> None | numpy.typing.NDArray[numpy.floating]:
        """Return None - regression has no survival log-variance."""
        return None

    def _compute_survival_metrics(
        self,
        y: numpy.typing.NDArray[numpy.floating],
        weights: numpy.typing.NDArray[numpy.floating],
    ) -> None | list[_node.SurvivalMetric]:
        """Return None - regression has no survival metrics."""
        return None

    def _compute_ranking_metrics(
        self,
        y: numpy.typing.NDArray[numpy.floating],
        weights: numpy.typing.NDArray[numpy.floating],
    ) -> None | list[_node.RankingMetric]:
        """Return None - regression has no ranking metrics."""
        return None

    def predict(
        self,
        X: numpy.typing.NDArray[numpy.floating] | pandas.DataFrame,
        offset: None | numpy.typing.NDArray[numpy.floating] = None,
    ) -> numpy.typing.NDArray[numpy.floating]:
        """Predict response values for the given samples.

        Args:
            X: Samples to predict, shape (n_samples, n_features).
            offset: Optional per-sample baseline, shape (n_samples,), in the
                response space. When None and the model was fit with an
                offset, defaults to zero. Ignored when the model was fit
                without offset and offset is None here.

        Returns:
            Predicted values, shape (n_samples,).
        """
        indices = self.predict_index(X)
        node_predictions = numpy.array(
            [node.prediction for node in self.nodes_]
        )
        base = node_predictions[indices]
        if offset is None:
            if not self._fit_with_offset:
                return base
            return base
        offset_new = self._validate_predict_offset(offset, len(base))
        predictions = base + offset_new
        return predictions

    def _validate_predict_offset(
        self,
        offset: numpy.typing.NDArray[numpy.floating],
        n_samples: int,
    ) -> numpy.typing.NDArray[numpy.floating]:
        """Validate the predict-time offset for regression."""
        offset_array = numpy.asarray(offset, dtype=float)
        offset_shape = offset_array.shape
        if offset_array.ndim != 1 or offset_shape[0] != n_samples:
            raise ValueError(
                f"offset must be 1D with length {n_samples},"
                f" got shape {offset_shape}"
            )
        if not numpy.all(numpy.isfinite(offset_array)):
            raise ValueError("offset values must be finite")
        return offset_array
