"""Enumeration types for the sigma package."""

import enum


def _validate_literal_param(
    value: str,
    enum_cls: type[enum.Enum],
    param_name: str,
) -> None:
    """Raise ValueError if value is not a valid member value of enum_cls."""
    valid_values = [member.value for member in enum_cls]
    if value not in valid_values:
        raise ValueError(
            f"{param_name} must be one of {valid_values}; got {value!r}"
        )


class TestStat(enum.Enum):
    """Test statistic form for conditional inference.

    See Hothorn et al. (2006), Section 3.

    MAXIMUM: Max-type statistic over standardized test components. Sensitive
        to a single strong-signal component.
    QUADRATIC: Quadratic-form statistic that pools signal across all
        components. Better for diffuse effects spread over several
        components.
    """

    __slots__ = ()

    MAXIMUM = "maximum"
    QUADRATIC = "quadratic"


class TestType(enum.Enum):
    """Multiplicity adjustment method for variable selection.

    See Hothorn et al. (2006), Section 4.

    BONFERRONI: Classical Bonferroni correction. The simplest and most
        conservative closed-form correction; strictly dominated by Sidak
        under independence or positive dependence of test statistics.
    MONTE_CARLO: Westfall-Young min-P resampling. More powerful than Sidak
        when covariates are correlated. Requires resamples > 0 on the
        estimator. See Westfall and Young (1993), Resampling-Based Multiple
        Testing.
    SIDAK: Closed-form Sidak correction. Uniformly powerful under
        independence or positive dependence of test statistics.
    """

    __slots__ = ()

    BONFERRONI = "bonferroni"
    MONTE_CARLO = "monte_carlo"
    SIDAK = "sidak"


class CovariateType(enum.Enum):
    """Type of a covariate, used to route variable selection and splitting.

    BOOLEAN: Boolean covariate (pandas BooleanDtype or numpy bool).
    CATEGORICAL: Unordered factor. Splits are binary partitions of the
        observed categories.
    INTEGER: Numeric covariate whose observed values are all integers. Split
        thresholds coincide with observed values.
    REAL: Numeric covariate with at least one non-integer observed value.
        Split thresholds fall between adjacent observed values.
    """

    __slots__ = ()

    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"
    INTEGER = "integer"
    REAL = "real"


class Correlation(enum.Enum):
    """Correlation type for association tests.

    NORMAL: Use raw values (Pearson-like). Original behavior from Hothorn et al.
        (2006).
    RANK: Rank-transform continuous covariates and responses before computing
        test statistics (Spearman-like). Robust to outliers and non-normality.
    """

    __slots__ = ()

    NORMAL = "normal"
    RANK = "rank"


class CiMethodRegressionTree(enum.Enum):
    """Confidence interval method for regression tree node predictions.

    BAYESIAN_BOOTSTRAP: Dirichlet-based Bayesian bootstrap of the weighted
        mean. Non-parametric; makes no assumption on the response
        distribution.
    BCA: Bias-corrected and accelerated bootstrap interval for the weighted
        mean. Non-parametric; makes no assumption on the response
        distribution. Frequentist counterpart to BAYESIAN_BOOTSTRAP.
    BETA: Exact Beta interval for a continuous proportional response.
        Requires responses in [0, 1]. Brackets the sample mean.
    EXPONENTIAL: Exact chi-squared interval for the mean of an Exponential
        response. Requires non-negative responses. Brackets the sample mean.
    GAMMA: Exact chi-squared interval for the mean of a Gamma response, with
        the shape estimated from the data. Requires non-negative responses.
        Brackets the sample mean.
    LOG_NORMAL: Cox's interval for the arithmetic mean of a log-normal
        response. Requires all responses strictly positive. Centered on the
        log-normal MLE of the mean, which is not in general equal to the
        sample arithmetic mean.
    LOG_NORMAL_GCI: Generalized confidence interval (Krishnamoorthy &
        Mathew, 2003) for the arithmetic mean of a log-normal response.
        Requires all responses strictly positive. Like LOG_NORMAL but
        built by Monte Carlo from a generalized pivotal quantity, giving
        asymmetric bounds. Non-deterministic.
    NORMAL: Normal-approximation interval on the weighted mean. Unequal
        weights widen the interval through an effective sample size.
    POISSON: Exact Garwood chi-squared interval for a Poisson mean rate.
        Requires non-negative responses. Brackets the sample mean.
    POISSON_JEFFREYS: Equal-tailed Jeffreys interval for a Poisson mean
        rate. Requires non-negative responses. Shorter than POISSON at
        moderate rates; use POISSON instead when guaranteed coverage at
        or above the nominal level matters.
    STUDENT_T: Student-t interval on the weighted mean. Wider than NORMAL
        for small effective sample sizes.
    """

    __slots__ = ()

    BAYESIAN_BOOTSTRAP = "bayesian_bootstrap"
    BCA = "bca"
    BETA = "beta"
    EXPONENTIAL = "exponential"
    GAMMA = "gamma"
    LOG_NORMAL = "log_normal"
    LOG_NORMAL_GCI = "log_normal_gci"
    NORMAL = "normal"
    POISSON = "poisson"
    POISSON_JEFFREYS = "poisson_jeffreys"
    STUDENT_T = "student_t"


class SurvivalMetricKind(enum.Enum):
    """Kind of per-leaf summary computed by SurvivalTree.

    MEDIAN: Median survival time. CI: Brookmeyer-Crowley.
    RISK_SCORE: Nelson-Aalen cumulative hazard summed across unique
        training event times. No CI.
    SURVIVAL: Kaplan-Meier survival probability at a fixed time supplied by
        the user. CI: log-log Greenwood.
    RMST: Restricted mean survival time up to a horizon supplied by the
        user. CI: Klein-Moeschberger integrated Greenwood.
    """

    __slots__ = ()

    MEDIAN = "median"
    RISK_SCORE = "risk_score"
    SURVIVAL = "survival"
    RMST = "rmst"


class CiMethodSurvival(enum.Enum):
    """Confidence interval method for survival node predictions.

    BROOKMEYER_CROWLEY: Brookmeyer-Crowley interval for the median survival
        time. See Brookmeyer and Crowley (1982), "A Confidence Interval for
        the Median Survival Time," Biometrics, 38(1), 29-41.
    """

    __slots__ = ()

    BROOKMEYER_CROWLEY = "brookmeyer_crowley"


class CiMethodClassificationTree(enum.Enum):
    """Confidence interval method for classification tree per-class proportions.

    AGRESTI_COULL: Closed-form adjusted Wald interval. Slightly wider than
        Wilson at small sample sizes; convergent to Wilson at large sample
        sizes.
    CLOPPER_PEARSON: Exact Beta-based interval. Guarantees coverage at least
        ci_coverage; conservative, with intervals wider than Wilson,
        Jeffreys, or Agresti-Coull on average.
    JEFFREYS: Bayesian interval using the Jeffreys non-informative prior.
        Shorter than Clopper-Pearson on average while retaining
        close-to-nominal coverage.
    MID_P_EXACT: Mid-p variant of Clopper-Pearson. Strictly narrower while
        keeping an exact-tail rationale; average coverage close to nominal.
    WILSON: Closed-form Wilson score interval. Cheap to compute and
        accurate for moderate sample sizes.
    WILSON_CC: Continuity-corrected Wilson score interval. Slightly wider
        than Wilson; restores lower-tail coverage at small sample sizes.
    """

    __slots__ = ()

    AGRESTI_COULL = "agresti_coull"
    CLOPPER_PEARSON = "clopper_pearson"
    JEFFREYS = "jeffreys"
    MID_P_EXACT = "mid_p_exact"
    WILSON = "wilson"
    WILSON_CC = "wilson_cc"
