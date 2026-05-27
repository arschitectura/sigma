"""Survival-analysis primitives for conditional inference trees.

Implements the influence-function and leaf-prediction helpers described in
Hothorn, Hornik, and Zeileis (2006), "Unbiased Recursive Partitioning: A
Conditional Inference Framework," *Journal of Computational and Graphical
Statistics*, 15(3), 651-674, Section 4 ("Censored regression"), and in
Hothorn, Buhlmann, Dudoit, Molinaro, and Van der Laan (2006), "Survival
Ensembles," *Biostatistics*, 7(3), 355-373.
"""

import numpy
import numpy.typing
import scipy.stats


def compute_logrank_scores(
    time: numpy.typing.NDArray[numpy.floating],
    event: numpy.typing.NDArray[numpy.floating],
) -> numpy.typing.NDArray[numpy.floating]:
    """Compute the log-rank score for each subject.

    Args:
        time: Observed times, shape (n,). Must be non-negative.
        event: Event indicators, shape (n,). 1 for an observed event, 0 for
            right-censored.

    Returns:
        Log-rank scores, shape (n,).
    """
    n = len(time)
    if n == 0:
        empty = numpy.empty(0, dtype=float)
        return empty
    order = numpy.argsort(time, kind="stable")
    t_sorted = time[order]
    e_sorted = event[order].astype(float)
    unique_times, first_idx, group_id = numpy.unique(
        t_sorted, return_index=True, return_inverse=True
    )
    n_unique = len(unique_times)
    d_per_time = numpy.bincount(group_id, weights=e_sorted, minlength=n_unique)
    r_per_time = (n - first_idx).astype(float)
    increment = numpy.where(r_per_time > 0, d_per_time / r_per_time, 0.0)
    cum_haz_per_time = numpy.cumsum(increment)
    cum_haz_sorted = cum_haz_per_time[group_id]
    score_sorted = e_sorted - cum_haz_sorted
    score = numpy.empty(n, dtype=float)
    score[order] = score_sorted
    return score


def compute_kaplan_meier(
    time: numpy.typing.NDArray[numpy.floating],
    event: numpy.typing.NDArray[numpy.floating],
    weights: numpy.typing.NDArray[numpy.floating],
) -> tuple[
    numpy.typing.NDArray[numpy.floating], numpy.typing.NDArray[numpy.floating]
]:
    """Estimate the (weighted) Kaplan-Meier survival curve.

    Computes S(t) at each unique observed time among the active samples
    (weight > 0). The returned curve is a right-continuous step function
    that drops only at event times.

    Args:
        time: Observed times, shape (n,). Must be non-negative.
        event: Event indicators, shape (n,).
        weights: Case weights, shape (n,). Samples with zero weight are
            excluded.

    Returns:
        Tuple (times, surv) of strictly-increasing unique active times and
        the survival probabilities at those times. Both arrays have shape
        (n_unique,) and may be empty when no sample is active.
    """
    active = weights > 0
    if not numpy.any(active):
        empty_t = numpy.empty(0, dtype=float)
        empty_s = numpy.empty(0, dtype=float)
        return empty_t, empty_s
    t = time[active].astype(float)
    e = event[active].astype(float)
    w = weights[active].astype(float)
    order = numpy.argsort(t, kind="stable")
    t_sorted = t[order]
    e_sorted = e[order]
    w_sorted = w[order]
    unique_times, first_idx, group_id = numpy.unique(
        t_sorted, return_index=True, return_inverse=True
    )
    n_unique = len(unique_times)
    d_w = numpy.bincount(
        group_id, weights=w_sorted * e_sorted, minlength=n_unique
    )
    cumw = numpy.cumsum(w_sorted)
    total_w = cumw[-1]
    cumw_before = numpy.concatenate([[0.0], cumw[:-1]])[first_idx]
    r_w = total_w - cumw_before
    safe_factor = numpy.where(r_w > 0, 1.0 - d_w / r_w, 1.0)
    surv = numpy.cumprod(safe_factor)
    return unique_times, surv


def compute_median_survival(
    times: numpy.typing.NDArray[numpy.floating],
    surv: numpy.typing.NDArray[numpy.floating],
) -> float:
    """Return the median survival time read from a Kaplan-Meier curve.

    The median is the smallest time t at which the survival curve drops to
    0.5 or below. When the curve never reaches 0.5, +inf is returned.

    Args:
        times: Strictly-increasing observed times, shape (n,).
        surv: Survival probabilities at those times, shape (n,).

    Returns:
        Median survival time, or +inf when the curve never reaches 0.5.
    """
    if len(times) == 0:
        return float("inf")
    below = numpy.where(surv <= 0.5)[0]
    if len(below) == 0:
        return float("inf")
    median = float(times[below[0]])
    return median


def compute_brookmeyer_crowley_ci(
    time: numpy.typing.NDArray[numpy.floating],
    event: numpy.typing.NDArray[numpy.floating],
    weights: numpy.typing.NDArray[numpy.floating],
    alpha: float,
) -> tuple[float, float]:
    """Compute the Brookmeyer-Crowley confidence interval for the median.

    Bounds are NaN when the corresponding confidence band never reaches 0.5.

    Args:
        time: Observed times, shape (n,).
        event: Event indicators, shape (n,).
        weights: Case weights, shape (n,).
        alpha: One-sided tail probability; the returned interval has
            nominal coverage 1 - 2 * alpha.

    Returns:
        Tuple (ci_low, ci_high) of confidence-interval bounds for the
        median. Either bound is NaN when not finite.
    """
    active = weights > 0
    if not numpy.any(active):
        return float("nan"), float("nan")
    unique_times, surv, var_log_s, _, _ = compute_kaplan_meier_with_variance(
        time, event, weights
    )
    s_lower_band, s_upper_band = compute_log_log_ci_band(surv, var_log_s, alpha)
    ci_low = _first_time_at_or_below(unique_times, s_lower_band, 0.5)
    ci_high = _first_time_at_or_below(unique_times, s_upper_band, 0.5)
    return ci_low, ci_high


def compute_kaplan_meier_with_variance(
    time: numpy.typing.NDArray[numpy.floating],
    event: numpy.typing.NDArray[numpy.floating],
    weights: numpy.typing.NDArray[numpy.floating],
) -> tuple[
    numpy.typing.NDArray[numpy.floating],
    numpy.typing.NDArray[numpy.floating],
    numpy.typing.NDArray[numpy.floating],
    numpy.typing.NDArray[numpy.floating],
    numpy.typing.NDArray[numpy.floating],
]:
    """Estimate the weighted Kaplan-Meier curve with Greenwood log-variance.

    Args:
        time: Observed times, shape (n,). Must be non-negative.
        event: Event indicators, shape (n,).
        weights: Case weights, shape (n,). Samples with zero weight are
            excluded.

    Returns:
        Tuple (times, surv, var_log_s, d_w, r_w) of strictly-increasing
        unique active times, the survival probabilities at those times,
        the Greenwood variance of log(S(t)) at those times, the weighted
        event counts, and the weighted at-risk counts. All arrays have
        shape (n_unique,) and may be empty when no sample is active.
    """
    active = weights > 0
    if not numpy.any(active):
        empty = numpy.empty(0, dtype=float)
        return empty, empty, empty, empty, empty
    t = time[active].astype(float)
    e = event[active].astype(float)
    w = weights[active].astype(float)
    order = numpy.argsort(t, kind="stable")
    t_sorted = t[order]
    e_sorted = e[order]
    w_sorted = w[order]
    unique_times, first_idx, group_id = numpy.unique(
        t_sorted, return_index=True, return_inverse=True
    )
    n_unique = len(unique_times)
    d_w = numpy.bincount(
        group_id, weights=w_sorted * e_sorted, minlength=n_unique
    )
    cumw = numpy.cumsum(w_sorted)
    total_w = cumw[-1]
    cumw_before = numpy.concatenate([[0.0], cumw[:-1]])[first_idx]
    r_w = total_w - cumw_before
    safe_factor = numpy.where(r_w > 0, 1.0 - d_w / r_w, 1.0)
    surv = numpy.cumprod(safe_factor)
    denom = r_w * (r_w - d_w)
    safe_denom = numpy.where(denom > 0, denom, 1.0)
    log_var_increment = numpy.where(denom > 0, d_w / safe_denom, 0.0)
    var_log_s = numpy.cumsum(log_var_increment)
    return (
        unique_times.astype(float),
        surv.astype(float),
        var_log_s.astype(float),
        d_w.astype(float),
        r_w.astype(float),
    )


def compute_survival_at(
    times: numpy.typing.NDArray[numpy.floating],
    surv: numpy.typing.NDArray[numpy.floating],
    query: float,
) -> float:
    """Evaluate a Kaplan-Meier step curve at a single time point.

    The curve is right-continuous: at each event time the value drops to
    surv[k]. Below the first event time the value is 1.0.

    Args:
        times: Strictly-increasing observed times, shape (n,).
        surv: Survival probabilities at those times, shape (n,).
        query: Time at which to evaluate S(t).

    Returns:
        S(query), or 1.0 when query is below the first observed time, or
        surv[-1] when query is at or beyond the last observed time.
    """
    if len(times) == 0:
        return 1.0
    index = int(numpy.searchsorted(times, query, side="right")) - 1
    if index < 0:
        return 1.0
    value = float(surv[index])
    return value


def compute_log_log_ci_at(
    times: numpy.typing.NDArray[numpy.floating],
    surv: numpy.typing.NDArray[numpy.floating],
    var_log_s: numpy.typing.NDArray[numpy.floating],
    query: float,
    alpha: float,
) -> tuple[float, float]:
    """Compute the log-log Greenwood pointwise CI for S(query).

    Returns (1.0, 1.0) when the curve is at 1.0 at query, and (S, S) when
    the curve is at 0.0.

    Args:
        times: Strictly-increasing observed times, shape (n,).
        surv: Survival probabilities at those times, shape (n,).
        var_log_s: Greenwood variance of log(S) at those times, shape (n,).
        query: Time at which to evaluate the CI.
        alpha: One-sided tail probability; the returned interval has
            nominal coverage 1 - 2 * alpha.

    Returns:
        Tuple (ci_low, ci_high) of confidence-interval bounds for
        S(query). Bounds are clamped to [0, 1].
    """
    if len(times) == 0:
        return 1.0, 1.0
    index = int(numpy.searchsorted(times, query, side="right")) - 1
    if index < 0:
        return 1.0, 1.0
    s = float(surv[index])
    if s <= 0.0:
        return 0.0, 0.0
    if s >= 1.0:
        return 1.0, 1.0
    v = float(var_log_s[index])
    if v <= 0.0:
        return s, s
    z = float(scipy.stats.norm.ppf(1.0 - alpha))
    log_s = numpy.log(s)
    se = float(numpy.sqrt(v)) / abs(log_s)
    psi_lower = numpy.log(-log_s) - z * se
    psi_upper = numpy.log(-log_s) + z * se
    ci_high = float(numpy.exp(-numpy.exp(psi_lower)))
    ci_low = float(numpy.exp(-numpy.exp(psi_upper)))
    ci_low = max(0.0, min(1.0, ci_low))
    ci_high = max(0.0, min(1.0, ci_high))
    return ci_low, ci_high


def compute_log_log_ci_band(
    surv: numpy.typing.NDArray[numpy.floating],
    var_log_s: numpy.typing.NDArray[numpy.floating],
    alpha: float,
) -> tuple[
    numpy.typing.NDArray[numpy.floating], numpy.typing.NDArray[numpy.floating]
]:
    """Compute the log-log Greenwood pointwise CI band over a KM curve.

    Bounds are clamped to [0, 1].

    Args:
        surv: Survival probabilities at each unique time, shape (n,).
        var_log_s: Greenwood variance of log(S) at those times, shape (n,).
        alpha: One-sided tail probability; the returned band has nominal
            coverage 1 - 2 * alpha at each time.

    Returns:
        Tuple (ci_low, ci_high) of confidence-band arrays for S, both
        shape (n,) with values in [0, 1].
    """
    n = len(surv)
    if n == 0:
        empty_low = numpy.empty(0, dtype=float)
        empty_high = numpy.empty(0, dtype=float)
        return empty_low, empty_high
    surv_arr = numpy.asarray(surv, dtype=float)
    var_arr = numpy.asarray(var_log_s, dtype=float)
    z = float(scipy.stats.norm.ppf(1.0 - alpha))
    ci_low = surv_arr.copy()
    ci_high = surv_arr.copy()
    is_one = surv_arr >= 1.0
    is_zero = surv_arr <= 0.0
    is_strict = (~is_one) & (~is_zero)
    has_var = is_strict & (var_arr > 0.0)
    ci_low[is_one] = 1.0
    ci_high[is_one] = 1.0
    ci_low[is_zero] = 0.0
    ci_high[is_zero] = 0.0
    if numpy.any(has_var):
        s = surv_arr[has_var]
        v = var_arr[has_var]
        log_s = numpy.log(s)
        se = numpy.sqrt(v) / numpy.abs(log_s)
        psi = numpy.log(-log_s)
        psi_lower = psi - z * se
        psi_upper = psi + z * se
        ci_high_strict = numpy.exp(-numpy.exp(psi_lower))
        ci_low_strict = numpy.exp(-numpy.exp(psi_upper))
        ci_low[has_var] = numpy.clip(ci_low_strict, 0.0, 1.0)
        ci_high[has_var] = numpy.clip(ci_high_strict, 0.0, 1.0)
    return ci_low, ci_high


def compute_rmst(
    times: numpy.typing.NDArray[numpy.floating],
    surv: numpy.typing.NDArray[numpy.floating],
    horizon: float,
) -> float:
    """Compute the restricted mean survival time up to a horizon.

    Above the last observed time the curve is treated as constant at
    surv[-1].

    Args:
        times: Strictly-increasing observed times, shape (n,).
        surv: Survival probabilities at those times, shape (n,).
        horizon: Upper integration bound; must be non-negative.

    Returns:
        The restricted mean survival time, in the same units as times.
    """
    if horizon <= 0.0:
        return 0.0
    if len(times) == 0:
        result = float(horizon)
        return result
    truncated = numpy.minimum(times, horizon)
    truncated = truncated[truncated <= horizon]
    bounds = numpy.concatenate([[0.0], truncated, [horizon]])
    bounds = numpy.unique(bounds)
    bounds = bounds[bounds <= horizon]
    if bounds[-1] < horizon:
        bounds = numpy.append(bounds, horizon)
    total = 0.0
    for k in range(len(bounds) - 1):
        left = bounds[k]
        right = bounds[k + 1]
        s_at_left = compute_survival_at(times, surv, left)
        total += s_at_left * (right - left)
    return total


def compute_rmst_ci(
    times: numpy.typing.NDArray[numpy.floating],
    surv: numpy.typing.NDArray[numpy.floating],
    d_w: numpy.typing.NDArray[numpy.floating],
    r_w: numpy.typing.NDArray[numpy.floating],
    horizon: float,
    alpha: float,
) -> tuple[float, float]:
    """Compute the Klein-Moeschberger CI for RMST up to a horizon.

    Bounds are clamped to [0, horizon].

    Args:
        times: Strictly-increasing observed times, shape (n,).
        surv: Survival probabilities at those times, shape (n,).
        d_w: Weighted event counts at those times, shape (n,).
        r_w: Weighted at-risk counts at those times, shape (n,).
        horizon: Upper integration bound.
        alpha: One-sided tail probability; the returned interval has
            nominal coverage 1 - 2 * alpha.

    Returns:
        Tuple (ci_low, ci_high) of CI bounds for RMST(horizon).
    """
    rmst = compute_rmst(times, surv, horizon)
    if horizon <= 0.0 or len(times) == 0:
        return rmst, rmst
    in_window = times <= horizon
    if not numpy.any(in_window):
        return rmst, rmst
    times_inner = times[in_window]
    d_w_inner = d_w[in_window]
    r_w_inner = r_w[in_window]
    integrals = numpy.empty(len(times_inner), dtype=float)
    for k, t_star in enumerate(times_inner):
        integrals[k] = compute_rmst(times, surv, horizon) - compute_rmst(
            times, surv, t_star
        )
    denom = r_w_inner * (r_w_inner - d_w_inner)
    safe = denom > 0
    contribs = numpy.where(
        safe, integrals**2 * d_w_inner / numpy.where(safe, denom, 1.0), 0.0
    )
    variance = float(contribs.sum())
    if variance <= 0.0:
        return rmst, rmst
    z = float(scipy.stats.norm.ppf(1.0 - alpha))
    half = z * float(numpy.sqrt(variance))
    ci_low = max(0.0, rmst - half)
    ci_high = min(float(horizon), rmst + half)
    return ci_low, ci_high


def compute_nelson_aalen(
    time: numpy.typing.NDArray[numpy.floating],
    event: numpy.typing.NDArray[numpy.floating],
    weights: numpy.typing.NDArray[numpy.floating],
) -> tuple[
    numpy.typing.NDArray[numpy.floating], numpy.typing.NDArray[numpy.floating]
]:
    """Estimate the weighted Nelson-Aalen cumulative hazard.

    Args:
        time: Observed times, shape (n,). Must be non-negative.
        event: Event indicators, shape (n,).
        weights: Case weights, shape (n,).

    Returns:
        Tuple (times, cum_haz) of strictly-increasing unique active times
        and the cumulative hazard estimate at those times.
    """
    active = weights > 0
    if not numpy.any(active):
        empty = numpy.empty(0, dtype=float)
        return empty, empty
    t = time[active].astype(float)
    e = event[active].astype(float)
    w = weights[active].astype(float)
    order = numpy.argsort(t, kind="stable")
    t_sorted = t[order]
    e_sorted = e[order]
    w_sorted = w[order]
    unique_times, first_idx, group_id = numpy.unique(
        t_sorted, return_index=True, return_inverse=True
    )
    n_unique = len(unique_times)
    d_w = numpy.bincount(
        group_id, weights=w_sorted * e_sorted, minlength=n_unique
    )
    cumw = numpy.cumsum(w_sorted)
    total_w = cumw[-1]
    cumw_before = numpy.concatenate([[0.0], cumw[:-1]])[first_idx]
    r_w = total_w - cumw_before
    increment = numpy.where(r_w > 0, d_w / r_w, 0.0)
    cum_haz = numpy.cumsum(increment)
    return unique_times, cum_haz


def compute_risk_score(
    time: numpy.typing.NDArray[numpy.floating],
    event: numpy.typing.NDArray[numpy.floating],
    weights: numpy.typing.NDArray[numpy.floating],
    reference_event_times: numpy.typing.NDArray[numpy.floating],
) -> float:
    """Compute the cumulative-hazard sum risk score for a leaf.

    Higher values indicate worse prognosis.

    Args:
        time: Observed times, shape (n,).
        event: Event indicators, shape (n,).
        weights: Case weights, shape (n,).
        reference_event_times: Strictly-increasing reference event times,
            typically the union of unique event times across the
            training data.

    Returns:
        The risk score, a non-negative finite float.
    """
    leaf_times, leaf_cum_haz = compute_nelson_aalen(time, event, weights)
    if len(leaf_times) == 0 or len(reference_event_times) == 0:
        return 0.0
    indices = (
        numpy.searchsorted(leaf_times, reference_event_times, side="right") - 1
    )
    valid = indices >= 0
    contributions = numpy.where(
        valid, leaf_cum_haz[numpy.maximum(indices, 0)], 0.0
    )
    score = float(contributions.sum())
    return score


def _first_time_at_or_below(
    times: numpy.typing.NDArray[numpy.floating],
    values: numpy.typing.NDArray[numpy.floating],
    threshold: float,
) -> float:
    """Return the smallest time at which values drops to threshold or below."""
    finite = numpy.isfinite(values)
    candidates = numpy.where(finite & (values <= threshold))[0]
    if len(candidates) == 0:
        return float("nan")
    result = float(times[candidates[0]])
    return result
