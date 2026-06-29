"""Unit tests for the survival-analysis helpers."""

import unittest

import numpy
import numpy.testing

import sigma._survival


class TestComputeLogrankScores(unittest.TestCase):
    """Tests for the log-rank (Savage) influence function."""

    __slots__ = ()

    def test_all_events_no_ties(self):
        """Recovers the centred Savage scores when every subject has an event."""
        time = numpy.array([1.0, 2.0, 3.0, 4.0])
        event = numpy.array([1.0, 1.0, 1.0, 1.0])
        scores = sigma._survival.compute_logrank_scores(time, event)
        expected = numpy.array(
            [
                1.0 - 1.0 / 4.0,
                1.0 - (1.0 / 4.0 + 1.0 / 3.0),
                1.0 - (1.0 / 4.0 + 1.0 / 3.0 + 1.0 / 2.0),
                1.0 - (1.0 / 4.0 + 1.0 / 3.0 + 1.0 / 2.0 + 1.0),
            ]
        )
        numpy.testing.assert_allclose(scores, expected)

    def test_with_censoring_subtracts_only_cumulative_hazard(self):
        """Censored subject score equals minus its cumulative hazard at T_i."""
        time = numpy.array([1.0, 2.0, 3.0, 4.0])
        event = numpy.array([1.0, 0.0, 1.0, 1.0])
        scores = sigma._survival.compute_logrank_scores(time, event)
        expected = numpy.array(
            [
                1.0 - 1.0 / 4.0,
                0.0 - 1.0 / 4.0,
                1.0 - (1.0 / 4.0 + 1.0 / 2.0),
                1.0 - (1.0 / 4.0 + 1.0 / 2.0 + 1.0),
            ]
        )
        numpy.testing.assert_allclose(scores, expected)

    def test_scores_sum_to_zero(self):
        """The vector of log-rank scores has mean zero by construction."""
        rng = numpy.random.RandomState(0)
        time = rng.exponential(size=50)
        event = rng.randint(0, 2, size=50).astype(float)
        scores = sigma._survival.compute_logrank_scores(time, event)
        numpy.testing.assert_allclose(scores.sum(), 0.0, atol=1e-12)

    def test_handles_ties(self):
        """Subjects sharing a time receive the same cumulative-hazard offset."""
        time = numpy.array([1.0, 1.0, 2.0, 3.0])
        event = numpy.array([1.0, 1.0, 1.0, 1.0])
        scores = sigma._survival.compute_logrank_scores(time, event)
        increment_t1 = 2.0 / 4.0
        increment_t2 = 1.0 / 2.0
        increment_t3 = 1.0 / 1.0
        expected = numpy.array(
            [
                1.0 - increment_t1,
                1.0 - increment_t1,
                1.0 - (increment_t1 + increment_t2),
                1.0 - (increment_t1 + increment_t2 + increment_t3),
            ]
        )
        numpy.testing.assert_allclose(scores, expected)

    def test_empty_input(self):
        """Returns an empty array when there are no observations."""
        scores = sigma._survival.compute_logrank_scores(
            numpy.empty(0), numpy.empty(0)
        )
        self.assertEqual(scores.shape, (0,))


class TestComputeKaplanMeier(unittest.TestCase):
    """Tests for the weighted Kaplan-Meier estimator."""

    __slots__ = ()

    def test_uncensored_uniform_weights(self):
        """Recovers the empirical survival function when nothing is censored."""
        time = numpy.array([1.0, 2.0, 3.0, 4.0])
        event = numpy.array([1.0, 1.0, 1.0, 1.0])
        weights = numpy.ones(4)
        times, surv = sigma._survival.compute_kaplan_meier(time, event, weights)
        numpy.testing.assert_allclose(times, numpy.array([1.0, 2.0, 3.0, 4.0]))
        numpy.testing.assert_allclose(surv, numpy.array([0.75, 0.5, 0.25, 0.0]))

    def test_with_censoring(self):
        """Censored observations leave the at-risk set without dropping S(t)."""
        time = numpy.array([1.0, 2.0, 3.0, 4.0])
        event = numpy.array([1.0, 0.0, 1.0, 1.0])
        weights = numpy.ones(4)
        times, surv = sigma._survival.compute_kaplan_meier(time, event, weights)
        numpy.testing.assert_allclose(times, numpy.array([1.0, 2.0, 3.0, 4.0]))
        expected = numpy.array(
            [
                1.0 - 1.0 / 4.0,
                (1.0 - 1.0 / 4.0),
                (1.0 - 1.0 / 4.0) * (1.0 - 1.0 / 2.0),
                (1.0 - 1.0 / 4.0) * (1.0 - 1.0 / 2.0) * (1.0 - 1.0 / 1.0),
            ]
        )
        numpy.testing.assert_allclose(surv, expected)

    def test_survival_is_non_increasing(self):
        """The KM curve is monotone non-increasing on a random sample."""
        rng = numpy.random.RandomState(0)
        time = rng.exponential(size=100)
        event = rng.randint(0, 2, size=100).astype(float)
        weights = numpy.ones(100)
        _, surv = sigma._survival.compute_kaplan_meier(time, event, weights)
        differences = numpy.diff(surv)
        self.assertTrue(numpy.all(differences <= 1e-12))

    def test_zero_weights_excluded(self):
        """Zero-weighted samples are dropped from the curve estimation."""
        time = numpy.array([1.0, 2.0, 3.0, 4.0])
        event = numpy.array([1.0, 1.0, 1.0, 1.0])
        weights = numpy.array([1.0, 0.0, 1.0, 1.0])
        times, surv = sigma._survival.compute_kaplan_meier(time, event, weights)
        numpy.testing.assert_allclose(times, numpy.array([1.0, 3.0, 4.0]))
        numpy.testing.assert_allclose(
            surv, numpy.array([1.0 - 1.0 / 3.0, (2.0 / 3.0) * 0.5, 0.0])
        )

    def test_no_active_samples_returns_empty(self):
        """Returns empty arrays when no sample has positive weight."""
        time = numpy.array([1.0, 2.0])
        event = numpy.array([1.0, 0.0])
        weights = numpy.zeros(2)
        times, surv = sigma._survival.compute_kaplan_meier(time, event, weights)
        self.assertEqual(times.shape, (0,))
        self.assertEqual(surv.shape, (0,))


class TestComputeMedianSurvival(unittest.TestCase):
    """Tests for reading the median off a Kaplan-Meier curve."""

    __slots__ = ()

    def test_median_below_half(self):
        """Returns the smallest time at which S(t) reaches 0.5 from above."""
        times = numpy.array([1.0, 2.0, 3.0, 4.0])
        surv = numpy.array([0.75, 0.5, 0.25, 0.0])
        self.assertEqual(
            sigma._survival.compute_median_survival(times, surv), 2.0
        )

    def test_curve_never_crosses_half_returns_nan(self):
        """Returns NaN when the survival curve never reaches 0.5."""
        times = numpy.array([1.0, 2.0, 3.0])
        surv = numpy.array([0.9, 0.8, 0.7])
        self.assertTrue(
            numpy.isnan(sigma._survival.compute_median_survival(times, surv))
        )

    def test_empty_returns_nan(self):
        """Returns NaN for an empty curve."""
        self.assertTrue(
            numpy.isnan(
                sigma._survival.compute_median_survival(
                    numpy.empty(0), numpy.empty(0)
                )
            )
        )


class TestComputeBrookmeyerCrowleyCi(unittest.TestCase):
    """Tests for the Brookmeyer-Crowley CI of the median."""

    __slots__ = ()

    def test_brackets_the_point_estimate(self):
        """The CI brackets the median when both bounds are finite."""
        rng = numpy.random.RandomState(0)
        time = rng.exponential(scale=2.0, size=200)
        event = (rng.uniform(size=200) < 0.8).astype(float)
        weights = numpy.ones(200)
        times, surv = sigma._survival.compute_kaplan_meier(time, event, weights)
        median = sigma._survival.compute_median_survival(times, surv)
        ci_low, ci_high = sigma._survival.compute_brookmeyer_crowley_ci(
            time, event, weights, alpha=0.025
        )
        self.assertTrue(numpy.isfinite(ci_low))
        self.assertTrue(numpy.isfinite(ci_high))
        self.assertLessEqual(ci_low, median)
        self.assertLessEqual(median, ci_high)

    def test_no_active_returns_nan(self):
        """Returns (NaN, NaN) when there are no active samples."""
        ci_low, ci_high = sigma._survival.compute_brookmeyer_crowley_ci(
            numpy.array([1.0]), numpy.array([1.0]), numpy.array([0.0]), 0.025
        )
        self.assertTrue(numpy.isnan(ci_low))
        self.assertTrue(numpy.isnan(ci_high))


class TestComputeSurvivalAt(unittest.TestCase):
    """Tests for evaluating a Kaplan-Meier curve at a single time."""

    __slots__ = ()

    def test_below_first_time_returns_one(self):
        """Returns S(t) = 1 below the first observed time."""
        times = numpy.array([1.0, 2.0, 3.0])
        surv = numpy.array([0.8, 0.6, 0.4])
        self.assertEqual(
            sigma._survival.compute_survival_at(times, surv, 0.5), 1.0
        )

    def test_at_event_time_returns_dropped_value(self):
        """At an event time, returns the post-drop survival."""
        times = numpy.array([1.0, 2.0, 3.0])
        surv = numpy.array([0.8, 0.6, 0.4])
        self.assertEqual(
            sigma._survival.compute_survival_at(times, surv, 1.0), 0.8
        )

    def test_beyond_last_time_returns_last_value(self):
        """Above the last observed time, returns the final S(t)."""
        times = numpy.array([1.0, 2.0, 3.0])
        surv = numpy.array([0.8, 0.6, 0.4])
        self.assertEqual(
            sigma._survival.compute_survival_at(times, surv, 10.0), 0.4
        )

    def test_empty_curve_returns_one(self):
        """An empty curve evaluates to 1.0 everywhere."""
        self.assertEqual(
            sigma._survival.compute_survival_at(
                numpy.empty(0), numpy.empty(0), 5.0
            ),
            1.0,
        )


class TestComputeLogLogCiAt(unittest.TestCase):
    """Tests for the log-log Greenwood pointwise CI of S(t)."""

    __slots__ = ()

    def test_brackets_point_estimate(self):
        """The log-log CI brackets S(query) when the curve has uncertainty."""
        rng = numpy.random.RandomState(0)
        time = rng.exponential(scale=2.0, size=100)
        event = (rng.uniform(size=100) < 0.7).astype(float)
        weights = numpy.ones(100)
        times, surv, var_log_s, _, _ = (
            sigma._survival.compute_kaplan_meier_with_variance(
                time, event, weights
            )
        )
        query = 1.0
        s = sigma._survival.compute_survival_at(times, surv, query)
        ci_low, ci_high = sigma._survival.compute_log_log_ci_at(
            times, surv, var_log_s, query, 0.025
        )
        self.assertLessEqual(ci_low, s)
        self.assertLessEqual(s, ci_high)
        self.assertGreaterEqual(ci_low, 0.0)
        self.assertLessEqual(ci_high, 1.0)

    def test_below_first_time_returns_one_one(self):
        """The CI is (1, 1) below the first observed time."""
        times = numpy.array([1.0])
        surv = numpy.array([0.5])
        var_log_s = numpy.array([1.0])
        ci = sigma._survival.compute_log_log_ci_at(
            times, surv, var_log_s, 0.5, 0.025
        )
        self.assertEqual(ci, (1.0, 1.0))


class TestComputeLogLogCiBand(unittest.TestCase):
    """Tests for the vectorized log-log Greenwood CI band."""

    __slots__ = ()

    def test_brackets_curve(self):
        """The band brackets S(t) at every time when the curve has uncertainty."""
        rng = numpy.random.RandomState(0)
        time = rng.exponential(scale=2.0, size=100)
        event = (rng.uniform(size=100) < 0.7).astype(float)
        weights = numpy.ones(100)
        _, surv, var_log_s, _, _ = (
            sigma._survival.compute_kaplan_meier_with_variance(
                time, event, weights
            )
        )
        ci_low, ci_high = sigma._survival.compute_log_log_ci_band(
            surv, var_log_s, 0.025
        )
        self.assertEqual(ci_low.shape, surv.shape)
        self.assertEqual(ci_high.shape, surv.shape)
        for index in range(len(surv)):
            self.assertLessEqual(ci_low[index], surv[index])
            self.assertLessEqual(surv[index], ci_high[index])
            self.assertGreaterEqual(ci_low[index], 0.0)
            self.assertLessEqual(ci_high[index], 1.0)

    def test_at_one_returns_one_one(self):
        """Where S(t) == 1, the band collapses to (1, 1)."""
        surv = numpy.array([1.0, 1.0])
        var_log_s = numpy.array([0.0, 0.5])
        ci_low, ci_high = sigma._survival.compute_log_log_ci_band(
            surv, var_log_s, 0.025
        )
        numpy.testing.assert_array_equal(ci_low, numpy.array([1.0, 1.0]))
        numpy.testing.assert_array_equal(ci_high, numpy.array([1.0, 1.0]))

    def test_at_zero_returns_zero_zero(self):
        """Where S(t) == 0, the band collapses to (0, 0)."""
        surv = numpy.array([0.0, 0.0])
        var_log_s = numpy.array([1.0, 0.5])
        ci_low, ci_high = sigma._survival.compute_log_log_ci_band(
            surv, var_log_s, 0.025
        )
        numpy.testing.assert_array_equal(ci_low, numpy.array([0.0, 0.0]))
        numpy.testing.assert_array_equal(ci_high, numpy.array([0.0, 0.0]))

    def test_zero_variance_returns_point(self):
        """Where var_log_s == 0 with 0 < S < 1, the band collapses to (S, S)."""
        surv = numpy.array([0.7])
        var_log_s = numpy.array([0.0])
        ci_low, ci_high = sigma._survival.compute_log_log_ci_band(
            surv, var_log_s, 0.025
        )
        numpy.testing.assert_array_equal(ci_low, surv)
        numpy.testing.assert_array_equal(ci_high, surv)

    def test_matches_pointwise(self):
        """Per index, the band agrees with compute_log_log_ci_at at that time."""
        rng = numpy.random.RandomState(1)
        time = rng.exponential(scale=2.0, size=80)
        event = (rng.uniform(size=80) < 0.6).astype(float)
        weights = numpy.ones(80)
        times, surv, var_log_s, _, _ = (
            sigma._survival.compute_kaplan_meier_with_variance(
                time, event, weights
            )
        )
        ci_low, ci_high = sigma._survival.compute_log_log_ci_band(
            surv, var_log_s, 0.025
        )
        for index in range(len(times)):
            point_low, point_high = sigma._survival.compute_log_log_ci_at(
                times, surv, var_log_s, float(times[index]), 0.025
            )
            self.assertAlmostEqual(float(ci_low[index]), point_low)
            self.assertAlmostEqual(float(ci_high[index]), point_high)

    def test_empty_inputs(self):
        """Empty inputs return empty arrays."""
        empty = numpy.empty(0, dtype=float)
        ci_low, ci_high = sigma._survival.compute_log_log_ci_band(
            empty, empty, 0.025
        )
        self.assertEqual(ci_low.shape, (0,))
        self.assertEqual(ci_high.shape, (0,))


class TestComputeRmst(unittest.TestCase):
    """Tests for restricted mean survival time."""

    __slots__ = ()

    def test_full_step_curve_integral(self):
        """Computes the area under a simple step curve."""
        times = numpy.array([1.0, 2.0, 3.0, 4.0])
        surv = numpy.array([0.75, 0.5, 0.25, 0.0])
        rmst = sigma._survival.compute_rmst(times, surv, 4.0)
        expected = 1.0 + 0.75 + 0.5 + 0.25
        self.assertAlmostEqual(rmst, expected)

    def test_horizon_below_first_event(self):
        """When the horizon is below the first event time, RMST = horizon."""
        times = numpy.array([5.0, 6.0])
        surv = numpy.array([0.5, 0.25])
        rmst = sigma._survival.compute_rmst(times, surv, 2.0)
        self.assertEqual(rmst, 2.0)

    def test_zero_horizon_returns_zero(self):
        """RMST(0) = 0."""
        times = numpy.array([1.0, 2.0])
        surv = numpy.array([0.5, 0.0])
        self.assertEqual(sigma._survival.compute_rmst(times, surv, 0.0), 0.0)


class TestComputeRmstCi(unittest.TestCase):
    """Tests for the integrated-Greenwood CI of RMST."""

    __slots__ = ()

    def test_brackets_point_estimate(self):
        """The CI brackets RMST when the leaf has uncertainty."""
        rng = numpy.random.RandomState(0)
        time = rng.exponential(scale=2.0, size=100)
        event = (rng.uniform(size=100) < 0.7).astype(float)
        weights = numpy.ones(100)
        times, surv, _, d_w, r_w = (
            sigma._survival.compute_kaplan_meier_with_variance(
                time, event, weights
            )
        )
        rmst = sigma._survival.compute_rmst(times, surv, 5.0)
        ci_low, ci_high = sigma._survival.compute_rmst_ci(
            times, surv, d_w, r_w, 5.0, 0.025
        )
        self.assertLessEqual(ci_low, rmst)
        self.assertLessEqual(rmst, ci_high)
        self.assertGreaterEqual(ci_low, 0.0)
        self.assertLessEqual(ci_high, 5.0)


class TestComputeRiskScore(unittest.TestCase):
    """Tests for the cumulative-hazard sum risk score."""

    __slots__ = ()

    def test_higher_hazard_yields_higher_score(self):
        """A leaf with more events has a strictly higher risk score."""
        time_low = numpy.array([1.0, 2.0, 3.0, 4.0, 5.0])
        event_low = numpy.array([0.0, 0.0, 0.0, 1.0, 0.0])
        time_high = numpy.array([1.0, 2.0, 3.0, 4.0, 5.0])
        event_high = numpy.array([1.0, 1.0, 1.0, 1.0, 0.0])
        ref = numpy.array([1.0, 2.0, 3.0, 4.0])
        score_low = sigma._survival.compute_risk_score(
            time_low, event_low, numpy.ones(5), ref
        )
        score_high = sigma._survival.compute_risk_score(
            time_high, event_high, numpy.ones(5), ref
        )
        self.assertGreater(score_high, score_low)

    def test_no_active_returns_zero(self):
        """A leaf with no active samples has a zero risk score."""
        score = sigma._survival.compute_risk_score(
            numpy.array([1.0]),
            numpy.array([1.0]),
            numpy.array([0.0]),
            numpy.array([1.0]),
        )
        self.assertEqual(score, 0.0)


if __name__ == "__main__":
    unittest.main()
