"""Unit tests for the Metric class hierarchy."""

import unittest

import sigma
import sigma._metric


class TestMedianSurvivalMetric(unittest.TestCase):
    """Tests for the median survival descriptor."""

    __slots__ = ()

    def test_carries_label_and_interval_flag(self):
        """The label and the interval flag come from the constructor."""
        metric = sigma.MedianSurvivalMetric("Median survival", True)
        self.assertEqual(metric.label, "Median survival")
        self.assertTrue(metric.has_ci)

    def test_renders_as_plain_value_where_higher_is_better(self):
        """Median survival formats as a number and prefers larger values."""
        metric = sigma.MedianSurvivalMetric("Median survival", True)
        self.assertEqual(metric.style, "value")
        self.assertEqual(metric.better_is, "higher")


class TestRiskScoreMetric(unittest.TestCase):
    """Tests for the risk score descriptor."""

    __slots__ = ()

    def test_reports_no_interval(self):
        """A risk score never carries confidence bounds."""
        metric = sigma.RiskScoreMetric("Risk score")
        self.assertFalse(metric.has_ci)

    def test_renders_as_plain_value_where_lower_is_better(self):
        """A risk score formats as a number and prefers smaller values."""
        metric = sigma.RiskScoreMetric("Risk score")
        self.assertEqual(metric.style, "value")
        self.assertEqual(metric.better_is, "lower")


class TestSurvivalAtMetric(unittest.TestCase):
    """Tests for the survival-at-a-time descriptor."""

    __slots__ = ()

    def test_carries_reference_time(self):
        """The reference time is available on the descriptor."""
        metric = sigma.SurvivalAtMetric("Survival at 5 years", True, 5.0)
        self.assertEqual(metric.time, 5.0)

    def test_renders_as_probability_where_higher_is_better(self):
        """S(t) formats as a percentage and prefers larger values."""
        metric = sigma.SurvivalAtMetric("Survival at 5 years", True, 5.0)
        self.assertEqual(metric.style, "probability")
        self.assertEqual(metric.better_is, "higher")


class TestRmstMetric(unittest.TestCase):
    """Tests for the restricted mean survival time descriptor."""

    __slots__ = ()

    def test_carries_horizon(self):
        """The restriction horizon is available on the descriptor."""
        metric = sigma.RmstMetric("RMST at 2 years", True, 2.0)
        self.assertEqual(metric.horizon, 2.0)

    def test_renders_as_plain_value_where_higher_is_better(self):
        """RMST formats as a number and prefers larger values."""
        metric = sigma.RmstMetric("RMST at 2 years", True, 2.0)
        self.assertEqual(metric.style, "value")
        self.assertEqual(metric.better_is, "higher")


class TestExpectedRankMetric(unittest.TestCase):
    """Tests for the per-item expected rank descriptor."""

    __slots__ = ()

    def test_renders_as_plain_value_where_lower_is_better(self):
        """An expected rank formats as a number and prefers smaller values."""
        metric = sigma.ExpectedRankMetric("Ebi rank", True)
        self.assertEqual(metric.style, "value")
        self.assertEqual(metric.better_is, "lower")


class TestHierarchy(unittest.TestCase):
    """Tests for the relationship between the descriptors."""

    __slots__ = ()

    def test_every_descriptor_is_a_metric(self):
        """All five descriptors share the Metric base class."""
        descriptors = [
            sigma.MedianSurvivalMetric("Median survival", True),
            sigma.RiskScoreMetric("Risk score"),
            sigma.SurvivalAtMetric("Survival at 5 years", True, 5.0),
            sigma.RmstMetric("RMST at 2 years", True, 2.0),
            sigma.ExpectedRankMetric("Ebi rank", True),
        ]
        for descriptor in descriptors:
            with self.subTest(cls=type(descriptor).__name__):
                self.assertIsInstance(descriptor, sigma.Metric)

    def test_constants_are_shared_not_stored_per_instance(self):
        """style and better_is live on the class, not in the instance state."""
        first = sigma.MedianSurvivalMetric("Median survival", True)
        second = sigma.MedianSurvivalMetric("Median survival", False)
        self.assertIs(type(first).style, type(second).style)
        self.assertNotIn("style", sigma._metric.Metric.__slots__)
        self.assertNotIn("better_is", sigma._metric.Metric.__slots__)


if __name__ == "__main__":
    unittest.main()
