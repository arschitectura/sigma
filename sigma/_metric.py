"""Metric class hierarchy describing the values a fitted tree reports."""

from __future__ import annotations

import abc
import typing


class Metric(abc.ABC):
    """One summary value that every node of a fitted tree reports.

    Attributes:
        label: Display label, e.g. "Median survival" or "Ebi rank".
        has_ci: Whether nodes carry confidence-interval bounds for this
            metric.
        style: Formatting style; "value" for plain numbers, "probability"
            for percentages.
        better_is: "higher" when larger values denote a better outcome,
            "lower" when smaller values do.
    """

    __slots__ = ("__weakref__", "has_ci", "label")

    style: typing.Literal["value", "probability"] = "value"
    better_is: typing.Literal["higher", "lower"] = "higher"

    def __init__(self, label: str, has_ci: bool) -> None:
        self.label = label
        self.has_ci = has_ci

    def _display_label(self, response_name: None | str) -> str:
        """Label shown by the renderers, left unchanged by the response name."""
        label = self.label
        return label


class MedianSurvivalMetric(Metric):
    """Median survival time, bounded by a Brookmeyer-Crowley interval.

    The value is NaN when the node's Kaplan-Meier curve never reaches 0.5.
    """

    __slots__ = ()

    def _display_label(self, response_name: None | str) -> str:
        """Label naming the response instead of survival when one is given."""
        if response_name is None:
            label = self.label
        else:
            label = f"Median {response_name}"
        return label


class RiskScoreMetric(Metric):
    """Nelson-Aalen cumulative hazard summed over the training event times.

    Reports no confidence interval.
    """

    __slots__ = ()

    better_is: typing.Literal["higher", "lower"] = "lower"

    def __init__(self, label: str) -> None:
        super().__init__(label, False)


class SurvivalAtMetric(Metric):
    """Kaplan-Meier survival probability at a reference time, bounded by a
    log-log Greenwood interval.

    Attributes:
        time: Reference time the probability is read at, in the time units
            of the response.
    """

    __slots__ = ("time",)

    style: typing.Literal["value", "probability"] = "probability"

    def __init__(self, label: str, has_ci: bool, time: float) -> None:
        super().__init__(label, has_ci)
        self.time = time


class RmstMetric(Metric):
    """Restricted mean survival time, bounded by an integrated Greenwood
    interval.

    Attributes:
        horizon: Time the mean is restricted to, in the time units of the
            response.
    """

    __slots__ = ("horizon",)

    def __init__(self, label: str, has_ci: bool, horizon: float) -> None:
        super().__init__(label, has_ci)
        self.horizon = horizon


class ExpectedRankMetric(Metric):
    """Plackett-Luce expected rank of one item, lying in [1, n_items]."""

    __slots__ = ()

    better_is: typing.Literal["higher", "lower"] = "lower"
