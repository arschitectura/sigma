"""Partition class hierarchy for routing records at internal tree nodes."""

from __future__ import annotations

import abc
import typing

import numpy
import numpy.typing

from . import _extension

if typing.TYPE_CHECKING:
    from . import _node

N = typing.TypeVar("N", bound="_node.Node")


# TODO use it
class UnknownCategoryError(ValueError):
    """Raised when a CategoricalPartition.route receives a category value
    that was not observed at the partition's node during training.

    Attributes:
        feature_name: Display name of the split feature, or None when no
            name source was available at fit time.
        value: The unobserved category value supplied to route().
    """

    __slots__ = ("feature_name", "value", "__weakref__")

    def __init__(self, feature_name: None | str, value: object) -> None:
        feature_label = "<unnamed>" if feature_name is None else feature_name
        message = (
            f"unknown category {value!r} for feature {feature_label}"
            " (not observed at this node during training)"
        )
        super().__init__(message)
        self.feature_name = feature_name
        self.value = value


class SplitStatistics:
    """Conditional-inference test backing a single binary split.

    Attributes:
        p_value: Significance p-value of the split. When a transmuter is
            used, this is the maximum of the variable selection p-value and
            the transmuter confirmation p-value.
        T: Observed linear test statistic for the split variable.
        mu: Expected value of T under the null of independence.
        Sigma: Covariance of T under the null of independence.
    """

    __slots__ = ("p_value", "T", "mu", "Sigma", "__weakref__")

    def __init__(
        self,
        p_value: float,
        T: numpy.typing.NDArray[numpy.floating],
        mu: numpy.typing.NDArray[numpy.floating],
        Sigma: numpy.typing.NDArray[numpy.floating],
    ) -> None:
        self.p_value = p_value
        self.T = T
        self.mu = mu
        self.Sigma = Sigma


class BranchCondition(abc.ABC):
    """Description of the records that a single branch of a partition admits."""

    __slots__ = ("__weakref__",)


class NumericInterval(BranchCondition):
    """Numeric branch range, open on the lower bound and closed on the upper.

    Attributes:
        lower: Exclusive lower bound, or None when the branch extends down
            without bound.
        upper: Inclusive upper bound, or None when the branch extends up
            without bound.
    """

    __slots__ = ("lower", "upper")

    def __init__(
        self, lower: None | int | float, upper: None | int | float
    ) -> None:
        self.lower = lower
        self.upper = upper


class CategorySubset(BranchCondition):
    """Categorical branch admitting a fixed set of category values.

    Attributes:
        categories: Category values routed to this branch.
    """

    __slots__ = ("categories",)

    def __init__(self, categories: frozenset) -> None:
        self.categories = categories


class BooleanValue(BranchCondition):
    """Boolean branch admitting one truth value.

    Attributes:
        value: The truth value routed to this branch.
    """

    __slots__ = ("value",)

    def __init__(self, value: bool) -> None:
        self.value = value


class Partition(_extension.Extension[N], typing.Generic[N]):
    """Routes records reaching an internal tree node to one of its children.

    Attributes:
        feature_index: Integer index of the partition covariate in the fit
            X matrix.
        feature_name: Display name of the partition covariate, or None when
            no name source (constructor feature_names or DataFrame columns)
            was available at fit time.
        statistics: Conditional-inference test backing the split, or None
            when no single test applies to this node.
        children: Child nodes in branch order, one per branch.
    """

    __slots__ = ("feature_index", "feature_name", "statistics", "children")

    def __init__(
        self,
        feature_index: int,
        feature_name: None | str,
        statistics: None | SplitStatistics,
        children: tuple[N, ...],
    ) -> None:
        self.feature_index = feature_index
        self.feature_name = feature_name
        self.statistics = statistics
        self.children = children

    @property
    @abc.abstractmethod
    def branch_conditions(self) -> tuple[BranchCondition, ...]:
        """Branch conditions in branch order, one per child."""

    @abc.abstractmethod
    def route(self, value: object) -> None | N:
        """Return the child to descend into for a record's feature value,
        or None when the value is not routable from this partition.
        """


class NumericalPartition(Partition[N], typing.Generic[N]):
    """Partition on a numeric covariate by ascending threshold cut points.

    A record routes to the first branch whose inclusive upper threshold is
    not exceeded, and to the last branch when it exceeds every threshold.

    Attributes:
        thresholds: Cut points in strictly ascending order. There is one
            more child than there are thresholds.
    """

    __slots__ = ("thresholds",)

    def __init__(
        self,
        feature_index: int,
        feature_name: None | str,
        statistics: None | SplitStatistics,
        children: tuple[N, ...],
        thresholds: tuple[int | float, ...],
    ) -> None:
        super().__init__(feature_index, feature_name, statistics, children)
        self.thresholds = thresholds

    @property
    def branch_conditions(self) -> tuple[BranchCondition, ...]:
        """Numeric intervals in ascending order, one per child."""
        lowers: tuple[None | int | float, ...] = (None,) + self.thresholds
        uppers: tuple[None | int | float, ...] = self.thresholds + (None,)
        conditions: list[BranchCondition] = []
        for lower, upper in zip(lowers, uppers):
            conditions.append(NumericInterval(lower, upper))
        result = tuple(conditions)
        return result

    def route(self, value: object) -> N:
        """Return the child for the interval that contains value."""
        numeric_value = typing.cast(float, value)
        for index, threshold in enumerate(self.thresholds):
            if numeric_value <= threshold:
                return self.children[index]
        return self.children[-1]


class BooleanPartition(Partition[N], typing.Generic[N]):
    """Binary partition on a boolean covariate.

    Records with value False (or 0.0) route to the first child; records with
    value True (or 1.0) route to the second child.
    """

    __slots__ = ()

    @property
    def branch_conditions(self) -> tuple[BranchCondition, ...]:
        """False branch first, then the true branch."""
        result = (BooleanValue(False), BooleanValue(True))
        return result

    def route(self, value: object) -> N:
        """Return the first child for False / 0.0, the second for True / 1.0.

        Raises:
            ValueError: When value is neither truthy-1 nor falsy-0.
        """
        numeric = float(typing.cast(float, value))
        if numeric == 0.0:
            return self.children[0]
        if numeric == 1.0:
            return self.children[1]
        raise ValueError(
            f"boolean feature {self.feature_name!r} got non-boolean"
            f" predict-time value {value!r}"
        )


class CategoricalPartition(Partition[N], typing.Generic[N]):
    """Partition on a categorical covariate by category membership.

    Attributes:
        category_groups: Disjoint category sets in branch order, one per
            child. A value outside every set is not routable.
    """

    __slots__ = ("category_groups",)

    def __init__(
        self,
        feature_index: int,
        feature_name: None | str,
        statistics: None | SplitStatistics,
        children: tuple[N, ...],
        category_groups: tuple[frozenset, ...],
    ) -> None:
        super().__init__(feature_index, feature_name, statistics, children)
        self.category_groups = category_groups

    @property
    def observed_categories(self) -> frozenset:
        """All categories observed at this node during training."""
        observed: frozenset = frozenset()
        for group in self.category_groups:
            observed = observed | group
        return observed

    @property
    def branch_conditions(self) -> tuple[BranchCondition, ...]:
        """Category subsets in branch order, one per child."""
        conditions: list[BranchCondition] = []
        for group in self.category_groups:
            conditions.append(CategorySubset(group))
        result = tuple(conditions)
        return result

    def route(self, value: object) -> None | N:
        """Return the child whose category set contains value, or None when
        value belongs to no branch's category set.
        """
        for group, child in zip(self.category_groups, self.children):
            if value in group:
                return child
        return None
