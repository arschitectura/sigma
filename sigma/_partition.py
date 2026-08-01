"""Partition class hierarchy for routing records at internal tree nodes."""

from __future__ import annotations

import abc
import math
import typing

from . import _extension, _feature

if typing.TYPE_CHECKING:
    import polars

    from . import _node

N = typing.TypeVar("N", bound="_node.Node")


class SplitStatistics:
    """Conditional-inference test backing the split at one internal node.

    Attributes:
        p_value: Significance p-value of the split. When a transmuter is
            used, this is the maximum of the variable selection p-value and
            the transmuter confirmation p-value.
    """

    __slots__ = ("__weakref__", "p_value")

    def __init__(self, p_value: float) -> None:
        self.p_value = p_value


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

    def __init__(self, lower: None | float, upper: None | float) -> None:
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


class MissingValue(BranchCondition):
    """Numeric branch admitting only missing (NaN) values."""

    __slots__ = ()


class Partition(_extension.Extension[N], typing.Generic[N]):
    """Routes records reaching an internal tree node to one of its children.

    Attributes:
        feature: Covariate column this partition splits on.
        statistics: Conditional-inference test backing the split, or None
            when no single test applies to this node.
        children: Child nodes in branch order, one per branch.
    """

    __slots__ = ("children", "feature", "statistics")

    def __init__(
        self,
        feature: _feature.Feature,
        statistics: None | SplitStatistics,
        children: tuple[N, ...],
    ) -> None:
        self.feature = feature
        self.statistics = statistics
        self.children = children

    @property
    def feature_index(self) -> int:
        """Position of the partition covariate in the fit X matrix."""
        index = self.feature.index
        return index

    @property
    @abc.abstractmethod
    def branch_conditions(self) -> tuple[BranchCondition, ...]:
        """Branch conditions in branch order, one per child."""

    @abc.abstractmethod
    def route(self, value: object) -> None | N:
        """Return the child to descend into for a record's feature value,
        or None when the value is not routable from this partition.
        """

    @abc.abstractmethod
    def _polars_condition(self, child: N) -> polars.Expr:
        """Polars predicate admitting the records routed to child."""

    def _child_index(self, child: N) -> int:
        """Index of child among children, matched by identity."""
        for index, candidate in enumerate(self.children):
            if candidate is child:
                return index
        raise ValueError("child is not one of this partition's children")

    def _polars_column(self) -> polars.Expr:
        """Polars column reference for this partition's feature."""
        import polars

        feature = self.feature
        if feature.name is None:
            name = f"X[{feature.index}]"
        else:
            name = feature.name
        column = polars.col(name)
        return column


class NumericalPartition(Partition[N], typing.Generic[N]):
    """Partition on a numeric covariate by ascending threshold cut points.

    An observed record routes to the first interval branch whose inclusive
    upper threshold is not exceeded, and to the last interval branch when it
    exceeds every threshold. A missing (NaN) record routes to the missing
    branch when one was learned, and is not routable otherwise.

    Attributes:
        thresholds: Cut points in strictly ascending order. There is one
            interval child more than there are thresholds.
        nan_child: Index into children of the branch that admits missing
            (NaN) records, or None when missing records are not routable.
    """

    __slots__ = ("nan_child", "thresholds")

    def __init__(
        self,
        feature: _feature.Feature,
        statistics: None | SplitStatistics,
        children: tuple[N, ...],
        thresholds: tuple[int | float, ...],
        nan_child: None | int = None,
    ) -> None:
        super().__init__(feature, statistics, children)
        self.thresholds = thresholds
        self.nan_child = nan_child

    @property
    def branch_conditions(self) -> tuple[BranchCondition, ...]:
        """Numeric intervals in ascending order, then a missing-value branch
        when a dedicated missing child is present.
        """
        lowers: tuple[None | int | float, ...] = (None,) + self.thresholds
        uppers: tuple[None | int | float, ...] = self.thresholds + (None,)
        conditions: list[BranchCondition] = []
        for lower, upper in zip(lowers, uppers):
            conditions.append(NumericInterval(lower, upper))
        if len(self.children) > len(self.thresholds) + 1:
            conditions.append(MissingValue())
        result = tuple(conditions)
        return result

    def route(self, value: object) -> None | N:
        """Return the child for the interval that contains value, the missing
        child for NaN, or None when value is NaN with no missing branch.
        """
        numeric_value = typing.cast(float, value)
        if math.isnan(numeric_value):
            if self.nan_child is None:
                return None
            return self.children[self.nan_child]
        for index, threshold in enumerate(self.thresholds):
            if numeric_value <= threshold:
                return self.children[index]
        return self.children[len(self.thresholds)]

    def _polars_condition(self, child: N) -> polars.Expr:
        """Polars predicate for the interval or missing branch routing to child."""
        index = self._child_index(child)
        column = self._polars_column()
        if index > len(self.thresholds):
            expression = column.is_null()
            return expression
        if index == 0:
            lower = None
        else:
            lower = self.thresholds[index - 1]
        if index == len(self.thresholds):
            upper = None
        else:
            upper = self.thresholds[index]
        match (lower, upper):
            case (None, None):
                expression = column.is_not_null()
            case (None, _):
                expression = column <= upper
            case (_, None):
                expression = column > lower
            case _:
                above = column > lower
                below = column <= upper
                expression = above & below
        if self.nan_child == index:
            missing = column.is_null()
            expression = expression | missing
        return expression


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

    def route(self, value: object) -> None | N:
        """Return the first child for False / 0.0, the second for True / 1.0,
        or None for a missing (NaN) value.

        Raises:
            ValueError: When value is neither NaN, truthy-1, nor falsy-0.
        """
        numeric = float(typing.cast(float, value))
        if math.isnan(numeric):
            return None
        if numeric == 0.0:
            return self.children[0]
        if numeric == 1.0:
            return self.children[1]
        raise ValueError(
            f"boolean feature {self.feature.name!r} got non-boolean"
            f" predict-time value {value!r}"
        )

    def _polars_condition(self, child: N) -> polars.Expr:
        """Polars predicate for the false or true branch routing to child."""
        index = self._child_index(child)
        column = self._polars_column()
        if index == 0:
            expression = ~column
        else:
            expression = column
        return expression


class CategoricalPartition(Partition[N], typing.Generic[N]):
    """Partition on a categorical covariate by category membership.

    Attributes:
        category_groups: Disjoint category sets in branch order, one per
            child. A value outside every set is not routable.
    """

    __slots__ = ("category_groups",)

    feature: _feature.CategoricalFeature

    def __init__(
        self,
        feature: _feature.CategoricalFeature,
        statistics: None | SplitStatistics,
        children: tuple[N, ...],
        category_groups: tuple[frozenset, ...],
    ) -> None:
        super().__init__(feature, statistics, children)
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

    def _polars_condition(self, child: N) -> polars.Expr:
        """Polars predicate for the category subset routing to child."""
        index = self._child_index(child)
        column = self._polars_column()
        group = self.category_groups[index]
        feature = self.feature
        na_code = feature.na_code
        sorted_codes = sorted(group)
        real_codes = [code for code in sorted_codes if code != na_code]
        parts: list[polars.Expr] = []
        if isinstance(feature, _feature.PromotedBooleanFeature):
            for code in real_codes:
                if code == 1.0:
                    parts.append(column)
                else:
                    negated = ~column
                    parts.append(negated)
        elif real_codes:
            labels = feature.category_labels
            values: list[float] | list[str]
            if labels is None:
                values = list(real_codes)
            else:
                values = [labels[code] for code in real_codes]
            if len(values) == 1:
                condition = column == values[0]
            else:
                condition = column.is_in(values)
            parts.append(condition)
        if na_code is not None and na_code in group:
            missing = column.is_null()
            parts.append(missing)
        expression = parts[0]
        for part in parts[1:]:
            expression = expression | part
        return expression
