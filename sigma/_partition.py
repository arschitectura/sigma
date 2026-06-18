"""Partition class hierarchy for routing records at internal tree nodes."""

from __future__ import annotations

import abc
import typing

import numpy
import numpy.typing

from . import _extension


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


class Partition(
    _extension.Extension[_extension.N], typing.Generic[_extension.N]
):
    """Routes records reaching an internal tree node to one of its children.

    Attributes:
        feature_index: Integer index of the partition covariate in the fit
            X matrix.
        feature_name: Display name of the partition covariate, or None when
            no name source (constructor feature_names or DataFrame columns)
            was available at fit time.
        p_value: P-value of the partition at this node. When a transmuter
            is used, this is the maximum of the variable selection p-value
            and the transmuter confirmation p-value.
        T: Observed linear test statistic for the split variable.
        mu: Expected value of T under the null of independence.
        Sigma: Covariance of T under the null of independence.
        left: Left child node.
        right: Right child node.
    """

    __slots__ = (
        "feature_index",
        "feature_name",
        "p_value",
        "T",
        "mu",
        "Sigma",
        "left",
        "right",
    )

    def __init__(
        self,
        feature_index: int,
        feature_name: None | str,
        p_value: float,
        T: numpy.typing.NDArray[numpy.floating],
        mu: numpy.typing.NDArray[numpy.floating],
        Sigma: numpy.typing.NDArray[numpy.floating],
        left: _extension.N,
        right: _extension.N,
    ) -> None:
        self.feature_index = feature_index
        self.feature_name = feature_name
        self.p_value = p_value
        self.T = T
        self.mu = mu
        self.Sigma = Sigma
        self.left = left
        self.right = right

    @abc.abstractmethod
    def route(self, value: object) -> None | _extension.N:
        """Return the child to descend into for a record's feature value,
        or None when the value is not routable from this partition.
        """


class NumericalPartition(Partition[_extension.N], typing.Generic[_extension.N]):
    """Binary partition on a numeric covariate by a threshold.

    Records with value <= threshold go left, others go right.

    Attributes:
        threshold: Numeric split point. Stored as a Python int when the
            split covariate is integer-valued, otherwise as a float.
    """

    __slots__ = ("threshold",)

    def __init__(
        self,
        feature_index: int,
        feature_name: None | str,
        p_value: float,
        T: numpy.typing.NDArray[numpy.floating],
        mu: numpy.typing.NDArray[numpy.floating],
        Sigma: numpy.typing.NDArray[numpy.floating],
        left: _extension.N,
        right: _extension.N,
        threshold: int | float,
    ) -> None:
        super().__init__(
            feature_index, feature_name, p_value, T, mu, Sigma, left, right
        )
        self.threshold = threshold

    def route(self, value: object) -> _extension.N:
        """Return left when value <= threshold, otherwise right."""
        threshold = self.threshold
        numeric_value = typing.cast(float, value)
        child = self.left if numeric_value <= threshold else self.right
        return child


class BooleanPartition(Partition[_extension.N], typing.Generic[_extension.N]):
    """Binary partition on a boolean covariate.

    Records with value False (or 0.0) route to the left child; records with
    value True (or 1.0) route to the right child.
    """

    __slots__ = ()

    def route(self, value: object) -> _extension.N:
        """Return left for False / 0.0, right for True / 1.0.

        Raises:
            ValueError: When value is neither truthy-1 nor falsy-0.
        """
        numeric = float(typing.cast(float, value))
        if numeric == 0.0:
            return self.left
        if numeric == 1.0:
            return self.right
        raise ValueError(
            f"boolean feature {self.feature_name!r} got non-boolean"
            f" predict-time value {value!r}"
        )


class CategoricalPartition(
    Partition[_extension.N], typing.Generic[_extension.N]
):
    """Binary partition on a categorical covariate by category membership.

    Attributes:
        left_categories: Categories observed at this node that route to
            the left child.
        right_categories: Categories observed at this node that route to
            the right child.
    """

    __slots__ = ("left_categories", "right_categories")

    def __init__(
        self,
        feature_index: int,
        feature_name: None | str,
        p_value: float,
        T: numpy.typing.NDArray[numpy.floating],
        mu: numpy.typing.NDArray[numpy.floating],
        Sigma: numpy.typing.NDArray[numpy.floating],
        left: _extension.N,
        right: _extension.N,
        left_categories: frozenset,
        right_categories: frozenset,
    ) -> None:
        super().__init__(
            feature_index, feature_name, p_value, T, mu, Sigma, left, right
        )
        self.left_categories = left_categories
        self.right_categories = right_categories

    @property
    def observed_categories(self) -> frozenset:
        """All categories observed at this node during training."""
        cats = self.left_categories | self.right_categories
        return cats

    def route(self, value: object) -> None | _extension.N:
        """Return the child whose category set contains value, or None
        when value belongs to neither the left nor the right category
        set.
        """
        if value in self.left_categories:
            return self.left
        if value in self.right_categories:
            return self.right
        return None
