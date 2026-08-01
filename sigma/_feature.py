"""Feature class hierarchy describing the covariates seen at fit time."""

from __future__ import annotations

import abc


class Feature(abc.ABC):
    """One covariate column of the matrix X seen at fit time.

    Attributes:
        index: Position of the column in the fit X matrix.
        name: Display name of the column, or None when the fit input
            carried no column names.
    """

    __slots__ = ("__weakref__", "index", "name")

    def __init__(self, index: int, name: None | str = None) -> None:
        self.index = index
        self.name = name


class NumericFeature(Feature):
    """Covariate column split by ascending threshold cut points.

    Attributes:
        integer: Whether every value observed at fit time was a whole
            number. Split thresholds then coincide with observed values
            instead of falling between adjacent ones.
    """

    __slots__ = ("integer",)

    def __init__(
        self,
        index: int,
        name: None | str = None,
        integer: bool = False,
    ) -> None:
        super().__init__(index, name)
        self.integer = integer


class BooleanFeature(Feature):
    """Covariate column of truth values, split into a false and a true branch."""

    __slots__ = ()


class CategoricalFeature(Feature):
    """Covariate column of unordered levels, split by category membership.

    Attributes:
        category_labels: Mapping from category code to display label, or
            None when the fit input carried no labels.
        na_code: Category code standing for a missing value, or None when
            the column learned no missing level.
        observed_codes: Category codes observed at fit time, recorded for a
            numeric column declared through categorical_features that
            carried missing values, and None otherwise. A predict-time value
            outside this set is not routable.
    """

    __slots__ = ("category_labels", "na_code", "observed_codes")

    def __init__(
        self,
        index: int,
        name: None | str = None,
        category_labels: None | dict[float, str] = None,
        na_code: None | float = None,
        observed_codes: None | frozenset[float] = None,
    ) -> None:
        super().__init__(index, name)
        self.category_labels = category_labels
        self.na_code = na_code
        self.observed_codes = observed_codes


class PromotedBooleanFeature(CategoricalFeature):
    """Boolean covariate column that carried missing values at fit time and is
    therefore split as a three-level categorical of false, true, and N/A.
    """

    __slots__ = ()
