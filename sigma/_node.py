"""Node class hierarchy and traversal for fitted conditional inference trees."""

from __future__ import annotations

import abc
import typing

import numpy
import numpy.typing
import typing_extensions

from . import _extension, _partition

if typing.TYPE_CHECKING:
    import polars


class Node(abc.ABC):
    """Abstract base for a node in a fitted conditional inference tree.

    Attributes:
        depth: Depth in the tree (root = 0).
        n_samples: Number of active samples reaching this node.
        share: Fraction of the total training samples that reached this
            node.
        decoration: Optional decoration produced by the tree decorator
            callable, or None when no decorator is set.
        extension: Partition on internal nodes, Leaf on leaves. Set by
            Tree.fit; defaults to a Leaf in unfitted nodes.
        node_id: Zero-based index of this node in Tree.nodes_. Set by
            Tree.fit; defaults to 0 in unfitted nodes.
        parent: Parent node, or None on the root. Set by Tree.fit;
            defaults to None in unfitted nodes.
    """

    __slots__ = (
        "__weakref__",
        "decoration",
        "depth",
        "extension",
        "n_samples",
        "node_id",
        "parent",
        "share",
    )

    extension: _extension.Extension[typing_extensions.Self]
    node_id: int
    parent: None | Node

    def __init__(
        self,
        depth: int,
        n_samples: int,
        share: float,
        decoration: None | object,
    ) -> None:
        self.depth = depth
        self.n_samples = n_samples
        self.share = share
        self.decoration = decoration
        self.extension = _extension.Leaf()
        self.node_id = 0
        self.parent = None

    def traverse(self, x: numpy.typing.NDArray) -> typing_extensions.Self:
        """Walk a single sample down the tree to its deepest reached node.

        Args:
            x: Feature vector for a single sample, shape (m,).

        Returns:
            The leaf node reached when traversal completes; otherwise the
            internal node whose partition did not route the value.
        """
        path = self.traverse_path(x)
        node = path[-1]
        return node

    def traverse_path(
        self, x: numpy.typing.NDArray
    ) -> list[typing_extensions.Self]:
        """Walk a single sample down the tree, listing the nodes it visits.

        Args:
            x: Feature vector for a single sample, shape (m,).

        Returns:
            The visited nodes ordered from the root to the deepest reached
            node. The final entry is the leaf reached when traversal
            completes, or the internal node whose partition did not route
            the value.
        """
        path: list[typing_extensions.Self] = []
        node = self
        while True:
            path.append(node)  # ty: ignore[invalid-argument-type]
            match node.extension:  # ty: ignore[unresolved-attribute]
                case _partition.Partition() as partition:
                    value = x[partition.feature_index]
                    child = partition.route(value)
                    if child is None:
                        break
                    node = child
                case _:
                    break
        return path

    def leaves(self) -> list[typing_extensions.Self]:
        """Return all leaf nodes in this subtree, in left-to-right order."""
        match self.extension:
            case _partition.Partition() as partition:
                result: list[typing_extensions.Self] = []
                for child in partition.children:
                    result.extend(child.leaves())  # ty: ignore[unresolved-attribute]
            case _:
                result = [self]
        return result

    def polars_expression(self) -> polars.Expr:
        """Build the polars filter expression selecting this node's rows.

        The expression AND-combines the branch conditions leading from the
        root to this node, so filtering the fit DataFrame with it keeps
        exactly the rows that reach this node or one of its descendants.
        The root node returns a literal true expression. Conditions
        reference columns by their fit-time feature names (X[i] when the
        fit input carried no names), categorical conditions compare
        against the fit-time category labels when the tree captured them,
        and missing-value branches test for null.

        Returns:
            A polars expression selecting this node's observations.

        Raises:
            ImportError: If polars is not installed.
        """
        import polars

        parent = self.parent
        if parent is None:
            expression = polars.lit(True)
            return expression
        parent_expression = parent.polars_expression()
        partition = typing.cast(_partition.Partition, parent.extension)
        condition = partition._polars_condition(self)
        expression = parent_expression & condition
        return expression

    @abc.abstractmethod
    def leaf_sort_key(self) -> tuple[float, ...]:
        """Sort key for ordering leaves of this task type."""

    @property
    @abc.abstractmethod
    def prediction(self) -> float:
        """The node's point prediction for its task."""


class RegressionNode(Node):
    """Node of a fitted RegressionTree.

    Attributes:
        prediction: Weighted mean of the response in this node's active
            samples.
        ci_low: Lower bound of the confidence interval for the prediction,
            or None when CI is disabled.
        ci_high: Upper bound of the confidence interval for the prediction,
            or None when CI is disabled.
        response_samples: Per-leaf array of response samples. Empty on
            internal nodes and on leaves when response_sample_size=0.
    """

    __slots__ = ("_prediction", "ci_high", "ci_low", "response_samples")

    def __init__(
        self,
        depth: int,
        n_samples: int,
        share: float,
        decoration: None | object,
        prediction: float,
        ci_low: None | float,
        ci_high: None | float,
        response_samples: numpy.typing.NDArray[numpy.floating],
    ) -> None:
        super().__init__(depth, n_samples, share, decoration)
        self._prediction = prediction
        self.ci_low = ci_low
        self.ci_high = ci_high
        self.response_samples = response_samples

    @property
    def prediction(self) -> float:
        """Weighted mean of the response in this node's active samples."""
        return self._prediction

    def leaf_sort_key(self) -> tuple[float, ...]:
        """Sort key: ascending by predicted mean."""
        key = (self.prediction,)
        return key


class ClassificationNode(Node):
    """Node of a fitted ClassificationTree.

    Attributes:
        prediction: Index of the majority class within the estimator's
            classes_ array.
        class_distribution: Class probability vector for this node, shape
            (n_classes,).
        ci_low: Lower CI bounds per class, shape (n_classes,), or None
            when CI is disabled.
        ci_high: Upper CI bounds per class, shape (n_classes,), or None
            when CI is disabled.
        mean_offset_proba: Weighted mean of the fit-time offset over
            this node's active samples, shape (n_classes,). None unless
            fit was called with an offset.
    """

    __slots__ = (
        "_prediction",
        "ci_high",
        "ci_low",
        "class_distribution",
        "mean_offset_proba",
    )

    def __init__(
        self,
        depth: int,
        n_samples: int,
        share: float,
        decoration: None | object,
        prediction: int,
        class_distribution: numpy.typing.NDArray[numpy.floating],
        ci_low: None | numpy.typing.NDArray[numpy.floating],
        ci_high: None | numpy.typing.NDArray[numpy.floating],
        mean_offset_proba: None | numpy.typing.NDArray[numpy.floating],
    ) -> None:
        super().__init__(depth, n_samples, share, decoration)
        self._prediction = prediction
        self.class_distribution = class_distribution
        self.ci_low = ci_low
        self.ci_high = ci_high
        self.mean_offset_proba = mean_offset_proba

    @property
    def prediction(self) -> int:
        """Index of the majority class within the estimator's classes_ array."""
        return self._prediction

    def leaf_sort_key(self) -> tuple[float, ...]:
        """Sort key: descending by class distribution tuple."""
        key = tuple(-p for p in self.class_distribution)
        return key


class SurvivalMetric:
    """Single per-node summary for a SurvivalTree estimator.

    Attributes:
        label: Display label, e.g., "Median survival" or "Survival at
            5 years".
        value: Numeric value of the metric. NaN and +/- inf are allowed.
        ci_low: Lower confidence-interval bound, or None when no CI is
            defined for this metric.
        ci_high: Upper confidence-interval bound, or None when no CI is
            defined for this metric.
        style: Formatting style; "value" for floats, "probability" for
            percentages.
        better_is: "higher" when larger values indicate a better
            prognosis (median, RMST, S(t)); "lower" when smaller values
            do (risk score).
    """

    __slots__ = (
        "__weakref__",
        "better_is",
        "ci_high",
        "ci_low",
        "label",
        "style",
        "value",
    )

    def __init__(
        self,
        label: str,
        value: float,
        ci_low: None | float,
        ci_high: None | float,
        style: typing.Literal["value", "probability"],
        better_is: typing.Literal["higher", "lower"],
    ) -> None:
        self.label = label
        self.value = value
        self.ci_low = ci_low
        self.ci_high = ci_high
        self.style = style
        self.better_is = better_is


class SurvivalNode(Node):
    """Node of a fitted SurvivalTree.

    Attributes:
        survival_function: Pair (times, surv) describing the
            Kaplan-Meier estimate of S(t) at this node.
        survival_log_variance: Greenwood variance of log S(t) at the same
            times as survival_function, shape (n,). Empty on internal
            nodes.
        metrics: Non-empty ordered list of per-node summary metrics.
    """

    __slots__ = ("metrics", "survival_function", "survival_log_variance")

    def __init__(
        self,
        depth: int,
        n_samples: int,
        share: float,
        decoration: None | object,
        survival_function: tuple[
            numpy.typing.NDArray[numpy.floating],
            numpy.typing.NDArray[numpy.floating],
        ],
        survival_log_variance: numpy.typing.NDArray[numpy.floating],
        metrics: list[SurvivalMetric],
    ) -> None:
        super().__init__(depth, n_samples, share, decoration)
        self.survival_function = survival_function
        self.survival_log_variance = survival_log_variance
        self.metrics = metrics

    @property
    def prediction(self) -> float:
        """First configured metric's value (typically median survival)."""
        value = self.metrics[0].value
        return value

    @property
    def ci_low(self) -> None | float:
        """Lower CI bound of the first configured metric."""
        bound = self.metrics[0].ci_low
        return bound

    @property
    def ci_high(self) -> None | float:
        """Upper CI bound of the first configured metric."""
        bound = self.metrics[0].ci_high
        return bound

    def leaf_sort_key(self) -> tuple[float, ...]:
        """Sort key: lexicographic on metrics, worst-prognosis-first."""
        components: list[float] = []
        for metric in self.metrics:
            sign = 1.0 if metric.better_is == "higher" else -1.0
            if numpy.isnan(metric.value):
                component = float("inf")
            else:
                component = sign * metric.value
            components.append(component)
        key = tuple(components)
        return key


class RankingMetric:
    """Single per-item summary for a RankingTree estimator.

    Attributes:
        label: Display label of the item being ranked.
        value: Plackett-Luce expected rank of the item, in [1, n_items].
        ci_low: Lower confidence-interval bound on the expected rank, or
            None when no CI is defined.
        ci_high: Upper confidence-interval bound on the expected rank,
            or None when no CI is defined.
        style: Formatting style; always "value" for ranking metrics.
        better_is: Prognostic direction; always "lower" because rank 1
            is the most-preferred position.
    """

    __slots__ = (
        "__weakref__",
        "better_is",
        "ci_high",
        "ci_low",
        "label",
        "style",
        "value",
    )

    def __init__(
        self,
        label: str,
        value: float,
        ci_low: None | float,
        ci_high: None | float,
    ) -> None:
        self.label = label
        self.value = value
        self.ci_low = ci_low
        self.ci_high = ci_high
        self.style: typing.Literal["value", "probability"] = "value"
        self.better_is: typing.Literal["higher", "lower"] = "lower"


class RankingNode(Node):
    """Node of a fitted RankingTree.

    Attributes:
        metrics: Ordered list of per-item RankingMetric records, one
            entry per item in item-index order. Non-empty.
    """

    __slots__ = ("metrics",)

    def __init__(
        self,
        depth: int,
        n_samples: int,
        share: float,
        decoration: None | object,
        metrics: list[RankingMetric],
    ) -> None:
        super().__init__(depth, n_samples, share, decoration)
        self.metrics = metrics

    @property
    def prediction(self) -> int:
        """Index of the item with the lowest expected rank."""
        values = numpy.array(
            [metric.value for metric in self.metrics], dtype=float
        )
        nan_mask = numpy.isnan(values)
        if numpy.all(nan_mask):
            return 0
        safe = numpy.where(nan_mask, numpy.inf, values)
        index = int(numpy.argmin(safe))
        return index

    @property
    def ci_low(self) -> None | float:
        """Lower CI bound on the favorite item's expected rank."""
        bound = self.metrics[self.prediction].ci_low
        return bound

    @property
    def ci_high(self) -> None | float:
        """Upper CI bound on the favorite item's expected rank."""
        bound = self.metrics[self.prediction].ci_high
        return bound

    def leaf_sort_key(self) -> tuple[float, ...]:
        """Sort key: ascending lexicographic on per-item expected ranks."""
        components: list[float] = []
        for metric in self.metrics:
            value = metric.value
            sort_value = float("inf") if numpy.isnan(value) else value
            components.append(sort_value)
        key = tuple(components)
        return key


def _populate_share(root: Node) -> None:
    """Set share on every node as n_samples / root.n_samples."""
    total = root.n_samples
    _assign_share(root, total)


def _assign_share(node: Node, total: int) -> None:
    """Assign share to node and its descendants relative to total."""
    node.share = node.n_samples / total
    match node.extension:
        case _partition.Partition() as partition:
            for child in partition.children:
                _assign_share(child, total)  # ty: ignore[invalid-argument-type]


def display_branches(
    node: Node, partition: _partition.Partition, best_first: bool
) -> list[tuple[_partition.BranchCondition, Node]]:
    """Order a partition's branches for display as (condition, child) pairs.

    Branches are ordered ascending by their child's leaf sort key, then
    reversed in full when best_first is True.
    """
    conditions = partition.branch_conditions
    children = partition.children
    pairs = list(zip(conditions, children))
    pairs.sort(key=lambda pair: pair[1].leaf_sort_key())
    if best_first:
        pairs.reverse()
    return pairs
