"""Node class hierarchy and traversal for fitted conditional inference trees."""

from __future__ import annotations

import abc
import typing

import numpy
import numpy.typing
import typing_extensions

from . import _extension, _metric, _partition

if typing.TYPE_CHECKING:
    import polars


class _DisplayedValue(typing.NamedTuple):
    """One labeled value a node reports, before any formatting."""

    label: str
    value: float
    ci_low: float
    ci_high: float
    style: typing.Literal["value", "probability"]
    has_ci: bool


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
    def leaf_sort_key(
        self, metrics: tuple[_metric.Metric, ...]
    ) -> tuple[float, ...]:
        """Build the sort key ordering the leaves of this task type.

        Args:
            metrics: Metric descriptors of the tree this node belongs to,
                in the order of the node's own value array.

        Returns:
            A tuple compared lexicographically, ascending, against the key
            of any other node of the same tree.
        """

    @abc.abstractmethod
    def _displayed_values(
        self,
        metrics: tuple[_metric.Metric, ...],
        class_names: None | list[str],
        response_name: None | str,
        displayed_indices: list[int],
    ) -> list[_DisplayedValue]:
        """Labeled values this node reports to the renderers, in display order."""

    @abc.abstractmethod
    def _sql_value(self, target_class_index: None | int) -> float:
        """Single numeric value the SQL export emits for this node."""

    def _top_displayed_indices(
        self, top_displayed_items: None | int
    ) -> list[int]:
        """Positions of the values this node displays when only its best ones are kept."""
        return []


class RegressionNode(Node):
    """Node of a fitted RegressionTree.

    Attributes:
        predicted_mean: Weighted mean of the response in this node's
            active samples. When the tree was fit with an offset, the mean
            is taken over the response minus that offset.
        ci_low: Lower bound of the confidence interval for the predicted
            mean, or None when CI is disabled.
        ci_high: Upper bound of the confidence interval for the predicted
            mean, or None when CI is disabled.
        response_samples: Per-leaf array of response samples. Empty on
            internal nodes and on leaves when response_sample_size=0.
    """

    __slots__ = ("ci_high", "ci_low", "predicted_mean", "response_samples")

    def __init__(
        self,
        depth: int,
        n_samples: int,
        share: float,
        decoration: None | object,
        predicted_mean: float,
        ci_low: None | float,
        ci_high: None | float,
        response_samples: numpy.typing.NDArray[numpy.floating],
    ) -> None:
        super().__init__(depth, n_samples, share, decoration)
        self.predicted_mean = predicted_mean
        self.ci_low = ci_low
        self.ci_high = ci_high
        self.response_samples = response_samples

    def leaf_sort_key(
        self, metrics: tuple[_metric.Metric, ...]
    ) -> tuple[float, ...]:
        """Sort key: ascending by predicted mean."""
        key = (self.predicted_mean,)
        return key

    def _displayed_values(
        self,
        metrics: tuple[_metric.Metric, ...],
        class_names: None | list[str],
        response_name: None | str,
        displayed_indices: list[int],
    ) -> list[_DisplayedValue]:
        """The predicted mean, labeled after the response when it is named."""
        if response_name is None:
            label = "Predicted mean"
        else:
            label = f"{response_name} mean"
        ci_low = self.ci_low
        ci_high = self.ci_high
        if ci_low is not None and ci_high is not None:
            displayed = _DisplayedValue(
                label, self.predicted_mean, ci_low, ci_high, "value", True
            )
        else:
            undefined = float("nan")
            displayed = _DisplayedValue(
                label, self.predicted_mean, undefined, undefined, "value", False
            )
        values = [displayed]
        return values

    def _sql_value(self, target_class_index: None | int) -> float:
        """The predicted mean."""
        value = self.predicted_mean
        return value


class ClassificationNode(Node):
    """Node of a fitted ClassificationTree.

    Attributes:
        predicted_class_index: Position of the predicted class within the
            estimator's classes_ array.
        predicted_proba: Class probability vector for this node, shape
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
        "ci_high",
        "ci_low",
        "mean_offset_proba",
        "predicted_class_index",
        "predicted_proba",
    )

    def __init__(
        self,
        depth: int,
        n_samples: int,
        share: float,
        decoration: None | object,
        predicted_class_index: int,
        predicted_proba: numpy.typing.NDArray[numpy.floating],
        ci_low: None | numpy.typing.NDArray[numpy.floating],
        ci_high: None | numpy.typing.NDArray[numpy.floating],
        mean_offset_proba: None | numpy.typing.NDArray[numpy.floating],
    ) -> None:
        super().__init__(depth, n_samples, share, decoration)
        self.predicted_class_index = predicted_class_index
        self.predicted_proba = predicted_proba
        self.ci_low = ci_low
        self.ci_high = ci_high
        self.mean_offset_proba = mean_offset_proba

    def leaf_sort_key(
        self, metrics: tuple[_metric.Metric, ...]
    ) -> tuple[float, ...]:
        """Sort key: descending by class probability tuple."""
        key = tuple(-p for p in self.predicted_proba)
        return key

    def _displayed_values(
        self,
        metrics: tuple[_metric.Metric, ...],
        class_names: None | list[str],
        response_name: None | str,
        displayed_indices: list[int],
    ) -> list[_DisplayedValue]:
        """One probability per class, labeled with the class display name."""
        predicted_proba = self.predicted_proba
        ci_low = self.ci_low
        ci_high = self.ci_high
        values: list[_DisplayedValue] = []
        for index in range(len(predicted_proba)):
            if class_names is None:
                name = str(index)
            else:
                name = class_names[index]
            label = f"{name} proba."
            if ci_low is not None and ci_high is not None:
                bound_low = float(ci_low[index])
                bound_high = float(ci_high[index])
                has_ci = True
            else:
                bound_low = float("nan")
                bound_high = float("nan")
                has_ci = False
            displayed = _DisplayedValue(
                label,
                predicted_proba[index],
                bound_low,
                bound_high,
                "probability",
                has_ci,
            )
            values.append(displayed)
        return values

    def _sql_value(self, target_class_index: None | int) -> float:
        """Probability of the target class."""
        if target_class_index is None:
            raise RuntimeError(
                "target_class_index must be resolved before rendering a"
                " classification leaf"
            )
        value = float(self.predicted_proba[target_class_index])
        return value


class SurvivalNode(Node):
    """Node of a fitted SurvivalTree.

    Attributes:
        predicted_survival: Pair (times, surv) describing the
            Kaplan-Meier estimate of S(t) at this node.
        survival_log_variance: Greenwood variance of log S(t) at the same
            times as predicted_survival, shape (n,). Empty on internal
            nodes.
        predicted_metrics: Value of each metric of the fitted tree, shape
            (n_metrics,), aligned with its metrics_. NaN and +/- inf are
            allowed.
        ci_low: Lower confidence-interval bound of each metric, shape
            (n_metrics,). NaN wherever the bound is undefined.
        ci_high: Upper confidence-interval bound of each metric, shape
            (n_metrics,). NaN wherever the bound is undefined.
    """

    __slots__ = (
        "ci_high",
        "ci_low",
        "predicted_metrics",
        "predicted_survival",
        "survival_log_variance",
    )

    def __init__(
        self,
        depth: int,
        n_samples: int,
        share: float,
        decoration: None | object,
        predicted_survival: tuple[
            numpy.typing.NDArray[numpy.floating],
            numpy.typing.NDArray[numpy.floating],
        ],
        survival_log_variance: numpy.typing.NDArray[numpy.floating],
        predicted_metrics: numpy.typing.NDArray[numpy.floating],
        ci_low: numpy.typing.NDArray[numpy.floating],
        ci_high: numpy.typing.NDArray[numpy.floating],
    ) -> None:
        super().__init__(depth, n_samples, share, decoration)
        self.predicted_survival = predicted_survival
        self.survival_log_variance = survival_log_variance
        self.predicted_metrics = predicted_metrics
        self.ci_low = ci_low
        self.ci_high = ci_high

    def leaf_sort_key(
        self, metrics: tuple[_metric.Metric, ...]
    ) -> tuple[float, ...]:
        """Sort key: lexicographic on metric values, worst-prognosis-first."""
        components: list[float] = []
        for index, metric in enumerate(metrics):
            value = float(self.predicted_metrics[index])
            sign = 1.0 if metric.better_is == "higher" else -1.0
            if numpy.isnan(value):
                component = float("inf")
            else:
                component = sign * value
            components.append(component)
        key = tuple(components)
        return key

    def _displayed_values(
        self,
        metrics: tuple[_metric.Metric, ...],
        class_names: None | list[str],
        response_name: None | str,
        displayed_indices: list[int],
    ) -> list[_DisplayedValue]:
        """One value per metric of the fitted tree, in metric order."""
        values: list[_DisplayedValue] = []
        for index, metric in enumerate(metrics):
            label = metric._display_label(response_name)
            displayed = _DisplayedValue(
                label,
                self.predicted_metrics[index],
                self.ci_low[index],
                self.ci_high[index],
                metric.style,
                metric.has_ci,
            )
            values.append(displayed)
        return values

    def _sql_value(self, target_class_index: None | int) -> float:
        """Value of the first metric of the fitted tree."""
        value = float(self.predicted_metrics[0])
        return value


class RankingNode(Node):
    """Node of a fitted RankingTree.

    Attributes:
        predicted_ranks: Plackett-Luce expected rank of each item, shape
            (n_items,), in item-index order. Each finite entry lies in
            [1, n_items].
        ci_low: Lower confidence-interval bound on each expected rank,
            shape (n_items,). NaN wherever the bound is undefined.
        ci_high: Upper confidence-interval bound on each expected rank,
            shape (n_items,). NaN wherever the bound is undefined.
    """

    __slots__ = ("ci_high", "ci_low", "predicted_ranks")

    def __init__(
        self,
        depth: int,
        n_samples: int,
        share: float,
        decoration: None | object,
        predicted_ranks: numpy.typing.NDArray[numpy.floating],
        ci_low: numpy.typing.NDArray[numpy.floating],
        ci_high: numpy.typing.NDArray[numpy.floating],
    ) -> None:
        super().__init__(depth, n_samples, share, decoration)
        self.predicted_ranks = predicted_ranks
        self.ci_low = ci_low
        self.ci_high = ci_high

    @property
    def predicted_item_index(self) -> int:
        """Position of the predicted item, the one with the lowest expected
        rank, within the estimator's item_names_ array."""
        nan_mask = numpy.isnan(self.predicted_ranks)
        if numpy.all(nan_mask):
            return 0
        safe = numpy.where(nan_mask, numpy.inf, self.predicted_ranks)
        index = int(numpy.argmin(safe))
        return index

    def leaf_sort_key(
        self, metrics: tuple[_metric.Metric, ...]
    ) -> tuple[float, ...]:
        """Sort key: ascending lexicographic on per-item expected ranks."""
        components: list[float] = []
        for value in self.predicted_ranks:
            sort_value = float("inf") if numpy.isnan(value) else float(value)
            components.append(sort_value)
        key = tuple(components)
        return key

    def _displayed_values(
        self,
        metrics: tuple[_metric.Metric, ...],
        class_names: None | list[str],
        response_name: None | str,
        displayed_indices: list[int],
    ) -> list[_DisplayedValue]:
        """One expected rank per displayed item, in item order."""
        values: list[_DisplayedValue] = []
        for index in displayed_indices:
            metric = metrics[index]
            label = metric._display_label(response_name)
            displayed = _DisplayedValue(
                label,
                self.predicted_ranks[index],
                self.ci_low[index],
                self.ci_high[index],
                metric.style,
                metric.has_ci,
            )
            values.append(displayed)
        return values

    def _sql_value(self, target_class_index: None | int) -> float:
        """Unsupported: a leaf predicts a per-item vector, not one scalar."""
        raise NotImplementedError(
            "SQL export is not supported for RankingTree: a single SQL"
            " scalar cannot represent the per-item expected-rank vector"
            " predicted at each leaf."
        )

    def _top_displayed_indices(
        self, top_displayed_items: None | int
    ) -> list[int]:
        """The node's own top items by lowest non-NaN expected rank."""
        if top_displayed_items is None:
            return []
        values = self.predicted_ranks
        nan_mask = numpy.isnan(values)
        valid_indices = numpy.flatnonzero(~nan_mask)
        if valid_indices.size == 0:
            return []
        take = min(top_displayed_items, valid_indices.size)
        valid_values = values[valid_indices]
        order = numpy.argsort(valid_values, kind="stable")[:take]
        result = [int(index) for index in valid_indices[order]]
        return result


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
    node: Node,
    partition: _partition.Partition,
    best_first: bool,
    metrics: tuple[_metric.Metric, ...],
) -> list[tuple[_partition.BranchCondition, Node]]:
    """Order a partition's branches for display as (condition, child) pairs.

    Branches are ordered ascending by their child's leaf sort key, built
    against the metric descriptors of the tree being rendered, then
    reversed in full when best_first is True.
    """
    conditions = partition.branch_conditions
    children = partition.children
    pairs = list(zip(conditions, children))
    pairs.sort(key=lambda pair: pair[1].leaf_sort_key(metrics))
    if best_first:
        pairs.reverse()
    return pairs
