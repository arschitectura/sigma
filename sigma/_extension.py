"""Extension hierarchy attached to each node of a fitted tree."""

from __future__ import annotations

import abc
import typing

if typing.TYPE_CHECKING:
    from . import _node


N = typing.TypeVar("N", bound="_node.Node")


class Extension(typing.Generic[N], abc.ABC):
    """Per-kind payload attached to every node of a fitted tree.

    Internal nodes carry a Partition (sigma._partition.Partition,
    routing records to children); leaves carry a Leaf.
    """

    __slots__ = ()


class Leaf(Extension[N], typing.Generic[N]):
    """Extension marking that the bearing Node is a leaf of the tree.

    Attributes:
        leaf_id: Zero-based index identifying the leaf among the tree's
            leaves. Leaves are numbered 0..N-1 in ascending
            leaf_sort_key() order, matching the position of each leaf
            in Tree.leaves_ (so leaves_[k].extension.leaf_id == k).
            Assigned at fit time and unaffected by display options
            such as reverse_order. Equal to the sentinel value 0 on a
            leaf that has not yet been incorporated into a fitted tree.
    """

    __slots__ = ("leaf_id",)

    def __init__(self, leaf_id: int = 0) -> None:
        """Initialize the leaf with a pre-assigned leaf_id.

        Args:
            leaf_id: Pre-assigned leaf identifier. Defaults to the
                sentinel value 0, used during tree construction before
                the final identifier is known.
        """
        self.leaf_id = leaf_id
