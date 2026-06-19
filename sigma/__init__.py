"""Conditional inference trees for Python.

Implements the algorithm from Hothorn, Hornik, and Zeileis (2006), "Unbiased
Recursive Partitioning: A Conditional Inference Framework," *Journal of
Computational and Graphical Statistics*, 15(3), 651-674.
"""

# TODO verify the pickling

from ._export import export_graphviz, export_image, export_sql, export_text
from ._extension import Extension, Leaf
from ._node import (
    ClassificationNode,
    Node,
    RankingMetric,
    RankingNode,
    RegressionNode,
    SurvivalMetric,
    SurvivalNode,
)
from ._partition import (
    BooleanPartition,
    BooleanValue,
    BranchCondition,
    CategoricalPartition,
    CategorySubset,
    NumericalPartition,
    NumericInterval,
    Partition,
    SplitStatistics,
    UnknownCategoryError,
)
from ._tree_classification import ClassificationTree
from ._tree_ranking import RankingTree
from ._tree_regression import RegressionTree
from ._tree_survival import SurvivalTree

__all__ = [
    "BooleanPartition",
    "BooleanValue",
    "BranchCondition",
    "CategoricalPartition",
    "CategorySubset",
    "ClassificationNode",
    "ClassificationTree",
    "Extension",
    "Leaf",
    "Node",
    "NumericalPartition",
    "NumericInterval",
    "Partition",
    "RankingMetric",
    "RankingNode",
    "RankingTree",
    "RegressionNode",
    "RegressionTree",
    "SplitStatistics",
    "SurvivalMetric",
    "SurvivalNode",
    "SurvivalTree",
    "UnknownCategoryError",
    "export_graphviz",
    "export_image",
    "export_sql",
    "export_text",
]
