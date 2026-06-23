"""Conditional inference trees for Python.

Implements the algorithm from Hothorn, Hornik, and Zeileis (2006), "Unbiased
Recursive Partitioning: A Conditional Inference Framework," *Journal of
Computational and Graphical Statistics*, 15(3), 651-674, reproducing the
reference implementation in Hothorn and Zeileis (2015), "partykit: A Modular
Toolkit for Recursive Partytioning in R," *Journal of Machine Learning
Research*, 16, 3905-3909.
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
