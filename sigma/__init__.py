"""Conditional inference trees for Python.

Provides four scikit-learn compatible estimators: ClassificationTree,
RegressionTree, SurvivalTree, and RankingTree. Each is trained with fit,
queried with predict, and rendered with export_text, export_sql,
export_graphviz, or export_image.

Example:
    import sigma
    tree = sigma.ClassificationTree(random_state=123)
    tree.fit(X, y)
    predictions = tree.predict(X)
    report = sigma.export_text(tree)
    print(report)

Documentation and source: https://github.com/arschitectura/sigma
"""

# TODO verify the pickling

import importlib.metadata

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

__version__ = importlib.metadata.version("ars-sigma")

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
