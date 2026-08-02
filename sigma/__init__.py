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

import importlib.metadata

from ._export import export_graphviz, export_image, export_sql, export_text
from ._extension import Extension, Leaf
from ._feature import (
    BooleanFeature,
    CategoricalFeature,
    Feature,
    NumericFeature,
    PromotedBooleanFeature,
)
from ._metric import (
    ExpectedRankMetric,
    MedianSurvivalMetric,
    Metric,
    RiskScoreMetric,
    RmstMetric,
    SurvivalAtMetric,
)
from ._node import (
    ClassificationNode,
    Node,
    RankingNode,
    RegressionNode,
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
)
from ._tree_classification import ClassificationTree
from ._tree_ranking import RankingTree
from ._tree_regression import RegressionTree
from ._tree_survival import SurvivalTree

__version__ = importlib.metadata.version("ars-sigma")

__all__ = [
    "BooleanFeature",
    "BooleanPartition",
    "BooleanValue",
    "BranchCondition",
    "CategoricalFeature",
    "CategoricalPartition",
    "CategorySubset",
    "ClassificationNode",
    "ClassificationTree",
    "ExpectedRankMetric",
    "Extension",
    "Feature",
    "Leaf",
    "MedianSurvivalMetric",
    "Metric",
    "Node",
    "NumericFeature",
    "NumericInterval",
    "NumericalPartition",
    "Partition",
    "PromotedBooleanFeature",
    "RankingNode",
    "RankingTree",
    "RegressionNode",
    "RegressionTree",
    "RiskScoreMetric",
    "RmstMetric",
    "SplitStatistics",
    "SurvivalAtMetric",
    "SurvivalNode",
    "SurvivalTree",
    "export_graphviz",
    "export_image",
    "export_sql",
    "export_text",
]
