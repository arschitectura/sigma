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

from . import (
    _export,
    _feature,
    _metric,
    _node,
    _partition,
    _tree,
    _tree_classification,
    _tree_ranking,
    _tree_regression,
    _tree_survival,
)

export_graphviz = _export.export_graphviz
export_image = _export.export_image
export_sql = _export.export_sql
export_text = _export.export_text
Extension = _partition.Extension
Leaf = _partition.Leaf
BooleanFeature = _feature.BooleanFeature
CategoricalFeature = _feature.CategoricalFeature
Feature = _feature.Feature
NumericFeature = _feature.NumericFeature
PromotedBooleanFeature = _feature.PromotedBooleanFeature
ExpectedRankMetric = _metric.ExpectedRankMetric
MedianSurvivalMetric = _metric.MedianSurvivalMetric
Metric = _metric.Metric
RiskScoreMetric = _metric.RiskScoreMetric
RmstMetric = _metric.RmstMetric
SurvivalAtMetric = _metric.SurvivalAtMetric
ClassificationNode = _node.ClassificationNode
Node = _node.Node
RankingNode = _node.RankingNode
RegressionNode = _node.RegressionNode
SurvivalNode = _node.SurvivalNode
BooleanPartition = _partition.BooleanPartition
BooleanValue = _partition.BooleanValue
BranchCondition = _partition.BranchCondition
CategoricalPartition = _partition.CategoricalPartition
CategorySubset = _partition.CategorySubset
NumericalPartition = _partition.NumericalPartition
NumericInterval = _partition.NumericInterval
Partition = _partition.Partition
SplitStatistics = _partition.SplitStatistics
InconsistentVersionWarning = _tree.InconsistentVersionWarning
ClassificationTree = _tree_classification.ClassificationTree
RankingTree = _tree_ranking.RankingTree
RegressionTree = _tree_regression.RegressionTree
SurvivalTree = _tree_survival.SurvivalTree

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
    "InconsistentVersionWarning",
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
