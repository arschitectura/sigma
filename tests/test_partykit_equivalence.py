"""Partykit-equivalence crosschecks for RegressionTree and ClassificationTree."""

import os
import typing
import unittest
import warnings

import numpy
import numpy.testing
import pandas

import sigma._node
import sigma._partition
import sigma._tree_classification
import sigma._tree_regression


_DATA_DIRECTORY = os.path.join(os.path.dirname(__file__), "data")


class TestRegressionTreePartykitEquivalence(unittest.TestCase):
    """Reproduce the partykit::ctree airquality regression tree."""

    __slots__ = ()

    def test_airquality_matches_partykit_reference(self):
        """Reproduces tree structure and leaf means from partykit on airquality."""
        frame = pandas.read_csv(os.path.join(_DATA_DIRECTORY, "airquality.csv"))
        frame = frame.dropna(subset=["Ozone"]).reset_index(drop=True)
        feature_columns = ["Wind", "Temp", "Month", "Day"]
        X = frame[feature_columns].to_numpy(dtype=float)
        y = frame["Ozone"].to_numpy(dtype=float)
        # correlation="normal" mirrors partykit's raw-value scoring;
        # test_type="sidak" matches partykit's "Bonferroni" (Sidak formula).
        estimator = sigma._tree_regression.RegressionTree(correlation="normal")
        # Ozone has many unique values, which makes sklearn's type_of_target
        # emit a "could represent a regression problem" UserWarning even
        # though RegressionTree is the right place to be.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            estimator.fit(X, y)
        root = estimator.content_
        root_partition = root.extension
        assert isinstance(root_partition, sigma._partition.NumericalPartition)
        root_partition = typing.cast(
            sigma._partition.NumericalPartition[sigma._node.RegressionNode],
            root_partition,
        )
        self.assertEqual(feature_columns[root_partition.feature_index], "Temp")
        self.assertEqual(root_partition.threshold, 82)
        left = root_partition.left
        right = root_partition.right
        left_partition = left.extension
        right_partition = right.extension
        assert isinstance(left_partition, sigma._partition.NumericalPartition)
        assert isinstance(right_partition, sigma._partition.NumericalPartition)
        self.assertEqual(feature_columns[left_partition.feature_index], "Wind")
        self.assertEqual(feature_columns[right_partition.feature_index], "Wind")
        leaves_by_n = {leaf.n_samples: leaf for leaf in estimator.leaves_}
        self.assertEqual(sorted(leaves_by_n.keys()), [7, 10, 21, 30, 48])
        self.assertAlmostEqual(leaves_by_n[10].prediction, 55.600, places=3)
        self.assertAlmostEqual(leaves_by_n[48].prediction, 18.479, places=3)
        self.assertAlmostEqual(leaves_by_n[21].prediction, 31.143, places=3)
        self.assertAlmostEqual(leaves_by_n[30].prediction, 81.633, places=3)
        self.assertAlmostEqual(leaves_by_n[7].prediction, 48.714, places=3)


class TestClassificationTreePartykitEquivalence(unittest.TestCase):
    """Reproduce the partykit::ctree GlaucomaM classification tree."""

    __slots__ = ()

    def test_glaucoma_m_matches_partykit_reference(self):
        """Reproduces tree structure and leaf class shares from partykit on GlaucomaM."""
        frame = pandas.read_csv(os.path.join(_DATA_DIRECTORY, "glaucoma_m.csv"))
        feature_columns = [c for c in frame.columns if c != "Class"]
        X = frame[feature_columns].to_numpy(dtype=float)
        y = numpy.asarray(frame["Class"] == "normal", dtype=float)
        # correlation="normal" mirrors partykit's raw-value scoring;
        # test_type="sidak" matches partykit's "Bonferroni" (Sidak formula).
        estimator = sigma._tree_classification.ClassificationTree(
            correlation="normal"
        )
        estimator.fit(X, y)
        root = estimator.content_
        root_partition = root.extension
        assert isinstance(root_partition, sigma._partition.NumericalPartition)
        root_partition = typing.cast(
            sigma._partition.NumericalPartition[sigma._node.ClassificationNode],
            root_partition,
        )
        self.assertEqual(feature_columns[root_partition.feature_index], "vari")
        left = root_partition.left
        right = root_partition.right
        left_partition = left.extension
        right_partition = right.extension
        assert isinstance(left_partition, sigma._partition.NumericalPartition)
        assert isinstance(right_partition, sigma._partition.NumericalPartition)
        self.assertEqual(feature_columns[left_partition.feature_index], "vasg")
        self.assertEqual(feature_columns[right_partition.feature_index], "tms")
        leaves_by_n = {leaf.n_samples: leaf for leaf in estimator.leaves_}
        self.assertEqual(sorted(leaves_by_n.keys()), [8, 44, 65, 79])
        numpy.testing.assert_allclose(
            leaves_by_n[79].class_distribution,
            numpy.array([74.0 / 79.0, 5.0 / 79.0]),
            atol=1e-6,
        )
        numpy.testing.assert_allclose(
            leaves_by_n[8].class_distribution,
            numpy.array([1.0 / 8.0, 7.0 / 8.0]),
            atol=1e-6,
        )
        numpy.testing.assert_allclose(
            leaves_by_n[65].class_distribution,
            numpy.array([6.0 / 65.0, 59.0 / 65.0]),
            atol=1e-6,
        )
        numpy.testing.assert_allclose(
            leaves_by_n[44].class_distribution,
            numpy.array([17.0 / 44.0, 27.0 / 44.0]),
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
