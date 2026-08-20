"""Unit tests for the sigma version stamp carried by pickled trees."""

import copy
import pickle
import unittest
import warnings

import _helpers
import numpy

import sigma


def _fit_one_of_each():
    """Fit one tree of every estimator type on the shared step fixtures."""
    regression = _helpers._fit_step_regression_tree()
    classification = _helpers._fit_step_classification_tree()
    survival = _helpers._fit_step_survival_tree()
    X, y = _helpers._step_X_y_regression()
    ranks = numpy.column_stack([y, y.max() - y])
    ranking = sigma.RankingTree(random_state=0)
    ranking.fit(X, ranks)
    trees = {
        "RegressionTree": regression,
        "ClassificationTree": classification,
        "SurvivalTree": survival,
        "RankingTree": ranking,
    }
    return trees


class TestVersionStamp(unittest.TestCase):
    """Tests that saved trees carry the sigma version that wrote them."""

    __slots__ = ()

    def test_state_carries_the_running_version(self):
        """Every estimator stamps the installed sigma version into its state."""
        trees = _fit_one_of_each()
        for name, tree in trees.items():
            with self.subTest(estimator=name):
                state = tree.__getstate__()
                self.assertEqual(state["_sigma_version"], sigma.__version__)

    def test_pickled_bytes_carry_the_version(self):
        """The stamp reaches the serialized bytes, not only the state dict."""
        tree = _helpers._fit_step_regression_tree()
        payload = pickle.dumps(tree)
        self.assertIn(b"_sigma_version", payload)

    def test_state_is_a_copy_of_the_instance_dictionary(self):
        """Stamping the state leaves the fitted estimator unstamped."""
        tree = _helpers._fit_step_regression_tree()
        state = tree.__getstate__()
        self.assertIsNot(state, tree.__dict__)
        self.assertNotIn("_sigma_version", tree.__dict__)

    def test_mutating_the_state_leaves_the_estimator_intact(self):
        """A caller writing into the returned state cannot corrupt the tree."""
        tree = _helpers._fit_step_regression_tree()
        expected = sorted(tree.__dict__)
        state = tree.__getstate__()
        state["_probe"] = 1
        state.pop("content_", None)
        self.assertEqual(sorted(tree.__dict__), expected)

    def test_restored_tree_does_not_keep_the_stamp(self):
        """The stamp is consumed on load and never becomes an attribute."""
        tree = _helpers._fit_step_regression_tree()
        restored = pickle.loads(pickle.dumps(tree))
        self.assertNotIn("_sigma_version", restored.__dict__)


class TestVersionMismatchWarning(unittest.TestCase):
    """Tests the warning raised when a stored version differs from the running one."""

    __slots__ = ()

    def test_matching_version_loads_silently(self):
        """A tree saved and loaded by the same version warns about nothing."""
        trees = _fit_one_of_each()
        for name, tree in trees.items():
            with self.subTest(estimator=name):
                payload = pickle.dumps(tree)
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    pickle.loads(payload)
                self.assertEqual(list(caught), [])

    def test_differing_version_warns(self):
        """Loading a state stamped by another version raises the warning."""
        tree = _helpers._fit_step_regression_tree()
        state = tree.__getstate__()
        state["_sigma_version"] = "0.9.0"
        restored = sigma.RegressionTree()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            restored.__setstate__(state)
        self.assertEqual(len(caught), 1)
        self.assertIs(caught[0].category, sigma.InconsistentVersionWarning)

    def test_warning_names_the_estimator_and_both_versions(self):
        """The message identifies the estimator, the stored and running versions."""
        tree = _helpers._fit_step_survival_tree()
        state = tree.__getstate__()
        state["_sigma_version"] = "0.9.0"
        restored = sigma.SurvivalTree()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            restored.__setstate__(state)
        message = str(caught[0].message)
        self.assertIn("SurvivalTree", message)
        self.assertIn("0.9.0", message)
        self.assertIn(sigma.__version__, message)

    def test_warning_exposes_its_fields(self):
        """The warning carries the estimator name and both versions as attributes."""
        warning = sigma.InconsistentVersionWarning(
            "RegressionTree", "0.9.0", "1.0.0"
        )
        self.assertEqual(warning.estimator_name, "RegressionTree")
        self.assertEqual(warning.original_sigma_version, "0.9.0")
        self.assertEqual(warning.current_sigma_version, "1.0.0")

    def test_mismatch_still_yields_a_usable_tree(self):
        """The mismatch warns without blocking the load or changing predictions."""
        tree = _helpers._fit_step_regression_tree()
        X, _unused = _helpers._step_X_y_regression()
        expected = tree.predict(X)
        state = tree.__getstate__()
        state["_sigma_version"] = "0.9.0"
        restored = sigma.RegressionTree()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            restored.__setstate__(state)
        numpy.testing.assert_array_equal(restored.predict(X), expected)
        self.assertEqual(sigma.export_text(restored), sigma.export_text(tree))

    def test_copying_a_tree_does_not_warn(self):
        """copy, deepcopy and compact go through the stamp without warning."""
        tree = _helpers._fit_step_regression_tree()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            copy.copy(tree)
            copy.deepcopy(tree)
            tree.compact()
        self.assertEqual(list(caught), [])
