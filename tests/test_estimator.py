"""Unit tests for sklearn estimator compliance."""

import unittest
import warnings

import numpy
import sklearn.exceptions
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.utils.estimator_checks

import sigma._tree_classification
import sigma._tree_ranking
import sigma._tree_regression
import sigma._tree_survival

_EXPECTED_FAILED_CHECKS = {
    "check_do_not_raise_errors_in_init_or_set_params": (
        "sigma validates typing.Literal parameters eagerly in __init__"
        " by design, which is stricter than the sklearn convention."
    ),
}

_SURVIVAL_Y_SHAPE_FAILURE_REASON = (
    "SurvivalTree requires y of shape (n, 2) carrying (time, event);"
    " the standard sklearn check supplies 1D y."
)
_SURVIVAL_Y_SHAPE_FAILED_CHECKS = (
    "check_dict_unchanged",
    "check_dont_overwrite_parameters",
    "check_dtype_object",
    "check_estimators_dtypes",
    "check_estimators_fit_returns_self",
    "check_estimators_nan_inf",
    "check_estimators_overwrite_params",
    "check_estimators_pickle",
    "check_f_contiguous_array_estimator",
    "check_fit2d_1feature",
    "check_fit2d_1sample",
    "check_fit2d_predict1d",
    "check_fit_check_is_fitted",
    "check_fit_idempotent",
    "check_fit_score_takes_y",
    "check_methods_sample_order_invariance",
    "check_methods_subset_invariance",
    "check_n_features_in",
    "check_n_features_in_after_fitting",
    "check_pipeline_consistency",
    "check_positive_only_tag_during_fit",
    "check_readonly_memmap_input",
    "check_sample_weight_equivalence_on_dense_data",
    "check_sample_weights_list",
    "check_sample_weights_not_an_array",
    "check_sample_weights_not_overwritten",
    "check_sample_weights_pandas_series",
    "check_sample_weights_shape",
)
_SURVIVAL_EXPECTED_FAILED_CHECKS = {
    **_EXPECTED_FAILED_CHECKS,
    **dict.fromkeys(
        _SURVIVAL_Y_SHAPE_FAILED_CHECKS, _SURVIVAL_Y_SHAPE_FAILURE_REASON
    ),
}

_RANKING_Y_SHAPE_FAILURE_REASON = (
    "RankingTree requires y of shape (n_obs, n_items) carrying per-item"
    " ranks (NaN for unranked); the standard sklearn check supplies 1D"
    " float y."
)
_RANKING_EXPECTED_FAILED_CHECKS = {
    **_EXPECTED_FAILED_CHECKS,
    **dict.fromkeys(
        _SURVIVAL_Y_SHAPE_FAILED_CHECKS, _RANKING_Y_SHAPE_FAILURE_REASON
    ),
    "check_all_zero_sample_weights_error": _RANKING_Y_SHAPE_FAILURE_REASON,
}


def _run_check_estimator(estimator, expected_failed_checks):
    """Run the scikit-learn estimator checks for the given estimator."""
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Skipping check check_array_api_input",
            category=sklearn.exceptions.SkipTestWarning,
        )
        sklearn.utils.estimator_checks.check_estimator(
            estimator,
            expected_failed_checks=expected_failed_checks,
        )


class TestSklearnCompliance(unittest.TestCase):
    """Tests for scikit-learn estimator contract compliance."""

    __slots__ = ()

    def test_check_regression_tree(self):
        """RegressionTree passes all scikit-learn estimator checks."""
        _run_check_estimator(
            sigma._tree_regression.RegressionTree(),
            _EXPECTED_FAILED_CHECKS,
        )

    def test_check_classification_tree(self):
        """ClassificationTree passes all scikit-learn estimator checks."""
        _run_check_estimator(
            sigma._tree_classification.ClassificationTree(),
            _EXPECTED_FAILED_CHECKS,
        )

    def test_check_survival_tree(self):
        """SurvivalTree passes all scikit-learn estimator checks."""
        _run_check_estimator(
            sigma._tree_survival.SurvivalTree(),
            _SURVIVAL_EXPECTED_FAILED_CHECKS,
        )

    def test_check_ranking_tree(self):
        """RankingTree passes all scikit-learn estimator checks."""
        _run_check_estimator(
            sigma._tree_ranking.RankingTree(),
            _RANKING_EXPECTED_FAILED_CHECKS,
        )


class TestPipelineIntegration(unittest.TestCase):
    """End-to-end Pipeline tests for all three Tree estimators."""

    __slots__ = ()

    def test_pipeline_fit_predict_score_for_all_three_trees(self):
        """Pipeline fit/predict/score runs without raising for every tree class."""
        rng = numpy.random.RandomState(0)
        n = 200
        X = rng.randn(n, 3)
        y_real = X[:, 0] + 0.1 * rng.randn(n)
        y_int = (X[:, 0] > 0.0).astype(int)
        time = numpy.minimum(rng.exponential(scale=5.0, size=n), 8.0)
        event = (time < 8.0).astype(float)
        y_2d = numpy.column_stack([time, event])
        cases = (
            (sigma._tree_regression.RegressionTree, y_real),
            (sigma._tree_classification.ClassificationTree, y_int),
            (sigma._tree_survival.SurvivalTree, y_2d),
        )
        for tree_cls, y in cases:
            with self.subTest(tree=tree_cls.__name__):
                pipeline = sklearn.pipeline.Pipeline(
                    [
                        ("scaler", sklearn.preprocessing.StandardScaler()),
                        (
                            "tree",
                            tree_cls(min_splits=10, min_buckets=5, max_depth=2),
                        ),
                    ]
                )
                pipeline.fit(X, y)
                predictions = pipeline.predict(X)
                self.assertEqual(predictions.shape[0], n)
                score = pipeline.score(X, y)
                self.assertTrue(numpy.isfinite(score))


if __name__ == "__main__":
    unittest.main()
