"""Reproduce scikit-learn's target-dependent estimator checks for the trees
whose target is two-dimensional (SurvivalTree, RankingTree).

scikit-learn's check_estimator feeds a one-dimensional y, so the 28 checks
that fit the estimator cannot run against these estimators and are declared
expected failures in test_estimator.py. This module re-runs each of those 28
checks with a valid two-dimensional target, reproducing scikit-learn's own
setup and assertions (and reusing its assertion helpers), then adds the
checks specific to the survival and ranking targets.
"""

import copy
import functools
import inspect
import pickle
import unittest

import joblib
import numpy
import pandas
import sklearn.datasets
import sklearn.exceptions
import sklearn.model_selection
import sklearn.pipeline
import sklearn.utils
import sklearn.utils._testing
import sklearn.utils.estimator_checks
import sklearn.utils.validation

import sigma._tree_ranking
import sigma._tree_survival


_RANKING_ITEMS = 3

_PREDICTION_METHOD_NAMES = (
    "predict",
    "transform",
    "decision_function",
    "predict_proba",
    "score_samples",
)

# Integer sample weights equal row repetition only when each observation's
# influence function depends on that observation alone. RankingTree (fixed
# per-row projected response) qualifies; SurvivalTree does not, because its
# log-rank scores depend on the node's full risk set, which row repetition
# alters through tied event times.
_WEIGHT_REPEAT_EQUIVALENT_LABELS = ("RankingTree",)


def _survival_target(X):
    """Build a valid (n, 2) (time, event) survival target from a design X."""
    X_array = numpy.asarray(X, dtype=float)
    feature = X_array[:, 0]
    pivot = numpy.median(feature)
    deviation = numpy.abs(feature - pivot)
    time = numpy.clip(1.0 + deviation, 0.1, None)
    event = (feature >= pivot).astype(float)
    target = numpy.column_stack([time, event])
    return target


def _ranking_target(X):
    """Build a valid (n, n_items) per-item rank target from a design X."""
    X_array = numpy.asarray(X, dtype=float)
    feature = X_array[:, 0]
    pivot = numpy.median(feature)
    ascending = numpy.arange(1.0, _RANKING_ITEMS + 1.0)
    descending = ascending[::-1]
    high_mask = feature >= pivot
    target = numpy.where(
        high_mask[:, None], ascending[None, :], descending[None, :]
    )
    return target


def _estimator_cases():
    """Return (label, factory, target_builder) for each 2D-target tree."""
    survival_factory = functools.partial(
        sigma._tree_survival.SurvivalTree, min_splits=2, min_buckets=1
    )
    ranking_factory = functools.partial(
        sigma._tree_ranking.RankingTree,
        pca_components=2,
        min_splits=2,
        min_buckets=1,
    )
    cases = [
        ("SurvivalTree", survival_factory, _survival_target),
        ("RankingTree", ranking_factory, _ranking_target),
    ]
    return cases


def _present_prediction_methods(estimator):
    """Return the estimator's available scikit-learn prediction methods."""
    methods = []
    for name in _PREDICTION_METHOD_NAMES:
        if hasattr(estimator, name):
            method = getattr(estimator, name)
            methods.append(method)
    return methods


def _is_public_attribute(name):
    """Return whether an attribute name is public (no leading/trailing _)."""
    is_public = not (name.startswith("_") or name.endswith("_"))
    return is_public


def _fitted_survival_tree(X, y):
    """Fit and return a SurvivalTree on the given design and target."""
    survival_tree = sigma._tree_survival.SurvivalTree(
        min_splits=2, min_buckets=1
    )
    survival_tree.fit(X, y)
    return survival_tree


def _fitted_ranking_tree(X, y):
    """Fit and return a RankingTree on the given design and target."""
    ranking_tree = sigma._tree_ranking.RankingTree(
        pca_components=2, min_splits=2, min_buckets=1
    )
    ranking_tree.fit(X, y)
    return ranking_tree


class TestSklearnCheckParity(unittest.TestCase):
    """Reproduce scikit-learn's 28 target-dependent checks with a 2D target."""

    __slots__ = ()

    def test_dict_unchanged(self):
        """Prediction methods do not mutate the fitted estimator's __dict__."""
        for label, factory, target in _estimator_cases():
            with self.subTest(estimator=label):
                rng = numpy.random.RandomState(0)
                X = 3.0 * rng.uniform(size=(20, 3))
                y = target(X)
                estimator = factory()
                sklearn.utils.estimator_checks.set_random_state(estimator, 1)
                estimator.fit(X, y)
                for method in _present_prediction_methods(estimator):
                    dict_before = estimator.__dict__.copy()
                    method(X)
                    unchanged = estimator.__dict__ == dict_before
                    self.assertTrue(
                        unchanged,
                        f"{label} changed __dict__ during {method.__name__}",
                    )

    def test_dont_overwrite_parameters(self):
        """fit adds or rebinds no public (non-underscore) attribute."""
        for label, factory, target in _estimator_cases():
            with self.subTest(estimator=label):
                rng = numpy.random.RandomState(0)
                X = 3.0 * rng.uniform(size=(20, 3))
                y = target(X)
                estimator = factory()
                sklearn.utils.estimator_checks.set_random_state(estimator, 1)
                dict_before = estimator.__dict__.copy()
                estimator.fit(X, y)
                dict_after = estimator.__dict__
                added = [
                    key
                    for key in dict_after
                    if key not in dict_before and _is_public_attribute(key)
                ]
                self.assertEqual(added, [], f"{label} added public attrs")
                changed = [
                    key
                    for key in dict_before
                    if _is_public_attribute(key)
                    and dict_before[key] is not dict_after[key]
                ]
                self.assertEqual(changed, [], f"{label} changed public attrs")

    def test_dtype_object(self):
        """Numeric object-dtype X fits; a non-numeric object cell raises."""
        for label, factory, target in _estimator_cases():
            with self.subTest(estimator=label):
                rng = numpy.random.RandomState(0)
                X_float = rng.uniform(size=(40, 10))
                X = X_float.astype(object)
                y = target(X)
                estimator = factory()
                estimator.fit(X, y)
                for method in _present_prediction_methods(estimator):
                    method(X)
                y_object = y.astype(object)
                with sklearn.utils._testing.raises(
                    Exception, match="Unknown label type", may_pass=True
                ):
                    factory().fit(X, y_object)
                X_bad = X.copy()
                X_bad[0, 0] = {"foo": "bar"}
                message = "argument must be .* string.* number"
                with sklearn.utils._testing.raises(TypeError, match=message):
                    factory().fit(X_bad, y)

    def test_estimators_dtypes(self):
        """fit and predict run across float32, float64, int64, int32 X."""
        for label, factory, target in _estimator_cases():
            with self.subTest(estimator=label):
                rng = numpy.random.RandomState(0)
                X_float32 = (3.0 * rng.uniform(size=(20, 5))).astype(
                    numpy.float32
                )
                y = target(X_float32)
                dtypes = (
                    numpy.float32,
                    numpy.float64,
                    numpy.int64,
                    numpy.int32,
                )
                for dtype in dtypes:
                    X = X_float32.astype(dtype)
                    estimator = factory()
                    sklearn.utils.estimator_checks.set_random_state(
                        estimator, 1
                    )
                    estimator.fit(X, y)
                    for method in _present_prediction_methods(estimator):
                        method(X)

    def test_estimators_fit_returns_self(self):
        """fit returns the estimator instance itself."""
        for label, factory, target in _estimator_cases():
            with self.subTest(estimator=label):
                X, _ = sklearn.datasets.make_blobs(random_state=0, n_samples=21)
                y = target(X)
                estimator = factory()
                sklearn.utils.estimator_checks.set_random_state(estimator)
                fitted = estimator.fit(X, y)
                self.assertIs(fitted, estimator)

    def test_estimators_nan_inf(self):
        """inf in X raises in fit and predict; NaN in X is accepted (MIA)."""
        for label, factory, target in _estimator_cases():
            with self.subTest(estimator=label):
                rng = numpy.random.RandomState(0)
                X_finite = rng.uniform(size=(10, 3))
                y = target(X_finite)
                X_inf = X_finite.copy()
                X_inf[0, 0] = numpy.inf
                X_nan = X_finite.copy()
                X_nan[0, 0] = numpy.nan
                estimator = factory()
                sklearn.utils.estimator_checks.set_random_state(estimator, 1)
                with sklearn.utils._testing.raises(
                    ValueError, match=["inf", "NaN"]
                ):
                    estimator.fit(X_inf, y)
                estimator.fit(X_nan, y)
                for method in _present_prediction_methods(estimator):
                    method(X_nan)
                    with sklearn.utils._testing.raises(
                        ValueError, match=["inf", "NaN"]
                    ):
                        method(X_inf)

    def test_estimators_overwrite_params(self):
        """fit mutates no constructor hyperparameter."""
        for label, factory, target in _estimator_cases():
            with self.subTest(estimator=label):
                X, _ = sklearn.datasets.make_blobs(random_state=0, n_samples=21)
                y = target(X)
                estimator = factory()
                sklearn.utils.estimator_checks.set_random_state(estimator)
                params = estimator.get_params()
                original = copy.deepcopy(params)
                estimator.fit(X, y)
                after = estimator.get_params()
                for name in original:
                    after_hash = joblib.hash(after[name])
                    original_hash = joblib.hash(original[name])
                    self.assertEqual(
                        after_hash,
                        original_hash,
                        f"{label} mutated parameter {name}",
                    )

    def test_estimators_pickle(self):
        """A fitted estimator pickles and predicts identically afterward."""
        for label, factory, target in _estimator_cases():
            with self.subTest(estimator=label):
                X, _ = sklearn.datasets.make_blobs(
                    n_samples=30,
                    centers=[[0, 0, 0], [1, 1, 1]],
                    n_features=2,
                    cluster_std=0.1,
                    random_state=0,
                )
                y = target(X)
                estimator = factory()
                sklearn.utils.estimator_checks.set_random_state(estimator)
                estimator.fit(X, y)
                results = {}
                for method in _present_prediction_methods(estimator):
                    results[method.__name__] = method(X)
                serialized = pickle.dumps(estimator)
                unpickled = pickle.loads(serialized)
                for name, expected in results.items():
                    bound = getattr(unpickled, name)
                    actual = bound(X)
                    sklearn.utils._testing.assert_allclose_dense_sparse(
                        expected, actual
                    )

    def test_f_contiguous_array_estimator(self):
        """fit and predict run on a Fortran-ordered design matrix."""
        for label, factory, target in _estimator_cases():
            with self.subTest(estimator=label):
                rng = numpy.random.RandomState(0)
                X_c = 3.0 * rng.uniform(size=(20, 3))
                y = target(X_c)
                X = numpy.asfortranarray(X_c)
                estimator = factory()
                estimator.fit(X, y)
                for method in _present_prediction_methods(estimator):
                    method(X)

    def test_fit2d_1feature(self):
        """A single-feature X either fits or raises an informative error."""
        for label, factory, target in _estimator_cases():
            with self.subTest(estimator=label):
                rng = numpy.random.RandomState(0)
                X = 3.0 * rng.uniform(size=(10, 1))
                y = target(X)
                estimator = factory()
                sklearn.utils.estimator_checks.set_random_state(estimator, 1)
                messages = [r"1 feature\(s\)", "n_features = 1", "n_features=1"]
                with sklearn.utils._testing.raises(
                    ValueError, match=messages, may_pass=True
                ):
                    estimator.fit(X, y)

    def test_fit2d_1sample(self):
        """A single-sample X either fits or raises an informative error."""
        for label, factory, target in _estimator_cases():
            with self.subTest(estimator=label):
                rng = numpy.random.RandomState(0)
                X = 3.0 * rng.uniform(size=(1, 10))
                y = target(X)
                estimator = factory()
                sklearn.utils.estimator_checks.set_random_state(estimator, 1)
                messages = [
                    "1 sample",
                    "n_samples = 1",
                    "n_samples=1",
                    "one sample",
                    "1 class",
                    "one class",
                ]
                with sklearn.utils._testing.raises(
                    ValueError, match=messages, may_pass=True
                ):
                    estimator.fit(X, y)

    def test_fit2d_predict1d(self):
        """Prediction methods reject a one-dimensional input."""
        for label, factory, target in _estimator_cases():
            with self.subTest(estimator=label):
                rng = numpy.random.RandomState(0)
                X = 3.0 * rng.uniform(size=(20, 3))
                y = target(X)
                estimator = factory()
                sklearn.utils.estimator_checks.set_random_state(estimator, 1)
                estimator.fit(X, y)
                for method in _present_prediction_methods(estimator):
                    with sklearn.utils._testing.raises(
                        ValueError, match="Reshape your data"
                    ):
                        method(X[0])

    def test_fit_check_is_fitted(self):
        """check_is_fitted fails before fit and passes after."""
        for label, factory, target in _estimator_cases():
            with self.subTest(estimator=label):
                rng = numpy.random.RandomState(0)
                X = rng.normal(loc=100.0, size=(100, 2))
                y = target(X)
                estimator = factory()
                sklearn.utils.estimator_checks.set_random_state(estimator)
                with sklearn.utils._testing.raises(
                    sklearn.exceptions.NotFittedError
                ):
                    sklearn.utils.validation.check_is_fitted(estimator)
                estimator.fit(X, y)
                sklearn.utils.validation.check_is_fitted(estimator)

    def test_fit_idempotent(self):
        """Fitting twice on the same data yields identical predictions."""
        for label, factory, target in _estimator_cases():
            with self.subTest(estimator=label):
                rng = numpy.random.RandomState(0)
                X = rng.normal(loc=100.0, size=(100, 2))
                y = target(X)
                splitter = sklearn.model_selection.ShuffleSplit(
                    test_size=0.2, random_state=rng
                )
                splits = splitter.split(X)
                train_index, test_index = next(splits)
                X_train = X[train_index]
                X_test = X[test_index]
                y_train = y[train_index]
                estimator = factory()
                sklearn.utils.estimator_checks.set_random_state(estimator)
                estimator.fit(X_train, y_train)
                first = {}
                for method in _present_prediction_methods(estimator):
                    first[method.__name__] = method(X_test)
                sklearn.utils.estimator_checks.set_random_state(estimator)
                estimator.fit(X_train, y_train)
                for name, expected in first.items():
                    bound = getattr(estimator, name)
                    actual = bound(X_test)
                    sklearn.utils._testing.assert_allclose_dense_sparse(
                        expected,
                        actual,
                        rtol=1e-7,
                        atol=1e-9,
                        err_msg=f"idempotency failed for {label}.{name}",
                    )

    def test_fit_score_takes_y(self):
        """fit and score accept y as their second positional argument."""
        for label, factory, target in _estimator_cases():
            with self.subTest(estimator=label):
                rng = numpy.random.RandomState(0)
                X = rng.uniform(size=(30, 3))
                y = target(X)
                estimator = factory()
                sklearn.utils.estimator_checks.set_random_state(estimator)
                for name in ("fit", "score", "fit_predict", "fit_transform"):
                    if not hasattr(estimator, name):
                        continue
                    func = getattr(estimator, name)
                    func(X, y)
                    signature = inspect.signature(func)
                    parameter_names = signature.parameters.keys()
                    parameters = list(parameter_names)
                    if parameters and parameters[0] == "self":
                        parameters = parameters[1:]
                    self.assertIn(
                        parameters[1],
                        ("y", "Y"),
                        f"{label}.{name} second argument is not y",
                    )

    def test_methods_sample_order_invariance(self):
        """Permuting input rows permutes predictions identically."""
        for label, factory, target in _estimator_cases():
            with self.subTest(estimator=label):
                rng = numpy.random.RandomState(0)
                X = 3.0 * rng.uniform(size=(20, 3))
                y = target(X)
                estimator = factory()
                sklearn.utils.estimator_checks.set_random_state(estimator, 1)
                estimator.fit(X, y)
                permutation = rng.permutation(X.shape[0])
                X_permuted = X[permutation]
                for method in _present_prediction_methods(estimator):
                    original = method(X)
                    on_permuted = method(X_permuted)
                    expected = sklearn.utils._safe_indexing(
                        original, permutation
                    )
                    sklearn.utils._testing.assert_allclose_dense_sparse(
                        expected,
                        on_permuted,
                        atol=1e-9,
                        err_msg=f"order invariance failed for {label}",
                    )

    def test_methods_subset_invariance(self):
        """Per-sample predictions are independent of batch composition."""
        for label, factory, target in _estimator_cases():
            with self.subTest(estimator=label):
                rng = numpy.random.RandomState(0)
                X = 3.0 * rng.uniform(size=(20, 3))
                y = target(X)
                estimator = factory()
                sklearn.utils.estimator_checks.set_random_state(estimator, 1)
                estimator.fit(X, y)
                n_features = X.shape[1]
                for method in _present_prediction_methods(estimator):
                    full = method(X)
                    per_row = []
                    for row in X:
                        reshaped = row.reshape(1, n_features)
                        single = method(reshaped)
                        per_row.append(single)
                    concatenated = numpy.concatenate(per_row)
                    batched = numpy.ravel(concatenated)
                    full_raveled = numpy.ravel(full)
                    sklearn.utils._testing.assert_allclose(
                        full_raveled,
                        batched,
                        atol=1e-7,
                        err_msg=f"subset invariance failed for {label}",
                    )

    def test_n_features_in(self):
        """n_features_in_ is absent before fit and correct after."""
        for label, factory, target in _estimator_cases():
            with self.subTest(estimator=label):
                rng = numpy.random.RandomState(0)
                X = rng.normal(loc=100.0, size=(100, 2))
                y = target(X)
                estimator = factory()
                sklearn.utils.estimator_checks.set_random_state(estimator)
                self.assertFalse(hasattr(estimator, "n_features_in_"))
                estimator.fit(X, y)
                self.assertTrue(hasattr(estimator, "n_features_in_"))
                self.assertEqual(estimator.n_features_in_, X.shape[1])

    def test_n_features_in_after_fitting(self):
        """Prediction methods reject inputs with a different feature count."""
        for label, factory, target in _estimator_cases():
            with self.subTest(estimator=label):
                rng = numpy.random.RandomState(0)
                X = rng.normal(loc=100.0, size=(10, 4))
                y = target(X)
                estimator = factory()
                sklearn.utils.estimator_checks.set_random_state(estimator)
                estimator.fit(X, y)
                self.assertEqual(estimator.n_features_in_, X.shape[1])
                X_bad = X[:, [1]]
                message = (
                    r"X has 1 features, but \w+ is expecting "
                    rf"{X.shape[1]} features as input"
                )
                for name in (
                    "predict",
                    "transform",
                    "decision_function",
                    "predict_proba",
                    "score",
                ):
                    if not hasattr(estimator, name):
                        continue
                    method = getattr(estimator, name)
                    if name == "score":
                        callable_method = functools.partial(method, y=y)
                    else:
                        callable_method = method
                    with sklearn.utils._testing.raises(
                        ValueError, match=message
                    ):
                        callable_method(X_bad)

    def test_pipeline_consistency(self):
        """A single-step Pipeline reproduces the estimator's predict and
        score outputs."""
        for label, factory, target in _estimator_cases():
            with self.subTest(estimator=label):
                X, _ = sklearn.datasets.make_blobs(
                    n_samples=30,
                    centers=[[0, 0, 0], [1, 1, 1]],
                    n_features=2,
                    cluster_std=0.1,
                    random_state=0,
                )
                y = target(X)
                estimator = factory()
                sklearn.utils.estimator_checks.set_random_state(estimator)
                pipeline = sklearn.pipeline.make_pipeline(estimator)
                estimator.fit(X, y)
                pipeline.fit(X, y)
                estimator_prediction = estimator.predict(X)
                pipeline_prediction = pipeline.predict(X)
                sklearn.utils._testing.assert_allclose_dense_sparse(
                    estimator_prediction, pipeline_prediction
                )
                for name in ("score", "fit_transform"):
                    has_both = hasattr(estimator, name) and hasattr(
                        pipeline, name
                    )
                    if not has_both:
                        continue
                    estimator_func = getattr(estimator, name)
                    pipeline_func = getattr(pipeline, name)
                    expected = estimator_func(X, y)
                    actual = pipeline_func(X, y)
                    sklearn.utils._testing.assert_allclose_dense_sparse(
                        expected, actual
                    )

    def test_positive_only_tag_during_fit(self):
        """With positive_only False, fit accepts negative feature values."""
        for label, factory, target in _estimator_cases():
            with self.subTest(estimator=label):
                X_iris, _ = sklearn.datasets.load_iris(return_X_y=True)
                y = target(X_iris)
                estimator = factory()
                sklearn.utils.estimator_checks.set_random_state(estimator)
                global_mean = X_iris.mean()
                centered_offset = global_mean.astype(X_iris.dtype)
                X = X_iris - centered_offset
                tags = sklearn.utils.get_tags(estimator)
                if tags.input_tags.positive_only:
                    with sklearn.utils._testing.raises(
                        ValueError, match="Negative values in data"
                    ):
                        estimator.fit(X, y)
                else:
                    try:
                        estimator.fit(X, y)
                    except Exception as exception:
                        raise AssertionError(
                            f"{label} raised {exception.__class__.__name__}"
                            " unexpectedly on centered data"
                        ) from exception

    def test_readonly_memmap_input(self):
        """fit accepts read-only memmap-backed X and y and returns self."""
        for label, factory, target in _estimator_cases():
            with self.subTest(estimator=label):
                X, _ = sklearn.datasets.make_blobs(random_state=0, n_samples=21)
                y = target(X)
                X_mmap, y_mmap = (
                    sklearn.utils._testing.create_memmap_backed_data([X, y])
                )
                estimator = factory()
                sklearn.utils.estimator_checks.set_random_state(estimator)
                fitted = estimator.fit(X_mmap, y_mmap)
                self.assertIs(fitted, estimator)

    def test_sample_weight_equivalence_on_dense_data(self):
        """Integer sample_weight reproduces row repetition for RankingTree and
        yields a valid full-length prediction for SurvivalTree."""
        for label, factory, target in _estimator_cases():
            with self.subTest(estimator=label):
                rng = numpy.random.RandomState(42)
                X = rng.rand(15, 30)
                y = target(X)
                sample_weight = rng.randint(0, 5, size=15)
                X_repeated = numpy.repeat(X, sample_weight, axis=0)
                y_repeated = numpy.repeat(y, sample_weight, axis=0)
                X_weighted, y_weighted, weight_weighted = sklearn.utils.shuffle(
                    X, y, sample_weight, random_state=0
                )
                estimator_repeated = factory()
                estimator_weighted = factory()
                sklearn.utils.estimator_checks.set_random_state(
                    estimator_repeated, 0
                )
                sklearn.utils.estimator_checks.set_random_state(
                    estimator_weighted, 0
                )
                estimator_repeated.fit(X_repeated, y_repeated)
                estimator_weighted.fit(
                    X_weighted, y_weighted, sample_weight=weight_weighted
                )
                expects_equivalence = label in _WEIGHT_REPEAT_EQUIVALENT_LABELS
                repeated_methods = _present_prediction_methods(
                    estimator_repeated
                )
                for method in repeated_methods:
                    name = method.__name__
                    weighted_method = getattr(estimator_weighted, name)
                    prediction_repeated = method(X)
                    prediction_weighted = weighted_method(X)
                    if expects_equivalence:
                        sklearn.utils._testing.assert_allclose_dense_sparse(
                            prediction_repeated,
                            prediction_weighted,
                            err_msg=f"weighted != repeated for {label}.{name}",
                        )
                    else:
                        expected_shape = (X.shape[0],)
                        self.assertEqual(
                            prediction_weighted.shape, expected_shape
                        )
                        is_nan = numpy.isnan(prediction_weighted)
                        is_finite = numpy.isfinite(prediction_weighted)
                        finite_or_nan = (is_nan | is_finite).all()
                        self.assertTrue(finite_or_nan)

    def test_sample_weights_list(self):
        """sample_weight may be supplied as a Python list."""
        for label, factory, target in _estimator_cases():
            with self.subTest(estimator=label):
                rng = numpy.random.RandomState(0)
                X = rng.uniform(size=(30, 3))
                y = target(X)
                estimator = factory()
                sample_weight = [3] * 30
                estimator.fit(X, y, sample_weight=sample_weight)

    def test_sample_weights_not_an_array(self):
        """sample_weight may be an array-convertible non-ndarray."""
        for label, factory, target in _estimator_cases():
            with self.subTest(estimator=label):
                rng = numpy.random.RandomState(0)
                X = rng.uniform(size=(12, 2))
                y = target(X)
                estimator = factory()
                X_wrapped = sklearn.utils.estimator_checks._NotAnArray(X)
                y_wrapped = sklearn.utils.estimator_checks._NotAnArray(y)
                weights = sklearn.utils.estimator_checks._NotAnArray([1] * 12)
                estimator.fit(X_wrapped, y_wrapped, sample_weight=weights)

    def test_sample_weights_not_overwritten(self):
        """fit does not mutate the caller's sample_weight array."""
        for label, factory, target in _estimator_cases():
            with self.subTest(estimator=label):
                rng = numpy.random.RandomState(0)
                X = rng.uniform(size=(16, 2))
                y = target(X)
                estimator = factory()
                sklearn.utils.estimator_checks.set_random_state(estimator, 0)
                sample_weight_original = numpy.ones(16)
                sample_weight_original[0] = 10.0
                sample_weight_fit = sample_weight_original.copy()
                estimator.fit(X, y, sample_weight=sample_weight_fit)
                sklearn.utils._testing.assert_allclose(
                    sample_weight_fit,
                    sample_weight_original,
                    err_msg=f"{label} overwrote the caller's sample_weight",
                )

    def test_sample_weights_pandas_series(self):
        """sample_weight may be a pandas Series with a DataFrame X and y."""
        for label, factory, target in _estimator_cases():
            with self.subTest(estimator=label):
                rng = numpy.random.RandomState(0)
                X_array = rng.uniform(size=(12, 2))
                y_array = target(X_array)
                X = pandas.DataFrame(X_array)
                y = pandas.DataFrame(y_array)
                ones = numpy.ones(12)
                weights = pandas.Series(ones)
                estimator = factory()
                try:
                    estimator.fit(X, y, sample_weight=weights)
                except ValueError as exception:
                    raise ValueError(
                        f"{label} rejects a pandas.Series sample_weight"
                    ) from exception

    def test_sample_weights_shape(self):
        """A sample_weight whose shape mismatches X raises ValueError."""
        for label, factory, target in _estimator_cases():
            with self.subTest(estimator=label):
                rng = numpy.random.RandomState(0)
                X = rng.uniform(size=(16, 2))
                y = target(X)
                estimator = factory()
                sklearn.utils.estimator_checks.set_random_state(estimator, 0)
                n_samples = len(y)
                full_weight = numpy.ones(n_samples)
                estimator.fit(X, y, sample_weight=full_weight)
                double_weight = numpy.ones(2 * n_samples)
                with sklearn.utils._testing.raises(ValueError):
                    estimator.fit(X, y, sample_weight=double_weight)
                matrix_weight = numpy.ones((n_samples, 2))
                with sklearn.utils._testing.raises(ValueError):
                    estimator.fit(X, y, sample_weight=matrix_weight)


class TestTwoDimensionalTargetContract(unittest.TestCase):
    """Checks specific to the survival and ranking two-dimensional targets."""

    __slots__ = ()

    def test_survival_rejects_malformed_target(self):
        """SurvivalTree.fit rejects a target that is not (time, event)."""
        rng = numpy.random.RandomState(0)
        X = rng.uniform(size=(40, 3))
        y = _survival_target(X)
        three_columns = numpy.column_stack([y, y[:, :1]])
        with self.assertRaises(ValueError):
            sigma._tree_survival.SurvivalTree().fit(X, three_columns)
        bad_event = y.copy()
        bad_event[0, 1] = 2.0
        with self.assertRaises(ValueError):
            sigma._tree_survival.SurvivalTree().fit(X, bad_event)
        negative_time = y.copy()
        negative_time[0, 0] = -1.0
        with self.assertRaises(ValueError):
            sigma._tree_survival.SurvivalTree().fit(X, negative_time)

    def test_ranking_rejects_malformed_target(self):
        """RankingTree.fit rejects 1D, single-column, or under-ranked y."""
        rng = numpy.random.RandomState(0)
        X = rng.uniform(size=(40, 3))
        y = _ranking_target(X)
        with self.assertRaises(ValueError):
            sigma._tree_ranking.RankingTree(pca_components=2).fit(X, y[:, 0])
        with self.assertRaises(ValueError):
            sigma._tree_ranking.RankingTree(pca_components=2).fit(X, y[:, :1])
        too_sparse = y.copy()
        too_sparse[0, 1:] = numpy.nan
        with self.assertRaises(ValueError):
            sigma._tree_ranking.RankingTree(pca_components=2).fit(X, too_sparse)

    def test_ranking_accepts_partial_rankings(self):
        """RankingTree fits and predicts when some items are unranked (NaN)."""
        rng = numpy.random.RandomState(0)
        X = rng.uniform(size=(60, 3))
        y = _ranking_target(X)
        y[0, 2] = numpy.nan
        y[1, 2] = numpy.nan
        ranking_tree = sigma._tree_ranking.RankingTree(
            pca_components=2, min_splits=2, min_buckets=1
        )
        ranking_tree.fit(X, y)
        predictions = ranking_tree.predict(X)
        self.assertEqual(predictions.shape, (60,))

    def test_survival_score_in_unit_interval(self):
        """SurvivalTree.score returns a concordance index within [0, 1]."""
        rng = numpy.random.RandomState(0)
        X = rng.uniform(size=(80, 3))
        y = _survival_target(X)
        survival_tree = sigma._tree_survival.SurvivalTree(
            min_splits=2, min_buckets=1
        )
        survival_tree.fit(X, y)
        score = survival_tree.score(X, y)
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)

    def test_survival_specific_methods_order_invariance(self):
        """predict_survival is row-order invariant and stable across pickle."""
        rng = numpy.random.RandomState(0)
        X = rng.uniform(size=(80, 3))
        y = _survival_target(X)
        survival_tree = _fitted_survival_tree(X, y)
        times = survival_tree.event_grid_
        n_times = len(times)
        original = survival_tree.predict_survival(X, times)
        self.assertEqual(original.shape, (80, n_times))
        permutation = rng.permutation(80)
        X_permuted = X[permutation]
        on_permuted = survival_tree.predict_survival(X_permuted, times)
        sklearn.utils._testing.assert_allclose(
            original[permutation], on_permuted
        )
        serialized = pickle.dumps(survival_tree)
        unpickled = pickle.loads(serialized)
        after_pickle = unpickled.predict_survival(X, times)
        sklearn.utils._testing.assert_allclose(original, after_pickle)

    def test_ranking_specific_methods_order_invariance(self):
        """predict_rank is row-order invariant and stable across pickle."""
        rng = numpy.random.RandomState(0)
        X = rng.uniform(size=(120, 3))
        y = _ranking_target(X)
        ranking_tree = _fitted_ranking_tree(X, y)
        original = ranking_tree.predict_rank(X)
        self.assertEqual(original.shape, (120, _RANKING_ITEMS))
        permutation = rng.permutation(120)
        X_permuted = X[permutation]
        on_permuted = ranking_tree.predict_rank(X_permuted)
        sklearn.utils._testing.assert_allclose(
            original[permutation], on_permuted
        )
        serialized = pickle.dumps(ranking_tree)
        unpickled = pickle.loads(serialized)
        after_pickle = unpickled.predict_rank(X)
        sklearn.utils._testing.assert_allclose(original, after_pickle)

    def test_survival_predict_survival_subset_invariance(self):
        """predict_survival rows are independent of batch composition."""
        rng = numpy.random.RandomState(0)
        X = rng.uniform(size=(60, 3))
        y = _survival_target(X)
        survival_tree = _fitted_survival_tree(X, y)
        times = survival_tree.event_grid_
        full = survival_tree.predict_survival(X, times)
        for index in range(X.shape[0]):
            single_X = X[index : index + 1]
            single = survival_tree.predict_survival(single_X, times)
            sklearn.utils._testing.assert_allclose(single[0], full[index])

    def test_ranking_predict_rank_subset_invariance(self):
        """predict_rank rows are independent of batch composition."""
        rng = numpy.random.RandomState(0)
        X = rng.uniform(size=(60, 3))
        y = _ranking_target(X)
        ranking_tree = _fitted_ranking_tree(X, y)
        full = ranking_tree.predict_rank(X)
        for index in range(X.shape[0]):
            single_X = X[index : index + 1]
            single = ranking_tree.predict_rank(single_X)
            sklearn.utils._testing.assert_allclose(single[0], full[index])

    def test_survival_predict_survival_unfitted_raises(self):
        """predict_survival on an unfitted SurvivalTree raises NotFittedError."""
        rng = numpy.random.RandomState(0)
        X = rng.uniform(size=(10, 3))
        times = numpy.array([1.0, 2.0])
        survival_tree = sigma._tree_survival.SurvivalTree()
        with self.assertRaises(sklearn.exceptions.NotFittedError):
            survival_tree.predict_survival(X, times)

    def test_ranking_predict_rank_unfitted_raises(self):
        """predict_rank on an unfitted RankingTree raises NotFittedError."""
        rng = numpy.random.RandomState(0)
        X = rng.uniform(size=(10, 3))
        ranking_tree = sigma._tree_ranking.RankingTree(pca_components=2)
        with self.assertRaises(sklearn.exceptions.NotFittedError):
            ranking_tree.predict_rank(X)

    def test_survival_predict_survival_rejects_bad_X(self):
        """predict_survival rejects a wrong feature count and inf in X."""
        rng = numpy.random.RandomState(0)
        X = rng.uniform(size=(40, 3))
        y = _survival_target(X)
        survival_tree = _fitted_survival_tree(X, y)
        times = survival_tree.event_grid_
        X_fewer = X[:, [0]]
        with self.assertRaises(ValueError):
            survival_tree.predict_survival(X_fewer, times)
        X_inf = X.copy()
        X_inf[0, 0] = numpy.inf
        with self.assertRaises(ValueError):
            survival_tree.predict_survival(X_inf, times)

    def test_ranking_predict_rank_rejects_bad_X(self):
        """predict_rank rejects a wrong feature count and non-finite X."""
        rng = numpy.random.RandomState(0)
        X = rng.uniform(size=(40, 3))
        y = _ranking_target(X)
        ranking_tree = _fitted_ranking_tree(X, y)
        X_fewer = X[:, [0]]
        with self.assertRaises(ValueError):
            ranking_tree.predict_rank(X_fewer)
        X_inf = X.copy()
        X_inf[0, 0] = numpy.inf
        with self.assertRaises(ValueError):
            ranking_tree.predict_rank(X_inf)

    def test_survival_score_y_encodings_equivalent(self):
        """score yields the same index for all three accepted y encodings."""
        rng = numpy.random.RandomState(0)
        X = rng.uniform(size=(80, 3))
        y = _survival_target(X)
        survival_tree = _fitted_survival_tree(X, y)
        time_column = y[:, 0]
        event_column = y[:, 1]
        structured = numpy.zeros(80, dtype=[("time", float), ("event", float)])
        structured["time"] = time_column
        structured["event"] = event_column
        age_encoded = numpy.where(
            event_column == 1.0, time_column, -time_column
        )
        score_2d = survival_tree.score(X, y)
        score_structured = survival_tree.score(X, structured)
        score_age = survival_tree.score(X, age_encoded)
        self.assertAlmostEqual(score_2d, score_structured)
        self.assertAlmostEqual(score_2d, score_age)

    def test_survival_predict_survival_curves_valid(self):
        """predict_survival returns probabilities in [0, 1] falling over time."""
        rng = numpy.random.RandomState(0)
        X = rng.uniform(size=(80, 3))
        y = _survival_target(X)
        survival_tree = _fitted_survival_tree(X, y)
        times = numpy.array([0.0, 0.5, 1.0, 2.0, 5.0])
        curves = survival_tree.predict_survival(X, times)
        self.assertEqual(curves.shape, (80, 5))
        nonnegative = numpy.all(curves >= 0.0)
        self.assertTrue(nonnegative)
        bounded = numpy.all(curves <= 1.0)
        self.assertTrue(bounded)
        time_diffs = numpy.diff(curves, axis=1)
        non_increasing = numpy.all(time_diffs <= 1e-12)
        self.assertTrue(non_increasing)

    def test_survival_predict_survival_offset_identity_and_rejection(self):
        """An all-ones offset is identity; out-of-range offsets raise."""
        rng = numpy.random.RandomState(0)
        X = rng.uniform(size=(40, 3))
        y = _survival_target(X)
        survival_tree = _fitted_survival_tree(X, y)
        times = numpy.array([0.5, 1.0, 2.0])
        n_samples = X.shape[0]
        n_times = len(times)
        bare = survival_tree.predict_survival(X, times)
        identity_offset = numpy.ones((n_samples, n_times))
        with_offset = survival_tree.predict_survival(
            X, times, offset=identity_offset
        )
        sklearn.utils._testing.assert_allclose(bare, with_offset)
        too_large = numpy.ones((n_samples, n_times))
        too_large[0, 0] = 1.5
        with self.assertRaises(ValueError):
            survival_tree.predict_survival(X, times, offset=too_large)
        increasing = numpy.full((n_samples, n_times), 0.5)
        increasing[:, -1] = 0.9
        with self.assertRaises(ValueError):
            survival_tree.predict_survival(X, times, offset=increasing)

    def test_ranking_predict_rank_within_range(self):
        """predict_rank finite entries lie in [1, n_items_]."""
        rng = numpy.random.RandomState(0)
        X = rng.uniform(size=(120, 3))
        y = _ranking_target(X)
        ranking_tree = _fitted_ranking_tree(X, y)
        ranks = ranking_tree.predict_rank(X)
        self.assertEqual(ranking_tree.n_items_, _RANKING_ITEMS)
        nan_mask = numpy.isnan(ranks)
        finite_mask = ~nan_mask
        finite_ranks = ranks[finite_mask]
        at_least_one = numpy.all(finite_ranks >= 1.0)
        self.assertTrue(at_least_one)
        at_most_items = numpy.all(finite_ranks <= _RANKING_ITEMS)
        self.assertTrue(at_most_items)

    def test_ranking_item_names_propagation(self):
        """item_names_ and predict labels follow the source of the item names."""
        rng = numpy.random.RandomState(0)
        X = rng.uniform(size=(120, 3))
        y = _ranking_target(X)
        default_tree = _fitted_ranking_tree(X, y)
        default_names = default_tree.item_names_.tolist()
        self.assertEqual(default_names, [0, 1, 2])
        names = ["soba", "udon", "ramen"]
        named_tree = sigma._tree_ranking.RankingTree(
            pca_components=2, min_splits=2, min_buckets=1, item_names=names
        )
        named_tree.fit(X, y)
        named_items = named_tree.item_names_.tolist()
        self.assertEqual(named_items, names)
        predictions = named_tree.predict(X)
        prediction_list = predictions.tolist()
        unique_predictions = set(prediction_list)
        names_set = set(names)
        is_subset = unique_predictions.issubset(names_set)
        self.assertTrue(is_subset)
        columns = ["a", "b", "c"]
        y_frame = pandas.DataFrame(y, columns=columns)
        frame_tree = _fitted_ranking_tree(X, y_frame)
        frame_items = frame_tree.item_names_.tolist()
        self.assertEqual(frame_items, columns)

    def test_ranking_rejects_offset(self):
        """RankingTree.fit rejects a fit-time offset."""
        rng = numpy.random.RandomState(0)
        X = rng.uniform(size=(40, 3))
        y = _ranking_target(X)
        offset = numpy.ones(40)
        ranking_tree = sigma._tree_ranking.RankingTree(pca_components=2)
        with self.assertRaises(ValueError):
            ranking_tree.fit(X, y, offset=offset)

    def test_feature_names_in_consistency(self):
        """Both trees record feature_names_in_ and reject renamed columns."""
        rng = numpy.random.RandomState(0)
        X_array = rng.uniform(size=(60, 3))
        columns = ["a", "b", "c"]
        X = pandas.DataFrame(X_array, columns=columns)
        survival_y = _survival_target(X_array)
        survival_tree = _fitted_survival_tree(X, survival_y)
        survival_names = survival_tree.feature_names_in_.tolist()
        self.assertEqual(survival_names, columns)
        ranking_y = _ranking_target(X_array)
        ranking_tree = _fitted_ranking_tree(X, ranking_y)
        ranking_names = ranking_tree.feature_names_in_.tolist()
        self.assertEqual(ranking_names, columns)
        renamed = pandas.DataFrame(X_array, columns=["a", "b", "d"])
        times = survival_tree.event_grid_
        with self.assertRaises(ValueError):
            survival_tree.predict_survival(renamed, times)
        with self.assertRaises(ValueError):
            ranking_tree.predict_rank(renamed)


if __name__ == "__main__":
    unittest.main()
