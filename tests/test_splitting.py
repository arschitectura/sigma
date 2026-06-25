"""Unit tests for the split search module."""

import unittest

import numpy

import sigma._splitting
import sigma._types


class TestFindBestSplitNumeric(unittest.TestCase):
    """Tests for numeric split search."""

    __slots__ = ()

    def test_step_function_finds_correct_threshold_integer(self):
        """Picks the left observed value when the covariate is integer-valued."""
        X_j = numpy.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
        y = numpy.array([0, 0, 0, 0, 10, 10, 10, 10], dtype=float)
        weights = numpy.ones(8)
        result = sigma._splitting.find_best_split_numeric(
            X_j,
            y.reshape(-1, 1),
            weights,
            sigma._types.TestStat.QUADRATIC,
            min_buckets=1,
            is_integer=True,
            correlation=sigma._types.Correlation.NORMAL,
        )
        assert result is not None
        split, statistic = result
        assert split.threshold == 4
        assert statistic > 0

    def test_step_function_finds_correct_threshold_real(self):
        """Picks the midpoint when the covariate has non-integer values."""
        X_j = numpy.array([1.0, 2.0, 3.0, 4.0, 5.5, 6.0, 7.0, 8.0], dtype=float)
        y = numpy.array([0, 0, 0, 0, 10, 10, 10, 10], dtype=float)
        weights = numpy.ones(8)
        result = sigma._splitting.find_best_split_numeric(
            X_j,
            y.reshape(-1, 1),
            weights,
            sigma._types.TestStat.QUADRATIC,
            min_buckets=1,
            is_integer=False,
            correlation=sigma._types.Correlation.NORMAL,
        )
        assert result is not None
        split, statistic = result
        assert split.threshold == 4.75
        assert statistic > 0

    def test_integer_with_gap_uses_left_value(self):
        """Picks the left observed value when integer values have a gap."""
        X_j = numpy.array([1, 1, 1, 5, 5, 5], dtype=float)
        y = numpy.array([0, 0, 0, 10, 10, 10], dtype=float)
        weights = numpy.ones(6)
        result = sigma._splitting.find_best_split_numeric(
            X_j,
            y.reshape(-1, 1),
            weights,
            sigma._types.TestStat.QUADRATIC,
            min_buckets=1,
            is_integer=True,
            correlation=sigma._types.Correlation.NORMAL,
        )
        assert result is not None
        split, statistic = result
        assert split.threshold == 1
        assert statistic > 0

    def test_returns_none_single_unique_value(self):
        """Returns None when all feature values are identical."""
        X_j = numpy.array([3, 3, 3, 3], dtype=float)
        y = numpy.array([1, 2, 3, 4], dtype=float)
        weights = numpy.ones(4)
        result = sigma._splitting.find_best_split_numeric(
            X_j,
            y.reshape(-1, 1),
            weights,
            sigma._types.TestStat.QUADRATIC,
            min_buckets=1,
            is_integer=True,
            correlation=sigma._types.Correlation.NORMAL,
        )
        assert result is None

    def test_min_buckets_enforcement(self):
        """Returns None when no split satisfies min_buckets."""
        X_j = numpy.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
        y = numpy.array([0, 0, 0, 0, 10, 10, 10, 10], dtype=float)
        weights = numpy.ones(8)
        result = sigma._splitting.find_best_split_numeric(
            X_j,
            y.reshape(-1, 1),
            weights,
            sigma._types.TestStat.QUADRATIC,
            min_buckets=5,
            is_integer=True,
            correlation=sigma._types.Correlation.NORMAL,
        )
        assert result is None

    def test_respects_weights(self):
        """Ignores zero-weight samples for split evaluation."""
        X_j = numpy.array([1, 2, 3, 4], dtype=float)
        y = numpy.array([0, 0, 10, 10], dtype=float)
        weights = numpy.array([1, 1, 1, 0], dtype=float)
        result = sigma._splitting.find_best_split_numeric(
            X_j,
            y.reshape(-1, 1),
            weights,
            sigma._types.TestStat.QUADRATIC,
            min_buckets=1,
            is_integer=True,
            correlation=sigma._types.Correlation.NORMAL,
        )
        assert result is not None
        split, statistic = result
        assert split.threshold == 2
        assert statistic > 0

    def test_works_with_maximum_test_stat(self):
        """Finds the correct split using maximum-type statistic."""
        X_j = numpy.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
        y = numpy.array([0, 0, 0, 0, 10, 10, 10, 10], dtype=float)
        weights = numpy.ones(8)
        result = sigma._splitting.find_best_split_numeric(
            X_j,
            y.reshape(-1, 1),
            weights,
            sigma._types.TestStat.MAXIMUM,
            min_buckets=1,
            is_integer=True,
            correlation=sigma._types.Correlation.NORMAL,
        )
        assert result is not None
        split, statistic = result
        assert split.threshold == 4
        assert statistic > 0


class TestFindBestSplitCategorical(unittest.TestCase):
    """Tests for categorical split search."""

    __slots__ = ()

    def test_two_category_split(self):
        """Splits two categories with distinct responses."""
        X_j = numpy.array([0, 0, 0, 1, 1, 1], dtype=float)
        y = numpy.array([1, 1, 1, 10, 10, 10], dtype=float)
        weights = numpy.ones(6)
        result = sigma._splitting.find_best_split_categorical(
            X_j,
            y.reshape(-1, 1),
            weights,
            sigma._types.TestStat.QUADRATIC,
            min_buckets=1,
            correlation=sigma._types.Correlation.NORMAL,
        )
        assert result is not None
        categories, statistic = result
        assert isinstance(categories, frozenset)
        assert categories == frozenset({0.0}) or categories == frozenset({1.0})
        assert statistic > 0

    def test_three_category_exhaustive(self):
        """Groups low-response categories together via exhaustive search."""
        X_j = numpy.array([0, 0, 1, 1, 2, 2], dtype=float)
        y = numpy.array([1, 1, 1, 1, 10, 10], dtype=float)
        weights = numpy.ones(6)
        result = sigma._splitting.find_best_split_categorical(
            X_j,
            y.reshape(-1, 1),
            weights,
            sigma._types.TestStat.QUADRATIC,
            min_buckets=1,
            correlation=sigma._types.Correlation.NORMAL,
        )
        assert result is not None
        categories, statistic = result
        assert isinstance(categories, frozenset)
        # Best split groups {0, 1} vs {2}
        assert categories == frozenset({0.0, 1.0}) or categories == frozenset(
            {2.0}
        )
        assert statistic > 0

    def test_returns_none_single_category(self):
        """Returns None when only one category exists."""
        X_j = numpy.array([0, 0, 0], dtype=float)
        y = numpy.array([1, 2, 3], dtype=float)
        weights = numpy.ones(3)
        result = sigma._splitting.find_best_split_categorical(
            X_j,
            y.reshape(-1, 1),
            weights,
            sigma._types.TestStat.QUADRATIC,
            min_buckets=1,
            correlation=sigma._types.Correlation.NORMAL,
        )
        assert result is None

    def test_min_buckets_enforcement_categorical(self):
        """Returns None when no partition satisfies min_buckets."""
        X_j = numpy.array([0, 1, 1, 1, 1], dtype=float)
        y = numpy.array([10, 0, 0, 0, 0], dtype=float)
        weights = numpy.ones(5)
        result = sigma._splitting.find_best_split_categorical(
            X_j,
            y.reshape(-1, 1),
            weights,
            sigma._types.TestStat.QUADRATIC,
            min_buckets=2,
            correlation=sigma._types.Correlation.NORMAL,
        )
        assert result is None

    def test_many_categories_mean_ordering(self):
        """Uses mean-response ordering for more than 10 categories."""
        n_cats = 11
        samples_per_cat = 3
        X_j = numpy.repeat(numpy.arange(n_cats, dtype=float), samples_per_cat)
        y = numpy.repeat(numpy.arange(n_cats, dtype=float), samples_per_cat)
        weights = numpy.ones(len(X_j))
        result = sigma._splitting.find_best_split_categorical(
            X_j,
            y.reshape(-1, 1),
            weights,
            sigma._types.TestStat.QUADRATIC,
            min_buckets=1,
            correlation=sigma._types.Correlation.NORMAL,
        )
        assert result is not None
        categories, statistic = result
        assert isinstance(categories, frozenset)
        assert statistic > 0
        # The result must be a contiguous prefix of mean-ordered categories
        sorted_cats = list(range(n_cats))
        cat_list = sorted(categories)
        k = len(cat_list)
        expected_prefix = [float(c) for c in sorted_cats[:k]]
        assert cat_list == expected_prefix


class TestFindBestSplitBoolean(unittest.TestCase):
    """Tests for boolean split search."""

    __slots__ = ()

    def test_returns_sentinel_and_statistic_on_signal(self):
        """A signal-bearing 0/1 column returns (True, positive statistic)."""
        X_j = numpy.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float)
        y = numpy.array([1, 1, 1, 1, 9, 9, 9, 9], dtype=float)
        weights = numpy.ones(8)
        result = sigma._splitting.find_best_split_boolean(
            X_j,
            y.reshape(-1, 1),
            weights,
            sigma._types.TestStat.QUADRATIC,
            min_buckets=1,
            correlation=sigma._types.Correlation.NORMAL,
        )
        assert result is not None
        criterion, statistic = result
        assert criterion is True
        assert statistic > 0

    def test_returns_none_when_constant(self):
        """A column with only one observed value returns None."""
        X_j = numpy.zeros(8, dtype=float)
        y = numpy.array([1, 1, 1, 1, 9, 9, 9, 9], dtype=float)
        weights = numpy.ones(8)
        result = sigma._splitting.find_best_split_boolean(
            X_j,
            y.reshape(-1, 1),
            weights,
            sigma._types.TestStat.QUADRATIC,
            min_buckets=1,
            correlation=sigma._types.Correlation.NORMAL,
        )
        assert result is None

    def test_returns_none_when_min_buckets_violated(self):
        """Returns None when either side has fewer samples than min_buckets."""
        X_j = numpy.array([0, 0, 0, 0, 0, 0, 0, 1], dtype=float)
        y = numpy.array([1, 1, 1, 1, 1, 1, 1, 9], dtype=float)
        weights = numpy.ones(8)
        result = sigma._splitting.find_best_split_boolean(
            X_j,
            y.reshape(-1, 1),
            weights,
            sigma._types.TestStat.QUADRATIC,
            min_buckets=2,
            correlation=sigma._types.Correlation.NORMAL,
        )
        assert result is None

    def test_respects_weights(self):
        """Zero-weight rows are ignored when checking feasibility."""
        X_j = numpy.array([0, 0, 1, 1], dtype=float)
        y = numpy.array([1, 1, 9, 9], dtype=float)
        weights = numpy.array([1, 1, 1, 0], dtype=float)
        result = sigma._splitting.find_best_split_boolean(
            X_j,
            y.reshape(-1, 1),
            weights,
            sigma._types.TestStat.QUADRATIC,
            min_buckets=1,
            correlation=sigma._types.Correlation.NORMAL,
        )
        assert result is not None
        criterion, _statistic = result
        assert criterion is True


class TestFindBestSplit(unittest.TestCase):
    """Tests for the split dispatch function."""

    __slots__ = ()

    def test_dispatches_to_boolean(self):
        """Routes BOOLEAN features to the boolean split branch."""
        X = numpy.array([0, 0, 0, 0, 1, 1, 1, 1], dtype=float).reshape(-1, 1)
        y = numpy.array([1, 1, 1, 1, 9, 9, 9, 9], dtype=float)
        weights = numpy.ones(8)
        feature_types = numpy.array([sigma._types.CovariateType.BOOLEAN])
        result = sigma._splitting.find_best_split(
            X,
            y.reshape(-1, 1),
            weights,
            0,
            feature_types,
            sigma._types.TestStat.QUADRATIC,
            min_buckets=1,
            correlation=sigma._types.Correlation.NORMAL,
        )
        assert result is not None
        criterion, statistic = result
        assert criterion is True
        assert statistic > 0

    def test_dispatches_to_integer(self):
        """Routes integer features to the left-observed-value branch."""
        X = numpy.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float).reshape(-1, 1)
        y = numpy.array([0, 0, 0, 0, 10, 10, 10, 10], dtype=float)
        weights = numpy.ones(8)
        feature_types = numpy.array([sigma._types.CovariateType.INTEGER])
        result = sigma._splitting.find_best_split(
            X,
            y.reshape(-1, 1),
            weights,
            0,
            feature_types,
            sigma._types.TestStat.QUADRATIC,
            min_buckets=1,
            correlation=sigma._types.Correlation.NORMAL,
        )
        assert result is not None
        split_value, statistic = result
        assert isinstance(split_value, sigma._splitting._NumericSplit)
        assert split_value.threshold == 4

    def test_dispatches_to_real(self):
        """Routes real-valued features to the midpoint branch."""
        X = numpy.array(
            [1.0, 2.0, 3.0, 4.0, 5.5, 6.0, 7.0, 8.0], dtype=float
        ).reshape(-1, 1)
        y = numpy.array([0, 0, 0, 0, 10, 10, 10, 10], dtype=float)
        weights = numpy.ones(8)
        feature_types = numpy.array([sigma._types.CovariateType.REAL])
        result = sigma._splitting.find_best_split(
            X,
            y.reshape(-1, 1),
            weights,
            0,
            feature_types,
            sigma._types.TestStat.QUADRATIC,
            min_buckets=1,
            correlation=sigma._types.Correlation.NORMAL,
        )
        assert result is not None
        split_value, statistic = result
        assert isinstance(split_value, sigma._splitting._NumericSplit)
        assert split_value.threshold == 4.75

    def test_dispatches_to_categorical(self):
        """Routes to categorical split for categorical features."""
        X = numpy.array([0, 0, 0, 1, 1, 1], dtype=float).reshape(-1, 1)
        y = numpy.array([1, 1, 1, 10, 10, 10], dtype=float)
        weights = numpy.ones(6)
        feature_types = numpy.array([sigma._types.CovariateType.CATEGORICAL])
        result = sigma._splitting.find_best_split(
            X,
            y.reshape(-1, 1),
            weights,
            0,
            feature_types,
            sigma._types.TestStat.QUADRATIC,
            min_buckets=1,
            correlation=sigma._types.Correlation.NORMAL,
        )
        assert result is not None
        split_value, statistic = result
        assert isinstance(split_value, frozenset)
        assert statistic > 0


class TestFindBestSplitNumericRank(unittest.TestCase):
    """Tests for numeric split search with rank correlation."""

    __slots__ = ()

    def test_step_function_finds_correct_threshold(self):
        """Finds the same split for a perfect step function."""
        X_j = numpy.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
        y = numpy.array([0, 0, 0, 0, 10, 10, 10, 10], dtype=float)
        weights = numpy.ones(8)
        result = sigma._splitting.find_best_split_numeric(
            X_j,
            y.reshape(-1, 1),
            weights,
            sigma._types.TestStat.QUADRATIC,
            min_buckets=1,
            is_integer=True,
            correlation=sigma._types.Correlation.RANK,
        )
        assert result is not None
        split, statistic = result
        assert split.threshold == 4
        assert statistic > 0

    def test_robust_to_response_outlier(self):
        """Rank-based split is not pulled by an extreme response."""
        X_j = numpy.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
        y = numpy.array([0, 0, 0, 0, 10, 10, 10, 1e6], dtype=float)
        weights = numpy.ones(8)
        result_rank = sigma._splitting.find_best_split_numeric(
            X_j,
            y.reshape(-1, 1),
            weights,
            sigma._types.TestStat.QUADRATIC,
            min_buckets=1,
            is_integer=True,
            correlation=sigma._types.Correlation.RANK,
        )
        result_normal = sigma._splitting.find_best_split_numeric(
            X_j,
            y.reshape(-1, 1),
            weights,
            sigma._types.TestStat.QUADRATIC,
            min_buckets=1,
            is_integer=True,
            correlation=sigma._types.Correlation.NORMAL,
        )
        assert result_rank is not None
        assert result_normal is not None
        assert result_rank[0].threshold == 4
