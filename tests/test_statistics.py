"""Unit tests for the statistical engine."""

import collections.abc
import itertools
import unittest

import numpy
import numpy.testing
import numpy.typing
import scipy.stats

import sigma._statistics
import sigma._types


# Hand-computed reference values for a simple numeric case:
#   g_j = [1, 2, 3, 4], h = [10, 20, 30, 40], weights = [1, 1, 1, 1]
#   w_. = 4, sum_wg = 10, E_h = 25, V_h = 125
#   G_quad = 30, G_outer = 100
#   T = 300, mu = 250, Sigma = 2500/3


def _moments(
    g_list: collections.abc.Sequence[numpy.typing.NDArray[numpy.floating]],
    h: numpy.typing.NDArray[numpy.floating],
    weights: numpy.typing.NDArray[numpy.floating],
) -> tuple[
    list[numpy.typing.NDArray[numpy.floating]],
    list[numpy.typing.NDArray[numpy.floating]],
]:
    """Pre-compute (mu_j, Sigma_j) per variable for Monte Carlo tests."""
    mu_list = [
        sigma._statistics.compute_conditional_expectation(g_j, h, weights)
        for g_j in g_list
    ]
    Sigma_list = [
        sigma._statistics.compute_conditional_covariance(g_j, h, weights)
        for g_j in g_list
    ]
    return mu_list, Sigma_list


class TestComputeLinearStatistic(unittest.TestCase):
    """Tests for the linear statistic T_j."""

    __slots__ = ()

    def test_numeric_covariate(self):
        """Computes the weighted sum of outer products for numeric data."""
        g_j = numpy.array([[1], [2], [3], [4]], dtype=float)
        h = numpy.array([[10], [20], [30], [40]], dtype=float)
        weights = numpy.ones(4)
        result = sigma._statistics.compute_linear_statistic(g_j, h, weights)
        numpy.testing.assert_allclose(result, numpy.array([300.0]))

    def test_categorical_covariate(self):
        """Computes correctly with one-hot encoded categorical data."""
        g_j = numpy.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)
        h = numpy.array([[10], [20], [30], [40]], dtype=float)
        weights = numpy.ones(4)
        result = sigma._statistics.compute_linear_statistic(g_j, h, weights)
        numpy.testing.assert_allclose(result, numpy.array([30.0, 70.0]))

    def test_nonuniform_weights(self):
        """Applies non-uniform weights correctly."""
        g_j = numpy.array([[1], [2]], dtype=float)
        h = numpy.array([[10], [20]], dtype=float)
        weights = numpy.array([2.0, 3.0])
        result = sigma._statistics.compute_linear_statistic(g_j, h, weights)
        expected = numpy.array([2.0 * 1 * 10 + 3.0 * 2 * 20])
        numpy.testing.assert_allclose(result, expected)

    def test_multivariate_response(self):
        """Handles multivariate influence function (q > 1)."""
        g_j = numpy.array([[1], [2]], dtype=float)
        h = numpy.array([[10, 100], [20, 200]], dtype=float)
        weights = numpy.ones(2)
        result = sigma._statistics.compute_linear_statistic(g_j, h, weights)
        # Matrix = g_j.T @ h = [[1,2]] @ [[10,100],[20,200]] = [[50, 500]]
        # vec (column-major) of (1, 2) matrix = [50, 500]
        numpy.testing.assert_allclose(result, numpy.array([50.0, 500.0]))


class TestComputeConditionalExpectation(unittest.TestCase):
    """Tests for the conditional expectation mu_j."""

    __slots__ = ()

    def test_numeric_covariate(self):
        """Computes mu_j for numeric covariate with uniform weights."""
        g_j = numpy.array([[1], [2], [3], [4]], dtype=float)
        h = numpy.array([[10], [20], [30], [40]], dtype=float)
        weights = numpy.ones(4)
        result = sigma._statistics.compute_conditional_expectation(
            g_j, h, weights
        )
        # sum_wg = 10, E_h = 25, mu = 250
        numpy.testing.assert_allclose(result, numpy.array([250.0]))

    def test_categorical_covariate(self):
        """Computes mu_j for categorical covariate."""
        g_j = numpy.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)
        h = numpy.array([[10], [20], [30], [40]], dtype=float)
        weights = numpy.ones(4)
        result = sigma._statistics.compute_conditional_expectation(
            g_j, h, weights
        )
        # sum_wg = [2, 2], E_h = 25, mu = [50, 50]
        numpy.testing.assert_allclose(result, numpy.array([50.0, 50.0]))

    def test_nonuniform_weights(self):
        """Applies non-uniform weights correctly."""
        g_j = numpy.array([[1], [2]], dtype=float)
        h = numpy.array([[10], [20]], dtype=float)
        weights = numpy.array([2.0, 3.0])
        # w_. = 5, sum_wg = 2*1 + 3*2 = 8
        # E_h = (2*10 + 3*20) / 5 = 80/5 = 16
        # mu = 8 * 16 = 128
        result = sigma._statistics.compute_conditional_expectation(
            g_j, h, weights
        )
        numpy.testing.assert_allclose(result, numpy.array([128.0]))


class TestComputeConditionalCovariance(unittest.TestCase):
    """Tests for the conditional covariance Sigma_j."""

    __slots__ = ()

    def test_numeric_covariate(self):
        """Computes Sigma_j for numeric covariate with uniform weights."""
        g_j = numpy.array([[1], [2], [3], [4]], dtype=float)
        h = numpy.array([[10], [20], [30], [40]], dtype=float)
        weights = numpy.ones(4)
        result = sigma._statistics.compute_conditional_covariance(
            g_j, h, weights
        )
        numpy.testing.assert_allclose(result, numpy.array([[2500.0 / 3.0]]))

    def test_constant_response(self):
        """Returns zero covariance when response is constant."""
        g_j = numpy.array([[1], [2], [3]], dtype=float)
        h = numpy.array([[5], [5], [5]], dtype=float)
        weights = numpy.ones(3)
        result = sigma._statistics.compute_conditional_covariance(
            g_j, h, weights
        )
        numpy.testing.assert_allclose(result, numpy.zeros((1, 1)))

    def test_singular_categorical(self):
        """Produces a singular covariance for one-hot encoding."""
        g_j = numpy.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)
        h = numpy.array([[10], [20], [30], [40]], dtype=float)
        weights = numpy.ones(4)
        result = sigma._statistics.compute_conditional_covariance(
            g_j, h, weights
        )
        # V_h = 125, w_.*G_quad - G_outer = 4*[[2,0],[0,2]] - [[4,4],[4,4]]
        #   = [[4,-4],[-4,4]]
        # Sigma = (125/3) * [[4,-4],[-4,4]]
        expected = (125.0 / 3.0) * numpy.array([[4, -4], [-4, 4]])
        numpy.testing.assert_allclose(result, expected)

    def test_nonuniform_weights(self):
        """Computes covariance with non-uniform weights."""
        g_j = numpy.array([[1], [2]], dtype=float)
        h = numpy.array([[10], [20]], dtype=float)
        weights = numpy.array([2.0, 3.0])
        # w_. = 5, E_h = 16
        # V_h = (1/5) * (2*(10-16)^2 + 3*(20-16)^2)
        #      = (1/5) * (72 + 48) = 24
        # G_quad = 2*1 + 3*4 = 14, G_outer = 8^2 = 64
        # inner = 5*14 - 64 = 6
        # Sigma = (1/4) * 24 * 6 = 36
        result = sigma._statistics.compute_conditional_covariance(
            g_j, h, weights
        )
        numpy.testing.assert_allclose(result, numpy.array([[36.0]]))


class TestComputeTestStatistic(unittest.TestCase):
    """Tests for the test statistic computation."""

    __slots__ = ()

    def test_quadratic_numeric(self):
        """Computes quadratic-form statistic for 1D case."""
        T = numpy.array([300.0])
        mu = numpy.array([250.0])
        Sigma = numpy.array([[2500.0 / 3.0]])
        result = sigma._statistics.compute_test_statistic(
            T, mu, Sigma, sigma._types.TestStat.QUADRATIC
        )
        numpy.testing.assert_allclose(result, 3.0)

    def test_maximum_numeric(self):
        """Computes maximum-type statistic for 1D case."""
        T = numpy.array([300.0])
        mu = numpy.array([250.0])
        Sigma = numpy.array([[2500.0 / 3.0]])
        result = sigma._statistics.compute_test_statistic(
            T, mu, Sigma, sigma._types.TestStat.MAXIMUM
        )
        expected = 50.0 / numpy.sqrt(2500.0 / 3.0)
        numpy.testing.assert_allclose(result, expected)

    def test_quadratic_singular_covariance(self):
        """Handles singular covariance via pseudo-inverse."""
        T = numpy.array([30.0, 70.0])
        mu = numpy.array([50.0, 50.0])
        Sigma = (125.0 / 3.0) * numpy.array([[4, -4], [-4, 4]])
        result = sigma._statistics.compute_test_statistic(
            T, mu, Sigma, sigma._types.TestStat.QUADRATIC
        )
        numpy.testing.assert_allclose(result, 2.4)

    def test_maximum_skips_zero_variance(self):
        """Skips dimensions with zero diagonal in max-type statistic."""
        T = numpy.array([5.0, 0.0])
        mu = numpy.array([0.0, 0.0])
        Sigma = numpy.array([[4.0, 0.0], [0.0, 0.0]])
        result = sigma._statistics.compute_test_statistic(
            T, mu, Sigma, sigma._types.TestStat.MAXIMUM
        )
        numpy.testing.assert_allclose(result, 2.5)

    def test_maximum_multivariate(self):
        """Takes the maximum across valid dimensions."""
        T = numpy.array([6.0, 10.0])
        mu = numpy.array([0.0, 0.0])
        Sigma = numpy.array([[4.0, 0.0], [0.0, 25.0]])
        result = sigma._statistics.compute_test_statistic(
            T, mu, Sigma, sigma._types.TestStat.MAXIMUM
        )
        # |6|/sqrt(4) = 3.0, |10|/sqrt(25) = 2.0 -> max = 3.0
        numpy.testing.assert_allclose(result, 3.0)


class TestComputePValue(unittest.TestCase):
    """Tests for p-value computation."""

    __slots__ = ()

    def test_quadratic_known_value(self):
        """Matches scipy chi2.sf for rank-1 covariance."""
        statistic = 3.0
        Sigma = numpy.array([[2500.0 / 3.0]])
        result = sigma._statistics.compute_p_value(
            statistic, Sigma, sigma._types.TestStat.QUADRATIC
        )
        expected = float(scipy.stats.chi2.sf(3.0, df=1))
        numpy.testing.assert_allclose(result, expected)

    def test_maximum_one_dimension(self):
        """Matches two-sided normal test for scalar case."""
        c = numpy.sqrt(3.0)
        Sigma = numpy.array([[2500.0 / 3.0]])
        result = sigma._statistics.compute_p_value(
            c, Sigma, sigma._types.TestStat.MAXIMUM
        )
        expected = 2.0 * float(scipy.stats.norm.sf(c))
        numpy.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_quadratic_zero_statistic(self):
        """Returns p-value of 1.0 when statistic is zero."""
        result = sigma._statistics.compute_p_value(
            0.0, numpy.eye(2), sigma._types.TestStat.QUADRATIC
        )
        numpy.testing.assert_allclose(result, 1.0)

    def test_quadratic_singular_covariance(self):
        """Uses rank of Sigma as degrees of freedom."""
        Sigma = (125.0 / 3.0) * numpy.array([[4, -4], [-4, 4]])
        statistic = 2.4
        result = sigma._statistics.compute_p_value(
            statistic, Sigma, sigma._types.TestStat.QUADRATIC
        )
        expected = float(scipy.stats.chi2.sf(2.4, df=1))
        numpy.testing.assert_allclose(result, expected)

    def test_maximum_multivariate(self):
        """Computes multivariate normal rectangular probability."""
        Sigma = numpy.array([[4.0, 1.0], [1.0, 4.0]])
        c = 2.0
        result = sigma._statistics.compute_p_value(
            c, Sigma, sigma._types.TestStat.MAXIMUM
        )
        R = numpy.array([[1.0, 0.25], [0.25, 1.0]])
        rv = scipy.stats.multivariate_normal(mean=[0, 0], cov=R)
        prob = 0.0
        for bits in itertools.product([0, 1], repeat=2):
            point = [-c if bits[k] else c for k in range(2)]
            sign = (-1) ** sum(bits)
            prob += sign * rv.cdf(point)
        expected = 1.0 - prob
        numpy.testing.assert_allclose(result, expected, atol=2e-3)

    def test_maximum_zero_variance(self):
        """Returns p-value of 1.0 when all variances are zero."""
        Sigma = numpy.zeros((2, 2))
        result = sigma._statistics.compute_p_value(
            1.0, Sigma, sigma._types.TestStat.MAXIMUM
        )
        numpy.testing.assert_allclose(result, 1.0)

    def test_maximum_singular_covariance(self):
        """Handles rank-deficient Sigma without raising."""
        g_j = numpy.array(
            [
                [1, 0, 0],
                [1, 0, 0],
                [0, 1, 0],
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 1],
            ],
            dtype=float,
        )
        h = numpy.array([[10], [20], [30], [40], [50], [60]], dtype=float)
        weights = numpy.ones(6)
        Sigma = sigma._statistics.compute_conditional_covariance(
            g_j, h, weights
        )
        self.assertEqual(numpy.linalg.matrix_rank(Sigma), 2)
        result = sigma._statistics.compute_p_value(
            1.5, Sigma, sigma._types.TestStat.MAXIMUM
        )
        assert numpy.isfinite(result)
        assert 0.0 < result <= 1.0


class TestComputePValueMaximumMultivariate(unittest.TestCase):
    """Correctness tests for the maximum-type multivariate p-value."""

    __slots__ = ()

    def test_matches_corner_sum_on_full_rank(self):
        """Matches the 2^d inclusion-exclusion reference on full-rank Sigma."""
        cases = [
            (2, numpy.array([[4.0, 0.0], [0.0, 4.0]]), 1.5),
            (2, numpy.array([[4.0, 1.0], [1.0, 4.0]]), 2.0),
            (
                3,
                numpy.array(
                    [
                        [4.0, 0.0, 0.0],
                        [0.0, 9.0, 0.0],
                        [0.0, 0.0, 16.0],
                    ]
                ),
                1.5,
            ),
            (
                3,
                numpy.array(
                    [
                        [4.0, 1.0, 0.5],
                        [1.0, 9.0, 1.5],
                        [0.5, 1.5, 16.0],
                    ]
                ),
                1.8,
            ),
            (
                4,
                numpy.array(
                    [
                        [1.0, 0.5, 0.25, 0.125],
                        [0.5, 1.0, 0.5, 0.25],
                        [0.25, 0.5, 1.0, 0.5],
                        [0.125, 0.25, 0.5, 1.0],
                    ]
                ),
                1.5,
            ),
            (5, numpy.eye(5) * 2.0, 2.0),
        ]
        for d, Sigma, c in cases:
            with self.subTest(d=d, c=c):
                result = sigma._statistics.compute_p_value(
                    c, Sigma, sigma._types.TestStat.MAXIMUM
                )
                sd = numpy.sqrt(numpy.diag(Sigma))
                R = Sigma / numpy.outer(sd, sd)
                rv = scipy.stats.multivariate_normal(mean=numpy.zeros(d), cov=R)
                prob = 0.0
                for bits in itertools.product([0, 1], repeat=d):
                    point = [-c if bits[k] else c for k in range(d)]
                    sign = (-1) ** sum(bits)
                    prob += sign * rv.cdf(point)
                expected = 1.0 - prob
                numpy.testing.assert_allclose(result, expected, rtol=1e-4)

    def test_singular_two_level_matches_analytical(self):
        """Matches 2*Phi(-c) for a 2-level one-hot whose Sigma has rank 1."""
        g_j = numpy.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=float)
        h = numpy.array([[10], [20], [30], [40]], dtype=float)
        weights = numpy.ones(4)
        Sigma = sigma._statistics.compute_conditional_covariance(
            g_j, h, weights
        )
        self.assertEqual(numpy.linalg.matrix_rank(Sigma), 1)
        for c in [0.5, 1.0, 1.5, 2.0, 3.0]:
            with self.subTest(c=c):
                result = sigma._statistics.compute_p_value(
                    c, Sigma, sigma._types.TestStat.MAXIMUM
                )
                expected = 2.0 * float(scipy.stats.norm.sf(c))
                numpy.testing.assert_allclose(result, expected, rtol=1e-5)

    def test_singular_three_level_matches_monte_carlo(self):
        """Matches a Monte Carlo estimate for a 3-level one-hot."""
        Sigma, R = _build_one_hot_sigma_and_correlation(num_levels=3, seed=0)
        self.assertEqual(numpy.linalg.matrix_rank(Sigma), 2)
        samples = _sample_singular_mvn(R, n_samples=200_000, seed=42)
        for c in [1.0, 1.5, 2.0]:
            with self.subTest(c=c):
                result = sigma._statistics.compute_p_value(
                    c, Sigma, sigma._types.TestStat.MAXIMUM
                )
                p_mc = float(
                    numpy.mean(numpy.max(numpy.abs(samples), axis=1) > c)
                )
                tolerance = max(
                    4.0 * numpy.sqrt(p_mc * (1 - p_mc) / samples.shape[0]),
                    1e-3,
                )
                assert abs(result - p_mc) < tolerance

    def test_singular_five_level_matches_monte_carlo(self):
        """Matches a Monte Carlo estimate for a 5-level one-hot."""
        Sigma, R = _build_one_hot_sigma_and_correlation(num_levels=5, seed=1)
        self.assertEqual(numpy.linalg.matrix_rank(Sigma), 4)
        samples = _sample_singular_mvn(R, n_samples=200_000, seed=43)
        for c in [1.5, 2.0, 2.5]:
            with self.subTest(c=c):
                result = sigma._statistics.compute_p_value(
                    c, Sigma, sigma._types.TestStat.MAXIMUM
                )
                p_mc = float(
                    numpy.mean(numpy.max(numpy.abs(samples), axis=1) > c)
                )
                tolerance = max(
                    4.0 * numpy.sqrt(p_mc * (1 - p_mc) / samples.shape[0]),
                    1e-3,
                )
                assert abs(result - p_mc) < tolerance

    def test_zero_statistic_returns_one(self):
        """Returns 1.0 when the statistic is zero, for full-rank and singular Sigma."""
        Sigma_full = numpy.array([[4.0, 1.0], [1.0, 4.0]])
        result_full = sigma._statistics.compute_p_value(
            0.0, Sigma_full, sigma._types.TestStat.MAXIMUM
        )
        numpy.testing.assert_allclose(result_full, 1.0)
        Sigma_singular = (125.0 / 3.0) * numpy.array([[4.0, -4.0], [-4.0, 4.0]])
        result_singular = sigma._statistics.compute_p_value(
            0.0, Sigma_singular, sigma._types.TestStat.MAXIMUM
        )
        numpy.testing.assert_allclose(result_singular, 1.0)

    def test_large_statistic_returns_near_zero(self):
        """Returns ~0 for very large statistic, for full-rank and singular Sigma."""
        Sigma_full = numpy.array([[4.0, 1.0], [1.0, 4.0]])
        result_full = sigma._statistics.compute_p_value(
            50.0, Sigma_full, sigma._types.TestStat.MAXIMUM
        )
        assert result_full < 1e-10
        Sigma_singular = (125.0 / 3.0) * numpy.array([[4.0, -4.0], [-4.0, 4.0]])
        result_singular = sigma._statistics.compute_p_value(
            50.0, Sigma_singular, sigma._types.TestStat.MAXIMUM
        )
        assert result_singular < 1e-10

    def test_monotone_in_statistic(self):
        """Strictly larger statistic yields strictly smaller p-value."""
        Sigma, _ = _build_one_hot_sigma_and_correlation(num_levels=4, seed=2)
        c_values = [0.5, 1.0, 1.5, 2.0, 2.5]
        p_values = [
            sigma._statistics.compute_p_value(
                c, Sigma, sigma._types.TestStat.MAXIMUM
            )
            for c in c_values
        ]
        for k in range(len(p_values) - 1):
            assert p_values[k] > p_values[k + 1]

    def test_high_dimensional_singular_completes(self):
        """Returns a finite probability for a 20-level one-hot Sigma."""
        K = 20
        Sigma, _ = _build_one_hot_sigma_and_correlation(num_levels=K, seed=3)
        self.assertEqual(numpy.linalg.matrix_rank(Sigma), K - 1)
        result = sigma._statistics.compute_p_value(
            1.5, Sigma, sigma._types.TestStat.MAXIMUM
        )
        assert numpy.isfinite(result)
        assert 0.0 < result <= 1.0


def _build_one_hot_sigma_and_correlation(
    num_levels: int, seed: int
) -> tuple[
    numpy.typing.NDArray[numpy.floating],
    numpy.typing.NDArray[numpy.floating],
]:
    """Build a singular one-hot Sigma plus its standardized correlation R."""
    rng = numpy.random.default_rng(seed)
    samples_per_level = 6
    n = num_levels * samples_per_level
    cat = numpy.repeat(numpy.arange(num_levels), samples_per_level).astype(
        float
    )
    rng.shuffle(cat)
    g_j = (cat[:, None] == numpy.arange(num_levels)).astype(float)
    h = rng.standard_normal((n, 1))
    weights = numpy.ones(n)
    Sigma = sigma._statistics.compute_conditional_covariance(g_j, h, weights)
    sd = numpy.sqrt(numpy.diag(Sigma))
    R = Sigma / numpy.outer(sd, sd)
    return Sigma, R


def _sample_singular_mvn(
    R: numpy.typing.NDArray[numpy.floating], n_samples: int, seed: int
) -> numpy.typing.NDArray[numpy.floating]:
    """Draw samples from MVN(0, R) via eigendecomposition with clipped eigenvalues."""
    rng = numpy.random.default_rng(seed)
    eigenvalues, eigenvectors = numpy.linalg.eigh(R)
    eigenvalues_clipped = numpy.maximum(eigenvalues, 0.0)
    sqrt_factor = eigenvectors * numpy.sqrt(eigenvalues_clipped)
    standard_samples = rng.standard_normal((n_samples, R.shape[0]))
    samples = standard_samples @ sqrt_factor.T
    return samples


def _assert_selects_associated_variable(test_type, **kwargs):
    """Assert select_variable picks the signal feature under test_type."""
    rng = numpy.random.default_rng(42)
    n = 200
    x_signal = numpy.linspace(0, 10, n)
    x_noise = rng.standard_normal(n)
    X = numpy.column_stack([x_noise, x_signal])
    y = 3.0 * x_signal + rng.standard_normal(n) * 0.1
    weights = numpy.ones(n)
    feature_types = numpy.array(
        [sigma._types.CovariateType.REAL, sigma._types.CovariateType.REAL]
    )
    result = sigma._statistics.select_variable(
        X,
        y.reshape(-1, 1),
        weights,
        feature_types,
        sigma._types.TestStat.QUADRATIC,
        test_type,
        correlation=sigma._types.Correlation.NORMAL,
        **kwargs,
    )
    assert result is not None
    assert result.feature_index == 1
    assert result.p_value < 0.05


def _assert_returns_none_when_independent(test_type, **kwargs):
    """Assert select_variable returns None on noise features under test_type."""
    rng = numpy.random.default_rng(123)
    n = 50
    X = rng.standard_normal((n, 3))
    y = rng.standard_normal(n)
    weights = numpy.ones(n)
    feature_types = numpy.array(
        [
            sigma._types.CovariateType.REAL,
            sigma._types.CovariateType.REAL,
            sigma._types.CovariateType.REAL,
        ]
    )
    result = sigma._statistics.select_variable(
        X,
        y.reshape(-1, 1),
        weights,
        feature_types,
        sigma._types.TestStat.QUADRATIC,
        test_type,
        correlation=sigma._types.Correlation.NORMAL,
        **kwargs,
    )
    assert result is None


class TestSelectVariable(unittest.TestCase):
    """Tests for variable selection."""

    __slots__ = ()

    def test_selects_associated_variable(self):
        """Selects the variable with the strongest association."""
        _assert_selects_associated_variable(sigma._types.TestType.SIDAK)

    def test_returns_none_when_independent(self):
        """Returns None when no variable is significantly associated."""
        _assert_returns_none_when_independent(sigma._types.TestType.SIDAK)

    def test_categorical_variable(self):
        """Correctly handles a categorical variable with strong signal."""
        rng = numpy.random.default_rng(7)
        n = 200
        categories = rng.integers(0, 3, size=n).astype(float)
        y = numpy.where(
            categories == 0,
            10.0,
            numpy.where(categories == 1, 20.0, 30.0),
        )
        y = y + rng.standard_normal(n) * 0.5
        X = categories.reshape(-1, 1)
        weights = numpy.ones(n)
        feature_types = numpy.array([sigma._types.CovariateType.CATEGORICAL])
        result = sigma._statistics.select_variable(
            X,
            y.reshape(-1, 1),
            weights,
            feature_types,
            sigma._types.TestStat.QUADRATIC,
            sigma._types.TestType.SIDAK,
            correlation=sigma._types.Correlation.NORMAL,
        )
        assert result is not None
        assert result[0] == 0
        assert result[1] < 0.05

    def test_maximum_test_stat(self):
        """Works with maximum-type test statistic."""
        rng = numpy.random.default_rng(42)
        n = 200
        x_signal = numpy.linspace(0, 10, n)
        X = x_signal.reshape(-1, 1)
        y = 3.0 * x_signal + rng.standard_normal(n) * 0.1
        weights = numpy.ones(n)
        feature_types = numpy.array([sigma._types.CovariateType.REAL])
        result = sigma._statistics.select_variable(
            X,
            y.reshape(-1, 1),
            weights,
            feature_types,
            sigma._types.TestStat.MAXIMUM,
            sigma._types.TestType.SIDAK,
            correlation=sigma._types.Correlation.NORMAL,
        )
        assert result is not None
        assert result[0] == 0
        assert result[1] < 0.05

    def test_categorical_with_maximum_test_stat(self):
        """Handles categorical covariates with maximum-type test statistic."""
        rng = numpy.random.default_rng(7)
        n = 200
        categories = rng.integers(0, 3, size=n).astype(float)
        y = numpy.where(
            categories == 0,
            10.0,
            numpy.where(categories == 1, 20.0, 30.0),
        )
        y = y + rng.standard_normal(n) * 0.5
        X = categories.reshape(-1, 1)
        weights = numpy.ones(n)
        feature_types = numpy.array([sigma._types.CovariateType.CATEGORICAL])
        result = sigma._statistics.select_variable(
            X,
            y.reshape(-1, 1),
            weights,
            feature_types,
            sigma._types.TestStat.MAXIMUM,
            sigma._types.TestType.SIDAK,
            correlation=sigma._types.Correlation.NORMAL,
        )
        assert result is not None
        assert result[0] == 0
        assert result[1] < 0.05

    def test_respects_weights(self):
        """Only considers samples with positive weights."""
        rng = numpy.random.default_rng(42)
        n = 200
        x = numpy.linspace(0, 10, n)
        X = x.reshape(-1, 1)
        y = 3.0 * x + rng.standard_normal(n) * 0.1
        weights = numpy.ones(n)
        # Zero out weights for all but a tiny random subset
        weights[: n - 3] = 0.0
        feature_types = numpy.array([sigma._types.CovariateType.REAL])
        result = sigma._statistics.select_variable(
            X,
            y.reshape(-1, 1),
            weights,
            feature_types,
            sigma._types.TestStat.QUADRATIC,
            sigma._types.TestType.SIDAK,
            correlation=sigma._types.Correlation.NORMAL,
        )
        # With only 3 samples, insufficient evidence
        assert result is None


class TestAdjustPValues(unittest.TestCase):
    """Tests for p-value multiplicity adjustment."""

    __slots__ = ()

    def test_sidak_formula(self):
        """Sidak adjustment matches the exact formula 1 - (1 - p)^m."""
        p_values = numpy.array([0.01, 0.05, 0.2])
        m = len(p_values)
        result = sigma._statistics._adjust_p_values(
            p_values, sigma._types.TestType.SIDAK
        )
        expected = 1.0 - (1.0 - p_values) ** m
        numpy.testing.assert_allclose(result, expected)

    def test_sidak_at_zero(self):
        """Sidak returns 0.0 for a raw p-value of 0."""
        p_values = numpy.array([0.0, 0.05])
        result = sigma._statistics._adjust_p_values(
            p_values, sigma._types.TestType.SIDAK
        )
        numpy.testing.assert_allclose(result[0], 0.0)

    def test_sidak_at_one(self):
        """Sidak returns 1.0 for a raw p-value of 1."""
        p_values = numpy.array([0.05, 1.0])
        result = sigma._statistics._adjust_p_values(
            p_values, sigma._types.TestType.SIDAK
        )
        numpy.testing.assert_allclose(result[1], 1.0)

    def test_bonferroni_formula(self):
        """Bonferroni adjustment matches the exact formula min(m * p, 1)."""
        p_values = numpy.array([0.01, 0.04, 0.1])
        m = len(p_values)
        result = sigma._statistics._adjust_p_values(
            p_values, sigma._types.TestType.BONFERRONI
        )
        expected = numpy.minimum(p_values * m, 1.0)
        numpy.testing.assert_allclose(result, expected)

    def test_bonferroni_at_zero(self):
        """Bonferroni returns 0.0 for a raw p-value of 0."""
        p_values = numpy.array([0.0, 0.05])
        result = sigma._statistics._adjust_p_values(
            p_values, sigma._types.TestType.BONFERRONI
        )
        numpy.testing.assert_allclose(result[0], 0.0)

    def test_bonferroni_at_one(self):
        """Bonferroni returns 1.0 for a raw p-value of 1."""
        p_values = numpy.array([0.05, 1.0])
        result = sigma._statistics._adjust_p_values(
            p_values, sigma._types.TestType.BONFERRONI
        )
        numpy.testing.assert_allclose(result[1], 1.0)

    def test_bonferroni_clamped_to_one(self):
        """Bonferroni clamps adjusted p-values at 1.0 when m * p exceeds 1."""
        p_values = numpy.array([0.5, 0.8])
        result = sigma._statistics._adjust_p_values(
            p_values, sigma._types.TestType.BONFERRONI
        )
        numpy.testing.assert_allclose(result, numpy.array([1.0, 1.0]))

    def test_bonferroni_dominates_sidak_dominates_raw(self):
        """Bonferroni is >= Sidak is >= raw element-wise, and bounded by 1."""
        p_values = numpy.array([0.001, 0.01, 0.05, 0.2, 0.5])
        bonferroni = sigma._statistics._adjust_p_values(
            p_values, sigma._types.TestType.BONFERRONI
        )
        sidak = sigma._statistics._adjust_p_values(
            p_values, sigma._types.TestType.SIDAK
        )
        assert numpy.all(p_values <= sidak)
        assert numpy.all(sidak <= bonferroni)
        assert numpy.all(bonferroni <= 1.0)


class TestSelectVariableSidak(unittest.TestCase):
    """Tests for variable selection with Sidak adjustment."""

    __slots__ = ()

    def test_selects_associated_variable(self):
        """Selects the variable with the strongest association."""
        _assert_selects_associated_variable(sigma._types.TestType.SIDAK)

    def test_returns_none_when_independent(self):
        """Returns None when no variable is significantly associated."""
        _assert_returns_none_when_independent(sigma._types.TestType.SIDAK)


class TestSelectVariableBonferroni(unittest.TestCase):
    """Tests for variable selection with Bonferroni adjustment."""

    __slots__ = ()

    def test_selects_associated_variable(self):
        """Selects the variable with the strongest association."""
        _assert_selects_associated_variable(sigma._types.TestType.BONFERRONI)

    def test_returns_none_when_independent(self):
        """Returns None when no variable is significantly associated."""
        _assert_returns_none_when_independent(sigma._types.TestType.BONFERRONI)


class TestRankTransform(unittest.TestCase):
    """Tests for the rank transformation helper."""

    __slots__ = ()

    def test_uniform_weights(self):
        """Ranks all samples when all weights are positive."""
        matrix = numpy.array([[3], [1], [2]], dtype=float)
        weights = numpy.ones(3)
        result = sigma._statistics._rank_transform(matrix, weights)
        numpy.testing.assert_allclose(
            result, numpy.array([[3], [1], [2]], dtype=float)
        )

    def test_zero_weight_excluded(self):
        """Assigns zero to samples with zero weight."""
        matrix = numpy.array([[10], [1], [5]], dtype=float)
        weights = numpy.array([0.0, 1.0, 1.0])
        result = sigma._statistics._rank_transform(matrix, weights)
        numpy.testing.assert_allclose(result[0], [0.0])
        numpy.testing.assert_allclose(result[1], [1.0])
        numpy.testing.assert_allclose(result[2], [2.0])

    def test_ties(self):
        """Uses average ranking for tied values."""
        matrix = numpy.array([[5], [5], [1]], dtype=float)
        weights = numpy.ones(3)
        result = sigma._statistics._rank_transform(matrix, weights)
        numpy.testing.assert_allclose(
            result, numpy.array([[2.5], [2.5], [1.0]])
        )

    def test_multiple_columns(self):
        """Ranks each column independently."""
        matrix = numpy.array([[3, 1], [1, 3], [2, 2]], dtype=float)
        weights = numpy.ones(3)
        result = sigma._statistics._rank_transform(matrix, weights)
        numpy.testing.assert_allclose(result[:, 0], [3.0, 1.0, 2.0])
        numpy.testing.assert_allclose(result[:, 1], [1.0, 3.0, 2.0])


class TestSelectVariableRank(unittest.TestCase):
    """Tests for variable selection with rank correlation."""

    __slots__ = ()

    def test_selects_associated_variable(self):
        """Selects the signal variable under rank correlation."""
        rng = numpy.random.default_rng(42)
        n = 200
        x_signal = numpy.linspace(0, 10, n)
        x_noise = rng.standard_normal(n)
        X = numpy.column_stack([x_noise, x_signal])
        y = 3.0 * x_signal + rng.standard_normal(n) * 0.1
        weights = numpy.ones(n)
        feature_types = numpy.array(
            [sigma._types.CovariateType.REAL, sigma._types.CovariateType.REAL]
        )
        result = sigma._statistics.select_variable(
            X,
            y.reshape(-1, 1),
            weights,
            feature_types,
            sigma._types.TestStat.QUADRATIC,
            sigma._types.TestType.SIDAK,
        )
        assert result is not None
        assert result.feature_index == 1
        assert result.p_value < 0.05

    def test_robust_to_outlier(self):
        """Rank mode resists a single extreme outlier."""
        rng = numpy.random.default_rng(99)
        n = 100
        x_real = numpy.linspace(0, 5, n)
        x_outlier = rng.standard_normal(n)
        x_outlier[0] = 1e6
        X = numpy.column_stack([x_outlier, x_real])
        y = 2.0 * x_real + rng.standard_normal(n) * 0.5
        y[0] = 1e6
        weights = numpy.ones(n)
        feature_types = numpy.array(
            [sigma._types.CovariateType.REAL, sigma._types.CovariateType.REAL]
        )
        result_rank = sigma._statistics.select_variable(
            X,
            y.reshape(-1, 1),
            weights,
            feature_types,
            sigma._types.TestStat.QUADRATIC,
            sigma._types.TestType.SIDAK,
        )
        assert result_rank is not None
        assert result_rank[0] == 1

    def test_categorical_variable_rank(self):
        """Handles categorical variable with rank correlation."""
        rng = numpy.random.default_rng(7)
        n = 200
        categories = rng.integers(0, 3, size=n).astype(float)
        y = numpy.where(
            categories == 0,
            10.0,
            numpy.where(categories == 1, 20.0, 30.0),
        )
        y = y + rng.standard_normal(n) * 0.5
        X = categories.reshape(-1, 1)
        weights = numpy.ones(n)
        feature_types = numpy.array([sigma._types.CovariateType.CATEGORICAL])
        result = sigma._statistics.select_variable(
            X,
            y.reshape(-1, 1),
            weights,
            feature_types,
            sigma._types.TestStat.QUADRATIC,
            sigma._types.TestType.SIDAK,
        )
        assert result is not None
        assert result[0] == 0
        assert result[1] < 0.05


class TestAdjustPValuesMonteCarlo(unittest.TestCase):
    """Tests for Westfall-Young min-P adjusted p-values."""

    __slots__ = ()

    def test_monte_carlo_raises_in_adjust_p_values(self):
        """Raises ValueError when called on _adjust_p_values directly."""
        p_values = numpy.array([0.01, 0.05])
        with self.assertRaises(ValueError):
            sigma._statistics._adjust_p_values(
                p_values, sigma._types.TestType.MONTE_CARLO
            )

    def test_minimum_adjusted_p(self):
        """Adjusted p-value is at least 1/(B+1) due to +1 correction."""
        rng = numpy.random.default_rng(0)
        n = 100
        x = numpy.linspace(0, 10, n)
        g_list = [x.reshape(-1, 1)]
        h = (3.0 * x).reshape(-1, 1)
        weights = numpy.ones(n)
        mu_list, Sigma_list = _moments(g_list, h, weights)
        p_values_obs = numpy.array([1e-20])
        resamples = 99
        result = sigma._statistics._adjust_p_values_monte_carlo(
            p_values_obs,
            g_list,
            h,
            weights,
            mu_list,
            Sigma_list,
            sigma._types.TestStat.QUADRATIC,
            resamples,
            rng,
        )
        assert result[0] >= 1.0 / (resamples + 1)

    def test_null_signal_yields_large_adjusted_p(self):
        """Constant response gives adjusted p-values of 1.0."""
        rng = numpy.random.default_rng(1)
        n = 50
        x = numpy.linspace(0, 5, n)
        g_list = [x.reshape(-1, 1)]
        h = numpy.ones((n, 1))
        weights = numpy.ones(n)
        mu_list, Sigma_list = _moments(g_list, h, weights)
        p_values_obs = numpy.array([1.0])
        result = sigma._statistics._adjust_p_values_monte_carlo(
            p_values_obs,
            g_list,
            h,
            weights,
            mu_list,
            Sigma_list,
            sigma._types.TestStat.QUADRATIC,
            99,
            rng,
        )
        numpy.testing.assert_allclose(result[0], 1.0)

    def test_strong_signal_yields_small_adjusted_p(self):
        """Near-zero observed p-value gives small adjusted p-value."""
        rng = numpy.random.default_rng(2)
        n = 200
        x = numpy.linspace(0, 10, n)
        g_list = [x.reshape(-1, 1)]
        h = (5.0 * x).reshape(-1, 1)
        weights = numpy.ones(n)
        mu_list, Sigma_list = _moments(g_list, h, weights)
        p_values_obs = numpy.array([1e-30])
        result = sigma._statistics._adjust_p_values_monte_carlo(
            p_values_obs,
            g_list,
            h,
            weights,
            mu_list,
            Sigma_list,
            sigma._types.TestStat.QUADRATIC,
            499,
            rng,
        )
        assert result[0] < 0.05

    def test_monotone_in_observed_p(self):
        """Larger observed p-values give larger or equal adjusted p-values."""
        rng = numpy.random.default_rng(3)
        n = 100
        x1 = numpy.linspace(0, 10, n)
        x2 = rng.standard_normal(n)
        g_list = [x1.reshape(-1, 1), x2.reshape(-1, 1)]
        h = (2.0 * x1).reshape(-1, 1)
        weights = numpy.ones(n)
        mu_list, Sigma_list = _moments(g_list, h, weights)
        p_values_obs = numpy.sort(numpy.array([0.001, 0.1]))
        result = sigma._statistics._adjust_p_values_monte_carlo(
            p_values_obs,
            g_list,
            h,
            weights,
            mu_list,
            Sigma_list,
            sigma._types.TestStat.QUADRATIC,
            499,
            rng,
        )
        assert result[0] <= result[1]


class TestSelectVariableMonteCarlo(unittest.TestCase):
    """Tests for variable selection with min-P resampling adjustment."""

    __slots__ = ()

    def test_selects_associated_variable(self):
        """Selects the signal variable with monte_carlo adjustment."""
        _assert_selects_associated_variable(
            sigma._types.TestType.MONTE_CARLO,
            resamples=499,
            rng=numpy.random.default_rng(0),
        )

    def test_returns_none_when_independent(self):
        """Returns None when no variable is associated under monte_carlo."""
        _assert_returns_none_when_independent(
            sigma._types.TestType.MONTE_CARLO,
            resamples=499,
            rng=numpy.random.default_rng(0),
        )

    def test_reproducible_with_same_seed(self):
        """Two calls with the same RNG seed return identical results."""
        rng = numpy.random.default_rng(42)
        n = 100
        x_signal = numpy.linspace(0, 10, n)
        x_noise = rng.standard_normal(n)
        X = numpy.column_stack([x_noise, x_signal])
        y = 3.0 * x_signal + rng.standard_normal(n) * 0.1
        weights = numpy.ones(n)
        feature_types = numpy.array(
            [sigma._types.CovariateType.REAL, sigma._types.CovariateType.REAL]
        )
        args = (
            X,
            y.reshape(-1, 1),
            weights,
            feature_types,
            sigma._types.TestStat.QUADRATIC,
            sigma._types.TestType.MONTE_CARLO,
            0.05,
            sigma._types.Correlation.NORMAL,
            99,
        )
        result1 = sigma._statistics.select_variable(
            *args,
            rng=numpy.random.default_rng(7),
        )
        result2 = sigma._statistics.select_variable(
            *args,
            rng=numpy.random.default_rng(7),
        )
        assert result1 is not None
        assert result2 is not None
        assert result1[0] == result2[0]
        numpy.testing.assert_allclose(result1[1], result2[1])

    def test_raises_without_rng(self):
        """Raises ValueError when rng is None for MONTE_CARLO."""
        X = numpy.array([[1.0], [2.0], [3.0]])
        h = numpy.array([[1.0], [2.0], [3.0]])
        weights = numpy.ones(3)
        feature_types = numpy.array([sigma._types.CovariateType.REAL])
        with self.assertRaises(ValueError):
            sigma._statistics.select_variable(
                X,
                h,
                weights,
                feature_types,
                sigma._types.TestStat.QUADRATIC,
                sigma._types.TestType.MONTE_CARLO,
                resamples=99,
                rng=None,
            )

    def test_raises_without_resamples(self):
        """Raises ValueError when resamples is None for MONTE_CARLO."""
        X = numpy.array([[1.0], [2.0], [3.0]])
        h = numpy.array([[1.0], [2.0], [3.0]])
        weights = numpy.ones(3)
        feature_types = numpy.array([sigma._types.CovariateType.REAL])
        with self.assertRaises(ValueError):
            sigma._statistics.select_variable(
                X,
                h,
                weights,
                feature_types,
                sigma._types.TestStat.QUADRATIC,
                sigma._types.TestType.MONTE_CARLO,
                resamples=None,
                rng=numpy.random.default_rng(0),
            )


class TestSelectVariableReturnsTestInputs(unittest.TestCase):
    """Tests that select_variable surfaces the selected variable's T, mu, Sigma."""

    __slots__ = ()

    def test_returns_hand_computed_T_mu_sigma_for_single_real_feature(self):
        """T, mu, Sigma on the result match the hand-computed reference values."""
        X = numpy.array([[1.0], [2.0], [3.0], [4.0]])
        y = numpy.array([10.0, 20.0, 30.0, 40.0]).reshape(-1, 1)
        weights = numpy.ones(4)
        feature_types = numpy.array([sigma._types.CovariateType.REAL])
        result = sigma._statistics.select_variable(
            X,
            y,
            weights,
            feature_types,
            sigma._types.TestStat.QUADRATIC,
            sigma._types.TestType.SIDAK,
            alpha=1.0,
            correlation=sigma._types.Correlation.NORMAL,
        )
        assert result is not None
        self.assertEqual(result.feature_index, 0)
        numpy.testing.assert_allclose(result.T, numpy.array([300.0]))
        numpy.testing.assert_allclose(result.mu, numpy.array([250.0]))
        numpy.testing.assert_allclose(result.Sigma, numpy.array([[2500.0 / 3]]))

    def test_returns_triple_for_winning_variable_among_many(self):
        """T, mu, Sigma correspond to the selected variable, not to any other."""
        rng = numpy.random.default_rng(42)
        n = 200
        x_noise = rng.standard_normal(n)
        x_signal = numpy.linspace(0, 10, n)
        X = numpy.column_stack([x_noise, x_signal])
        y = 3.0 * x_signal + rng.standard_normal(n) * 0.1
        weights = numpy.ones(n)
        feature_types = numpy.array(
            [sigma._types.CovariateType.REAL, sigma._types.CovariateType.REAL]
        )
        result = sigma._statistics.select_variable(
            X,
            y.reshape(-1, 1),
            weights,
            feature_types,
            sigma._types.TestStat.QUADRATIC,
            sigma._types.TestType.SIDAK,
            correlation=sigma._types.Correlation.NORMAL,
        )
        assert result is not None
        self.assertEqual(result.feature_index, 1)
        g_winner = X[:, 1:2]
        expected_T = sigma._statistics.compute_linear_statistic(
            g_winner, y.reshape(-1, 1), weights
        )
        expected_mu = sigma._statistics.compute_conditional_expectation(
            g_winner, y.reshape(-1, 1), weights
        )
        expected_Sigma = sigma._statistics.compute_conditional_covariance(
            g_winner, y.reshape(-1, 1), weights
        )
        numpy.testing.assert_allclose(result.T, expected_T)
        numpy.testing.assert_allclose(result.mu, expected_mu)
        numpy.testing.assert_allclose(result.Sigma, expected_Sigma)

    def test_returns_triple_for_categorical_winner(self):
        """T, mu, Sigma are returned with the categorical-onehot shape for a categorical winner."""
        rng = numpy.random.default_rng(7)
        n = 200
        categories = rng.integers(0, 3, size=n).astype(float)
        y = numpy.where(
            categories == 0,
            10.0,
            numpy.where(categories == 1, 20.0, 30.0),
        )
        y = y + rng.standard_normal(n) * 0.5
        X = categories.reshape(-1, 1)
        weights = numpy.ones(n)
        feature_types = numpy.array([sigma._types.CovariateType.CATEGORICAL])
        result = sigma._statistics.select_variable(
            X,
            y.reshape(-1, 1),
            weights,
            feature_types,
            sigma._types.TestStat.QUADRATIC,
            sigma._types.TestType.SIDAK,
            correlation=sigma._types.Correlation.NORMAL,
        )
        assert result is not None
        self.assertEqual(result.feature_index, 0)
        self.assertEqual(result.T.shape, (3,))
        self.assertEqual(result.mu.shape, (3,))
        self.assertEqual(result.Sigma.shape, (3, 3))
