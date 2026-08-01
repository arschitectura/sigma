"""End-to-end RankingTree equivalence tests against the legacy primitives.

Two scenarios:

* `TestSushiEndToEndEquivalence` runs side-by-side: it monkey-patches
  `sigma._ranking` to use embedded legacy implementations of
  `compute_pl_mle`, `_compute_pl_mle_from_cache` and `pl_expected_rank`
  during the legacy fit, then re-fits with the production code and
  asserts tree shape + per-node metrics agree at rtol=1e-9, atol=1e-9.
* `TestMovieLensEndToEndEquivalence` is fixture-based: it loads the
  pre-captured per-node metric arrays from
  `tests/data/ranking_reference_movielens.npz`, re-fits with the
  production code, and asserts identical tree shape + close metrics.
  Gated behind the `SIGMA_RUN_SLOW_RANKING_EQUIV=1` env var because the
  MovieLens fit takes minutes.
"""

import contextlib
import os
import unittest
import zipfile

import _helpers
import numpy
import numpy.testing
import pandas

import sigma
import sigma._partition
import sigma._ranking
from tests.test_ranking_equivalence import (
    _legacy_compute_pl_mle,
    _legacy_pl_expected_rank,
)

_RTOL = 1e-9
_ATOL = 1e-9
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
_DEMO_DATA_DIR = os.path.join(_REPO_ROOT, ".demo_data")
_MOVIELENS_REFERENCE = os.path.join(
    os.path.dirname(__file__), "data", "ranking_reference_movielens.npz"
)


def _legacy_compute_pl_mle_from_cache(
    cache, ordering_weights, n_items, npseudo, tolerance, max_iter
):
    """Reconstruct y from the cache, then run the legacy MM loop on it."""
    n_orderings = int(cache.row_sizes.size)
    if n_orderings == 0:
        empty = numpy.full(n_items, numpy.nan, dtype=float)
        return empty
    y_local = numpy.full((n_orderings, n_items), numpy.nan, dtype=float)
    for i in range(n_orderings):
        start = int(cache.row_starts[i])
        end = int(cache.row_ends_inclusive[i]) + 1
        ord_items = cache.flat_idx[start:end]
        for position, item in enumerate(ord_items):
            y_local[i, int(item)] = float(position + 1)
    weights_arr = numpy.asarray(ordering_weights, dtype=float)
    alpha = _legacy_compute_pl_mle(
        y_local,
        weights_arr,
        npseudo=npseudo,
        tolerance=tolerance,
        max_iter=max_iter,
    )
    return alpha


@contextlib.contextmanager
def _ranking_legacy_active():
    """Swap the production PL primitives for embedded legacy ones during a fit."""
    targets = {
        "_compute_pl_mle_from_cache": _legacy_compute_pl_mle_from_cache,
        "pl_expected_rank": _legacy_pl_expected_rank,
        "compute_pl_mle": _legacy_compute_pl_mle,
    }
    original = {name: getattr(sigma._ranking, name) for name in targets}
    try:
        for name, replacement in targets.items():
            setattr(sigma._ranking, name, replacement)
        yield
    finally:
        for name, replacement in original.items():
            setattr(sigma._ranking, name, replacement)


def _assert_same_tree_shape(test_case, tree_a, tree_b):
    """Compare two fitted RankingTree instances for structural equality."""
    test_case.assertEqual(len(tree_a.nodes_), len(tree_b.nodes_))
    test_case.assertEqual(len(tree_a.leaves_), len(tree_b.leaves_))
    for node_a, node_b in zip(tree_a.nodes_, tree_b.nodes_):
        test_case.assertEqual(node_a.node_id, node_b.node_id)
        test_case.assertEqual(node_a.depth, node_b.depth)
        test_case.assertEqual(node_a.n_samples, node_b.n_samples)
        ext_a = node_a.extension
        ext_b = node_b.extension
        test_case.assertIs(type(ext_a), type(ext_b))
        match ext_a:
            case sigma._partition.Partition() as part_a:
                part_b = ext_b
                test_case.assertEqual(
                    part_a.feature_index, part_b.feature_index
                )
                match part_a:
                    case sigma._partition.NumericalPartition():
                        test_case.assertEqual(
                            part_a.thresholds, part_b.thresholds
                        )
                    case sigma._partition.CategoricalPartition():
                        test_case.assertEqual(
                            part_a.category_groups, part_b.category_groups
                        )


def _assert_same_metrics(test_case, tree_a, tree_b):
    """Compare per-node ranking metrics across two fitted RankingTrees."""
    for node_a, node_b in zip(tree_a.nodes_, tree_b.nodes_):
        values_a = numpy.array(
            [metric.value for metric in node_a.metrics], dtype=float
        )
        values_b = numpy.array(
            [metric.value for metric in node_b.metrics], dtype=float
        )
        numpy.testing.assert_allclose(
            values_a,
            values_b,
            rtol=_RTOL,
            atol=_ATOL,
            err_msg=f"metric.value mismatch at node_id={node_a.node_id}",
        )
        for field in ("ci_low", "ci_high"):
            arr_a = numpy.array(
                [
                    numpy.nan
                    if getattr(metric, field) is None
                    else float(getattr(metric, field))
                    for metric in node_a.metrics
                ],
                dtype=float,
            )
            arr_b = numpy.array(
                [
                    numpy.nan
                    if getattr(metric, field) is None
                    else float(getattr(metric, field))
                    for metric in node_b.metrics
                ],
                dtype=float,
            )
            numpy.testing.assert_allclose(
                arr_a,
                arr_b,
                rtol=_RTOL,
                atol=_ATOL,
                err_msg=(
                    f"metric.{field} mismatch at node_id={node_a.node_id}"
                ),
            )


class TestSushiEndToEndEquivalence(unittest.TestCase):
    """Side-by-side equivalence on a 1000-user SUSHI3A subset, max_depth=2."""

    __slots__ = ()

    @classmethod
    def setUpClass(cls):
        """Load SUSHI3A once for the whole class."""
        cls.X, cls.rankings = _helpers._load_sushi3a_subset(1000)

    def test_tree_shape_and_metrics_match(self):
        """Production tree matches a tree fitted with the legacy PL primitives."""
        with _ranking_legacy_active():
            tree_legacy = sigma.RankingTree(
                random_state=123, max_depth=2, ci_replicates=20
            )
            tree_legacy.fit(self.X, self.rankings)
        tree_new = sigma.RankingTree(
            random_state=123, max_depth=2, ci_replicates=20
        )
        tree_new.fit(self.X, self.rankings)
        _assert_same_tree_shape(self, tree_legacy, tree_new)
        _assert_same_metrics(self, tree_legacy, tree_new)


def _load_movielens_for_fit():
    """Load MovieLens-1M from the cached zip exactly as build_demos does."""
    zip_path = os.path.join(_DEMO_DATA_DIR, "ml-1m.zip")
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open("ml-1m/ratings.dat") as ratings_file:
            ratings_text = ratings_file.read().decode("latin-1")
        with archive.open("ml-1m/movies.dat") as movies_file:
            movies_text = movies_file.read().decode("latin-1")
        with archive.open("ml-1m/users.dat") as users_file:
            users_text = users_file.read().decode("latin-1")
    ratings_dataframe = pandas.DataFrame(
        [line.split("::") for line in ratings_text.splitlines() if line],
        columns=["user_id", "movie_id", "rating", "timestamp"],
    ).astype(
        {
            "user_id": "int64",
            "movie_id": "int64",
            "rating": "int64",
            "timestamp": "int64",
        }
    )
    movies_dataframe = pandas.DataFrame(
        [line.split("::") for line in movies_text.splitlines() if line],
        columns=["movie_id", "title", "genres"],
    ).astype({"movie_id": "int64"})
    users_dataframe = pandas.DataFrame(
        [line.split("::") for line in users_text.splitlines() if line],
        columns=["user_id", "gender", "age", "occupation", "zip_code"],
    ).astype({"user_id": "int64", "age": "int64", "occupation": "int64"})
    sorted_ratings = ratings_dataframe.sort_values(
        ["user_id", "rating", "timestamp"], ascending=[True, False, True]
    ).copy()
    sorted_ratings["personal_rank"] = (
        sorted_ratings.groupby("user_id").cumcount() + 1
    )
    rankings = sorted_ratings.pivot(
        index="user_id", columns="movie_id", values="personal_rank"
    )
    movie_id_to_title = dict(
        zip(movies_dataframe["movie_id"], movies_dataframe["title"])
    )
    rankings.columns = [
        movie_id_to_title.get(int(column), str(column))
        for column in rankings.columns
    ]
    rated_count_per_user = rankings.notna().sum(axis=1)
    qualifying_mask = rated_count_per_user >= 2
    rankings = rankings.loc[qualifying_mask]
    qualifying_users = rankings.index.tolist()
    user_demographics = (
        users_dataframe.set_index("user_id").loc[qualifying_users].reset_index()
    )
    age_label = {
        1: "<18",
        18: "18-24",
        25: "25-34",
        35: "35-44",
        45: "45-49",
        50: "50-55",
        56: "56+",
    }
    occupation_label = {
        0: "other",
        1: "academic/educator",
        2: "artist",
        3: "clerical/admin",
        4: "college/grad student",
        5: "customer service",
        6: "doctor/health care",
        7: "executive/managerial",
        8: "farmer",
        9: "homemaker",
        10: "K-12 student",
        11: "lawyer",
        12: "programmer",
        13: "retired",
        14: "sales/marketing",
        15: "scientist",
        16: "self-employed",
        17: "technician/engineer",
        18: "tradesman/craftsman",
        19: "unemployed",
        20: "writer",
    }
    X = pandas.DataFrame(
        {
            "Age band": pandas.Categorical(
                user_demographics["age"].map(age_label),
                categories=[
                    "<18",
                    "18-24",
                    "25-34",
                    "35-44",
                    "45-49",
                    "50-55",
                    "56+",
                ],
                ordered=True,
            ),
            "Gender": pandas.Categorical(
                user_demographics["gender"], categories=["F", "M"]
            ),
            "Occupation": pandas.Categorical(
                user_demographics["occupation"].map(occupation_label)
            ),
        }
    )
    return X, rankings


@unittest.skipUnless(
    os.environ.get("SIGMA_RUN_SLOW_RANKING_EQUIV"),
    "slow; opt in by setting SIGMA_RUN_SLOW_RANKING_EQUIV=1",
)
class TestMovieLensEndToEndEquivalence(unittest.TestCase):
    """Fixture-based regression: refit MovieLens, compare to captured arrays."""

    __slots__ = ()

    def test_against_captured_reference(self):
        """Production refit matches the captured pre-change tree structure and metrics."""
        if not os.path.exists(_MOVIELENS_REFERENCE):
            self.fail(
                f"Fixture not found at {_MOVIELENS_REFERENCE!s}; rerun"
                f" tests/data/build_ranking_reference.py to regenerate it."
            )
        reference = numpy.load(_MOVIELENS_REFERENCE, allow_pickle=True)
        X, rankings = _load_movielens_for_fit()
        tree = sigma.RankingTree(random_state=123, max_depth=4)
        tree.fit(X, rankings)
        self._assert_tree_shape_matches_reference(tree, reference)
        self._assert_metric_arrays_match_reference(tree, reference)

    def _assert_tree_shape_matches_reference(self, tree, reference):
        """Compare tree topology against the saved depth / split / leaf arrays."""
        n_nodes_expected = int(reference["n_nodes"])
        self.assertEqual(len(tree.nodes_), n_nodes_expected)
        for node in tree.nodes_:
            node_id = node.node_id
            self.assertEqual(node.depth, int(reference["depth"][node_id]))
            self.assertEqual(
                node.n_samples, int(reference["n_samples"][node_id])
            )
            is_leaf_expected = bool(reference["is_leaf"][node_id])
            match node.extension:
                case sigma._partition.Partition() as partition:
                    self.assertFalse(is_leaf_expected)
                    self.assertEqual(
                        partition.feature_index,
                        int(reference["feature_index"][node_id]),
                    )
                    partition_kind = int(reference["partition_kind"][node_id])
                    match partition_kind:
                        case 1:
                            self.assertIsInstance(
                                partition, sigma._partition.NumericalPartition
                            )
                            self.assertEqual(
                                float(partition.thresholds[0]),
                                float(reference["threshold"][node_id]),
                            )
                        case 2:
                            self.assertIsInstance(
                                partition, sigma._partition.BooleanPartition
                            )
                        case 3:
                            self.assertIsInstance(
                                partition,
                                sigma._partition.CategoricalPartition,
                            )
                case _:
                    self.assertTrue(is_leaf_expected)
        ref_leaf_ids = sorted(int(x) for x in reference["leaf_ids"])
        actual_leaf_ids = sorted(int(leaf.node_id) for leaf in tree.leaves_)
        self.assertEqual(actual_leaf_ids, ref_leaf_ids)

    def _assert_metric_arrays_match_reference(self, tree, reference):
        """Compare per-node (value, ci_low, ci_high) arrays at rtol=1e-9."""
        ref_value = reference["metric_value"]
        ref_ci_low = reference["metric_ci_low"]
        ref_ci_high = reference["metric_ci_high"]
        for node in tree.nodes_:
            node_id = node.node_id
            actual_value = numpy.array(
                [metric.value for metric in node.metrics], dtype=float
            )
            actual_ci_low = numpy.array(
                [
                    numpy.nan if metric.ci_low is None else float(metric.ci_low)
                    for metric in node.metrics
                ],
                dtype=float,
            )
            actual_ci_high = numpy.array(
                [
                    numpy.nan
                    if metric.ci_high is None
                    else float(metric.ci_high)
                    for metric in node.metrics
                ],
                dtype=float,
            )
            numpy.testing.assert_allclose(
                actual_value,
                ref_value[node_id],
                rtol=_RTOL,
                atol=_ATOL,
                err_msg=f"metric.value mismatch at node_id={node_id}",
            )
            numpy.testing.assert_allclose(
                actual_ci_low,
                ref_ci_low[node_id],
                rtol=_RTOL,
                atol=_ATOL,
                err_msg=f"metric.ci_low mismatch at node_id={node_id}",
            )
            numpy.testing.assert_allclose(
                actual_ci_high,
                ref_ci_high[node_id],
                rtol=_RTOL,
                atol=_ATOL,
                err_msg=f"metric.ci_high mismatch at node_id={node_id}",
            )


if __name__ == "__main__":
    unittest.main()
