"""Capture the MovieLens RankingTree reference fixture for equivalence tests.

Run this script ONCE against the current Sigma codebase to produce
`ranking_reference_movielens.npz` next to it. The downstream regression
test `tests.test_tree_ranking_equivalence.
TestMovieLensEndToEndEquivalence` loads the saved arrays and asserts
that a freshly-fitted RankingTree reproduces the captured tree shape
and per-node ranking-metric arrays at rtol=1e-9, atol=1e-9.

Invocation:
    mamba run -n standard python tests/data/build_ranking_reference.py

The script re-uses the demo's MovieLens-1M loading pipeline from
`build_demos._build_movielens_tree` so the captured data matches what
`build_demos.py` would feed to the tree.
"""

import os
import sys
import time
import zipfile

import numpy
import pandas

_REPO_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), os.pardir, os.pardir)
)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import sigma
import sigma._partition

_CACHE_DIR = os.path.join(_REPO_ROOT, ".demo_data")
_OUTPUT_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "ranking_reference_movielens.npz",
)


def run() -> int:
    """Build the MovieLens reference fixture and write it to disk."""
    print(f"Loading MovieLens-1M from {_CACHE_DIR} ...")
    t0 = time.perf_counter()
    X, rankings = _load_movielens()
    print(
        f"  loaded in {time.perf_counter() - t0:.1f}s; X={X.shape}, y={rankings.shape}"
    )
    n_items = rankings.shape[1]
    item_names = list(rankings.columns)
    print("Fitting RankingTree(random_state=123, max_depth=4) ...")
    t0 = time.perf_counter()
    tree = sigma.RankingTree(random_state=123, max_depth=4)
    tree.fit(X, rankings)
    fit_seconds = time.perf_counter() - t0
    print(
        f"  fitted in {fit_seconds:.1f}s; n_nodes={len(tree.nodes_)}, n_leaves={len(tree.leaves_)}"
    )
    payload = _serialize_tree(
        tree, n_items=n_items, item_names=item_names, fit_seconds=fit_seconds
    )
    numpy.savez_compressed(_OUTPUT_PATH, **payload)
    size_bytes = os.path.getsize(_OUTPUT_PATH)
    print(f"Wrote {_OUTPUT_PATH} ({size_bytes / 1024 / 1024:.2f} MB)")
    return 0


def _load_movielens():
    """Replicate build_demos._build_movielens_tree's data loading."""
    zip_path = os.path.join(_CACHE_DIR, "ml-1m.zip")
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
    ).astype(
        {
            "user_id": "int64",
            "age": "int64",
            "occupation": "int64",
        }
    )
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


def _serialize_tree(tree, n_items: int, item_names, fit_seconds: float) -> dict:
    """Flatten a fitted RankingTree into npz-friendly arrays."""
    nodes = tree.nodes_
    n_nodes = len(nodes)
    depth = numpy.empty(n_nodes, dtype=numpy.int64)
    n_samples = numpy.empty(n_nodes, dtype=numpy.int64)
    is_leaf = numpy.empty(n_nodes, dtype=bool)
    feature_index = numpy.full(n_nodes, -1, dtype=numpy.int64)
    partition_kind = numpy.zeros(n_nodes, dtype=numpy.int64)
    threshold = numpy.full(n_nodes, numpy.nan, dtype=float)
    left_child = numpy.full(n_nodes, -1, dtype=numpy.int64)
    right_child = numpy.full(n_nodes, -1, dtype=numpy.int64)
    metric_value = numpy.full((n_nodes, n_items), numpy.nan, dtype=float)
    metric_ci_low = numpy.full((n_nodes, n_items), numpy.nan, dtype=float)
    metric_ci_high = numpy.full((n_nodes, n_items), numpy.nan, dtype=float)
    left_categories_list: list[list[str]] = []
    right_categories_list: list[list[str]] = []
    for node in nodes:
        i = node.node_id
        depth[i] = node.depth
        n_samples[i] = node.n_samples
        metric_value[i] = node.predicted_ranks
        metric_ci_low[i] = node.ci_low
        metric_ci_high[i] = node.ci_high
        match node.extension:
            case sigma._partition.Partition() as partition:
                is_leaf[i] = False
                feature_index[i] = partition.feature_index
                left_child[i] = partition.children[0].node_id
                right_child[i] = partition.children[1].node_id
                match partition:
                    case sigma._partition.NumericalPartition():
                        partition_kind[i] = 1
                        threshold[i] = float(partition.thresholds[0])
                    case sigma._partition.BooleanPartition():
                        partition_kind[i] = 2
                    case sigma._partition.CategoricalPartition():
                        partition_kind[i] = 3
                        left_categories_list.append(
                            sorted(partition.category_groups[0], key=repr)
                        )
                        right_categories_list.append(
                            sorted(partition.category_groups[1], key=repr)
                        )
            case _:
                is_leaf[i] = True
    leaf_ids = numpy.array(
        [leaf.node_id for leaf in tree.leaves_], dtype=numpy.int64
    )
    categorical_node_ids = numpy.flatnonzero(partition_kind == 3).astype(
        numpy.int64
    )
    categorical_left = numpy.array(
        [_join(cats) for cats in left_categories_list], dtype=object
    )
    categorical_right = numpy.array(
        [_join(cats) for cats in right_categories_list], dtype=object
    )
    payload = {
        "n_items": numpy.int64(n_items),
        "n_nodes": numpy.int64(n_nodes),
        "fit_seconds": numpy.float64(fit_seconds),
        "item_names": numpy.array(item_names, dtype=object),
        "depth": depth,
        "n_samples": n_samples,
        "is_leaf": is_leaf,
        "feature_index": feature_index,
        "partition_kind": partition_kind,
        "threshold": threshold,
        "left_child": left_child,
        "right_child": right_child,
        "leaf_ids": leaf_ids,
        "categorical_node_ids": categorical_node_ids,
        "categorical_left": categorical_left,
        "categorical_right": categorical_right,
        "metric_value": metric_value,
        "metric_ci_low": metric_ci_low,
        "metric_ci_high": metric_ci_high,
    }
    return payload


def _join(items) -> str:
    """Join an iterable of category labels into a stable string key."""
    rendered = "␟".join(str(item) for item in items)
    return rendered


if __name__ == "__main__":
    sys.exit(run())
