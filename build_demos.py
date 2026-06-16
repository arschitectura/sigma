#!/usr/bin/env python3
"""Generate eight demo decision-tree PNGs at <dir>/<prefix><dataset>.png."""

import argparse
import os
import sys
import time
import urllib.request
import zipfile

import numpy
import pandas
import scipy.io.arff
import sklearn.datasets

import sigma

_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), ".demo_data"
)


def run() -> int:
    """Parse CLI args and write the six demo tree PNGs with response companions.

    Returns:
        0 on success. argparse may exit the process with status 2 before
        this function returns if argument parsing or validation fails
        (e.g., a missing or non-directory --dir).
    """
    args = _parse_args()
    output_dir = args.dir
    prefix = args.prefix
    dpi = args.dpi
    _build_diabetes_tree(
        os.path.join(output_dir, f"{prefix}diabetes.png"), dpi
    )
    _build_titanic_tree(
        os.path.join(output_dir, f"{prefix}titanic.png"), dpi
    )
    _build_german_credit_tree(
        os.path.join(output_dir, f"{prefix}german_credit.png"), dpi
    )
    _build_insurance_tree(
        os.path.join(output_dir, f"{prefix}insurance.png"), dpi
    )
    _build_breast_cancer_tree(
        os.path.join(output_dir, f"{prefix}breast_cancer.png"), dpi
    )
    _build_telco_churn_tree(
        os.path.join(output_dir, f"{prefix}telco_churn.png"), dpi
    )
    _build_movielens_tree(
        os.path.join(output_dir, f"{prefix}movielens.png"), dpi
    )
    _build_sushi_tree(
        os.path.join(output_dir, f"{prefix}sushi.png"), dpi
    )
    return 0


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for run()."""
    default_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(
        description=(
            "Fit eight demo conditional inference trees (Diabetes, Titanic, "
            "German Credit, Insurance, Breast Cancer, Telco Churn, MovieLens, "
            "Sushi) and write one PNG per dataset."
        ),
    )
    parser.add_argument(
        "--dir",
        default=default_dir,
        type=_existing_directory,
        help=(
            "Directory in which to write the PNG files; must already exist. "
            "Defaults to the directory of this script."
        ),
    )
    parser.add_argument(
        "--prefix",
        default="demo_",
        help='Filename prefix for each PNG (default: "demo_").',
    )
    parser.add_argument(
        "--dpi",
        default=96,
        type=_positive_integer,
        help=(
            "Output resolution in dots per inch for both the tree and "
            "response PNGs. Defaults to 96 (screen-friendly)."
        ),
    )
    args = parser.parse_args()
    return args


def _existing_directory(value: str) -> str:
    """Argparse type that accepts only paths to existing directories."""
    if not os.path.isdir(value):
        raise argparse.ArgumentTypeError(
            f"directory does not exist: {value}"
        )
    return value


def _positive_integer(value: str) -> int:
    """Argparse type that accepts only positive integers."""
    try:
        parsed = int(value)
    except ValueError as value_error:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer; got {value!r}"
        ) from value_error
    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            f"expected a positive integer; got {parsed}"
        )
    return parsed


def _build_diabetes_tree(output_path: str, dpi: int) -> None:
    """Fit a regression tree on the sklearn Diabetes dataset and write a PNG to
    output_path at the requested dpi.
    """
    _print_dataset_header("Diabetes")
    cache_path = os.path.join(_CACHE_DIR, "diabetes.csv")
    if not os.path.exists(cache_path):
        os.makedirs(_CACHE_DIR, exist_ok=True)
        diabetes_bunch = sklearn.datasets.load_diabetes(
            as_frame=True, scaled=False
        )
        target_name = diabetes_bunch.target.name
        if target_name != "target":
            raise ValueError(
                f"load_diabetes target name drift: {target_name!r}"
            )
        diabetes_bunch.frame.to_csv(cache_path, index=False)
    diabetes_frame = pandas.read_csv(
        cache_path, float_precision="round_trip"
    )
    diabetes_data = diabetes_frame[[
        "age", "sex", "bmi", "bp",
        "s1", "s2", "s3", "s4", "s5", "s6",
    ]].astype({
        "age": "float64",
        "sex": "float64",
        "bmi": "float64",
        "bp": "float64",
        "s1": "float64",
        "s2": "float64",
        "s3": "float64",
        "s4": "float64",
        "s5": "float64",
        "s6": "float64",
    })
    X_diabetes = diabetes_data.rename(columns={
        "age": "Age",
        "sex": "Sex",
        "bmi": "BMI",
        "bp": "Blood pressure",
        "s1": "Total cholesterol",
        "s2": "LDL cholesterol",
        "s3": "HDL cholesterol",
        "s4": "Total-to-HDL ratio",
        "s5": "Triglycerides (log)",
        "s6": "Blood sugar",
    })
    y_diabetes = diabetes_frame["target"].astype("float64").rename(
        "Disease progression"
    )
    regression_tree = sigma.RegressionTree(
        test_type="monte_carlo",
        resamples=2000,
        random_state=123,
        reverse_order=True,
    )
    _build_and_export_tree(
        regression_tree,
        X_diabetes,
        y_diabetes,
        output_path,
        dpi,
        precision=1,
        orientation="left-to-right",
    )


def _build_titanic_tree(output_path: str, dpi: int) -> None:
    """Fit a classification tree on the Titanic survival dataset and write a PNG
    to output_path at the requested dpi.
    """
    _print_dataset_header("Titanic")
    url = (
        "https://raw.githubusercontent.com/datasciencedojo"
        "/datasets/master/titanic.csv"
    )
    titanic_dataframe = pandas.read_csv(
        _cached_download(url, "titanic.csv"),
        usecols=["Pclass", "Sex", "Age", "Embarked", "Survived"],
        dtype={
            "Pclass": "int64",
            "Sex": "object",
            "Age": "float64",
            "Embarked": "object",
            "Survived": "int64",
        },
    ).dropna()
    X_titanic = pandas.DataFrame({
        "Passenger class": pandas.Categorical(
            titanic_dataframe["Pclass"].map({1: "1st", 2: "2nd", 3: "3rd"}),
            categories=["1st", "2nd", "3rd"],
        ),
        "Sex": pandas.Categorical(
            titanic_dataframe["Sex"], categories=["female", "male"]
        ),
        "Age": titanic_dataframe["Age"].astype("float64"),
        "Port of embarkation": pandas.Categorical(
            titanic_dataframe["Embarked"].map(
                {"C": "Cherbourg", "Q": "Queenstown", "S": "Southampton"}
            ),
            categories=["Cherbourg", "Queenstown", "Southampton"],
        ),
    })
    y_titanic = pandas.Series(
        pandas.Categorical(
            titanic_dataframe["Survived"].map({0: "died", 1: "survived"}),
            categories=["died", "survived"],
        ),
        name="Survival",
    )
    classification_tree = sigma.ClassificationTree(
        test_type="monte_carlo",
        resamples=2000,
        random_state=123,
    )
    _build_and_export_tree(
        classification_tree,
        X_titanic,
        y_titanic,
        output_path,
        dpi,
        precision=1,
    )


def _build_german_credit_tree(output_path: str, dpi: int) -> None:
    """Fit a classification tree on the OpenML German Credit dataset and write a
    PNG to output_path at the requested dpi.
    """
    _print_dataset_header("German Credit")
    url = "https://www.openml.org/data/v1/download/31/credit-g.arff"
    arff_data, _ = scipy.io.arff.loadarff(
        _cached_download(url, "german_credit.arff")
    )
    credit_frame = pandas.DataFrame(arff_data)
    for column in credit_frame.select_dtypes([object]).columns:
        credit_frame[column] = credit_frame[column].str.decode("utf-8")
    credit_dataframe = credit_frame[
        [
            "checking_status",
            "duration",
            "credit_amount",
            "savings_status",
            "age",
            "housing",
            "class",
        ]
    ].astype({
        "checking_status": "category",
        "duration": "int64",
        "credit_amount": "int64",
        "savings_status": "category",
        "age": "int64",
        "housing": "category",
        "class": "category",
    }).dropna()
    X_german_credit = pandas.DataFrame({
        "Checking account balance": credit_dataframe["checking_status"].cat.rename_categories(
            {"0<=X<200": "0-200", "no checking": "no account"}
        ),
        "Loan duration": credit_dataframe["duration"],
        "Loan amount": credit_dataframe["credit_amount"],
        "Savings balance": credit_dataframe["savings_status"].cat.rename_categories({
            "100<=X<500": "100-500",
            "500<=X<1000": "500-1000",
            "no known savings": "no account",
        }),
        "Age": credit_dataframe["age"],
        "Housing": credit_dataframe["housing"],
    })
    y_german_credit = pandas.Series(
        pandas.Categorical(
            credit_dataframe["class"].map(
                {"good": "Met all payments", "bad": "Missed payments"}
            ),
            categories=["Met all payments", "Missed payments"],
        ),
        name="Payments",
    )
    classification_tree = sigma.ClassificationTree(
        test_type="monte_carlo",
        resamples=2000,
        random_state=123,
        reverse_order=True,
    )
    _build_and_export_tree(
        classification_tree,
        X_german_credit,
        y_german_credit,
        output_path,
        dpi,
        precision=1,
    )


def _build_insurance_tree(output_path: str, dpi: int) -> None:
    """Fit a regression tree on the Medical Insurance Charges dataset and write
    a PNG to output_path at the requested dpi.
    """
    _print_dataset_header("Insurance")
    url = (
        "https://raw.githubusercontent.com/stedy"
        "/Machine-Learning-with-R-datasets/master/insurance.csv"
    )
    insurance_dataframe = pandas.read_csv(
        _cached_download(url, "insurance.csv"),
        usecols=["age", "sex", "bmi", "children", "smoker", "region", "charges"],
        dtype={
            "age": "int64",
            "sex": "object",
            "bmi": "float64",
            "children": "int64",
            "smoker": "object",
            "region": "object",
            "charges": "float64",
        },
    ).dropna()
    X_insurance = pandas.DataFrame({
        "Age": insurance_dataframe["age"],
        "Sex": pandas.Categorical(
            insurance_dataframe["sex"], categories=["female", "male"]
        ),
        "BMI": insurance_dataframe["bmi"],
        "Number of children": insurance_dataframe["children"],
        "Smoking status": insurance_dataframe["smoker"].map(
            {"no": False, "yes": True}
        ).astype(bool),
        "Region": pandas.Categorical(
            insurance_dataframe["region"],
            categories=["northeast", "northwest", "southeast", "southwest"],
        ),
    })
    y_insurance = insurance_dataframe["charges"].rename("Charges")
    regression_tree = sigma.RegressionTree(
        test_type="monte_carlo",
        resamples=2000,
        random_state=123,
        max_depth=4,
        reverse_order=True,
    )
    _build_and_export_tree(
        regression_tree,
        X_insurance,
        y_insurance,
        output_path,
        dpi,
        precision=0,
        orientation="left-to-right",
    )


def _build_breast_cancer_tree(output_path: str, dpi: int) -> None:
    """Fit a survival tree on the breast cancer dataset and write a PNG to
    output_path at the requested dpi.
    """
    _print_dataset_header("Breast Cancer")
    url = (
        "https://raw.githubusercontent.com/sebp/scikit-survival"
        "/master/sksurv/datasets/data/GBSG2.arff"
    )
    arff_data, arff_meta = scipy.io.arff.loadarff(
        _cached_download(url, "breast_cancer.arff")
    )
    expected_arff_attributes = [
        ("horTh", "nominal"),
        ("age", "numeric"),
        ("menostat", "nominal"),
        ("tsize", "numeric"),
        ("tgrade", "nominal"),
        ("pnodes", "numeric"),
        ("progrec", "numeric"),
        ("estrec", "numeric"),
        ("time", "numeric"),
        ("cens", "nominal"),
    ]
    actual_arff_attributes = [
        (name, arff_meta[name][0]) for name in arff_meta.names()
    ]
    if actual_arff_attributes != expected_arff_attributes:
        raise ValueError(
            f"GBSG2.arff schema drift: {actual_arff_attributes!r}"
        )
    breast_cancer_dataframe = pandas.DataFrame(arff_data)
    for column in breast_cancer_dataframe.select_dtypes([object]).columns:
        breast_cancer_dataframe[column] = breast_cancer_dataframe[column].str.decode("utf-8")
    X_breast_cancer = pandas.DataFrame({
        "Hormone therapy": breast_cancer_dataframe["horTh"].map(
            {"no": False, "yes": True}
        ).astype(bool),
        "Age": breast_cancer_dataframe["age"].astype("float64"),
        "Menopausal status": pandas.Categorical(
            breast_cancer_dataframe["menostat"], categories=["Pre", "Post"]
        ).rename_categories({"Pre": "pre", "Post": "post"}),
        "Tumor size": breast_cancer_dataframe["tsize"].astype("float64"),
        "Tumor grade": pandas.Categorical(
            breast_cancer_dataframe["tgrade"], categories=["I", "II", "III"]
        ),
        "Positive lymph nodes": breast_cancer_dataframe["pnodes"].astype("float64"),
        "Progesterone receptor level": breast_cancer_dataframe["progrec"].astype("float64"),
        "Estrogen receptor level": breast_cancer_dataframe["estrec"].astype("float64"),
    })
    y_breast_cancer = pandas.DataFrame({
        "recurrence-free years": (
            breast_cancer_dataframe["time"].astype("float64") / 365.25
        ),
        "event": breast_cancer_dataframe["cens"].astype("float64"),
    })
    survival_tree = sigma.SurvivalTree(
        test_type="monte_carlo",
        resamples=2000,
        random_state=123,
        metrics=("median", ("survival", 5.0, "years")),
    )
    _build_and_export_tree(
        survival_tree,
        X_breast_cancer,
        y_breast_cancer,
        output_path,
        dpi,
        precision=1,
    )


def _build_telco_churn_tree(output_path: str, dpi: int) -> None:
    """Fit a survival tree on the IBM Telco Customer Churn dataset and write a
    PNG to output_path at the requested dpi.
    """
    _print_dataset_header("Telco Churn")
    url = (
        "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d"
        "/master/data/Telco-Customer-Churn.csv"
    )
    telco_dataframe = pandas.read_csv(
        _cached_download(url, "telco_churn.csv"),
        usecols=[
            "Contract",
            "InternetService",
            "OnlineSecurity",
            "TechSupport",
            "PaymentMethod",
            "MonthlyCharges",
            "Partner",
            "Dependents",
            "tenure",
            "Churn",
        ],
        dtype={
            "Contract": "object",
            "InternetService": "object",
            "OnlineSecurity": "object",
            "TechSupport": "object",
            "PaymentMethod": "object",
            "MonthlyCharges": "float64",
            "Partner": "object",
            "Dependents": "object",
            "tenure": "int64",
            "Churn": "object",
        },
    )
    telco_dataframe = telco_dataframe[telco_dataframe["tenure"] > 0].copy()
    X_telco = pandas.DataFrame({
        "Contract type": pandas.Categorical(
            telco_dataframe["Contract"],
            categories=["Month-to-month", "One year", "Two year"],
        ).rename_categories({
            "Month-to-month": "month-to-month",
            "One year": "1 year",
            "Two year": "2 years",
        }),
        "Internet service": pandas.Categorical(
            telco_dataframe["InternetService"],
            categories=["DSL", "Fiber optic", "No"],
        ).rename_categories({
            "DSL": "DSL", "Fiber optic": "fiber", "No": "no internet",
        }),
        "Online security": pandas.Categorical(
            telco_dataframe["OnlineSecurity"],
            categories=["No", "Yes", "No internet service"],
        ).rename_categories({
            "No": "no", "Yes": "yes", "No internet service": "no internet",
        }),
        "Tech support": pandas.Categorical(
            telco_dataframe["TechSupport"],
            categories=["No", "Yes", "No internet service"],
        ).rename_categories({
            "No": "no", "Yes": "yes", "No internet service": "no internet",
        }),
        "Payment method": pandas.Categorical(
            telco_dataframe["PaymentMethod"],
            categories=[
                "Bank transfer (automatic)",
                "Credit card (automatic)",
                "Electronic check",
                "Mailed check",
            ],
        ).rename_categories({
            "Bank transfer (automatic)": "bank transfer",
            "Credit card (automatic)": "credit card",
            "Electronic check": "electronic check",
            "Mailed check": "mailed check",
        }),
        "Monthly charges": telco_dataframe["MonthlyCharges"],
        "Has a partner": telco_dataframe["Partner"].map(
            {"No": False, "Yes": True}
        ).astype(bool),
        "Has dependents": telco_dataframe["Dependents"].map(
            {"No": False, "Yes": True}
        ).astype(bool),
    })
    y_telco = pandas.DataFrame({
        "Tenure (months)": telco_dataframe["tenure"].astype("float64"),
        "event": telco_dataframe["Churn"].map({"No": 0.0, "Yes": 1.0}),
    })
    survival_tree = sigma.SurvivalTree(
        test_type="monte_carlo",
        resamples=2000,
        random_state=123,
        max_depth=4,
        metrics=("median", ("survival", 12.0, "months")),
    )
    _build_and_export_tree(
        survival_tree,
        X_telco,
        y_telco,
        output_path,
        dpi,
        precision=0,
        orientation="left-to-right",
    )


def _build_movielens_tree(output_path: str, dpi: int) -> None:
    """Fit a ranking tree on the MovieLens-1M dataset and write a PNG to
    output_path at the requested dpi.

    Source: Harper and Konstan (2016), "The MovieLens Datasets: History
    and Context," ACM TiiS, 5(4), 19. The 1M dataset (6040 users, 1M
    ratings, 3883 movies) is fetched from GroupLens as a zip of
    double-colon-separated text files.

    The ranking is derived from each user's 1-5 star ratings by sorting
    rating descending, timestamp ascending as the tie-breaker, and
    assigning the personal rank 1, 2, ... to the resulting positions.
    The full catalogue ranks are passed to RankingTree, which internally
    selects the items used for statistical tests by descending log-rank
    variance and collapses the remaining items into a single "Others"
    column on the test side.
    """
    _print_dataset_header("MovieLens")
    url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
    zip_path = _cached_download(url, "ml-1m.zip")
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
    ).astype({
        "user_id": "int64",
        "movie_id": "int64",
        "rating": "int64",
        "timestamp": "int64",
    })
    movies_dataframe = pandas.DataFrame(
        [line.split("::") for line in movies_text.splitlines() if line],
        columns=["movie_id", "title", "genres"],
    ).astype({"movie_id": "int64"})
    users_dataframe = pandas.DataFrame(
        [line.split("::") for line in users_text.splitlines() if line],
        columns=["user_id", "gender", "age", "occupation", "zip_code"],
    ).astype({
        "user_id": "int64",
        "age": "int64",
        "occupation": "int64",
    })
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
    user_demographics = users_dataframe.set_index("user_id").loc[
        qualifying_users
    ].reset_index()
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
    X_movielens = pandas.DataFrame({
        "Age band": pandas.Categorical(
            user_demographics["age"].map(age_label),
            categories=[
                "<18", "18-24", "25-34", "35-44", "45-49", "50-55", "56+"
            ],
            ordered=True,
        ),
        "Gender": pandas.Categorical(
            user_demographics["gender"], categories=["F", "M"]
        ),
        "Occupation": pandas.Categorical(
            user_demographics["occupation"].map(occupation_label)
        ),
    })
    ranking_tree = sigma.RankingTree(
        random_state=123,
        max_depth=4,
    )
    _build_and_export_tree(
        ranking_tree,
        X_movielens,
        rankings,
        output_path,
        dpi,
        precision=2,
        orientation="left-to-right",
        top_displayed_items=1,
    )


def _build_sushi_tree(output_path: str, dpi: int) -> None:
    """Fit a ranking tree on the Sushi preference dataset and write a PNG to
    output_path at the requested dpi.

    Source: Kamishima (2003), "Nantonac collaborative filtering:
    recommendation based on order responses," KDD '03. The SUSHI3A
    dataset (10 sushi items, 5000 respondents) is fetched from
    kamishima.net under a research-only, no-redistribution license; the
    cache file is kept under .demo_data/.
    """
    _print_dataset_header("Sushi")
    url = "https://www.kamishima.net/asset/sushi3-2016.zip"
    zip_path = _cached_download(url, "sushi3-2016.zip")
    with zipfile.ZipFile(zip_path) as archive:
        with archive.open("sushi3-2016/sushi3a.5000.10.order") as order_file:
            order_lines = order_file.read().decode("utf-8").splitlines()
        with archive.open("sushi3-2016/sushi3.udata") as udata_file:
            udata_lines = udata_file.read().decode("utf-8").splitlines()
        with archive.open("sushi3-2016/sushi3.idata") as idata_file:
            idata_lines = idata_file.read().decode("utf-8").splitlines()
    sushi3a_item_count = 10
    all_item_names = [
        line.split("\t")[1] for line in idata_lines if line.strip()
    ]
    item_names = all_item_names[:sushi3a_item_count]
    n_items = len(item_names)
    if n_items != sushi3a_item_count:
        raise ValueError(
            f"SUSHI3A schema drift: expected {sushi3a_item_count}"
            f" items, got {n_items}"
        )
    n_users = sum(1 for line in order_lines[1:] if line.strip())
    ranks_in_cell = numpy.full((n_users, n_items), numpy.nan, dtype=float)
    row_index = 0
    for line in order_lines[1:]:
        if not line.strip():
            continue
        tokens = line.split()
        for position in range(n_items):
            item_id = int(tokens[2 + position])
            ranks_in_cell[row_index, item_id] = float(position + 1)
        row_index += 1
    rankings = pandas.DataFrame(ranks_in_cell, columns=item_names)
    demographic_rows = []
    for line in udata_lines:
        if not line.strip():
            continue
        demographic_rows.append(line.split("\t"))
    demographic_frame = pandas.DataFrame(
        demographic_rows,
        columns=[
            "user_id",
            "gender",
            "age_group",
            "completion_seconds",
            "childhood_prefecture",
            "childhood_region",
            "childhood_east_west",
            "current_prefecture",
            "current_region",
            "current_east_west",
            "migrated",
        ],
    )
    gender_labels = {"0": "male", "1": "female"}
    age_labels = {
        "0": "15-19",
        "1": "20-29",
        "2": "30-39",
        "3": "40-49",
        "4": "50-59",
        "5": "60+",
    }
    region_labels = {
        "0": "Hokkaido",
        "1": "Tohoku",
        "2": "Hokuriku",
        "3": "Kanto+Shizuoka",
        "4": "Nagoya",
        "5": "Kinki",
        "6": "Chugoku",
        "7": "Shikoku",
        "8": "Kyushu",
        "9": "Okinawa",
        "10": "abroad",
        "11": "missing",
    }
    east_west_labels = {"0": "Eastern Japan", "1": "Western Japan"}
    X_sushi = pandas.DataFrame({
        "Gender": pandas.Categorical(
            demographic_frame["gender"].map(gender_labels),
            categories=["female", "male"],
        ),
        "Age group": pandas.Categorical(
            demographic_frame["age_group"].map(age_labels),
            categories=list(age_labels.values()),
            ordered=True,
        ),
        "Childhood region": pandas.Categorical(
            demographic_frame["childhood_region"].map(region_labels),
            categories=list(region_labels.values()),
        ),
        "Childhood part of Japan": pandas.Categorical(
            demographic_frame["childhood_east_west"].map(east_west_labels),
            categories=list(east_west_labels.values()),
        ),
        "Current region": pandas.Categorical(
            demographic_frame["current_region"].map(region_labels),
            categories=list(region_labels.values()),
        ),
        "Migrated since age 15": demographic_frame["migrated"].map(
            {"0": False, "1": True}
        ).astype(bool),
    })
    ranking_tree = sigma.RankingTree(
        random_state=123,
        max_depth=3,
        ci_method="gaussian_multiplier",
    )
    _build_and_export_tree(
        ranking_tree,
        X_sushi,
        rankings,
        output_path,
        dpi,
        precision=2,
        orientation="left-to-right",
    )


def _build_and_export_tree(
    tree,
    X,
    y,
    output_path: str,
    dpi: int,
    precision: int,
    orientation: str = "top-down",
    top_displayed_items: None | int = None,
) -> None:
    """Fit tree on (X, y), print its text, and write the tree and response PNGs."""
    fit_start = time.perf_counter()
    tree.fit(X, y)
    elapsed = time.perf_counter() - fit_start
    print(f"Fitted in {elapsed:.1f}s")
    text = tree.to_text(
        precision=precision, top_displayed_items=top_displayed_items
    )
    print(text)
    print()
    tree_png = tree.to_image(
        "png", dpi=dpi, orientation=orientation, precision=precision
    )
    _write_png(output_path, tree_png, "tree")
    response_png = tree.to_image("png", kind="response", dpi=dpi)
    _write_png(_response_png_path(output_path), response_png, "responses")


def _print_dataset_header(name: str) -> None:
    """Print a section header for a dataset."""
    title = f"{name} dataset"
    left = f"======== {title} "
    right = "=" * (120 - len(left))
    print(f"\n{left}{right}\n")


def _write_png(output_path: str, png_bytes: bytes, label: str) -> None:
    """Write png_bytes to output_path and print a confirmation line."""
    with open(output_path, "wb") as file:
        file.write(png_bytes)
    filename = os.path.basename(output_path)
    print(f"Saved {label} to {filename} ({len(png_bytes)} bytes)")


def _response_png_path(output_path: str) -> str:
    """Return the companion <stem>_response.png path for output_path."""
    stem, extension = os.path.splitext(output_path)
    companion = f"{stem}_response{extension}"
    return companion


def _cached_download(url: str, filename: str) -> str:
    """Return the cached local path for url, downloading it on first miss."""
    path = os.path.join(_CACHE_DIR, filename)
    if not os.path.exists(path):
        os.makedirs(_CACHE_DIR, exist_ok=True)
        with urllib.request.urlopen(url) as response:
            payload = response.read()
        with open(path, "wb") as file:
            file.write(payload)
    return path


if __name__ == "__main__":
    status = run()
    sys.exit(status)
