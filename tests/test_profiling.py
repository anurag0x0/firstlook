import numpy as np
import pandas as pd

from firstlook.profiling import (
    infer_semantic_type,
    profile_dataframe,
)


def test_numeric_type():
    series = pd.Series(
        [10, 20, 30, 40, 50],
        name="revenue",
    )

    assert infer_semantic_type(series) == "numeric"


def test_categorical_type():
    series = pd.Series(
        [
            "Delhi",
            "Mumbai",
            "Delhi",
            "Pune",
            "Mumbai",
        ],
        name="region",
    )

    assert infer_semantic_type(series) == "categorical"


def test_datetime_type():
    series = pd.Series(
        [
            "2025-01-01",
            "2025-01-02",
            "2025-01-03",
            "2025-01-04",
            "2025-01-05",
        ],
        name="order_date",
    )

    assert infer_semantic_type(series) == "datetime"


def test_boolean_type():
    series = pd.Series(
        [True, False, True, False, True],
        name="is_active",
    )

    assert infer_semantic_type(series) == "boolean"


def test_identifier_type():
    series = pd.Series(
        [
            "U0001",
            "U0002",
            "U0003",
            "U0004",
            "U0005",
        ],
        name="user_id",
    )

    assert infer_semantic_type(series) == "identifier"


def test_text_type():
    series = pd.Series(
        [
            (
                "This is a very long customer feedback message "
                "that contains enough text to be classified as text."
            ),
            (
                "Another long customer feedback message "
                "with enough characters to trigger text detection."
            ),
            (
                "This customer provided a detailed explanation "
                "about their experience with the product."
            ),
        ],
        name="feedback",
    )

    assert infer_semantic_type(series) == "text"


def test_constant_type():
    series = pd.Series(
        [
            "INR",
            "INR",
            "INR",
            "INR",
            "INR",
        ],
        name="currency",
    )

    assert infer_semantic_type(series) == "constant"


def test_empty_type():
    series = pd.Series(
        [
            np.nan,
            np.nan,
            np.nan,
            np.nan,
            np.nan,
        ],
        name="empty_col",
    )

    assert infer_semantic_type(series) == "empty"


def test_profile_dataframe():
    df = pd.DataFrame(
        {
            "user_id": [
                "U001",
                "U002",
                "U003",
                "U004",
                "U005",
            ],
            "revenue": [
                100,
                200,
                300,
                400,
                500,
            ],
            "region": [
                "Delhi",
                "Mumbai",
                "Delhi",
                "Pune",
                "Mumbai",
            ],
        }
    )

    profile = profile_dataframe(df)

    assert profile.n_rows == 5
    assert profile.n_cols == 3
    assert profile.n_duplicate_rows == 0

    assert (
        profile.get("user_id").semantic_type
        == "identifier"
    )

    assert (
        profile.get("revenue").semantic_type
        == "numeric"
    )

    assert (
        profile.get("region").semantic_type
        == "categorical"
    )


def test_clean_fixture_profile(clean_df):
    profile = profile_dataframe(clean_df)

    assert profile.n_rows == 200
    assert profile.n_cols == 7

    assert (
        profile.get("user_id").semantic_type
        == "identifier"
    )

    assert (
        profile.get("revenue").semantic_type
        == "numeric"
    )

    assert (
        profile.get("orders").semantic_type
        == "numeric"
    )

    assert (
        profile.get("region").semantic_type
        == "categorical"
    )

    assert (
        profile.get("channel").semantic_type
        == "categorical"
    )

    assert (
        profile.get("order_date").semantic_type
        == "datetime"
    )

    assert (
        profile.get("is_active").semantic_type
        == "boolean"
    )