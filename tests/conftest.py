import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def clean_df():
    """
    Clean synthetic dataset for testing.

    Contains:
    - numeric columns
    - categorical columns
    - datetime column
    - boolean column
    - identifier column
    """

    np.random.seed(42)

    n = 200

    df = pd.DataFrame(
        {
            "user_id": [
                f"U{i:04d}"
                for i in range(1, n + 1)
            ],
            "revenue": np.random.normal(
                loc=1000,
                scale=150,
                size=n,
            ).round(2),
            "orders": np.random.randint(
                5,
                25,
                size=n,
            ),
            "region": np.random.choice(
                [
                    "Delhi",
                    "Mumbai",
                    "Pune",
                    "Bangalore",
                ],
                size=n,
            ),
            "channel": np.random.choice(
                [
                    "App",
                    "Web",
                ],
                size=n,
            ),
            "order_date": pd.date_range(
                start="2025-01-01",
                periods=n,
                freq="D",
            ),
            "is_active": np.random.choice(
                [
                    True,
                    False,
                ],
                size=n,
            ),
        }
    )

    return df


@pytest.fixture
def messy_df(clean_df):
    """
    Messy version of clean_df.

    Adds:
    - missing values
    - duplicate rows
    - constant column
    - outlier
    """

    df = clean_df.copy()

    # ---------------------------------------------
    # Missing values
    # ---------------------------------------------

    df.loc[0:9, "revenue"] = np.nan

    df.loc[10:19, "region"] = None

    # ---------------------------------------------
    # Constant column
    # ---------------------------------------------

    df["constant_col"] = "same_value"

    # ---------------------------------------------
    # Outlier
    # ---------------------------------------------

    df.loc[20, "revenue"] = 100000

    # ---------------------------------------------
    # Duplicate rows
    # ---------------------------------------------

    duplicates = df.iloc[
        30:35
    ].copy()

    df = pd.concat(
        [
            df,
            duplicates,
        ],
        ignore_index=True,
    )

    return df