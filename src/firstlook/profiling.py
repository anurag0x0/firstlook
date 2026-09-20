from dataclasses import dataclass
from enum import Enum

import pandas as pd


class SemanticType(str, Enum):
    """Semantic types used to describe dataset columns."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    IDENTIFIER = "identifier"
    TEXT = "text"
    CONSTANT = "constant"
    EMPTY = "empty"


@dataclass
class ColumnProfile:
    """Profile information for a single dataset column."""

    name: str
    dtype: str
    semantic_type: str
    n_rows: int
    n_missing: int
    missing_pct: float
    n_unique: int
    unique_pct: float
    sample_values: list
    stats: dict


def infer_semantic_type(
    series: pd.Series,
    column_name: str = "",
) -> SemanticType:
    """Infer the semantic type of a pandas Series."""

    name = column_name.lower().strip()

    non_null = series.dropna()

    # 1. Empty column
    if non_null.empty:
        return SemanticType.EMPTY

    # 2. Constant column
    if non_null.nunique(dropna=True) == 1:
        return SemanticType.CONSTANT

    # 3. Boolean dtype
    if pd.api.types.is_bool_dtype(series):
        return SemanticType.BOOLEAN

    # 4. Datetime dtype
    if pd.api.types.is_datetime64_any_dtype(series):
        return SemanticType.DATETIME

    # 5. Boolean-like values
    unique_values = {
        str(value).strip().lower()
        for value in non_null.unique()
    }

    boolean_values = {
        "0",
        "1",
        "yes",
        "no",
        "true",
        "false",
    }

    if (
        len(unique_values) == 2
        and unique_values.issubset(boolean_values)
    ):
        return SemanticType.BOOLEAN

    # 6. Numeric columns
    if pd.api.types.is_numeric_dtype(series):

        unique_count = non_null.nunique()
        unique_ratio = unique_count / len(non_null)

        id_keywords = (
            "id",
            "key",
            "code",
        )

        has_id_name = any(
            keyword in name
            for keyword in id_keywords
        )

        # Numeric identifier
        if has_id_name and unique_ratio >= 0.95:
            return SemanticType.IDENTIFIER

        # Low-cardinality integer category
        if (
            pd.api.types.is_integer_dtype(series)
            and unique_count <= 10
            and unique_ratio < 0.05
        ):
            return SemanticType.CATEGORICAL

        return SemanticType.NUMERIC

    # 7. String / object columns
    if (
        pd.api.types.is_string_dtype(series)
        or pd.api.types.is_object_dtype(series)
    ):

        values = non_null.astype(str).str.strip()

        date_keywords = (
            "date",
            "time",
            "timestamp",
            "created",
            "updated",
        )

        has_date_name = any(
            keyword in name
            for keyword in date_keywords
        )

        # -------------------------------------------------
        # Date detection
        # -------------------------------------------------
        #
        # Only attempt date parsing when:
        # 1. Column name strongly suggests date/time, OR
        # 2. Values are clearly date-like.
        #
        # This prevents warnings for columns such as:
        # city, country, product, category, etc.
        # -------------------------------------------------

        parse_ratio = 0.0

        if has_date_name:

            parsed_dates = pd.to_datetime(
                values,
                errors="coerce",
                format="mixed",
            )

            parse_ratio = parsed_dates.notna().mean()

        else:

            # Only attempt parsing for values that
            # look like dates.
            date_like_mask = values.str.match(
                r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}"
            )

            if date_like_mask.any():

                date_like_values = values[date_like_mask]

                parsed_dates = pd.to_datetime(
                    date_like_values,
                    errors="coerce",
                    format="mixed",
                )

                parse_ratio = (
                    len(parsed_dates.dropna())
                    / len(values)
                )

        # Date column
        if parse_ratio >= 0.95:
            return SemanticType.DATETIME

        # String statistics
        unique_ratio = (
            values.nunique()
            / len(values)
        )

        average_length = (
            values.str.len().mean()
        )

        # String identifier
        if (
            unique_ratio >= 0.95
            and average_length <= 30
        ):
            return SemanticType.IDENTIFIER

        # Long text
        if average_length > 60:
            return SemanticType.TEXT

        # Otherwise categorical
        return SemanticType.CATEGORICAL

    # Fallback
    return SemanticType.CATEGORICAL


def profile_column(
    series: pd.Series,
    column_name: str = "",
) -> ColumnProfile:
    """Create a detailed profile for a single dataset column."""

    # Remove missing values
    non_null = series.dropna()

    # Basic information
    n_rows = len(series)

    n_missing = int(
        series.isna().sum()
    )

    missing_pct = (
        (n_missing / n_rows) * 100
        if n_rows > 0
        else 0.0
    )

    n_unique = int(
        series.nunique(dropna=True)
    )

    unique_pct = (
        (n_unique / n_rows) * 100
        if n_rows > 0
        else 0.0
    )

    # Sample values
    sample_values = (
        series
        .dropna()
        .head(3)
        .tolist()
    )

    # Semantic type
    semantic_type = infer_semantic_type(
        series,
        column_name,
    )

    stats = {}

    # =====================================================
    # NUMERIC
    # =====================================================
    if semantic_type == SemanticType.NUMERIC:

        numeric = pd.to_numeric(
            series,
            errors="coerce",
        ).dropna()

        if not numeric.empty:

            q1 = float(
                numeric.quantile(0.25)
            )

            q3 = float(
                numeric.quantile(0.75)
            )

            iqr = q3 - q1

            stats = {
                "min": float(numeric.min()),
                "max": float(numeric.max()),
                "mean": float(numeric.mean()),
                "median": float(numeric.median()),
                "std": float(numeric.std()),
                "q1": q1,
                "q3": q3,
                "iqr": iqr,
                "skew": float(numeric.skew()),
                "n_zero": int(
                    (numeric == 0).sum()
                ),
                "n_negative": int(
                    (numeric < 0).sum()
                ),
            }

    # =====================================================
    # CATEGORICAL
    # =====================================================
    elif semantic_type == SemanticType.CATEGORICAL:

        value_counts = (
            series
            .dropna()
            .value_counts()
        )

        if not value_counts.empty:

            top_value = value_counts.index[0]

            top_count = int(
                value_counts.iloc[0]
            )

            non_null_count = len(
                series.dropna()
            )

            top_pct = (
                (top_count / non_null_count) * 100
                if non_null_count > 0
                else 0.0
            )

            stats = {
                "top_value": top_value,
                "top_count": top_count,
                "top_pct": float(top_pct),
                "value_counts": {
                    str(key): int(value)
                    for key, value
                    in value_counts.head(10).items()
                },
            }

    # =====================================================
    # DATETIME
    # =====================================================
    elif semantic_type == SemanticType.DATETIME:

        dates = pd.to_datetime(
            series,
            errors="coerce",
            format="mixed",
        ).dropna()

        if not dates.empty:

            min_date = dates.min()
            max_date = dates.max()

            span_days = (
                max_date - min_date
            ).days

            stats = {
                "min": min_date.isoformat(),
                "max": max_date.isoformat(),
                "span_days": int(span_days),
            }

    # =====================================================
    # BOOLEAN
    # =====================================================
    elif semantic_type == SemanticType.BOOLEAN:

        value_counts = (
            series
            .dropna()
            .value_counts()
        )

        stats = {
            "value_counts": {
                str(key): int(value)
                for key, value
                in value_counts.items()
            }
        }

    # =====================================================
    # IDENTIFIER
    # =====================================================
    elif semantic_type == SemanticType.IDENTIFIER:

        stats = {
            "n_unique": n_unique,
            "unique_pct": round(
                unique_pct,
                2,
            ),
        }

    # =====================================================
    # TEXT
    # =====================================================
    elif semantic_type == SemanticType.TEXT:

        text_values = (
            series
            .dropna()
            .astype(str)
        )

        if not text_values.empty:

            stats = {
                "avg_length": float(
                    text_values.str.len().mean()
                ),
                "min_length": int(
                    text_values.str.len().min()
                ),
                "max_length": int(
                    text_values.str.len().max()
                ),
            }

    # =====================================================
    # CONSTANT
    # =====================================================
    elif semantic_type == SemanticType.CONSTANT:

        stats = {
            "value": (
                non_null.iloc[0]
                if not non_null.empty
                else None
            )
        }

    # =====================================================
    # EMPTY
    # =====================================================
    elif semantic_type == SemanticType.EMPTY:

        stats = {}

    # =====================================================
    # RETURN
    # =====================================================
    return ColumnProfile(
        name=column_name,
        dtype=str(series.dtype),
        semantic_type=semantic_type.value,
        n_rows=n_rows,
        n_missing=n_missing,
        missing_pct=round(
            missing_pct,
            2,
        ),
        n_unique=n_unique,
        unique_pct=round(
            unique_pct,
            2,
        ),
        sample_values=sample_values,
        stats=stats,
    )