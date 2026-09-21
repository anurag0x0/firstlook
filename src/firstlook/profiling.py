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


@dataclass
class DatasetProfile:
    """Profile information for an entire dataset."""

    n_rows: int
    n_cols: int
    n_duplicate_rows: int
    memory_mb: float
    columns: list[ColumnProfile]

    def by_type(self, t):
        """Return all column profiles matching a semantic type."""

        if isinstance(t, SemanticType):
            t = t.value

        return [
            column
            for column in self.columns
            if column.semantic_type == t
        ]

    def get(self, name):
        """Return the profile for a column by name."""

        for column in self.columns:
            if column.name == name:
                return column

        return None


def infer_semantic_type(
    series: pd.Series,
    column_name: str = "",
) -> SemanticType:
    """Infer the semantic type of a pandas Series."""

    name = column_name.lower().strip()

    non_null = series.dropna()

    # =========================================================
    # 1. EMPTY COLUMN
    # =========================================================
    if non_null.empty:
        return SemanticType.EMPTY

    # =========================================================
    # 2. CONSTANT COLUMN
    # =========================================================
    if non_null.nunique(dropna=True) == 1:
        return SemanticType.CONSTANT

    # =========================================================
    # 3. BOOLEAN DTYPE
    # =========================================================
    if pd.api.types.is_bool_dtype(series):
        return SemanticType.BOOLEAN

    # =========================================================
    # 4. DATETIME DTYPE
    # =========================================================
    if pd.api.types.is_datetime64_any_dtype(series):
        return SemanticType.DATETIME

    # =========================================================
    # 5. BOOLEAN-LIKE VALUES
    # =========================================================
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

    # =========================================================
    # 6. NUMERIC COLUMNS
    # =========================================================
    if pd.api.types.is_numeric_dtype(series):

        unique_count = non_null.nunique()
        unique_ratio = unique_count / len(non_null)

        id_keywords = (
            "id",
            "key",
            "code",
            "uuid",
            "guid",
        )

        has_id_name = any(
            keyword in name
            for keyword in id_keywords
        )

        # -----------------------------------------------------
        # Numeric identifier
        # -----------------------------------------------------
        if (
            has_id_name
            and unique_ratio >= 0.95
        ):
            return SemanticType.IDENTIFIER

        # -----------------------------------------------------
        # Low-cardinality integer category
        # -----------------------------------------------------
        if (
            pd.api.types.is_integer_dtype(series)
            and unique_count <= 10
            and unique_ratio < 0.05
        ):
            return SemanticType.CATEGORICAL

        return SemanticType.NUMERIC

    # =========================================================
    # 7. STRING / OBJECT COLUMNS
    # =========================================================
    if (
        pd.api.types.is_string_dtype(series)
        or pd.api.types.is_object_dtype(series)
    ):

        values = non_null.astype(str).str.strip()

        # -----------------------------------------------------
        # Date keywords
        # -----------------------------------------------------
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

        # -----------------------------------------------------
        # Date detection
        # -----------------------------------------------------
        parse_ratio = 0.0

        if has_date_name:

            parsed_dates = pd.to_datetime(
                values,
                errors="coerce",
                format="mixed",
            )

            parse_ratio = parsed_dates.notna().mean()

        else:

            date_like_mask = values.str.match(
                r"^\d{4}[-/]\d{1,2}[-/]\d{1,2}"
            )

            if date_like_mask.any():

                date_like_values = values[
                    date_like_mask
                ]

                parsed_dates = pd.to_datetime(
                    date_like_values,
                    errors="coerce",
                    format="mixed",
                )

                parse_ratio = (
                    len(parsed_dates.dropna())
                    / len(values)
                )

        # -----------------------------------------------------
        # Datetime
        # -----------------------------------------------------
        if parse_ratio >= 0.95:
            return SemanticType.DATETIME

        # =====================================================
        # STRING STATISTICS
        # =====================================================

        unique_ratio = (
            values.nunique()
            / len(values)
        )

        average_length = (
            values.str.len().mean()
        )

        # =====================================================
        # STRING IDENTIFIER DETECTION
        # =====================================================
        #
        # Important:
        #
        # A transactional dataset can contain repeated
        # identifiers.
        #
        # Example:
        #
        # customer_id
        #
        # C00001
        # C00001
        # C00001
        # C00002
        #
        # Therefore we should NOT require 95% uniqueness
        # when the column name clearly indicates an ID.
        #
        # =====================================================

        identifier_name = (
            name == "id"
            or name.endswith("_id")
            or name.endswith("_key")
            or name.endswith("_code")
            or name.endswith("_uuid")
            or name.endswith("_guid")
            or name.startswith("id_")
            or name.startswith("key_")
            or name.startswith("code_")
            or name.startswith("uuid_")
            or name.startswith("guid_")
        )

        # -----------------------------------------------------
        # Named identifier
        # -----------------------------------------------------
        if (
            identifier_name
            and average_length <= 50
        ):
            return SemanticType.IDENTIFIER

        # -----------------------------------------------------
        # High-cardinality string identifier
        # -----------------------------------------------------
        if (
            unique_ratio >= 0.95
            and average_length <= 50
        ):
            return SemanticType.IDENTIFIER

        # =====================================================
        # LONG TEXT
        # =====================================================
        if average_length > 60:
            return SemanticType.TEXT

        # =====================================================
        # CATEGORICAL
        # =====================================================
        return SemanticType.CATEGORICAL

    # =========================================================
    # 8. FALLBACK
    # =========================================================
    return SemanticType.CATEGORICAL


def profile_column(
    series: pd.Series,
    column_name: str = "",
) -> ColumnProfile:
    """Create a detailed profile for a single dataset column."""

    # =========================================================
    # REMOVE MISSING VALUES
    # =========================================================
    non_null = series.dropna()

    # =========================================================
    # BASIC INFORMATION
    # =========================================================
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

    # =========================================================
    # SAMPLE VALUES
    # =========================================================
    sample_values = (
        series
        .dropna()
        .head(3)
        .tolist()
    )

    # =========================================================
    # SEMANTIC TYPE
    # =========================================================
    semantic_type = infer_semantic_type(
        series,
        column_name,
    )

    stats = {}

    # =========================================================
    # NUMERIC
    # =========================================================
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

    # =========================================================
    # CATEGORICAL
    # =========================================================
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

    # =========================================================
    # DATETIME
    # =========================================================
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

    # =========================================================
    # BOOLEAN
    # =========================================================
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

    # =========================================================
    # IDENTIFIER
    # =========================================================
    elif semantic_type == SemanticType.IDENTIFIER:

        stats = {
            "n_unique": n_unique,
            "unique_pct": round(
                unique_pct,
                2,
            ),
        }

    # =========================================================
    # TEXT
    # =========================================================
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

    # =========================================================
    # CONSTANT
    # =========================================================
    elif semantic_type == SemanticType.CONSTANT:

        stats = {
            "value": (
                non_null.iloc[0]
                if not non_null.empty
                else None
            )
        }

    # =========================================================
    # EMPTY
    # =========================================================
    elif semantic_type == SemanticType.EMPTY:

        stats = {}

    # =========================================================
    # RETURN COLUMN PROFILE
    # =========================================================
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


def profile_dataframe(
    df: pd.DataFrame,
) -> DatasetProfile:
    """Create a profile for an entire pandas DataFrame."""

    # =========================================================
    # DATASET BASIC INFORMATION
    # =========================================================

    n_rows = len(df)

    n_cols = len(df.columns)

    # =========================================================
    # DUPLICATE ROWS
    # =========================================================

    n_duplicate_rows = int(
        df.duplicated().sum()
    )

    # =========================================================
    # MEMORY USAGE
    # =========================================================

    memory_bytes = df.memory_usage(
        deep=True
    ).sum()

    memory_mb = (
        memory_bytes / (1024 ** 2)
    )

    # =========================================================
    # PROFILE EVERY COLUMN
    # =========================================================

    columns = []

    for column_name in df.columns:

        column_profile = profile_column(
            df[column_name],
            str(column_name),
        )

        columns.append(
            column_profile
        )

    # =========================================================
    # RETURN DATASET PROFILE
    # =========================================================

    return DatasetProfile(
        n_rows=n_rows,
        n_cols=n_cols,
        n_duplicate_rows=n_duplicate_rows,
        memory_mb=round(
            memory_mb,
            4,
        ),
        columns=columns,
    )