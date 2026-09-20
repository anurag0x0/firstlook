import pandas as pd
from enum import Enum


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


def infer_semantic_type(
    series: pd.Series,
    column_name: str = "",
) -> SemanticType:
    """Infer the semantic type of a pandas Series."""

    # Normalize column name
    name = column_name.lower().strip()

    # Remove missing values for value-based checks
    non_null = series.dropna()

    # 1. All values are NULL
    if non_null.empty:
        return SemanticType.EMPTY

    # 2. Only one unique value
    if non_null.nunique(dropna=True) == 1:
        return SemanticType.CONSTANT

    # 3. Pandas boolean dtype
    if pd.api.types.is_bool_dtype(series):
        return SemanticType.BOOLEAN

    # 4. Pandas datetime dtype
    if pd.api.types.is_datetime64_any_dtype(series):
        return SemanticType.DATETIME

    # 5. Exactly 2 unique boolean-like values
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
    # IMPORTANT:
    # Numeric check happens before string/date parsing.
    # Otherwise numbers like 1, 2, 3 may be interpreted as dates.
    if pd.api.types.is_numeric_dtype(series):

        unique_count = non_null.nunique()
        unique_ratio = unique_count / len(non_null)

        # 6a. Identifier detection
        id_keywords = ("id", "key", "code")

        has_id_name = any(
            keyword in name
            for keyword in id_keywords
        )

        if has_id_name and unique_ratio >= 0.95:
            return SemanticType.IDENTIFIER

        # 6b. Low-cardinality integer → categorical
        if (
            pd.api.types.is_integer_dtype(series)
            and unique_count <= 10
            and unique_ratio < 0.05
        ):
            return SemanticType.CATEGORICAL

        # 6c. Otherwise numeric
        return SemanticType.NUMERIC

    # 7. String / object columns
    if (
        pd.api.types.is_string_dtype(series)
        or pd.api.types.is_object_dtype(series)
    ):

        values = non_null.astype(str).str.strip()

        # Try parsing strings as dates
        parsed_dates = pd.to_datetime(
            values,
            errors="coerce",
        )

        parse_ratio = parsed_dates.notna().mean()

        # 7a. Column name suggests date/time
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

        if has_date_name and parse_ratio >= 0.95:
            return SemanticType.DATETIME

        # 7b. 95%+ values parse successfully as dates
        if parse_ratio >= 0.95:
            return SemanticType.DATETIME

        # Calculate string statistics
        unique_ratio = values.nunique() / len(values)
        average_length = values.str.len().mean()

        # 7c. Mostly unique + short strings → identifier
        if unique_ratio >= 0.95 and average_length <= 30:
            return SemanticType.IDENTIFIER

        # 7d. Long strings → free text
        if average_length > 60:
            return SemanticType.TEXT

        # 7e. Otherwise categorical
        return SemanticType.CATEGORICAL

    # Fallback
    return SemanticType.CATEGORICAL