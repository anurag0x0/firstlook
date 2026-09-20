from dataclasses import dataclass

import pandas as pd


@dataclass
class Issue:
    """Represents a single data-quality issue."""

    severity: str
    kind: str
    column: str | None
    message: str
    suggestion: str
    detail: dict


# =========================================================
# MISSING VALUES
# =========================================================

def check_missing_values(
    df: pd.DataFrame,
    warning_threshold: float = 20.0,
    critical_threshold: float = 50.0,
) -> list[Issue]:
    """
    Detect columns with significant missing values.
    """

    issues = []

    for column in df.columns:

        missing_count = int(
            df[column].isna().sum()
        )

        if missing_count == 0:
            continue

        missing_pct = (
            missing_count / len(df) * 100
            if len(df) > 0
            else 0.0
        )

        if missing_pct >= critical_threshold:

            severity = "critical"

            message = (
                f"{column} has "
                f"{missing_pct:.1f}% missing values"
            )

            suggestion = (
                "Investigate the source data; "
                "consider recovering the missing values "
                "or dropping the column if it is not required."
            )

        elif missing_pct >= warning_threshold:

            severity = "warning"

            message = (
                f"{column} has "
                f"{missing_pct:.1f}% missing values"
            )

            suggestion = (
                "Investigate the cause of missing values "
                "and choose an appropriate imputation "
                "or data-cleaning strategy."
            )

        else:
            continue

        issues.append(
            Issue(
                severity=severity,
                kind="missing",
                column=str(column),
                message=message,
                suggestion=suggestion,
                detail={
                    "missing_count": missing_count,
                    "missing_pct": round(
                        missing_pct,
                        2,
                    ),
                    "total_rows": len(df),
                },
            )
        )

    return issues


# =========================================================
# DUPLICATE ROWS
# =========================================================

def check_duplicate_rows(
    df: pd.DataFrame,
    warning_threshold: float = 5.0,
    critical_threshold: float = 20.0,
) -> list[Issue]:
    """
    Detect duplicate rows in the dataset.
    """

    issues = []

    duplicate_count = int(
        df.duplicated().sum()
    )

    if duplicate_count == 0:
        return issues

    duplicate_pct = (
        duplicate_count / len(df) * 100
        if len(df) > 0
        else 0.0
    )

    if duplicate_pct >= critical_threshold:

        severity = "critical"

        message = (
            f"Dataset contains "
            f"{duplicate_count} duplicate rows "
            f"({duplicate_pct:.1f}%)"
        )

        suggestion = (
            "Investigate why duplicate records exist "
            "and remove them if they do not represent "
            "legitimate repeated events."
        )

    elif duplicate_pct >= warning_threshold:

        severity = "warning"

        message = (
            f"Dataset contains "
            f"{duplicate_count} duplicate rows "
            f"({duplicate_pct:.1f}%)"
        )

        suggestion = (
            "Review duplicate records and remove them "
            "if they are accidental duplicates."
        )

    else:

        severity = "info"

        message = (
            f"Dataset contains "
            f"{duplicate_count} duplicate rows "
            f"({duplicate_pct:.1f}%)"
        )

        suggestion = (
            "Review duplicate records to confirm "
            "whether they are expected."
        )

    issues.append(
        Issue(
            severity=severity,
            kind="duplicate_rows",
            column=None,
            message=message,
            suggestion=suggestion,
            detail={
                "duplicate_count": duplicate_count,
                "duplicate_pct": round(
                    duplicate_pct,
                    2,
                ),
                "total_rows": len(df),
            },
        )
    )

    return issues


# =========================================================
# CONSTANT COLUMNS
# =========================================================

def check_constant_columns(
    df: pd.DataFrame,
) -> list[Issue]:
    """
    Detect columns containing only one unique value.
    """

    issues = []

    for column in df.columns:

        non_null = df[column].dropna()

        if non_null.empty:
            continue

        unique_count = int(
            non_null.nunique()
        )

        if unique_count != 1:
            continue

        constant_value = non_null.iloc[0]

        issues.append(
            Issue(
                severity="warning",
                kind="constant",
                column=str(column),
                message=(
                    f"{column} contains only one "
                    f"unique value: {constant_value}"
                ),
                suggestion=(
                    "Verify whether this column provides "
                    "any analytical value; drop it if it "
                    "does not vary."
                ),
                detail={
                    "unique_count": unique_count,
                    "value": constant_value,
                    "non_null_count": len(non_null),
                },
            )
        )

    return issues


# =========================================================
# OUTLIERS
# =========================================================

def check_outliers(
    df: pd.DataFrame,
    min_rows: int = 4,
) -> list[Issue]:
    """
    Detect numeric outliers using the IQR method.

    Outlier rule:

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
    """

    issues = []

    for column in df.select_dtypes(
        include="number"
    ).columns:

        numeric = pd.to_numeric(
            df[column],
            errors="coerce",
        ).dropna()

        if len(numeric) < min_rows:
            continue

        q1 = float(
            numeric.quantile(0.25)
        )

        q3 = float(
            numeric.quantile(0.75)
        )

        iqr = q3 - q1

        # If IQR is zero, normal IQR-based
        # outlier detection is not useful.
        if iqr <= 0:
            continue

        lower_bound = q1 - (
            1.5 * iqr
        )

        upper_bound = q3 + (
            1.5 * iqr
        )

        outliers = numeric[
            (numeric < lower_bound)
            | (numeric > upper_bound)
        ]

        outlier_count = len(outliers)

        if outlier_count == 0:
            continue

        outlier_pct = (
            outlier_count / len(numeric) * 100
        )

        if outlier_pct >= 20:

            severity = "critical"

        elif outlier_pct >= 5:

            severity = "warning"

        else:

            severity = "info"

        issues.append(
            Issue(
                severity=severity,
                kind="outliers",
                column=str(column),
                message=(
                    f"{column} contains "
                    f"{outlier_count} outlier(s) "
                    f"({outlier_pct:.1f}% of non-null values)"
                ),
                suggestion=(
                    "Investigate the extreme values "
                    "and verify whether they are genuine "
                    "observations or data-entry errors."
                ),
                detail={
                    "outlier_count": outlier_count,
                    "outlier_pct": round(
                        outlier_pct,
                        2,
                    ),
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "lower_bound": lower_bound,
                    "upper_bound": upper_bound,
                    "outlier_values": outliers.tolist(),
                },
            )
        )

    return issues


# =========================================================
# RUN ALL QUALITY CHECKS
# =========================================================

def run_quality_checks(
    df: pd.DataFrame,
) -> list[Issue]:
    """
    Run all available data-quality checks.
    """

    issues = []

    issues.extend(
        check_missing_values(df)
    )

    issues.extend(
        check_duplicate_rows(df)
    )

    issues.extend(
        check_constant_columns(df)
    )

    issues.extend(
        check_outliers(df)
    )

    return issues


# =========================================================
# QUALITY SCORE
# =========================================================

def quality_score(
    issues: list[Issue],
) -> int:
    """
    Calculate an overall data-quality score.

    Score:
        100 - (12 × critical)
            - (5 × warning)
            - (1 × info)

    The final score is clamped to a minimum of 0.
    """

    critical_count = sum(
        1
        for issue in issues
        if issue.severity.lower() == "critical"
    )

    warning_count = sum(
        1
        for issue in issues
        if issue.severity.lower() == "warning"
    )

    info_count = sum(
        1
        for issue in issues
        if issue.severity.lower() == "info"
    )

    score = (
        100
        - (12 * critical_count)
        - (5 * warning_count)
        - (1 * info_count)
    )

    return max(0, score)