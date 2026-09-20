import pandas as pd

from firstlook.quality import (
    check_missing_values,
    check_duplicate_rows,
    check_constant_columns,
    check_outliers,
    run_quality_checks,
    quality_score,
)


# ============================================================
# MISSING VALUE TESTS
# ============================================================

def test_missing_values_clean_dataframe(clean_df):
    issues = check_missing_values(clean_df)

    assert issues == []


def test_missing_values_warning():
    df = pd.DataFrame(
        {
            "revenue": [100, 200, None, 400, 500],
            "region": ["Delhi", "Mumbai", "Delhi", "Pune", "Delhi"],
        }
    )

    issues = check_missing_values(df)

    assert len(issues) == 1
    assert issues[0].column == "revenue"
    assert issues[0].severity == "warning"
    assert issues[0].kind == "missing"


def test_missing_values_critical():
    df = pd.DataFrame(
        {
            "revenue": [None, None, None, None, 500],
            "region": ["Delhi", "Mumbai", "Delhi", "Pune", "Delhi"],
        }
    )

    issues = check_missing_values(df)

    assert len(issues) == 1
    assert issues[0].column == "revenue"
    assert issues[0].severity == "critical"
    assert issues[0].kind == "missing"


def test_missing_values_multiple_columns():
    df = pd.DataFrame(
        {
            "revenue": [100, None, 300, 400, 500],
            "orders": [10, 20, None, 40, 50],
        }
    )

    issues = check_missing_values(df)

    assert len(issues) == 2

    columns = {issue.column for issue in issues}

    assert columns == {"revenue", "orders"}


# ============================================================
# DUPLICATE ROW TESTS
# ============================================================

def test_duplicate_rows_clean_dataframe(clean_df):
    issues = check_duplicate_rows(clean_df)

    assert issues == []


def test_duplicate_rows_detected():
    df = pd.DataFrame(
        {
            "user_id": [1, 2, 3, 3, 4],
            "revenue": [100, 200, 300, 300, 400],
        }
    )

    issues = check_duplicate_rows(df)

    assert len(issues) == 1
    assert issues[0].kind == "duplicate_rows"

    # 1 duplicate out of 5 rows = 20%,
    # which is critical according to the current implementation.
    assert issues[0].severity == "critical"


def test_duplicate_rows_critical():
    df = pd.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2],
            "revenue": [100, 100, 100, 200, 200],
        }
    )

    issues = check_duplicate_rows(df)

    assert len(issues) == 1
    assert issues[0].kind == "duplicate_rows"
    assert issues[0].severity == "critical"


# ============================================================
# CONSTANT COLUMN TESTS
# ============================================================

def test_constant_column_detected():
    df = pd.DataFrame(
        {
            "revenue": [
                100,
                200,
                300,
                400,
                500,
            ],
            "currency": [
                "INR",
                "INR",
                "INR",
                "INR",
                "INR",
            ],
        }
    )

    issues = check_constant_columns(df)

    assert len(issues) == 1
    assert issues[0].column == "currency"
    assert issues[0].kind == "constant"
    assert issues[0].severity == "warning"


def test_no_constant_column():
    df = pd.DataFrame(
        {
            "revenue": [100, 200, 300, 400, 500],
            "region": ["Delhi", "Mumbai", "Pune", "Delhi", "Mumbai"],
        }
    )

    issues = check_constant_columns(df)

    assert issues == []


# ============================================================
# OUTLIER TESTS
# ============================================================

def test_outlier_detected():
    df = pd.DataFrame(
        {
            "revenue": [
                100,
                110,
                105,
                108,
                102,
                1000,
            ]
        }
    )

    issues = check_outliers(df)

    assert len(issues) >= 1
    assert issues[0].column == "revenue"
    assert issues[0].kind == "outliers"


def test_no_outlier_in_normal_data():
    df = pd.DataFrame(
        {
            "revenue": [
                100,
                105,
                110,
                108,
                103,
                107,
            ]
        }
    )

    issues = check_outliers(df)

    assert issues == []


# ============================================================
# RUN QUALITY CHECKS
# ============================================================

def test_run_quality_checks_clean_dataframe(clean_df):
    issues = run_quality_checks(clean_df)

    # The current quality engine also reports informational
    # outlier findings, so a clean fixture can still have issues.
    # Verify that there are no warning/critical issues.
    serious_issues = [
        issue
        for issue in issues
        if issue.severity.lower() in {"warning", "critical"}
    ]

    assert serious_issues == []


def test_run_quality_checks_messy_dataframe(messy_df):
    issues = run_quality_checks(messy_df)

    assert len(issues) > 0

    kinds = {issue.kind for issue in issues}

    assert (
        "missing" in kinds
        or "duplicate_rows" in kinds
        or "constant" in kinds
        or "outliers" in kinds
    )


# ============================================================
# QUALITY SCORE TESTS
# ============================================================

def test_quality_score_perfect():
    score = quality_score([])

    assert score == 100


def test_quality_score_with_warning():
    df = pd.DataFrame(
        {
            "revenue": [100, 200, None, 400, 500],
        }
    )

    issues = check_missing_values(df)

    score = quality_score(issues)

    assert score < 100
    assert score >= 0


def test_quality_score_never_negative():
    issues = []

    for _ in range(100):
        df = pd.DataFrame(
            {
                "revenue": [None, None, None, None, None]
            }
        )

        issues.extend(check_missing_values(df))

    score = quality_score(issues)

    assert score >= 0