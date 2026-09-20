import pandas as pd


def top_correlations(
    df: pd.DataFrame,
    threshold: float = 0.3,
) -> pd.DataFrame:
    """
    Find pairwise Pearson correlations between numeric columns.

    Only correlations with absolute value >= threshold
    are returned.
    """

    result_columns = [
        "column_1",
        "column_2",
        "correlation",
        "abs_correlation",
        "strength",
    ]

    numeric_df = df.select_dtypes(include="number")

    if numeric_df.shape[1] < 2:
        return pd.DataFrame(columns=result_columns)

    correlation_matrix = numeric_df.corr(
        method="pearson"
    )

    results = []

    columns = correlation_matrix.columns

    for i in range(len(columns)):
        for j in range(i + 1, len(columns)):

            column_1 = columns[i]
            column_2 = columns[j]

            correlation = correlation_matrix.loc[
                column_1,
                column_2,
            ]

            if pd.isna(correlation):
                continue

            correlation = float(correlation)
            abs_correlation = abs(correlation)

            if abs_correlation < threshold:
                continue

            if abs_correlation >= 0.7:
                strength = "strong"
            elif abs_correlation >= 0.5:
                strength = "moderate"
            else:
                strength = "weak"

            results.append(
                {
                    "column_1": column_1,
                    "column_2": column_2,
                    "correlation": correlation,
                    "abs_correlation": abs_correlation,
                    "strength": strength,
                }
            )

    if not results:
        return pd.DataFrame(columns=result_columns)

    result = pd.DataFrame(
        results,
        columns=result_columns,
    )

    result = result.sort_values(
        by="abs_correlation",
        ascending=False,
    ).reset_index(drop=True)

    return result


def segment_comparison(
    df: pd.DataFrame,
    metric: str,
    segment: str,
    min_group_size: int = 5,
) -> pd.DataFrame:
    """
    Compare a numeric metric across different segments.

    Example:
        segment = "region"
        metric = "revenue"

    Returns:
        segment
        mean
        median
        std
        count
        vs_overall_pct

    Groups smaller than min_group_size are excluded.
    """

    result_columns = [
        segment,
        "mean",
        "median",
        "std",
        "count",
        "vs_overall_pct",
    ]

    # ---------------------------------------------------------
    # Validate columns
    # ---------------------------------------------------------

    if metric not in df.columns:
        raise ValueError(
            f"Metric column '{metric}' not found in DataFrame."
        )

    if segment not in df.columns:
        raise ValueError(
            f"Segment column '{segment}' not found in DataFrame."
        )

    # ---------------------------------------------------------
    # Metric should be numeric
    # ---------------------------------------------------------

    if not pd.api.types.is_numeric_dtype(df[metric]):
        raise ValueError(
            f"Metric column '{metric}' must be numeric."
        )

    # ---------------------------------------------------------
    # Overall mean
    # ---------------------------------------------------------

    overall_mean = df[metric].mean()

    # ---------------------------------------------------------
    # Group by segment
    # ---------------------------------------------------------

    grouped = (
        df.groupby(segment, dropna=False)[metric]
        .agg(
            mean="mean",
            median="median",
            std="std",
            count="count",
        )
        .reset_index()
    )

    # ---------------------------------------------------------
    # Minimum group size
    # ---------------------------------------------------------

    grouped = grouped[
        grouped["count"] >= min_group_size
    ].copy()

    # ---------------------------------------------------------
    # No valid groups
    # ---------------------------------------------------------

    if grouped.empty:
        return pd.DataFrame(columns=result_columns)

    # ---------------------------------------------------------
    # Compare each group against overall mean
    # ---------------------------------------------------------

    if pd.isna(overall_mean) or overall_mean == 0:
        grouped["vs_overall_pct"] = pd.NA
    else:
        grouped["vs_overall_pct"] = (
            (
                grouped["mean"] - overall_mean
            )
            / overall_mean
            * 100
        )

    # ---------------------------------------------------------
    # Round numeric output
    # ---------------------------------------------------------

    grouped["mean"] = grouped["mean"].round(2)
    grouped["median"] = grouped["median"].round(2)
    grouped["std"] = grouped["std"].round(2)
    grouped["vs_overall_pct"] = grouped[
        "vs_overall_pct"
    ].round(2)

    return grouped[result_columns].reset_index(drop=True)