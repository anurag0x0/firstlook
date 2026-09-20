import numpy as np
import pandas as pd


def top_correlations(
    df: pd.DataFrame,
    threshold: float = 0.3,
) -> pd.DataFrame:
    """
    Find pairwise Pearson correlations between numeric columns.
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
    """

    result_columns = [
        segment,
        "mean",
        "median",
        "std",
        "count",
        "vs_overall_pct",
    ]

    if metric not in df.columns:
        raise ValueError(
            f"Metric column '{metric}' not found in DataFrame."
        )

    if segment not in df.columns:
        raise ValueError(
            f"Segment column '{segment}' not found in DataFrame."
        )

    if not pd.api.types.is_numeric_dtype(df[metric]):
        raise ValueError(
            f"Metric column '{metric}' must be numeric."
        )

    overall_mean = df[metric].mean()

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

    grouped = grouped[
        grouped["count"] >= min_group_size
    ].copy()

    if grouped.empty:
        return pd.DataFrame(columns=result_columns)

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

    grouped["mean"] = grouped["mean"].round(2)
    grouped["median"] = grouped["median"].round(2)
    grouped["std"] = grouped["std"].round(2)
    grouped["vs_overall_pct"] = grouped[
        "vs_overall_pct"
    ].round(2)

    return grouped[
        result_columns
    ].reset_index(drop=True)


def rank_segment_drivers(
    df: pd.DataFrame,
    metric: str,
    categorical_columns: list[str] | None = None,
    min_group_size: int = 5,
) -> pd.DataFrame:
    """
    Rank categorical columns by how strongly they split a metric.
    """

    result_columns = [
        "segment",
        "metric",
        "spread_pct",
        "highest_group",
        "highest_group_mean",
        "lowest_group",
        "lowest_group_mean",
        "n_groups",
    ]

    if metric not in df.columns:
        raise ValueError(
            f"Metric column '{metric}' not found in DataFrame."
        )

    if not pd.api.types.is_numeric_dtype(df[metric]):
        raise ValueError(
            f"Metric column '{metric}' must be numeric."
        )

    overall_mean = df[metric].mean()

    if pd.isna(overall_mean) or overall_mean == 0:
        return pd.DataFrame(columns=result_columns)

    if categorical_columns is None:
        categorical_columns = (
            df.select_dtypes(
                include=["object", "category", "string"]
            )
            .columns
            .tolist()
        )

    results = []

    for segment in categorical_columns:

        if segment == metric:
            continue

        if segment not in df.columns:
            continue

        comparison = segment_comparison(
            df=df,
            metric=metric,
            segment=segment,
            min_group_size=min_group_size,
        )

        if comparison.empty:
            continue

        if len(comparison) < 2:
            continue

        highest_idx = comparison["mean"].idxmax()
        lowest_idx = comparison["mean"].idxmin()

        highest_row = comparison.loc[highest_idx]
        lowest_row = comparison.loc[lowest_idx]

        highest_mean = float(
            highest_row["mean"]
        )

        lowest_mean = float(
            lowest_row["mean"]
        )

        spread_pct = (
            (highest_mean - lowest_mean)
            / overall_mean
            * 100
        )

        results.append(
            {
                "segment": segment,
                "metric": metric,
                "spread_pct": round(
                    spread_pct,
                    2,
                ),
                "highest_group": highest_row[segment],
                "highest_group_mean": highest_mean,
                "lowest_group": lowest_row[segment],
                "lowest_group_mean": lowest_mean,
                "n_groups": len(comparison),
            }
        )

    if not results:
        return pd.DataFrame(columns=result_columns)

    result = pd.DataFrame(
        results,
        columns=result_columns,
    )

    result = result.sort_values(
        by="spread_pct",
        ascending=False,
    ).reset_index(drop=True)

    return result


def trend_over_time(
    df: pd.DataFrame,
    date_column: str,
    metric: str,
    freq: str = "ME",
) -> pd.DataFrame:
    """
    Aggregate a numeric metric over time.

    Default frequency is monthly end ('ME').

    Returns:
        period
        value
    """

    result_columns = [
        "period",
        "value",
    ]

    # ---------------------------------------------------------
    # Validate columns
    # ---------------------------------------------------------

    if date_column not in df.columns:
        raise ValueError(
            f"Date column '{date_column}' not found in DataFrame."
        )

    if metric not in df.columns:
        raise ValueError(
            f"Metric column '{metric}' not found in DataFrame."
        )

    # ---------------------------------------------------------
    # Validate metric
    # ---------------------------------------------------------

    if not pd.api.types.is_numeric_dtype(df[metric]):
        raise ValueError(
            f"Metric column '{metric}' must be numeric."
        )

    # ---------------------------------------------------------
    # Copy data
    # ---------------------------------------------------------

    working_df = df[
        [date_column, metric]
    ].copy()

    # ---------------------------------------------------------
    # Convert date column
    # ---------------------------------------------------------

    working_df[date_column] = pd.to_datetime(
        working_df[date_column],
        errors="coerce",
    )

    # Remove invalid dates
    working_df = working_df.dropna(
        subset=[date_column]
    )

    # Remove rows with missing metric
    working_df = working_df.dropna(
        subset=[metric]
    )

    if working_df.empty:
        return pd.DataFrame(
            columns=result_columns
        )

    # ---------------------------------------------------------
    # Monthly aggregation
    # ---------------------------------------------------------

    trend = (
        working_df
        .set_index(date_column)[metric]
        .resample(freq)
        .mean()
        .dropna()
        .reset_index()
    )

    trend = trend.rename(
        columns={
            date_column: "period",
            metric: "value",
        }
    )

    return trend[result_columns].reset_index(
        drop=True
    )


def trend_direction(
    trend: pd.DataFrame,
) -> dict:
    """
    Determine trend direction from a time-series DataFrame.

    Rules:
        > +5%  -> increasing
        < -5%  -> decreasing
        else   -> stable

    Fewer than 3 points -> insufficient_data.
    """

    result = {
        "direction": "insufficient_data",
        "slope": None,
        "change_pct": None,
        "n_points": 0,
    }

    # ---------------------------------------------------------
    # Validate input
    # ---------------------------------------------------------

    if trend.empty:
        return result

    if "value" not in trend.columns:
        raise ValueError(
            "Trend DataFrame must contain a 'value' column."
        )

    values = pd.to_numeric(
        trend["value"],
        errors="coerce",
    ).dropna()

    n = len(values)

    result["n_points"] = n

    # ---------------------------------------------------------
    # Need at least 3 points
    # ---------------------------------------------------------

    if n < 3:
        return result

    values_array = values.to_numpy(
        dtype=float
    )

    x = np.arange(n)

    # ---------------------------------------------------------
    # Calculate slope
    # ---------------------------------------------------------

    slope = float(
        np.polyfit(
            x,
            values_array,
            1,
        )[0]
    )

    # ---------------------------------------------------------
    # Calculate percentage change
    # ---------------------------------------------------------

    first = float(
        values_array[0]
    )

    last = float(
        values_array[-1]
    )

    if first == 0:
        change_pct = None
        direction = "insufficient_data"

    else:
        change_pct = (
            (last - first)
            / abs(first)
            * 100
        )

        if change_pct > 5:
            direction = "increasing"

        elif change_pct < -5:
            direction = "decreasing"

        else:
            direction = "stable"

    result.update(
        {
            "direction": direction,
            "slope": round(slope, 4),
            "change_pct": (
                round(change_pct, 2)
                if change_pct is not None
                else None
            ),
        }
    )

    return result