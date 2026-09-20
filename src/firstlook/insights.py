from dataclasses import dataclass

import pandas as pd


@dataclass
class Finding:
    importance: int
    category: str
    headline: str
    detail: str
    recommendation: str


CORRELATION_DISCLAIMER = (
    "Correlation is not causation — this is a lead, not a conclusion."
)


def findings_from_quality(issues: list) -> list[Finding]:
    findings = []

    for issue in issues:
        severity = issue.severity.lower()

        if severity == "critical":
            importance = 1
        elif severity == "warning":
            importance = 2
        else:
            importance = 4

        column_text = (
            f" in {issue.column}"
            if issue.column
            else ""
        )

        findings.append(
            Finding(
                importance=importance,
                category="Data Quality",
                headline=(
                    f"{issue.kind.replace('_', ' ').title()}"
                    f"{column_text} needs attention."
                ),
                detail=issue.message,
                recommendation=issue.suggestion,
            )
        )

    return findings


def findings_from_trend(
    trend_results: dict,
    metric: str | None = None,
) -> list[Finding]:
    findings = []

    if not trend_results:
        return findings

    direction = trend_results.get("direction")
    change_pct = trend_results.get("change_pct")

    if (
        direction == "insufficient_data"
        or change_pct is None
    ):
        return findings

    if abs(change_pct) <= 5:
        return findings

    metric_text = metric or "The metric"

    if direction == "increasing":
        headline = (
            f"{metric_text} increased by "
            f"{abs(change_pct):.1f}% over time."
        )
        recommendation = (
            "Identify the drivers behind the increase "
            "and determine whether the trend is sustainable."
        )

    elif direction == "decreasing":
        headline = (
            f"{metric_text} decreased by "
            f"{abs(change_pct):.1f}% over time."
        )
        recommendation = (
            "Investigate the causes of the decline "
            "and identify segments or periods requiring attention."
        )

    else:
        return findings

    findings.append(
        Finding(
            importance=1,
            category="Trend",
            headline=headline,
            detail=(
                f"The observed trend is {direction}. "
                f"The estimated slope is "
                f"{trend_results.get('slope')}."
            ),
            recommendation=recommendation,
        )
    )

    return findings


def findings_from_segments(
    segment_results: pd.DataFrame,
) -> list[Finding]:
    findings = []

    if segment_results.empty:
        return findings

    for _, row in segment_results.iterrows():
        spread_pct = float(row["spread_pct"])

        if spread_pct > 50:
            importance = 1
        elif spread_pct > 10:
            importance = 2
        else:
            importance = 3

        segment = row["segment"]
        metric = row["metric"]

        highest_group = row["highest_group"]
        lowest_group = row["lowest_group"]

        highest_mean = row["highest_group_mean"]
        lowest_mean = row["lowest_group_mean"]

        findings.append(
            Finding(
                importance=importance,
                category="Segments",
                headline=(
                    f"{segment} creates a "
                    f"{spread_pct:.1f}% spread in {metric}."
                ),
                detail=(
                    f"{highest_group} has the highest average "
                    f"{metric} ({highest_mean:.2f}), while "
                    f"{lowest_group} has the lowest "
                    f"({lowest_mean:.2f})."
                ),
                recommendation=(
                    f"Investigate why {segment} differs across "
                    f"these groups and identify practices that "
                    f"could improve lower-performing groups."
                ),
            )
        )

    return findings


def findings_from_correlations(
    correlation_results: pd.DataFrame,
) -> list[Finding]:
    findings = []

    if correlation_results.empty:
        return findings

    strong_results = correlation_results[
        correlation_results["strength"] == "strong"
    ]

    for _, row in strong_results.iterrows():
        column_1 = row["column_1"]
        column_2 = row["column_2"]
        correlation = float(row["correlation"])

        direction = (
            "positive"
            if correlation > 0
            else "negative"
        )

        findings.append(
            Finding(
                importance=2,
                category="Relationships",
                headline=(
                    f"{column_1} and {column_2} "
                    f"have a strong {direction} relationship."
                ),
                detail=(
                    f"Pearson correlation is "
                    f"{correlation:.2f}, indicating a "
                    f"strong statistical relationship. "
                    f"{CORRELATION_DISCLAIMER}"
                ),
                recommendation=(
                    f"Investigate the relationship between "
                    f"{column_1} and {column_2} using business "
                    f"context or controlled analysis."
                ),
            )
        )

    return findings


def findings_from_shape(profile) -> list[Finding]:
    if profile is None:
        return []

    return [
        Finding(
            importance=5,
            category="Dataset Overview",
            headline=(
                f"Dataset contains {profile.n_rows:,} "
                f"rows and {profile.n_cols:,} columns."
            ),
            detail=(
                f"The dataset contains "
                f"{profile.n_duplicate_rows:,} duplicate "
                f"row(s) and uses approximately "
                f"{profile.memory_mb:.2f} MB of memory."
            ),
            recommendation=(
                "Use the profiling and quality results "
                "to understand the dataset before making "
                "business decisions."
            ),
        )
    ]


def generate_findings(
    issues: list | None = None,
    trend_results: dict | None = None,
    segment_results: pd.DataFrame | None = None,
    correlation_results: pd.DataFrame | None = None,
    profile=None,
    metric: str | None = None,
) -> list[Finding]:

    findings = []

    findings.extend(
        findings_from_quality(issues or [])
    )

    findings.extend(
        findings_from_trend(
            trend_results or {},
            metric=metric,
        )
    )

    if segment_results is not None:
        findings.extend(
            findings_from_segments(segment_results)
        )

    if correlation_results is not None:
        findings.extend(
            findings_from_correlations(
                correlation_results
            )
        )

    findings.extend(
        findings_from_shape(profile)
    )

    findings.sort(
        key=lambda finding: finding.importance
    )

    return findings