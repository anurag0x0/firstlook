from pathlib import Path

import pandas as pd


def _safe_markdown_table(
    df: pd.DataFrame | None,
) -> str:
    """Convert a DataFrame to a Markdown table safely."""

    if df is None or df.empty:
        return "_No data available._"

    return df.to_markdown(index=False)


def _format_finding(
    finding,
    number: int,
) -> str:
    """Format one Finding for the report."""

    return (
        f"### {number}. {finding.headline}\n\n"
        f"**Category:** {finding.category}  \n"
        f"**Importance:** {finding.importance}/5\n\n"
        f"{finding.detail}\n\n"
        f"**Recommendation:** {finding.recommendation}\n"
    )


def build_markdown_report(
    profile,
    quality_score_value: int,
    issues: list,
    findings: list,
    cleaning_log: list | None = None,
    column_reference: pd.DataFrame | None = None,
    numeric_summary: pd.DataFrame | None = None,
    correlations: pd.DataFrame | None = None,
    segment_analysis: pd.DataFrame | None = None,
    trend_over_time: pd.DataFrame | None = None,
    primary_metric: str | None = None,
) -> str:
    """
    Build a complete Markdown business report.
    """

    rows = profile.n_rows
    columns = profile.n_cols
    issue_count = len(issues)

    metric_text = (
        primary_metric
        if primary_metric
        else "Not specified"
    )

    sections = []

    # ---------------------------------------------------------
    # Title
    # ---------------------------------------------------------

    sections.append(
        "# FirstLook — Automated Data Analysis Report"
    )

    # ---------------------------------------------------------
    # 1. Executive Summary
    # ---------------------------------------------------------

    sections.append(
        "## 1. Executive Summary\n\n"
        f"- **Rows:** {rows:,}\n"
        f"- **Columns:** {columns:,}\n"
        f"- **Quality Score:** {quality_score_value}/100\n"
        f"- **Primary Metric:** {metric_text}\n"
        f"- **Issues Detected:** {issue_count}\n"
    )

    # ---------------------------------------------------------
    # 2. Key Findings
    # ---------------------------------------------------------

    findings_section = [
        "## 2. Key Findings"
    ]

    top_findings = findings[:8]

    if not top_findings:
        findings_section.append(
            "_No significant findings were generated._"
        )
    else:
        for number, finding in enumerate(
            top_findings,
            start=1,
        ):
            findings_section.append(
                _format_finding(
                    finding,
                    number,
                )
            )

    sections.append(
        "\n".join(findings_section)
    )

    # ---------------------------------------------------------
    # 3. Data Quality
    # ---------------------------------------------------------

    quality_section = [
        "## 3. Data Quality"
    ]

    if not issues:
        quality_section.append(
            "No data quality issues were detected."
        )
    else:
        quality_rows = []

        for issue in issues:
            quality_rows.append(
                {
                    "Severity": issue.severity,
                    "Issue": issue.kind,
                    "Column": (
                        issue.column
                        if issue.column
                        else "-"
                    ),
                    "Message": issue.message,
                    "Suggested Fix": issue.suggestion,
                }
            )

        quality_df = pd.DataFrame(
            quality_rows
        )

        quality_section.append(
            _safe_markdown_table(
                quality_df
            )
        )

    sections.append(
        "\n".join(quality_section)
    )

    # ---------------------------------------------------------
    # 4. Cleaning Applied
    # ---------------------------------------------------------

    cleaning_section = [
        "## 4. Cleaning Applied"
    ]

    if not cleaning_log:
        cleaning_section.append(
            "_No cleaning steps were applied._"
        )
    else:
        cleaning_rows = []

        for log_item in cleaning_log:
            cleaning_rows.append(
                {
                    "Action": log_item.get(
                        "action",
                        "-",
                    ),
                    "Column": log_item.get(
                        "column",
                        "-",
                    ),
                    "Reason": log_item.get(
                        "reason",
                        "-",
                    ),
                    "Shape Before": log_item.get(
                        "shape_before",
                        "-",
                    ),
                    "Shape After": log_item.get(
                        "shape_after",
                        "-",
                    ),
                    "Rows Affected": log_item.get(
                        "rows_affected",
                        "-",
                    ),
                }
            )

        cleaning_df = pd.DataFrame(
            cleaning_rows
        )

        cleaning_section.append(
            _safe_markdown_table(
                cleaning_df
            )
        )

    sections.append(
        "\n".join(cleaning_section)
    )

    # ---------------------------------------------------------
    # 5. Column Reference
    # ---------------------------------------------------------

    column_section = [
        "## 5. Column Reference"
    ]

    if column_reference is None:
        column_section.append(
            "_No column reference available._"
        )
    else:
        column_section.append(
            _safe_markdown_table(
                column_reference
            )
        )

    sections.append(
        "\n".join(column_section)
    )

    # ---------------------------------------------------------
    # 6. Numeric Summary
    # ---------------------------------------------------------

    numeric_section = [
        "## 6. Numeric Summary"
    ]

    if numeric_summary is None:
        numeric_section.append(
            "_No numeric summary available._"
        )
    else:
        numeric_section.append(
            _safe_markdown_table(
                numeric_summary
            )
        )

    sections.append(
        "\n".join(numeric_section)
    )

    # ---------------------------------------------------------
    # 7. Correlations
    # ---------------------------------------------------------

    correlation_section = [
        "## 7. Correlations"
    ]

    if correlations is None or correlations.empty:
        correlation_section.append(
            "_No significant correlations detected._"
        )
    else:
        correlation_section.append(
            _safe_markdown_table(
                correlations
            )
        )

    correlation_section.append(
        "\n> Correlation is not causation — "
        "this is a lead, not a conclusion."
    )

    sections.append(
        "\n".join(correlation_section)
    )

    # ---------------------------------------------------------
    # 8. Segment Analysis
    # ---------------------------------------------------------

    segment_section = [
        "## 8. Segment Analysis"
    ]

    if (
        segment_analysis is None
        or segment_analysis.empty
    ):
        segment_section.append(
            "_No segment analysis available._"
        )
    else:
        segment_section.append(
            _safe_markdown_table(
                segment_analysis
            )
        )

    sections.append(
        "\n".join(segment_section)
    )

    # ---------------------------------------------------------
    # 9. Trend Over Time
    # ---------------------------------------------------------

    trend_section = [
        "## 9. Trend Over Time"
    ]

    if (
        trend_over_time is None
        or trend_over_time.empty
    ):
        trend_section.append(
            "_No trend data available._"
        )
    else:
        trend_section.append(
            _safe_markdown_table(
                trend_over_time
            )
        )

    sections.append(
        "\n".join(trend_section)
    )

    # ---------------------------------------------------------
    # 10. Suggested Next Steps
    # ---------------------------------------------------------

    next_steps = [
        "## 10. Suggested Next Steps",
        "",
        "1. Review the highest-importance findings first.",
        "2. Validate unusual trends and segment differences "
        "with business stakeholders.",
        "3. Investigate the underlying causes of major "
        "data quality issues.",
        "4. Use correlations as leads for further analysis, "
        "not as evidence of causation.",
        "5. Define experiments or deeper analysis for the "
        "most material business questions.",
    ]

    sections.append(
        "\n".join(next_steps)
    )

    # ---------------------------------------------------------
    # Final disclaimer
    # ---------------------------------------------------------

    sections.append(
        "---\n\n"
        "*This is an automated first pass. Validate findings "
        "against business context before making decisions.*"
    )

    return "\n\n".join(sections)


def save_markdown_report(
    content: str,
    output_path: str = "reports/business_report.md",
) -> Path:
    """Save Markdown report to disk."""

    path = Path(output_path)

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    path.write_text(
        content,
        encoding="utf-8",
    )

    return path