from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .analysis import (
    rank_segment_drivers,
    top_correlations,
    trend_direction,
    trend_over_time,
)
from .cleaning import (
    apply_cleaning_plan,
    build_cleaning_plan,
)
from .insights import generate_findings
from .profiling import profile_dataframe
from .quality import quality_score, run_quality_checks
from .report import build_markdown_report, save_markdown_report


@dataclass
class AnalysisResult:
    """
    Complete output of the FirstLook analysis pipeline.
    """

    dataset_name: str

    original_df: pd.DataFrame
    clean_df: pd.DataFrame

    profile: object
    clean_profile: object

    issues: list
    quality_score: int

    cleaning_plan: list
    cleaning_log: list

    analysis: dict
    findings: list

    report_markdown: str

    primary_metric: str | None = None

    def to_markdown(self) -> str:
        """
        Return the generated Markdown report.
        """

        return self.report_markdown

    def save_report(
        self,
        output_path: str | Path = "reports/business_report.md",
    ) -> Path:
        """
        Save the generated Markdown report.
        """

        return save_markdown_report(
            self.report_markdown,
            str(output_path),
        )


def load_dataset(
    path: str | Path,
) -> pd.DataFrame:
    """
    Load a dataset based on its file extension.

    Supported formats:
        .csv
        .tsv
        .xlsx
        .parquet
    """

    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    extension = path.suffix.lower()

    if extension == ".csv":
        return pd.read_csv(path)

    if extension == ".tsv":
        return pd.read_csv(
            path,
            sep="\t",
        )

    if extension == ".xlsx":
        return pd.read_excel(path)

    if extension == ".parquet":
        return pd.read_parquet(path)

    raise ValueError(
        f"Unsupported file format: '{extension}'. "
        "Supported formats are: "
        ".csv, .tsv, .xlsx, .parquet"
    )


def _infer_primary_metric(
    df: pd.DataFrame,
    profile,
) -> str | None:
    """
    Automatically select the first numeric metric.

    Identifier-like columns are excluded by semantic profiling.
    """

    numeric_columns = profile.by_type(
        "numeric"
    )

    if not numeric_columns:
        return None

    return numeric_columns[0].name


def _infer_date_column(
    profile,
) -> str | None:
    """
    Automatically find the first datetime column.
    """

    datetime_columns = profile.by_type(
        "datetime"
    )

    if not datetime_columns:
        return None

    return datetime_columns[0].name


def _numeric_summary(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create a compact numeric summary.
    """

    numeric_df = df.select_dtypes(
        include="number"
    )

    if numeric_df.empty:
        return pd.DataFrame()

    summary = (
        numeric_df
        .agg(
            [
                "count",
                "mean",
                "median",
                "min",
                "max",
                "std",
            ]
        )
        .T
        .reset_index()
        .rename(
            columns={
                "index": "column"
            }
        )
    )

    for column in [
        "mean",
        "median",
        "min",
        "max",
        "std",
    ]:
        summary[column] = summary[
            column
        ].round(2)

    return summary


def _column_reference(
    profile,
) -> pd.DataFrame:
    """
    Build a column reference table.
    """

    rows = []

    for column in profile.columns:
        rows.append(
            {
                "column": column.name,
                "dtype": column.dtype,
                "semantic_type": column.semantic_type,
                "missing_pct": column.missing_pct,
                "unique_pct": column.unique_pct,
            }
        )

    return pd.DataFrame(rows)


def run_analysis(
    df: pd.DataFrame,
    profile,
    metric: str | None = None,
    date_col: str | None = None,
) -> dict:
    """
    Run all analytical modules.
    """

    # -------------------------------------------------
    # 1. Infer primary metric
    # -------------------------------------------------

    if metric is None:
        metric = _infer_primary_metric(
            df,
            profile,
        )

    # -------------------------------------------------
    # 2. Infer date column
    # -------------------------------------------------

    if date_col is None:
        date_col = _infer_date_column(
            profile
        )

    # -------------------------------------------------
    # 3. Correlations
    # -------------------------------------------------

    correlations = top_correlations(
        df
    )

    # -------------------------------------------------
    # 4. Segment drivers
    # -------------------------------------------------

    segment_drivers = pd.DataFrame()

    if metric is not None:
        segment_drivers = rank_segment_drivers(
            df=df,
            metric=metric,
        )

    # -------------------------------------------------
    # 5. Trend analysis
    # -------------------------------------------------

    trend = pd.DataFrame()

    trend_summary = {
        "direction": "insufficient_data",
        "slope": None,
        "change_pct": None,
        "n_points": 0,
    }

    if (
        metric is not None
        and date_col is not None
    ):
        trend = trend_over_time(
            df,
            date_col,
            metric,
        )

        trend_summary = trend_direction(
            trend
        )

    # -------------------------------------------------
    # 6. Numeric summary
    # -------------------------------------------------

    numeric_summary = _numeric_summary(
        df
    )

    # -------------------------------------------------
    # 7. Column reference
    # -------------------------------------------------

    column_reference = _column_reference(
        profile
    )

    return {
        "metric": metric,
        "date_column": date_col,
        "correlations": correlations,
        "segment_drivers": segment_drivers,
        "trend": trend,
        "trend_summary": trend_summary,
        "numeric_summary": numeric_summary,
        "column_reference": column_reference,
    }


def analyze(
    df: pd.DataFrame,
    dataset_name: str,
    metric: str | None = None,
    date_col: str | None = None,
    auto_clean: bool = True,
) -> AnalysisResult:
    """
    Run the complete FirstLook analysis pipeline.

    Pipeline:

        DataFrame
            ↓
        Profiling
            ↓
        Quality Checks
            ↓
        Cleaning Plan
            ↓
        Cleaning
            ↓
        Re-Profiling
            ↓
        Analysis
            ↓
        Findings
            ↓
        Markdown Report
    """

    # -------------------------------------------------
    # Validate input
    # -------------------------------------------------

    if not isinstance(
        df,
        pd.DataFrame,
    ):
        raise TypeError(
            "df must be a pandas DataFrame."
        )

    # -------------------------------------------------
    # PHASE 1 — Profile original dataset
    # -------------------------------------------------

    profile = profile_dataframe(
        df
    )

    # -------------------------------------------------
    # PHASE 2 — Quality checks
    # -------------------------------------------------

    # IMPORTANT:
    # Current quality.py expects only df.
    issues = run_quality_checks(
        df
    )

    # -------------------------------------------------
    # Quality score
    # -------------------------------------------------

    score = quality_score(
        issues
    )

    # -------------------------------------------------
    # PHASE 3 — Build cleaning plan
    # -------------------------------------------------

    plan = build_cleaning_plan(
        df,
        profile,
    )

    # -------------------------------------------------
    # Apply cleaning
    # -------------------------------------------------

    if auto_clean:
        clean_df, cleaning_log = (
            apply_cleaning_plan(
                df,
                plan,
            )
        )

    else:
        clean_df = df.copy()
        cleaning_log = []

    # -------------------------------------------------
    # PHASE 4 — Re-profile cleaned dataset
    # -------------------------------------------------

    clean_profile = profile_dataframe(
        clean_df
    )

    # -------------------------------------------------
    # PHASE 5 — Run analysis
    # -------------------------------------------------

    analysis = run_analysis(
        clean_df,
        clean_profile,
        metric=metric,
        date_col=date_col,
    )

    primary_metric = analysis[
        "metric"
    ]

    # -------------------------------------------------
    # PHASE 6 — Generate findings
    # -------------------------------------------------

    findings = generate_findings(
        issues=issues,
        trend_results=analysis[
            "trend_summary"
        ],
        segment_results=analysis[
            "segment_drivers"
        ],
        correlation_results=analysis[
            "correlations"
        ],
        profile=clean_profile,
        metric=primary_metric,
    )

    # -------------------------------------------------
    # PHASE 7 — Generate report
    # -------------------------------------------------

    report_markdown = build_markdown_report(
        profile=clean_profile,
        quality_score_value=score,
        issues=issues,
        findings=findings,
        cleaning_log=cleaning_log,
        column_reference=analysis[
            "column_reference"
        ],
        numeric_summary=analysis[
            "numeric_summary"
        ],
        correlations=analysis[
            "correlations"
        ],
        segment_analysis=analysis[
            "segment_drivers"
        ],
        trend_over_time=analysis[
            "trend"
        ],
        primary_metric=primary_metric,
    )

    # -------------------------------------------------
    # Final result
    # -------------------------------------------------

    return AnalysisResult(
        dataset_name=dataset_name,
        original_df=df.copy(),
        clean_df=clean_df,
        profile=profile,
        clean_profile=clean_profile,
        issues=issues,
        quality_score=score,
        cleaning_plan=plan,
        cleaning_log=cleaning_log,
        analysis=analysis,
        findings=findings,
        report_markdown=report_markdown,
        primary_metric=primary_metric,
    )