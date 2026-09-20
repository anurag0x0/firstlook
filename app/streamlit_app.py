import sys
from pathlib import Path
from dataclasses import asdict, is_dataclass

import pandas as pd
import streamlit as st


# ============================================================
# PROJECT PATH
# ============================================================

ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


import firstlook


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="FirstLook | Analytics Intelligence",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# GLOBAL CSS
# ============================================================

st.markdown(
    """
    <style>

    /* ================================
       GLOBAL
       ================================ */

    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 4rem;
    }

    /* ================================
       SIDEBAR
       ================================ */

    section[data-testid="stSidebar"] {
        border-right: 1px solid rgba(128, 128, 128, 0.18);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1.5rem;
    }

    /* ================================
       METRICS
       ================================ */

    [data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 12px;
        padding: 1rem;
        background: rgba(128, 128, 128, 0.025);
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.78rem;
    }

    /* ================================
       BUTTONS
       ================================ */

    .stButton > button {
        border-radius: 9px;
        min-height: 2.6rem;
        font-weight: 650;
    }

    /* ================================
       TABS
       ================================ */

    button[data-baseweb="tab"] {
        font-weight: 650;
    }

    /* ================================
       TABLE
       ================================ */

    [data-testid="stDataFrame"] {
        border-radius: 10px;
    }

    /* ================================
       SMALL MUTED TEXT
       ================================ */

    .muted {
        opacity: 0.6;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SESSION STATE
# ============================================================

if "result" not in st.session_state:
    st.session_state.result = None

if "dataset_name" not in st.session_state:
    st.session_state.dataset_name = None

if "raw_df" not in st.session_state:
    st.session_state.raw_df = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_get(obj, name, default=None):
    """
    Safely retrieve an attribute from an object or dictionary.
    """
    if obj is None:
        return default

    if isinstance(obj, dict):
        return obj.get(name, default)

    return getattr(obj, name, default)


def to_dict(obj):
    """
    Convert dataclass/dict/object into a dictionary.
    """
    if obj is None:
        return {}

    if isinstance(obj, dict):
        return obj

    if is_dataclass(obj):
        return asdict(obj)

    if hasattr(obj, "__dict__"):
        return vars(obj)

    return {}


def to_dataframe(data):
    """
    Safely convert common result structures to DataFrame.
    """
    if data is None:
        return pd.DataFrame()

    if isinstance(data, pd.DataFrame):
        return data

    if isinstance(data, list):
        if not data:
            return pd.DataFrame()

        rows = []

        for item in data:
            if isinstance(item, dict):
                rows.append(item)
            elif is_dataclass(item):
                rows.append(asdict(item))
            elif hasattr(item, "__dict__"):
                rows.append(vars(item))

        if rows:
            return pd.DataFrame(rows)

    if isinstance(data, dict):
        try:
            return pd.DataFrame(data)
        except Exception:
            return pd.DataFrame()

    return pd.DataFrame()


def format_number(value):
    """
    Format numbers for UI.
    """
    if value is None:
        return "—"

    try:
        number = float(value)

        if number.is_integer():
            return f"{int(number):,}"

        return f"{number:,.2f}"

    except Exception:
        return str(value)


def format_percent(value):
    """
    Format percentage values.
    """
    if value is None:
        return "—"

    try:
        return f"{float(value):.1f}%"
    except Exception:
        return str(value)


def get_profile_columns(profile):
    """
    Retrieve column profiles.
    """
    columns = safe_get(profile, "columns", [])

    if columns is None:
        return []

    return columns


def get_issues(result):
    """
    Retrieve quality issues.
    """
    return safe_get(result, "issues", []) or []


def get_findings(result):
    """
    Retrieve business findings.
    """
    return safe_get(result, "findings", []) or []


def get_cleaning_log(result):
    """
    Retrieve cleaning audit log.
    """
    return safe_get(result, "cleaning_log", []) or []


def get_analysis(result):
    """
    Retrieve analysis dictionary.
    """
    analysis = safe_get(result, "analysis", {})

    if analysis is None:
        return {}

    if isinstance(analysis, dict):
        return analysis

    if hasattr(analysis, "__dict__"):
        return vars(analysis)

    return {}


def quality_status(score):
    """
    Convert quality score into a human-readable status.
    """
    if score >= 90:
        return "Excellent"

    if score >= 75:
        return "Good"

    if score >= 50:
        return "Needs attention"

    return "Poor"


def severity_counts(issues):
    """
    Count issue severities.
    """
    counts = {
        "critical": 0,
        "warning": 0,
        "info": 0,
    }

    for issue in issues:
        severity = str(
            safe_get(issue, "severity", "info")
        ).lower()

        if severity in counts:
            counts[severity] += 1

    return counts


def render_issue_table(issues):
    """
    Render quality issues as a table.
    """
    if not issues:
        st.success(
            "No data-quality issues were detected."
        )
        return

    rows = []

    for issue in issues:

        rows.append(
            {
                "Severity": str(
                    safe_get(issue, "severity", "")
                ).title(),
                "Type": str(
                    safe_get(issue, "kind", "")
                ),
                "Column": safe_get(
                    issue,
                    "column",
                    "Dataset",
                ),
                "Message": safe_get(
                    issue,
                    "message",
                    "",
                ),
                "Recommendation": safe_get(
                    issue,
                    "suggestion",
                    "",
                ),
            }
        )

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )


def render_findings(findings):
    """
    Render business findings.
    """
    if not findings:
        st.info(
            "No material business findings were generated."
        )
        return

    sorted_findings = sorted(
        findings,
        key=lambda x: safe_get(
            x,
            "importance",
            99,
        ),
    )

    for index, finding in enumerate(
        sorted_findings,
        start=1,
    ):

        importance = safe_get(
            finding,
            "importance",
            5,
        )

        category = safe_get(
            finding,
            "category",
            "General",
        )

        headline = safe_get(
            finding,
            "headline",
            "Finding",
        )

        detail = safe_get(
            finding,
            "detail",
            "",
        )

        recommendation = safe_get(
            finding,
            "recommendation",
            "",
        )

        with st.container(border=True):

            st.caption(
                f"FINDING {index:02d} · "
                f"{str(category).upper()} · "
                f"IMPORTANCE {importance}"
            )

            st.subheader(headline)

            if detail:
                st.write(detail)

            if recommendation:
                st.info(
                    f"Next step: {recommendation}"
                )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown("◈")

    st.markdown(
        "## FirstLook"
    )

    st.caption(
        "Analytics Intelligence"
    )

    st.divider()

    st.markdown(
        "### Data source"
    )

    uploaded = st.file_uploader(
        "Upload dataset",
        type=["csv", "xlsx"],
        help="Upload a CSV or Excel dataset.",
    )

    if uploaded is not None:

        st.caption(
            f"Selected: **{uploaded.name}**"
        )

        if st.button(
            "Run analysis",
            type="primary",
            use_container_width=True,
        ):

            try:

                # --------------------------------------------
                # LOAD
                # --------------------------------------------

                if uploaded.name.lower().endswith(".csv"):

                    df = pd.read_csv(uploaded)

                elif uploaded.name.lower().endswith(".xlsx"):

                    df = pd.read_excel(uploaded)

                else:

                    st.error(
                        "Unsupported file format."
                    )

                    st.stop()


                # --------------------------------------------
                # ANALYZE ONCE
                # --------------------------------------------

                with st.spinner(
                    "Running FirstLook analysis..."
                ):

                    result = firstlook.analyze(
                        df,
                        uploaded.name,
                    )


                # --------------------------------------------
                # SESSION STATE
                # --------------------------------------------

                st.session_state.result = result
                st.session_state.dataset_name = uploaded.name
                st.session_state.raw_df = df.copy()

                st.rerun()

            except Exception as error:

                st.error(
                    f"Analysis failed: {error}"
                )


    # --------------------------------------------
    # DATASET STATUS
    # --------------------------------------------

    if st.session_state.result is not None:

        st.divider()

        st.markdown(
            "### Current dataset"
        )

        st.caption(
            st.session_state.dataset_name
        )

        result = st.session_state.result

        profile = safe_get(
            result,
            "profile",
        )

        if profile is not None:

            c1, c2 = st.columns(2)

            with c1:
                st.metric(
                    "Rows",
                    format_number(
                        safe_get(
                            profile,
                            "n_rows",
                            0,
                        )
                    ),
                )

            with c2:
                st.metric(
                    "Columns",
                    format_number(
                        safe_get(
                            profile,
                            "n_cols",
                            0,
                        )
                    ),
                )

        score = safe_get(
            result,
            "quality_score",
            0,
        )

        st.metric(
            "Quality",
            f"{score}/100",
        )

        st.divider()

        if st.button(
            "Start new analysis",
            use_container_width=True,
        ):

            st.session_state.result = None
            st.session_state.dataset_name = None
            st.session_state.raw_df = None

            st.rerun()


# ============================================================
# LANDING PAGE
# ============================================================

if st.session_state.result is None:

    st.caption(
        "DATA ANALYSIS WORKFLOW"
    )

    st.title(
        "FirstLook"
    )

    st.write(
        """
        Transform raw business data into a structured analytical view —
        from data quality and automated cleaning to trends, segment
        analysis, business findings and decision-ready reporting.
        """
    )

    st.divider()

    st.subheader(
        "Start a new analysis"
    )

    st.info(
        "Upload a CSV or XLSX dataset from the sidebar. "
        "FirstLook will profile, validate, clean and analyze it."
    )

    st.divider()

    st.caption(
        "ANALYTICAL WORKFLOW"
    )

    st.subheader(
        "From raw data to business insight"
    )

    # --------------------------------------------------------
    # ROW 1
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        with st.container(border=True):

            st.caption("01")

            st.subheader(
                "Dataset Profiling"
            )

            st.write(
                "Understand dataset dimensions, "
                "semantic types, distributions and "
                "column behavior."
            )


    with col2:

        with st.container(border=True):

            st.caption("02")

            st.subheader(
                "Data Quality"
            )

            st.write(
                "Detect missing values, duplicate "
                "records, constants and statistical "
                "outliers."
            )


    with col3:

        with st.container(border=True):

            st.caption("03")

            st.subheader(
                "Automated Cleaning"
            )

            st.write(
                "Apply transparent cleaning actions "
                "while maintaining an auditable "
                "transformation log."
            )


    # --------------------------------------------------------
    # ROW 2
    # --------------------------------------------------------

    col1, col2, col3 = st.columns(3)

    with col1:

        with st.container(border=True):

            st.caption("04")

            st.subheader(
                "Analytical Engine"
            )

            st.write(
                "Analyze trends, correlations and "
                "meaningful performance differences "
                "across segments."
            )


    with col2:

        with st.container(border=True):

            st.caption("05")

            st.subheader(
                "Business Findings"
            )

            st.write(
                "Translate analytical results into "
                "structured findings, implications "
                "and next steps."
            )


    with col3:

        with st.container(border=True):

            st.caption("06")

            st.subheader(
                "Decision-Ready Report"
            )

            st.write(
                "Generate a structured report covering "
                "quality, cleaning, analysis and "
                "business findings."
            )


    st.divider()

    st.caption(
        "FirstLook · Analytics Intelligence Platform"
    )

    st.stop()


# ============================================================
# RESULT
# ============================================================

result = st.session_state.result

dataset_name = (
    st.session_state.dataset_name
    or "Dataset"
)

profile = safe_get(
    result,
    "profile",
)

issues = get_issues(result)

findings = get_findings(result)

cleaning_log = get_cleaning_log(result)

analysis = get_analysis(result)

quality_score_value = safe_get(
    result,
    "quality_score",
    0,
)


# ============================================================
# RESULT HEADER
# ============================================================

st.caption(
    "ANALYSIS WORKSPACE"
)

st.title(
    dataset_name
)

st.write(
    "FirstLook has completed profiling, "
    "data-quality assessment, automated cleaning "
    "and analytical processing."
)


# ============================================================
# TOP KPI BAR
# ============================================================

if profile is not None:

    total_rows = safe_get(
        profile,
        "n_rows",
        0,
    )

    total_columns = safe_get(
        profile,
        "n_cols",
        0,
    )

else:

    total_rows = 0
    total_columns = 0


critical_count = severity_counts(
    issues
)["critical"]

warning_count = severity_counts(
    issues
)["warning"]


k1, k2, k3, k4, k5 = st.columns(5)


with k1:

    st.metric(
        "Rows",
        format_number(total_rows),
    )


with k2:

    st.metric(
        "Columns",
        format_number(total_columns),
    )


with k3:

    st.metric(
        "Quality",
        f"{quality_score_value}/100",
    )


with k4:

    st.metric(
        "Warnings",
        warning_count,
    )


with k5:

    st.metric(
        "Critical",
        critical_count,
    )


st.divider()


# ============================================================
# MAIN TABS
# ============================================================

overview_tab, quality_tab, cleaning_tab, analysis_tab, findings_tab, report_tab = st.tabs(
    [
        "Overview",
        "Data Quality",
        "Cleaning",
        "Analysis",
        "Findings",
        "Report",
    ]
)


# ============================================================
# 1. OVERVIEW
# ============================================================

with overview_tab:

    st.header(
        "Executive Overview"
    )

    st.caption(
        "A high-level view of dataset structure, "
        "data health and analytical signals."
    )


    # --------------------------------------------------------
    # DATASET SNAPSHOT
    # --------------------------------------------------------

    st.subheader(
        "Dataset snapshot"
    )

    if profile is not None:

        columns = get_profile_columns(
            profile
        )

        semantic_counts = {}

        for column in columns:

            semantic_type = safe_get(
                column,
                "semantic_type",
                "unknown",
            )

            semantic_counts[
                semantic_type
            ] = (
                semantic_counts.get(
                    semantic_type,
                    0,
                )
                + 1
            )

        snapshot_col1, snapshot_col2 = st.columns(2)

        with snapshot_col1:

            st.write(
                "**Dataset size**"
            )

            st.write(
                f"{format_number(total_rows)} rows × "
                f"{format_number(total_columns)} columns"
            )

            st.write(
                "**Duplicate rows**"
            )

            st.write(
                format_number(
                    safe_get(
                        profile,
                        "n_duplicate_rows",
                        0,
                    )
                )
            )


        with snapshot_col2:

            st.write(
                "**Semantic types**"
            )

            if semantic_counts:

                semantic_df = (
                    pd.DataFrame(
                        {
                            "Type": semantic_counts.keys(),
                            "Columns": semantic_counts.values(),
                        }
                    )
                    .sort_values(
                        "Columns",
                        ascending=False,
                    )
                )

                st.dataframe(
                    semantic_df,
                    use_container_width=True,
                    hide_index=True,
                )


    st.divider()


    # --------------------------------------------------------
    # DATA HEALTH
    # --------------------------------------------------------

    st.subheader(
        "Data health"
    )

    status = quality_status(
        quality_score_value
    )

    q1, q2, q3 = st.columns(3)

    with q1:

        st.metric(
            "Quality score",
            f"{quality_score_value}/100",
        )


    with q2:

        st.metric(
            "Status",
            status,
        )


    with q3:

        st.metric(
            "Detected issues",
            len(issues),
        )


    st.divider()


    # --------------------------------------------------------
    # TOP FINDINGS
    # --------------------------------------------------------

    st.subheader(
        "Top findings"
    )

    if findings:

        for finding in sorted(
            findings,
            key=lambda x: safe_get(
                x,
                "importance",
                99,
            ),
        )[:3]:

            headline = safe_get(
                finding,
                "headline",
                "Finding",
            )

            detail = safe_get(
                finding,
                "detail",
                "",
            )

            st.markdown(
                f"**{headline}**"
            )

            if detail:
                st.caption(
                    detail
                )

    else:

        st.info(
            "No material findings were generated."
        )


# ============================================================
# 2. DATA QUALITY
# ============================================================

with quality_tab:

    st.header(
        "Data Quality"
    )

    st.caption(
        "Understand whether the dataset is reliable enough "
        "for downstream analysis."
    )


    q1, q2, q3, q4 = st.columns(4)

    counts = severity_counts(
        issues
    )

    with q1:
        st.metric(
            "Quality score",
            f"{quality_score_value}/100",
        )

    with q2:
        st.metric(
            "Critical",
            counts["critical"],
        )

    with q3:
        st.metric(
            "Warnings",
            counts["warning"],
        )

    with q4:
        st.metric(
            "Information",
            counts["info"],
        )


    st.divider()

    st.subheader(
        "Detected issues"
    )

    render_issue_table(
        issues
    )


    st.divider()

    st.subheader(
        "Column profile"
    )

    if profile is not None:

        columns = get_profile_columns(
            profile
        )

        rows = []

        for column in columns:

            rows.append(
                {
                    "Column": safe_get(
                        column,
                        "name",
                        "",
                    ),
                    "Type": safe_get(
                        column,
                        "semantic_type",
                        "",
                    ),
                    "Missing %": round(
                        float(
                            safe_get(
                                column,
                                "missing_pct",
                                0,
                            )
                        ),
                        2,
                    ),
                    "Unique %": round(
                        float(
                            safe_get(
                                column,
                                "unique_pct",
                                0,
                            )
                        ),
                        2,
                    ),
                    "Unique": safe_get(
                        column,
                        "n_unique",
                        0,
                    ),
                }
            )

        if rows:

            st.dataframe(
                pd.DataFrame(rows),
                use_container_width=True,
                hide_index=True,
            )


# ============================================================
# 3. CLEANING
# ============================================================

with cleaning_tab:

    st.header(
        "Cleaning Audit"
    )

    st.caption(
        "Every automated transformation is surfaced "
        "for transparency and auditability."
    )


    if not cleaning_log:

        st.success(
            "No cleaning actions were required."
        )

    else:

        st.metric(
            "Cleaning actions applied",
            len(cleaning_log),
        )

        st.divider()

        cleaning_rows = []

        for item in cleaning_log:

            if isinstance(
                item,
                dict,
            ):

                cleaning_rows.append(
                    item
                )

            elif is_dataclass(item):

                cleaning_rows.append(
                    asdict(item)
                )

            elif hasattr(
                item,
                "__dict__",
            ):

                cleaning_rows.append(
                    vars(item)
                )

        if cleaning_rows:

            st.dataframe(
                pd.DataFrame(
                    cleaning_rows
                ),
                use_container_width=True,
                hide_index=True,
            )

        else:

            st.write(
                cleaning_log
            )


    st.divider()

    st.subheader(
        "Cleaning principles"
    )

    st.write(
        """
        FirstLook uses a rule-based cleaning plan rather than
        silently modifying the source dataset. Cleaning actions
        are generated from detected data characteristics and
        recorded as an audit trail.
        """
    )


# ============================================================
# 4. ANALYSIS
# ============================================================

with analysis_tab:

    st.header(
        "Analytical Engine"
    )

    st.caption(
        "Trends, correlations and segment-level performance."
    )


    # --------------------------------------------------------
    # TREND
    # --------------------------------------------------------

    st.subheader(
        "Trend analysis"
    )

    trend_data = analysis.get(
        "trend",
        analysis.get(
            "trends",
            None,
        ),
    )

    trend_df = to_dataframe(
        trend_data
    )

    if not trend_df.empty:

        st.dataframe(
            trend_df,
            use_container_width=True,
            hide_index=True,
        )

        # Try to identify numeric columns
        numeric_columns = trend_df.select_dtypes(
            include="number"
        ).columns.tolist()

        if numeric_columns:

            value_column = numeric_columns[-1]

            try:

                st.line_chart(
                    trend_df[
                        [value_column]
                    ]
                )

            except Exception:
                pass

    else:

        st.info(
            "No trend result is available for this dataset."
        )


    st.divider()


    # --------------------------------------------------------
    # CORRELATIONS
    # --------------------------------------------------------

    st.subheader(
        "Correlations"
    )

    correlation_data = analysis.get(
        "correlations",
        None,
    )

    correlation_df = to_dataframe(
        correlation_data
    )

    if not correlation_df.empty:

        st.dataframe(
            correlation_df,
            use_container_width=True,
            hide_index=True,
        )

        st.caption(
            "Correlation indicates association, not causation."
        )

    else:

        st.info(
            "No material correlations were detected."
        )


    st.divider()


    # --------------------------------------------------------
    # SEGMENTS
    # --------------------------------------------------------

    st.subheader(
        "Segment comparison"
    )

    segment_data = analysis.get(
        "segments",
        analysis.get(
            "segment_comparison",
            None,
        ),
    )

    segment_df = to_dataframe(
        segment_data
    )

    if not segment_df.empty:

        st.dataframe(
            segment_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "No segment comparison is available."
        )


    st.divider()


    # --------------------------------------------------------
    # SEGMENT DRIVERS
    # --------------------------------------------------------

    st.subheader(
        "Segment drivers"
    )

    driver_data = analysis.get(
        "drivers",
        analysis.get(
            "segment_drivers",
            None,
        ),
    )

    driver_df = to_dataframe(
        driver_data
    )

    if not driver_df.empty:

        st.dataframe(
            driver_df,
            use_container_width=True,
            hide_index=True,
        )

    else:

        st.info(
            "No segment-driver analysis is available."
        )


# ============================================================
# 5. FINDINGS
# ============================================================

with findings_tab:

    st.header(
        "Business Findings"
    )

    st.caption(
        "Analytical evidence translated into structured "
        "business findings and next-step recommendations."
    )

    render_findings(
        findings
    )


# ============================================================
# 6. REPORT
# ============================================================

with report_tab:

    st.header(
        "Decision-Ready Report"
    )

    st.caption(
        "A structured report generated from the same "
        "analysis result used throughout the workspace."
    )


    report_markdown = safe_get(
        result,
        "report_markdown",
        "",
    )


    if report_markdown:

        st.download_button(
            label="Download Markdown Report",
            data=report_markdown,
            file_name=(
                Path(dataset_name).stem
                + "_firstlook_report.md"
            ),
            mime="text/markdown",
            type="primary",
        )

        st.divider()

        st.markdown(
            report_markdown
        )

    else:

        st.warning(
            "No report content is available."
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "FirstLook · Analytics Intelligence Platform"
)