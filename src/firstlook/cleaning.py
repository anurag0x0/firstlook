from dataclasses import dataclass

import pandas as pd

from .profiling import DatasetProfile


@dataclass
class CleaningStep:
    """
    Represents one proposed data-cleaning action.
    """

    action: str
    column: str | None
    reason: str
    params: dict
    enabled: bool = True


def build_cleaning_plan(
    df: pd.DataFrame,
    profile: DatasetProfile,
) -> list[CleaningStep]:
    """
    Build a cleaning plan without modifying the DataFrame.

    Rules:
    1. Duplicate rows -> drop_duplicate_rows
    2. Empty / constant columns -> drop_column
    3. Missing >= 50% -> drop_column
    4. Datetime stored as string -> parse_datetime
    5. Numeric + missing -> fill_missing using median
    6. Categorical + missing -> fill_missing using Unknown
    7. Categorical -> strip_whitespace
    """

    plan: list[CleaningStep] = []

    # =========================================================
    # RULE 1: Duplicate rows
    # =========================================================

    duplicate_count = int(df.duplicated().sum())

    if duplicate_count > 0:
        step = CleaningStep(
            action="drop_duplicate_rows",
            column=None,
            reason=(
                f"Dataset contains {duplicate_count} "
                f"duplicate row(s)."
            ),
            params={
                "duplicate_count": duplicate_count,
            },
        )

        plan.append(step)

    # =========================================================
    # COLUMN-LEVEL RULES
    # =========================================================

    for column_profile in profile.columns:

        column_name = column_profile.name
        semantic_type = column_profile.semantic_type
        missing_count = column_profile.n_missing
        missing_pct = column_profile.missing_pct

        # -----------------------------------------------------
        # RULE 2: Empty column
        # -----------------------------------------------------

        if semantic_type == "empty":

            step = CleaningStep(
                action="drop_column",
                column=column_name,
                reason=(
                    f"{column_name} contains no usable values."
                ),
                params={
                    "reason_type": "empty_column",
                },
            )

            plan.append(step)
            continue

        # -----------------------------------------------------
        # RULE 2: Constant column
        # -----------------------------------------------------

        if semantic_type == "constant":

            step = CleaningStep(
                action="drop_column",
                column=column_name,
                reason=(
                    f"{column_name} contains only one "
                    "unique value."
                ),
                params={
                    "reason_type": "constant_column",
                },
            )

            plan.append(step)
            continue

        # -----------------------------------------------------
        # RULE 3: Missing >= 50%
        # -----------------------------------------------------

        if missing_pct >= 50.0:

            step = CleaningStep(
                action="drop_column",
                column=column_name,
                reason=(
                    f"{column_name} has "
                    f"{missing_pct:.1f}% missing values."
                ),
                params={
                    "reason_type": "high_missingness",
                    "missing_pct": missing_pct,
                    "threshold": 50.0,
                },
            )

            plan.append(step)
            continue

        # -----------------------------------------------------
        # RULE 4: Datetime stored as string/object
        # -----------------------------------------------------

        if semantic_type == "datetime":

            if pd.api.types.is_object_dtype(df[column_name]):

                step = CleaningStep(
                    action="parse_datetime",
                    column=column_name,
                    reason=(
                        f"{column_name} contains datetime "
                        "values stored as strings."
                    ),
                    params={
                        "format": "mixed",
                    },
                )

                plan.append(step)

        # -----------------------------------------------------
        # RULE 5: Numeric + missing -> median
        # -----------------------------------------------------

        if semantic_type == "numeric" and missing_count > 0:

            median_value = column_profile.stats.get("median")

            step = CleaningStep(
                action="fill_missing",
                column=column_name,
                reason=(
                    f"{column_name} is numeric and contains "
                    f"{missing_count} missing value(s)."
                ),
                params={
                    "method": "median",
                    "value": median_value,
                },
            )

            plan.append(step)

        # -----------------------------------------------------
        # RULE 6: Categorical + missing -> Unknown
        # -----------------------------------------------------

        if semantic_type == "categorical" and missing_count > 0:

            step = CleaningStep(
                action="fill_missing",
                column=column_name,
                reason=(
                    f"{column_name} is categorical and contains "
                    f"{missing_count} missing value(s)."
                ),
                params={
                    "method": "constant",
                    "value": "Unknown",
                },
            )

            plan.append(step)

        # -----------------------------------------------------
        # RULE 7: Categorical -> strip whitespace
        # -----------------------------------------------------

        if semantic_type == "categorical":

            step = CleaningStep(
                action="strip_whitespace",
                column=column_name,
                reason=(
                    f"{column_name} is categorical and may contain "
                    "leading or trailing whitespace."
                ),
                params={
                    "side": "both",
                },
            )

            plan.append(step)

    return plan


def apply_cleaning_plan(
    df: pd.DataFrame,
    plan: list[CleaningStep],
) -> tuple[pd.DataFrame, list[dict]]:
    """
    Apply an approved cleaning plan to a copy of the DataFrame.

    The original DataFrame is never modified.

    Returns:
        cleaned_df: cleaned copy of the original DataFrame
        log: audit trail of all applied cleaning steps
    """

    # IMPORTANT:
    # Never modify the original DataFrame.
    cleaned_df = df.copy()

    # Audit trail
    log: list[dict] = []

    # =========================================================
    # APPLY EACH CLEANING STEP
    # =========================================================

    for step in plan:

        # -----------------------------------------------------
        # Skip disabled steps
        # -----------------------------------------------------

        if not step.enabled:
            continue

        # Shape before applying the step
        shape_before = cleaned_df.shape

        # =====================================================
        # 1. Drop duplicate rows
        # =====================================================

        if step.action == "drop_duplicate_rows":

            duplicate_count = int(
                cleaned_df.duplicated().sum()
            )

            cleaned_df = (
                cleaned_df
                .drop_duplicates()
                .reset_index(drop=True)
            )

            shape_after = cleaned_df.shape

            log.append(
                {
                    "action": step.action,
                    "column": step.column,
                    "reason": step.reason,
                    "shape_before": shape_before,
                    "shape_after": shape_after,
                    "rows_affected": duplicate_count,
                }
            )

        # =====================================================
        # 2. Drop column
        # =====================================================

        elif step.action == "drop_column":

            column = step.column

            if column is None:
                continue

            if column not in cleaned_df.columns:
                continue

            cleaned_df = cleaned_df.drop(
                columns=[column]
            )

            shape_after = cleaned_df.shape

            log.append(
                {
                    "action": step.action,
                    "column": column,
                    "reason": step.reason,
                    "shape_before": shape_before,
                    "shape_after": shape_after,
                    "rows_affected": 0,
                }
            )

        # =====================================================
        # 3. Parse datetime
        # =====================================================

        elif step.action == "parse_datetime":

            column = step.column

            if column is None:
                continue

            if column not in cleaned_df.columns:
                continue

            cleaned_df[column] = pd.to_datetime(
                cleaned_df[column],
                format="mixed",
                errors="coerce",
            )

            shape_after = cleaned_df.shape

            log.append(
                {
                    "action": step.action,
                    "column": column,
                    "reason": step.reason,
                    "shape_before": shape_before,
                    "shape_after": shape_after,
                    "rows_affected": int(
                        cleaned_df[column].notna().sum()
                    ),
                }
            )

        # =====================================================
        # 4. Fill missing values
        # =====================================================

        elif step.action == "fill_missing":

            column = step.column

            if column is None:
                continue

            if column not in cleaned_df.columns:
                continue

            # Count missing values BEFORE cleaning
            missing_before = int(
                cleaned_df[column].isna().sum()
            )

            method = step.params.get("method")

            # -------------------------------------------------
            # Numeric -> median
            # -------------------------------------------------

            if method == "median":

                value = step.params.get("value")

                cleaned_df[column] = (
                    cleaned_df[column]
                    .fillna(value)
                )

            # -------------------------------------------------
            # Categorical -> Unknown
            # -------------------------------------------------

            elif method == "constant":

                value = step.params.get(
                    "value",
                    "Unknown",
                )

                cleaned_df[column] = (
                    cleaned_df[column]
                    .fillna(value)
                )

            shape_after = cleaned_df.shape

            log.append(
                {
                    "action": step.action,
                    "column": column,
                    "reason": step.reason,
                    "shape_before": shape_before,
                    "shape_after": shape_after,
                    "rows_affected": missing_before,
                }
            )

        # =====================================================
        # 5. Strip whitespace
        # =====================================================

        elif step.action == "strip_whitespace":

            column = step.column

            if column is None:
                continue

            if column not in cleaned_df.columns:
                continue

            before_values = cleaned_df[column].copy()

            cleaned_df[column] = cleaned_df[column].map(
                lambda value: (
                    value.strip()
                    if isinstance(value, str)
                    else value
                )
            )

            changed_count = int(
                (
                    before_values
                    != cleaned_df[column]
                )
                .fillna(False)
                .sum()
            )

            shape_after = cleaned_df.shape

            log.append(
                {
                    "action": step.action,
                    "column": column,
                    "reason": step.reason,
                    "shape_before": shape_before,
                    "shape_after": shape_after,
                    "rows_affected": changed_count,
                }
            )

    return cleaned_df, log