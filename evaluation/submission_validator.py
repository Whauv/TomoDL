"""Submission validation utilities for Kaggle compatibility."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

SUBMISSION_COLUMNS = ["tomo_id", "Motor axis 0", "Motor axis 1", "Motor axis 2"]


class SubmissionValidationError(ValueError):
    """Raised when submission dataframe violates competition contract."""


def _ensure_columns(df: pd.DataFrame) -> None:
    if list(df.columns) != SUBMISSION_COLUMNS:
        raise SubmissionValidationError(
            f"Submission columns must be {SUBMISSION_COLUMNS}, got {list(df.columns)}"
        )


def _ensure_numeric_axes(df: pd.DataFrame) -> None:
    axis_cols = SUBMISSION_COLUMNS[1:]
    for col in axis_cols:
        if not pd.api.types.is_numeric_dtype(df[col]):
            raise SubmissionValidationError(f"Column '{col}' must be numeric.")
        arr = df[col].to_numpy(dtype=np.float64)
        if not np.all(np.isfinite(arr)):
            raise SubmissionValidationError(f"Column '{col}' contains non-finite values.")


def _ensure_minus_one_policy(df: pd.DataFrame) -> None:
    axes = df[["Motor axis 0", "Motor axis 1", "Motor axis 2"]].to_numpy(dtype=np.float64)
    any_minus_one = (axes == -1.0).any(axis=1)
    all_minus_one = (axes == -1.0).all(axis=1)
    bad = any_minus_one & (~all_minus_one)
    if bool(np.any(bad)):
        idx = int(np.where(bad)[0][0])
        raise SubmissionValidationError(
            "Invalid -1 policy: if any motor axis is -1, all three must be -1 "
            f"(first bad row index: {idx})."
        )


def _ensure_row_contract(df: pd.DataFrame, expected_tomo_ids: Iterable[str] | None) -> None:
    if expected_tomo_ids is None:
        return
    expected = [str(x) for x in expected_tomo_ids]
    got = [str(x) for x in df["tomo_id"].tolist()]
    if len(got) != len(expected):
        raise SubmissionValidationError(
            f"Row count mismatch. expected={len(expected)} got={len(got)}"
        )
    if set(got) != set(expected):
        missing = sorted(set(expected) - set(got))[:5]
        extra = sorted(set(got) - set(expected))[:5]
        raise SubmissionValidationError(
            f"tomo_id mismatch. missing={missing} extra={extra}"
        )


def validate_submission_df(df: pd.DataFrame, expected_tomo_ids: Iterable[str] | None = None) -> None:
    """Validate submission dataframe against Kaggle schema/rules."""
    _ensure_columns(df)
    _ensure_numeric_axes(df)
    _ensure_minus_one_policy(df)
    _ensure_row_contract(df, expected_tomo_ids)


def validate_submission_csv(path: str, expected_tomo_ids: Iterable[str] | None = None) -> None:
    """Load and validate a submission CSV file."""
    df = pd.read_csv(path)
    validate_submission_df(df, expected_tomo_ids=expected_tomo_ids)
