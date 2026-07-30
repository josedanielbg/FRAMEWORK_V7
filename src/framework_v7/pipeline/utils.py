"""Utility helpers shared by notebook pipeline modules."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    """Read CSV or Excel artifacts with one function.

    Args:
        path: File path to read.
        **kwargs: Extra arguments passed to pandas readers.

    Returns:
        Loaded DataFrame, or an empty DataFrame when the file does not exist.

    Raises:
        ValueError: If the suffix is not supported.
    """

    if not path.exists():
        return pd.DataFrame()

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, **kwargs)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path, **kwargs)
    raise ValueError(f"Unsupported table format: {path.suffix}")


def ensure_columns(df: pd.DataFrame, required_columns: Iterable[str], dataset_name: str = "dataset") -> None:
    """Validate that a DataFrame contains required columns.

    Args:
        df: Dataset to validate.
        required_columns: Required column names.
        dataset_name: Human-readable dataset label for error messages.

    Raises:
        ValueError: If at least one required column is missing.
    """

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")


def existing_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    """Return columns that exist in a DataFrame preserving order.

    Args:
        df: Dataset to inspect.
        columns: Candidate columns.

    Returns:
        Ordered list of columns present in ``df``.
    """

    return [column for column in columns if column in df.columns]


def save_table(df: pd.DataFrame, path: Path) -> Path:
    """Save a table as CSV or Excel depending on the suffix.

    Args:
        df: Dataset to write.
        path: Destination path.

    Returns:
        Written path.

    Raises:
        ValueError: If the suffix is not supported.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    suffix = path.suffix.lower()
    if suffix == ".csv":
        df.to_csv(path, index=False)
        return path
    if suffix in {".xlsx", ".xls"}:
        df.to_excel(path, index=False)
        return path
    raise ValueError(f"Unsupported table format: {path.suffix}")


def save_csv_and_excel(df: pd.DataFrame, csv_path: Path, excel_path: Path | None = None) -> list[Path]:
    """Save a DataFrame as CSV and optionally Excel.

    Args:
        df: Dataset to persist.
        csv_path: CSV destination.
        excel_path: Optional Excel destination.

    Returns:
        List of written paths.
    """

    written = [save_table(df, csv_path)]
    if excel_path is not None:
        written.append(save_table(df, excel_path))
    return written
