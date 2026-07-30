"""Utility helpers shared by notebook pipeline modules."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pandas as pd


def read_table(path: Path, **kwargs) -> pd.DataFrame:
    """Read a tabular artifact from disk.

    Centralizes notebook IO for CSV and Excel files. Missing files return an
    empty DataFrame so exploratory notebooks can continue and decide how to
    report missing artifacts.

    Args:
        path (Path): File path to read. Supported suffixes are ``.csv``,
            ``.xlsx`` and ``.xls``.
        **kwargs: Extra keyword arguments forwarded to ``pandas.read_csv`` or
            ``pandas.read_excel``.

    Returns:
        pd.DataFrame: Loaded table. Returns an empty DataFrame when ``path``
        does not exist.

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
    """Validate the presence of required columns.

    This helper is used by notebook modules before joins, model preparation or
    sequence construction so errors fail with a clear stage-specific message.

    Args:
        df (pd.DataFrame): Dataset to validate.
        required_columns (Iterable[str]): Column names that must be present.
        dataset_name (str): Human-readable dataset label included in the error
            message.

    Returns:
        None: The function only validates inputs.

    Raises:
        ValueError: If at least one required column is missing.
    """

    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"{dataset_name} is missing required columns: {missing}")


def existing_columns(df: pd.DataFrame, columns: Iterable[str]) -> list[str]:
    """Filter a candidate column list to columns present in a DataFrame.

    Args:
        df (pd.DataFrame): Dataset to inspect.
        columns (Iterable[str]): Candidate column names.

    Returns:
        list[str]: Ordered list with only the columns found in ``df``.
    """

    return [column for column in columns if column in df.columns]


def save_table(df: pd.DataFrame, path: Path) -> Path:
    """Save a table using the format implied by its file suffix.

    Args:
        df (pd.DataFrame): Dataset to persist.
        path (Path): Destination path. Supported suffixes are ``.csv``,
            ``.xlsx`` and ``.xls``.

    Returns:
        Path: Path written to disk.

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
    """Save a dataset as CSV and optionally as Excel.

    Args:
        df (pd.DataFrame): Dataset to persist.
        csv_path (Path): CSV destination path.
        excel_path (Path | None): Optional Excel destination path.

    Returns:
        list[Path]: Paths written by the function.
    """

    written = [save_table(df, csv_path)]
    if excel_path is not None:
        written.append(save_table(df, excel_path))
    return written
