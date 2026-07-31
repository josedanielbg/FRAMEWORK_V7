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


def read_key_value_table(
    path: Path,
    key_column: str = "Parametro",
    value_column: str = "Valor",
) -> dict[str, str]:
    """Read a two-column metadata table as a dictionary.

    Notebook stages C13 to C16 export metadata as CSV or Excel files with a
    parameter column and a value column. This helper normalizes that repeated
    pattern so downstream code can retrieve values by semantic key.

    Args:
        path (Path): Path to the metadata table.
        key_column (str): Name of the column containing metadata keys.
        value_column (str): Name of the column containing metadata values.

    Returns:
        dict[str, str]: Mapping from key names to string values. Returns an
        empty dictionary when the file or required columns are missing.
    """

    table = read_table(path)
    if table.empty or not {key_column, value_column}.issubset(table.columns):
        return {}
    clean = table[[key_column, value_column]].dropna(subset=[key_column])
    return dict(zip(clean[key_column].astype(str), clean[value_column].astype(str)))


def discover_experiment_dirs(root: Path) -> list[Path]:
    """List experiment directories below an artifact root.

    Args:
        root (Path): Directory that contains experiment-specific subfolders,
            for example ``DATA/EVALUACIONES`` or ``DATA/MODELADO/Tensores``.

    Returns:
        list[Path]: Sorted list of existing experiment directories. Missing
        roots return an empty list.
    """

    if not root.exists():
        return []
    return sorted(path for path in root.iterdir() if path.is_dir())


def discover_experiments(root: Path) -> list[str]:
    """List experiment identifiers available below an artifact root.

    Args:
        root (Path): Directory containing one subdirectory per experiment.

    Returns:
        list[str]: Sorted experiment identifiers such as ``Exp01`` or
        ``Exp04``.
    """

    return [path.name for path in discover_experiment_dirs(root)]


def latest_existing_path(paths: Iterable[Path]) -> Path | None:
    """Return the first path from a candidate sequence that exists.

    Args:
        paths (Iterable[Path]): Ordered candidate paths.

    Returns:
        Path | None: First existing path, or ``None`` when no candidate exists.
    """

    for path in paths:
        if path.exists():
            return path
    return None


def artifact_inventory(root: Path, experiment: str | None = None) -> pd.DataFrame:
    """Build an inventory of files produced by notebook experiments.

    Args:
        root (Path): Artifact directory to scan recursively.
        experiment (str | None): Optional experiment folder name used to limit
            the inventory.

    Returns:
        pd.DataFrame: Inventory with relative path, suffix, size in bytes and
        inferred experiment identifier.
    """

    scan_root = root / experiment if experiment else root
    if not scan_root.exists():
        return pd.DataFrame(columns=["Experimento", "Archivo", "Formato", "Bytes"])

    rows = []
    for file_path in sorted(path for path in scan_root.rglob("*") if path.is_file()):
        try:
            relative_path = file_path.relative_to(root)
        except ValueError:
            relative_path = file_path.name
        parts = Path(relative_path).parts
        rows.append(
            {
                "Experimento": parts[0] if parts else experiment,
                "Archivo": str(relative_path).replace("\\", "/"),
                "Formato": file_path.suffix.lower() or "sin_extension",
                "Bytes": file_path.stat().st_size,
            }
        )
    return pd.DataFrame(rows)


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
