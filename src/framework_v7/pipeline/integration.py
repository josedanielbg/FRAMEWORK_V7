"""Integration helpers extracted from notebook C08."""

from __future__ import annotations

from functools import reduce
from typing import Mapping

import pandas as pd

from .utils import ensure_columns


def build_node(df: pd.DataFrame, municipality_col: str = "municipio", station_col: str | None = None) -> pd.Series:
    """Build the normalized integration node used by layer datasets.

    The C08 notebook creates a common spatial key before joining hydrology,
    water-quality, hydraulic and governance layers. This function reproduces
    that key construction in a reusable form.

    Args:
        df (pd.DataFrame): Input layer dataset.
        municipality_col (str): Column containing municipality names.
        station_col (str | None): Optional station column appended to the node
            when it exists and is not empty.

    Returns:
        pd.Series: Uppercase, stripped node labels suitable for joins.
    """

    ensure_columns(df, [municipality_col], "layer dataset")
    node = df[municipality_col].astype(str).str.strip().str.upper()
    if station_col and station_col in df.columns:
        station = df[station_col].astype(str).str.strip().str.upper()
        node = node.where(station.eq("") | station.eq("NAN"), node + " - " + station)
    return node


def standardize_date_column(df: pd.DataFrame, date_col: str = "Fecha") -> pd.DataFrame:
    """Convert a date-like column to monthly timestamps.

    Dates are coerced with pandas and then normalized to the first day of each
    month. This matches the temporal grain used by the master dataset.

    Args:
        df (pd.DataFrame): Input dataset.
        date_col (str): Date column name to normalize.

    Returns:
        pd.DataFrame: Copy of ``df`` with standardized monthly dates.
    """

    ensure_columns(df, [date_col], "layer dataset")
    output = df.copy()
    output[date_col] = pd.to_datetime(output[date_col], errors="coerce").dt.to_period("M").dt.to_timestamp()
    return output


def standardize_layer_keys(
    df: pd.DataFrame,
    date_col: str = "Fecha",
    node_col: str = "Nodo",
    municipality_col: str | None = None,
    station_col: str | None = None,
) -> pd.DataFrame:
    """Ensure a layer dataset has the common master keys.

    The function standardizes temporal keys, creates ``Nodo`` when a
    municipality column is available and normalizes node text for reliable
    outer joins across layers.

    Args:
        df (pd.DataFrame): Input layer dataset.
        date_col (str): Name of the source date column.
        node_col (str): Name of the source or output node column.
        municipality_col (str | None): Municipality column used when ``node_col``
            is not already present.
        station_col (str | None): Optional station column used to enrich nodes.

    Returns:
        pd.DataFrame: Copy with standardized ``Fecha`` and ``Nodo`` columns when
        source columns are available.
    """

    output = df.copy()
    if date_col in output.columns:
        output = standardize_date_column(output, date_col)
        if date_col != "Fecha":
            output = output.rename(columns={date_col: "Fecha"})
    if node_col not in output.columns and municipality_col:
        output[node_col] = build_node(output, municipality_col, station_col)
    if node_col in output.columns and node_col != "Nodo":
        output = output.rename(columns={node_col: "Nodo"})
    if "Nodo" in output.columns:
        output["Nodo"] = output["Nodo"].astype(str).str.strip().str.upper()
    return output


def merge_layer_frames(frames: Mapping[str, pd.DataFrame], keys: list[str] | None = None) -> pd.DataFrame:
    """Merge standardized layer frames into one master dataset.

    Empty frames are skipped. Non-empty frames must contain every merge key;
    otherwise a clear validation error is raised before the join.

    Args:
        frames (Mapping[str, pd.DataFrame]): Mapping of layer label to
            DataFrame.
        keys (list[str] | None): Merge keys. Defaults to ``["Fecha", "Nodo"]``.

    Returns:
        pd.DataFrame: Outer-joined master dataset. Returns an empty DataFrame
        when all inputs are empty.
    """

    keys = keys or ["Fecha", "Nodo"]
    prepared = []
    for label, frame in frames.items():
        if frame.empty:
            continue
        ensure_columns(frame, keys, label)
        prepared.append(frame.copy())
    if not prepared:
        return pd.DataFrame()
    return reduce(lambda left, right: pd.merge(left, right, on=keys, how="outer"), prepared)


def integration_quality(master: pd.DataFrame, keys: list[str] | None = None) -> pd.DataFrame:
    """Build structural quality indicators for the integrated dataset.

    Args:
        master (pd.DataFrame): Integrated master dataset.
        keys (list[str] | None): Key columns used for duplicate checks.
            Defaults to ``["Fecha", "Nodo"]``.

    Returns:
        pd.DataFrame: Table with row count, column count, nulls and duplicate
        indicators.
    """

    keys = keys or ["Fecha", "Nodo"]
    rows = [
        {"Indicador": "filas", "Valor": len(master)},
        {"Indicador": "columnas", "Valor": master.shape[1]},
        {"Indicador": "nulos", "Valor": int(master.isna().sum().sum())},
        {"Indicador": "duplicados_totales", "Valor": int(master.duplicated().sum())},
    ]
    if all(column in master.columns for column in keys):
        rows.append({"Indicador": "duplicados_llave", "Valor": int(master.duplicated(subset=keys).sum())})
    return pd.DataFrame(rows)
