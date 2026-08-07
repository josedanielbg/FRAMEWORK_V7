"""Framework artifact builders for notebooks C01-C07.

The extraction notebooks collect raw information for each system layer. The
``FW7_C0X_Framework`` notebooks then repeat the same governance pattern:
record identifiers, hashes, metadata, dictionaries, audits, indicators and
exploratory summaries. This module centralizes that shared logic so every
layer can reuse the same functions without depending on Streamlit.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from framework_v7.catalog import LAYER_CATALOG
from framework_v7.layers import LAYER_MODULES
from framework_v7.paths import NOTEBOOKS_DIR, rel

from .utils import save_table


FRAMEWORK_NOTEBOOK_PATTERN = "FW7_C0*_Framework.ipynb"


def generate_record_hash(row: pd.Series, algorithm: str = "md5") -> str:
    """Generate a deterministic hash for one dataset row.

    Args:
        row (pd.Series): Row values to concatenate before hashing.
        algorithm (str): Hash algorithm supported by ``hashlib``. The default
            value reproduces the MD5 behavior used in the original notebooks.

    Returns:
        str: Hexadecimal hash string for the row.

    Raises:
        ValueError: If ``algorithm`` is not available in ``hashlib``.
    """

    try:
        hasher = hashlib.new(algorithm)
    except ValueError as exc:
        raise ValueError(f"Unsupported hash algorithm: {algorithm}") from exc
    hasher.update("".join(row.astype(str)).encode("utf-8"))
    return hasher.hexdigest()


def add_framework_governance_columns(
    df: pd.DataFrame,
    layer_code: str,
    version: str,
    responsible: str,
    generated_at: datetime | None = None,
    status: str = "VALIDO",
    id_column: str = "ID_Registro",
) -> pd.DataFrame:
    """Add standard governance columns used by C01-C07 framework notebooks.

    Args:
        df (pd.DataFrame): Layer dataset to enrich.
        layer_code (str): Canonical layer code such as ``C01`` or ``GOB``.
        version (str): Framework version stored in ``Version_Framework``.
        responsible (str): Person or team responsible for the transformation.
        generated_at (datetime | None): Execution timestamp. When ``None``,
            the current local timestamp is used.
        status (str): Record status written to ``Estado_Registro``.
        id_column (str): Name of the generated identifier column.

    Returns:
        pd.DataFrame: Copy of ``df`` with governance and hash columns.
    """

    timestamp = generated_at or datetime.now()
    output = df.copy()
    if id_column not in output.columns:
        record_ids = [f"{layer_code}-{index + 1:07d}" for index in range(len(output))]
        output.insert(0, id_column, record_ids)
    if "ID_Capa" not in output.columns:
        output.insert(1, "ID_Capa", layer_code)
    output["Fecha_ETL"] = timestamp.strftime("%Y-%m-%d")
    output["Hora_ETL"] = timestamp.strftime("%H:%M:%S")
    output["Version_Framework"] = version
    output["Estado_Registro"] = status
    output["Responsable"] = responsible
    output["Hash"] = output.astype(str).apply(generate_record_hash, axis=1)
    return output


def build_layer_metadata(
    df: pd.DataFrame,
    layer_name: str,
    layer_code: str,
    version: str,
    responsible: str,
    source: str = "",
    source_url: str = "",
    generated_at: datetime | None = None,
    date_column: str | None = None,
    node_column: str | None = None,
) -> pd.DataFrame:
    """Build the metadata table exported as ``03_Metadata.xlsx``.

    Args:
        df (pd.DataFrame): Framework layer dataset.
        layer_name (str): Human-readable layer name.
        layer_code (str): Layer code used by the framework.
        version (str): Framework version.
        responsible (str): Responsible person or team.
        source (str): Data source name.
        source_url (str): Source URL or repository reference.
        generated_at (datetime | None): Execution timestamp. Uses current time
            when omitted.
        date_column (str | None): Optional temporal column used to calculate
            coverage.
        node_column (str | None): Optional node or municipality column.

    Returns:
        pd.DataFrame: Two-column metadata table with ``Propiedad`` and
        ``Valor``.
    """

    timestamp = generated_at or datetime.now()
    numeric_count = len(df.select_dtypes(include="number").columns)
    categorical_count = len(df.select_dtypes(exclude="number").columns)
    start_date = ""
    end_date = ""
    if date_column and date_column in df.columns:
        dates = pd.to_datetime(df[date_column], errors="coerce")
        start_date = dates.min()
        end_date = dates.max()

    node_count = df[node_column].nunique() if node_column and node_column in df.columns else ""
    values = {
        "Nombre de la capa": layer_name,
        "Codigo": layer_code,
        "Version": version,
        "Responsable": responsible,
        "Fuente": source,
        "URL Fuente": source_url,
        "Fecha ETL": timestamp.strftime("%Y-%m-%d"),
        "Hora ETL": timestamp.strftime("%H:%M:%S"),
        "Registros": len(df),
        "Variables": len(df.columns),
        "Variables Numericas": numeric_count,
        "Variables Categoricas": categorical_count,
        "Valores Nulos": int(df.isna().sum().sum()),
        "Duplicados": int(df.duplicated().sum()),
        "Cobertura Temporal Inicio": start_date,
        "Cobertura Temporal Fin": end_date,
        "Numero de Nodos": node_count,
        "Estado": "Generado",
        "Framework": "FRAMEWORK V7",
    }
    return pd.DataFrame({"Propiedad": list(values), "Valor": list(values.values())})


def build_data_dictionary(
    df: pd.DataFrame,
    source: str = "",
    descriptions: Mapping[str, str] | None = None,
    units: Mapping[str, str] | None = None,
    required_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Build the data dictionary exported as ``04_Diccionario_Datos.xlsx``.

    Args:
        df (pd.DataFrame): Dataset whose columns will be documented.
        source (str): Source label assigned to variables without a specific
            origin.
        descriptions (Mapping[str, str] | None): Optional descriptions keyed
            by column name.
        units (Mapping[str, str] | None): Optional measurement units keyed by
            column name.
        required_columns (Sequence[str] | None): Columns marked as mandatory.

    Returns:
        pd.DataFrame: Column-level dictionary with types, nulls, examples,
        descriptions, units and origin.
    """

    descriptions = descriptions or {}
    units = units or {}
    required = set(required_columns or [])
    rows = []
    for column in df.columns:
        sample = df[column].dropna().head(1)
        rows.append(
            {
                "Campo": column,
                "Tipo_Dato": str(df[column].dtype),
                "Nulos": int(df[column].isna().sum()),
                "No_Nulos": int(df[column].notna().sum()),
                "Valores_Unicos": int(df[column].nunique(dropna=True)),
                "Ejemplo": str(sample.iloc[0]) if not sample.empty else "",
                "Descripcion": descriptions.get(column, ""),
                "Unidad": units.get(column, ""),
                "Origen": source,
                "Obligatorio": "Si" if column in required else "No",
            }
        )
    return pd.DataFrame(rows)


def audit_layer_dataset(
    df: pd.DataFrame,
    date_column: str | None = None,
    node_column: str | None = None,
) -> pd.DataFrame:
    """Create a compact audit table for a framework layer.

    Args:
        df (pd.DataFrame): Dataset to audit.
        date_column (str | None): Optional date column used for temporal
            coverage metrics.
        node_column (str | None): Optional node column used for spatial or
            territorial coverage.

    Returns:
        pd.DataFrame: Audit metrics with ``Indicador`` and ``Valor`` columns.
    """

    metrics: list[dict[str, object]] = [
        {"Indicador": "Registros", "Valor": len(df)},
        {"Indicador": "Variables", "Valor": len(df.columns)},
        {"Indicador": "Valores Nulos", "Valor": int(df.isna().sum().sum())},
        {"Indicador": "Duplicados", "Valor": int(df.duplicated().sum())},
        {
            "Indicador": "Variables Numericas",
            "Valor": len(df.select_dtypes(include="number").columns),
        },
    ]
    if date_column and date_column in df.columns:
        dates = pd.to_datetime(df[date_column], errors="coerce")
        metrics.extend(
            [
                {"Indicador": "Cobertura Temporal Inicio", "Valor": dates.min()},
                {"Indicador": "Cobertura Temporal Fin", "Valor": dates.max()},
            ]
        )
    if node_column and node_column in df.columns:
        metrics.append({"Indicador": "Nodos", "Valor": int(df[node_column].nunique())})
    return pd.DataFrame(metrics)


def quality_indicators(
    df: pd.DataFrame,
    date_column: str | None = None,
    node_column: str | None = None,
) -> pd.DataFrame:
    """Calculate data-quality indicators for a framework layer.

    Args:
        df (pd.DataFrame): Dataset to evaluate.
        date_column (str | None): Optional date column for temporal span.
        node_column (str | None): Optional node column for territorial span.

    Returns:
        pd.DataFrame: Indicator table with completeness, null percentage,
        duplicate percentage, numeric-variable count and optional coverage.
    """

    total_cells = max(len(df) * len(df.columns), 1)
    nulls = int(df.isna().sum().sum())
    duplicates = int(df.duplicated().sum())
    rows: list[dict[str, object]] = [
        {
            "Indicador": "Completitud (%)",
            "Valor": round((1 - nulls / total_cells) * 100, 2),
        },
        {
            "Indicador": "Porcentaje Nulos (%)",
            "Valor": round(nulls / total_cells * 100, 2),
        },
        {
            "Indicador": "Duplicidad (%)",
            "Valor": round(duplicates / max(len(df), 1) * 100, 2),
        },
        {
            "Indicador": "Variables Numericas",
            "Valor": len(df.select_dtypes(include="number").columns),
        },
        {
            "Indicador": "Hashes Unicos",
            "Valor": int(df["Hash"].nunique()) if "Hash" in df.columns else "",
        },
    ]
    if date_column and date_column in df.columns:
        dates = pd.to_datetime(df[date_column], errors="coerce")
        valid_dates = dates.dropna()
        rows.append(
            {
                "Indicador": "Meses Observados",
                "Valor": int(valid_dates.dt.to_period("M").nunique()),
            }
        )
    if node_column and node_column in df.columns:
        rows.append({"Indicador": "Nodos Observados", "Valor": int(df[node_column].nunique())})
    return pd.DataFrame(rows)


def eda_summary(
    df: pd.DataFrame,
    group_columns: Sequence[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Build exploratory summary tables for notebook reports.

    Args:
        df (pd.DataFrame): Dataset to profile.
        group_columns (Sequence[str] | None): Optional categorical columns used
            to calculate frequency tables.

    Returns:
        dict[str, pd.DataFrame]: Dictionary with descriptive statistics,
        missing-value summary and one frequency table per requested column.
    """

    reports: dict[str, pd.DataFrame] = {
        "descriptivos": (
            df.describe(include="all").T.reset_index().rename(columns={"index": "Variable"})
        ),
        "nulos": df.isna().sum().reset_index(name="Nulos").rename(columns={"index": "Variable"}),
    }
    for column in group_columns or []:
        if column in df.columns:
            reports[f"frecuencia_{column}"] = df[column].value_counts(dropna=False).reset_index()
    return reports


def build_layer_framework_artifacts(
    df: pd.DataFrame,
    layer_name: str,
    layer_code: str,
    version: str,
    responsible: str,
    source: str = "",
    source_url: str = "",
    date_column: str | None = None,
    node_column: str | None = None,
    descriptions: Mapping[str, str] | None = None,
    units: Mapping[str, str] | None = None,
    required_columns: Sequence[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """Build all standard C01-C07 framework artifacts in memory.

    Args:
        df (pd.DataFrame): Input layer dataset.
        layer_name (str): Human-readable layer name.
        layer_code (str): Layer code used to generate record identifiers.
        version (str): Framework version.
        responsible (str): Responsible person or team.
        source (str): Source label.
        source_url (str): Source URL or reference.
        date_column (str | None): Optional date column used for coverage.
        node_column (str | None): Optional node column used for coverage.
        descriptions (Mapping[str, str] | None): Optional data dictionary
            descriptions.
        units (Mapping[str, str] | None): Optional units by column.
        required_columns (Sequence[str] | None): Optional mandatory columns.

    Returns:
        dict[str, pd.DataFrame]: Artifacts keyed by logical name:
        ``dataset``, ``metadata``, ``diccionario``, ``auditoria``,
        ``indicadores`` and EDA summaries.
    """

    dataset = add_framework_governance_columns(df, layer_code, version, responsible)
    return {
        "dataset": dataset,
        "metadata": build_layer_metadata(
            dataset,
            layer_name=layer_name,
            layer_code=layer_code,
            version=version,
            responsible=responsible,
            source=source,
            source_url=source_url,
            date_column=date_column,
            node_column=node_column,
        ),
        "diccionario": build_data_dictionary(
            dataset,
            source,
            descriptions,
            units,
            required_columns,
        ),
        "auditoria": audit_layer_dataset(dataset, date_column, node_column),
        "indicadores": quality_indicators(dataset, date_column, node_column),
        **eda_summary(dataset, [column for column in [date_column, node_column] if column]),
    }


def export_layer_framework_artifacts(
    artifacts: Mapping[str, pd.DataFrame],
    output_dir: Path,
    main_filename: str,
) -> list[Path]:
    """Export standard framework artifacts to disk.

    Args:
        artifacts (Mapping[str, pd.DataFrame]): Output from
            ``build_layer_framework_artifacts``.
        output_dir (Path): Destination directory.
        main_filename (str): File name for the main framework dataset.

    Returns:
        list[Path]: Paths written to disk.
    """

    file_names = {
        "dataset": main_filename,
        "metadata": "03_Metadata.xlsx",
        "diccionario": "04_Diccionario_Datos.xlsx",
        "auditoria": "05_Auditoria.xlsx",
        "indicadores": "06_Indicadores.xlsx",
        "descriptivos": "07_EDA_Descriptivos.xlsx",
        "nulos": "07_EDA_Nulos.xlsx",
    }
    written = []
    for artifact_name, table in artifacts.items():
        file_name = file_names.get(artifact_name)
        if file_name and isinstance(table, pd.DataFrame):
            written.append(save_table(table, output_dir / file_name))
    return written


def framework_notebook_inventory() -> pd.DataFrame:
    """Inventory the framework-construction notebooks for C01-C07.

    Returns:
        pd.DataFrame: One row per ``FW7_C0X_Framework`` notebook with code
        cells, markdown cells, detected repeated helper functions and path.
    """

    rows = []
    for notebook_path in sorted(NOTEBOOKS_DIR.glob(f"C0*/{FRAMEWORK_NOTEBOOK_PATTERN}")):
        text = notebook_path.read_text(encoding="utf-8")
        rows.append(
            {
                "Notebook": notebook_path.name,
                "Ruta": rel(notebook_path),
                "Usa_Hash_Local": "generar_hash" in text,
                "Exporta_Metadata": "03_Metadata.xlsx" in text,
                "Exporta_Diccionario": "04_Diccionario_Datos.xlsx" in text,
                "Exporta_Auditoria": "05_Auditoria.xlsx" in text,
                "Exporta_Indicadores": "06_Indicadores.xlsx" in text,
            }
        )
    return pd.DataFrame(rows)


def summarize_layer_framework_outputs() -> pd.DataFrame:
    """Summarize curated layer outputs already present in ``DATA/MASTER``.

    Returns:
        pd.DataFrame: Summary with layer name, main artifact availability and
        dimensions loaded through the layer modules.
    """

    rows = []
    for layer_name, module in LAYER_MODULES.items():
        config = LAYER_CATALOG[layer_name]
        dataset = module.load_dataset()
        main_path = config["folder"] / config["main"]
        rows.append(
            {
                "Capa": layer_name,
                "Archivo_Principal": rel(main_path),
                "Disponible": main_path.exists(),
                "Filas": len(dataset),
                "Columnas": len(dataset.columns),
                "Variables_Clave": len(module.available_key_variables()),
            }
        )
    return pd.DataFrame(rows)
