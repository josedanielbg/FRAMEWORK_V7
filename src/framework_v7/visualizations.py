"""Reusable Streamlit visualization components.

This module contains UI widgets and charts that can be shared across dashboard
views. Keeping visual functions outside ``app.py`` makes the app easier to read
and makes each visualization independently testable at the import level.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from .paths import rel
from .profiling import dataset_profile, find_date_column, find_node_column, missing_profile, numeric_columns


def render_dataset_metrics(df: pd.DataFrame) -> None:
    """Render common dataset quality metrics.

    Args:
        df: Dataset to summarize.

    Returns:
        None. The function writes Streamlit components.
    """

    profile = dataset_profile(df)
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Filas", f"{profile['filas']:,}")
    c2.metric("Columnas", f"{profile['columnas']:,}")
    c3.metric("Variables numericas", f"{profile['numericas']:,}")
    c4.metric("Nulos", f"{profile['nulos']:,}")
    c5.metric("Duplicados", f"{profile['duplicados']:,}")


def render_missing_file(path: Path) -> None:
    """Render a Streamlit error for a missing artifact.

    Args:
        path: Expected file path.

    Returns:
        None.
    """

    st.error(f"No se encontro el archivo: `{rel(path)}`")


def render_missing_profile(df: pd.DataFrame) -> None:
    """Render a bar chart of variable coverage.

    Args:
        df: Dataset to inspect.

    Returns:
        None.
    """

    missing = missing_profile(df)
    if missing.empty:
        return
    fig = px.bar(
        missing.sort_values("Cobertura_%"),
        x="Cobertura_%",
        y="Variable",
        orientation="h",
        title="Cobertura de variables principales",
        color="Cobertura_%",
        color_continuous_scale=["#E76F51", "#F4A261", "#2A9D8F"],
    )
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, use_container_width=True)


def render_numeric_overview(df: pd.DataFrame, key_prefix: str) -> None:
    """Render boxplots and correlations for numeric variables.

    Args:
        df: Dataset containing numeric columns.
        key_prefix: Unique prefix for Streamlit widget keys.

    Returns:
        None.
    """

    cols = numeric_columns(df)
    if not cols:
        st.info("Este dataset no tiene columnas numericas para graficar.")
        return
    selected = st.multiselect(
        "Variables numericas",
        cols,
        default=cols[: min(6, len(cols))],
        key=f"{key_prefix}_numeric_cols",
    )
    if not selected:
        return
    melted = df[selected].melt(var_name="Variable", value_name="Valor")
    fig = px.box(melted, x="Variable", y="Valor", points=False, title="Distribucion por variable")
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, use_container_width=True)

    if len(selected) >= 2:
        corr = df[selected].corr(numeric_only=True)
        fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Matriz de correlacion")
        fig.update_layout(height=520, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)


def render_time_series(df: pd.DataFrame, key_prefix: str) -> None:
    """Render a time-series chart when temporal data is available.

    Args:
        df: Dataset with a date-like column and numeric variables.
        key_prefix: Unique prefix for Streamlit widget keys.

    Returns:
        None.
    """

    date_col = find_date_column(df)
    cols = numeric_columns(df)
    if date_col is None or not cols:
        st.info("No se detecto una columna temporal y numerica compatible.")
        return
    view = df.copy()
    view[date_col] = pd.to_datetime(view[date_col], errors="coerce")
    view = view.dropna(subset=[date_col])
    if view.empty:
        st.info("La columna temporal no pudo convertirse a fechas.")
        return
    variable = st.selectbox("Variable", cols, key=f"{key_prefix}_series_var")
    node_col = find_node_column(view)
    color = node_col if node_col and view[node_col].nunique() <= 12 else None
    fig = px.line(view.sort_values(date_col), x=date_col, y=variable, color=color, title=f"Serie temporal: {variable}")
    fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, use_container_width=True)


def render_layer_images(folder: Path) -> None:
    """Render PNG artifacts stored in a layer folder.

    Args:
        folder: Layer folder containing optional PNG charts.

    Returns:
        None.
    """

    images = sorted(folder.glob("*.png"))
    if not images:
        return
    cols = st.columns(min(3, len(images)))
    for index, image in enumerate(images):
        with cols[index % len(cols)]:
            st.image(str(image), caption=image.name, use_container_width=True)


def render_system_map() -> None:
    """Render the multicapa Sankey diagram.

    Returns:
        None. The function writes a Plotly chart to Streamlit.
    """

    nodes = [
        "Clima",
        "Hidrologia",
        "Calidad",
        "ONI",
        "Hidraulica",
        "Percepcion",
        "Gobernanza",
        "Dataset maestro",
        "Modelo Exp01",
        "IRCA",
    ]
    links = [
        ("Clima", "Dataset maestro", 4),
        ("Hidrologia", "Dataset maestro", 3),
        ("Calidad", "Dataset maestro", 3),
        ("ONI", "Dataset maestro", 2),
        ("Hidraulica", "Dataset maestro", 2),
        ("Percepcion", "Dataset maestro", 2),
        ("Gobernanza", "Dataset maestro", 2),
        ("Dataset maestro", "Modelo Exp01", 6),
        ("Modelo Exp01", "IRCA", 6),
    ]
    index = {name: position for position, name in enumerate(nodes)}
    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=18,
                    thickness=18,
                    label=nodes,
                    color=[
                        "#2A9D8F",
                        "#457B9D",
                        "#E76F51",
                        "#F4A261",
                        "#264653",
                        "#8E7DBE",
                        "#6A994E",
                        "#A8DADC",
                        "#1D3557",
                        "#E63946",
                    ],
                ),
                link=dict(
                    source=[index[source] for source, _, _ in links],
                    target=[index[target] for _, target, _ in links],
                    value=[value for _, _, value in links],
                    color="rgba(69, 123, 157, 0.25)",
                ),
            )
        ]
    )
    fig.update_layout(title_text="Mapa multicapa del experimento", height=430, margin=dict(l=10, r=10, t=55, b=10))
    st.plotly_chart(fig, use_container_width=True)
