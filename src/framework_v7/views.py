"""Streamlit view functions for FRAMEWORK V7.

Each function renders one high-level screen in the application. The view layer
depends on data-access, profiling and visualization helpers, but it does not
define business catalog constants or read files directly except through helper
functions.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from .catalog import FEATURE_GROUPS, LAYER_CATALOG, MASTER_FILES, SUPPORT_FILES
from .data_access import ProjectData, load_csv, load_excel, read_text
from .layers import LAYER_MODULES
from .paths import BASE_DIR, COVERAGE_PATH, ML_DATASET_PATH, NOTEBOOKS_DIR, PREDICTIONS_PATH, rel
from .profiling import find_date_column, layer_summary, normalize_01, quality_badge
from .visualizations import (
    render_dataset_metrics,
    render_layer_images,
    render_missing_file,
    render_missing_profile,
    render_numeric_overview,
    render_system_map,
    render_time_series,
)


def render_sidebar(data: ProjectData) -> str:
    """Render the sidebar and return the selected section.

    Args:
        data: Loaded project datasets and metadata.

    Returns:
        Name of the selected dashboard section.
    """

    with st.sidebar:
        st.title("FRAMEWORK V7")
        section = st.radio(
            "Vista",
            [
                "Dashboard",
                "Experimento 01",
                "Datasets por capas",
                "Dataset maestro",
                "Notebooks",
            ],
        )
        st.divider()
        st.caption("Experimento activo")
        st.write(data.meta.get("Experimento", "Exp01"))
        st.caption("Variable objetivo")
        st.write(data.meta.get("Variable Objetivo", "irca"))
        st.caption("Fecha ejecucion")
        st.write(data.meta.get("Fecha Ejecucion", "-"))
    return section


def render_dashboard(data: ProjectData) -> None:
    """Render the executive multicapa dashboard.

    Args:
        data: Loaded project datasets and metadata.

    Returns:
        None.
    """

    st.subheader("Resumen ejecutivo")
    layers = layer_summary()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Capas sistemicas", f"{int(layers['Disponible'].sum())}/{len(layers)}")
    c2.metric("Dataset maestro", f"{len(data.master):,} filas")
    c3.metric("Variables maestro", f"{data.master.shape[1]:,}")
    c4.metric("Predicciones Exp01", data.meta.get("Predicciones Generadas", str(len(data.predictions))))

    tab_map, tab_pred, tab_layers, tab_quality = st.tabs(
        ["Mapa del sistema", "Predicciones", "Capas", "Calidad de datos"]
    )
    with tab_map:
        render_system_map()
        st.dataframe(layers[["Capa", "Rol sistemico", "Filas", "Columnas", "Nulos"]], use_container_width=True, hide_index=True)

    with tab_pred:
        if data.predictions.empty:
            render_missing_file(PREDICTIONS_PATH)
        else:
            view = data.predictions.copy()
            if "Prediccion" in view.columns:
                view["Prediccion"] = pd.to_numeric(view["Prediccion"], errors="coerce")
                view["Prediccion_suavizada"] = view["Prediccion"].rolling(12, min_periods=1).mean()
                fig = px.line(view, x="Registro", y=["Prediccion", "Prediccion_suavizada"], title="Prediccion y tendencia movil")
                fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
                st.plotly_chart(fig, use_container_width=True)

    with tab_layers:
        fig = px.bar(
            layers,
            x="Capa",
            y="Filas",
            color="Nulos",
            title="Volumen de datos por capa",
            color_continuous_scale=["#2A9D8F", "#F4A261", "#E76F51"],
        )
        fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)

    with tab_quality:
        if not data.coverage.empty:
            st.dataframe(data.coverage, use_container_width=True, hide_index=True)
        else:
            render_missing_profile(data.master)


def render_experiment(data: ProjectData) -> None:
    """Render the Exp01 experiment view.

    Args:
        data: Loaded project datasets and metadata.

    Returns:
        None.
    """

    tab_summary, tab_series, tab_dist, tab_ml, tab_metadata = st.tabs(
        ["Resumen", "Serie", "Distribucion", "Variables ML", "Metadata"]
    )
    with tab_summary:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Objetivo", data.meta.get("Variable Objetivo", "irca"))
        col2.metric("Modelo", data.meta.get("Modelo", "-"))
        col3.metric("Ventana", data.meta.get("Ventana", "12"))
        col4.metric("Predictoras", data.meta.get("Variables Predictoras", "-"))
        if not data.predictions.empty and "Prediccion" in data.predictions.columns:
            values = pd.to_numeric(data.predictions["Prediccion"], errors="coerce")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Media", f"{values.mean():.3f}")
            c2.metric("Minimo", f"{values.min():.3f}")
            c3.metric("Maximo", f"{values.max():.3f}")
            c4.metric("Desviacion", f"{values.std():.3f}")

    with tab_series:
        if data.predictions.empty:
            render_missing_file(PREDICTIONS_PATH)
        else:
            view = data.predictions.copy()
            view["Prediccion"] = pd.to_numeric(view["Prediccion"], errors="coerce")
            view["Normalizada"] = normalize_01(view["Prediccion"])
            view["Categoria"] = view["Normalizada"].apply(quality_badge)
            selected = st.slider(
                "Rango de registros",
                int(view["Registro"].min()),
                int(view["Registro"].max()),
                (int(view["Registro"].min()), int(view["Registro"].max())),
            )
            view = view[(view["Registro"] >= selected[0]) & (view["Registro"] <= selected[1])]
            fig = px.area(view, x="Registro", y="Normalizada", color="Categoria", title="Intensidad relativa de prediccion")
            fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(view, use_container_width=True, hide_index=True)

    with tab_dist:
        if not data.predictions.empty and "Prediccion" in data.predictions.columns:
            values_df = data.predictions.copy()
            values_df["Prediccion"] = pd.to_numeric(values_df["Prediccion"], errors="coerce")
            left, right = st.columns(2)
            with left:
                fig = px.histogram(values_df, x="Prediccion", nbins=35, marginal="box", title="Histograma con caja")
                fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
                st.plotly_chart(fig, use_container_width=True)
            with right:
                values_df["Categoria"] = normalize_01(values_df["Prediccion"]).apply(quality_badge)
                counts = values_df["Categoria"].value_counts().reset_index()
                counts.columns = ["Categoria", "Registros"]
                fig = px.pie(counts, names="Categoria", values="Registros", hole=0.45, title="Categorias relativas")
                fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
                st.plotly_chart(fig, use_container_width=True)

    with tab_ml:
        if data.ml_dataset.empty:
            render_missing_file(ML_DATASET_PATH)
        else:
            render_dataset_metrics(data.ml_dataset)
            render_numeric_overview(data.ml_dataset, "exp_ml")

    with tab_metadata:
        st.dataframe(data.metadata, use_container_width=True, hide_index=True)
        if not data.diagnostic.empty:
            st.markdown("**Diagnostico estadistico**")
            st.dataframe(data.diagnostic, use_container_width=True, hide_index=True)


def render_layers() -> None:
    """Render the dataset explorer for every system layer.

    Returns:
        None.
    """

    st.subheader("Datasets de las capas")
    layers = layer_summary()
    st.dataframe(layers, use_container_width=True, hide_index=True)
    layer_tabs = st.tabs(list(LAYER_MODULES))
    for tab, (layer_name, layer_module) in zip(layer_tabs, LAYER_MODULES.items()):
        with tab:
            config = LAYER_CATALOG[layer_name]
            folder = config["folder"]
            main_path = folder / config["main"]
            df = layer_module.load_dataset()
            st.markdown(f"**Rol sistemico:** {config['role']}")
            st.caption(rel(main_path))
            if df.empty:
                render_missing_file(main_path)
                continue

            sub_summary, sub_data, sub_visual, sub_docs = st.tabs(["Resumen", "Datos", "Visual", "Soporte"])
            with sub_summary:
                render_dataset_metrics(df)
                key_variables = layer_module.available_key_variables()
                if key_variables:
                    st.markdown("**Variables clave detectadas**")
                    st.write(", ".join(key_variables))
                else:
                    st.info("No se detectaron variables clave declaradas para esta capa.")
                render_missing_profile(df)

            with sub_data:
                compact = layer_module.feature_frame()
                data_tab, compact_tab = st.tabs(["Dataset completo", "Vista compacta"])
                with data_tab:
                    st.dataframe(df.head(500), use_container_width=True, hide_index=True)
                with compact_tab:
                    st.dataframe(compact.head(500), use_container_width=True, hide_index=True)
                st.download_button(
                    f"Descargar muestra {layer_name}",
                    data=df.head(500).to_csv(index=False).encode("utf-8"),
                    file_name=f"{layer_name.replace(' ', '_').replace('-', '').lower()}_muestra.csv",
                    mime="text/csv",
                    key=f"download_{layer_name}",
                )

            with sub_visual:
                render_time_series(df, layer_name)
                render_numeric_overview(df, f"layer_{layer_name}")
                render_layer_images(folder)

            with sub_docs:
                for label, file_name in SUPPORT_FILES.items():
                    support_path = folder / file_name
                    support_df = load_excel(support_path)
                    with st.expander(label, expanded=label == "Metadata"):
                        if support_df.empty:
                            render_missing_file(support_path)
                        else:
                            st.caption(rel(support_path))
                            st.dataframe(support_df.head(300), use_container_width=True, hide_index=True)
                readme = read_text(folder / "08_README.md")
                if readme:
                    with st.expander("README de la capa"):
                        st.markdown(readme)


def render_master_dataset() -> None:
    """Render the master dataset explorer.

    Returns:
        None.
    """

    st.subheader("Explorador del dataset maestro")
    selected_file = st.selectbox("Archivo maestro", list(MASTER_FILES))
    selected_path = MASTER_FILES[selected_file]
    dataset = load_csv(selected_path)
    if dataset.empty:
        render_missing_file(selected_path)
        return

    tab_general, tab_series, tab_groups, tab_quality, tab_table = st.tabs(
        ["General", "Series", "Grupos", "Cobertura", "Tabla"]
    )
    with tab_general:
        render_dataset_metrics(dataset)
        st.caption(rel(selected_path))
        date_col = find_date_column(dataset)
        if date_col:
            dates = pd.to_datetime(dataset[date_col], errors="coerce")
            c1, c2 = st.columns(2)
            c1.metric("Inicio", dates.min().date() if pd.notna(dates.min()) else "-")
            c2.metric("Fin", dates.max().date() if pd.notna(dates.max()) else "-")
    with tab_series:
        render_time_series(dataset, "master")
    with tab_groups:
        available_groups = {group: [col for col in cols if col in dataset.columns] for group, cols in FEATURE_GROUPS.items()}
        group = st.selectbox("Grupo", [name for name, cols in available_groups.items() if cols])
        cols = available_groups[group]
        render_numeric_overview(dataset[cols], f"master_group_{group}")
    with tab_quality:
        render_missing_profile(dataset)
        if selected_file != "Cobertura variables" and COVERAGE_PATH.exists():
            st.markdown("**Resumen de cobertura precomputado**")
            st.dataframe(load_csv(COVERAGE_PATH), use_container_width=True, hide_index=True)
    with tab_table:
        st.dataframe(dataset.head(1000), use_container_width=True, hide_index=True)
        st.download_button(
            "Descargar vista completa",
            data=dataset.to_csv(index=False).encode("utf-8"),
            file_name=f"{selected_file.replace(' ', '_').lower()}.csv",
            mime="text/csv",
        )


def render_notebooks() -> None:
    """Render the notebook inventory and documentation view.

    Returns:
        None.
    """

    st.subheader("Memoria metodologica en notebooks")
    st.write(
        "Esta carpeta conserva la exploracion y el paso a paso. La app y los datos quedan separados para que el repositorio funcione como producto reproducible."
    )
    notebooks = sorted(NOTEBOOKS_DIR.rglob("*.ipynb"))
    rows = []
    for notebook in notebooks:
        rows.append({"Notebook": notebook.name, "Carpeta": rel(notebook.parent), "Tamano KB": round(notebook.stat().st_size / 1024, 1)})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    md_files = sorted(NOTEBOOKS_DIR.rglob("*.md"))
    if md_files:
        selected_doc = st.selectbox("Documento", [rel(path) for path in md_files])
        st.markdown(read_text(BASE_DIR / selected_doc))
