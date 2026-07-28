from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


BASE_DIR = Path(__file__).resolve().parent
PREDICTIONS_PATH = BASE_DIR / "DATA" / "EVALUACIONES" / "Exp01" / "predicciones.csv"
METADATA_PATH = BASE_DIR / "DATA" / "EVALUACIONES" / "Exp01" / "metadata_prediccion.csv"
ML_DATASET_PATH = (
    BASE_DIR
    / "DATA"
    / "MACHINE_LEARNING"
    / "C13_MACHINE_LEARNING"
    / "Transformaciones"
    / "Exp01"
    / "dataset_machine_learning_transformado.csv"
)
DIAGNOSTIC_PATH = (
    BASE_DIR
    / "DATA"
    / "MACHINE_LEARNING"
    / "C13_MACHINE_LEARNING"
    / "Diagnostico"
    / "diagnostico_estadistico_ml.csv"
)
MASTER_PATH = BASE_DIR / "DATA" / "MASTER" / "C09_MASTER" / "Dataset_Maestro_Framework_v03_Con_Imputaciones.csv"

LAYERS = {
    "C01 - Climatica": BASE_DIR / "DATA" / "MASTER" / "C01_MASTER" / "02_Capa_Climatica_Framework.xlsx",
    "C02 - Hidrologica": BASE_DIR / "DATA" / "MASTER" / "C02_MASTER" / "02_Capa_Hidrologica_Framework.xlsx",
    "C03 - Calidad de agua": BASE_DIR / "DATA" / "MASTER" / "C03_MASTER" / "02_Capa_Calidad_Agua_Framework.xlsx",
    "C04 - ONI": BASE_DIR / "DATA" / "MASTER" / "C04_MASTER" / "02_Capa_ONI_Framework.xlsx",
    "C05 - Hidraulica": BASE_DIR / "DATA" / "MASTER" / "C05_MASTER" / "02_Capa_Hidraulica_Framework.xlsx",
    "C06 - Percepcion": BASE_DIR / "DATA" / "MASTER" / "C06_MASTER" / "02_Capa_Percepcion_Framework.xlsx",
    "C07 - Gobernanza": BASE_DIR / "DATA" / "MASTER" / "CO7_MASTER" / "02_Capa_Gobernanza_Framework.xlsx",
}

FEATURE_GROUPS = {
    "Clima": ["Precipitacion_mm", "Temp_Min_C", "Radiacion_Solar", "Humedad_Relativa", "Velocidad_Viento", "ONI"],
    "Hidraulica": ["VolumenUtilDiarioMasa"],
    "Objetivo": ["irca"],
}


st.set_page_config(
    page_title="FRAMEWORK V7 | Experimento 01",
    page_icon="DATA/MASTER/C01_MASTER/Correlacion.png",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_excel(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_excel(path)


def numeric_columns(df: pd.DataFrame) -> list[str]:
    return list(df.select_dtypes(include="number").columns)


def metadata_dict(df: pd.DataFrame) -> dict[str, str]:
    if {"Parametro", "Valor"}.issubset(df.columns):
        return dict(zip(df["Parametro"].astype(str), df["Valor"].astype(str)))
    return {}


def quality_badge(value: float) -> str:
    if value <= 0.25:
        return "Bajo"
    if value <= 0.50:
        return "Medio"
    if value <= 0.75:
        return "Alto"
    return "Critico"


def normalize_01(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    span = values.max() - values.min()
    if pd.isna(span) or span == 0:
        return pd.Series([0.0] * len(values), index=series.index)
    return (values - values.min()) / span


def render_missing_file(path: Path) -> None:
    st.error(f"No se encontro el archivo: `{path.relative_to(BASE_DIR)}`")


metadata = load_csv(METADATA_PATH)
predictions = load_csv(PREDICTIONS_PATH)
ml_dataset = load_csv(ML_DATASET_PATH)
diagnostic = load_csv(DIAGNOSTIC_PATH)
master = load_csv(MASTER_PATH)
meta = metadata_dict(metadata)

with st.sidebar:
    st.title("FRAMEWORK V7")
    section = st.radio(
        "Vista",
        [
            "Resumen",
            "Predicciones",
            "Dataset ML",
            "Capas sistemicas",
            "Datos maestros",
        ],
    )
    st.divider()
    st.caption("Experimento activo")
    st.write(meta.get("Experimento", "Exp01"))
    st.caption("Variable objetivo")
    st.write(meta.get("Variable Objetivo", "irca"))

st.title("Framework de gestion hidrica - Experimento 01")
st.caption("Visor Streamlit para resultados predictivos, dataset transformado y capas del sistema.")

if section == "Resumen":
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Objetivo", meta.get("Variable Objetivo", "irca"))
    col2.metric("Ventana", meta.get("Ventana", "12"))
    col3.metric("Predictoras", meta.get("Variables Predictoras", "-"))
    col4.metric("Predicciones", meta.get("Predicciones Generadas", str(len(predictions))))

    if not predictions.empty and "Prediccion" in predictions.columns:
        prediction_values = pd.to_numeric(predictions["Prediccion"], errors="coerce")
        c1, c2, c3 = st.columns(3)
        c1.metric("Prediccion media", f"{prediction_values.mean():.3f}")
        c2.metric("Prediccion minima", f"{prediction_values.min():.3f}")
        c3.metric("Prediccion maxima", f"{prediction_values.max():.3f}")

        fig = px.line(
            predictions,
            x="Registro" if "Registro" in predictions.columns else predictions.index,
            y="Prediccion",
            title="Serie de predicciones del experimento",
            markers=False,
        )
        fig.update_layout(height=390, margin=dict(l=10, r=10, t=55, b=10))
        st.plotly_chart(fig, use_container_width=True)
    else:
        render_missing_file(PREDICTIONS_PATH)

    st.subheader("Metadata")
    st.dataframe(metadata, use_container_width=True, hide_index=True)

elif section == "Predicciones":
    if predictions.empty:
        render_missing_file(PREDICTIONS_PATH)
    else:
        view = predictions.copy()
        if "Prediccion" in view.columns:
            view["Prediccion"] = pd.to_numeric(view["Prediccion"], errors="coerce")
            view["Prediccion_normalizada"] = normalize_01(view["Prediccion"])
            view["Categoria_relativa"] = view["Prediccion_normalizada"].apply(quality_badge)

        min_reg = int(view["Registro"].min()) if "Registro" in view.columns else 0
        max_reg = int(view["Registro"].max()) if "Registro" in view.columns else len(view) - 1
        if "Registro" in view.columns and min_reg < max_reg:
            selected = st.slider("Rango de registros", min_reg, max_reg, (min_reg, max_reg))
            view = view[(view["Registro"] >= selected[0]) & (view["Registro"] <= selected[1])]

        left, right = st.columns([2, 1])
        with left:
            fig = px.line(
                view,
                x="Registro" if "Registro" in view.columns else view.index,
                y="Prediccion",
                color="Categoria_relativa" if "Categoria_relativa" in view.columns else None,
                title="Predicciones filtradas",
            )
            fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with right:
            fig = px.histogram(view, x="Prediccion", nbins=30, title="Distribucion")
            fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig, use_container_width=True)

        st.dataframe(view, use_container_width=True, hide_index=True)
        st.download_button(
            "Descargar predicciones filtradas",
            data=view.to_csv(index=False).encode("utf-8"),
            file_name="predicciones_exp01_filtradas.csv",
            mime="text/csv",
        )

elif section == "Dataset ML":
    if ml_dataset.empty:
        render_missing_file(ML_DATASET_PATH)
    else:
        st.subheader("Dataset transformado para machine learning")
        cols = numeric_columns(ml_dataset)
        selected_cols = st.multiselect("Variables", cols, default=cols[: min(8, len(cols))])

        c1, c2, c3 = st.columns(3)
        c1.metric("Filas", f"{len(ml_dataset):,}")
        c2.metric("Columnas", f"{ml_dataset.shape[1]:,}")
        c3.metric("Nulos", f"{int(ml_dataset.isna().sum().sum()):,}")

        if selected_cols:
            melted = ml_dataset[selected_cols].melt(var_name="Variable", value_name="Valor")
            fig = px.box(melted, x="Variable", y="Valor", points=False, title="Distribucion de variables")
            fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig, use_container_width=True)

            corr = ml_dataset[selected_cols].corr(numeric_only=True)
            fig = px.imshow(corr, text_auto=".2f", aspect="auto", title="Correlacion de variables seleccionadas")
            fig.update_layout(height=520, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig, use_container_width=True)

        with st.expander("Diagnostico estadistico"):
            st.dataframe(diagnostic, use_container_width=True, hide_index=True)

        st.dataframe(ml_dataset.head(200), use_container_width=True, hide_index=True)

elif section == "Capas sistemicas":
    st.subheader("Lectura por capas")
    rows = []
    for name, path in LAYERS.items():
        df = load_excel(path)
        rows.append(
            {
                "Capa": name,
                "Archivo": str(path.relative_to(BASE_DIR)),
                "Filas": len(df),
                "Columnas": df.shape[1],
                "Nulos": int(df.isna().sum().sum()) if not df.empty else None,
                "Disponible": path.exists(),
            }
        )
    summary = pd.DataFrame(rows)
    st.dataframe(summary, use_container_width=True, hide_index=True)

    selected_layer = st.selectbox("Capa", list(LAYERS))
    layer_df = load_excel(LAYERS[selected_layer])
    if layer_df.empty:
        render_missing_file(LAYERS[selected_layer])
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("Filas", f"{len(layer_df):,}")
        c2.metric("Columnas", f"{layer_df.shape[1]:,}")
        c3.metric("Nulos", f"{int(layer_df.isna().sum().sum()):,}")
        st.dataframe(layer_df.head(300), use_container_width=True, hide_index=True)

elif section == "Datos maestros":
    if master.empty:
        render_missing_file(MASTER_PATH)
    else:
        st.subheader("Dataset maestro con imputaciones")
        nodes = sorted(master["Nodo"].dropna().unique()) if "Nodo" in master.columns else []
        selected_nodes = st.multiselect("Nodos", nodes, default=nodes[: min(4, len(nodes))])
        view = master.copy()
        if selected_nodes and "Nodo" in view.columns:
            view = view[view["Nodo"].isin(selected_nodes)]

        if "Fecha" in view.columns:
            view["Fecha"] = pd.to_datetime(view["Fecha"], errors="coerce")
            min_date = view["Fecha"].min()
            max_date = view["Fecha"].max()
            c1, c2, c3 = st.columns(3)
            c1.metric("Filas", f"{len(view):,}")
            c2.metric("Inicio", min_date.date() if pd.notna(min_date) else "-")
            c3.metric("Fin", max_date.date() if pd.notna(max_date) else "-")

        available_groups = {
            group: [col for col in cols if col in view.columns] for group, cols in FEATURE_GROUPS.items()
        }
        for group, cols in available_groups.items():
            if cols:
                st.markdown(f"**{group}**")
                chart_data = view[["Fecha", "Nodo", *cols]].copy() if {"Fecha", "Nodo"}.issubset(view.columns) else view[cols].copy()
                variable = st.selectbox(f"Variable {group}", cols, key=group)
                if {"Fecha", "Nodo", variable}.issubset(chart_data.columns):
                    fig = px.line(chart_data, x="Fecha", y=variable, color="Nodo", title=f"{variable} por nodo")
                    fig.update_layout(height=390, margin=dict(l=10, r=10, t=55, b=10))
                    st.plotly_chart(fig, use_container_width=True)

        st.dataframe(view.head(500), use_container_width=True, hide_index=True)
