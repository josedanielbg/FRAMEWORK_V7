"""Streamlit view functions for FRAMEWORK V7.

Each function renders one high-level screen in the application. The view layer
depends on data-access, profiling and visualization helpers, but it does not
define business catalog constants or read files directly except through helper
functions.
"""

from __future__ import annotations

import base64

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

from .catalog import EXPERIMENT_DESIGN_FILES, FEATURE_GROUPS, LAYER_CATALOG, MASTER_FILES, SUPPORT_FILES
from .data_access import ProjectData, load_csv, load_excel, read_text
from .layers import LAYER_MODULES
from .paths import (
    BASE_DIR,
    COVERAGE_PATH,
    EVALUATIONS_DIR,
    EXPERIMENT_DESIGN_DIR,
    INTERPRETATION_DIR,
    MACHINE_LEARNING_DIR,
    ML_DATASET_PATH,
    MODEL_ACCURACY_IMAGE_PATH,
    MODEL_CARDS_DIR,
    MODEL_DIAGNOSTIC_PATH,
    MODEL_LOSS_IMAGE_PATH,
    MODELING_DIR,
    NOTEBOOKS_DIR,
    PREDICTIONS_PATH,
    rel,
)
from .pipeline.evaluation import (
    evaluation_artifact_inventory,
    load_prediction_metadata,
    load_predictions,
    prediction_distribution,
    summarize_evaluation_experiments,
)
from .pipeline.interpretation import (
    dimension_coverage,
    interpretation_artifact_inventory,
    load_interpretation_summary,
    summarize_interpretation_experiments,
)
from .pipeline.machine_learning import (
    load_sequence_metadata,
    load_transformed_dataset,
    ml_artifact_inventory,
    summarize_ml_experiments,
)
from .pipeline.modeling import (
    load_model_diagnostic,
    load_model_record,
    load_model_recommendations,
    modeling_artifact_inventory,
    summarize_modeling_experiments,
)
from .profiling import find_date_column, layer_summary, normalize_01, quality_badge
from .utils import format_metric_date
from .visualizations import (
    render_dataset_metrics,
    render_layer_images,
    render_missing_file,
    render_missing_profile,
    render_numeric_overview,
    render_system_map,
    render_time_series,
)


def _experiment_names(*frames: pd.DataFrame) -> list[str]:
    """Return sorted experiment identifiers from summary tables.

    Args:
        *frames: DataFrames that may contain an ``Experimento`` column.

    Returns:
        Sorted list of unique experiment names.
    """

    names = set()
    for frame in frames:
        if not frame.empty and "Experimento" in frame.columns:
            names.update(frame["Experimento"].dropna().astype(str))
    return sorted(names)


def _numeric_metric_frame(summary: pd.DataFrame) -> pd.DataFrame:
    """Convert wide experiment metrics into a long chart-ready table.

    Args:
        summary: Experiment summary with metric columns.

    Returns:
        DataFrame with ``Experimento``, ``Metrica`` and ``Valor`` columns.
    """

    metric_columns = ["Accuracy", "Precision", "Recall", "F1", "MAE", "RMSE", "MAPE", "R2"]
    available = [column for column in metric_columns if column in summary.columns]
    if summary.empty or not available or "Experimento" not in summary.columns:
        return pd.DataFrame(columns=["Experimento", "Metrica", "Valor"])

    metrics = summary[["Experimento", *available]].melt(
        id_vars="Experimento",
        var_name="Metrica",
        value_name="Valor",
    )
    metrics["Valor"] = pd.to_numeric(metrics["Valor"], errors="coerce")
    return metrics.dropna(subset=["Valor"])


def _prediction_view(experiment: str) -> pd.DataFrame:
    """Load prediction rows and attach the experiment name.

    Args:
        experiment: Experiment identifier.

    Returns:
        Prediction DataFrame prepared for plotting.
    """

    predictions = load_predictions(experiment).copy()
    if predictions.empty or "Prediccion" not in predictions.columns:
        return pd.DataFrame()
    if "Registro" not in predictions.columns:
        predictions["Registro"] = range(1, len(predictions) + 1)
    predictions["Prediccion"] = pd.to_numeric(predictions["Prediccion"], errors="coerce")
    predictions["Experimento"] = experiment
    predictions["Tendencia"] = predictions["Prediccion"].rolling(12, min_periods=1).mean()
    predictions["Intensidad"] = normalize_01(predictions["Prediccion"])
    return predictions


def _artifact_inventory() -> pd.DataFrame:
    """Build a consolidated artifact inventory for executed experiments.

    Returns:
        DataFrame with stage, experiment, file path, format and size.
    """

    inventories = [
        ("C13 Machine Learning", ml_artifact_inventory()),
        ("C14 Modelado", modeling_artifact_inventory()),
        ("C15 Evaluacion", evaluation_artifact_inventory()),
        ("C16 Interpretacion", interpretation_artifact_inventory()),
    ]
    frames = []
    for stage, inventory in inventories:
        if inventory.empty:
            continue
        frame = inventory.copy()
        frame.insert(0, "Etapa", stage)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _model_card_inventory() -> pd.DataFrame:
    """Build an inventory of available model-card PDF files.

    Returns:
        DataFrame with experiment label, file name, repository path and size.
    """

    if not MODEL_CARDS_DIR.exists():
        return pd.DataFrame(columns=["Experimento", "Archivo", "Ruta", "Tamano_MB"])

    rows = []
    for pdf_path in sorted(MODEL_CARDS_DIR.glob("*.pdf")):
        experiment = pdf_path.stem.replace("_", "-")
        rows.append(
            {
                "Experimento": experiment,
                "Archivo": pdf_path.name,
                "Ruta": rel(pdf_path),
                "Tamano_MB": round(pdf_path.stat().st_size / (1024 * 1024), 2),
            }
        )
    return pd.DataFrame(rows)


def _render_pdf_preview(pdf_path, height: int = 720) -> None:
    """Render a local PDF file inside Streamlit.

    Args:
        pdf_path: Path to the PDF file.
        height: Viewer height in pixels.

    Returns:
        None.
    """

    if not pdf_path.exists():
        render_missing_file(pdf_path)
        return

    encoded_pdf = base64.b64encode(pdf_path.read_bytes()).decode("utf-8")
    components.html(
        f"""
        <iframe
            src="data:application/pdf;base64,{encoded_pdf}"
            width="100%"
            height="{height}"
            style="border: 1px solid #D0D7DE; border-radius: 6px;"
        ></iframe>
        """,
        height=height + 20,
        scrolling=True,
    )


def _research_memory(data: ProjectData) -> pd.DataFrame:
    """Build a research-memory table from project artifacts.

    Args:
        data: Loaded project datasets and metadata.

    Returns:
        DataFrame with research blocks, visible evidence and next steps for
        the thesis defense narrative.
    """

    layers = layer_summary()
    evaluation_summary = summarize_evaluation_experiments()
    interpretation_summary = summarize_interpretation_experiments()
    catalog = data.experiment_design.get("Catalogo de experimentos", pd.DataFrame())
    predictors = data.experiment_design.get("Variables predictoras", pd.DataFrame())
    notebook_count = len(list(NOTEBOOKS_DIR.rglob("*.ipynb")))
    artifact_count = len(_artifact_inventory())
    model_card_count = len(_model_card_inventory())
    pipeline_modules = len(list((BASE_DIR / "src" / "framework_v7" / "pipeline").glob("*.py")))

    layers_available = int(layers["Disponible"].sum()) if not layers.empty else 0
    experiments_available = len(evaluation_summary)
    interpreted_experiments = len(interpretation_summary)
    executed_designs = 0
    if not catalog.empty and "Estado" in catalog.columns:
        executed_designs = int((catalog["Estado"].astype(str) == "Ejecutado").sum())

    rows = [
        {
            "Bloque": "Problema y pertinencia",
            "Estado": "Consolidado",
            "Evidencia": "Tema aplicado a gestion hidrica del Rio Bogota y variable objetivo documentada.",
            "Siguiente paso": "Conectar cada objetivo con una decision concreta de gestion hidrica.",
        },
        {
            "Bloque": "Pensamiento sistemico",
            "Estado": "Consolidado" if layers_available == len(layers) else "Por completar",
            "Evidencia": (
                f"{layers_available}/{len(layers)} capas disponibles: clima, "
                "hidrologia, calidad, ONI, hidraulica, percepcion y gobernanza."
            ),
            "Siguiente paso": "Explicitar relaciones causa-efecto entre capas y variable objetivo.",
        },
        {
            "Bloque": "Datos y trazabilidad",
            "Estado": "Consolidado" if not data.master.empty and layers_available else "Por completar",
            "Evidencia": (
                f"Dataset maestro con {len(data.master):,} filas y "
                f"{data.master.shape[1]:,} columnas."
            ),
            "Siguiente paso": "Mostrar cobertura, nulos, imputaciones y supuestos por capa.",
        },
        {
            "Bloque": "Metodologia reproducible",
            "Estado": "Consolidado" if pipeline_modules >= 10 else "En progreso",
            "Evidencia": (
                f"{pipeline_modules} modulos en src/framework_v7/pipeline y "
                f"{notebook_count} notebooks como memoria."
            ),
            "Siguiente paso": "Agregar pruebas unitarias ligeras para funciones criticas del pipeline.",
        },
        {
            "Bloque": "Diseno experimental",
            "Estado": "Consolidado" if len(catalog) >= 3 else "En progreso",
            "Evidencia": (
                f"{len(catalog)} experimentos definidos y {executed_designs} "
                "marcados como ejecutados."
            ),
            "Siguiente paso": "Justificar por que cada experimento cambia objetivo, arquitectura o horizonte.",
        },
        {
            "Bloque": "Modelado y evaluacion",
            "Estado": "Consolidado" if experiments_available >= 3 else "En progreso",
            "Evidencia": (
                f"{experiments_available} experimentos con predicciones, "
                f"artefactos de evaluacion y {model_card_count} model cards."
            ),
            "Siguiente paso": "Comparar contra modelos base simples y explicar trade-offs.",
        },
        {
            "Bloque": "Interpretacion e impacto",
            "Estado": "Consolidado" if interpreted_experiments else "En progreso",
            "Evidencia": f"{interpreted_experiments} resumenes de interpretacion C16 consolidados.",
            "Siguiente paso": "Traducir metricas a umbrales de accion y limitaciones operativas.",
        },
        {
            "Bloque": "Comunicacion del proyecto",
            "Estado": "En progreso",
            "Evidencia": (
                f"{artifact_count} artefactos versionados entre ML, modelado, "
                "evaluacion e interpretacion."
            ),
            "Siguiente paso": "Cerrar la historia con aporte, riesgos, limites y trabajo futuro.",
        },
    ]
    memory = pd.DataFrame(rows)
    memory["Predictoras documentadas"] = len(predictors)
    return memory


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
                "Experimentos",
                "Diseno experimental",
                "Memoria de investigacion",
                "Datasets por capas",
                "Dataset maestro",
                "Notebooks",
            ],
        )
        st.divider()
        evaluation_summary = summarize_evaluation_experiments()
        st.caption("Experimentos evaluados")
        st.write(str(len(evaluation_summary)) if not evaluation_summary.empty else "1")
        st.caption("Base historica")
        st.write(data.meta.get("Experimento", "Exp01"))
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
    evaluation_summary = summarize_evaluation_experiments()
    interpretation_summary = summarize_interpretation_experiments()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Capas sistemicas", f"{int(layers['Disponible'].sum())}/{len(layers)}")
    c2.metric("Dataset maestro", f"{len(data.master):,} filas")
    c3.metric("Variables maestro", f"{data.master.shape[1]:,}")
    c4.metric("Experimentos evaluados", f"{len(evaluation_summary):,}")

    tab_map, tab_pred, tab_layers, tab_quality = st.tabs(
        ["Mapa del sistema", "Predicciones", "Capas", "Calidad de datos"]
    )
    with tab_map:
        render_system_map()
        st.dataframe(
            layers[["Capa", "Rol sistemico", "Filas", "Columnas", "Nulos"]],
            use_container_width=True,
            hide_index=True,
        )

    with tab_pred:
        experiments = _experiment_names(evaluation_summary)
        prediction_frames = [_prediction_view(experiment) for experiment in experiments]
        prediction_frames = [frame for frame in prediction_frames if not frame.empty]
        if not prediction_frames and data.predictions.empty:
            render_missing_file(PREDICTIONS_PATH)
        else:
            view = pd.concat(prediction_frames, ignore_index=True) if prediction_frames else data.predictions.copy()
            fig = px.line(
                view,
                x="Registro",
                y="Prediccion",
                color="Experimento" if "Experimento" in view.columns else None,
                title="Predicciones por experimento",
            )
            fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig, use_container_width=True)
            if not interpretation_summary.empty:
                metrics = _numeric_metric_frame(interpretation_summary)
                if not metrics.empty:
                    fig = px.bar(
                        metrics,
                        x="Experimento",
                        y="Valor",
                        color="Metrica",
                        barmode="group",
                        title="Metricas consolidadas de interpretacion",
                    )
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


def render_experiments(data: ProjectData) -> None:
    """Render a multi-experiment analysis center.

    Args:
        data: Loaded project datasets and metadata.

    Returns:
        None.
    """

    st.subheader("Centro de experimentos")
    ml_summary = summarize_ml_experiments()
    modeling_summary = summarize_modeling_experiments()
    evaluation_summary = summarize_evaluation_experiments()
    interpretation_summary = summarize_interpretation_experiments()
    experiments = _experiment_names(ml_summary, modeling_summary, evaluation_summary, interpretation_summary)

    if not experiments:
        render_experiment(data)
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Experimentos", f"{len(experiments):,}")
    c2.metric("Con predicciones", f"{len(evaluation_summary):,}")
    c3.metric("Con modelado", f"{len(modeling_summary):,}")
    c4.metric("Interpretados", f"{len(interpretation_summary):,}")

    selected_experiment = st.selectbox("Experimento", experiments, key="experiment_center_selected")
    (
        tab_compare,
        tab_detail,
        tab_predictions,
        tab_modeling,
        tab_model_cards,
        tab_interpretation,
        tab_artifacts,
    ) = st.tabs(
        [
            "Comparativo",
            "Detalle",
            "Predicciones",
            "Modelado",
            "Model cards",
            "Interpretacion",
            "Artefactos",
        ]
    )

    with tab_compare:
        left, right = st.columns([1, 1])
        with left:
            if evaluation_summary.empty:
                render_missing_file(EVALUATIONS_DIR)
            else:
                fig = px.bar(
                    evaluation_summary,
                    x="Experimento",
                    y="Predicciones",
                    color="Variable_Objetivo",
                    title="Predicciones generadas por experimento",
                )
                fig.update_layout(height=420, margin=dict(l=10, r=10, t=55, b=10))
                st.plotly_chart(fig, use_container_width=True)
        with right:
            metrics = _numeric_metric_frame(interpretation_summary)
            if metrics.empty:
                render_missing_file(INTERPRETATION_DIR)
            else:
                fig = px.bar(
                    metrics,
                    x="Experimento",
                    y="Valor",
                    color="Metrica",
                    barmode="group",
                    title="Metricas comparables C16",
                )
                fig.update_layout(height=420, margin=dict(l=10, r=10, t=55, b=10))
                st.plotly_chart(fig, use_container_width=True)

        summary_tabs = st.tabs(["Preparacion ML", "Modelado", "Evaluacion", "Interpretacion"])
        with summary_tabs[0]:
            st.caption(rel(MACHINE_LEARNING_DIR / "Transformaciones"))
            st.dataframe(ml_summary, use_container_width=True, hide_index=True)
        with summary_tabs[1]:
            st.caption(rel(MODELING_DIR))
            st.dataframe(modeling_summary, use_container_width=True, hide_index=True)
        with summary_tabs[2]:
            st.caption(rel(EVALUATIONS_DIR))
            st.dataframe(evaluation_summary, use_container_width=True, hide_index=True)
        with summary_tabs[3]:
            st.caption(rel(INTERPRETATION_DIR))
            st.dataframe(interpretation_summary, use_container_width=True, hide_index=True)

    with tab_detail:
        metadata = load_prediction_metadata(selected_experiment)
        sequence_metadata = load_sequence_metadata(selected_experiment)
        ml_dataset = load_transformed_dataset(selected_experiment)
        model_record = load_model_record(selected_experiment)
        predictions = load_predictions(selected_experiment)
        target = metadata.get("Variable Objetivo", sequence_metadata.get("Variable Objetivo", "-"))

        d1, d2, d3, d4 = st.columns(4)
        d1.metric("Variable objetivo", target)
        d2.metric("Ventana", metadata.get("Ventana", sequence_metadata.get("Ventana", "-")))
        d3.metric("Filas ML", f"{len(ml_dataset):,}")
        d4.metric("Predicciones", f"{len(predictions):,}")

        left, right = st.columns([1, 1])
        with left:
            st.markdown("**Metadata de prediccion**")
            metadata_table = pd.DataFrame(
                [{"Parametro": key, "Valor": value} for key, value in metadata.items()]
            )
            st.dataframe(metadata_table, use_container_width=True, hide_index=True)
        with right:
            st.markdown("**Registro de modelado**")
            if model_record.empty:
                render_missing_file(MODELING_DIR / "Modelos" / selected_experiment)
            else:
                st.dataframe(model_record, use_container_width=True, hide_index=True)

        if not ml_dataset.empty:
            with st.expander("Dataset transformado C13", expanded=False):
                render_dataset_metrics(ml_dataset)
                st.dataframe(ml_dataset.head(300), use_container_width=True, hide_index=True)

    with tab_predictions:
        prediction_view = _prediction_view(selected_experiment)
        if prediction_view.empty:
            render_missing_file(EVALUATIONS_DIR / selected_experiment / "predicciones.csv")
        else:
            p1, p2, p3, p4 = st.columns(4)
            distribution = prediction_distribution(prediction_view)
            row = distribution.iloc[0].to_dict() if not distribution.empty else {}
            p1.metric("Conteo", f"{int(row.get('Conteo', len(prediction_view))):,}")
            p2.metric("Media", f"{row.get('Media', 0):.3f}")
            p3.metric("Minimo", f"{row.get('Minimo', 0):.3f}")
            p4.metric("Maximo", f"{row.get('Maximo', 0):.3f}")

            left, right = st.columns([2, 1])
            with left:
                fig = px.line(
                    prediction_view,
                    x="Registro",
                    y=["Prediccion", "Tendencia"],
                    title=f"Serie de prediccion {selected_experiment}",
                )
                fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
                st.plotly_chart(fig, use_container_width=True)
            with right:
                prediction_view["Categoria"] = prediction_view["Intensidad"].apply(quality_badge)
                counts = prediction_view["Categoria"].value_counts().reset_index()
                counts.columns = ["Categoria", "Registros"]
                fig = px.pie(
                    counts,
                    names="Categoria",
                    values="Registros",
                    hole=0.45,
                    title="Intensidad relativa",
                )
                fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
                st.plotly_chart(fig, use_container_width=True)

            st.dataframe(prediction_view, use_container_width=True, hide_index=True)

    with tab_modeling:
        diagnostic = load_model_diagnostic(selected_experiment)
        recommendations = load_model_recommendations(selected_experiment)
        if diagnostic.empty:
            render_missing_file(MODELING_DIR / "Diagnosticos" / selected_experiment)
        else:
            metric_values = diagnostic.copy()
            metric_values["Valor_Numerico"] = pd.to_numeric(metric_values["Valor"], errors="coerce")
            chart_values = metric_values.dropna(subset=["Valor_Numerico"])
            if not chart_values.empty:
                fig = px.bar(
                    chart_values,
                    x="Indicador",
                    y="Valor_Numerico",
                    color="Estado" if "Estado" in chart_values.columns else None,
                    title=f"Diagnostico del modelo {selected_experiment}",
                    color_discrete_map={
                        "Excelente": "#2A9D8F",
                        "Aceptable": "#E9C46A",
                        "Baja": "#F4A261",
                        "Critico": "#E76F51",
                        "Crítico": "#E76F51",
                    },
                )
                fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(diagnostic, use_container_width=True, hide_index=True)

        if not recommendations.empty:
            st.markdown("**Recomendaciones**")
            st.dataframe(recommendations, use_container_width=True, hide_index=True)

        metric_images = sorted((MODELING_DIR / "Metricas" / selected_experiment).glob("*.png"))
        if metric_images:
            cols = st.columns(min(3, len(metric_images)))
            for index, image in enumerate(metric_images):
                with cols[index % len(cols)]:
                    st.image(str(image), caption=image.name, use_container_width=True)

    with tab_model_cards:
        cards = _model_card_inventory()
        if cards.empty:
            render_missing_file(MODEL_CARDS_DIR)
        else:
            normalized_experiment = selected_experiment.replace("_", "-")
            options = cards["Archivo"].tolist()
            matched = cards[cards["Experimento"].astype(str) == normalized_experiment]
            default_index = 0
            if not matched.empty:
                default_file = matched["Archivo"].iloc[0]
                default_index = options.index(default_file)

            selected_card = st.selectbox(
                "Model card",
                options,
                index=default_index,
                key=f"model_card_{selected_experiment}",
            )
            card_path = MODEL_CARDS_DIR / selected_card
            c1, c2, c3 = st.columns(3)
            card_row = cards[cards["Archivo"] == selected_card].iloc[0]
            c1.metric("Experimento", card_row["Experimento"])
            c2.metric("Archivo", selected_card)
            c3.metric("Tamano", f"{float(card_row['Tamano_MB']):.2f} MB")
            st.caption(rel(card_path))
            _render_pdf_preview(card_path)
            st.download_button(
                "Descargar model card",
                data=card_path.read_bytes(),
                file_name=selected_card,
                mime="application/pdf",
                key=f"download_model_card_{selected_card}",
            )
            st.dataframe(cards, use_container_width=True, hide_index=True)

    with tab_interpretation:
        summary = load_interpretation_summary(selected_experiment)
        interpreted = interpretation_summary[
            interpretation_summary.get("Experimento", pd.Series(dtype=str)).astype(str) == selected_experiment
        ]
        if summary.empty and interpreted.empty:
            render_missing_file(INTERPRETATION_DIR / selected_experiment / "resumen_experimento.csv")
        else:
            view = interpreted if not interpreted.empty else summary
            st.dataframe(view, use_container_width=True, hide_index=True)
            if "Interpretacion_Tecnica" in view.columns:
                st.info(str(view["Interpretacion_Tecnica"].iloc[0]))

            sequence_metadata = load_sequence_metadata(selected_experiment)
            variables = str(sequence_metadata.get("Variables Predictoras", "")).split(";")
            variables = [variable.strip() for variable in variables if variable.strip()]
            coverage = dimension_coverage(variables)
            if not coverage.empty:
                fig = px.bar(
                    coverage,
                    x="Dimension",
                    y="Variables",
                    color="Dimension",
                    title="Cobertura sistemica de variables predictoras",
                )
                fig.update_layout(height=420, margin=dict(l=10, r=10, t=55, b=10))
                st.plotly_chart(fig, use_container_width=True)
                st.dataframe(coverage, use_container_width=True, hide_index=True)

    with tab_artifacts:
        inventory = _artifact_inventory()
        if inventory.empty:
            render_missing_file(BASE_DIR / "DATA")
        else:
            filtered = inventory[inventory["Experimento"].astype(str) == selected_experiment]
            if filtered.empty:
                filtered = inventory
            counts = filtered.groupby(["Etapa", "Formato"]).size().reset_index(name="Archivos")
            fig = px.bar(
                counts,
                x="Etapa",
                y="Archivos",
                color="Formato",
                barmode="group",
                title="Artefactos disponibles por etapa",
            )
            fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.dataframe(filtered, use_container_width=True, hide_index=True)


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
            fig = px.area(
                view,
                x="Registro",
                y="Normalizada",
                color="Categoria",
                title="Intensidad relativa de prediccion",
            )
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


def render_experiment_design(data: ProjectData) -> None:
    """Render the experimental design view.

    Args:
        data: Loaded project datasets and metadata.

    Returns:
        None.
    """

    st.subheader("Diseno experimental")
    readme = read_text(EXPERIMENT_DESIGN_DIR / "README.md")
    if readme:
        st.markdown(readme)

    catalog = data.experiment_design.get("Catalogo de experimentos", pd.DataFrame())
    config = data.experiment_design.get("Configuracion", pd.DataFrame())
    predictors = data.experiment_design.get("Variables predictoras", pd.DataFrame())
    status = data.experiment_design.get("Estado de experimentos", pd.DataFrame())

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Experimentos", f"{len(catalog):,}")
    c2.metric("Ejecutados", f"{(catalog.get('Estado', pd.Series(dtype=str)) == 'Ejecutado').sum():,}")
    c3.metric("Pendientes", f"{(catalog.get('Estado', pd.Series(dtype=str)) == 'Pendiente').sum():,}")
    c4.metric("Predictoras", f"{len(predictors):,}")

    tab_map, tab_config, tab_diagnostic, tab_files = st.tabs(
        ["Mapa experimental", "Configuracion", "Diagnostico Exp01", "Archivos"]
    )

    with tab_map:
        if catalog.empty:
            render_missing_file(EXPERIMENT_DESIGN_FILES["Catalogo de experimentos"])
        else:
            left, right = st.columns([1, 2])
            with left:
                selected = st.selectbox("Experimento", catalog["Experimento"].tolist())
                experiment = catalog[catalog["Experimento"] == selected].iloc[0]
                st.metric("Objetivo", experiment.get("Variable_Objetivo", "-"))
                st.metric("Tipo", experiment.get("Tipo_Problema", "-"))
                st.metric("Estado", experiment.get("Estado", "-"))
                st.caption(str(experiment.get("Pregunta_Investigacion", "")))
            with right:
                counts = catalog.groupby(["Tipo_Problema", "Estado"]).size().reset_index(name="Experimentos")
                fig = px.bar(
                    counts,
                    x="Tipo_Problema",
                    y="Experimentos",
                    color="Estado",
                    barmode="group",
                    title="Plan experimental por tipo de problema",
                    color_discrete_map={"Ejecutado": "#2A9D8F", "Pendiente": "#E9C46A"},
                )
                fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(catalog, use_container_width=True, hide_index=True)

    with tab_config:
        config_tab, predictors_tab, criteria_tab, status_tab = st.tabs(
            ["Parametros", "Predictoras", "Criterios", "Estado"]
        )
        with config_tab:
            st.dataframe(config, use_container_width=True, hide_index=True)
        with predictors_tab:
            st.dataframe(predictors, use_container_width=True, hide_index=True)
            if not predictors.empty and "Variable" in predictors.columns:
                st.write(", ".join(predictors["Variable"].astype(str).tolist()))
        with criteria_tab:
            for label in ["Criterios clasificacion", "Criterios regresion"]:
                criteria = data.experiment_design.get(label, pd.DataFrame())
                with st.expander(label, expanded=label == "Criterios clasificacion"):
                    st.dataframe(criteria, use_container_width=True, hide_index=True)
        with status_tab:
            st.dataframe(status, use_container_width=True, hide_index=True)

    with tab_diagnostic:
        diagnostic = data.model_diagnostic.copy()
        recommendations = data.model_recommendations.copy()
        if diagnostic.empty:
            render_missing_file(MODEL_DIAGNOSTIC_PATH)
        else:
            st.markdown("**Resultado del modelo Exp01**")
            metric_values = diagnostic.copy()
            metric_values["Valor_Numerico"] = pd.to_numeric(metric_values["Valor"], errors="coerce")
            chart_values = metric_values.dropna(subset=["Valor_Numerico"])
            if not chart_values.empty:
                fig = px.bar(
                    chart_values,
                    x="Indicador",
                    y="Valor_Numerico",
                    color="Estado",
                    title="Metricas de clasificacion Exp01",
                    color_discrete_map={
                        "Excelente": "#2A9D8F",
                        "Aceptable": "#E9C46A",
                        "Baja": "#F4A261",
                        "Critico": "#E76F51",
                    },
                )
                fig.update_layout(height=430, margin=dict(l=10, r=10, t=55, b=10))
                st.plotly_chart(fig, use_container_width=True)
            st.dataframe(diagnostic, use_container_width=True, hide_index=True)

        if not recommendations.empty and "Recomendacion" in recommendations.columns:
            st.markdown("**Recomendaciones metodologicas**")
            for recommendation in recommendations["Recomendacion"].dropna().astype(str):
                st.write(recommendation)

        images = [path for path in [MODEL_ACCURACY_IMAGE_PATH, MODEL_LOSS_IMAGE_PATH] if path.exists()]
        if images:
            cols = st.columns(len(images))
            for column, image in zip(cols, images):
                with column:
                    st.image(str(image), caption=image.name, use_container_width=True)

    with tab_files:
        selected_label = st.selectbox("Dataset de diseno", list(EXPERIMENT_DESIGN_FILES))
        selected_path = EXPERIMENT_DESIGN_FILES[selected_label]
        dataset = data.experiment_design.get(selected_label, pd.DataFrame())
        st.caption(rel(selected_path))
        if dataset.empty:
            render_missing_file(selected_path)
        else:
            render_dataset_metrics(dataset)
            st.dataframe(dataset, use_container_width=True, hide_index=True)
            st.download_button(
                "Descargar dataset",
                data=dataset.to_csv(index=False).encode("utf-8"),
                file_name=selected_path.name,
                mime="text/csv",
                key=f"download_design_{selected_label}",
            )


def render_research_memory(data: ProjectData) -> None:
    """Render research memory and thesis-defense evidence.

    Args:
        data: Loaded project datasets and metadata.

    Returns:
        None.
    """

    st.subheader("Memoria de investigacion")
    memory = _research_memory(data)
    catalog = data.experiment_design.get("Catalogo de experimentos", pd.DataFrame())
    model_cards = _model_card_inventory()
    consolidated = int((memory["Estado"] == "Consolidado").sum())
    experiments = len(summarize_evaluation_experiments())
    layers = layer_summary()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Bloques documentados", f"{len(memory):,}")
    c2.metric("Bloques consolidados", f"{consolidated:,}")
    c3.metric("Experimentos evaluados", f"{experiments:,}")
    c4.metric("Capas disponibles", f"{int(layers['Disponible'].sum())}/{len(layers)}")

    tab_context, tab_method, tab_evidence, tab_closure = st.tabs(
        ["Planteamiento", "Metodologia", "Evidencias", "Cierre"]
    )

    with tab_context:
        st.markdown("**Problema de investigacion**")
        st.write(
            "El proyecto organiza informacion ambiental, hidrica, social e "
            "institucional para analizar y predecir variables asociadas a la "
            "gestion hidrica del Rio Bogota."
        )
        st.markdown("**Objetivo del tablero**")
        st.write(
            "Mostrar la trazabilidad entre datos, capas, diseno experimental, "
            "modelos, resultados e interpretacion, sin mezclar la aplicacion "
            "con la evaluacion academica."
        )
        st.dataframe(
            memory[["Bloque", "Estado", "Evidencia"]],
            use_container_width=True,
            hide_index=True,
        )

    with tab_method:
        st.markdown("**Flujo metodologico**")
        flow = pd.DataFrame(
            [
                {
                    "Etapa": "1. Datos raw",
                    "Salida": "Fuentes por capa en DATA/RAW",
                    "Vista": "Datasets por capas",
                },
                {
                    "Etapa": "2. Capas framework",
                    "Salida": "C01-C07 con metadata, auditoria e indicadores",
                    "Vista": "Datasets por capas",
                },
                {
                    "Etapa": "3. Dataset maestro",
                    "Salida": "Base integrada e imputada para modelado",
                    "Vista": "Dataset maestro",
                },
                {
                    "Etapa": "4. Diseno experimental",
                    "Salida": "Catalogo, variables, criterios y estado",
                    "Vista": "Diseno experimental",
                },
                {
                    "Etapa": "5. Modelado y evaluacion",
                    "Salida": "Predicciones, diagnosticos y model cards",
                    "Vista": "Experimentos",
                },
                {
                    "Etapa": "6. Interpretacion",
                    "Salida": "Lectura sistemica y limites del resultado",
                    "Vista": "Experimentos",
                },
            ]
        )
        st.dataframe(flow, use_container_width=True, hide_index=True)

        if not catalog.empty:
            st.markdown("**Experimentos como decisiones metodologicas**")
            visible_columns = [
                column
                for column in [
                    "Experimento",
                    "Variable_Objetivo",
                    "Tipo_Problema",
                    "Estado",
                    "Pregunta_Investigacion",
                ]
                if column in catalog.columns
            ]
            st.dataframe(catalog[visible_columns], use_container_width=True, hide_index=True)

    with tab_evidence:
        st.markdown("**Evidencias conectadas a la investigacion**")
        evidence_rows = [
            {
                "Bloque": "Datos multicapa",
                "Ruta": "DATA/MASTER",
                "Uso en la app": (
                    "Demostrar integracion de clima, hidrologia, calidad, "
                    "ONI, hidraulica, percepcion y gobernanza."
                ),
            },
            {
                "Bloque": "Diseno experimental",
                "Ruta": "DATA/DISENO_EXPERIMENTAL",
                "Uso en la app": (
                    "Mostrar preguntas, variables objetivo, configuraciones "
                    "y estado de experimentos."
                ),
            },
            {
                "Bloque": "Pipeline modular",
                "Ruta": "src/framework_v7/pipeline",
                "Uso en la app": (
                    "Sustentar reproducibilidad, PEP8, funciones "
                    "reutilizables y separacion notebook/app."
                ),
            },
            {
                "Bloque": "Resultados",
                "Ruta": "DATA/EVALUACIONES",
                "Uso en la app": "Comparar salidas predictivas y metricas entre experimentos.",
            },
            {
                "Bloque": "Model cards",
                "Ruta": "FRAMEWORK_STREAMLIT/model_cards",
                "Uso en la app": (
                    "Presentar proposito, datos, desempeno, limitaciones y "
                    "uso responsable de cada modelo."
                ),
            },
            {
                "Bloque": "Interpretacion",
                "Ruta": "DATA/INTERPRETACION_RESULTADOS",
                "Uso en la app": "Traducir metricas a lectura sistemica y decision hidrica.",
            },
            {
                "Bloque": "Memoria metodologica",
                "Ruta": "NOTEBOOKS",
                "Uso en la app": "Conservar trazabilidad del proceso exploratorio y experimental.",
            },
        ]
        st.dataframe(pd.DataFrame(evidence_rows), use_container_width=True, hide_index=True)

        if not model_cards.empty:
            st.markdown("**Model cards disponibles**")
            st.dataframe(model_cards, use_container_width=True, hide_index=True)

    with tab_closure:
        st.markdown("**Narrativa para presentar el proyecto**")
        story = pd.DataFrame(
            [
                {
                    "Momento": "1. Problema",
                    "Mensaje": (
                        "La gestion hidrica necesita integrar variables "
                        "biofisicas, sociales e institucionales."
                    ),
                    "Vista": "Dashboard",
                },
                {
                    "Momento": "2. Sistema multicapa",
                    "Mensaje": "El framework organiza el problema en capas conectadas y auditables.",
                    "Vista": "Datasets por capas",
                },
                {
                    "Momento": "3. Dataset maestro",
                    "Mensaje": "Las capas se consolidan en una base temporal lista para modelado.",
                    "Vista": "Dataset maestro",
                },
                {
                    "Momento": "4. Experimentos",
                    "Mensaje": "Los experimentos comparan objetivos, horizontes y artefactos predictivos.",
                    "Vista": "Experimentos",
                },
                {
                    "Momento": "5. Interpretacion",
                    "Mensaje": "Las metricas se convierten en lectura sistemica, limites y acciones futuras.",
                    "Vista": "Memoria de investigacion",
                },
            ]
        )
        st.dataframe(story, use_container_width=True, hide_index=True)

        left, right = st.columns(2)
        with left:
            st.markdown("**Preguntas que debe responder la app**")
            questions = [
                "Que se esta prediciendo y por que importa para el Rio Bogota.",
                "Que capas alimentan el modelo y cual es su rol sistemico.",
                "Como se construyo, limpio e integro el dataset maestro.",
                "Que diferencia hay entre los experimentos ejecutados.",
                "Cuales son las limitaciones y el siguiente experimento necesario.",
            ]
            for question in questions:
                st.write(question)
        with right:
            st.markdown("**Aporte diferencial**")
            contributions = [
                "Framework multicapa con datos tecnicos, percepcion y gobernanza.",
                "Repositorio reproducible con notebooks como memoria y modulos en `src`.",
                "Tablero de resultados para evaluar datos, modelos e interpretacion.",
                "Diseno experimental versionado para ampliar el trabajo.",
            ]
            for contribution in contributions:
                st.write(contribution)

        st.markdown("**Pendientes metodologicos**")
        checklist = pd.DataFrame(
            [
                {"Item": "Explicar objetivo general y objetivos especificos", "Estado": "Listo"},
                {"Item": "Mostrar trazabilidad de datos por capa", "Estado": "Listo"},
                {"Item": "Defender el diseno de los 3 experimentos ejecutados", "Estado": "Listo"},
                {"Item": "Agregar comparacion contra baseline simple", "Estado": "Por reforzar"},
                {"Item": "Cerrar con limitaciones, riesgos y trabajo futuro", "Estado": "Por reforzar"},
            ]
        )
        st.dataframe(checklist, use_container_width=True, hide_index=True)


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
            start_date = dates.min()
            end_date = dates.max()
            c1, c2 = st.columns(2)
            c1.metric("Inicio", format_metric_date(start_date))
            c2.metric("Fin", format_metric_date(end_date))
    with tab_series:
        render_time_series(dataset, "master")
    with tab_groups:
        available_groups = {
            group: [col for col in cols if col in dataset.columns]
            for group, cols in FEATURE_GROUPS.items()
        }
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
        "Esta carpeta conserva la exploracion y el paso a paso. La app y los "
        "datos quedan separados para que el repositorio funcione como producto reproducible."
    )
    notebooks = sorted(NOTEBOOKS_DIR.rglob("*.ipynb"))
    rows = []
    for notebook in notebooks:
        rows.append(
            {
                "Notebook": notebook.name,
                "Carpeta": rel(notebook.parent),
                "Tamano KB": round(notebook.stat().st_size / 1024, 1),
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    md_files = sorted(NOTEBOOKS_DIR.rglob("*.md"))
    if md_files:
        selected_doc = st.selectbox("Documento", [rel(path) for path in md_files])
        st.markdown(read_text(BASE_DIR / selected_doc))
