"""Command-line execution helpers for the notebook pipeline."""

from __future__ import annotations

import pandas as pd

from framework_v7.data_access import ProjectData, load_project_data

from . import experiment_design, machine_learning, ml_preparation


def run_pipeline_summary(data: ProjectData | None = None) -> pd.DataFrame:
    """Execute a lightweight modular validation of notebook artifacts.

    Args:
        data: Optional preloaded project data.

    Returns:
        Summary table by notebook stage.
    """

    data = data or load_project_data()
    rows = []

    design_summary = experiment_design.design_summary(data.experiment_design)
    rows.append(
        {
            "Etapa": "DISENO_EXPERIMENTAL",
            "Modulo": "experiment_design.py",
            "Resultado": f"{int(design_summary['Disponible'].sum())}/{len(design_summary)} artefactos",
        }
    )

    target = data.meta.get("Variable Objetivo", "irca")
    if not data.master.empty:
        rows.append(
            {
                "Etapa": "C08-C09",
                "Modulo": "integration.py / feature_engineering.py",
                "Resultado": f"{data.master.shape[0]} filas x {data.master.shape[1]} columnas maestro",
            }
        )

    if not data.ml_dataset.empty and target in data.ml_dataset.columns:
        diagnostic = ml_preparation.modeling_dataset_diagnostic(data.ml_dataset, target)
        rows.append(
            {
                "Etapa": "C12",
                "Modulo": "ml_preparation.py",
                "Resultado": (
                    f"{int(diagnostic.loc[diagnostic['Indicador'] == 'predictoras_numericas', 'Valor'].iloc[0])} "
                    "predictoras numericas"
                ),
            }
        )
        transform = machine_learning.transformation_diagnostic(data.ml_dataset)
        rows.append(
            {
                "Etapa": "C13",
                "Modulo": "machine_learning.py",
                "Resultado": (
                    f"{int(transform.loc[transform['Indicador'] == 'variables_constantes', 'Valor'].iloc[0])} "
                    "variables constantes"
                ),
            }
        )

    if not data.model_diagnostic.empty:
        rows.append(
            {
                "Etapa": "C14-C15",
                "Modulo": "modeling.py / evaluation.py",
                "Resultado": f"{len(data.model_diagnostic)} indicadores de diagnostico Exp01",
            }
        )

    return pd.DataFrame(rows)


def print_pipeline_summary(summary: pd.DataFrame) -> None:
    """Print pipeline summary rows to stdout.

    Args:
        summary: Output from ``run_pipeline_summary``.

    Returns:
        None.
    """

    print("PIPELINE MODULAR DE NOTEBOOKS")
    print("=" * 72)
    for row in summary.to_dict("records"):
        print(f"{row['Etapa']}: {row['Modulo']} -> {row['Resultado']}")


def main() -> None:
    """Run the notebook pipeline validation from the command line."""

    print_pipeline_summary(run_pipeline_summary())


if __name__ == "__main__":
    main()
