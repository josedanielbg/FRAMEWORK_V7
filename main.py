from __future__ import annotations

"""Command-line summary for FRAMEWORK V7.

This file is a lightweight execution entrypoint complementary to ``app.py``.
Use it when you need a quick repository health check without launching
Streamlit.
"""

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from framework_v7.data_access import load_project_data
from framework_v7.layers import LAYER_MODULES
from framework_v7.pipeline import PIPELINE_MODULES
from framework_v7.pipeline.main import run_pipeline_summary
from framework_v7.profiling import layer_summary


def main() -> None:
    """Print a concise project summary.

    Returns:
        None.
    """

    data = load_project_data()
    layers = layer_summary()
    print("FRAMEWORK V7")
    print(f"Experimento: {data.meta.get('Experimento', 'Exp01')}")
    print(f"Variable objetivo: {data.meta.get('Variable Objetivo', 'irca')}")
    print(f"Predicciones: {len(data.predictions)}")
    print(f"Dataset maestro: {data.master.shape[0]} filas x {data.master.shape[1]} columnas")
    print(f"Capas disponibles: {int(layers['Disponible'].sum())}/{len(layers)}")
    print(f"Modulos de capa: {len(LAYER_MODULES)}")
    design_catalog = data.experiment_design.get("Catalogo de experimentos")
    if design_catalog is not None:
        print(f"Experimentos disenados: {len(design_catalog)}")
    if not data.model_diagnostic.empty:
        print(f"Diagnostico modelo Exp01: {len(data.model_diagnostic)} indicadores")
    print(f"Modulos de pipeline notebook: {len(PIPELINE_MODULES)}")
    print()
    pipeline_summary = run_pipeline_summary(data)
    for row in pipeline_summary.to_dict("records"):
        print(f"{row['Etapa']}: {row['Resultado']}")


if __name__ == "__main__":
    main()
