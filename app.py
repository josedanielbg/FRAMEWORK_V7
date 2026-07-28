from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from framework_v7.data_access import load_project_data
from framework_v7.views import (
    render_dashboard,
    render_experiment,
    render_layers,
    render_master_dataset,
    render_notebooks,
    render_sidebar,
)


st.set_page_config(
    page_title="FRAMEWORK V7 | Experimento 01",
    page_icon="DATA/MASTER/C01_MASTER/Correlacion.png",
    layout="wide",
)

data = load_project_data()
section = render_sidebar(data)

st.title("Framework de gestion hidrica - V7")
st.caption("Tablero multicapa para explorar datos, resultados de Exp01 y memoria metodologica.")

if section == "Dashboard":
    render_dashboard(data)
elif section == "Experimento 01":
    render_experiment(data)
elif section == "Datasets por capas":
    render_layers()
elif section == "Dataset maestro":
    render_master_dataset()
elif section == "Notebooks":
    render_notebooks()
