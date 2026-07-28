"""Business anatomy and dataset catalog for FRAMEWORK V7.

The project is organized around system layers. Each layer has a business role,
an input folder, a main dataset and support artifacts used by the dashboard.
This module is intentionally declarative: it describes the domain structure
without loading data or rendering UI.
"""

from __future__ import annotations

from .paths import COVERAGE_PATH, MASTER_DIR, MASTER_PATH


LAYER_CATALOG = {
    "C01 - Climatica": {
        "folder": MASTER_DIR / "C01_MASTER",
        "main": "02_Capa_Climatica_Framework.xlsx",
        "role": "Forzantes climaticos locales que condicionan el sistema hidrico.",
        "color": "#2A9D8F",
    },
    "C02 - Hidrologica": {
        "folder": MASTER_DIR / "C02_MASTER",
        "main": "02_Capa_Hidrologica_Framework.xlsx",
        "role": "Respuesta fisica del rio y comportamiento de niveles hidricos.",
        "color": "#457B9D",
    },
    "C03 - Calidad de agua": {
        "folder": MASTER_DIR / "C03_MASTER",
        "main": "02_Capa_Calidad_Agua_Framework.xlsx",
        "role": "Estado fisicoquimico y sanitario del recurso hidrico.",
        "color": "#E76F51",
    },
    "C04 - ONI": {
        "folder": MASTER_DIR / "C04_MASTER",
        "main": "02_Capa_ONI_Framework.xlsx",
        "role": "Senal macroclimatica asociada a variabilidad ENSO.",
        "color": "#F4A261",
    },
    "C05 - Hidraulica": {
        "folder": MASTER_DIR / "C05_MASTER",
        "main": "02_Capa_Hidraulica_Framework.xlsx",
        "role": "Operacion y disponibilidad hidraulica del sistema.",
        "color": "#264653",
    },
    "C06 - Percepcion": {
        "folder": MASTER_DIR / "C06_MASTER",
        "main": "02_Capa_Percepcion_Framework.xlsx",
        "role": "Lectura social del problema: riesgo, confianza y preocupacion ciudadana.",
        "color": "#8E7DBE",
    },
    "C07 - Gobernanza": {
        "folder": MASTER_DIR / "CO7_MASTER",
        "main": "02_Capa_Gobernanza_Framework.xlsx",
        "role": "Capacidad institucional, control, cobertura y gestion publica.",
        "color": "#6A994E",
    },
}

SUPPORT_FILES = {
    "Metadata": "03_Metadata.xlsx",
    "Diccionario": "04_Diccionario_Datos.xlsx",
    "Auditoria": "05_Auditoria.xlsx",
    "Indicadores": "06_Indicadores.xlsx",
    "EDA": "07_EDA.xlsx",
}

MASTER_FILES = {
    "Maestro con imputaciones": MASTER_PATH,
    "Maestro v04": MASTER_DIR / "C09_MASTER" / "Dataset_Maestro_Framework_v04.csv",
    "Maestro IRCA": MASTER_DIR / "C09_MASTER" / "Dataset_Maestro_Framework_v02_Irca.csv",
    "Volumen": MASTER_DIR / "C09_MASTER" / "Dataset_Maestro_Framework_v03_Volumen.csv",
    "Cobertura variables": COVERAGE_PATH,
}

FEATURE_GROUPS = {
    "Clima": [
        "Precipitacion_mm",
        "Temp_Min_C",
        "Temp_Max_C",
        "Temp_Media_C",
        "Radiacion_Solar",
        "Humedad_Relativa",
        "Velocidad_Viento",
        "ONI",
    ],
    "Hidraulica": ["VolumenUtilDiarioMasa", "Nivel_Minimo"],
    "Calidad": [
        "irca",
        "pH",
        "TURBIDEZ",
        "OXIGENO DISUELTO (OD)",
        "DEMANDA BIOQUIMICA DE OXIGENO (DBO5)",
        "DEMANDA QUIMICA DE OXIGENO (DQO)",
    ],
    "Gobernanza": [
        "INDICE DE DESEMPENO INSTITUCIONAL",
        "ACCESO A AGUA POTABLE ADECUADO",
        "COBERTURA DE ACUEDUCTO URBANO",
        "COBERTURA DE ALCANTARILLADO RURAL",
    ],
}
