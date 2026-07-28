# FRAMEWORK V7

Framework para la gestion hidrica del Rio Bogota con capas biofisicas, hidraulicas, de percepcion y gobernanza.

## Estructura

- `DATA/RAW`: fuentes originales.
- `DATA/MASTER`: datasets consolidados por capa y datasets maestros.
- `DATA/MACHINE_LEARNING`: transformaciones, diagnosticos y secuencias para modelado.
- `DATA/EVALUACIONES`: resultados del experimento predictivo.
- `NOTEBOOKS`: memoria metodologica y notebooks del flujo.
- `app.py`: visor Streamlit del experimento y de las capas.

## App Streamlit

La aplicacion muestra:

- dashboard ejecutivo del sistema multicapa;
- mapa visual de flujo desde capas hasta modelo;
- predicciones de `Exp01`;
- dataset transformado para machine learning;
- tabs por cada capa del framework;
- datasets maestros y cobertura de variables;
- indice de notebooks.

## Ejecucion local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Archivos principales

- `DATA/EVALUACIONES/Exp01/predicciones.csv`
- `DATA/EVALUACIONES/Exp01/metadata_prediccion.csv`
- `DATA/MACHINE_LEARNING/C13_MACHINE_LEARNING/Transformaciones/Exp01/dataset_machine_learning_transformado.csv`
- `DATA/MASTER/C09_MASTER/Dataset_Maestro_Framework_v03_Con_Imputaciones.csv`

## Objetivo del experimento

La metadata actual define `irca` como variable objetivo. La app interpreta este primer experimento como una primera salida predictiva para evaluar el comportamiento del indicador de riesgo/calidad dentro del enfoque multicapa del framework.
