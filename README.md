# FRAMEWORK V7

Framework para la gestion hidrica del Rio Bogota con capas biofisicas, hidraulicas, de percepcion y gobernanza.

## App Streamlit

La aplicacion `app.py` muestra los resultados del primer experimento (`Exp01`):

- metadata de la prediccion;
- serie y distribucion de predicciones;
- dataset transformado para machine learning;
- diagnostico estadistico;
- lectura por capas sistemicas;
- dataset maestro con imputaciones.

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
