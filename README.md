# FRAMEWORK V7

Framework para la gestion hidrica del Rio Bogota con capas biofisicas, hidraulicas, de percepcion y gobernanza.

## Estructura

- `DATA/RAW`: fuentes originales.
- `DATA/MASTER`: datasets consolidados por capa y datasets maestros.
- `DATA/MACHINE_LEARNING`: transformaciones, diagnosticos y secuencias para modelado.
- `DATA/EVALUACIONES`: resultados del experimento predictivo.
- `NOTEBOOKS`: memoria metodologica y notebooks del flujo.
- `src/framework_v7`: modulos reutilizables del proyecto.
- `app.py`: entrada principal del visor Streamlit.
- `main.py`: entrada de consola para revisar el estado del proyecto.

## Anatomia del codigo

- `catalog.py`: anatomia del negocio, capas, archivos maestros y grupos de variables.
- `data_access.py`: carga cacheada de CSV, Excel y metadata.
- `profiling.py`: funciones de perfilado, cobertura y transformaciones ligeras.
- `visualizations.py`: componentes visuales reutilizables.
- `views.py`: pantallas Streamlit por seccion.
- `paths.py`: rutas canonicas del repositorio.
- `utils.py`: utilidades generales.
- `layers/`: modulos reutilizables por capa de los Colabs C01-C07.

## Capas modularizadas

- `layers/climate.py`: capa climatica.
- `layers/hydrology.py`: capa hidrologica.
- `layers/water_quality.py`: capa de calidad de agua.
- `layers/oni.py`: capa macroclimatica ONI.
- `layers/hydraulic.py`: capa hidraulica.
- `layers/perception.py`: capa de percepcion.
- `layers/governance.py`: capa de gobernanza.

Cada modulo de capa expone una interfaz comun:

- `load_dataset()`
- `available_key_variables()`
- `feature_frame()`
- `summary()`

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

Si se quiere instalar el paquete localmente:

```bash
pip install -e .
```

Para una revision rapida por consola:

```bash
python main.py
```

## Archivos principales

- `DATA/EVALUACIONES/Exp01/predicciones.csv`
- `DATA/EVALUACIONES/Exp01/metadata_prediccion.csv`
- `DATA/MACHINE_LEARNING/C13_MACHINE_LEARNING/Transformaciones/Exp01/dataset_machine_learning_transformado.csv`
- `DATA/MASTER/C09_MASTER/Dataset_Maestro_Framework_v03_Con_Imputaciones.csv`

## Objetivo del experimento

La metadata actual define `irca` como variable objetivo. La app interpreta este primer experimento como una primera salida predictiva para evaluar el comportamiento del indicador de riesgo/calidad dentro del enfoque multicapa del framework.
