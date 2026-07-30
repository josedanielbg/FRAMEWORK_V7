# FRAMEWORK V7

Framework para la gestion hidrica del Rio Bogota con capas biofisicas, hidraulicas, de percepcion y gobernanza.

## Estructura

- `DATA/RAW`: fuentes originales.
- `DATA/MASTER`: datasets consolidados por capa y datasets maestros.
- `DATA/MACHINE_LEARNING`: transformaciones, diagnosticos y secuencias para modelado.
- `DATA/EVALUACIONES`: resultados del experimento predictivo.
- `DATA/DISENO_EXPERIMENTAL`: catalogo, configuracion, criterios y estado de los experimentos.
- `DATA/MODELADO`: tensores, registros, metricas, modelos y diagnosticos de entrenamiento.
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
- `pipeline/`: funciones reutilizables extraidas de los notebooks C08-C15 y diseno experimental.

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

## Notebooks modularizados

Los notebooks se conservan como memoria metodologica, pero la logica reutilizable vive en `src/framework_v7/pipeline`. Esta separacion evita que el proyecto dependa de ejecutar celdas manuales y permite reutilizar funciones desde notebooks, scripts o validaciones automatizadas.

Mapa de modulos:

- `pipeline/integration.py`: integracion C08 y construccion de llaves `Fecha`/`Nodo`.
- `pipeline/feature_engineering.py`: preparacion C09, variables temporales, imputacion y cobertura.
- `pipeline/domain_knowledge.py`: catalogo de conocimiento C10.
- `pipeline/ipml.py`: calculo del indice de pertinencia para machine learning C11.
- `pipeline/ml_preparation.py`: seleccion de predictoras y dataset de modelado C12.
- `pipeline/machine_learning.py`: escalamiento, diagnostico y secuencias temporales C13.
- `pipeline/modeling.py`: configuracion experimental, validacion de tensores y salidas de prediccion C14.
- `pipeline/evaluation.py`: metricas de clasificacion/regresion y recomendaciones C15.
- `pipeline/experiment_design.py`: carga, resumen y plan de diseno experimental.
- `pipeline/utils.py`: lectura/escritura de tablas y validaciones comunes.
- `pipeline/main.py`: ejecucion ligera del pipeline modular sin usar Streamlit.

Uso desde un notebook:

```python
from framework_v7.pipeline.ml_preparation import build_modeling_dataset
from framework_v7.pipeline.machine_learning import create_temporal_sequences
```

Ejecucion por consola:

```bash
python main.py
```

## App Streamlit

La aplicacion muestra:

- dashboard ejecutivo del sistema multicapa;
- mapa visual de flujo desde capas hasta modelo;
- predicciones de `Exp01`;
- diseno experimental de `Exp01` a `Exp08`;
- diagnostico y recomendaciones del modelo `Exp01`;
- dataset transformado para machine learning;
- tabs por cada capa del framework;
- datasets maestros y cobertura de variables;
- indice de notebooks.

## Diseno experimental

La carpeta `DATA/DISENO_EXPERIMENTAL` define la planeacion de los experimentos del framework. Su objetivo es separar la memoria metodologica de los datos operativos para que cada experimento tenga pregunta, objetivo, variable objetivo, tipo de problema, modelo, ventana temporal, horizonte predictivo y estado de ejecucion.

Artefactos principales:

- `catalogo_experimentos.csv`: inventario de `Exp01` a `Exp08`, con pregunta de investigacion, objetivo, variable objetivo y estado.
- `configuracion_experimentos.csv`: parametros comunes de entrenamiento, como modelo, ventana, horizonte, transformacion, optimizador, learning rate, batch size, epochs, loss y metrica.
- `variables_predictoras.csv`: variables usadas como entrada inicial del experimento.
- `estado_experimentos.csv`: bitacora de avance, resultados y observaciones por experimento.
- `criterios_clasificacion.csv`: reglas de evaluacion para experimentos de clasificacion.
- `criterios_regresion.csv`: reglas de evaluacion para experimentos de regresion.

El primer experimento (`Exp01`) predice `irca` como problema de clasificacion usando una ventana temporal de 12 registros y un horizonte de 1. Los experimentos pendientes amplian el framework hacia calidad del agua, nivel minimo, volumen util, DBO5, DQO, oxigeno disuelto y pH.

La app Streamlit incluye la seccion `Diseno experimental`, donde se puede revisar:

- mapa experimental por tipo de problema y estado;
- configuracion de modelos;
- variables predictoras;
- criterios de evaluacion;
- diagnostico del modelo `Exp01`;
- recomendaciones metodologicas para mejorar el siguiente entrenamiento.

La memoria de este flujo esta en `NOTEBOOKS/DISENO_EXPERIMENTAL/FW7_Diseño_Experimental.ipynb`.

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
- `DATA/DISENO_EXPERIMENTAL/catalogo_experimentos.csv`
- `DATA/DISENO_EXPERIMENTAL/configuracion_experimentos.csv`
- `DATA/MODELADO/Diagnosticos/Exp01/diagnostico_modelo_Exp01.csv`
- `DATA/MODELADO/Diagnosticos/Exp01/recomendaciones_Exp01.csv`
- `DATA/MACHINE_LEARNING/C13_MACHINE_LEARNING/Transformaciones/Exp01/dataset_machine_learning_transformado.csv`
- `DATA/MASTER/C09_MASTER/Dataset_Maestro_Framework_v03_Con_Imputaciones.csv`

## Objetivo del experimento

La metadata actual define `irca` como variable objetivo. La app interpreta este primer experimento como una primera salida predictiva para evaluar el comportamiento del indicador de riesgo/calidad dentro del enfoque multicapa del framework.
