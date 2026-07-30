# FRAMEWORK V7

Framework para la aplicacion de tecnologias 4.0 y ciencia de datos en la
gestion hidrica del Rio Bogota. El proyecto organiza datos biofisicos,
hidraulicos, sociales e institucionales en un flujo multicapa que permite
explorar informacion, documentar experimentos y evaluar un primer modelo
predictivo sobre indicadores del agua.

## Objetivo

El objetivo del repositorio es convertir el trabajo exploratorio de notebooks
en un proyecto reproducible, modular y consultable. Para eso separa:

- datos fuente y datos consolidados en `DATA`;
- memoria metodologica en `NOTEBOOKS`;
- funciones reutilizables en `src/framework_v7`;
- visualizacion de resultados en `app.py`;
- validacion y ejecucion ligera en `main.py`.

## Estructura Del Repositorio

```text
FRAMEWORK_V7/
├── DATA/
│   ├── RAW/
│   ├── MASTER/
│   ├── MACHINE_LEARNING/
│   ├── MODELADO/
│   ├── EVALUACIONES/
│   └── DISENO_EXPERIMENTAL/
├── NOTEBOOKS/
├── src/
│   └── framework_v7/
│       ├── layers/
│       └── pipeline/
├── app.py
├── main.py
├── pyproject.toml
└── requirements.txt
```

## Capas Del Framework

El sistema esta organizado en siete capas principales:

- `C01 - Climatica`: precipitacion, temperatura, humedad, radiacion y viento.
- `C02 - Hidrologica`: respuesta fisica del rio y niveles hidricos.
- `C03 - Calidad de agua`: variables fisicoquimicas y sanitarias.
- `C04 - ONI`: variabilidad macroclimatica asociada a ENSO.
- `C05 - Hidraulica`: disponibilidad y operacion hidraulica.
- `C06 - Percepcion`: lectura social del problema hidrico.
- `C07 - Gobernanza`: capacidad institucional, cobertura y gestion publica.

Cada capa tiene un modulo en `src/framework_v7/layers` con una interfaz comun:

- `load_dataset()`
- `available_key_variables()`
- `feature_frame()`
- `summary()`

## Pipeline Modular De Notebooks

Los notebooks se conservan como memoria metodologica, pero la logica
reutilizable vive en `src/framework_v7/pipeline`. Esta separacion permite que
los notebooks importen funciones, en vez de concentrar toda la logica en celdas
manuales.

Mapa de modulos:

- `pipeline/utils.py`: lectura, escritura y validaciones comunes.
- `pipeline/integration.py`: integracion C08 y construccion de llaves
  `Fecha`/`Nodo`.
- `pipeline/feature_engineering.py`: preparacion C09, temporalidad, imputacion
  y cobertura.
- `pipeline/domain_knowledge.py`: catalogo de conocimiento C10.
- `pipeline/ipml.py`: indice de pertinencia para machine learning C11.
- `pipeline/ml_preparation.py`: seleccion de predictoras y target C12.
- `pipeline/machine_learning.py`: escalamiento, diagnostico y secuencias C13.
- `pipeline/modeling.py`: configuracion, tensores y predicciones C14.
- `pipeline/evaluation.py`: metricas y recomendaciones C15.
- `pipeline/experiment_design.py`: diseno experimental y plan de experimentos.
- `pipeline/main.py`: validacion ligera del pipeline modular.

Ejemplo de uso desde un notebook:

```python
from framework_v7.pipeline.ml_preparation import build_modeling_dataset
from framework_v7.pipeline.machine_learning import create_temporal_sequences
```

Ejemplo de ejecucion del pipeline modular:

```bash
python -m framework_v7.pipeline.main
```

## Diseno Experimental

La carpeta `DATA/DISENO_EXPERIMENTAL` define la planeacion de los experimentos
del framework. Cada experimento declara pregunta de investigacion, objetivo,
variable objetivo, tipo de problema, ventana temporal, horizonte predictivo,
modelo y estado.

Artefactos principales:

- `catalogo_experimentos.csv`: inventario de `Exp01` a `Exp08`.
- `configuracion_experimentos.csv`: parametros de entrenamiento.
- `variables_predictoras.csv`: variables de entrada iniciales.
- `estado_experimentos.csv`: bitacora y avance de experimentos.
- `criterios_clasificacion.csv`: criterios para evaluar clasificacion.
- `criterios_regresion.csv`: criterios para evaluar regresion.

`Exp01` usa `irca` como variable objetivo y se plantea como problema de
clasificacion con ventana temporal de 12 registros y horizonte de 1. Los
experimentos siguientes amplian el analisis hacia calidad general del agua,
nivel minimo, volumen util, DBO5, DQO, oxigeno disuelto y pH.

## Aplicacion Streamlit

La aplicacion `app.py` permite explorar el proyecto sin ejecutar notebooks.
Incluye:

- dashboard ejecutivo multicapa;
- mapa del flujo desde capas hasta modelo;
- predicciones del experimento `Exp01`;
- diseno experimental de `Exp01` a `Exp08`;
- diagnostico y recomendaciones del modelo `Exp01`;
- explorador de datasets por capa;
- explorador del dataset maestro;
- indice de notebooks.

Ejecucion local:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Ejecucion Por Consola

Para una validacion rapida del proyecto:

```bash
python main.py
```

La salida resume datos cargados, capas disponibles, experimentos disenados,
diagnosticos y modulos del pipeline de notebooks.

## Instalacion Como Paquete Local

```bash
pip install -e .
```

Esto permite importar los modulos desde notebooks o scripts:

```python
from framework_v7.pipeline.experiment_design import load_design_artifacts
```

## Artefactos Principales

- `DATA/MASTER/C09_MASTER/Dataset_Maestro_Framework_v03_Con_Imputaciones.csv`
- `DATA/MACHINE_LEARNING/C13_MACHINE_LEARNING/Transformaciones/Exp01/dataset_machine_learning_transformado.csv`
- `DATA/EVALUACIONES/Exp01/predicciones.csv`
- `DATA/EVALUACIONES/Exp01/metadata_prediccion.csv`
- `DATA/MODELADO/Diagnosticos/Exp01/diagnostico_modelo_Exp01.csv`
- `DATA/MODELADO/Diagnosticos/Exp01/recomendaciones_Exp01.csv`
- `DATA/DISENO_EXPERIMENTAL/catalogo_experimentos.csv`

## Convenciones De Desarrollo

- Mantener notebooks como evidencia y explicacion metodologica.
- Mover funciones reutilizables a `src/framework_v7/pipeline`.
- Mantener salidas consolidadas en `DATA`.
- Evitar que `app.py` contenga logica de transformacion pesada.
- Usar `main.py` y `python -m framework_v7.pipeline.main` como chequeos
  ligeros antes de publicar cambios.

## Autores

Proyecto desarrollado por Jose Barreto y Juan Riataga.
