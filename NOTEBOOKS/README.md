# NOTEBOOKS

Esta carpeta conserva la memoria metodologica del proyecto, siguiendo la separacion recomendada entre experimentacion en notebooks y producto reproducible en repositorio.

## Organizacion

- `C01` a `C07`: extraccion y framework por capa.
- `C08` y `C09`: integracion y preparacion del dataset maestro.
- `C10` y `C11`: catalogo de conocimiento e indicadores.
- `C12` a `C15`: preparacion de machine learning, modelado y evaluacion.

## Uso recomendado

1. Mantener los notebooks como evidencia de exploracion.
2. Guardar salidas consolidadas en `DATA`.
3. Usar `app.py` para consultar resultados sin ejecutar manualmente el notebook completo.
4. Ejecutar logica reutilizable desde los modulos de `src/framework_v7/pipeline`.

## Modularizacion

Los notebooks no deben concentrar toda la logica. Cada etapa tiene funciones
reutilizables en `src/framework_v7/pipeline`:

- `C08_INTEGRACION` usa `pipeline/integration.py`.
- `C09_INGENIERIA_DATOS` usa `pipeline/feature_engineering.py`.
- `C10_CKD` usa `pipeline/domain_knowledge.py`.
- `C11_IPML` usa `pipeline/ipml.py`.
- `C12_PREPARACION_MACHINE_LEARNING` usa `pipeline/ml_preparation.py`.
- `C13_MACHINE_LEARNING` usa `pipeline/machine_learning.py`.
- `C14_MODELADO` usa `pipeline/modeling.py`.
- `C15_EVALUACION` usa `pipeline/evaluation.py`.
- `DISENO_EXPERIMENTAL` usa `pipeline/experiment_design.py`.

Patron recomendado dentro de cada notebook:

```python
from framework_v7.pipeline.feature_engineering import build_engineered_master
from framework_v7.pipeline.machine_learning import create_temporal_sequences
```

El archivo `src/framework_v7/pipeline/utils.py` contiene funciones de ayuda
para lectura, escritura y validacion de columnas. El archivo
`src/framework_v7/pipeline/main.py` ejecuta una validacion ligera del pipeline
sin depender de la app Streamlit.
