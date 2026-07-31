# DISEÑO EXPERIMENTAL DEL FRAMEWORK V7

Este directorio contiene la configuración oficial utilizada por el Framework para planear, ejecutar y evaluar experimentos de ciencia de datos aplicados a la gestión hídrica del Río Bogotá.

## Propósito

El diseño experimental conecta las preguntas de investigación con variables objetivo, variables predictoras, tipo de problema, ventana temporal, horizonte predictivo, modelo y criterios de evaluación. Los archivos de esta carpeta son consumidos por los notebooks C13, C14, C15 y C16, así como por la app Streamlit.

## Artefactos

- `catalogo_experimentos.csv`: plan experimental de `Exp01` a `Exp08`.
- `configuracion_experimentos.csv`: parámetros generales de modelado.
- `variables_predictoras.csv`: variables iniciales usadas como entrada.
- `estado_experimentos.csv`: bitácora de avance y resultados por experimento ejecutado.
- `criterios_clasificacion.csv`: criterios para interpretar modelos de clasificación.
- `criterios_regresion.csv`: criterios para interpretar modelos de regresión.

## Experimentos realizados

A la fecha se realizaron tres ejecuciones experimentales:

| Experimento | Tipo | Variable objetivo | Estado | Resultado principal |
|---|---|---|---|---|
| Exp01 | Clasificación | irca | Ejecutado | Línea base no apta para detectar el objetivo; accuracy 0.8462, precisión 0.0000, recall 0.0000 y F1 0.0000. |
| Exp01-V3 | Clasificación | irca | Ejecutado | Versión mejorada con capacidad predictiva parcial; accuracy 0.9226, precisión 0.5645, recall 0.7292 y F1 0.6364. |
| Exp04 | Regresión | VolumenUtilDiarioMasa | Ejecutado | Modelo aceptable para volumen útil; MAE 0.0824, RMSE 0.1470, MAPE 23.8944 y R2 0.7225. |

## Lectura metodológica

`Exp01` debe leerse como línea base: muestra que una exactitud global alta no garantiza capacidad de detección cuando las clases están desbalanceadas. `Exp01-V3` documenta la mejora del experimento IRCA y permite justificar iteración metodológica. `Exp04` amplía el framework hacia regresión y evalúa disponibilidad hídrica mediante `VolumenUtilDiarioMasa`.

Los experimentos pendientes (`Exp02`, `Exp03`, `Exp05`, `Exp06`, `Exp07` y `Exp08`) quedan como agenda de ampliación para evaluar calidad general del agua, nivel mínimo, contaminación orgánica/química, oxígeno disuelto y pH.
