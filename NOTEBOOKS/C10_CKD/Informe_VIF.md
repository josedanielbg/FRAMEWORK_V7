# INFORME TÉCNICO DEL ANÁLISIS DE MULTICOLINEALIDAD

Fecha de ejecución: 2026-07-28 03:20

## Objetivo

Evaluar el grado de multicolinealidad existente entre las variables numéricas del Dataset Maestro mediante el cálculo del Factor de Inflación de la Varianza (VIF), con el propósito de identificar relaciones lineales que puedan afectar la estabilidad de futuros modelos de Machine Learning.

## Metodología

- Construcción de la matriz numérica.
- Validación del Universo Analítico.
- Identificación y exclusión de variables constantes.
- Imputación de valores faltantes mediante la mediana.
- Cálculo del VIF para cada variable.
- Clasificación automática del VIF.
- Integración de resultados al Universo Analítico.

## Resultados

- Variables evaluadas: **35**
- Multicolinealidad perfecta: **7**
- Muy alta: **4**
- Alta: **5**
- Moderada: **3**
- Baja: **15**
- Sin multicolinealidad: **0**
- No calculado: **0**

## Interpretación

El análisis permitió identificar variables con multicolinealidad perfecta, asociadas principalmente a indicadores demográficos y de cobertura, confirmando los resultados obtenidos previamente mediante el análisis de correlación y el Índice de Redundancia. Asimismo, se identificaron variables con multicolinealidad alta y moderada, especialmente dentro del componente climático y de calidad del agua, evidenciando dependencias lineales que deberán ser consideradas durante el proceso de selección de variables.

## Conclusión

Los resultados obtenidos constituyen una fuente adicional de evidencia para el cálculo del Índice de Pertinencia para Machine Learning (IPML), permitiendo incorporar el efecto de la multicolinealidad dentro del proceso de priorización de variables del Framework.
