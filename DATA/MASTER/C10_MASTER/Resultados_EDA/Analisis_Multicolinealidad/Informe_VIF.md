# INFORME TÉCNICO DEL ANÁLISIS DE MULTICOLINEALIDAD MEDIANTE VIF

Fecha de ejecución: 2026-08-12 06:21

## 1. Objetivo

Evaluar la multicolinealidad del universo analítico numérico mediante el Factor de Inflación de la Varianza (VIF), como diagnóstico complementario al análisis de redundancia estructural realizado previamente.

## 2. Metodología

- Universo analítico inicial: **35 variables**.
- Se excluyeron del cálculo matemático del VIF las variables con varianza nula.
- Los valores faltantes se imputaron temporalmente mediante la mediana exclusivamente para permitir el cálculo del VIF.
- Esta imputación auxiliar no modifica el Dataset Maestro V04.
- Las regresiones auxiliares utilizadas para el VIF incluyeron explícitamente un intercepto.
- La constante utilizada como intercepto no forma parte del universo analítico.

### Clasificación utilizada

| VIF | Estado |
|---:|---|
| ≤ 1 | Sin Multicolinealidad |
| > 1 y < 5 | Baja |
| ≥ 5 y < 10 | Moderada |
| ≥ 10 y < 20 | Alta |
| ≥ 20 | Muy Alta |
| ∞ | Perfecta |

## 3. Resultados

- Variables del universo: **35**
- Variables con VIF calculado: **34**
- Variables con VIF no calculado: **1**
- Multicolinealidad perfecta: **0**
- Muy alta: **8**
- Alta: **3**
- Moderada: **6**
- Baja: **17**
- Sin multicolinealidad: **0**
- No calculado: **1**

## 4. Variables con mayor VIF

| Variable                       |      VIF | Estado_VIF   |
|:-------------------------------|---------:|:-------------|
| DENSIDAD POBLACIONAL           | 1329.16  | Muy Alta     |
| POBLACION TOTAL                | 1182.4   | Muy Alta     |
| Temp_Media_C                   |   82.34  | Muy Alta     |
| COBERTURA DE ACUEDUCTO URBANO  |   42.952 | Muy Alta     |
| ACCESO A AGUA POTABLE ADECUADO |   37.091 | Muy Alta     |
| Temp_Max_C                     |   36.388 | Muy Alta     |
| COBERTURA DE ACUEDUCTO RURAL   |   32.439 | Muy Alta     |
| CONDUCTIVIDAD ELECTRICA        |   22.148 | Muy Alta     |
| FOSFORO TOTAL                  |   14.559 | Alta         |
| Temp_Min_C                     |   13.221 | Alta         |

## 5. Variables con VIF no calculable

- **CONTINUIDAD DE ACUEDUCTO URBANO**: VIF no calculable por varianza nula.

## 6. Interpretación

El diagnóstico evidencia distintos niveles de dependencia lineal multivariable dentro del universo analítico. Los valores elevados de VIF indican que parte de la variabilidad de determinadas variables puede ser explicada por combinaciones lineales de otras variables del conjunto.

El VIF y el Índice de Redundancia no representan el mismo fenómeno: el primero evalúa dependencia lineal multivariable, mientras que el segundo resume las relaciones bivariadas fuertes que cumplen el criterio establecido en el análisis previo.

## 7. Consideraciones metodológicas

- El VIF fue calculado sobre una matriz auxiliar completada mediante imputación por mediana.
- Las variables con baja cobertura original deben interpretarse conjuntamente con el diagnóstico de calidad y disponibilidad de datos.
- Un VIF elevado no implica eliminación automática de una variable.
- La variable con varianza nula permanece dentro del universo analítico con estado `No Calculado`.
- Estos resultados constituyen evidencia diagnóstica que será integrada posteriormente con otros criterios en el IPML.

## 8. Conclusión

El análisis VIF proporciona una medida complementaria de multicolinealidad para las variables del Dataset Maestro V04. Sus resultados se conservan como insumo del Framework y no constituyen, por sí solos, una decisión de selección o descarte de variables.
