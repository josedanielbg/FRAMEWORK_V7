# DIAGNÓSTICO DE REDUNDANCIA ESTRUCTURAL

## Objetivo

Cuantificar la redundancia estructural entre las variables numéricas analíticas del Dataset Maestro V04 como insumo para etapas posteriores del Framework y para la construcción del Índice de Pertinencia para Machine Learning (IPML).

## Criterio metodológico

Se consideran relaciones de redundancia aquellas asociaciones con correlación absoluta de Pearson >= 0.90 y con un soporte mínimo de 30 observaciones pareadas. Las correlaciones con menor soporte no se utilizan para construir este diagnóstico.

El Índice de Redundancia se obtiene normalizando el número de relaciones fuertes de cada variable respecto al máximo observado en el universo analítico. Por tanto, es un indicador relativo dentro del Dataset Maestro V04.

## Resultados generales

- Variables analíticas evaluadas: **35**
- Relaciones redundantes con soporte suficiente: **8**
- Variables involucradas en alguna relación fuerte: **14**
- Máximo de relaciones fuertes observado: **2**

## Distribución del Índice de Redundancia

- Muy Alta: 2 variables
- Alta: 0 variables
- Moderada: 12 variables
- Baja: 0 variables
- Muy Baja: 21 variables

## Variables con mayor redundancia

- AGUAS RESIDUALES TRATADAS (relaciones=2, IR=1.000, nivel=Muy Alta)
- INDICE DE DESEMPENO INSTITUCIONAL (relaciones=2, IR=1.000, nivel=Muy Alta)
- ACCESO A AGUA POTABLE ADECUADO (relaciones=1, IR=0.500, nivel=Moderada)
- CALIDAD DEL AGUA (relaciones=1, IR=0.500, nivel=Moderada)
- COBERTURA DE ACUEDUCTO RURAL (relaciones=1, IR=0.500, nivel=Moderada)
- COBERTURA DE ACUEDUCTO URBANO (relaciones=1, IR=0.500, nivel=Moderada)
- COBERTURA DE ALCANTARILLADO RURAL (relaciones=1, IR=0.500, nivel=Moderada)
- COBERTURA DE ALCANTARILLADO URBANO (relaciones=1, IR=0.500, nivel=Moderada)
- CONDUCTIVIDAD ELECTRICA (relaciones=1, IR=0.500, nivel=Moderada)
- DENSIDAD POBLACIONAL (relaciones=1, IR=0.500, nivel=Moderada)

## Interpretación

Un Índice de Redundancia elevado indica que una variable participa en una mayor proporción de las relaciones fuertes identificadas dentro del universo analítico. Un índice igual a cero indica que la variable no participa en relaciones que cumplan simultáneamente el umbral de correlación y el soporte mínimo definidos para este análisis.

Este resultado no implica que una variable sea irrelevante, ni que deba ser eliminada automáticamente. La redundancia constituye únicamente un componente diagnóstico que debe evaluarse posteriormente junto con otros criterios, incluido el VIF, la cobertura, la calidad de los datos y el conocimiento del dominio.

## Conclusión

El diagnóstico caracteriza la redundancia estructural de las 35 variables numéricas analíticas sin realizar eliminación automática. Sus resultados quedan disponibles como insumo para los análisis posteriores del Framework y para la integración final del IPML.
