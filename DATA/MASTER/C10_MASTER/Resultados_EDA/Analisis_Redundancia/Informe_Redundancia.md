# DIAGNÓSTICO DE REDUNDANCIA ESTRUCTURAL

## Objetivo

Cuantificar la redundancia estructural entre las variables numéricas del Dataset Maestro V04 como insumo para la construcción del Índice de Pertinencia para Machine Learning (IPML).

## Resultados Generales

- Variables evaluadas: **35**
- Relaciones redundantes: **32**

## Distribución del Índice de Redundancia

- Muy Baja: 24 variables
- Muy Alta: 8 variables
- Baja: 3 variables

## Variables con mayor redundancia

- DENSIDAD POBLACIONAL (IR=1.000)
- ACCESO A AGUA POTABLE ADECUADO (IR=1.000)
- PORCENTAJE DE LA POBLACION CON ACCESO A METODOS DE SANEAMIENTO ADECUADOS (IR=1.000)
- COBERTURA DE ACUEDUCTO URBANO (IR=1.000)
- COBERTURA DE ACUEDUCTO RURAL (IR=1.000)
- COBERTURA DE ALCANTARILLADO URBANO (IR=1.000)
- COBERTURA DE ALCANTARILLADO RURAL (IR=1.000)
- INDICE DE DESEMPENO INSTITUCIONAL (IR=0.833)
- CALIDAD DEL AGUA (IR=0.333)
- CONDUCTIVIDAD ELECTRICA (IR=0.333)

## Interpretación

Las variables con mayor Índice de Redundancia representan grupos de información altamente relacionados. Las variables con IR cercano a cero aportan información más independiente dentro del Dataset Maestro.

## Conclusiones

El Índice de Redundancia constituye el primer criterio objetivo para la evaluación de variables dentro del Framework y será integrado posteriormente con el VIF, la cobertura, la calidad de datos y el conocimiento del dominio para construir el IPML.
