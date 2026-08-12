
# RESUMEN EJECUTIVO DEL ANÁLISIS EXPLORATORIO DE DATOS

## Dataset

Dataset Maestro V04

## Variables analizadas

35

## Cobertura promedio

60.76 %

## Variables con cobertura menor al 20 %

15

   INDICE DE DESEMPENO INSTITUCIONAL
     CONTINUIDAD DE ACUEDUCTO URBANO
                    CALIDAD DEL AGUA
           AGUAS RESIDUALES TRATADAS
                        Nivel_Minimo
             CONDUCTIVIDAD ELECTRICA
DEMANDA BIOQUIMICA DE OXIGENO (DBO5)
    DEMANDA QUIMICA DE OXIGENO (DQO)
                       FOSFORO TOTAL
                     NITROGENO TOTAL
               OXIGENO DISUELTO (OD)
         SOLIDOS SUSPENDIDOS TOTALES
                         TEMPERATURA
                            TURBIDEZ
                                  pH

## Variables con alta variabilidad (CV > 50 %)

18

                    Precipitacion_mm
                                 Mes
                                 ONI
                                irca
                     POBLACION TOTAL
                DENSIDAD POBLACIONAL
   COBERTURA DE ALCANTARILLADO RURAL
                    CALIDAD DEL AGUA
           AGUAS RESIDUALES TRATADAS
                        Nivel_Minimo
             CONDUCTIVIDAD ELECTRICA
DEMANDA BIOQUIMICA DE OXIGENO (DBO5)
    DEMANDA QUIMICA DE OXIGENO (DQO)
                       FOSFORO TOTAL
                     NITROGENO TOTAL
               OXIGENO DISUELTO (OD)
         SOLIDOS SUSPENDIDOS TOTALES
                            TURBIDEZ

## Variables con mayor presencia de valores atípicos

| Variable                             |   %Outliers |
|:-------------------------------------|------------:|
| POBLACION TOTAL                      |       25    |
| COBERTURA DE ACUEDUCTO RURAL         |       25    |
| INDICE DE DESEMPENO INSTITUCIONAL    |       25    |
| ACCESO A AGUA POTABLE ADECUADO       |       20.45 |
| COBERTURA DE ACUEDUCTO URBANO        |       18.18 |
| DEMANDA BIOQUIMICA DE OXIGENO (DBO5) |       15.38 |
| CALIDAD DEL AGUA                     |       12.5  |
| pH                                   |       12    |
| CONDUCTIVIDAD ELECTRICA              |        8    |
| SOLIDOS SUSPENDIDOS TOTALES          |        8    |

## Variables aproximadamente simétricas

12

                        Temp_Max_C
                        Temp_Min_C
                      Temp_Media_C
                   Radiacion_Solar
                              Anio
                               Mes
             VolumenUtilDiarioMasa
COBERTURA DE ALCANTARILLADO URBANO
   CONTINUIDAD DE ACUEDUCTO URBANO
                  CALIDAD DEL AGUA
             OXIGENO DISUELTO (OD)
                                pH

## Variables altamente asimétricas

14

                                  Humedad_Relativa
                                              irca
                                   POBLACION TOTAL
                    ACCESO A AGUA POTABLE ADECUADO
PORCENTAJE DE LA POBLACION CON ACCESO A METODOS...
                     COBERTURA DE ACUEDUCTO URBANO
                      COBERTURA DE ACUEDUCTO RURAL
                                      Nivel_Minimo
                           CONDUCTIVIDAD ELECTRICA
              DEMANDA BIOQUIMICA DE OXIGENO (DBO5)
                                     FOSFORO TOTAL
                                   NITROGENO TOTAL
                       SOLIDOS SUSPENDIDOS TOTALES
                                          TURBIDEZ

## Conclusiones

El análisis exploratorio univariado del Dataset Maestro V04
permitió caracterizar 35 variables numéricas analíticas en términos
de cobertura, tendencia central, dispersión, asimetría, curtosis
y presencia de valores atípicos.

Las variables con baja cobertura fueron identificadas y documentadas
para su consideración en las etapas posteriores del Framework.

Las variables con alta variabilidad, distribuciones asimétricas y
presencia de valores atípicos requieren especial atención durante
los análisis posteriores y la preparación de datos.

Con estos resultados se completa la caracterización univariada y
se continúa con la siguiente etapa de C10 correspondiente al
Análisis Bivariado y Correlaciones.

La evaluación de la pertinencia de las variables para procesos de
modelado se realizará posteriormente mediante el componente IPML
del Framework.
