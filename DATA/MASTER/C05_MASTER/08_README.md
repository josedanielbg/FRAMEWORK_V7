# FRAMEWORK V7

# Capa Hidráulica

## Información General

Nombre de la capa: Hidráulica

Código: HDR

Versión: 1.0

Responsable: Juan Carlos Riatiga

Fuente: Reservas Hidráulicas en Masa - XM

URL Fuente:
https://www.datos.gov.co/dataset/Reservas-Hidr-ulicas-en-Masa/5cbd-e6b5

Archivo origen:
https://raw.githubusercontent.com/jriatiga/dataset/refs/heads/main/proyectoGradoV4/05_Capa_Hidraulica/01_Capa_Reservas_Hidraulicas_Masa_V1.csv

Fecha ETL:
2026-07-14

Hora ETL:
21:21:28

------------------------------------------------------------

## Objetivo

Consolidar la información hidráulica del sistema de embalses de la cuenca del río Bogotá mediante la agregación mensual del volumen útil almacenado (VolumenUtilDiarioMasa), proporcionando una serie temporal consistente para su integración con las demás capas del Framework V7 y su utilización como variable explicativa en modelos predictivos basados en LSTM.

------------------------------------------------------------

## Descripción

La capa se construyó a partir de los registros diarios de reservas hidráulicas publicados por XM. Para garantizar la compatibilidad temporal con las demás capas del Framework, los registros fueron filtrados para el embalse AGREGADO_BOGOTA y posteriormente consolidados mediante el promedio mensual del volumen útil almacenado.

Durante el proceso ETL se aplicaron procedimientos de lectura, validación estructural, estandarización, gobierno del dato, generación de metadatos, auditoría, indicadores de calidad y análisis exploratorio, garantizando la trazabilidad y reproducibilidad de la información.

------------------------------------------------------------

## Variables principales

- Fecha
- CodigoEmbalse
- VolumenUtilDiarioMasa

------------------------------------------------------------

## Cobertura temporal

Inicio:
2019-01-31

Fin:
2025-01-31

Número de registros:
73

------------------------------------------------------------

## Calidad de los datos

Valores nulos:
0

Registros duplicados:
0

Hashes únicos:
73

------------------------------------------------------------

## Resultado

La capa representa la evolución mensual del volumen útil del sistema AGREGADO_BOGOTA, constituyendo una variable hidráulica estratégica para explicar la disponibilidad de agua en la cuenca y fortalecer la capacidad predictiva del Framework V7 mediante su integración con variables climáticas, hidrológicas y de calidad del agua.

