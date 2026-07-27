
# Informe de la Capa Hidrologica (HID)

## 1. Visión General

Este documento describe el proceso de ingestión, estandarización y análisis exploratorio de datos para la Capa **Hidrologica** (Código: **HID**).

## 2. Detalles del Proyecto

*   **Versión del Framework:** 1.0
*   **Responsable:** Juan Carlos Riatiga
*   **Fuente de Datos:** Datos Abiertos Colombia
*   **URL de la Fuente:** https://www.datos.gov.co
*   **Fecha de Ejecución del ETL:** 2026-07-15 18:54:21

## 3. Resumen del Dataset

Después del procesamiento, el dataset `02_Capa_Hidrologica_Framework.xlsx` contiene:

*   **Registros Totales:** 5259
*   **Variables (Columnas):** 20
*   **Cobertura Temporal:** Del 2006-08-24 16:00:00 al 2019-08-27 23:56:00
*   **Estaciones Únicas:** 21
*   **Municipios Únicos:** 16
*   **Departamentos Únicos:** 2

## 4. Estructura de Archivos Generada

Al finalizar la ejecución del pipeline, se generan los siguientes archivos:

*   `02_Capa_Hidrologica_Framework.xlsx`: Dataset estandarizado y limpio.
*   `03_Metadata.xlsx`: Registro de propiedades y esquema del dataset.
*   `04_Diccionario_Datos.xlsx`: Definición técnica y de negocio de las variables.
*   `05_Auditoria.xlsx`: Logs de procesamiento y control de registros.
*   `06_Indicadores.xlsx`: Métricas de calidad del dato.
*   `07_EDA.xlsx`: Reporte estadístico del Análisis Exploratorio.
*   `08_README.md`: Este documento.
*   `Graficos/`: Directorio que contiene artefactos visuales generados durante el EDA.

## 5. Contacto

Para consultas o soporte, por favor contacte a Juan Carlos Riatiga.
