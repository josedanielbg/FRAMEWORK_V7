
# Capa Climática - Climatica

**Fecha de Generación:** 2026-07-14 00:22:15

## 1. Descripción General
Este documento describe el proceso de generación de la capa climática, que integra datos de diversas fuentes para proporcionar información relevante sobre el clima. La capa es parte del **FRAMEWORK V7**, un ecosistema estandarizado para la gestión y análisis de datos.

## 2. Origen de los Datos
- **Fuente:** NASA POWER
- **URL:** https://power.larc.nasa.gov

## 3. Metodología del Procesamiento (Pipeline)
El procesamiento de la Capa Climática sigue la siguiente secuencia de módulos:

*   **M0 Configuración:** Montaje de Google Drive, importación de librerías y definición de rutas globales.
*   **M1 Lectura:** Conexión e ingesta del archivo original (`01_Capa_Climatica_V1.xlsx`).
*   **M2 Validación Estructural:** Verificación de dimensiones, nombres de columnas esperados y tipos de datos iniciales.
*   **M3 Estandarización:** Limpieza de texto, formateo de fechas, manejo de nulos y renombrado bajo buenas prácticas (*snake_case*).
*   **M4 Gobierno del Dato:** Aplicación de reglas de negocio, validación de rangos lógicos (ej. temperaturas viables) y seguridad. Creación de ID_Registro, ID_Capa y Hash.
*   **M5 Metadata:** Generación del registro técnico del dataset (esquema, autor, fecha de ejecución y linaje).
*   **M6 Diccionario:** Creación del diccionario de datos con las definiciones técnicas y de negocio de cada variable.
*   **M7 Auditoría:** Control de registros (filas procesadas vs. descartadas) y logs de errores.
*   **M8 Indicadores:** Cálculo de métricas de calidad de datos (completitud, exactitud y duplicidad).
*   **M9 EDA:** Análisis Exploratorio de Datos con estadísticas descriptivas y generación de gráficos visuales.
*   **M10 README:** Documentación general del proceso (este componente).

## 4. Resultados Clave
- **Registros Procesados:** 528 
- **Columnas Generadas:** 20
- **Valores Nulos Totales:** 0
- **Filas Duplicadas:** 0

## 5. Estructura de Archivos Generada
Los artefactos generados por este pipeline se organizan en el directorio `01_Capa_Climatica/` de la siguiente manera:

```text
01_Capa_Climatica/
├── 01_Capa_Climatica_V1.xlsx          ← Evidencia original (Datos crudos)
├── 02_Capa_Climatica_Framework.xlsx   ← Dataset estandarizado y limpio (Listo para usar)
├── 03_Metadata.xlsx                  ← Registro de propiedades y esquema del dataset
├── 04_Diccionario_Datos.xlsx         ← Definición técnica y de negocio de las variables
├── 05_Auditoria.xlsx                 ← Logs de procesamiento y control de registros
├── 06_Indicadores.xlsx               ← Métricas de calidad del dato
├── 07_EDA.xlsx                       ← Reporte estadístico del Análisis Exploratorio
├── 08_README.md                      ← Copia de respaldo de esta documentación
└── Graficos/                         ← Artefactos visuales exportados en el M9
    ├── Serie_Precipitacion.png       ← Tendencia temporal de lluvias
    ├── Serie_Temperatura.png         ← Tendencia temporal de temperatura
    ├── Boxplot_Variables.png         ← Identificación de valores atípicos (outliers)
    ├── Histograma_Lluvia.png         ← Distribución de la frecuencia de precipitaciones
    ├── Correlacion.png               ← Matriz de relaciones numéricas
    └── Heatmap.png                   ← Mapa de calor de densidad/correlación
```

Este README se guarda como `08_README.md` dentro de la estructura de la capa.
