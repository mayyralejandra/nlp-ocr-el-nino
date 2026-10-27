# Aplicación de OCR y NLP para el análisis automático de documentos en contexto peruano

Caso de estudio: comunicados institucionales sobre el Fenómeno El Niño en el Perú.
Aplicación Streamlit con OCR (Tesseract), limpieza de texto, análisis NLP, clasificación
(El Niño / No El Niño) y exploración del conjunto de datos del equipo.

## Estructura

```text
app.py              Interfaz (6 secciones)          ocr.py        Extracción de texto (sin cambios)
limpieza.py         Limpieza conservadora (sin cambios)   modelo.py     Clasificador (sin cambios)
nlp.py              Normalización del modelo + análisis descriptivo (NUEVO)
scripts/generar_resultados.py   Regenera las métricas de validación
data/               dataset_ocr_utilizado.csv, stopwords_espanol.json
models/             modelo_pliegue_1..5.joblib
resultados/         métricas, matriz de confusión y predicciones de validación
notebooks/          Notebook de la Persona 2
originales_backup/  Copia de app.py, requirements.txt y módulos originales
.streamlit/config.toml   Tema claro        packages.txt   Tesseract para Streamlit Cloud
```

## Instalación en Windows (CMD)

1. Instala **Python 3.11 o 3.12** (marca "Add Python to PATH").
2. Instala **Tesseract OCR** (instalador de UB Mannheim) y en "Additional language data" marca **Spanish**.
   Sin el paquete `spa`, el OCR usa inglés y falla con tildes y ñ. Si Tesseract no está en
   `C:\Program Files\Tesseract-OCR\`, define `set TESSERACT_CMD=C:\ruta\tesseract.exe`.
3. En la carpeta del proyecto:

```cmd
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
streamlit run app.py
```

Para regenerar las métricas de validación (opcional): `python scripts\generar_resultados.py`

## Compatibilidad de versiones

Los modelos `.joblib` se guardaron con **scikit-learn 1.6.1**; `requirements.txt` lo fija con `==`.
Con otra versión la app carga los modelos pero muestra un aviso. Cargar un `.joblib` ejecuta código:
usa solo los del equipo.

## Qué se usa y de dónde sale

- **Normalización del clasificador**: la `normalizar` del notebook, con las palabras vacías de NLTK
  (español, 313) más `ser, asi, ademas, tambien, peru, perú`, guardadas en `data/stopwords_espanol.json`
  (el archivo `stopwords_espanol.json` del equipo no se adjuntó). Se comprobó que reentrenar con ella
  reproduce vocabularios y coeficientes de los 5 modelos.
- **Métricas**: no se adjuntó `metricas_clasificador.csv`. Se regeneraron con el procedimiento del notebook
  (`StratifiedGroupKFold`, 5 particiones, `random_state=42`) usando los modelos entregados. Son métricas de
  cada modelo sobre su pliegue; el promedio de los 5 modelos **no fue evaluado como conjunto**.
- **Técnicas NLP sobre el documento cargado**: resumen extractivo, palabras clave (frecuencia y TF-IDF), nube de palabras (`wordcloud`), entidades y temas (por listas y patrones, no es un modelo entrenado) y clasificación. Los cálculos del corpus del equipo se hacen en la app con los métodos del notebook sobre el
  texto OCR del CSV (no se adjuntaron los CSV precalculados).

## Despliegue en Streamlit Community Cloud

1. Sube el proyecto a GitHub (incluye `packages.txt`, que instala `tesseract-ocr` y `tesseract-ocr-spa`).
2. Crea la app en share.streamlit.io apuntando a `app.py`.
3. Usa una versión de Python compatible con scikit-learn 1.6.1 (3.11 o 3.12) en *Advanced settings*.

Limitaciones: memoria limitada (por eso 20 MB por archivo y 10 páginas por PDF), OCR lento en PDF escaneados,
el sistema de archivos es efímero (la app no guarda nada) y el repositorio no debe contener datos privados.

## Lista de pruebas

| Área | Prueba | Resultado esperado |
|---|---|---|
| Carga | Subir JPG, PNG y PDF a la vez | Una fila por archivo en el resumen |
| Carga | Subir un archivo dañado o de 0 bytes | Estado "Error" con mensaje claro, sin caída |
| OCR | Imagen con texto en español | Texto con tildes y ñ (requiere `spa`) |
| PDF | PDF digital / PDF escaneado | Método "Texto digital" / "OCR"; PDF >10 páginas: advertencia |
| Limpieza | Alternar "Unir saltos de línea" | El texto limpio cambia; el extraído no |
| NLP | Corpus del equipo: filtrar por clase | Frecuencias, TF-IDF, estadísticas y resumen se actualizan |
| NLP | Documentos cargados | Estadísticas del documento elegido, no del corpus |
| Clasificación | Documento cargado | Clase, probabilidad, 5 probabilidades, cobertura y términos |
| Clasificación | Texto de menos de 10 términos | Mensaje de error, sin predicción |
| Clasificación | Quitar `models/` | Mensaje "faltan modelos"; no hay predicciones |
| Datos | Filtros por institución, etiqueta y formato | Conteo y tabla coherentes; selección muestra texto OCR |
| Descargas | TXT, CSV de resultados, CSV de clasificación, tablas | Archivos con tildes correctas en Excel |
