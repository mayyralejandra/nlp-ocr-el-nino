"""Aplicación de OCR y NLP para el análisis automático de documentos en contexto peruano.

Caso de estudio: comunicados institucionales sobre el Fenómeno El Niño en el Perú.
Secciones: Inicio, Carga y extracción, Análisis NLP, Clasificación, Conjunto de datos, Descargas.
"""
from __future__ import annotations

import re
from pathlib import Path, PurePath

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

import limpieza
import modelo
import nlp
import ocr

RAIZ = Path(__file__).resolve().parent
RUTA_CSV = RAIZ / "data" / "dataset_ocr_utilizado.csv"
DIR_RESULTADOS = RAIZ / "resultados"
MAX_MB = 20
ACENTO, GRIS, ROJO = "#1F4E79", "#9AA5B1", "#B5473A"
NOMBRE_CLASE = {0: "No El Niño", 1: "El Niño"}

st.set_page_config(page_title="OCR y NLP – Comunicados institucionales", page_icon=None, layout="wide")

st.markdown(
    """
    <style>
    .block-container {padding-top: 2.2rem; max-width: 1200px;}
    h1, h2, h3 {color: #1F2933; letter-spacing: -0.01em;}
    h1 {font-size: 1.9rem !important;}
    [data-testid="stMetric"] {background: #FFFFFF; border: 1px solid #D9DEE5; border-radius: 6px;
                              padding: 0.6rem 0.9rem;}
    [data-testid="stMetricLabel"] {color: #52606D;}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================================
# Utilidades
# ============================================================================
@st.cache_data(show_spinner=False, max_entries=30)
def procesar_archivo(nombre: str, datos: bytes) -> ocr.ResultadoExtraccion:
    """Extrae el texto; se cachea por contenido para no repetir el OCR en cada interacción."""
    return ocr.extraer_texto(nombre, datos)


@st.cache_resource(show_spinner="Cargando modelos…")
def obtener_modelos():
    return modelo.cargar_modelos()


@st.cache_data(show_spinner=False, max_entries=100)
def clasificar(texto: str) -> dict:
    conjunto, error = obtener_modelos()
    if conjunto is None:
        return {"error": error}
    return modelo.predecir(conjunto, texto)


@st.cache_data(show_spinner=False)
def cargar_dataset() -> tuple[pd.DataFrame | None, str]:
    if not RUTA_CSV.exists():
        return None, "No se encontró data/dataset_ocr_utilizado.csv."
    try:
        df = pd.read_csv(RUTA_CSV, keep_default_na=False)
    except Exception as e:
        return None, f"No se pudo leer el CSV: {e}"
    df["etiqueta_nombre"] = df["etiqueta"].map(NOMBRE_CLASE).fillna("Sin etiqueta")
    df["texto_norm"] = df["texto_ocr"].map(nlp.normalizar_nlp)
    df["palabras"] = df["texto_ocr"].map(limpieza.contar_palabras)
    df["terminos"] = df["texto_norm"].str.split().str.len()
    df["con_texto"] = df["texto_norm"].str.strip() != ""
    return df, ""


def cargar_resultado(nombre: str) -> pd.DataFrame | None:
    ruta = DIR_RESULTADOS / nombre
    return pd.read_csv(ruta, index_col=0 if "metricas" in nombre or "matriz" in nombre else None) \
        if ruta.exists() else None


def registrar_tabla(nombre: str, df: pd.DataFrame) -> None:
    st.session_state.setdefault("tablas", {})[nombre] = df


def csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8-sig")  # utf-8-sig: Excel muestra bien tildes y ñ


def nombre_seguro(nombre: str) -> str:
    return re.sub(r"[^\w\-]+", "_", PurePath(nombre).stem).strip("_") or "documento"


def estado_de(res: ocr.ResultadoExtraccion, texto: str) -> str:
    if not res.exito:
        return "Error"
    return "Procesado" if texto.strip() else "Sin texto"


def barras_h(df: pd.DataFrame, x: str, y: str, titulo: str, xlabel: str, color: str = ACENTO):
    """Gráfico de barras horizontales sobrio con etiquetas legibles."""
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.32 * len(df) + 1)))
    ax.barh(df[y].astype(str)[::-1], df[x][::-1], color=color)
    ax.set_title(titulo, fontsize=11, loc="left")
    ax.set_xlabel(xlabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(labelsize=9)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def disponibilidad() -> dict:
    version = ocr.configurar_tesseract()
    df, err_df = cargar_dataset()
    conjunto, err_mod = obtener_modelos()
    return {"version_ocr": version, "df": df, "err_df": err_df, "conjunto": conjunto, "err_mod": err_mod}


# ============================================================================
# Navegación
# ============================================================================
SECCIONES = ["Inicio", "Carga y extracción", "Análisis NLP", "Clasificación",
             "Conjunto de datos", "Descargas"]
with st.sidebar:
    st.markdown("### OCR y NLP")
    st.caption("Comunicados institucionales sobre El Fenómeno El Niño")
    seccion = st.radio("Sección", SECCIONES, label_visibility="collapsed")
    st.divider()
    estado = disponibilidad()
    st.caption("Estado del sistema")
    st.write(("✓ " if estado["version_ocr"] else "✗ ") + "OCR (Tesseract)")
    st.write(("✓ " if estado["df"] is not None else "✗ ") + "Conjunto de datos")
    st.write(("✓ " if estado["conjunto"] else "✗ ") + "Modelos de clasificación")
    st.caption(f"Límites: {MAX_MB} MB por archivo y {ocr.MAX_PAGINAS_PDF} páginas por PDF.")


# ============================================================================
# Inicio
# ============================================================================
def pagina_inicio():
    st.title("Aplicación de OCR y NLP para el análisis automático de documentos en contexto peruano")
    st.markdown(
        "Esta herramienta extrae texto de imágenes y PDF, lo limpia, calcula estadísticas de lenguaje "
        "y clasifica los documentos como **El Niño** o **No El Niño** con los modelos entrenados por el "
        "equipo. Sirve para explorar y comprender comunicados institucionales; **no evalúa el impacto "
        "de ninguna institución ni demuestra efectos a partir de un documento**."
    )
    st.markdown("**Caso de estudio:** comunicados institucionales sobre el Fenómeno El Niño en el Perú.")

    st.subheader("Cómo usar la aplicación")
    st.markdown(
        "1. En **Carga y extracción**, sube imágenes (JPG, JPEG, PNG) o PDF y revisa el texto extraído y limpio.\n"
        "2. En **Análisis NLP**, consulta los términos frecuentes, palabras clave y resúmenes del corpus del "
        "equipo o de tus documentos.\n"
        "3. En **Clasificación**, obtén la clase predicha, su probabilidad y los términos que más pesan.\n"
        "4. En **Conjunto de datos**, filtra los documentos registrados por el equipo.\n"
        "5. En **Descargas**, guarda textos y tablas generados."
    )

    st.subheader("Disponibilidad de componentes")
    df, conjunto = estado["df"], estado["conjunto"]
    filas = [
        ("OCR (Tesseract)",
         f"Disponible (versión {estado['version_ocr']})" if estado["version_ocr"] else
         "No disponible: los PDF digitales se leen, las imágenes y PDF escaneados no"),
        ("Conjunto de datos",
         f"Disponible: {len(df)} registros" if df is not None else estado["err_df"]),
        ("Modelos de clasificación",
         f"Disponibles: {len(conjunto.pipelines)} modelos" if conjunto else estado["err_mod"]),
    ]
    st.dataframe(pd.DataFrame(filas, columns=["Componente", "Estado"]), hide_index=True, width="stretch")
    if conjunto and conjunto.aviso_version:
        st.warning(conjunto.aviso_version)
    lang, aviso = ocr.idioma_ocr() if estado["version_ocr"] else ("", "")
    if aviso:
        st.warning(aviso)

    st.subheader("Alcance y limitaciones")
    st.markdown(
        "- Las clases del conjunto de datos están asociadas a la institución y al formato del archivo; el "
        "modelo podría haber aprendido esas diferencias y no solo el tema.\n"
        "- Con pocos documentos y clases desbalanceadas, las métricas del equipo son exploratorias.\n"
        "- El OCR puede cometer errores, y estos se propagan al análisis."
    )


# ============================================================================
# Carga y extracción
# ============================================================================
def construir_docs(archivos, unir_lineas: bool) -> list[dict]:
    docs: list[dict] = []
    barra = st.progress(0.0, text="Procesando archivos…")
    for i, f in enumerate(archivos):
        datos = f.getvalue()
        if len(datos) > MAX_MB * 1024 * 1024:
            res = ocr.ResultadoExtraccion(error=f"El archivo supera el límite de {MAX_MB} MB.")
        else:
            barra.progress(i / len(archivos), text=f"Procesando {f.name}…")
            res = procesar_archivo(f.name, datos)
        texto = res.texto if res.exito else ""
        ext = PurePath(f.name).suffix.lower().lstrip(".")
        docs.append({
            "nombre": f.name, "ext": ext, "res": res, "texto": texto,
            "limpio": limpieza.limpiar_texto(texto, unir_lineas=unir_lineas),
            "estado": estado_de(res, texto),
            "imagen": datos if ext in {"jpg", "jpeg", "png"} and res.exito else None,
        })
    barra.empty()
    return docs


def tabla_resultados(docs: list[dict]) -> pd.DataFrame:
    return pd.DataFrame([{
        "archivo": d["nombre"], "formato": d["ext"], "estado": d["estado"],
        "metodo_extraccion": d["res"].metodo if d["res"].exito else "",
        "paginas_procesadas": d["res"].paginas_procesadas if d["res"].exito else None,
        "paginas_totales": d["res"].paginas_totales if d["res"].exito else None,
        "advertencias": " | ".join(d["res"].advertencias), "error": d["res"].error,
        "palabras_extraido": limpieza.contar_palabras(d["texto"]),
        "palabras_limpio": limpieza.contar_palabras(d["limpio"]),
        "texto_extraido": d["texto"], "texto_limpio": d["limpio"],
    } for d in docs])


def pagina_carga():
    st.title("Carga y extracción de documentos")
    unir = st.checkbox(
        "Unir saltos de línea dentro de cada párrafo", value=True,
        help="El OCR corta las líneas de forma arbitraria. Si lo desactivas se conservan los saltos "
             "originales (útil para tablas o listas).")
    archivos = st.file_uploader(
        "Selecciona imágenes (JPG, JPEG, PNG) o documentos PDF", type=["jpg", "jpeg", "png", "pdf"],
        accept_multiple_files=True, key="cargador",
        help="Los archivos se procesan en memoria; no se guardan ni se modifican.")

    if archivos:
        st.session_state["docs"] = construir_docs(archivos, unir)
    docs = st.session_state.get("docs", [])
    if not docs:
        st.info("Carga uno o más archivos. Los PDF con texto digital se leen directamente; las imágenes y "
                "los PDF escaneados se procesan con OCR.")
        return
    if not archivos:
        st.caption("Se muestran los documentos procesados anteriormente en esta sesión.")
        if st.button("Quitar documentos de la sesión"):
            st.session_state.pop("docs", None)
            st.session_state.pop("tablas", None)
            st.rerun()

    st.subheader("Resumen del procesamiento")
    tabla = pd.DataFrame([{
        "Archivo": d["nombre"], "Formato": d["ext"].upper(),
        "Método": d["res"].metodo if d["res"].exito else "—",
        "Páginas": f'{d["res"].paginas_procesadas}/{d["res"].paginas_totales}' if d["res"].exito else "—",
        "Palabras (extraído)": limpieza.contar_palabras(d["texto"]),
        "Palabras (limpio)": limpieza.contar_palabras(d["limpio"]), "Estado": d["estado"],
    } for d in docs])
    st.dataframe(tabla, hide_index=True, width="stretch")

    st.subheader("Texto extraído y texto limpio")
    i = st.selectbox("Archivo a visualizar", range(len(docs)),
                     format_func=lambda k: f'{k + 1}. {docs[k]["nombre"]}')
    d, res = docs[i], docs[i]["res"]
    if not res.exito:
        st.error(f'**{d["nombre"]}**: {res.error}')
    else:
        for aviso in res.advertencias:
            st.warning(aviso)
        if not d["texto"].strip():
            st.warning("No se detectó texto. Prueba con una imagen más nítida, de mayor resolución o con "
                       "mejor contraste.")
        c = st.columns(4)
        c[0].metric("Método", res.metodo)
        c[1].metric("Páginas procesadas", f"{res.paginas_procesadas} de {res.paginas_totales}")
        c[2].metric("Palabras (extraído)", limpieza.contar_palabras(d["texto"]))
        c[3].metric("Palabras (limpio)", limpieza.contar_palabras(d["limpio"]))
        if d["imagen"]:
            with st.expander("Ver imagen cargada"):
                st.image(d["imagen"], width=480, caption=d["nombre"])
        if d["ext"] == "pdf" and res.detalle_paginas:
            with st.expander("Detalle por página"):
                for linea in res.detalle_paginas:
                    st.write(f"- {linea}")
        a, b = st.columns(2)
        a.markdown("**Texto extraído (sin modificar)**")
        a.text_area("Texto extraído", d["texto"], height=400, disabled=True,
                    label_visibility="collapsed", key=f"orig_{i}")
        b.markdown("**Texto limpio**")
        b.text_area("Texto limpio", d["limpio"], height=400, disabled=True,
                    label_visibility="collapsed", key=f"limpio_{i}")
        st.caption("La limpieza corrige espacios, saltos de línea y caracteres corruptos; conserva tildes, "
                   "mayúsculas, números y puntuación. No es la normalización que usa el clasificador.")
        base = nombre_seguro(d["nombre"])
        x, y, _ = st.columns([1, 1, 2])
        x.download_button("Texto extraído (.txt)", d["texto"].encode("utf-8"),
                          file_name=f"{base}_texto_extraido.txt", mime="text/plain", key=f"dl_o_{i}")
        y.download_button("Texto limpio (.txt)", d["limpio"].encode("utf-8"),
                          file_name=f"{base}_texto_limpio.txt", mime="text/plain", key=f"dl_l_{i}")

    df = tabla_resultados(docs)
    registrar_tabla("resultados_ocr.csv", df)
    st.download_button("Resultados y metadatos de todos los archivos (.csv)", csv_bytes(df),
                       file_name="resultados_ocr.csv", mime="text/csv")
    st.caption("El OCR de esta aplicación usa Tesseract. El texto del conjunto de datos del equipo se "
               "obtuvo con otro procedimiento de OCR, por lo que el mismo documento puede dar resultados "
               "distintos.")


# ============================================================================
# Análisis NLP
# ============================================================================
def bloque_terminos(textos_norm, clave: str, etiqueta_tabla: str):
    n = st.slider("Número de términos", 5, 30, 15, key=f"n_{clave}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Frecuencia de términos**")
        st.caption("Veces que aparece cada término (tras normalizar y quitar palabras vacías) y en cuántos "
                   "documentos aparece.")
        frec = nlp.frecuencia_terminos(textos_norm, n)
        if frec.empty:
            st.info("No hay términos para mostrar.")
        else:
            barras_h(frec, "frecuencia", "termino", "Frecuencia absoluta", "Apariciones")
            st.dataframe(frec, hide_index=True, width="stretch")
            registrar_tabla(f"frecuencia_terminos_{etiqueta_tabla}.csv", frec)
    with c2:
        st.markdown("**Palabras clave (TF-IDF promedio)**")
        st.caption("Términos y bigramas con mayor peso TF-IDF medio. Destacan palabras características "
                   "del conjunto, no necesariamente las más repetidas.")
        kw = nlp.palabras_clave_tfidf(textos_norm, n)
        if kw.empty:
            st.info("No hay términos para mostrar.")
        else:
            barras_h(kw, "tfidf_promedio", "termino", "TF-IDF promedio", "TF-IDF promedio", color=GRIS)
            st.dataframe(kw.round(4), hide_index=True, width="stretch")
            registrar_tabla(f"palabras_clave_{etiqueta_tabla}.csv", kw)


def pagina_nlp():
    st.title("Análisis NLP")
    t1, t2 = st.tabs(["Corpus del equipo", "Documentos cargados"])

    with t1:
        df = estado["df"]
        if df is None:
            st.error(estado["err_df"])
        else:
            st.caption("Cálculos sobre los textos OCR del conjunto de datos del equipo. Los documentos "
                       "sin texto se excluyen.")
            clase = st.radio("Documentos incluidos", ["Todos", "El Niño", "No El Niño"], horizontal=True)
            sub = df[df["con_texto"]]
            if clase != "Todos":
                sub = sub[sub["etiqueta_nombre"] == clase]
            st.write(f"**{len(sub)}** documentos con texto ({'ambas clases' if clase == 'Todos' else clase}); "
                     f"{int((~df['con_texto']).sum())} registro(s) del conjunto sin texto excluido(s).")
            bloque_terminos(sub["texto_norm"], "corpus", clase.lower().replace(" ", "_"))

            st.subheader("Estadísticas descriptivas")
            est = (df[df["con_texto"]].groupby("etiqueta_nombre")
                   .agg(documentos=("archivo", "count"), palabras_media=("palabras", "mean"),
                        palabras_mediana=("palabras", "median"), palabras_min=("palabras", "min"),
                        palabras_max=("palabras", "max")).round(1).reset_index()
                   .rename(columns={"etiqueta_nombre": "clase"}))
            st.dataframe(est, hide_index=True, width="stretch")
            registrar_tabla("estadisticas_descriptivas_corpus.csv", est)

            st.subheader("Resumen extractivo de un documento")
            st.caption("Selecciona las 3 oraciones con mayor peso TF-IDF del propio texto (método del "
                       "notebook). No es un resumen redactado: puede heredar errores del OCR y debe "
                       "revisarse manualmente.")
            fila = st.selectbox("Documento", sub.index, format_func=lambda k: sub.loc[k, "archivo"],
                                key="res_corpus")
            if fila is not None:
                st.write(nlp.resumen_extractivo(sub.loc[fila, "texto_ocr"]))

    with t2:
        docs = [d for d in st.session_state.get("docs", []) if d["estado"] == "Procesado"]
        if not docs:
            st.info("Aún no hay documentos procesados. Cárgalos en **Carga y extracción**.")
        else:
            st.caption("Cálculos sobre el documento seleccionado; no representan al conjunto de datos.")
            k = st.selectbox("Documento", range(len(docs)), format_func=lambda i: docs[i]["nombre"],
                             key="nlp_doc")
            d = docs[k]
            norm = nlp.normalizar_nlp(d["texto"])
            est = nlp.estadisticas_documento(d["texto"], norm)
            c = st.columns(4)
            c[0].metric("Palabras", est["palabras"])
            c[1].metric("Términos normalizados", est["terminos_normalizados"])
            c[2].metric("Términos únicos", est["terminos_unicos"])
            c[3].metric("Caracteres", est["caracteres"])
            if est["terminos_normalizados"] < modelo.MIN_TOKENS:
                st.warning("Texto muy corto: los resultados no son representativos.")
            else:
                n = st.slider("Número de términos", 5, 30, 15, key="n_doc")
                frec = nlp.frecuencia_terminos([norm], n)
                barras_h(frec[["termino", "frecuencia"]], "frecuencia", "termino",
                         "Frecuencia en el documento", "Apariciones")
                registrar_tabla(f"frecuencia_terminos_{nombre_seguro(d['nombre'])}.csv", frec)
                st.markdown("**Resumen extractivo**")
                st.write(nlp.resumen_extractivo(d["texto"]))
                st.caption("Las oraciones provienen del texto extraído. Revísalas contra el original.")


# ============================================================================
# Clasificación
# ============================================================================
def mostrar_validacion():
    st.subheader("Validación del equipo (sobre el conjunto de datos, no sobre documentos nuevos)")
    metricas = cargar_resultado("metricas_clasificador.csv")
    if metricas is None:
        st.info("No hay métricas de validación disponibles en la carpeta resultados/.")
        return
    st.markdown(
        "Estas cifras son predicciones **fuera de entrenamiento** de la validación cruzada "
        "(5 particiones por grupo): cada documento se predijo con el modelo que no lo vio. "
        "Corresponden a los modelos individuales; **el promedio de los 5 modelos usado en esta aplicación "
        "no fue evaluado como conjunto.**"
    )
    st.dataframe(metricas.round(3).reset_index(names="clase / promedio"), hide_index=True, width="stretch")
    cm = cargar_resultado("matriz_confusion.csv")
    if cm is not None:
        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        ax.imshow(cm.values, cmap="Blues")
        ax.set_xticks(range(2), [c.replace("pred: ", "") for c in cm.columns], fontsize=9)
        ax.set_yticks(range(2), [i.replace("real: ", "") for i in cm.index], fontsize=9)
        ax.set_xlabel("Predicción"); ax.set_ylabel("Etiqueta real")
        ax.set_title("Matriz de confusión (validación cruzada)", fontsize=10)
        for (r, c), v in zip([(r, c) for r in range(2) for c in range(2)], cm.values.ravel()):
            ax.text(c, r, int(v), ha="center", va="center",
                    color="white" if v > cm.values.max() / 2 else "#1F2933", fontsize=12)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
        registrar_tabla("matriz_confusion_validacion.csv", cm.reset_index(names="real"))
    registrar_tabla("metricas_validacion.csv", metricas.reset_index(names="clase"))
    st.caption(
        "Estas métricas fueron regeneradas con `scripts/generar_resultados.py` a partir del CSV y de los "
        "modelos entregados (se comprobó que cada partición coincide con su modelo). Con 47 documentos y "
        "clases desbalanceadas son exploratorias; además la clase está asociada a institución y formato, "
        "y los resultados podrían no generalizar a fuentes nuevas."
    )


def pagina_clasificacion():
    st.title("Clasificación de documentos")
    conjunto = estado["conjunto"]
    if conjunto is None:
        st.error(f"No se pueden realizar predicciones. {estado['err_mod']}")
        mostrar_validacion()
        return
    if conjunto.aviso_version:
        st.warning(conjunto.aviso_version)

    docs = [d for d in st.session_state.get("docs", []) if d["estado"] == "Procesado"]
    if not docs:
        st.info("Carga documentos en **Carga y extracción** para clasificarlos.")
    else:
        st.caption("Clase según el promedio de las probabilidades de los 5 modelos (TF-IDF + regresión "
                   "logística). El texto se normaliza exactamente como en el entrenamiento.")
        filas = []
        for d in docs:
            r = clasificar(d["texto"])
            fila = {"archivo": d["nombre"], "error": r.get("error", "")}
            if not r["error"]:
                clase = r["clase"]
                fila.update({
                    "clase_predicha": NOMBRE_CLASE[clase],
                    "prob_el_nino": round(r["prob_nino"], 4),
                    "prob_clase_predicha": round(r["prob_nino"] if clase == 1 else 1 - r["prob_nino"], 4),
                    "modelos_a_favor_el_nino": f'{r["votos_nino"]}/{r["n_modelos"]}',
                    "cobertura_vocabulario": round(r["cobertura_vocabulario"], 3),
                    "terminos_normalizados": r["n_terminos"],
                    "metodo_extraccion": d["res"].metodo,
                })
                for nom, p in r["por_modelo"]:
                    fila[f"prob_{nom.replace(' ', '_')}"] = round(p, 4)
            filas.append(fila)
        res_df = pd.DataFrame(filas)
        st.dataframe(res_df.drop(columns=[c for c in res_df if c.startswith("prob_pliegue")]),
                     hide_index=True, width="stretch")
        if (res_df["error"] == "").any():
            registrar_tabla("clasificacion_documentos.csv", res_df)

        k = st.selectbox("Detalle del documento", range(len(docs)),
                         format_func=lambda i: docs[i]["nombre"], key="clf_doc")
        d, r = docs[k], clasificar(docs[k]["texto"])
        if r["error"]:
            st.error(r["error"])
        else:
            clase = r["clase"]
            p_clase = r["prob_nino"] if clase == 1 else 1 - r["prob_nino"]
            c = st.columns(4)
            c[0].metric("Clase predicha", NOMBRE_CLASE[clase])
            c[1].metric("Probabilidad de esa clase", f"{p_clase:.1%}")
            c[2].metric("Modelos a favor de El Niño", f'{r["votos_nino"]} de {r["n_modelos"]}')
            c[3].metric("Cobertura del vocabulario", f'{r["cobertura_vocabulario"]:.0%}')
            if r["cobertura_vocabulario"] < 0.5:
                st.warning("Menos de la mitad de los términos del documento está en el vocabulario de los "
                           "modelos: la predicción es poco fiable.")
            if r["n_terminos"] < 30:
                st.warning(f'El texto es corto ({r["n_terminos"]} términos tras normalizar); la predicción '
                           "puede ser inestable.")
            if 0.4 <= r["prob_nino"] <= 0.6:
                st.warning("La probabilidad está cerca de 50 %: el resultado es poco concluyente.")
            for a in d["res"].advertencias:
                st.warning(f"Extracción: {a}")
            st.caption("La probabilidad proviene de regresión logística y no está calibrada; no equivale a "
                       "la certeza de que el documento pertenezca a la clase.")

            pm = pd.DataFrame(r["por_modelo"], columns=["modelo", "prob_nino"])
            fig, ax = plt.subplots(figsize=(6, 2.6))
            ax.bar(pm["modelo"], pm["prob_nino"], color=ACENTO)
            ax.axhline(0.5, color=ROJO, linestyle="--", linewidth=1)
            ax.set_ylim(0, 1); ax.set_ylabel("P(El Niño)")
            ax.set_title("Probabilidad de El Niño por modelo (línea: umbral 0,5)", fontsize=10, loc="left")
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=8)
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)

            st.markdown("**Términos que más contribuyen a la clasificación**")
            st.caption("Contribución = coeficiente del modelo × peso TF-IDF del término, promediada entre los "
                       "5 modelos. Describe cómo pesa el modelo cada término; no es una explicación causal.")
            a, b = st.columns(2)
            for col, clave, titulo in ((a, "hacia_nino", "Empujan hacia El Niño"),
                                       (b, "hacia_no_nino", "Empujan hacia No El Niño")):
                col.markdown(f"*{titulo}*")
                datos = r["contribuciones"][clave]
                if datos:
                    col.dataframe(pd.DataFrame(datos, columns=["término", "contribución"]).round(4),
                                  hide_index=True, width="stretch")
                else:
                    col.write("Sin términos relevantes.")
    st.divider()
    mostrar_validacion()


# ============================================================================
# Conjunto de datos
# ============================================================================
def pagina_datos():
    st.title("Conjunto de datos del proyecto")
    df = estado["df"]
    if df is None:
        st.error(estado["err_df"])
        return
    st.write(f"**{len(df)}** registros y **{df.shape[1] - 5}** variables originales: "
             + ", ".join(f"`{c}`" for c in df.columns[:10]) + ".")
    sin_texto = int((~df["con_texto"]).sum())
    if sin_texto:
        st.caption(f"{sin_texto} registro(s) no tiene(n) texto OCR. En el CSV, `error` está vacío y `paginas` "
                   "es “No indicado” en todos los registros.")

    c = st.columns(3)
    inst = c[0].multiselect("Institución", sorted(df["institucion"].unique()))
    etiq = c[1].multiselect("Etiqueta", ["El Niño", "No El Niño"])
    fmt = c[2].multiselect("Formato", sorted(df["formato"].unique()))
    sub = df
    if inst:
        sub = sub[sub["institucion"].isin(inst)]
    if etiq:
        sub = sub[sub["etiqueta_nombre"].isin(etiq)]
    if fmt:
        sub = sub[sub["formato"].isin(fmt)]
    st.write(f"{len(sub)} de {len(df)} registros.")
    if sub.empty:
        st.info("Ningún registro cumple los filtros.")
        return

    vista = sub[["archivo", "institucion", "etiqueta_nombre", "formato", "palabras", "grupo"]].rename(
        columns={"etiqueta_nombre": "etiqueta", "palabras": "palabras_ocr"})
    st.dataframe(vista, hide_index=True, width="stretch")
    registrar_tabla("dataset_filtrado.csv", sub.drop(columns=["texto_norm", "con_texto", "terminos"]))

    st.subheader("Documento seleccionado")
    k = st.selectbox("Documento", sub.index, format_func=lambda i: sub.loc[i, "archivo"])
    fila = sub.loc[k]
    m = st.columns(4)
    m[0].metric("Institución", fila["institucion"])
    m[1].metric("Etiqueta", fila["etiqueta_nombre"])
    m[2].metric("Formato", fila["formato"].upper())
    m[3].metric("Palabras (OCR)", int(fila["palabras"]))
    pv = cargar_resultado("predicciones_validacion.csv")
    if pv is not None and fila["archivo"] in set(pv["archivo"]):
        p = pv[pv["archivo"] == fila["archivo"]].iloc[0]
        st.caption(f"Predicción de validación cruzada (modelo del pliegue {int(p['pliegue_prueba'])}, que no "
                   f"vio este documento): {NOMBRE_CLASE[int(p['prediccion_cv'])]} "
                   f"(P El Niño = {p['prob_nino']:.2f}).")
    if not fila["con_texto"]:
        st.warning("Este registro no tiene texto OCR.")
    st.text_area("Texto OCR", fila["texto_ocr"], height=300, disabled=True, key=f"ds_{k}")

    st.subheader("Comparación entre documentos e instituciones")
    g1, g2 = st.columns(2)
    with g1:
        cuenta = sub.groupby(["institucion", "etiqueta_nombre"]).size().unstack(fill_value=0)
        cuenta = cuenta.loc[cuenta.sum(axis=1).sort_values().index]
        fig, ax = plt.subplots(figsize=(6.5, max(2.8, 0.35 * len(cuenta) + 1)))
        izq = pd.Series(0, index=cuenta.index)
        for nombre, color in (("El Niño", ACENTO), ("No El Niño", GRIS)):
            if nombre in cuenta:
                ax.barh(cuenta.index, cuenta[nombre], left=izq, color=color, label=nombre)
                izq = izq + cuenta[nombre]
        ax.set_title("Documentos por institución y etiqueta", fontsize=11, loc="left")
        ax.set_xlabel("Documentos"); ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=8)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with g2:
        cx = pd.crosstab(sub["etiqueta_nombre"], sub["formato"]).reset_index()
        st.markdown("**Etiqueta según formato del archivo**")
        st.dataframe(cx, hide_index=True, width="stretch")
        st.caption("Si una clase se concentra en un formato o institución, un clasificador puede aprender "
                   "esa diferencia en lugar del tema.")
        fig, ax = plt.subplots(figsize=(6, 3))
        for nombre, color in (("El Niño", ACENTO), ("No El Niño", GRIS)):
            v = sub[(sub["etiqueta_nombre"] == nombre) & sub["con_texto"]]["palabras"]
            if len(v):
                ax.hist(v, bins=10, alpha=0.7, color=color, label=nombre)
        ax.set_title("Palabras por documento (OCR)", fontsize=11, loc="left")
        ax.set_xlabel("Palabras"); ax.set_ylabel("Documentos"); ax.legend(fontsize=8, frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        fig.tight_layout(); st.pyplot(fig); plt.close(fig)


# ============================================================================
# Descargas
# ============================================================================
def pagina_descargas():
    st.title("Descargas")
    docs = st.session_state.get("docs", [])
    if docs:
        st.subheader("Textos de los documentos cargados")
        for i, d in enumerate(docs):
            if d["estado"] != "Procesado":
                continue
            base = nombre_seguro(d["nombre"])
            a, b, c = st.columns([2, 1, 1])
            a.write(d["nombre"])
            b.download_button("Extraído (.txt)", d["texto"].encode("utf-8"), file_name=f"{base}_texto_extraido.txt",
                              mime="text/plain", key=f"d_o_{i}")
            c.download_button("Limpio (.txt)", d["limpio"].encode("utf-8"), file_name=f"{base}_texto_limpio.txt",
                              mime="text/plain", key=f"d_l_{i}")
    tablas = st.session_state.get("tablas", {})
    if tablas:
        st.subheader("Tablas generadas en esta sesión")
        st.caption("Solo aparecen las tablas que ya se calcularon al visitar cada sección.")
        for nombre, t in tablas.items():
            a, b = st.columns([3, 1])
            a.write(f"{nombre} ({len(t)} filas)")
            b.download_button("Descargar CSV", csv_bytes(t), file_name=nombre, mime="text/csv",
                              key=f"t_{nombre}")
    if not docs and not tablas:
        st.info("Aún no hay nada que descargar. Procesa documentos o visita las secciones de análisis.")


{"Inicio": pagina_inicio, "Carga y extracción": pagina_carga, "Análisis NLP": pagina_nlp,
 "Clasificación": pagina_clasificacion, "Conjunto de datos": pagina_datos,
 "Descargas": pagina_descargas}[seccion]()
