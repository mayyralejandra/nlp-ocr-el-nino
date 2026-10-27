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
        "2. En **Análisis NLP**, obtén para tu documento un resumen automático, palabras clave, nube de "
        "palabras, entidades (instituciones, lugares, fechas) y temas.\n"
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
# Análisis NLP (sobre el documento cargado)
# ============================================================================
def fuentes_nlp() -> tuple[list[dict], bool]:
    """Documentos cargados; si no hay, permite probar con uno del conjunto del equipo (ejemplo)."""
    docs = [d for d in st.session_state.get("docs", []) if d["estado"] == "Procesado"]
    if docs:
        return docs, False
    df = estado["df"]
    if df is None:
        return [], False
    st.info("Aún no cargaste documentos (hazlo en **Carga y extracción**). Mientras tanto puedes probar "
            "las técnicas con un documento de ejemplo del equipo.")
    ej = df[df["con_texto"]]
    k = st.selectbox("Documento de ejemplo", ej.index, format_func=lambda i: ej.loc[i, "archivo"],
                     key="nlp_ejemplo")
    return [{"nombre": ej.loc[k, "archivo"], "texto": ej.loc[k, "texto_ocr"]}], True


def nube_palabras(frecuencias: dict):
    from wordcloud import WordCloud
    wc = WordCloud(width=1000, height=450, background_color="white", colormap="Blues",
                   collocations=False, prefer_horizontal=0.9).generate_from_frequencies(frecuencias)
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def pagina_nlp():
    st.title("Análisis de lenguaje (NLP)")
    st.markdown("El **Procesamiento de Lenguaje Natural (NLP)** son técnicas para que el computador "
                "\"lea\" un texto. Aquí se aplican al **documento que cargaste**: resumen automático, "
                "palabras clave, nube de palabras, extracción de entidades, temas y clasificación.")
    docs, es_ejemplo = fuentes_nlp()
    if not docs:
        return
    k = 0
    if len(docs) > 1:
        k = st.selectbox("Documento a analizar", range(len(docs)), format_func=lambda i: docs[i]["nombre"],
                         key="nlp_doc")
    d = docs[k]
    base = nombre_seguro(d["nombre"])
    texto = d["texto"]
    norm = nlp.normalizar_nlp(texto)
    est = nlp.estadisticas_documento(texto, norm)
    st.caption(f"Documento: **{d['nombre']}**" + (" (ejemplo del conjunto del equipo)" if es_ejemplo else ""))
    c = st.columns(4)
    c[0].metric("Palabras", est["palabras"])
    c[1].metric("Palabras relevantes", est["terminos_normalizados"],
                help="Palabras que quedan al quitar artículos, preposiciones y otras palabras vacías.")
    c[2].metric("Palabras distintas", est["terminos_unicos"])
    c[3].metric("Caracteres", est["caracteres"])
    if est["terminos_normalizados"] < modelo.MIN_TOKENS:
        st.warning("El texto es muy corto o no se extrajo bien; los resultados no serán representativos.")

    t_res, t_kw, t_nube, t_ent, t_tema, t_clf, t_corp = st.tabs(
        ["Resumen", "Palabras clave", "Nube de palabras", "Entidades", "Temas", "Clasificación",
         "Conjunto del equipo"])

    with t_res:
        st.markdown("**Resumen automático (extractivo)**")
        st.caption("El programa elige las oraciones más importantes del propio documento. No escribe "
                   "texto nuevo, así que puede arrastrar errores del OCR. Conviene leerlo junto al original.")
        n_or = st.slider("Número de oraciones del resumen", 2, 6, 3, key="n_or")
        resumen = nlp.resumen_extractivo(texto, n=n_or)
        st.write(resumen if resumen.strip() else "No hay suficiente texto para resumir.")
        registrar_tabla(f"resumen_{base}.csv", pd.DataFrame({"archivo": [d["nombre"]], "resumen": [resumen]}))

    with t_kw:
        n = st.slider("Cantidad de palabras", 5, 25, 12, key="n_kw")
        a, b = st.columns(2)
        with a:
            st.markdown("**Más repetidas**")
            st.caption("Palabras relevantes que más veces aparecen en el documento.")
            frec = nlp.frecuencia_terminos([norm], n)[["termino", "frecuencia"]]
            if frec.empty:
                st.info("No hay palabras para mostrar.")
            else:
                barras_h(frec, "frecuencia", "termino", "Veces que aparece", "Apariciones")
                registrar_tabla(f"frecuencia_{base}.csv", frec)
        with b:
            st.markdown("**Más distintivas de este documento**")
            df_ref = estado["df"]
            if df_ref is None:
                st.info("Se necesita el conjunto de datos para esta comparación.")
            else:
                st.caption("Palabras importantes aquí y poco comunes en los demás documentos del conjunto "
                           "del equipo (técnica TF-IDF). Ayudan a ver de qué trata específicamente.")
                ref = df_ref[df_ref["archivo"] != d["nombre"]]["texto_norm"]
                kw = nlp.palabras_distintivas(norm, ref, n)
                if kw.empty:
                    st.info("No hay palabras para mostrar.")
                else:
                    barras_h(kw, "peso", "termino", "Importancia relativa", "Peso TF-IDF", color=GRIS)
                    registrar_tabla(f"palabras_clave_{base}.csv", kw)

    with t_nube:
        st.caption("Cuanto más grande aparece una palabra, más veces se repite en el documento.")
        frecs = nlp.frecuencia_terminos([norm], 60)
        if len(frecs) < 3:
            st.info("No hay suficientes palabras para dibujar la nube.")
        else:
            nube_palabras(dict(zip(frecs["termino"], frecs["frecuencia"])))

    with t_ent:
        st.markdown("**Entidades encontradas en el texto**")
        st.caption("Se buscan instituciones, lugares, fechas y cifras con listas y patrones de escritura. "
                   "No es un modelo entrenado: puede omitir entidades que no estén en la lista o escribirse "
                   "distinto por errores del OCR.")
        ents = nlp.extraer_entidades(texto)
        cols = st.columns(2)
        for i, (nombre, tabla) in enumerate(ents.items()):
            with cols[i % 2]:
                st.markdown(f"*{nombre}*")
                if tabla.empty:
                    st.write("No se encontraron.")
                else:
                    st.dataframe(tabla, hide_index=True, width="stretch")
        todo = pd.concat([t.rename(columns={t.columns[0]: "entidad"}).assign(tipo=n)
                          for n, t in ents.items() if not t.empty], ignore_index=True) \
            if any(not t.empty for t in ents.values()) else pd.DataFrame()
        if not todo.empty:
            registrar_tabla(f"entidades_{base}.csv", todo[["tipo", "entidad", "veces"]])

    with t_tema:
        st.markdown("**Temas mencionados**")
        st.caption("Cuenta cuántas palabras de cada grupo temático aparecen en el documento (por ejemplo, "
                   "\"lluvia\", \"temperatura\" para Clima). Es un conteo con listas de palabras, no un modelo; "
                   "sirve como orientación general.")
        temas = nlp.temas_mencionados(texto)
        if temas["menciones"].sum() == 0:
            st.info("No se encontraron palabras de los grupos temáticos.")
        else:
            barras_h(temas, "menciones", "tema", "Menciones por tema", "Palabras encontradas")
            registrar_tabla(f"temas_{base}.csv", temas)

    with t_clf:
        r = clasificar(texto)
        if r["error"]:
            st.warning(r["error"])
        else:
            mostrar_resultado_simple(r)
            st.caption("Más detalle en la sección **Clasificación**.")

    with t_corp:
        st.caption("Datos del conjunto de documentos reunido por el equipo (no del documento cargado). "
                   "Sirve para comparar.")
        df = estado["df"]
        if df is None:
            st.error(estado["err_df"])
        else:
            clase = st.radio("Documentos incluidos", ["Todos", "El Niño", "No El Niño"], horizontal=True)
            sub = df[df["con_texto"]]
            if clase != "Todos":
                sub = sub[sub["etiqueta_nombre"] == clase]
            st.write(f"{len(sub)} documentos con texto.")
            bloque_terminos(sub["texto_norm"], "corpus", clase.lower().replace(" ", "_"))


def bloque_terminos(textos_norm, clave: str, etiqueta_tabla: str):
    n = st.slider("Número de términos", 5, 30, 15, key=f"n_{clave}")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Frecuencia de términos**")
        frec = nlp.frecuencia_terminos(textos_norm, n)
        if not frec.empty:
            barras_h(frec, "frecuencia", "termino", "Frecuencia absoluta", "Apariciones")
            st.dataframe(frec, hide_index=True, width="stretch")
            registrar_tabla(f"frecuencia_terminos_{etiqueta_tabla}.csv", frec)
    with c2:
        st.markdown("**Palabras clave (TF-IDF promedio)**")
        kw = nlp.palabras_clave_tfidf(textos_norm, n)
        if not kw.empty:
            barras_h(kw, "tfidf_promedio", "termino", "TF-IDF promedio", "TF-IDF promedio", color=GRIS)
            st.dataframe(kw.round(4), hide_index=True, width="stretch")
            registrar_tabla(f"palabras_clave_{etiqueta_tabla}.csv", kw)


# ============================================================================
# Clasificación
# ============================================================================
def mostrar_resultado_simple(r: dict):
    """Resultado en lenguaje sencillo para cualquier persona."""
    clase = r["clase"]
    p = r["prob_nino"] if clase == 1 else 1 - r["prob_nino"]
    acuerdo = r["votos_nino"] if clase == 1 else r["n_modelos"] - r["votos_nino"]
    if clase == 1:
        st.success("**Este documento parece tratar sobre El Fenómeno El Niño.**")
    else:
        st.info("**Este documento parece NO tratar sobre El Fenómeno El Niño.**")
    nivel = "alta" if p >= 0.75 and acuerdo == r["n_modelos"] else "media" if p >= 0.6 else "baja"
    c = st.columns(3)
    c[0].metric("Seguridad del resultado", nivel.capitalize(),
                help="Orientativa: depende de la probabilidad y de si los 5 modelos coinciden.")
    c[1].metric("Probabilidad estimada", f"{p:.0%}")
    c[2].metric("Modelos que coinciden", f"{acuerdo} de {r['n_modelos']}")
    if r["cobertura_vocabulario"] < 0.5:
        st.warning("El documento usa pocas palabras que los modelos conocen; el resultado es poco fiable.")
    if r["n_terminos"] < 30:
        st.warning("El texto es corto; el resultado puede cambiar fácilmente.")
    if 0.4 <= r["prob_nino"] <= 0.6:
        st.warning("El resultado está muy cerca del punto de duda (50 %); no es concluyente.")


def mostrar_validacion():
    metricas = cargar_resultado("metricas_clasificador.csv")
    if metricas is None:
        st.info("No hay métricas de validación disponibles en la carpeta resultados/.")
        return
    st.markdown(
        "Estas cifras salen de probar cada modelo con documentos que **no vio al entrenarse** (validación "
        "cruzada de 5 particiones). Corresponden a los modelos individuales; **el promedio de los 5 modelos "
        "que usa esta aplicación no fue evaluado como conjunto.**"
    )
    st.dataframe(metricas.round(3).reset_index(names="clase / promedio"), hide_index=True, width="stretch")
    cm = cargar_resultado("matriz_confusion.csv")
    if cm is not None:
        st.markdown("**Matriz de confusión** (filas: clase real; columnas: clase predicha)")
        st.dataframe(cm.reset_index(names=""), hide_index=True, width="stretch")
        registrar_tabla("matriz_confusion_validacion.csv", cm.reset_index(names="real"))
    registrar_tabla("metricas_validacion.csv", metricas.reset_index(names="clase"))
    st.caption(
        "Regeneradas con `scripts/generar_resultados.py` a partir del CSV y de los modelos entregados. Con "
        "47 documentos y clases desbalanceadas son exploratorias; la clase está asociada a institución y "
        "formato, así que podrían no generalizar a fuentes nuevas."
    )


def pagina_clasificacion():
    st.title("Clasificación de documentos")
    st.markdown("Indica si un documento **trata o no sobre El Fenómeno El Niño**, usando los modelos "
                "entrenados por el equipo.")
    conjunto = estado["conjunto"]
    if conjunto is None:
        st.error(f"No se pueden realizar predicciones. {estado['err_mod']}")
        return
    if conjunto.aviso_version:
        st.warning(conjunto.aviso_version)

    docs = [d for d in st.session_state.get("docs", []) if d["estado"] == "Procesado"]
    if not docs:
        st.info("Carga documentos en **Carga y extracción** para clasificarlos.")
    else:
        filas, simple = [], []
        for d in docs:
            r = clasificar(d["texto"])
            fila = {"archivo": d["nombre"], "error": r.get("error", "")}
            if not r["error"]:
                clase = r["clase"]
                p = r["prob_nino"] if clase == 1 else 1 - r["prob_nino"]
                fila.update({
                    "clase_predicha": NOMBRE_CLASE[clase], "prob_el_nino": round(r["prob_nino"], 4),
                    "prob_clase_predicha": round(p, 4),
                    "modelos_a_favor_el_nino": f'{r["votos_nino"]}/{r["n_modelos"]}',
                    "cobertura_vocabulario": round(r["cobertura_vocabulario"], 3),
                    "terminos_normalizados": r["n_terminos"], "metodo_extraccion": d["res"].metodo})
                for nom, pm in r["por_modelo"]:
                    fila[f"prob_{nom.replace(' ', '_')}"] = round(pm, 4)
                simple.append({"Archivo": d["nombre"], "Resultado": NOMBRE_CLASE[clase],
                               "Probabilidad": f"{p:.0%}"})
            else:
                simple.append({"Archivo": d["nombre"], "Resultado": "No se pudo clasificar",
                               "Probabilidad": r["error"]})
            filas.append(fila)
        st.dataframe(pd.DataFrame(simple), hide_index=True, width="stretch")
        res_df = pd.DataFrame(filas)
        if (res_df["error"] == "").any():
            registrar_tabla("clasificacion_documentos.csv", res_df)

        k = st.selectbox("Ver detalle de", range(len(docs)), format_func=lambda i: docs[i]["nombre"],
                         key="clf_doc")
        d, r = docs[k], clasificar(docs[k]["texto"])
        if r["error"]:
            st.error(r["error"])
        else:
            mostrar_resultado_simple(r)
            for a in d["res"].advertencias:
                st.warning(f"Extracción: {a}")
            st.markdown("**Palabras que más influyeron en la decisión**")
            st.caption("Son las palabras del documento que más pesan para el modelo. Muestran cómo decidió "
                       "el modelo; no prueban que una cosa cause la otra.")
            a, b = st.columns(2)
            for col, clave, titulo in ((a, "hacia_nino", "Apuntan a: trata de El Niño"),
                                       (b, "hacia_no_nino", "Apuntan a: no trata de El Niño")):
                col.markdown(f"*{titulo}*")
                datos = r["contribuciones"][clave]
                if datos:
                    col.dataframe(pd.DataFrame(datos, columns=["palabra", "influencia"]).round(3),
                                  hide_index=True, width="stretch")
                else:
                    col.write("Ninguna destacada.")
            with st.expander("Detalle técnico"):
                st.write(f"Cobertura del vocabulario: {r['cobertura_vocabulario']:.0%} de las palabras "
                         f"relevantes del documento están en el vocabulario de los modelos. La probabilidad "
                         "viene de una regresión logística y no está calibrada.")
                pm = pd.DataFrame(r["por_modelo"], columns=["modelo", "prob_nino"])
                fig, ax = plt.subplots(figsize=(6, 2.6))
                ax.bar(pm["modelo"], pm["prob_nino"], color=ACENTO)
                ax.axhline(0.5, color=ROJO, linestyle="--", linewidth=1)
                ax.set_ylim(0, 1); ax.set_ylabel("Prob. El Niño")
                ax.set_title("Probabilidad de El Niño según cada modelo", fontsize=10, loc="left")
                ax.spines[["top", "right"]].set_visible(False); ax.tick_params(labelsize=8)
                fig.tight_layout(); st.pyplot(fig); plt.close(fig)
    with st.expander("Para el informe del equipo: cómo se evaluaron los modelos"):
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
