"""Procesamiento NLP: normalización del clasificador y análisis descriptivo del corpus.

La función `normalizar_nlp` es la MISMA `normalizar` del notebook de la Persona 2
(NFKC, minúsculas, solo letras con tilde/ñ, sin palabras vacías, longitud > 2). Las palabras vacías
son las de NLTK (español, 313) más {'ser','asi','ademas','tambien','peru','perú'}, guardadas en
data/stopwords_espanol.json para no depender de descargas de NLTK. Se comprobó que reentrenando con
esta normalización se obtienen vocabularios y coeficientes idénticos a los .joblib entregados.
"""
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

RAIZ = Path(__file__).resolve().parent
RUTA_STOPWORDS = RAIZ / "data" / "stopwords_espanol.json"


@lru_cache(maxsize=1)
def stopwords() -> frozenset:
    return frozenset(json.loads(RUTA_STOPWORDS.read_text(encoding="utf-8")))


def normalizar_nlp(texto) -> str:
    """Normalización idéntica a la usada al entrenar los modelos."""
    sw = stopwords()
    s = unicodedata.normalize("NFKC", str(texto)).lower()
    s = re.sub(r"[^a-záéíóúüñ\s]", " ", s)
    return " ".join(w for w in s.split() if w not in sw and len(w) > 2)


# ----------------------------------------------------------------------------
# Análisis descriptivo (se calcula explícitamente sobre los textos que se le pasen)
# ----------------------------------------------------------------------------
def frecuencia_terminos(textos_norm, n: int = 20) -> pd.DataFrame:
    """Frecuencia absoluta y nº de documentos donde aparece cada término (ya normalizados)."""
    total, docs = Counter(), Counter()
    for t in textos_norm:
        toks = t.split()
        total.update(toks)
        docs.update(set(toks))
    filas = [(w, c, docs[w]) for w, c in total.most_common(n)]
    return pd.DataFrame(filas, columns=["termino", "frecuencia", "documentos"])


def palabras_clave_tfidf(textos_norm, n: int = 20) -> pd.DataFrame:
    """TF-IDF promedio (unigramas y bigramas, max 5000), igual que el notebook del equipo."""
    textos = [t for t in textos_norm if t.strip()]
    if not textos:
        return pd.DataFrame(columns=["termino", "tfidf_promedio"])
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)
    X = vec.fit_transform(textos)
    df = pd.DataFrame({"termino": vec.get_feature_names_out(),
                       "tfidf_promedio": np.asarray(X.mean(axis=0)).ravel()})
    return df.sort_values("tfidf_promedio", ascending=False).head(n).reset_index(drop=True)


def resumen_extractivo(texto: str, n: int = 3) -> str:
    """Resumen extractivo del notebook: elige las n oraciones con mayor suma TF-IDF.
    Selecciona oraciones existentes, por lo que puede heredar errores del OCR."""
    texto = str(texto)
    oraciones = [re.sub(r"\s+", " ", x).strip()
                 for x in re.split(r"(?<=[.!?])\s+|\n+", texto) if len(x.strip()) >= 40]
    if len(oraciones) <= n:
        return " ".join(oraciones) if oraciones else texto[:800]
    limpias = [normalizar_nlp(x) for x in oraciones]
    if not any(limpias):
        return " ".join(oraciones[:n])
    try:
        mat = TfidfVectorizer().fit_transform(limpias)
        scores = np.asarray(mat.sum(axis=1)).ravel()
        idx = sorted(np.argsort(scores)[-n:])
        return " ".join(oraciones[i] for i in idx)
    except ValueError:
        return " ".join(oraciones[:n])


def estadisticas_documento(texto: str, texto_norm: str) -> dict:
    toks = texto_norm.split()
    return {
        "palabras": len(re.findall(r"\w+", texto or "", flags=re.UNICODE)),
        "terminos_normalizados": len(toks),
        "terminos_unicos": len(set(toks)),
        "caracteres": len(texto or ""),
    }


# ----------------------------------------------------------------------------
# Extracción de entidades y temas (basada en reglas y listas; NO es un modelo entrenado)
# ----------------------------------------------------------------------------
INSTITUCIONES = ["ENFEN", "INDECI", "SENAMHI", "IMARPE", "COEN", "MINSA", "MINAM", "MIDAGRI", "PRODUCE",
                 "IGP", "ANA", "CENEPRED", "PCM", "MEF", "MINEDU", "INGEMMET", "ULADECH", "CIPLIMA",
                 "DHN", "EsSalud", "SUNASS", "MTC", "MVCS", "CONIDA"]
_PAT_INST = re.compile(
    r"\b(?:Municipalidad\s+(?:Distrital\s+|Provincial\s+Metropolitana\s+|Provincial\s+|Metropolitana\s+)?de\s+"
    r"[A-ZÁÉÍÓÚÑ][\wáéíóúñ]+(?:\s+(?:de\s+|del\s+)?[A-ZÁÉÍÓÚÑ][\wáéíóúñ]+){0,2}"
    r"|Ministerio\s+de(?:l|\s+la)?\s+[A-ZÁÉÍÓÚÑ][\wáéíóúñ]+(?:\s+y\s+[A-ZÁÉÍÓÚÑ][\wáéíóúñ]+)?"
    r"|Instituto\s+(?:Nacional|Geofísico|del\s+Mar)\s+(?:de\s+)?[A-ZÁÉÍÓÚÑ][\wáéíóúñ]+(?:\s+[A-ZÁÉÍÓÚÑ][\wáéíóúñ]+)?"
    r"|Gobierno\s+Regional\s+de\s+[A-ZÁÉÍÓÚÑ][\wáéíóúñ]+)")
LUGARES = ["Amazonas", "Áncash", "Ancash", "Apurímac", "Arequipa", "Ayacucho", "Cajamarca", "Callao", "Cusco",
           "Huancavelica", "Huánuco", "Ica", "Junín", "La Libertad", "Lambayeque", "Lima", "Loreto",
           "Madre de Dios", "Moquegua", "Pasco", "Piura", "Puno", "San Martín", "Tacna", "Tumbes", "Ucayali",
           "Chimbote", "Trujillo", "Chiclayo", "Paita", "Sullana", "Chorrillos", "Comas", "Carabayllo",
           "Los Olivos", "Surco", "Ancón", "Rímac", "Costa norte", "Costa central", "Costa sur"]
_MESES = "enero|febrero|marzo|abril|mayo|junio|julio|agosto|setiembre|septiembre|octubre|noviembre|diciembre"
_PAT_FECHA = re.compile(rf"\b\d{{1,2}}\s+de\s+(?:{_MESES})(?:\s+(?:de|del)\s+\d{{4}})?|\b\d{{1,2}}/\d{{1,2}}/\d{{2,4}}\b",
                        re.IGNORECASE)
_PAT_CIFRA = re.compile(r"\b\d+(?:[.,]\d+)?\s?(?:°\s?C|ºC|°|%|mm|cm|m³/s|km|metros|grados|días|horas|personas|familias)\b",
                        re.IGNORECASE)
_PAT_NUMERO = re.compile(r"\b(?:N[°º]\.?|N\.º|Nro\.?)\s?\d{1,4}(?:[-–]\d{2,4})?", re.IGNORECASE)

# Temas: lista de palabras (ya sin tildes) que se buscan en el texto. Es un conteo de coincidencias.
TEMAS = {
    "Clima y lluvias": ["lluvia", "lluvias", "precipitaciones", "precipitacion", "temperatura", "temperaturas",
                        "clima", "climatico", "senamhi", "pronostico", "tormenta", "oleaje"],
    "Mar y pesca": ["mar", "marino", "pesca", "pesquera", "anchoveta", "oceano", "costero", "costera", "imarpe"],
    "Fenómeno El Niño": ["niño", "nino", "enfen", "anomalia", "anomalias", "alerta", "advertencia", "magnitud"],
    "Emergencias y riesgo": ["emergencia", "peligro", "riesgo", "inundacion", "inundaciones", "huaico", "huaicos",
                             "deslizamiento", "damnificados", "indeci", "defensa", "evacuacion", "desastre"],
    "Salud": ["salud", "minsa", "hospital", "enfermedades", "dengue", "vacunacion", "atencion"],
    "Agricultura y ambiente": ["agricultura", "cultivos", "agricola", "sequia", "agua", "ambiente", "cosecha"],
}


def _sin_tildes(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn")


def _unicos(seq):
    vistos, out = set(), []
    for x in seq:
        k = re.sub(r"\s+", " ", x).strip()
        if k and k.lower() not in vistos:
            vistos.add(k.lower())
            out.append(k)
    return out


def extraer_entidades(texto: str) -> dict[str, pd.DataFrame]:
    """Busca instituciones, lugares, fechas y cifras con listas y expresiones regulares.
    No es un modelo estadístico: puede omitir entidades o equivocarse con texto de OCR defectuoso."""
    t = unicodedata.normalize("NFKC", str(texto or ""))
    plano = re.sub(r"\s+", " ", t)
    res: dict[str, pd.DataFrame] = {}

    def tabla(items, col):
        cont = Counter(re.sub(r"\s+", " ", i).strip() for i in items)
        return pd.DataFrame(sorted(cont.items(), key=lambda kv: -kv[1]), columns=[col, "veces"])

    inst = [m for sigla in INSTITUCIONES for m in re.findall(rf"\b{sigla}\b", plano)]
    inst += _PAT_INST.findall(plano)
    res["Instituciones"] = tabla(inst, "institución")
    lug = [l for l in LUGARES for _ in re.findall(rf"\b{re.escape(l)}\b", plano)]
    res["Lugares"] = tabla(lug, "lugar")
    res["Fechas"] = tabla(_PAT_FECHA.findall(plano), "fecha")
    res["Cifras y medidas"] = tabla(_PAT_CIFRA.findall(plano) + _PAT_NUMERO.findall(plano), "cifra")
    return res


def temas_mencionados(texto: str) -> pd.DataFrame:
    """Cuenta cuántas veces aparecen, en el texto, las palabras de cada lista temática."""
    toks = _sin_tildes(str(texto or "").lower())
    toks = re.findall(r"[a-zñ]+", toks)
    cont = Counter(toks)
    filas = []
    for tema, palabras in TEMAS.items():
        n = sum(cont.get(_sin_tildes(p), 0) for p in palabras)
        filas.append((tema, n))
    df = pd.DataFrame(filas, columns=["tema", "menciones"])
    return df.sort_values("menciones", ascending=False).reset_index(drop=True)


def palabras_distintivas(texto_norm: str, corpus_norm, n: int = 15) -> pd.DataFrame:
    """Términos con mayor peso TF-IDF en UN documento, comparado con el conjunto de referencia.
    Un término pesa más si es frecuente en el documento y poco común en el resto de documentos."""
    refs = [t for t in corpus_norm if t.strip()]
    if not texto_norm.strip():
        return pd.DataFrame(columns=["termino", "peso"])
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    X = vec.fit_transform(refs + [texto_norm])
    fila = X[X.shape[0] - 1].toarray().ravel()
    orden = np.argsort(fila)[::-1][:n]
    nombres = vec.get_feature_names_out()
    return pd.DataFrame({"termino": nombres[orden], "peso": fila[orden]}).query("peso > 0").reset_index(drop=True)
