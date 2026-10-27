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
