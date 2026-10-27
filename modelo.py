"""Clasificador El Niño / No El Niño.

Usa los 5 modelos de la validación cruzada entregados por el equipo (models/modelo_pliegue_*.joblib).
Cada modelo es un Pipeline TF-IDF + regresión logística. La predicción final promedia las
probabilidades de los 5 modelos. Ese promedio NO fue evaluado como conjunto: las métricas del
equipo corresponden a cada modelo sobre su propio pliegue.

Cargar un .joblib ejecuta código: solo se cargan archivos de origen conocido (el equipo).
"""
from __future__ import annotations

import re
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

import nlp

DIR_MODELOS = Path(__file__).resolve().parent / "models"
PATRON = "modelo_pliegue_*.joblib"
MIN_TOKENS = 10  # con menos términos tras la normalización la predicción no es fiable
ETIQUETAS = {0: "No El Niño", 1: "El Niño"}


@dataclass
class ConjuntoModelos:
    pipelines: list
    nombres: list[str]
    aviso_version: str = ""
    vocabulario: frozenset = frozenset()
    _nombres_terminos: dict = field(default_factory=dict)

    def terminos(self, i: int):
        if i not in self._nombres_terminos:
            self._nombres_terminos[i] = self.pipelines[i].named_steps["tfidf"].get_feature_names_out()
        return self._nombres_terminos[i]


def cargar_modelos() -> tuple[ConjuntoModelos | None, str]:
    """Devuelve (conjunto, error). Si faltan o fallan los archivos no inventa nada."""
    archivos = sorted(DIR_MODELOS.glob(PATRON),
                      key=lambda p: int(re.search(r"(\d+)", p.stem).group(1)))
    if not archivos:
        return None, ("No se encontraron modelos en la carpeta models/ "
                      "(se esperan archivos modelo_pliegue_1.joblib, ... modelo_pliegue_5.joblib).")
    pipelines, nombres, avisos = [], [], set()
    for ruta in archivos:
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                p = joblib.load(ruta)
            for x in w:
                original = getattr(x.message, "original_sklearn_version", None)
                actual = getattr(x.message, "current_sklearn_version", None)
                if original and actual:
                    avisos.add(f"Los modelos se entrenaron con scikit-learn {original} y está "
                               f"instalada la {actual}. Instala la {original} para obtener "
                               "resultados idénticos a los del equipo.")
            if not hasattr(p, "named_steps") or "tfidf" not in p.named_steps \
                    or "clasificador" not in p.named_steps:
                return None, f"{ruta.name} no tiene la estructura esperada (tfidf + clasificador)."
            if list(p.named_steps["clasificador"].classes_) != [0, 1]:
                return None, f"{ruta.name} no tiene las clases esperadas (0 y 1)."
            pipelines.append(p)
            nombres.append(ruta.stem.replace("modelo_", "").replace("_", " "))
        except Exception as e:
            return None, f"No se pudo cargar {ruta.name}: {e}"
    vocab = set()
    for p in pipelines:
        vocab.update(t for t in p.named_steps["tfidf"].vocabulary_ if " " not in t)
    return ConjuntoModelos(pipelines, nombres, " ".join(sorted(avisos)), frozenset(vocab)), ""


def predecir(conjunto: ConjuntoModelos, texto: str) -> dict:
    """Clasifica un texto extraído. Devuelve un dict con 'error' si no se puede predecir."""
    norm = nlp.normalizar_nlp(texto)
    tokens = norm.split()
    if len(tokens) < MIN_TOKENS:
        return {"error": f"El texto es demasiado corto para clasificarlo ({len(tokens)} términos "
                         f"tras la normalización; se requieren al menos {MIN_TOKENS})."}

    probs = [float(p.predict_proba([norm])[0][1]) for p in conjunto.pipelines]
    media = float(np.mean(probs))
    votos = sum(p >= 0.5 for p in probs)
    cobertura = sum(t in conjunto.vocabulario for t in tokens) / len(tokens)
    return {
        "error": "",
        "clase": int(media >= 0.5),
        "prob_nino": media,
        "votos_nino": votos,
        "n_modelos": len(probs),
        "por_modelo": list(zip(conjunto.nombres, probs)),
        "cobertura_vocabulario": cobertura,
        "n_terminos": len(tokens),
        "contribuciones": _contribuciones(conjunto, norm),
    }


def _contribuciones(conjunto: ConjuntoModelos, norm: str, n: int = 6) -> dict:
    """Términos que más empujan la decisión (coeficiente x peso TF-IDF), promediados en los modelos."""
    acum: dict[str, float] = defaultdict(float)
    k = len(conjunto.pipelines)
    for i, p in enumerate(conjunto.pipelines):
        X = p.named_steps["tfidf"].transform([norm])
        coef = p.named_steps["clasificador"].coef_[0]
        nombres = conjunto.terminos(i)
        for idx, val in zip(X.indices, X.data):
            acum[nombres[idx]] += float(val * coef[idx]) / k
    orden = sorted(acum.items(), key=lambda kv: kv[1])
    return {
        "hacia_nino": [(t, v) for t, v in orden[::-1][:n] if v > 0],
        "hacia_no_nino": [(t, v) for t, v in orden[:n] if v < 0],
    }
