"""Limpieza del texto extraído.

Es una limpieza CONSERVADORA: corrige espacios, saltos de línea y caracteres corruptos, pero
conserva tildes, mayúsculas, números y signos de puntuación. No elimina palabras.

NO es la normalización del clasificador (minúsculas, sin stopwords, solo letras). Esa
función se incorporará en la etapa del modelo, copiada del notebook de la Persona 2 para que
coincida exactamente con la usada al entrenar.
"""
from __future__ import annotations

import re
import unicodedata

# Caracteres de control, BOM, espacios de ancho cero y el carácter de reemplazo '�'
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\ufeff\u200b-\u200d\u2060\ufffd]")
_GUION_CORTE = re.compile(r"(?<=[a-záéíóúüñ])-[ \t]*\n[ \t]*(?=[a-záéíóúüñ])")
_SALTO_SIMPLE = re.compile(r"(?<!\n)\n(?!\n)")


def limpiar_texto(texto: str, unir_lineas: bool = True) -> str:
    """Devuelve el texto con espacios, saltos de línea y caracteres problemáticos corregidos.

    unir_lineas=True convierte los saltos de línea simples en espacios (las líneas de un OCR
    suelen cortarse arbitrariamente) y conserva los párrafos separados por línea en blanco.
    """
    if not isinstance(texto, str) or not texto.strip():
        return ""

    t = unicodedata.normalize("NFKC", texto)          # unifica formas Unicode; \xa0 -> espacio
    t = t.replace("\u00ad", "")                        # guion blando invisible
    t = t.replace("\r\n", "\n").replace("\r", "\n").replace("\f", "\n\n")
    t = _CONTROL.sub("", t)
    t = t.replace("\t", " ")
    t = _GUION_CORTE.sub("", t)                        # "comuni-\ncado" -> "comunicado"
    t = re.sub(r"[ ]{2,}", " ", t)
    t = re.sub(r" *\n *", "\n", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    if unir_lineas:
        t = _SALTO_SIMPLE.sub(" ", t)
        t = re.sub(r"[ ]{2,}", " ", t)
    return t.strip()


def contar_palabras(texto: str) -> int:
    """Número de palabras (secuencias de letras/dígitos)."""
    return len(re.findall(r"\w+", texto or "", flags=re.UNICODE))
