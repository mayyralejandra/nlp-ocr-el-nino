"""Extracción de texto de imágenes y PDF.

Este módulo SOLO extrae texto; la limpieza está en limpieza.py.
Trabaja siempre con los bytes en memoria: nunca escribe ni modifica el archivo original.
"""
from __future__ import annotations

import io
import os
import shutil
import sys
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import PurePath

import pymupdf  # PyMuPDF: texto digital de PDF y renderizado de páginas (no requiere poppler)
import pytesseract
from PIL import Image, ImageOps

EXT_IMAGEN = {".jpg", ".jpeg", ".png"}
EXT_PDF = {".pdf"}

MAX_PAGINAS_PDF = 10         # protege la memoria en Streamlit Community Cloud
MIN_CARS_TEXTO_DIGITAL = 30  # menos caracteres que esto => se considera página escaneada
DPI_OCR = 250                # resolución al convertir páginas de PDF a imagen
LADO_MIN_IMAGEN = 1500       # imágenes más pequeñas se amplían antes del OCR

_RUTAS_WINDOWS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)


@dataclass
class ResultadoExtraccion:
    texto: str = ""
    metodo: str = ""                      # "OCR (Tesseract)", "Texto digital del PDF" o "Mixto"
    paginas_totales: int = 1
    paginas_procesadas: int = 1
    detalle_paginas: list[str] = field(default_factory=list)
    advertencias: list[str] = field(default_factory=list)
    error: str = ""                       # no vacío => no se pudo procesar

    @property
    def exito(self) -> bool:
        return not self.error


# ----------------------------------------------------------------------------
# Configuración de Tesseract
# ----------------------------------------------------------------------------
@lru_cache(maxsize=1)
def configurar_tesseract() -> str:
    """Localiza el ejecutable de Tesseract. Devuelve su versión, o '' si no está instalado."""
    ruta = os.environ.get("TESSERACT_CMD", "").strip()
    if not ruta and shutil.which("tesseract") is None and sys.platform.startswith("win"):
        ruta = next((r for r in _RUTAS_WINDOWS if os.path.exists(r)), "")
    if ruta:
        pytesseract.pytesseract.tesseract_cmd = ruta
    try:
        return str(pytesseract.get_tesseract_version())
    except Exception:
        return ""


@lru_cache(maxsize=1)
def idioma_ocr() -> tuple[str, str]:
    """Devuelve (idioma, aviso). Usa español si está instalado; si no, inglés con aviso."""
    try:
        idiomas = set(pytesseract.get_languages(config=""))
    except Exception:
        return "eng", "No se pudo consultar los idiomas de Tesseract; se usará 'eng'."
    if "spa" in idiomas:
        return "spa", ""
    if "eng" in idiomas:
        return "eng", ("El paquete de idioma español de Tesseract ('spa') no está instalado: "
                       "el OCR usa inglés y fallará con tildes y ñ. Instala 'spa' (ver README).")
    return "", "Tesseract no tiene idiomas instalados."


# ----------------------------------------------------------------------------
# OCR de imágenes
# ----------------------------------------------------------------------------
def preparar_imagen(img: Image.Image) -> Image.Image:
    """Mejora ligera para OCR: orientación EXIF, escala de grises, contraste y ampliación."""
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        fondo = Image.new("RGBA", img.size, (255, 255, 255, 255))  # transparencia -> blanco
        img = Image.alpha_composite(fondo, img)
    img = ImageOps.grayscale(img.convert("RGB"))
    img = ImageOps.autocontrast(img, cutoff=1)
    lado = max(img.size)
    if lado < LADO_MIN_IMAGEN:
        factor = LADO_MIN_IMAGEN / lado
        img = img.resize((int(img.width * factor), int(img.height * factor)), Image.LANCZOS)
    return img


def ocr_imagen(img: Image.Image) -> str:
    """Aplica Tesseract a una imagen PIL ya cargada."""
    lang, _ = idioma_ocr()
    return pytesseract.image_to_string(preparar_imagen(img), lang=lang or None)


# ----------------------------------------------------------------------------
# Extracción por tipo de archivo
# ----------------------------------------------------------------------------
def _extraer_imagen(datos: bytes) -> ResultadoExtraccion:
    res = ResultadoExtraccion(metodo="OCR (Tesseract)", detalle_paginas=["Imagen: OCR"])
    try:
        with Image.open(io.BytesIO(datos)) as img:
            img.load()
            res.texto = ocr_imagen(img)
    except pytesseract.TesseractNotFoundError:
        res.error = "Tesseract OCR no está instalado o no se encuentra en el PATH (ver README)."
    except (OSError, ValueError, Image.DecompressionBombError) as e:
        res.error = f"No se pudo abrir la imagen: archivo dañado o formato no válido ({e})."
    return res


def _extraer_pdf(datos: bytes) -> ResultadoExtraccion:
    res = ResultadoExtraccion()
    try:
        doc = pymupdf.open(stream=datos, filetype="pdf")
    except Exception as e:
        res.error = f"No se pudo abrir el PDF: archivo dañado o no válido ({e})."
        return res

    with doc:
        if doc.needs_pass:
            res.error = "El PDF está protegido con contraseña; no se puede procesar."
            return res
        total = len(doc)
        if total == 0:
            res.error = "El PDF no contiene páginas."
            return res

        res.paginas_totales = total
        res.paginas_procesadas = min(total, MAX_PAGINAS_PDF)
        if total > MAX_PAGINAS_PDF:
            res.advertencias.append(
                f"El PDF tiene {total} páginas; solo se procesaron las primeras {MAX_PAGINAS_PDF}."
            )

        partes: list[str] = []
        usos: set[str] = set()
        ocr_omitido = False
        for i in range(res.paginas_procesadas):
            pagina = doc[i]
            try:
                digital = pagina.get_text("text") or ""
            except Exception:
                digital = ""

            if len(digital.strip()) >= MIN_CARS_TEXTO_DIGITAL:
                partes.append(digital)
                usos.add("digital")
                res.detalle_paginas.append(f"Página {i + 1}: texto digital")
                continue

            # Página escaneada o sin capa de texto -> OCR
            if not configurar_tesseract():
                ocr_omitido = True
                res.detalle_paginas.append(f"Página {i + 1}: requiere OCR (Tesseract no disponible)")
                if digital.strip():
                    partes.append(digital)
                continue
            try:
                pix = pagina.get_pixmap(dpi=DPI_OCR, colorspace=pymupdf.csRGB, alpha=False)
                img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
                partes.append(ocr_imagen(img))
                usos.add("ocr")
                res.detalle_paginas.append(f"Página {i + 1}: OCR")
            except Exception as e:
                res.advertencias.append(f"Página {i + 1}: no se pudo aplicar OCR ({e}).")
                res.detalle_paginas.append(f"Página {i + 1}: error de OCR")

    res.texto = "\n\n".join(p for p in partes if p.strip())
    if usos == {"digital"}:
        res.metodo = "Texto digital del PDF"
    elif usos == {"ocr"}:
        res.metodo = "OCR (Tesseract)"
    elif usos:
        res.metodo = "Mixto (texto digital + OCR)"
    else:
        res.metodo = "Sin método aplicable"

    if ocr_omitido and not res.texto.strip():
        res.error = ("El PDF es escaneado y requiere OCR, pero Tesseract no está instalado "
                     "o no se encuentra en el PATH (ver README).")
    elif ocr_omitido:
        res.advertencias.append("Algunas páginas escaneadas se omitieron: Tesseract no está disponible.")
    return res


def extraer_texto(nombre: str, datos: bytes) -> ResultadoExtraccion:
    """Punto de entrada: decide el método según la extensión y nunca lanza excepciones."""
    ext = PurePath(nombre).suffix.lower()
    if not datos:
        return ResultadoExtraccion(error="El archivo está vacío (0 bytes).")
    try:
        if ext in EXT_IMAGEN:
            if not configurar_tesseract():
                return ResultadoExtraccion(
                    error="Tesseract OCR no está instalado o no se encuentra en el PATH (ver README)."
                )
            res = _extraer_imagen(datos)
        elif ext in EXT_PDF:
            res = _extraer_pdf(datos)
        else:
            return ResultadoExtraccion(error=f"Formato '{ext or 'desconocido'}' no admitido. Usa JPG, JPEG, PNG o PDF.")
    except Exception as e:  # último recurso: la app no debe caerse por un archivo problemático
        return ResultadoExtraccion(error=f"Error inesperado al procesar el archivo: {e}")

    if res.exito and ext in EXT_IMAGEN | EXT_PDF:
        _, aviso = idioma_ocr()
        if aviso and "OCR" in res.metodo:
            res.advertencias.append(aviso)
    return res
