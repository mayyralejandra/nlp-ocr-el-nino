import json, re, unicodedata
from pathlib import Path
import joblib

BASE = Path(__file__).resolve().parent
with open(BASE / 'stopwords_espanol.json', encoding='utf-8') as archivo:
    STOPWORDS = set(json.load(archivo))

def limpiar_texto(texto):
    texto = unicodedata.normalize('NFKC', str(texto)).lower()
    texto = re.sub(r'[^a-záéíóúüñ\s]', ' ', texto)
    return ' '.join(palabra for palabra in texto.split() if palabra not in STOPWORDS and len(palabra) > 2)

def predecir(texto):
    limpio = limpiar_texto(texto)
    if not limpio:
        raise ValueError('El texto está vacío después de la limpieza.')
    modelo = joblib.load(BASE / 'modelo_clasificador.joblib')
    clase = int(modelo.predict([limpio])[0])
    return {'etiqueta': clase, 'categoria': 'El Niño' if clase == 1 else 'No El Niño'}
