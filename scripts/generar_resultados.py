"""Regenera las métricas de validación cruzada a partir de los datos y modelos del equipo.

Reproduce el procedimiento del notebook (Persona 2): StratifiedGroupKFold(5, shuffle, random_state=42)
sobre los documentos con texto válido. Comprueba que cada partición coincide con el modelo entregado
(mismo vocabulario TF-IDF) y que las predicciones fuera de entrenamiento salen de ESOS modelos .joblib.
Uso (desde la raíz del proyecto):  python scripts/generar_resultados.py
"""
import json
import sys
import warnings
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedGroupKFold

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))
import nlp  # noqa: E402

warnings.filterwarnings("ignore")
df = pd.read_csv(RAIZ / "data" / "dataset_ocr_utilizado.csv", keep_default_na=False)
df["texto_limpio"] = df.texto_ocr.map(nlp.normalizar_nlp)
v = df[(df.error == "") & (df.texto_limpio.str.strip() != "")].reset_index(drop=True)

modelos = [joblib.load(RAIZ / "models" / f"modelo_pliegue_{i}.joblib") for i in range(1, 6)]
splits = list(StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
              .split(v.texto_limpio, v.etiqueta, v.grupo))

pred = pd.Series(index=v.index, dtype=int)
prob = pd.Series(index=v.index, dtype=float)
pliegue = pd.Series(index=v.index, dtype=int)
coinciden = []
for i, ((tr, te), m) in enumerate(zip(splits, modelos), 1):
    nuevo = clone(m).fit(v.texto_limpio.iloc[tr], v.etiqueta.iloc[tr])
    coinciden.append(nuevo.named_steps["tfidf"].vocabulary_ == m.named_steps["tfidf"].vocabulary_)
    p = m.predict_proba(v.texto_limpio.iloc[te])[:, 1]
    prob.iloc[te], pliegue.iloc[te] = p, i
    pred.iloc[te] = (p >= 0.5).astype(int)
assert all(coinciden), "Alguna partición no coincide con el modelo entregado"
pred = pred.astype(int)

nombres = ["No El Niño", "El Niño"]
rep = classification_report(v.etiqueta, pred, target_names=nombres, output_dict=True, zero_division=0)
pd.DataFrame(rep).T.to_csv(RAIZ / "resultados" / "metricas_clasificador.csv", encoding="utf-8-sig")
pd.DataFrame(confusion_matrix(v.etiqueta, pred, labels=[0, 1]),
             index=["real: " + n for n in nombres], columns=["pred: " + n for n in nombres]
             ).to_csv(RAIZ / "resultados" / "matriz_confusion.csv", encoding="utf-8-sig")
out = v[["archivo", "institucion", "grupo", "etiqueta"]].copy()
out["pliegue_prueba"], out["prob_nino"], out["prediccion_cv"] = pliegue.astype(int), prob.round(4), pred
out.to_csv(RAIZ / "resultados" / "predicciones_validacion.csv", index=False, encoding="utf-8-sig")
(RAIZ / "resultados" / "verificacion.json").write_text(json.dumps({
    "documentos_validos": len(v), "documentos_excluidos_sin_texto": int(len(df) - len(v)),
    "particiones_coinciden_con_modelos": coinciden, "sklearn_usado": sklearn.__version__,
    "metodo": "StratifiedGroupKFold(5, shuffle=True, random_state=42); predicciones de los .joblib entregados",
}, ensure_ascii=False, indent=1), encoding="utf-8")
print(classification_report(v.etiqueta, pred, target_names=nombres, zero_division=0))
print(confusion_matrix(v.etiqueta, pred, labels=[0, 1]))
