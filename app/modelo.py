import joblib
import numpy as np
import pandas as pd

from .features import COLUNAS_FEATURES


class ModeloCNH:
    def __init__(self, caminho: str):
        obj = joblib.load(caminho)
        features = None
        encoder = None

        # aceita tanto o estimador/Pipeline direto quanto um dict {"modelo": ..., "features": ...}
        if isinstance(obj, dict):
            modelo = next((obj[k] for k in ("pipeline", "modelo", "model", "clf") if k in obj), None)
            if modelo is None:
                raise ValueError(f"Dict do joblib sem o modelo. Chaves encontradas: {list(obj)}")
            features = next((obj[k] for k in ("features", "colunas", "colunas_features") if k in obj), None)
            encoder = obj.get("label_encoder") or obj.get("encoder")
        else:
            modelo = obj

        self.usa_nomes = hasattr(modelo, "feature_names_in_")
        if features is None:
            features = list(modelo.feature_names_in_) if self.usa_nomes else list(COLUNAS_FEATURES)
        features = list(features)

        desconhecidas = set(features) - set(COLUNAS_FEATURES)
        if desconhecidas:
            raise ValueError(f"O modelo espera features que a API não calcula: {sorted(desconhecidas)}")

        classes = list(modelo.classes_)
        if encoder is not None:
            classes = list(encoder.inverse_transform(classes))

        self.modelo = modelo
        self.features = features
        self.classes = [str(c) for c in classes]

    def prever(self, feats: dict) -> dict:
        linha = [[feats[c] for c in self.features]]
        X = pd.DataFrame(linha, columns=self.features) if self.usa_nomes else np.array(linha)

        if hasattr(self.modelo, "predict_proba"):
            proba = self.modelo.predict_proba(X)[0]
            idx = int(np.argmax(proba))
            probs = {c: round(float(p), 4) for c, p in zip(self.classes, proba)}
            return {"label": self.classes[idx], "confianca": round(float(proba[idx]), 4),
                    "probabilidades": probs}

        pred = self.modelo.predict(X)[0]
        label = self.classes[list(self.modelo.classes_).index(pred)]
        return {"label": label, "confianca": None, "probabilidades": None}
