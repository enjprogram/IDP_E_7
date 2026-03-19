"""
fastapi/nlp/classifier_service.py
----------------------------------
Thin wrapper around the trained BERT ticket classifier.
Falls back to a keyword-based heuristic if the MLflow model is unavailable.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

# -- Keyword fallback -----------------------------------------------------------------------------
_KEYWORD_RULES: List[tuple] = [
    ("Delivery",      ["tracking", "shipped", "shipping", "transit", "delivered", "parcel",
                       "courier", "hasn't shown up", "hasn't arrived", "can't find the package",
                       "in transit", "safe place"]),
    ("Refund",        ["refund", "money back", "charged twice", "duplicate charge", "cancelled",
                       "pending charge", "reimburs", "taking too long", "when will i get"]),
    ("Account",       ["log in", "login", "password", "locked", "unlock", "two-factor", "2fa",
                       "email address", "update my email", "account", "verify"]),
    ("Product Issue", ["scratched", "faulty", "won't turn on", "missing parts", "doesn't match",
                       "wrong", "size/fit", "defect", "broken", "arrived damaged", "replacement",
                       "warranty"]),
    ("Other",         ["size guide", "discount code", "back in stock", "ship internationally",
                       "where can i find", "how do i", "question about"]),
]

CATEGORIES = ["Delivery", "Refund", "Account", "Product Issue", "Other"]


def _keyword_predict(text: str) -> Dict:
    text_lower = text.lower()
    scores = {cat: 0 for cat, _ in _KEYWORD_RULES}
    for cat, keywords in _KEYWORD_RULES:
        for kw in keywords:
            if kw in text_lower:
                scores[cat] += 1
    best  = max(scores, key=lambda c: scores[c])
    total = sum(scores.values()) or 1
    return {
        "predicted_category": best if scores[best] > 0 else "Other",
        "confidence": round(min(scores[best] / total + 0.4, 0.95), 3),
        "scores":     {c: round(s / total, 3) for c, s in scores.items()},
        "method":     "keyword_fallback",
    }


def _load_bert_from_mlflow(mlflow_model_name: str, mlflow_stage: str):
    """Load fine-tuned BERT from MLflow registry only. No local files."""
    try:
        import torch
        import mlflow
        import mlflow.transformers
        from mlflow.tracking import MlflowClient

        mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

        model_uri  = f"models:/{mlflow_model_name}/{mlflow_stage}"
        print(f"[INFO] Loading BERT from MLflow: {model_uri}")

        components = mlflow.transformers.load_model(model_uri, return_type="components")
        model      = components["model"]
        tokenizer  = components["tokenizer"]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        # Download label map from MLflow artifact
        client   = MlflowClient()
        versions = client.get_latest_versions(mlflow_model_name, stages=[mlflow_stage])
        if not versions:
            raise ValueError(f"No {mlflow_stage} version found for {mlflow_model_name}")

        run_id         = versions[0].run_id
        label_map_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="label_map.json"
        )
        with open(label_map_path) as f:
            label_map = {int(k): v for k, v in json.load(f).items()}

        print(f"[INFO] BERT loaded from MLflow — run_id: {run_id}")
        return model, tokenizer, (device, label_map)

    except Exception as e:
        print(f"[WARN] MLflow BERT load failed: {e} — falling back to keyword classifier")
        return None, None, None


class TicketClassifierService:
    def __init__(
        self,
        mlflow_model_name: str = "bert-ticket-classifier",
        mlflow_stage:      str = "Production",
    ):
        # Load from mlflow
        self.model, self.tokenizer, self._meta = _load_bert_from_mlflow(
            mlflow_model_name, mlflow_stage
        )
        self.using_bert = self.model is not None

    def predict(self, text: str) -> Dict:
        if not self.using_bert:
            return _keyword_predict(text)

        import torch
        device, label_map = self._meta
        enc = self.tokenizer(text, max_length=128, padding="max_length",
                             truncation=True, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
            ).logits
        probs   = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
        pred_id = int(probs.argmax())
        return {
            "predicted_category": label_map[pred_id],
            "confidence":         round(float(probs[pred_id]), 3),
            "scores":             {label_map[i]: round(float(p), 3) for i, p in enumerate(probs)},
            "method":             "bert",
        }

    def predict_batch(self, texts: List[str], batch_size: int = 32) -> List[Dict]:
        if not self.using_bert:
            return [_keyword_predict(t) for t in texts]

        import torch
        device, label_map = self._meta
        results = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            enc   = self.tokenizer(batch, max_length=128, padding=True,
                                   truncation=True, return_tensors="pt")
            with torch.no_grad():
                logits = self.model(
                    input_ids=enc["input_ids"].to(device),
                    attention_mask=enc["attention_mask"].to(device),
                ).logits
            for p in torch.softmax(logits, dim=1).cpu().numpy():
                pid = int(p.argmax())
                results.append({
                    "predicted_category": label_map[pid],
                    "confidence":         round(float(p[pid]), 3),
                    "scores":             {label_map[i]: round(float(v), 3) for i, v in enumerate(p)},
                    "method":             "bert",
                })
        return results
