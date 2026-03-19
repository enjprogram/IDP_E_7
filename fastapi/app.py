"""
fastapi/app.py
==============
Unified FastAPI backend — CNN species classifier + NLP support ticket services.

Run from inside the fastapi/ folder:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Routes
------
CNN:
    POST /predict               — species image classification

NLP:
    POST /nlp/classify          — ticket category classification
    POST /nlp/classify/batch    — batch classification (≤ 100 tickets)
    POST /nlp/ner               — named entity extraction
    GET  /nlp/ner/evaluate      — precision/recall on 50-ticket gold standard
    POST /nlp/draft             — LLM draft response generation
    POST /nlp/analyse           — full pipeline: classify + NER + draft

System:
    GET  /health                — service health + classifier mode
    GET  /nlp/categories        — list of supported categories
"""


# fastapi/app.py

from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()  # reading fastapi/.env into os.environ

import io
import os
import time
...

from nlp.classifier_service import TicketClassifierService
from nlp.ner_service import extract_entities, evaluate_on_annotations
from nlp.draft_service import generate_draft  # ← by the time this is imported, key is already set

import io
import os
import time
from functools import lru_cache
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, Field

from nlp.classifier_service import TicketClassifierService
from nlp.ner_service import extract_entities, evaluate_on_annotations
from nlp.draft_service import generate_draft

import os

# ── App ───────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="AI Services API",
    description="CNN Species Classifier + NLP Support Ticket Services",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths are relative to fastapi/ (the working directory when uvicorn is run)
# CNN_MODEL_PATH        = os.getenv("CNN_MODEL_PATH",        r"exported_model\1\model.keras")
# CNN_LABELS_PATH       = os.getenv("CNN_LABELS_PATH",       r"data\v1\cnn_label_classes.json")
# NER_ANNOTATIONS_PATH  = os.getenv("NER_ANNOTATIONS_PATH",  r"data\ner_annotations.jsonl")
# BERT_MODEL_DIR        = os.getenv("BERT_MODEL_DIR",        r"models\best_checkpoint")
# BERT_LABEL_MAP        = os.getenv("BERT_LABEL_MAP",        r"models\label_map.json")

# For loading from the files
# CNN_MODEL_PATH        = os.getenv("CNN_MODEL_PATH",        "exported_model/1/model.keras")
# CNN_LABELS_PATH       = os.getenv("CNN_LABELS_PATH",       "data/v1/cnn_label_classes.json")
# NER_ANNOTATIONS_PATH  = os.getenv("NER_ANNOTATIONS_PATH",  "data/ner_annotations.jsonl")
# BERT_MODEL_DIR        = os.getenv("BERT_MODEL_DIR",        "models/best_checkpoint")
# BERT_LABEL_MAP        = os.getenv("BERT_LABEL_MAP",        "models/label_map.json")

# For loading from mlflow:
import mlflow
import mlflow.keras
import mlflow.transformers

MLFLOW_TRACKING_URI   = os.getenv("MLFLOW_TRACKING_URI",   "http://mlflow:5000")
CNN_MODEL_NAME        = os.getenv("CNN_MODEL_NAME",         "cnn-species-classifier")
CNN_MODEL_STAGE       = os.getenv("CNN_MODEL_STAGE",        "Production")
BERT_MODEL_NAME       = os.getenv("BERT_MODEL_NAME",        "bert-ticket-classifier")
BERT_MODEL_STAGE      = os.getenv("BERT_MODEL_STAGE",       "Production")
NER_ANNOTATIONS_PATH  = os.getenv("NER_ANNOTATIONS_PATH",   "data/ner_annotations.jsonl")

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

CATEGORIES = ["Delivery", "Refund", "Account", "Product Issue", "Other"]


# ═══════════════════════════════════════════════════════════════════════════════
# CNN — existing logic, untouched
# ═══════════════════════════════════════════════════════════════════════════════

# Loading from file
# @lru_cache(maxsize=1)
# def _load_cnn_model():
#     try:
#         from tensorflow.keras.models import load_model
#         if not os.path.exists(CNN_MODEL_PATH):
#             raise FileNotFoundError(f"Model not found at: {CNN_MODEL_PATH}")
#         model = load_model(CNN_MODEL_PATH)
#         print(f"[INFO] CNN model loaded from {CNN_MODEL_PATH}")
#         return model
#     except Exception as e:
#         print(f"[ERROR] CNN model load failed: {e}")
#         return None


# For loading from mlflow:
@lru_cache(maxsize=1)
def _load_cnn_model():
    try:
        model_uri = f"models:/{CNN_MODEL_NAME}/{CNN_MODEL_STAGE}"
        model = mlflow.keras.load_model(model_uri)
        print(f"[INFO] CNN model loaded from MLflow: {model_uri}")
        return model
    except Exception as e:
        print(f"[ERROR] CNN model load failed from MLflow: {e}")
        return None


# Loading from file
# @lru_cache(maxsize=1)
# def _load_cnn_labels() -> list:
#     """Load class names from cnn_label_classes.json.
#     Falls back to searching data/vN/ folders newest-first if CNN_LABELS_PATH
#     doesn't exist yet (e.g. before the first notebook run)."""
#     import json, glob
#     # 1. Explicit configured path
#     if os.path.exists(CNN_LABELS_PATH):
#         with open(CNN_LABELS_PATH) as f:
#             return json.load(f)
#     # 2. Search data/vN/ folders newest-first
#     candidates = sorted(
#         glob.glob(os.path.join("data", "v*", "cnn_label_classes.json")),
#         key=lambda p: int(os.path.basename(os.path.dirname(p)).lstrip("v")),
#         reverse=True,
#     )
#     if candidates:
#         print(f"[INFO] CNN labels loaded from {candidates[0]}")
#         with open(candidates[0]) as f:
#             return json.load(f)
#     print("[WARN] cnn_label_classes.json not found — class names unavailable")
#     return []


# Loading from mlflow:
@lru_cache(maxsize=1)
def _load_cnn_labels() -> list:
    try:
        import json
        client    = mlflow.tracking.MlflowClient()
        # Get the latest production version
        versions  = client.get_latest_versions(CNN_MODEL_NAME, stages=[CNN_MODEL_STAGE])
        if not versions:
            raise ValueError(f"No {CNN_MODEL_STAGE} version found for {CNN_MODEL_NAME}")
        run_id    = versions[0].run_id
        # Download labels artifact from the run
        local_path = mlflow.artifacts.download_artifacts(
            run_id=run_id, artifact_path="labels/cnn_label_classes.json"
        )
        with open(local_path) as f:
            labels = json.load(f)
        print(f"[INFO] CNN labels loaded from MLflow run {run_id}")
        return labels
    except Exception as e:
        print(f"[WARN] Could not load CNN labels from MLflow: {e}")
        return []

def _preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.resize((128, 128))
    return np.expand_dims(np.array(image) / 255.0, axis=0)


@app.post("/predict", tags=["CNN"])
async def predict_species(file: UploadFile = File(...)):
    """Classify an uploaded image using the CNN species model."""
    model = _load_cnn_model()
    try:
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = _preprocess_image(image)
        predictions = model.predict(x)
        probs       = predictions[0]                    # shape (n_classes,)
        class_id    = int(np.argmax(probs))
        labels      = _load_cnn_labels()
        class_name  = labels[class_id] if labels and class_id < len(labels) else f"Class {class_id}"
        # Build per-class score dict — same pattern as NLP classifier
        scores = {
            (labels[i] if labels and i < len(labels) else f"Class {i}"): round(float(p), 4)
            for i, p in enumerate(probs)
        }
        return {
            "class_id":   class_id,
            "class_name": class_name,
            "confidence": float(np.max(probs)),
            "scores":     scores,
        }
    except Exception as e:
        return {"error": str(e)}


# ═══════════════════════════════════════════════════════════════════════════════
# NLP — lazy-loaded classifier
# ═══════════════════════════════════════════════════════════════════════════════

# Load from file
# @lru_cache(maxsize=1)
# def _get_classifier() -> TicketClassifierService:
#     svc = TicketClassifierService(model_dir=BERT_MODEL_DIR, label_map_path=BERT_LABEL_MAP)
#     mode = "BERT" if svc.using_bert else "keyword fallback"
#     print(f"[INFO] Ticket classifier loaded — using {mode}")
#     return svc


# Load from mlflow:
@lru_cache(maxsize=1)
def _get_classifier() -> TicketClassifierService:
    svc  = TicketClassifierService(
        mlflow_model_name = BERT_MODEL_NAME,   # from env var
        mlflow_stage      = BERT_MODEL_STAGE,  # from env var
    )
    mode = "BERT" if svc.using_bert else "keyword fallback"
    print(f"[INFO] Ticket classifier loaded — using {mode}")
    return svc


# ── Request / Response schemas ────────────────────────────────────────────────

class ClassifyRequest(BaseModel):
    ticket_id: Optional[str] = Field(None,  example="TKT-0001")
    text:      str            = Field(...,   min_length=3, max_length=2000,
                                            example="My order ORD-12345678 hasn't arrived.")

class BatchClassifyRequest(BaseModel):
    tickets: List[ClassifyRequest]

class NERRequest(BaseModel):
    ticket_id: Optional[str] = None
    text:      str            = Field(..., min_length=3, max_length=2000)

class DraftRequest(BaseModel):
    ticket_id:      Optional[str] = None
    text:           str            = Field(..., min_length=3, max_length=2000)
    category:       str            = Field(..., example="Delivery")
    openai_api_key: Optional[str]  = Field(None, description="Overrides OPENAI_API_KEY env var")

class AnalyseRequest(BaseModel):
    ticket_id:      Optional[str] = None
    text:           str            = Field(..., min_length=3, max_length=2000)
    openai_api_key: Optional[str]  = None


# ── Endpoints ─────────────────────────────────────────────────────────────────

# @app.get("/health", tags=["System"])
# def health():
#     clf = _get_classifier()
#     return {
#         "status":          "ok",
#         "classifier":      "bert" if clf.using_bert else "keyword_fallback",
#         "cnn_model_path":  CNN_MODEL_PATH,
#     }

# # Load from file:
# @app.get("/health", tags=["System"])
# def health():
#     clf        = _get_classifier()
#     cnn_model  = _load_cnn_model()
#     cnn_labels = _load_cnn_labels()
#     # show model information used by the classification service
#     return {
#         "status": "ok",
#         "models": {
#             "nlp": {
#                 "classifier":  "bert" if clf.using_bert else "keyword_fallback",
#                 "loaded":      True,
#             },
#             "cnn": {
#                 "loaded":      cnn_model is not None,
#                 "labels":      len(cnn_labels) if cnn_labels else 0,
#                 "model_path":  CNN_MODEL_PATH,
#             }
#         }
#     }


# Load from mlflow:
@app.get("/health", tags=["System"])
def health():
    clf       = _get_classifier()
    cnn_model = _load_cnn_model()
    cnn_labels = _load_cnn_labels()
    return {
        "status": "ok",
        "mlflow_uri": MLFLOW_TRACKING_URI,
        "models": {
            "nlp": {
                "classifier":  "bert" if clf.using_bert else "keyword_fallback",
                "loaded":      True,
                "source":      f"models:/{BERT_MODEL_NAME}/{BERT_MODEL_STAGE}",
            },
            "cnn": {
                "loaded":      cnn_model is not None,
                "labels":      len(cnn_labels) if cnn_labels else 0,
                "source":      f"models:/{CNN_MODEL_NAME}/{CNN_MODEL_STAGE}",
            }
        }
    }


@app.on_event("startup")
async def startup_event():
    print("[INFO] Pre-loading all models on startup...")
    _get_classifier()   # pre-load BERT
    _load_cnn_model()   # pre-load CNN
    _load_cnn_labels()  # pre-load labels
    print("[INFO] All models loaded and ready.")

@app.get("/nlp/categories", tags=["NLP"])
def list_categories():
    return {"categories": CATEGORIES}


@app.post("/nlp/classify", tags=["NLP"])
def classify_ticket(req: ClassifyRequest):
    """Classify a single support ticket into one of 5 categories."""
    t0 = time.perf_counter()
    result = _get_classifier().predict(req.text)
    return {
        "ticket_id": req.ticket_id,
        **result,
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


@app.post("/nlp/classify/batch", tags=["NLP"])
def classify_batch(req: BatchClassifyRequest):
    """Classify up to 100 tickets in a single call."""
    if len(req.tickets) > 100:
        raise HTTPException(400, "Batch size must be ≤ 100")
    t0 = time.perf_counter()
    results = _get_classifier().predict_batch([t.text for t in req.tickets])
    return {
        "results": [
            {"ticket_id": req.tickets[i].ticket_id, **r}
            for i, r in enumerate(results)
        ],
        "total_latency_ms": round((time.perf_counter() - t0) * 1000, 2),
    }


@app.post("/nlp/ner", tags=["NLP"])
def run_ner(req: NERRequest):
    """Extract ORDER_ID, DATE, and EMAIL entities from ticket text."""
    entities = extract_entities(req.text)
    return {
        "ticket_id":    req.ticket_id,
        "text":         req.text,
        "entities":     entities,
        "entity_count": len(entities),
    }


@app.get("/nlp/ner/evaluate", tags=["NLP"])
def evaluate_ner(annotations_path: str = Query(default=NER_ANNOTATIONS_PATH)):
    """Run precision/recall/F1 evaluation against the 50-ticket gold standard."""
    if not os.path.exists(annotations_path):
        raise HTTPException(404, f"Annotations file not found: {annotations_path}")
    try:
        return {"evaluation": evaluate_on_annotations(annotations_path),
                "annotations_file": annotations_path}
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post("/nlp/draft", tags=["NLP"])
def draft_response(req: DraftRequest):
    """Generate a professional first-response draft (LLM or template fallback)."""
    if req.category not in CATEGORIES:
        raise HTTPException(400, f"category must be one of {CATEGORIES}")
    t0 = time.perf_counter()
    result = generate_draft(
        text=req.text, category=req.category,
        ticket_id=req.ticket_id, api_key=req.openai_api_key,
    )
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    return result


@app.post("/nlp/analyse", tags=["NLP"])
def full_pipeline(req: AnalyseRequest):
    """Full NLP pipeline: classify → NER → draft, returned as one structured response."""
    t0 = time.perf_counter()

    clf_result = _get_classifier().predict(req.text)
    category   = clf_result["predicted_category"]
    entities   = extract_entities(req.text)
    draft      = generate_draft(
        text=req.text, category=category,
        ticket_id=req.ticket_id, api_key=req.openai_api_key,
    )

    return {
        "ticket_id":          req.ticket_id,
        "category":           category,
        "confidence":         clf_result["confidence"],
        "classifier_scores":  clf_result["scores"],
        "classifier_method":  clf_result.get("method"),
        "entities":           entities,
        "draft_response":     draft["draft_response"],
        "draft_error":        draft.get("error"),
        "model":              draft.get("model"),
        "total_latency_ms":   round((time.perf_counter() - t0) * 1000, 2),
    }
