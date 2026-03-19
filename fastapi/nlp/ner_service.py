"""
fastapi/nlp/ner_service.py
---------------------------
Rule-based NER using regex for three entity types found in support tickets:

    ORDER_ID  — ORD-XXXXXXXX  (fixed prefix + 8-digit number)
    DATE      — YYYY-MM-DD    (ISO 8601)
    EMAIL     — RFC-5322 email addresses

Justification for rule-based approach over a transformer NER:
    All three entity types follow rigid, unambiguous surface patterns.
    Regex achieves near-perfect precision/recall with zero training data,
    zero model-loading latency, and full explainability. A pre-trained
    transformer (e.g. dslim/bert-base-NER) would add ~200 ms latency and
    hundreds of MB of weights for no measurable quality gain on these
    highly-structured entity types.
"""

import re
import json
from typing import List, Dict, Any

PATTERNS: List[tuple] = [
    ("ORDER_ID", re.compile(r'\bORD-\d{8}\b')),
    ("DATE",     re.compile(r'\b\d{4}-\d{2}-\d{2}\b')),
    ("EMAIL",    re.compile(r'[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}')),
]

ENTITY_COLORS: Dict[str, str] = {
    "ORDER_ID": "#3B82F6",
    "DATE":     "#10B981",
    "EMAIL":    "#F59E0B",
}


def extract_entities(text: str) -> List[Dict[str, Any]]:
    """Return sorted list of {label, start, end, text} dicts."""
    entities = []
    for label, pattern in PATTERNS:
        for m in pattern.finditer(text):
            entities.append({
                "label": label,
                "start": m.start(),
                "end":   m.end(),
                "text":  m.group(),
            })
    return sorted(entities, key=lambda e: e["start"])


def evaluate_on_annotations(jsonl_path: str) -> Dict[str, Any]:
    """
    Compute exact-match precision, recall, F1 per entity type and micro-average
    against a JSONL gold-standard file (one JSON object per line).

    Each line must contain: ticket_id, text, entities (list of {label, start, end, text}).
    """
    gold_by_label: Dict[str, list] = {"ORDER_ID": [], "DATE": [], "EMAIL": []}
    pred_by_label: Dict[str, list] = {"ORDER_ID": [], "DATE": [], "EMAIL": []}

    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            text   = record["text"]
            gold   = {(e["label"], e["start"], e["end"]) for e in record["entities"]}
            pred   = {(e["label"], e["start"], e["end"]) for e in extract_entities(text)}

            for label in gold_by_label:
                gold_by_label[label].append({e for e in gold if e[0] == label})
                pred_by_label[label].append({e for e in pred if e[0] == label})

    results: Dict[str, Any] = {}
    for label in gold_by_label:
        tp = sum(len(g & p) for g, p in zip(gold_by_label[label], pred_by_label[label]))
        fp = sum(len(p - g) for g, p in zip(gold_by_label[label], pred_by_label[label]))
        fn = sum(len(g - p) for g, p in zip(gold_by_label[label], pred_by_label[label]))
        p_ = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r_ = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p_ * r_ / (p_ + r_) if (p_ + r_) > 0 else 0.0
        results[label] = {"precision": round(p_, 3), "recall": round(r_, 3),
                          "f1": round(f1, 3), "tp": tp, "fp": fp, "fn": fn}

    tp_all = sum(v["tp"] for v in results.values())
    fp_all = sum(v["fp"] for v in results.values())
    fn_all = sum(v["fn"] for v in results.values())
    mp = tp_all / (tp_all + fp_all) if (tp_all + fp_all) > 0 else 0.0
    mr = tp_all / (tp_all + fn_all) if (tp_all + fn_all) > 0 else 0.0
    mf = 2 * mp * mr / (mp + mr) if (mp + mr) > 0 else 0.0
    results["micro_avg"] = {"precision": round(mp, 3), "recall": round(mr, 3), "f1": round(mf, 3)}

    return results
