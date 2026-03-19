"""
fastapi/scripts/evaluate_ner.py
--------------------------------
Standalone NER evaluation script. Run from the fastapi/ folder:

    python scripts/evaluate_ner.py
    python scripts/evaluate_ner.py --annotations data/ner_annotations.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

# Ensure nlp/ package is importable when run from fastapi/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from nlp.ner_service import extract_entities, evaluate_on_annotations


def main(annotations_path: str):
    print("=" * 62)
    print("NER Evaluation — regex rule-based (ORDER_ID, DATE, EMAIL)")
    print("=" * 62)

    results = evaluate_on_annotations(annotations_path)

    print(f"\n{'Entity':<14} {'Precision':>10} {'Recall':>10} {'F1':>8} {'TP':>5} {'FP':>5} {'FN':>5}")
    print("-" * 62)
    for label in ["ORDER_ID", "DATE", "EMAIL"]:
        m = results[label]
        print(f"{label:<14} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>8.3f} "
              f"{m['tp']:>5} {m['fp']:>5} {m['fn']:>5}")
    print("-" * 62)
    m = results["micro_avg"]
    print(f"{'micro_avg':<14} {m['precision']:>10.3f} {m['recall']:>10.3f} {m['f1']:>8.3f}")

    print("\n" + "=" * 62)
    print("Qualitative examples (tickets with entities)")
    print("=" * 62)

    with open(annotations_path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    shown = 0
    for rec in records:
        gold  = rec["entities"]
        preds = extract_entities(rec["text"])
        if not gold and not preds:
            continue
        gold_set = {(e["label"], e["start"], e["end"]) for e in gold}
        pred_set = {(e["label"], e["start"], e["end"]) for e in preds}
        print(f"\n[{rec['ticket_id']}]")
        print(f"  Text : {rec['text']}")
        print(f"  Gold : {[(e['label'], e['text']) for e in gold]}")
        print(f"  Pred : {[(e['label'], e['text']) for e in preds]}")
        tp = gold_set & pred_set
        fp = pred_set - gold_set
        fn = gold_set - pred_set
        if tp: print(f"  TP: {len(tp)}")
        if fp: print(f"  FP: {fp}")
        if fn: print(f"  FN: {fn}")
        shown += 1
        if shown >= 8:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", default="data/ner_annotations.jsonl")
    main(parser.parse_args().annotations)
