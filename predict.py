#!/usr/bin/env python3
"""
Tira submission entrypoint.

Usage:
    python predict.py <input_dir> <output_dir>

    <input_dir>  must contain dataset.jsonl
    <output_dir> predictions.jsonl will be written here
"""
import json
import sys
import os
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).parent
CATBOOST_PATH = SCRIPT_DIR / "catboost_model.pkl"
DEBERTA_PATH = SCRIPT_DIR / "deberta_v2_best.pt"  # copied flat into /app by Dockerfile
DEBERTA_MDL = "microsoft/deberta-v3-large"


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(l) for l in f]


def write_jsonl(path, ids, scores):
    with open(path, "w") as f:
        for doc_id, score in zip(ids, scores):
            f.write(json.dumps({"id": doc_id, "label": round(float(score), 6)}) + "\n")


def main():
    if len(sys.argv) == 3:
        input_dir = Path(sys.argv[1])
        output_dir = Path(sys.argv[2])
    elif os.environ.get("inputDataset") and os.environ.get("outputDir"):
        input_dir = Path(os.environ["inputDataset"])
        output_dir = Path(os.environ["outputDir"])
    else:
        print("usage: predict.py <input_dir> <output_dir>", file=sys.stderr)
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_jsonl(input_dir / "dataset.jsonl")
    print(f"loaded {len(records)} records")

    predictions = {}

    # --- CatBoost (always runs, no GPU needed) ---
    if CATBOOST_PATH.exists():
        sys.path.insert(0, str(SCRIPT_DIR))
        from models.catboost_clf import CatBoostStyloClf
        print("running CatBoost...")
        clf = CatBoostStyloClf.load(CATBOOST_PATH)
        ids, proba = clf.predict_proba(records)
        predictions["catboost"] = dict(zip(ids, proba))
        print(f"  CatBoost done")
    else:
        print("WARNING: catboost_model.pkl not found, skipping", file=sys.stderr)

    # --- DeBERTa (runs if checkpoint present) ---
    if DEBERTA_PATH.exists():
        from models.deberta_inf import DeBERTaInf
        print("running DeBERTa...")
        deberta = DeBERTaInf(DEBERTA_PATH, mdl_nm=DEBERTA_MDL)
        ids, proba = deberta.predict_proba(records)
        predictions["deberta"] = dict(zip(ids, proba))
        print(f"  DeBERTa done")
    else:
        print("NOTE: deberta_v2_best.pt not found, running CatBoost-only")

    if not predictions:
        print("ERROR: no models available", file=sys.stderr)
        sys.exit(1)

    # --- Ensemble ---
    all_ids = [r["id"] for r in records]
    if len(predictions) == 1:
        name = list(predictions.keys())[0]
        final_scores = [predictions[name][doc_id] for doc_id in all_ids]
    else:
        # Simple average — both models are well-calibrated
        final_scores = [
            np.mean([predictions[m][doc_id] for m in predictions])
            for doc_id in all_ids
        ]

    out_path = output_dir / "predictions.jsonl"
    write_jsonl(out_path, all_ids, final_scores)
    print(f"wrote {len(all_ids)} predictions → {out_path}")


if __name__ == "__main__":
    main()
