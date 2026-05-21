import json
import pickle
from pathlib import Path

import numpy as np
from catboost import CatBoostClassifier, Pool

import sys
sys.path.insert(0, str(Path(__file__).parent))
from features import (
    get_all_ftrs, fingerprint_ftrs,
    FTR_NAMES_STYLO, FTR_NAMES_ALL, FTR_NAMES_FINGERPRINT,
)


class CatBoostStyloClf:
    """
    CatBoost classifier on stylometric + fingerprint features.

    Improvements over the LGBM StyloClf:
      - 'genre' as a native categorical feature (no one-hot, genre-aware splits)
      - 11 new fingerprint features (RLHF vocab, em dash, think blocks, etc.)
      - Ordered boosting → better probability calibration (Brier score)
      - auto_class_weights='Balanced' → handles 38/62 imbalance cleanly
      - No StandardScaler needed
    """

    def __init__(self, use_gltr=False, gltr_mdl="gpt2", use_genre=True):
        self.use_gltr = use_gltr
        self.gltr = None
        self.gltr_mdl = gltr_mdl
        self.use_genre = use_genre
        self.clf = None
        self._build_feature_meta()

    def _build_feature_meta(self):
        base = FTR_NAMES_ALL if self.use_gltr else FTR_NAMES_STYLO
        self.feature_names = list(base) + list(FTR_NAMES_FINGERPRINT)
        self.cat_feature_indices = []
        if self.use_genre:
            self.feature_names = self.feature_names + ["genre"]
            self.cat_feature_indices = [len(self.feature_names) - 1]

    def _init_gltr(self):
        if self.use_gltr and self.gltr is None:
            from features import GLTRFeatures
            self.gltr = GLTRFeatures(self.gltr_mdl)

    def _extract(self, records, desc=""):
        X, ids = [], []
        for i, r in enumerate(records):
            ftrs = get_all_ftrs(r["text"], self.gltr)
            ftrs.update(fingerprint_ftrs(r["text"]))
            row = list(ftrs.values())
            if self.use_genre:
                row.append(r.get("genre", "unknown"))
            X.append(row)
            ids.append(r["id"])
            if (i + 1) % 500 == 0:
                print(f"  {desc}{i+1}/{len(records)}")
        return X, ids

    def _pool(self, X, y=None):
        return Pool(
            X, y,
            feature_names=self.feature_names,
            cat_features=self.cat_feature_indices,
        )

    def fit(self, trn_recs, val_recs=None):
        self._init_gltr()
        print("extracting train features...")
        X_trn, _ = self._extract(trn_recs, "train ")
        y_trn = [r["label"] for r in trn_recs]

        n_feat = len(self.feature_names)
        print(f"features: {n_feat} ({n_feat - int(self.use_genre)} numeric + {int(self.use_genre)} cat)")

        self.clf = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            l2_leaf_reg=3.0,
            border_count=128,
            loss_function="Logloss",
            eval_metric="AUC",
            auto_class_weights="Balanced",
            early_stopping_rounds=50,
            random_seed=42,
            verbose=100,
        )

        if val_recs:
            print("extracting val features...")
            X_val, _ = self._extract(val_recs, "val ")
            y_val = [r["label"] for r in val_recs]
            self.clf.fit(self._pool(X_trn, y_trn), eval_set=self._pool(X_val, y_val))
        else:
            self.clf.fit(self._pool(X_trn, y_trn))

        print(f"best iteration: {self.clf.best_iteration_}")

    def predict_proba(self, records):
        self._init_gltr()
        X, ids = self._extract(records)
        proba = self.clf.predict_proba(self._pool(X))[:, 1]
        return ids, proba.tolist()

    def feature_importance(self):
        imp = self.clf.get_feature_importance()
        return sorted(zip(self.feature_names, imp), key=lambda x: -x[1])

    def save(self, path):
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path):
        with open(path, "rb") as f:
            return pickle.load(f)
