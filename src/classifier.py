"""
src/classifier.py - SVM + kNN Ensemble Classifier
===================================================
Classifies a face embedding into one of the enrolled student identities.

THREE-GATE UNKNOWN REJECTION (see full explanation in comments below):
  Gate 1 — Global nearest-neighbour distance  (primary unknown gate)
  Gate 2 — Ensemble confidence threshold
  Gate 3 — Class-specific cosine similarity

No roll numbers — identity is name only.
"""

import numpy as np
import pickle
import logging
from pathlib import Path
from typing import Optional, List, Dict
from dataclasses import dataclass, field

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

UNKNOWN_LABEL = "Unknown"


@dataclass
class RecognitionResult:
    """
    Result of classifying one face embedding.

    Attributes:
        name:          Student name, or "Unknown"
        confidence:    Final ensemble confidence [0, 1]
        svm_pred:      SVM's predicted name
        svm_conf:      SVM's confidence score
        knn_pred:      kNN's predicted name
        knn_conf:      kNN's confidence score
        agreed:        True if SVM and kNN predicted same class
        global_nn_sim: Cosine sim to nearest enrollment embedding (Gate 1)
        reject_reason: Which gate triggered rejection, or None if accepted
    """
    name:          str
    confidence:    float
    svm_pred:      str
    svm_conf:      float
    knn_pred:      str
    knn_conf:      float
    agreed:        bool
    global_nn_sim: float         = 0.0
    reject_reason: Optional[str] = None

    @property
    def is_known(self) -> bool:
        return self.name != UNKNOWN_LABEL


class EnsembleClassifier:
    """
    SVM + kNN ensemble with three-gate unknown rejection.

    Three rejection gates applied in sequence:

      Gate 1 — global_nn_sim < global_nn_threshold
        Cosine similarity to the SINGLE nearest enrollment embedding
        across ALL students. An unknown person will be far from everyone.
        This is the most effective gate for rejecting unknown faces.
        Recommended: 0.50 for 8–80 student datasets.

      Gate 2 — ensemble confidence < confidence_threshold
        Requires both classifiers to be confident. Extra penalty applied
        when SVM and kNN disagree (raises effective threshold by 0.08).

      Gate 3 — class cosine sim < min_cosine_sim
        Verifies geometric proximity specifically to the predicted student,
        not just any enrolled person.
    """

    def __init__(self,
                 svm_C:                float = 10.0,
                 svm_kernel:           str   = "rbf",
                 svm_gamma:            str   = "scale",
                 knn_k:                int   = 5,
                 confidence_threshold: float = 0.70,
                 min_cosine_sim:       float = 0.45,
                 global_nn_threshold:  float = 0.50):
        self.confidence_threshold = confidence_threshold
        self.min_cosine_sim       = min_cosine_sim
        self.global_nn_threshold  = global_nn_threshold

        self._svm = SVC(
            C=svm_C, kernel=svm_kernel, gamma=svm_gamma,
            probability=True, class_weight="balanced",
        )
        self._knn = KNeighborsClassifier(
            n_neighbors=knn_k, metric="cosine", algorithm="brute",
        )
        self._le               = LabelEncoder()
        self._label_to_name:   Dict[int, str]          = {}
        self._train_embeddings: Optional[np.ndarray]   = None
        self._train_labels:     Optional[np.ndarray]   = None
        self._fitted           = False

    def fit(self,
            embeddings: np.ndarray,
            labels:     np.ndarray,
            names_map:  Dict[int, str]):
        """
        Train SVM + kNN on enrollment embeddings.

        Args:
            embeddings: (N, 512) float32 L2-normalised embeddings
            labels:     (N,) int array of student label indices
            names_map:  {label_index: student_name}
        """
        if len(embeddings) == 0:
            raise ValueError("No embeddings provided.")

        logger.info(f"[Classifier] Training on {len(embeddings)} embeddings, "
                    f"{len(np.unique(labels))} students.")

        self._label_to_name    = names_map
        self._train_embeddings = embeddings.copy()
        self._train_labels     = labels.copy()

        encoded = self._le.fit_transform(labels)
        logger.info("[Classifier] Training SVM...")
        self._svm.fit(embeddings, encoded)
        logger.info("[Classifier] Training kNN...")
        self._knn.fit(embeddings, labels)

        self._fitted = True
        logger.info("[Classifier] Training complete.")
        self._log_nn_stats()

    def predict(self, embedding: np.ndarray) -> RecognitionResult:
        """
        Classify one face embedding through three rejection gates.

        Returns RecognitionResult. Check .is_known to see if accepted.
        """
        if not self._fitted:
            raise RuntimeError("Classifier not trained. Call fit() first.")

        emb = embedding.reshape(1, -1)

        # ── Gate 1: Global nearest-neighbour ───────────────────────────
        global_nn_sim = self._global_nn_sim(embedding)
        if global_nn_sim < self.global_nn_threshold:
            return RecognitionResult(
                name=UNKNOWN_LABEL, confidence=global_nn_sim,
                svm_pred=UNKNOWN_LABEL, svm_conf=0.0,
                knn_pred=UNKNOWN_LABEL, knn_conf=0.0,
                agreed=False, global_nn_sim=global_nn_sim,
                reject_reason="gate1_nn",
            )

        # ── SVM prediction ──────────────────────────────────────────────
        svm_proba   = self._svm.predict_proba(emb)[0]
        svm_encoded = np.argmax(svm_proba)
        svm_conf    = float(svm_proba[svm_encoded])
        svm_label   = int(self._le.inverse_transform([svm_encoded])[0])
        svm_name    = self._label_to_name.get(svm_label, UNKNOWN_LABEL)

        # ── kNN prediction ──────────────────────────────────────────────
        knn_proba     = self._knn.predict_proba(emb)[0]
        knn_label_idx = np.argmax(knn_proba)
        knn_conf      = float(knn_proba[knn_label_idx])
        knn_label     = int(self._knn.classes_[knn_label_idx])
        knn_name      = self._label_to_name.get(knn_label, UNKNOWN_LABEL)

        # ── Ensemble ────────────────────────────────────────────────────
        agreed = (svm_label == knn_label)
        if agreed:
            final_label = svm_label
            final_conf  = (svm_conf + knn_conf) / 2.0
        else:
            if svm_conf >= knn_conf:
                final_label, final_conf = svm_label, svm_conf * 0.75
            else:
                final_label, final_conf = knn_label, knn_conf * 0.75

        final_name = self._label_to_name.get(final_label, UNKNOWN_LABEL)

        # ── Gate 2: Confidence ──────────────────────────────────────────
        threshold = self.confidence_threshold + (0.08 if not agreed else 0.0)
        if final_conf < threshold:
            return RecognitionResult(
                name=UNKNOWN_LABEL, confidence=final_conf,
                svm_pred=svm_name, svm_conf=svm_conf,
                knn_pred=knn_name, knn_conf=knn_conf,
                agreed=agreed, global_nn_sim=global_nn_sim,
                reject_reason="gate2_conf",
            )

        # ── Gate 3: Class cosine similarity ────────────────────────────
        class_sim = self._class_cosine_sim(embedding, final_label)
        if class_sim < self.min_cosine_sim:
            return RecognitionResult(
                name=UNKNOWN_LABEL, confidence=final_conf,
                svm_pred=svm_name, svm_conf=svm_conf,
                knn_pred=knn_name, knn_conf=knn_conf,
                agreed=agreed, global_nn_sim=global_nn_sim,
                reject_reason="gate3_class",
            )

        # ── All gates passed ────────────────────────────────────────────
        return RecognitionResult(
            name=final_name, confidence=final_conf,
            svm_pred=svm_name, svm_conf=svm_conf,
            knn_pred=knn_name, knn_conf=knn_conf,
            agreed=agreed, global_nn_sim=global_nn_sim,
            reject_reason=None,
        )

    def predict_batch(self, embeddings: List[np.ndarray]) -> List[RecognitionResult]:
        return [self.predict(e) for e in embeddings]

    # ── Private helpers ───────────────────────────────────────────────

    def _global_nn_sim(self, embedding: np.ndarray) -> float:
        """Max cosine similarity to any enrollment embedding."""
        if self._train_embeddings is None:
            return 1.0
        return float(np.max(self._train_embeddings @ embedding))

    def _class_cosine_sim(self, embedding: np.ndarray, label: int) -> float:
        """Max cosine similarity to enrollment embeddings of one class."""
        if self._train_embeddings is None:
            return 1.0
        mask = self._train_labels == label
        embs = self._train_embeddings[mask]
        return float(np.max(embs @ embedding)) if len(embs) else 0.0

    def _log_nn_stats(self):
        """Log within-class NN stats to help tune global_nn_threshold."""
        if self._train_embeddings is None:
            return
        per_class_mins = []
        for label in np.unique(self._train_labels):
            mask = self._train_labels == label
            embs = self._train_embeddings[mask]
            sim  = embs @ embs.T
            np.fill_diagonal(sim, -1)
            per_class_mins.append(float(np.max(sim, axis=1).min()))
        mean_min  = float(np.mean(per_class_mins))
        std_min   = float(np.std(per_class_mins))
        suggested = round(mean_min - std_min, 3)
        logger.info(
            f"[Classifier] Within-class NN sim: mean={mean_min:.3f}  std={std_min:.3f}\n"
            f"  Suggested global_nn_threshold: {suggested}  "
            f"(current: {self.global_nn_threshold})"
        )

    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "svm":                  self._svm,
                "knn":                  self._knn,
                "label_encoder":        self._le,
                "label_to_name":        self._label_to_name,
                "train_embeddings":     self._train_embeddings,
                "train_labels":         self._train_labels,
                "confidence_threshold": self.confidence_threshold,
                "min_cosine_sim":       self.min_cosine_sim,
                "global_nn_threshold":  self.global_nn_threshold,
                "fitted":               self._fitted,
            }, f)
        logger.info(f"[Classifier] Saved to {path}")

    @classmethod
    def load(cls, path: str) -> "EnsembleClassifier":
        with open(path, "rb") as f:
            p = pickle.load(f)
        obj = cls(
            confidence_threshold=p["confidence_threshold"],
            min_cosine_sim=p["min_cosine_sim"],
            global_nn_threshold=p.get("global_nn_threshold", 0.50),
        )
        obj._svm               = p["svm"]
        obj._knn               = p["knn"]
        obj._le                = p["label_encoder"]
        obj._label_to_name     = p["label_to_name"]
        obj._train_embeddings  = p["train_embeddings"]
        obj._train_labels      = p["train_labels"]
        obj._fitted            = p["fitted"]
        # Handle old classifiers that stored label_to_roll
        # (gracefully ignored — roll numbers removed from system)
        logger.info(f"[Classifier] Loaded from {path}")
        return obj
