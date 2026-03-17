"""
src/enrollment.py - Student Enrollment Pipeline
================================================
Processes per-student image folders to build the embedding database.

ENROLLMENT FOLDER STRUCTURE (created by extract_frames.py):
  data/enrollment/
    ├── John_Doe/
    │   ├── frame_0000.jpg
    │   ├── frame_0005.jpg
    │   └── ...
    ├── Jane_Smith/
    │   └── ...
    └── ...

  Folder name = student name (underscores used as word separators).
  No roll numbers — identity is name only.

TRAINING DATA:
  ALL enrollment frames are used to train the production classifier.
  No frames are withheld — every angle and expression seen in enrollment
  contributes to recognition accuracy.

  Evaluation (evaluate.py) uses leave-one-out cross-validation on the
  training embeddings themselves, which gives an honest accuracy estimate
  without sacrificing any enrollment data.
"""

import os
import cv2
import numpy as np
import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from src.detector   import YOLOFaceDetector, FaceDetection
from src.aligner    import FaceAligner
from src.embedder   import FaceEmbedder
from src.classifier import EnsembleClassifier

logger = logging.getLogger(__name__)


class EnrollmentPipeline:
    """
    End-to-end enrollment: frame images → saved embeddings → trained classifier.

    Example:
        pipeline = EnrollmentPipeline(detector, aligner, embedder)
        pipeline.enroll_all("data/enrollment/", "data/embeddings/")
        pipeline.train_and_save("data/embeddings/", "models/classifier.pkl")
    """

    def __init__(self,
                 detector: YOLOFaceDetector,
                 aligner:  FaceAligner,
                 embedder: FaceEmbedder,
                 augment: bool = True):
        self.detector = detector
        self.aligner  = aligner
        self.embedder = embedder
        self.augment  = augment

    def enroll_all(self, enrollment_dir: str, output_dir: str) -> dict:
        """
        Process all student folders in enrollment_dir.

        Args:
            enrollment_dir: Root folder with per-student subfolders
            output_dir:     Where to write .npy arrays + metadata.json

        Returns:
            Summary dict keyed by student name.
        """
        enrollment_path = Path(enrollment_dir)
        output_path     = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        student_dirs = sorted([
            d for d in enrollment_path.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])

        if not student_dirs:
            raise ValueError(f"No student folders found in {enrollment_dir}")

        logger.info(f"[Enrollment] Found {len(student_dirs)} students.")

        all_train_embs: List[np.ndarray] = []
        all_train_lbls: List[int]        = []
        all_test_embs:  List[np.ndarray] = []
        all_test_lbls:  List[int]        = []

        names_map: Dict[int, str] = {}   # label_index → display name
        summary:   Dict           = {}

        for label_idx, student_dir in enumerate(student_dirs):
            name = self._folder_to_name(student_dir.name)
            names_map[label_idx] = name

            logger.info(f"[Enrollment] {name}  [label={label_idx}]")

            image_files = self._get_image_files(student_dir)
            if not image_files:
                logger.warning(f"  No images found for {name} — skipping.")
                continue

            # Use ALL frames for training — no data withheld.
            # Evaluation uses cross-validation on training embeddings (evaluate.py).
            logger.info(f"  {len(image_files)} images → all used for training")

            train_embs = self._process_images(image_files, augment=self.augment)
            test_embs  = []   # kept empty — evaluate.py does CV instead

            if not train_embs:
                logger.warning(f"  No valid embeddings for {name} — skipping.")
                continue

            all_train_embs.extend(train_embs)
            all_train_lbls.extend([label_idx] * len(train_embs))
            all_test_embs.extend(test_embs)
            all_test_lbls.extend([label_idx] * len(test_embs))

            summary[name] = {
                "label":            label_idx,
                "train_embeddings": len(train_embs),
                "total_images":     len(image_files),
            }

        # ── Save arrays ───────────────────────────────────────────────
        np.save(output_path / "train_embeddings.npy", np.array(all_train_embs, dtype=np.float32))
        np.save(output_path / "train_labels.npy",     np.array(all_train_lbls, dtype=np.int32))
        np.save(output_path / "test_embeddings.npy",
                np.array(all_test_embs, dtype=np.float32) if all_test_embs
                else np.zeros((0, 512), dtype=np.float32))
        np.save(output_path / "test_labels.npy",      np.array(all_test_lbls,  dtype=np.int32))

        metadata = {
            "names_map":               {str(k): v for k, v in names_map.items()},
            "summary":                 summary,
            "total_students":          len(names_map),
            "total_train_embeddings":  len(all_train_embs),
            "total_test_embeddings":   len(all_test_embs),
        }
        with open(output_path / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"[Enrollment] Saved to {output_path}")
        logger.info(f"  Train: {len(all_train_embs)} embs  |  Test: {len(all_test_embs)} embs")
        return summary

    def train_and_save(self, embeddings_dir: str, classifier_path: str,
                       confidence_threshold: float = 0.70,
                       min_cosine_sim:       float = 0.45,
                       global_nn_threshold:  float = 0.50):
        """Load saved embeddings and train + save the ensemble classifier."""
        emb_path = Path(embeddings_dir)

        embeddings = np.load(emb_path / "train_embeddings.npy")
        labels     = np.load(emb_path / "train_labels.npy")

        with open(emb_path / "metadata.json") as f:
            metadata = json.load(f)

        names_map = {int(k): v for k, v in metadata["names_map"].items()}

        clf = EnsembleClassifier(
            confidence_threshold=confidence_threshold,
            min_cosine_sim=min_cosine_sim,
            global_nn_threshold=global_nn_threshold,
        )
        clf.fit(embeddings, labels, names_map)
        clf.save(classifier_path)

        logger.info(f"[Enrollment] Classifier saved to {classifier_path}")
        return clf

    # ── Private helpers ───────────────────────────────────────────────

    def _process_images(self, image_files: List[Path],
                        augment: bool = False) -> List[np.ndarray]:
        """detect → align → embed for a list of image files."""
        embeddings = []
        for img_path in image_files:
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            dets = self.detector.detect(image)
            if not dets:
                continue
            best = max(dets, key=lambda d: d.confidence)
            aligned = self.aligner.align(image, best)
            if aligned is None:
                continue
            if augment:
                embeddings.extend(self.embedder.embed_with_augmentation(aligned))
            else:
                emb = self.embedder.embed(aligned)
                if emb is not None:
                    embeddings.append(emb)
        return embeddings

    @staticmethod
    def _folder_to_name(folder_name: str) -> str:
        """
        Convert folder name to display name.
        Underscores → spaces.  e.g. "John_Doe" → "John Doe"
        """
        return folder_name.replace("_", " ").strip()

    @staticmethod
    def _get_image_files(directory: Path) -> List[Path]:
        """Sorted image files — alphabetical order preserves frame sequence."""
        exts = {".jpg", ".jpeg", ".png", ".bmp"}
        return sorted([f for f in directory.iterdir() if f.suffix.lower() in exts])
