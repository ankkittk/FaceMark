"""
scripts/run_enrollment.py
==========================
Run the full enrollment pipeline to process student images and train
the classifier. Run this ONCE before using the attendance system.

USAGE:
  python scripts/run_enrollment.py

PREREQUISITES:
  1. Place student images in data/enrollment/<Name_RollNo>/ folders
  2. Download yolov8n-face.pt to models/
  3. Install requirements: pip install -r requirements.txt

OUTPUT:
  - data/embeddings/train_embeddings.npy
  - data/embeddings/train_labels.npy
  - data/embeddings/test_embeddings.npy  (held-out, for evaluation)
  - data/embeddings/test_labels.npy
  - data/embeddings/metadata.json
  - models/classifier.pkl
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
from config import PATHS, DETECTION, EMBEDDING, CLASSIFIER, RECOGNITION, EVALUATION
from src.utils import setup_logging, ensure_directories
from src.detector import YOLOFaceDetector
from src.aligner import FaceAligner
from src.embedder import FaceEmbedder
from src.enrollment import EnrollmentPipeline


def main():
    setup_logging("INFO", PATHS["log_file"])
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("  ATTENDANCE SYSTEM — ENROLLMENT PHASE")
    logger.info("=" * 60)

    # Create output directories
    ensure_directories(PATHS)

    # ── Initialize pipeline components ───────────────────────────────────────
    logger.info("Loading models...")

    detector = YOLOFaceDetector(
        weights_path=PATHS["yolo_weights"],
        conf=DETECTION["conf_threshold"],
        nms_iou=DETECTION["nms_iou"],
        min_area=DETECTION["min_face_area"],
    )

    aligner = FaceAligner()

    embedder = FaceEmbedder(model_name=EMBEDDING["model_name"])

    pipeline = EnrollmentPipeline(
        detector=detector,
        aligner=aligner,
        embedder=embedder,
        augment=EMBEDDING["augment_enrollment"],
        temporal_split=1.0 - EVALUATION["test_split"],  # e.g. 0.70
    )

    # ── Run enrollment ────────────────────────────────────────────────────────
    logger.info(f"Processing enrollment folder: {PATHS['enrollment_dir']}")

    summary = pipeline.enroll_all(
        enrollment_dir=PATHS["enrollment_dir"],
        output_dir=PATHS["embeddings_dir"],
    )

    # Print per-student summary
    logger.info("\n── Enrollment Summary ──")
    for name, info in summary.items():
        logger.info(f"  {name} [{info['roll']}]: "
                    f"{info['train_embeddings']} train embs, "
                    f"{info['test_embeddings']} test embs")

    # ── Train classifier ──────────────────────────────────────────────────────
    logger.info("\nTraining SVM + kNN ensemble classifier...")

    pipeline.train_and_save(
        embeddings_dir=PATHS["embeddings_dir"],
        classifier_path=PATHS["classifier"],
        confidence_threshold=RECOGNITION["confidence_threshold"],
        min_cosine_sim=RECOGNITION["min_cosine_sim"],
    )

    logger.info("\n✓ Enrollment complete. Run scripts/run_attendance.py for a session.")


if __name__ == "__main__":
    main()
