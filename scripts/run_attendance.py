"""
scripts/run_attendance.py
==========================
Run attendance recognition for a classroom session.

USAGE:
  # Process a specific session folder:
  python scripts/run_attendance.py --session data/classroom_sessions/session_001

  # Process all sessions in the sessions directory:
  python scripts/run_attendance.py --all

PREREQUISITES:
  - run_enrollment.py must have been run first
  - Session images must be in data/classroom_sessions/<session_id>/

OUTPUT:
  - outputs/attendance_reports/<session_id>_<timestamp>_attendance.csv
  - outputs/attendance_reports/<session_id>_<timestamp>_report.json
"""

import sys
import os
import argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import logging
from pathlib import Path

from config import PATHS, DETECTION, EMBEDDING, RECOGNITION, AGGREGATION
from src.utils import setup_logging, ensure_directories
from src.detector import YOLOFaceDetector
from src.aligner import FaceAligner
from src.embedder import FaceEmbedder
from src.classifier import EnsembleClassifier
from src.recognizer import AttendanceRecognizer


def load_student_lists(embeddings_dir: str):
    """Load all student names and roll numbers from metadata."""
    metadata_path = Path(embeddings_dir) / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"metadata.json not found in {embeddings_dir}.\n"
            f"Run scripts/run_enrollment.py first."
        )
    with open(metadata_path) as f:
        metadata = json.load(f)

    names_map   = {int(k): v for k, v in metadata["names_map"].items()}
    rollnos_map = {int(k): v for k, v in metadata["rollnos_map"].items()}

    # Build sorted lists
    indices = sorted(names_map.keys())
    names   = [names_map[i] for i in indices]
    rolls   = [rollnos_map[i] for i in indices]
    return names, rolls


def process_session(recognizer: AttendanceRecognizer,
                    session_path: Path,
                    all_names: list,
                    all_rolls: list,
                    output_dir: str):
    """Process one session directory."""
    logger = logging.getLogger(__name__)
    logger.info(f"\n{'='*50}")
    logger.info(f"Processing session: {session_path.name}")

    result = recognizer.process_session(
        session_dir=str(session_path),
        all_student_names=all_names,
        all_roll_nos=all_rolls,
    )

    csv_path, json_path = recognizer.save_attendance(result, output_dir)

    # Print summary to console
    print(f"\n{'='*50}")
    print(f"Session: {result.session_id}")
    print(f"Timestamp: {result.timestamp}")
    print(f"{'─'*50}")
    print(f"Present ({len(result.present_students)}):")
    for s in sorted(result.present_students, key=lambda x: x['roll_no']):
        print(f"  ✓ [{s['roll_no']}] {s['name']:<25} "
              f"conf={s['avg_confidence']:.2f}  seen in {s['images_seen']}/{s['total_images']} images")

    print(f"\nAbsent ({len(result.absent_students)}):")
    for s in sorted(result.absent_students, key=lambda x: x['roll_no']):
        print(f"  ✗ [{s['roll_no']}] {s['name']}")

    print(f"\nUnknown faces detected: {result.unknown_detections}")
    print(f"Attendance saved to: {csv_path}")
    print(f"{'='*50}\n")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Run classroom attendance recognition"
    )
    parser.add_argument("--session", type=str, default=None,
                        help="Path to a specific session folder")
    parser.add_argument("--all", action="store_true",
                        help="Process all sessions in sessions_dir")
    args = parser.parse_args()

    setup_logging("INFO", PATHS["log_file"])
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info("  ATTENDANCE SYSTEM — RECOGNITION PHASE")
    logger.info("=" * 60)

    ensure_directories(PATHS)

    # ── Verify classifier exists ──────────────────────────────────────────────
    if not Path(PATHS["classifier"]).exists():
        logger.error(
            f"Classifier not found at {PATHS['classifier']}.\n"
            f"Run scripts/run_enrollment.py first."
        )
        sys.exit(1)

    # ── Load models ───────────────────────────────────────────────────────────
    logger.info("Loading models...")

    detector   = YOLOFaceDetector(
        PATHS["yolo_weights"],
        conf=DETECTION["conf_threshold"],
        nms_iou=DETECTION["nms_iou"],
        min_area=DETECTION["min_face_area"],
    )
    aligner    = FaceAligner()
    embedder   = FaceEmbedder(EMBEDDING["model_name"])
    classifier = EnsembleClassifier.load(PATHS["classifier"])

    recognizer = AttendanceRecognizer(
        detector=detector,
        aligner=aligner,
        embedder=embedder,
        classifier=classifier,
        min_images_required=AGGREGATION["min_images_required"],
        high_conf_override=AGGREGATION["high_conf_override"],
    )

    all_names, all_rolls = load_student_lists(PATHS["embeddings_dir"])
    logger.info(f"Loaded {len(all_names)} enrolled students.")

    # ── Process sessions ──────────────────────────────────────────────────────
    if args.session:
        session_path = Path(args.session)
        if not session_path.exists():
            logger.error(f"Session path not found: {args.session}")
            sys.exit(1)
        process_session(recognizer, session_path, all_names, all_rolls, PATHS["output_dir"])

    elif args.all:
        sessions_path = Path(PATHS["sessions_dir"])
        session_dirs = sorted([d for d in sessions_path.iterdir() if d.is_dir()])
        if len(session_dirs) == 0:
            logger.warning(f"No session folders found in {PATHS['sessions_dir']}")
        for session_dir in session_dirs:
            process_session(recognizer, session_dir, all_names, all_rolls, PATHS["output_dir"])

    else:
        parser.print_help()
        print("\nExample: python scripts/run_attendance.py --session data/classroom_sessions/session_001")


if __name__ == "__main__":
    main()
