"""
src/recognizer.py - Session Recognition & Attendance Aggregation
================================================================
Processes a classroom session (multiple row images) and produces
a consolidated attendance list.

No roll numbers — students identified by name only.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Dict, Optional, Set
import json
import csv
from datetime import datetime

from src.detector   import YOLOFaceDetector
from src.aligner    import FaceAligner
from src.embedder   import FaceEmbedder
from src.classifier import EnsembleClassifier, RecognitionResult, UNKNOWN_LABEL

logger = logging.getLogger(__name__)


@dataclass
class FaceResult:
    """Recognition result for one detected face in one image."""
    image_name:  str
    bbox:        np.ndarray
    recognition: RecognitionResult
    face_crop:   Optional[np.ndarray] = None


@dataclass
class SessionResult:
    """Aggregated attendance result for an entire session."""
    session_id:          str
    timestamp:           str
    present_students:    List[Dict]
    absent_students:     List[Dict]
    unknown_detections:  int
    total_detections:    int
    per_image_results:   Dict


class AttendanceRecognizer:
    """
    Runs recognition on a session and produces attendance.

    Majority voting rule:
      PRESENT if student appears in ≥ min_images_required session images
              OR  confidence > high_conf_override in any single image.
    """

    def __init__(self,
                 detector:           YOLOFaceDetector,
                 aligner:            FaceAligner,
                 embedder:           FaceEmbedder,
                 classifier:         EnsembleClassifier,
                 min_images_required: int   = 2,
                 high_conf_override:  float = 0.82):
        self.detector            = detector
        self.aligner             = aligner
        self.embedder            = embedder
        self.classifier          = classifier
        self.min_images_required = min_images_required
        self.high_conf_override  = high_conf_override

    def process_session(self,
                        session_dir:       str,
                        all_student_names: List[str]) -> SessionResult:
        """
        Process all images in a session directory.

        Args:
            session_dir:       Path to folder with session images
            all_student_names: Complete list of enrolled student names

        Returns:
            SessionResult with consolidated attendance.
        """
        session_path  = Path(session_dir)
        session_id    = session_path.name
        image_files   = sorted([
            f for f in session_path.iterdir()
            if f.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ])

        logger.info(f"[Recognizer] Session '{session_id}' — {len(image_files)} images")

        votes:            Dict[str, List[float]] = defaultdict(list)
        per_image_results: Dict                  = {}
        total_detections  = 0
        unknown_count     = 0

        for img_file in image_files:
            image = cv2.imread(str(img_file))
            if image is None:
                continue

            face_results = self._process_single_image(image, img_file.name)
            per_image_results[img_file.name] = face_results
            total_detections += len(face_results)

            seen: Set[str] = set()
            for fr in face_results:
                if not fr.recognition.is_known:
                    unknown_count += 1
                    continue
                name = fr.recognition.name
                if name not in seen:
                    votes[name].append(fr.recognition.confidence)
                    seen.add(name)

            logger.info(f"  [{img_file.name}] {len(face_results)} faces, "
                        f"{len(seen)} recognised")

        # ── Majority voting ───────────────────────────────────────────
        present, absent = [], []
        for name in all_student_names:
            student_votes = votes.get(name, [])
            images_seen   = len(student_votes)
            max_conf      = max(student_votes)  if student_votes else 0.0
            avg_conf      = float(np.mean(student_votes)) if student_votes else 0.0
            is_present    = (images_seen >= self.min_images_required or
                             max_conf >= self.high_conf_override)

            entry = dict(
                name=name,
                avg_confidence=round(avg_conf, 3),
                max_confidence=round(max_conf, 3),
                images_seen=images_seen,
                total_images=len(image_files),
                status="Present" if is_present else "Absent",
            )
            (present if is_present else absent).append(entry)

        logger.info(f"[Recognizer] {len(present)} present, {len(absent)} absent, "
                    f"{unknown_count} unknown detections.")

        return SessionResult(
            session_id=session_id,
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            present_students=present,
            absent_students=absent,
            unknown_detections=unknown_count,
            total_detections=total_detections,
            per_image_results={
                k: [self._face_result_to_dict(fr) for fr in v]
                for k, v in per_image_results.items()
            },
        )

    def _process_single_image(self, image: np.ndarray,
                               image_name: str) -> List[FaceResult]:
        """Full pipeline for one image: detect → align → embed → classify."""
        results = []
        for det in self.detector.detect(image):
            aligned = self.aligner.align(image, det)
            if aligned is None:
                continue
            emb = self.embedder.embed(aligned)
            if emb is None:
                continue
            rec = self.classifier.predict(emb)
            results.append(FaceResult(
                image_name=image_name,
                bbox=det.bbox,
                recognition=rec,
                face_crop=aligned,
            ))
        return results

    def save_attendance(self, result: SessionResult, output_dir: str):
        """Save attendance to CSV and JSON."""
        out_path  = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        base_name = (f"{result.session_id}_"
                     f"{result.timestamp.replace(':', '-').replace(' ', '_')}")

        # CSV — sorted by name
        csv_path = out_path / f"{base_name}_attendance.csv"
        all_rows = sorted(result.present_students + result.absent_students,
                          key=lambda s: s["name"])
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "name", "status", "images_seen",
                "total_images", "avg_confidence", "max_confidence"
            ])
            writer.writeheader()
            writer.writerows(all_rows)

        # JSON — full report
        json_path = out_path / f"{base_name}_report.json"
        with open(json_path, "w") as f:
            json.dump({
                "session_id": result.session_id,
                "timestamp":  result.timestamp,
                "summary": {
                    "total_enrolled":    len(result.present_students) + len(result.absent_students),
                    "present":           len(result.present_students),
                    "absent":            len(result.absent_students),
                    "unknown_faces":     result.unknown_detections,
                    "total_detections":  result.total_detections,
                },
                "present_students":   result.present_students,
                "absent_students":    result.absent_students,
                "per_image_results":  result.per_image_results,
            }, f, indent=2)

        logger.info(f"[Recognizer] Saved:\n  CSV  → {csv_path}\n  JSON → {json_path}")
        return csv_path, json_path

    @staticmethod
    def _face_result_to_dict(fr: FaceResult) -> dict:
        return {
            "image":         fr.image_name,
            "bbox":          fr.bbox.tolist(),
            "name":          fr.recognition.name,
            "confidence":    round(fr.recognition.confidence, 3),
            "svm_pred":      fr.recognition.svm_pred,
            "svm_conf":      round(fr.recognition.svm_conf, 3),
            "knn_pred":      fr.recognition.knn_pred,
            "knn_conf":      round(fr.recognition.knn_conf, 3),
            "agreed":        fr.recognition.agreed,
            "global_nn_sim": round(fr.recognition.global_nn_sim, 3),
            "reject_reason": fr.recognition.reject_reason,
        }
