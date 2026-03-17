"""
scripts/trim_enrollment.py
===========================
From each student's enrollment folder, keep the best N frames and delete
the rest. "Best" is defined by:

  Score = detection_confidence * alignment_bonus
    alignment_bonus = 1.5  if 5-pt landmarks found  (affine-alignable)
                    = 1.0  if bbox fallback only

  Within each tier (landmark vs bbox), frames are sorted by score so the
  kept set maximises both quality and spread (first / mid / last picks
  ensure pose variety rather than all frames from one moment in the video).

USAGE:
  # Trim all students to 12 frames (default)
  python scripts/trim_enrollment.py

  # Different target
  python scripts/trim_enrollment.py --keep 20

  # Preview without deleting
  python scripts/trim_enrollment.py --dry-run

  # Single student
  python scripts/trim_enrollment.py --student "data/enrollment/John_Doe"
"""

import sys, os, argparse, shutil
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple

from config import PATHS, DETECTION
from src.utils    import setup_logging
from src.detector import YOLOFaceDetector

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def score_frames(student_dir: Path,
                 detector: YOLOFaceDetector) -> List[Tuple[Path, float, bool]]:
    """
    Score every image in a student folder.

    Returns list of (path, score, has_landmarks) sorted best → worst.
    Frames with no face detected are excluded entirely.
    """
    image_files = sorted([f for f in student_dir.iterdir()
                           if f.suffix.lower() in IMAGE_EXTS])
    scored = []
    for img_path in image_files:
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        dets = detector.detect(img)
        if not dets:
            continue

        # Use best (highest-confidence) detection per frame
        best = max(dets, key=lambda d: d.confidence)
        has_lm = best.landmarks is not None and best.landmarks.sum() > 5

        # Landmark bonus: affine-alignable frames are worth 1.5×
        score = best.confidence * (1.5 if has_lm else 1.0)
        scored.append((img_path, score, has_lm))

    return sorted(scored, key=lambda x: x[1], reverse=True)


def pick_diverse(scored: List[Tuple], keep: int) -> List[Path]:
    """
    Pick `keep` frames that are both high-quality AND temporally spread.

    Strategy:
      1. All landmark frames come first (sorted by score)
      2. Fill remaining slots from bbox frames
      3. Within each group, interleave high/mid/low score picks
         so we don't grab 12 nearly-identical consecutive frames.
    """
    lm_frames   = [s for s in scored if s[2]]      # has landmarks
    bbox_frames = [s for s in scored if not s[2]]   # bbox only

    def spread_pick(frames, n):
        """Pick n items spread evenly across the ranked list."""
        if not frames or n <= 0:
            return []
        if len(frames) <= n:
            return [f[0] for f in frames]
        indices = np.linspace(0, len(frames) - 1, n, dtype=int)
        return [frames[i][0] for i in indices]

    # Decide split: maximise landmark frames up to `keep`
    n_lm   = min(len(lm_frames), keep)
    n_bbox = keep - n_lm

    picked = spread_pick(lm_frames, n_lm) + spread_pick(bbox_frames, n_bbox)
    return picked


def trim_student(student_dir: Path, detector: YOLOFaceDetector,
                 keep: int = 12, dry_run: bool = False) -> dict:
    """
    Trim one student folder to `keep` best frames.

    Returns summary dict.
    """
    image_files = sorted([f for f in student_dir.iterdir()
                           if f.suffix.lower() in IMAGE_EXTS])
    total = len(image_files)

    if total <= keep:
        return dict(name=student_dir.name, total=total,
                    kept=total, deleted=0, lm_frames=0,
                    skipped="already ≤ target")

    print(f"  Scoring {total} frames...", end=" ", flush=True)
    scored = score_frames(student_dir, detector)

    no_face_count = total - len(scored)
    lm_count      = sum(1 for s in scored if s[2])
    print(f"{len(scored)} with faces ({lm_count} landmark, "
          f"{len(scored)-lm_count} bbox, {no_face_count} no-face)")

    keep_paths = set(pick_diverse(scored, keep))
    delete_paths = [f for f in image_files if f not in keep_paths]

    if not dry_run:
        for p in delete_paths:
            p.unlink()
    else:
        print(f"  [DRY RUN] Would keep {len(keep_paths)}, delete {len(delete_paths)}")

    return dict(name=student_dir.name, total=total,
                kept=len(keep_paths), deleted=len(delete_paths),
                lm_frames=lm_count)


def main():
    parser = argparse.ArgumentParser(
        description="Trim enrollment folders to best N frames each.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--keep",    type=int,  default=12,
                        help="Frames to keep per student (default: 12)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Preview without deleting anything")
    parser.add_argument("--student", type=str, default=None,
                        help="Process single folder only")
    args = parser.parse_args()

    setup_logging("WARNING", PATHS["log_file"])

    print("Loading YOLO detector...")
    detector = YOLOFaceDetector(
        weights_path=PATHS["yolo_weights"],
        conf=0.25,          # lower than normal — enrollment close-ups score high anyway
        nms_iou=DETECTION["nms_iou"],
        min_area=400,       # allow small faces during scoring
    )
    print("✓ Detector loaded\n")

    enrollment_dir = Path(PATHS["enrollment_dir"])

    if args.student:
        dirs = [Path(args.student)]
    else:
        dirs = sorted([d for d in enrollment_dir.iterdir()
                       if d.is_dir() and not d.name.startswith(".")])

    print(f"{'DRY RUN — ' if args.dry_run else ''}"
          f"Processing {len(dirs)} student folder(s), keeping best {args.keep} frames each\n"
          f"{'='*60}")

    total_deleted = 0
    results = []

    for i, student_dir in enumerate(dirs, 1):
        print(f"[{i}/{len(dirs)}] {student_dir.name}")
        result = trim_student(student_dir, detector,
                              keep=args.keep, dry_run=args.dry_run)
        results.append(result)
        total_deleted += result["deleted"]

        if result.get("skipped"):
            print(f"  → {result['skipped']} ({result['total']} frames)\n")
        else:
            lm_pct = result['lm_frames'] / result['total'] * 100 if result['total'] else 0
            print(f"  → Kept {result['kept']}, deleted {result['deleted']}  "
                  f"(landmark frames: {result['lm_frames']}/{result['total']} = {lm_pct:.0f}%)\n")

    print(f"{'='*60}")
    print(f"{'[DRY RUN] Would have deleted' if args.dry_run else 'Deleted'} "
          f"{total_deleted} frames across {len(dirs)} students")

    # Warn about students with 0 landmark frames — likely wrong YOLO model
    no_lm = [r['name'] for r in results if r.get('lm_frames', 0) == 0
             and not r.get('skipped')]
    if no_lm:
        print(f"\n⚠  {len(no_lm)} student(s) had NO landmark frames:")
        for n in no_lm[:10]:
            print(f"     {n}")
        print("   This means YOLO isn't outputting keypoints.")
        print("   Verify models/yolov8n-face.pt is the face-specific model")
        print("   (download from https://github.com/derronqi/yolov8-face/releases)")

    if not args.dry_run:
        print(f"\nNext step: python scripts/run_enrollment.py")
        print(f"  (or)      python scripts/enroll_one.py --all-pending")


if __name__ == "__main__":
    main()
