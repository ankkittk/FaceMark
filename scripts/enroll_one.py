"""
scripts/enroll_one.py
======================
Incrementally enroll a single student without re-processing everyone else.

Appends the new student's embeddings to the existing .npy arrays, then
retrains only the classifier (fast — seconds, not minutes).

USAGE:
  # Enroll from a video file
  python scripts/enroll_one.py --video "C:/path/to/VID - John Doe.mp4"

  # Enroll from an already-extracted frame folder
  python scripts/enroll_one.py --folder "data/enrollment/John_Doe"

  # List currently enrolled students
  python scripts/enroll_one.py --list

  # Remove a student (useful for correcting mistakes)
  python scripts/enroll_one.py --remove "John Doe"

WORKFLOW:
  First time (no existing embeddings):
    python scripts/run_enrollment.py        ← processes everyone at once

  Adding more students later:
    python scripts/enroll_one.py --video "path/VID - Jane Smith.mp4"
    python scripts/enroll_one.py --video "path/VID - Bob Jones.mp4"
    ...each call takes ~1-2 minutes per student
"""

import sys, os, json, argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import numpy as np
from pathlib import Path

from config import PATHS, DETECTION, EMBEDDING, RECOGNITION, EVALUATION
from src.utils      import setup_logging
from src.detector   import YOLOFaceDetector
from src.aligner    import FaceAligner
from src.embedder   import FaceEmbedder
from src.enrollment import EnrollmentPipeline
from src.classifier import EnsembleClassifier


def load_existing(embeddings_dir: Path):
    """Load current embeddings, labels and metadata. Returns empty state if none exist."""
    meta_path = embeddings_dir / "metadata.json"

    if not meta_path.exists():
        return (
            np.zeros((0, 512), dtype=np.float32),  # train_embs
            np.zeros((0,),     dtype=np.int32),     # train_lbls
            np.zeros((0, 512), dtype=np.float32),  # test_embs
            np.zeros((0,),     dtype=np.int32),     # test_lbls
            {},                                      # names_map  {idx: name}
            {},                                      # summary
        )

    train_embs = np.load(embeddings_dir / "train_embeddings.npy")
    train_lbls = np.load(embeddings_dir / "train_labels.npy")
    test_path  = embeddings_dir / "test_embeddings.npy"
    test_embs  = np.load(test_path) if test_path.exists() else np.zeros((0, 512), dtype=np.float32)
    test_path2 = embeddings_dir / "test_labels.npy"
    test_lbls  = np.load(test_path2) if test_path2.exists() else np.zeros((0,), dtype=np.int32)

    with open(meta_path) as f:
        meta = json.load(f)

    names_map = {int(k): v for k, v in meta["names_map"].items()}
    summary   = meta.get("summary", {})
    return train_embs, train_lbls, test_embs, test_lbls, names_map, summary


def save_all(embeddings_dir: Path, train_embs, train_lbls,
             test_embs, test_lbls, names_map, summary):
    """Overwrite .npy arrays and metadata.json."""
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    np.save(embeddings_dir / "train_embeddings.npy", train_embs)
    np.save(embeddings_dir / "train_labels.npy",     train_lbls)
    np.save(embeddings_dir / "test_embeddings.npy",  test_embs)
    np.save(embeddings_dir / "test_labels.npy",      test_lbls)

    meta = {
        "names_map":              {str(k): v for k, v in names_map.items()},
        "summary":                summary,
        "total_students":         len(names_map),
        "total_train_embeddings": int(len(train_embs)),
        "total_test_embeddings":  int(len(test_embs)),
    }
    with open(embeddings_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)


def retrain_classifier(embeddings_dir: Path, classifier_path: Path,
                        names_map: dict, train_embs, train_lbls):
    """Retrain SVM+kNN on the full (now updated) embedding set."""
    print("Retraining classifier... ", end="", flush=True)
    clf = EnsembleClassifier(
        confidence_threshold=RECOGNITION.get("confidence_threshold", 0.70),
        min_cosine_sim=RECOGNITION.get("min_cosine_sim", 0.45),
        global_nn_threshold=RECOGNITION.get("global_nn_threshold", 0.50),
    )
    clf.fit(train_embs, train_lbls, names_map)
    clf.save(str(classifier_path))
    print("✓")
    return clf


def enroll_student(name: str, student_dir: Path, pipeline: EnrollmentPipeline,
                   label_idx: int) -> tuple:
    """
    Extract embeddings for one student folder.
    Returns (train_embs, test_embs) as numpy arrays.
    """
    image_exts  = {".jpg", ".jpeg", ".png", ".bmp"}
    image_files = sorted([f for f in student_dir.iterdir()
                          if f.suffix.lower() in image_exts])

    if not image_files:
        raise ValueError(f"No images found in {student_dir}")

    # All frames go to training — no data withheld
    print(f"  Images: {len(image_files)} (all used for training)")

    train_embs = pipeline._process_images(image_files, augment=EMBEDDING["augment_enrollment"])
    test_embs  = []

    if not train_embs:
        raise ValueError(f"No valid face embeddings extracted for {name}. "
                         f"Check that images contain clear frontal faces.")

    print(f"  Embeddings: {len(train_embs)} train, {len(test_embs)} test")
    return (
        np.array(train_embs, dtype=np.float32),
        np.array(test_embs,  dtype=np.float32) if test_embs
        else np.zeros((0, 512), dtype=np.float32),
    )


def cmd_list(embeddings_dir: Path):
    """Print currently enrolled students."""
    _, _, _, _, names_map, summary = load_existing(embeddings_dir)
    if not names_map:
        print("No students enrolled yet.")
        return
    print(f"\nEnrolled students ({len(names_map)}):\n{'─'*45}")
    for idx in sorted(names_map.keys()):
        name = names_map[idx]
        info = summary.get(name, {})
        tr   = info.get("train_embeddings", "?")
        te   = info.get("test_embeddings",  "?")
        print(f"  [{idx}] {name:<30}  {tr} train embs  /  {te} test embs")
    print()


def cmd_remove(name_to_remove: str, embeddings_dir: Path,
               classifier_path: Path):
    """Remove a student from the embedding database and retrain."""
    train_embs, train_lbls, test_embs, test_lbls, names_map, summary = \
        load_existing(embeddings_dir)

    if not names_map:
        print("No students enrolled yet."); return

    # Find label index for this name (case-insensitive)
    target_idx = None
    for idx, name in names_map.items():
        if name.lower() == name_to_remove.lower().replace("_", " "):
            target_idx = idx; break

    if target_idx is None:
        print(f"Student '{name_to_remove}' not found.")
        print("Enrolled: " + ", ".join(names_map.values()))
        return

    target_name = names_map[target_idx]
    print(f"Removing: {target_name} (label {target_idx})")

    # Remove embeddings
    keep_train  = train_lbls != target_idx
    keep_test   = test_lbls  != target_idx
    train_embs  = train_embs[keep_train]
    train_lbls  = train_lbls[keep_train]
    test_embs   = test_embs[keep_test]
    test_lbls   = test_lbls[keep_test]

    del names_map[target_idx]
    summary.pop(target_name, None)

    save_all(embeddings_dir, train_embs, train_lbls,
             test_embs, test_lbls, names_map, summary)

    if len(names_map) >= 2:
        retrain_classifier(embeddings_dir, classifier_path, names_map,
                           train_embs, train_lbls)
    else:
        print("Need ≥2 students to train classifier.")

    print(f"✓ Removed {target_name}. {len(names_map)} students remain.")



def _enroll_batch(folder_list, embeddings_dir, classifier_path,
                  enrollment_dir, args):
    """
    Enroll a list of student folders sequentially.
    Models are loaded once and reused for all students.
    Classifier is retrained once at the very end (not after each student).
    """
    print("Loading models (once for all students)...")
    detector = YOLOFaceDetector(
        weights_path=PATHS["yolo_weights"],
        conf=DETECTION["conf_threshold"],
        nms_iou=DETECTION["nms_iou"],
        min_area=DETECTION["min_face_area"],
    )
    aligner  = FaceAligner()
    embedder = FaceEmbedder(model_name=EMBEDDING["model_name"])
    pipeline = EnrollmentPipeline(detector, aligner, embedder,
                                   augment=EMBEDDING["augment_enrollment"])
    print(f"✓ Models loaded — processing {len(folder_list)} students\n")

    succeeded, failed = [], []

    for i, student_dir in enumerate(folder_list, 1):
        student_name = student_dir.name.replace("_", " ")
        print(f"[{i}/{len(folder_list)}] {student_name}")

        # Load fresh state each iteration (previous student may have added data)
        train_embs, train_lbls, test_embs, test_lbls, names_map, summary = \
            load_existing(embeddings_dir)

        # Skip if already enrolled
        enrolled_lower = {v.lower(): k for k, v in names_map.items()}
        if student_name.lower() in enrolled_lower:
            print(f"  Already enrolled — skipping.\n")
            continue

        if not student_dir.exists():
            print(f"  Folder not found: {student_dir} — skipping.\n")
            failed.append(student_name)
            continue

        new_idx = max(names_map.keys()) + 1 if names_map else 0

        try:
            new_train_embs, new_test_embs = enroll_student(
                student_name, student_dir, pipeline, new_idx
            )
        except Exception as e:
            print(f"  ERROR: {e} — skipping.\n")
            failed.append(student_name)
            continue

        # Append
        new_train_lbls = np.full(len(new_train_embs), new_idx, dtype=np.int32)
        new_test_lbls  = np.full(len(new_test_embs),  new_idx, dtype=np.int32)

        train_embs = np.vstack([train_embs, new_train_embs]) if len(train_embs) else new_train_embs
        train_lbls = np.concatenate([train_lbls, new_train_lbls])
        test_embs  = np.vstack([test_embs, new_test_embs]) if len(test_embs) and len(new_test_embs) \
                     else (new_test_embs if len(new_test_embs) else test_embs)
        test_lbls  = np.concatenate([test_lbls, new_test_lbls])

        names_map[new_idx] = student_name
        summary[student_name] = {
            "label":            new_idx,
            "train_embeddings": len(new_train_embs),
        }

        # Save embeddings after each student (safe to interrupt)
        save_all(embeddings_dir, train_embs, train_lbls,
                 test_embs, test_lbls, names_map, summary)
        succeeded.append(student_name)
        print(f"  ✓ Saved. Total enrolled so far: {len(names_map)}\n")

    # Retrain classifier once at the end
    train_embs, train_lbls, _, _, names_map, _ = load_existing(embeddings_dir)
    if len(names_map) >= 2:
        retrain_classifier(embeddings_dir, classifier_path, names_map,
                           train_embs, train_lbls)

    print(f"\n{'='*50}")
    print(f"Batch complete.  ✓ {len(succeeded)} enrolled  |  ✗ {len(failed)} failed")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    print(f"Total enrolled: {len(names_map)} students")


def main():
    parser = argparse.ArgumentParser(
        description="Incrementally enroll students.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video",  type=str,
                       help="Single video file")
    group.add_argument("--folder", type=str,
                       help="Single frame folder")
    group.add_argument("--folders", type=str, nargs="+", metavar="FOLDER",
                       help="Multiple frame folders enrolled sequentially")
    group.add_argument("--all-pending", action="store_true",
                       help="Enroll all folders in data/enrollment/ not yet enrolled")
    group.add_argument("--list",   action="store_true",
                       help="List currently enrolled students")
    group.add_argument("--remove", type=str, metavar="NAME",
                       help="Remove a student by name")

    parser.add_argument("--max-frames",  type=int, default=150)
    parser.add_argument("--skip-frames", type=int, default=5)
    args = parser.parse_args()

    setup_logging("WARNING", PATHS["log_file"])   # quiet — only show prints

    embeddings_dir  = Path(PATHS["embeddings_dir"])
    classifier_path = Path(PATHS["classifier"])
    enrollment_dir  = Path(PATHS["enrollment_dir"])

    # ── List / Remove (no model loading needed) ───────────────────────
    if args.list:
        cmd_list(embeddings_dir); return

    if args.remove:
        cmd_remove(args.remove, embeddings_dir, classifier_path); return

    # ── Resolve folder list for --folders and --all-pending ───────────
    if args.folders:
        folder_list = [Path(f) for f in args.folders]
        _enroll_batch(folder_list, embeddings_dir, classifier_path,
                      enrollment_dir, args)
        return

    if args.all_pending:
        _, _, _, _, names_map, _ = load_existing(embeddings_dir)
        enrolled_names = {v.lower().replace(" ", "_") for v in names_map.values()}
        all_dirs = sorted([
            d for d in enrollment_dir.iterdir()
            if d.is_dir() and not d.name.startswith(".")
        ])
        pending = [d for d in all_dirs
                   if d.name.lower() not in enrolled_names]
        if not pending:
            print("All folders in data/enrollment/ are already enrolled.")
            return
        print(f"Found {len(pending)} unenrolled folders:")
        for d in pending:
            print(f"  {d.name}")
        print()
        _enroll_batch(pending, embeddings_dir, classifier_path,
                      enrollment_dir, args)
        return

    # ── Enroll one student ────────────────────────────────────────────
    print("Loading models...")
    detector = YOLOFaceDetector(
        weights_path=PATHS["yolo_weights"],
        conf=DETECTION["conf_threshold"],
        nms_iou=DETECTION["nms_iou"],
        min_area=DETECTION["min_face_area"],
    )
    aligner  = FaceAligner()
    embedder = FaceEmbedder(model_name=EMBEDDING["model_name"])
    pipeline = EnrollmentPipeline(detector, aligner, embedder,
                                   augment=EMBEDDING["augment_enrollment"])
    print("✓ Models loaded\n")

    # ── Determine student name and frame folder ───────────────────────
    if args.video:
        from scripts.extract_frames import parse_name_from_video, extract_for_video
        video_path   = Path(args.video)
        student_name = parse_name_from_video(video_path).replace("_", " ")
        folder_name  = student_name.replace(" ", "_")
        student_dir  = enrollment_dir / folder_name

        if student_dir.exists() and any(student_dir.iterdir()):
            print(f"Frame folder already exists: {student_dir}")
            ans = input("Re-extract frames? [y/N]: ").strip().lower()
            if ans == "y":
                import shutil; shutil.rmtree(student_dir)

        if not student_dir.exists() or not any(student_dir.iterdir()):
            print(f"Extracting frames from video...")
            _, count = extract_for_video(
                video_path, enrollment_dir,
                max_frames=args.max_frames,
                skip_frames=args.skip_frames,
            )
            print(f"  {count} frames extracted → {student_dir}")

    else:
        student_dir  = Path(args.folder)
        folder_name  = student_dir.name
        student_name = folder_name.replace("_", " ")

    if not student_dir.exists():
        print(f"Folder not found: {student_dir}"); sys.exit(1)

    print(f"Student : {student_name}")
    print(f"Folder  : {student_dir}\n")

    # ── Load existing state ───────────────────────────────────────────
    train_embs, train_lbls, test_embs, test_lbls, names_map, summary = \
        load_existing(embeddings_dir)

    # Check for duplicate
    existing_names_lower = {v.lower(): k for k, v in names_map.items()}
    if student_name.lower() in existing_names_lower:
        existing_idx = existing_names_lower[student_name.lower()]
        print(f"⚠  '{student_name}' is already enrolled (label {existing_idx}).")
        ans = input("Re-enroll (overwrites existing embeddings)? [y/N]: ").strip().lower()
        if ans != "y":
            print("Aborted."); return
        # Remove old embeddings first
        cmd_remove(student_name, embeddings_dir, classifier_path)
        train_embs, train_lbls, test_embs, test_lbls, names_map, summary = \
            load_existing(embeddings_dir)

    # Assign next label index
    new_idx = max(names_map.keys()) + 1 if names_map else 0
    print(f"Assigning label index: {new_idx}\n")

    # ── Extract embeddings ────────────────────────────────────────────
    print(f"Processing {student_name}...")
    new_train_embs, new_test_embs = enroll_student(
        student_name, student_dir, pipeline, new_idx
    )

    # ── Append to arrays ──────────────────────────────────────────────
    new_train_lbls = np.full(len(new_train_embs), new_idx, dtype=np.int32)
    new_test_lbls  = np.full(len(new_test_embs),  new_idx, dtype=np.int32)

    train_embs = np.vstack([train_embs, new_train_embs]) if len(train_embs) else new_train_embs
    train_lbls = np.concatenate([train_lbls, new_train_lbls])
    test_embs  = np.vstack([test_embs, new_test_embs]) if len(test_embs) and len(new_test_embs) else (
                 new_test_embs if len(new_test_embs) else test_embs)
    test_lbls  = np.concatenate([test_lbls, new_test_lbls])

    names_map[new_idx] = student_name
    summary[student_name] = {
        "label":            new_idx,
        "train_embeddings": len(new_train_embs),
    }

    # ── Save & retrain ────────────────────────────────────────────────
    save_all(embeddings_dir, train_embs, train_lbls,
             test_embs, test_lbls, names_map, summary)
    print(f"Embeddings saved.")

    if len(names_map) >= 2:
        retrain_classifier(embeddings_dir, classifier_path, names_map,
                           train_embs, train_lbls)
    else:
        print(f"Only 1 student enrolled — need ≥2 to train classifier.")

    print(f"\n✓ Done. Total enrolled: {len(names_map)} students")
    print(f"  {', '.join(names_map.values())}")


if __name__ == "__main__":
    main()
