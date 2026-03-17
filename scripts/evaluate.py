"""
scripts/evaluate.py
====================
Evaluate classifier accuracy using Leave-One-Out Cross-Validation (LOOCV)
on the training embeddings — no held-out data needed.

WHY LOOCV INSTEAD OF TRAIN/TEST SPLIT?
  With limited enrollment data (20-50 frames per student), a 70/30 split
  wastes 30% of data that could help the production classifier recognise
  difficult angles. LOOCV evaluates honestly without sacrificing any frames:

    For each embedding e_i in the training set:
      1. Train a temporary classifier on all embeddings EXCEPT e_i
      2. Predict the label for e_i
      3. Record whether it was correct

  This gives an unbiased accuracy estimate while ALL embeddings remain
  available to the production classifier.

USAGE:
  python scripts/evaluate.py
  python scripts/evaluate.py --student "John Doe"   # one student only
"""

import sys, os, json, argparse
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import logging
from pathlib import Path
from collections import defaultdict

from config import PATHS, RECOGNITION
from src.utils      import setup_logging
from src.classifier import EnsembleClassifier


def loocv_accuracy(train_embs: np.ndarray, train_lbls: np.ndarray,
                   names_map: dict, target_label: int = None) -> dict:
    """
    Leave-One-Out Cross-Validation on training embeddings.

    For each embedding, trains a mini-classifier on everything else
    and predicts. Reports per-student and overall accuracy.

    Args:
        train_embs:   (N, 512) all training embeddings
        train_lbls:   (N,) corresponding labels
        names_map:    {label_idx: name}
        target_label: if set, only evaluate this one student

    Returns:
        dict with accuracy stats per student and overall
    """
    n = len(train_embs)
    correct   = defaultdict(int)
    total     = defaultdict(int)
    conf_sums = defaultdict(float)

    labels_to_eval = (
        [target_label] if target_label is not None
        else list(np.unique(train_lbls))
    )

    print(f"Running LOOCV on {n} embeddings...")

    for i in range(n):
        lbl = int(train_lbls[i])
        if lbl not in labels_to_eval:
            continue

        # Leave out embedding i — train on everything else
        mask       = np.arange(n) != i
        loo_embs   = train_embs[mask]
        loo_lbls   = train_lbls[mask]

        # Skip if only one class remains (can't train classifier)
        if len(np.unique(loo_lbls)) < 2:
            continue

        # Lightweight classifier for this fold
        clf = EnsembleClassifier(
            confidence_threshold=RECOGNITION.get("confidence_threshold", 0.70),
            min_cosine_sim=RECOGNITION.get("min_cosine_sim", 0.45),
            global_nn_threshold=RECOGNITION.get("global_nn_threshold", 0.50),
        )
        clf.fit(loo_embs, loo_lbls, names_map)
        result = clf.predict(train_embs[i])

        total[lbl]     += 1
        conf_sums[lbl] += result.confidence
        if result.name == names_map.get(lbl, ""):
            correct[lbl] += 1

        # Progress indicator
        if (i + 1) % max(1, n // 10) == 0:
            pct = (i + 1) / n * 100
            print(f"  {pct:.0f}%  ({i+1}/{n})", flush=True)

    # ── Compile results ────────────────────────────────────────────────
    per_student = {}
    for lbl in sorted(labels_to_eval):
        name     = names_map.get(lbl, f"label_{lbl}")
        n_total  = total[lbl]
        n_correct= correct[lbl]
        acc      = n_correct / n_total if n_total > 0 else 0.0
        avg_conf = conf_sums[lbl] / n_total if n_total > 0 else 0.0
        per_student[name] = {
            "accuracy":      round(acc, 4),
            "correct":       n_correct,
            "total":         n_total,
            "avg_confidence":round(avg_conf, 3),
        }

    all_correct = sum(correct.values())
    all_total   = sum(total.values())
    overall_acc = all_correct / all_total if all_total > 0 else 0.0

    return {
        "method":       "Leave-One-Out Cross-Validation",
        "overall":      round(overall_acc, 4),
        "per_student":  per_student,
        "total_folds":  all_total,
    }


def print_results(results: dict):
    print(f"\n{'═'*55}")
    print(f"  Evaluation: {results['method']}")
    print(f"  Overall Accuracy: {results['overall']*100:.1f}%  "
          f"({sum(s['correct'] for s in results['per_student'].values())}"
          f"/{results['total_folds']} correct)")
    print(f"{'═'*55}")
    print(f"  {'Student':<28} {'Acc':>6}  {'Correct':>8}  {'Avg Conf':>9}")
    print(f"  {'─'*52}")

    for name, info in sorted(results["per_student"].items(),
                              key=lambda x: x[1]["accuracy"]):
        acc_bar = "█" * int(info["accuracy"] * 10)
        status  = "✓" if info["accuracy"] >= 0.80 else "⚠"
        print(f"  {status} {name:<26} {info['accuracy']*100:>5.1f}%  "
              f"{info['correct']:>3}/{info['total']:<4}  "
              f"{info['avg_confidence']:>8.3f}  {acc_bar}")

    print(f"\n  Students with accuracy < 80% need more enrollment images.")
    print(f"  Run:  python scripts/enroll_one.py --folder data/enrollment/<name>")


def main():
    parser = argparse.ArgumentParser(description="Evaluate classifier via LOOCV")
    parser.add_argument("--student", type=str, default=None,
                        help="Evaluate one specific student only")
    args = parser.parse_args()

    setup_logging("WARNING", PATHS["log_file"])

    emb_dir = Path(PATHS["embeddings_dir"])
    if not (emb_dir / "train_embeddings.npy").exists():
        print("No embeddings found. Run scripts/run_enrollment.py first.")
        sys.exit(1)

    train_embs = np.load(emb_dir / "train_embeddings.npy")
    train_lbls = np.load(emb_dir / "train_labels.npy")

    with open(emb_dir / "metadata.json") as f:
        meta = json.load(f)
    names_map = {int(k): v for k, v in meta["names_map"].items()}

    print(f"Loaded {len(train_embs)} embeddings, {len(names_map)} students.")

    target_label = None
    if args.student:
        for idx, name in names_map.items():
            if name.lower() == args.student.lower():
                target_label = idx
                break
        if target_label is None:
            print(f"Student '{args.student}' not found.")
            print("Enrolled: " + ", ".join(names_map.values()))
            sys.exit(1)
        print(f"Evaluating: {args.student} only\n")

    results = loocv_accuracy(train_embs, train_lbls, names_map, target_label)
    print_results(results)


if __name__ == "__main__":
    main()
