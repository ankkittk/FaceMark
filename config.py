"""
config.py - Central Configuration File
=======================================
All tunable parameters live here. No magic numbers in code.
Modify this file to adapt the system to your hardware/dataset.

ARCHITECTURE SUMMARY:
  Image → YOLO (face detection) → Alignment (5-pt affine)
       → ArcFace (frozen, 512-D embedding) → Ensemble (SVM + kNN)
       → Confidence thresholding → Majority voting → Attendance CSV
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── File Paths ──────────────────────────────────────────────────────────────
PATHS = {
    # Enrollment data: data/enrollment/<StudentName_RollNo>/<images>
    "enrollment_dir": os.path.join(BASE_DIR, "data", "enrollment"),

    # Session images: data/classroom_sessions/<session_id>/<row1.jpg ...>
    "sessions_dir":   os.path.join(BASE_DIR, "data", "classroom_sessions"),

    # Saved numpy arrays: embeddings.npy, labels.npy, names.npy, rollnos.npy
    "embeddings_dir": os.path.join(BASE_DIR, "data", "embeddings"),

    # YOLOv8-face weights. Download from:
    # https://github.com/derronqi/yolov8-face/releases (yolov8n-face.pt)
    "yolo_weights":   os.path.join(BASE_DIR, "models", "yolov8n-face.pt"),

    # Trained ensemble classifier saved after enrollment phase
    "classifier":     os.path.join(BASE_DIR, "models", "classifier.pkl"),

    # Attendance CSV reports
    "output_dir":     os.path.join(BASE_DIR, "outputs", "attendance_reports"),

    # Log file
    "log_file":       os.path.join(BASE_DIR, "outputs", "system.log"),
}

# ── YOLO Detection ───────────────────────────────────────────────────────────
DETECTION = {
    # Face confidence threshold.
    # ↓ Lowered to 0.30 — classroom images have small/angled faces in back rows
    #   that a stricter threshold misses entirely.
    "conf_threshold": 0.30,

    # Non-max suppression IoU. Removes duplicate boxes for same face.
    "nms_iou": 0.45,

    # Min face size (pixels²).
    # ↓ Lowered from 1600 (40×40) to 625 (25×25) — back-row students appear
    #   small. ArcFace can still embed 25×25 crops after bicubic upscale.
    "min_face_area": 625,

    # YOLO input size. 640 is standard for YOLOv8. Don't change.
    "input_size": 640,
}

# ── Face Alignment ───────────────────────────────────────────────────────────
ALIGNMENT = {
    # ArcFace was trained on exactly 112×112 aligned faces.
    # Changing this breaks embedding compatibility.
    "face_size": (112, 112),

    # Canonical 5-point landmark positions for the 112×112 target space.
    # [left_eye, right_eye, nose_tip, left_mouth, right_mouth]
    # Source: InsightFace reference implementation.
    "arcface_template": [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
}

# ── ArcFace Embedding ────────────────────────────────────────────────────────
EMBEDDING = {
    # InsightFace model name. 'buffalo_l' = ResNet100, highest accuracy.
    # 'buffalo_s' is faster but slightly less accurate.
    # For 80 students offline, use 'buffalo_l'.
    "model_name": "buffalo_l",

    # Fixed by ArcFace architecture. Do not change.
    "dim": 512,

    # L2-normalize all embeddings. REQUIRED for cosine similarity to work.
    "normalize": True,

    # Augment enrollment images before extracting embeddings.
    # This synthetically expands the embedding database (like having more angles).
    # Highly recommended — improves SVM boundary quality with no model changes.
    "augment_enrollment": True,
}

# ── Classifier (SVM + kNN Ensemble) ─────────────────────────────────────────
CLASSIFIER = {
    # SVM: C=10 works well for 512-D normalized embeddings with 80 classes.
    # Tune with GridSearchCV in evaluate.py if needed.
    "svm_C": 10.0,
    "svm_kernel": "rbf",        # RBF handles residual non-linearity in embedding space
    "svm_gamma": "scale",       # 1/(n_features * X.var()) — adapts to your data
    "svm_class_weight": "balanced",  # Compensates for unequal frames per student
    "svm_probability": True,    # Required for confidence thresholding

    # kNN: k=5 gives stable votes; cosine metric suits L2-normalized embeddings
    "knn_k": 5,
    "knn_metric": "cosine",

    # Ensemble: when SVM and kNN agree → use their average confidence
    #           when they disagree → pick the one with higher confidence
    "ensemble": True,
}

# ── Recognition Thresholds ───────────────────────────────────────────────────
RECOGNITION = {
    # Gate 2: min ensemble confidence to accept a prediction.
    # ↓ Lowered from 0.70 — bbox fallback (no landmarks) produces lower-confidence
    #   embeddings. With 62 students, the SVM is less decisive too.
    "confidence_threshold": 0.55,

    # Gate 3: min cosine sim to the predicted student's own embeddings.
    # ↓ Lowered from 0.45 — bbox-cropped faces have higher pose variance.
    "min_cosine_sim": 0.28,

    # Gate 1 (PRIMARY): min cosine sim to ANY enrolled face.
    # ↓ Lowered from 0.50 — this was rejecting most real students.
    #   With 62 students + bbox fallback, within-class sim drops significantly.
    #   Run the threshold-tuning notebook cell to find your exact sweet spot.
    "global_nn_threshold": 0.746,
}

# ── Attendance Aggregation ───────────────────────────────────────────────────
AGGREGATION = {
    # Student must appear in this many session images to be marked present.
    # ↓ Lowered from 2 → 1. With recognition rate currently ~37%, requiring
    #   2 appearances misses students who are only clearly visible in one image.
    #   Raise back to 2 once recognition rate improves (landmarks fixed).
    "min_images_required": 1,

    # Override: if confidence in ANY image exceeds this, mark present regardless.
    "high_conf_override": 0.75,
}

# ── Evaluation ───────────────────────────────────────────────────────────────
EVALUATION = {
    # Hold-out fraction for testing.
    "test_split": 0.30,

    # MUST be True: split by temporal order (first 70% train, last 30% test).
    # Random splitting causes data leakage — near-duplicate frames in both sets.
    "temporal_split": True,

    # Cross-validation folds for SVM tuning
    "cv_folds": 5,
}
