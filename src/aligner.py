"""
src/aligner.py - 5-Point Landmark Face Alignment
==================================================
Aligns detected faces to ArcFace's canonical 112x112 coordinate space.

WHY ALIGNMENT MATTERS SO MUCH:
  ArcFace was trained on faces aligned to a specific 112x112 template
  using a similarity transform. If you feed it unaligned crops:
    - The same student looks different across images (intra-class variance ↑)
    - Different students look more similar (inter-class distance ↓)
    - SVM decision boundaries become unreliable
  Proper alignment is the single highest-impact preprocessing step.

WHAT THIS MODULE DOES:
  1. Takes the 5 landmarks from YOLO (eyes, nose, mouth corners)
  2. Computes the similarity transform (rotation + scale + translation)
     mapping detected landmarks → canonical ArcFace template positions
  3. Applies affine warp to produce exactly 112x112 aligned face

FALLBACK (no landmarks):
  If YOLO doesn't output landmarks (shouldn't happen with YOLOv8-face),
  we fall back to a simple bounding-box crop + resize.
  This is strictly worse — flag a warning if this path is taken.
"""

import cv2
import numpy as np
import logging
from typing import Optional, List

from src.detector import FaceDetection

logger = logging.getLogger(__name__)

# Canonical 5-point template for ArcFace 112x112 space
# Source: InsightFace / ArcFace paper reference implementation
# Order: [left_eye, right_eye, nose_tip, left_mouth_corner, right_mouth_corner]
ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963],
    [73.5318, 51.5014],
    [56.0252, 71.7366],
    [41.5493, 92.3655],
    [70.7299, 92.2041],
], dtype=np.float32)

FACE_SIZE = (112, 112)


class FaceAligner:
    """
    Aligns face crops to ArcFace canonical coordinate space.

    Example:
        aligner = FaceAligner()
        aligned = aligner.align(image, detection)
        # aligned is a 112x112 BGR array ready for ArcFace
    """

    def __init__(self, face_size: tuple = FACE_SIZE,
                 template: np.ndarray = ARCFACE_TEMPLATE):
        self.face_size = face_size
        self.template = template

    def align(self, image: np.ndarray,
              detection: FaceDetection) -> Optional[np.ndarray]:
        """
        Align a single detected face.

        Args:
            image:     Full BGR image
            detection: FaceDetection from detector (with landmarks if available)

        Returns:
            112x112 BGR aligned face, or None if alignment fails.
        """
        if detection.landmarks is not None and self._landmarks_valid(detection.landmarks, image.shape):
            return self._align_with_landmarks(image, detection.landmarks)
        else:
            logger.debug("[Aligner] No valid landmarks — bbox fallback.")
            return self._align_bbox_fallback(image, detection.bbox)

    def align_batch(self, image: np.ndarray,
                    detections: List[FaceDetection]) -> List[Optional[np.ndarray]]:
        """
        Align all faces detected in a single image.

        Returns list of aligned crops (None for any that fail).
        """
        aligned = []
        for det in detections:
            crop = self.align(image, det)
            det.face_crop = crop   # Store back on detection object
            aligned.append(crop)
        return aligned

    def _landmarks_valid(self, landmarks: np.ndarray,
                         image_shape: tuple) -> bool:
        """
        Sanity-check landmarks before using them for alignment.
        Rejects landmarks that are outside image bounds or degenerate.
        """
        if landmarks is None or landmarks.shape != (5, 2):
            return False

        h, w = image_shape[:2]

        # All landmark coordinates must be within image bounds
        if np.any(landmarks[:, 0] < 0) or np.any(landmarks[:, 0] > w):
            return False
        if np.any(landmarks[:, 1] < 0) or np.any(landmarks[:, 1] > h):
            return False

        # Eyes should be horizontally separated (not degenerate/collapsed)
        eye_dist = np.linalg.norm(landmarks[0] - landmarks[1])
        if eye_dist < 10:  # Less than 10px apart = degenerate
            return False

        return True

    def _align_with_landmarks(self, image: np.ndarray,
                               landmarks: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute similarity transform from detected landmarks to ArcFace
        template, then warp the image.

        The similarity transform preserves shape (no shear), allowing only:
          - Rotation (corrects head tilt)
          - Uniform scaling (corrects distance)
          - Translation (corrects position)

        This is exactly what ArcFace training used — matching it at inference
        time is what makes the embeddings reliable.
        """
        try:
            src_pts = landmarks.astype(np.float32)
            dst_pts = self.template.astype(np.float32)

            # estimateAffinePartial2D = similarity transform (not full affine)
            # This is correct — full affine allows shear which distorts face geometry
            M, inliers = cv2.estimateAffinePartial2D(
                src_pts, dst_pts,
                method=cv2.LMEDS  # Robust to landmark outliers
            )

            if M is None:
                logger.warning("[Aligner] estimateAffinePartial2D failed.")
                return None

            aligned = cv2.warpAffine(
                image, M, self.face_size,
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT  # Reflect edges (better than black padding)
            )
            return aligned

        except Exception as e:
            logger.error(f"[Aligner] Alignment failed: {e}")
            return None

    def _align_bbox_fallback(self, image: np.ndarray,
                              bbox: np.ndarray) -> Optional[np.ndarray]:
        """
        Improved bbox fallback: estimates facial landmark positions from
        standard face proportions, then runs the SAME affine transform
        as the landmark path.

        Why this is much better than simple crop+resize:
          A plain crop passes the face to ArcFace in whatever orientation
          it appears in the photo.  ArcFace was trained on faces aligned so
          that eyes sit at ~y=52, nose at ~y=72, mouth at ~y=92 inside the
          112x112 frame.  When we estimate landmark positions from the bbox
          and warp to that template, we approximately reproduce that canonical
          orientation even without real landmark output.

        Face proportion constants (empirically derived from 300W / WIDER Face):
          - Eyes:      ~30% down from top, at 30% and 70% of width
          - Nose tip:  ~55% down, centred
          - Mouth:     ~75% down, at 37% and 63% of width
        """
        try:
            x1, y1, x2, y2 = bbox.astype(float)
            bw = x2 - x1
            bh = y2 - y1

            if bw <= 0 or bh <= 0:
                return None

            # Estimate 5 landmark positions from bbox proportions
            # These match the average face geometry seen in frontal photos
            est_landmarks = np.array([
                [x1 + bw * 0.30, y1 + bh * 0.30],   # left eye
                [x1 + bw * 0.70, y1 + bh * 0.30],   # right eye
                [x1 + bw * 0.50, y1 + bh * 0.55],   # nose tip
                [x1 + bw * 0.37, y1 + bh * 0.75],   # left mouth corner
                [x1 + bw * 0.63, y1 + bh * 0.75],   # right mouth corner
            ], dtype=np.float32)

            # Validate estimated landmarks are within image bounds
            h, w = image.shape[:2]
            est_landmarks[:, 0] = np.clip(est_landmarks[:, 0], 0, w - 1)
            est_landmarks[:, 1] = np.clip(est_landmarks[:, 1], 0, h - 1)

            # Run the same similarity transform as the landmark path
            aligned = self._align_with_landmarks(image, est_landmarks)
            if aligned is not None:
                return aligned

            # Last resort: plain crop if affine transform itself fails
            pad_x = int(bw * 0.05)
            pad_y = int(bh * 0.05)
            cx1 = max(0, int(x1) - pad_x)
            cy1 = max(0, int(y1) - pad_y)
            cx2 = min(w, int(x2) + pad_x)
            cy2 = min(h, int(y2) + pad_y)
            crop = image[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                return None
            return cv2.resize(crop, self.face_size, interpolation=cv2.INTER_CUBIC)

        except Exception as e:
            logger.error(f"[Aligner] Bbox fallback failed: {e}")
            return None
