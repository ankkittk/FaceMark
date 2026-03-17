"""
src/detector.py - YOLO Face Detector
======================================
Detects all faces in a classroom image using YOLOv8-face.

WHY YOLOv8-face (not generic YOLO)?
  Generic YOLO (COCO-trained) detects "person" class, not faces.
  YOLOv8-face is trained on WIDER Face and outputs:
    - Tight face bounding boxes
    - 5-point facial landmarks (eyes, nose, mouth corners)
  The landmarks are CRITICAL — they feed the affine alignment step,
  which ensures ArcFace receives faces in its expected coordinate space.

MODEL DOWNLOAD:
  https://github.com/derronqi/yolov8-face/releases
  File: yolov8n-face.pt  (nano, fast, good accuracy for classroom use)
  Place in: models/yolov8n-face.pt
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class FaceDetection:
    """
    Container for a single detected face.

    Attributes:
        bbox:        [x1, y1, x2, y2] pixel coordinates
        confidence:  YOLO score [0.0, 1.0]
        landmarks:   5x2 array [(x,y) per landmark] or None
                     Order: left_eye, right_eye, nose, left_mouth, right_mouth
        face_crop:   Aligned 112x112 BGR crop (filled by aligner, not detector)
    """
    bbox: np.ndarray
    confidence: float
    landmarks: Optional[np.ndarray] = None
    face_crop: Optional[np.ndarray] = None

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.bbox
        return float((x2 - x1) * (y2 - y1))

    @property
    def width(self) -> float:
        return float(self.bbox[2] - self.bbox[0])

    @property
    def height(self) -> float:
        return float(self.bbox[3] - self.bbox[1])


class YOLOFaceDetector:
    """
    YOLOv8-face wrapper for classroom multi-face detection.

    Example:
        detector = YOLOFaceDetector("models/yolov8n-face.pt")
        image = cv2.imread("classroom_row1.jpg")
        faces = detector.detect(image)
        print(f"Detected {len(faces)} faces")
    """

    def __init__(self, weights_path: str, conf: float = 0.45,
                 nms_iou: float = 0.45, min_area: int = 1600):
        """
        Args:
            weights_path: Path to yolov8n-face.pt
            conf:         Detection confidence threshold
            nms_iou:      Non-max suppression IoU threshold
            min_area:     Minimum face bounding box area (pixels²)
                          Default 1600 = 40×40 px minimum face
        """
        self.weights_path = weights_path
        self.conf = conf
        self.nms_iou = nms_iou
        self.min_area = min_area
        self._model = None
        self._load()

    def _load(self):
        """Load YOLO weights. Raises FileNotFoundError if not found."""
        path = Path(self.weights_path)
        if not path.exists():
            raise FileNotFoundError(
                f"\n[YOLO] Weights not found: {self.weights_path}\n"
                f"Download yolov8n-face.pt from:\n"
                f"  https://github.com/derronqi/yolov8-face/releases\n"
                f"Place it in the models/ folder."
            )
        from ultralytics import YOLO
        self._model = YOLO(self.weights_path)
        logger.info(f"[Detector] YOLOv8-face loaded from {self.weights_path}")

    def detect(self, image: np.ndarray) -> List[FaceDetection]:
        """
        Run face detection on a single BGR image.

        Returns faces sorted left-to-right (by x-coordinate of bbox center).
        This preserves rough seating order across classroom rows.

        Args:
            image: BGR numpy array (from cv2.imread)

        Returns:
            List[FaceDetection], sorted by x position.
        """
        if image is None or image.size == 0:
            logger.warning("[Detector] Received empty image.")
            return []

        # Use larger inference size for classroom photos — small back-row faces
        # get severely downsampled at 640px.  1280 detects ~2× more faces with
        # only ~1.5× compute cost. Falls back to 640 if image is already small.
        h, w = image.shape[:2]
        infer_size = 1280 if max(h, w) > 800 else 640

        results = self._model(
            image,
            conf=self.conf,
            iou=self.nms_iou,
            imgsz=infer_size,
            verbose=False,
        )

        detections: List[FaceDetection] = []

        for result in results:
            boxes = result.boxes
            if boxes is None or len(boxes) == 0:
                continue

            bboxes = boxes.xyxy.cpu().numpy()   # (N, 4)
            confs  = boxes.conf.cpu().numpy()   # (N,)

            # Extract landmarks if model supports them (YOLOv8-face does)
            landmarks_batch = None
            if hasattr(result, "keypoints") and result.keypoints is not None:
                try:
                    kp = result.keypoints.xy.cpu().numpy()  # (N, 5, 2) pixel coords
                    # Validity check: reject if ALL points are zero (degenerate output).
                    # Use sum > 5 instead of np.any(kp > 1) — the old check failed
                    # for small/distant faces where individual coords may be < 1px
                    # but the sum of all 10 coordinates is always > 5 for real faces.
                    if kp.shape == (len(bboxes), 5, 2) and kp.sum() > 5:
                        landmarks_batch = kp
                    else:
                        # Try .data as fallback for older ultralytics builds
                        try:
                            kp2 = result.keypoints.data.cpu().numpy()  # (N, 5, 3) with conf
                            if kp2.ndim == 3 and kp2.shape[2] >= 2:
                                kp_xy = kp2[:, :, :2]
                                if kp_xy.sum() > 5:
                                    landmarks_batch = kp_xy
                        except Exception:
                            pass
                except Exception:
                    landmarks_batch = None

            for i in range(len(bboxes)):
                bbox = bboxes[i].astype(int)
                confidence = float(confs[i])

                # Filter by minimum face size
                x1, y1, x2, y2 = bbox
                area = (x2 - x1) * (y2 - y1)
                if area < self.min_area:
                    logger.debug(f"[Detector] Skipping tiny face: area={area}px²")
                    continue

                # Clamp bbox to image bounds
                h, w = image.shape[:2]
                bbox = np.array([
                    max(0, x1), max(0, y1),
                    min(w, x2), min(h, y2)
                ])

                landmarks = None
                if landmarks_batch is not None:
                    lm = landmarks_batch[i]  # (5, 2)
                    # Reject degenerate landmarks — sum > 5 means at least some
                    # real pixel coordinates are present (not all zero)
                    if lm.sum() > 5:
                        landmarks = lm

                detections.append(FaceDetection(
                    bbox=bbox,
                    confidence=confidence,
                    landmarks=landmarks,
                ))

        # Sort left-to-right by x-center (preserves classroom seating order)
        detections.sort(key=lambda d: (d.bbox[0] + d.bbox[2]) / 2)

        logger.info(f"[Detector] Found {len(detections)} valid faces.")
        return detections

    def detect_batch(self, images: List[np.ndarray]) -> List[List[FaceDetection]]:
        """
        Detect faces in multiple images.

        Args:
            images: List of BGR images

        Returns:
            List of detection lists, one per image.
        """
        return [self.detect(img) for img in images]

    def draw_detections(self, image: np.ndarray,
                        detections: List[FaceDetection]) -> np.ndarray:
        """
        Draw bounding boxes and landmarks on image for visualization/debugging.

        Args:
            image:      BGR image
            detections: Output from detect()

        Returns:
            Annotated BGR image (copy of input)
        """
        vis = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(vis, f"{det.confidence:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            if det.landmarks is not None:
                colors = [(255,0,0),(0,0,255),(0,255,255),(255,0,255),(0,165,255)]
                for (lx, ly), color in zip(det.landmarks.astype(int), colors):
                    cv2.circle(vis, (lx, ly), 3, color, -1)
        return vis
