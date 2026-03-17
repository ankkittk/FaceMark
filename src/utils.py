"""
src/utils.py - Shared Utility Functions
========================================
Logging setup, frame extraction from video, visualization helpers.
"""

import cv2
import numpy as np
import logging
import os
import sys
from pathlib import Path
from typing import List, Optional


def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """
    Configure logging for the entire system.
    Outputs to both console and optionally a file.

    Args:
        level:    Log level string: DEBUG, INFO, WARNING, ERROR
        log_file: If provided, also log to this file path
    """
    fmt = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format=fmt,
        handlers=handlers,
    )


def extract_frames_from_video(video_path: str,
                               output_dir: str,
                               max_frames: int = 200,
                               skip_frames: int = 5) -> List[str]:
    """
    Extract frames from an enrollment video.

    Why skip_frames=5?
      Consecutive video frames are nearly identical. Extracting every 5th frame
      gives temporal diversity while keeping the dataset manageable.
      At 30fps video, every 5th frame = 6fps effective → good pose variation.

    Args:
        video_path:  Path to enrollment video
        output_dir:  Where to save extracted frames
        max_frames:  Maximum frames to extract (prevents huge datasets)
        skip_frames: Extract one frame every N frames

    Returns:
        List of saved frame file paths
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {video_path}")

    saved_paths = []
    frame_idx = 0
    saved_count = 0

    while saved_count < max_frames:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_idx % skip_frames == 0:
            filename = out_path / f"frame_{saved_count:04d}.jpg"
            cv2.imwrite(str(filename), frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            saved_paths.append(str(filename))
            saved_count += 1

        frame_idx += 1

    cap.release()
    logging.getLogger(__name__).info(
        f"Extracted {saved_count} frames from {Path(video_path).name}"
    )
    return saved_paths


def visualize_session_results(image: np.ndarray,
                               face_results: list,
                               output_path: Optional[str] = None) -> np.ndarray:
    """
    Draw recognition results on a classroom image.

    Each detected face gets:
      - Green box + name if recognized above threshold
      - Red box + "Unknown" if not recognized
      - Confidence score displayed above box

    Args:
        image:        BGR classroom image
        face_results: List of FaceResult objects from recognizer
        output_path:  If provided, save annotated image here

    Returns:
        Annotated BGR image
    """
    vis = image.copy()

    for fr in face_results:
        x1, y1, x2, y2 = fr.bbox.astype(int)
        is_known = fr.recognition.is_known

        color = (0, 200, 0) if is_known else (0, 0, 220)
        label = f"{fr.recognition.name} ({fr.recognition.confidence:.2f})"

        # Draw bounding box
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)

        # Draw label background for readability
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(vis, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(vis, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    if output_path:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(output_path, vis)

    return vis


def ensure_directories(paths: dict):
    """Create all required directories if they don't exist."""
    for key, path in paths.items():
        if path.endswith((".npy", ".pkl", ".pt", ".csv", ".json", ".log")):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
        else:
            Path(path).mkdir(parents=True, exist_ok=True)
