"""
src/embedder.py - ArcFace Face Embedder (InsightFace) [v2 - direct model load]
================================================================================
Extracts 512-D identity embeddings from aligned 112x112 face crops.

FIX NOTE (v2):
  FaceAnalysis asserts that a detection model must be present even when
  allowed_modules=["recognition"] is passed. This is a known InsightFace
  issue on certain versions. Fix: load the ArcFace ONNX model DIRECTLY
  via insightface.model_zoo, bypassing FaceAnalysis entirely.
  Since YOLO + our aligner handle detection and alignment, we only need
  the recognition model here.

WHY ARCFACE (FROZEN)?
  Trained on 5.8M images / 85K identities with angular margin loss.
  Generalizes to new identities without fine-tuning. For 80 students,
  frozen backbone + SVM is architecturally correct — fine-tuning with
  limited data risks overfitting with minimal accuracy gain (2-5%).
"""

import cv2
import numpy as np
import logging
import os
from typing import Optional, List

logger = logging.getLogger(__name__)

# The recognition model filename inside the buffalo_l pack
BUFFALO_L_REC_FILENAME = "w600k_r50.onnx"


class FaceEmbedder:
    """
    Loads ArcFace recognition ONNX model directly (no FaceAnalysis).
    Returns L2-normalized 512-D embeddings from 112x112 aligned faces.

    Example:
        embedder = FaceEmbedder("buffalo_l")
        emb = embedder.embed(aligned_112x112_bgr_face)  # → (512,)
    """

    def __init__(self, model_name: str = "buffalo_l"):
        self.model_name = model_name
        self._rec_model = None
        self._load()

    def _load(self):
        """
        Load ArcFace recognition model directly via insightface.model_zoo.
        First run downloads ~300MB to ~/.insightface/models/buffalo_l/.
        All subsequent runs are fully offline.
        """
        try:
            from insightface.model_zoo import model_zoo

            models_root = os.path.join(
                os.path.expanduser("~"), ".insightface", "models", self.model_name
            )
            rec_path = os.path.join(models_root, BUFFALO_L_REC_FILENAME)

            # If model not cached yet, trigger download via FaceAnalysis
            if not os.path.exists(rec_path):
                logger.info("[Embedder] First run — downloading InsightFace models (~300MB)...")
                from insightface.app import FaceAnalysis
                _dl = FaceAnalysis(name=self.model_name,
                                   providers=["CPUExecutionProvider"])
                _dl.prepare(ctx_id=0, det_size=(640, 640))
                del _dl
                logger.info("[Embedder] Download complete.")

            if not os.path.exists(rec_path):
                raise FileNotFoundError(
                    f"Recognition model missing: {rec_path}\n"
                    f"Ensure internet access for first-run download."
                )

            # Load ONLY the recognition model — bypasses detection assertion
            self._rec_model = model_zoo.get_model(
                rec_path, providers=["CPUExecutionProvider"]
            )
            self._rec_model.prepare(ctx_id=0)
            logger.info(f"[Embedder] ArcFace loaded: {rec_path}")

        except ImportError:
            raise ImportError(
                "InsightFace not installed.\n"
                "Run: pip install insightface onnxruntime"
            )
        except Exception as e:
            raise RuntimeError(f"[Embedder] Failed to load model: {e}")

    def embed(self, aligned_face: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract 512-D L2-normalized embedding from a 112x112 BGR face.

        Args:
            aligned_face: BGR numpy array (112, 112, 3) from FaceAligner.

        Returns:
            L2-normalized (512,) float32 array, or None on failure.
        """
        if aligned_face is None:
            return None

        if aligned_face.shape[:2] != (112, 112):
            aligned_face = cv2.resize(aligned_face, (112, 112))

        try:
            face_input = aligned_face.astype(np.float32)
            # get_feat() is the direct inference method on recognition models
            embedding = self._rec_model.get_feat(face_input)
            if embedding is None:
                return None

            emb = embedding.flatten().astype(np.float32)
            norm = np.linalg.norm(emb)
            if norm < 1e-6:
                logger.warning("[Embedder] Near-zero norm — skipping face.")
                return None

            return emb / norm   # L2-normalize onto unit hypersphere

        except Exception as e:
            logger.error(f"[Embedder] embed() failed: {e}")
            return None

    def embed_with_augmentation(self, aligned_face: np.ndarray) -> List[np.ndarray]:
        """
        Extract embeddings from face + augmented versions.
        Used during enrollment to bridge the domain gap between
        close-up enrollment videos and wide-angle session photos.

        Augmentations:
          1. Original
          2. Horizontal flip           → head angle variation
          3. Brightness +20%           → well-lit classroom
          4. Brightness -20%           → backlit / shadowed
          5. Contrast boost +15%       → sharpness variation
          6. Downscale 50% + upscale   → simulates back-row distance
          7. Downscale 35% + upscale   → simulates far back-row distance
          8. Slight Gaussian blur       → simulates motion/focus blur
          9. Flip + brightness -20%    → combined variation
        """
        if aligned_face is None:
            return []

        h, w = aligned_face.shape[:2]

        def _downscale_sim(face, scale):
            """Simulate distant face: shrink then blow back up."""
            small = cv2.resize(face, (int(w*scale), int(h*scale)),
                               interpolation=cv2.INTER_AREA)
            return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)

        def _blur(face, k=3):
            return cv2.GaussianBlur(face, (k, k), 0)

        faces = [
            aligned_face,
            cv2.flip(aligned_face, 1),
            self._adjust_brightness(aligned_face, 1.2),
            self._adjust_brightness(aligned_face, 0.8),
            self._adjust_contrast(aligned_face, 1.15),
            _downscale_sim(aligned_face, 0.50),
            _downscale_sim(aligned_face, 0.35),
            _blur(aligned_face, k=3),
            self._adjust_brightness(cv2.flip(aligned_face, 1), 0.85),
        ]

        results = []
        for face in faces:
            emb = self.embed(face)
            if emb is not None:
                results.append(emb)
        return results

    def embed_batch(self, faces: List[np.ndarray],
                    augment: bool = False) -> List[Optional[np.ndarray]]:
        """Embed a list of aligned faces. Set augment=True for enrollment."""
        if augment:
            out = []
            for face in faces:
                out.extend(self.embed_with_augmentation(face))
            return out
        return [self.embed(f) for f in faces]

    @staticmethod
    def _adjust_brightness(img: np.ndarray, factor: float) -> np.ndarray:
        return np.clip(img.astype(np.float32) * factor, 0, 255).astype(np.uint8)

    @staticmethod
    def _adjust_contrast(img: np.ndarray, factor: float) -> np.ndarray:
        mean = np.mean(img)
        return np.clip(mean + factor * (img.astype(np.float32) - mean),
                       0, 255).astype(np.uint8)
