"""Face detection (YuNet) + identity check (SFace) against reference portraits.

Verification levels written to faces.json per person per link:
  "sface"  - face embedding matched the person's reference portrait(s)
  "manual" - a reviewer (the Claude session, looking at review.jpg) pinned the face
  "unverified" - neither; render refuses unless --allow-unverified
Paintings, statues and cartoons rarely match SFace, so those need "manual".
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from .common import MODELS

SAME_PERSON = 0.363   # SFace cosine threshold from the OpenCV model card
_det = None
_rec = None


def _models():
    global _det, _rec
    if _det is None:
        _det = cv2.FaceDetectorYN.create(str(MODELS / "face_detection_yunet_2023mar.onnx"), "", (320, 320),
                                         score_threshold=0.6, nms_threshold=0.3, top_k=200)
        _rec = cv2.FaceRecognizerSF.create(str(MODELS / "face_recognition_sface_2021dec.onnx"), "")
    return _det, _rec


def load(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"cannot read image {path}")
    return img


def detect(img: np.ndarray) -> list[dict]:
    """Faces as dicts with box [x,y,w,h] in image pixels, score, and raw row for SFace."""
    det, _ = _models()
    h, w = img.shape[:2]
    faces_all = []
    # run at two scales: small faces in group shots need upscaling
    for target in (1600, 2600):
        s = min(1.0 * target / max(h, w), 4.0)
        im = cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_CUBIC if s > 1 else cv2.INTER_AREA)
        det.setInputSize((im.shape[1], im.shape[0]))
        _, f = det.detect(im)
        if f is None:
            continue
        for row in f:
            r = row.copy()
            r[:14] /= s
            faces_all.append(r)
    # merge duplicates across scales
    keep: list[np.ndarray] = []
    for r in sorted(faces_all, key=lambda r: -r[14]):
        if all(_iou(r[:4], k[:4]) < 0.4 for k in keep):
            keep.append(r)
    keep.sort(key=lambda r: r[0])  # left to right
    return [{"box": [float(x) for x in r[:4]], "score": float(r[14]), "row": r} for r in keep]


def _iou(a, b):
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    return inter / (a[2] * a[3] + b[2] * b[3] - inter + 1e-9)


def embed(img: np.ndarray, face: dict) -> np.ndarray:
    _, rec = _models()
    aligned = rec.alignCrop(img, face["row"])
    return rec.feature(aligned).flatten()


def sim(f1: np.ndarray, f2: np.ndarray) -> float:
    _, rec = _models()
    return float(rec.match(f1.reshape(1, -1), f2.reshape(1, -1), cv2.FaceRecognizerSF_FR_COSINE))


def reference_features(paths: list[Path]) -> list[np.ndarray]:
    """Embedding of the largest face in each reference portrait."""
    feats = []
    for p in paths:
        try:
            img = load(p)
        except ValueError:
            continue
        fs = detect(img)
        if not fs:
            continue
        big = max(fs, key=lambda f: f["box"][2] * f["box"][3])
        feats.append(embed(img, big))
    return feats


def best_match(img: np.ndarray, faces: list[dict], refs: list[np.ndarray]) -> list[float]:
    """Max cosine similarity of each face to any reference feature."""
    if not refs:
        return [0.0] * len(faces)
    out = []
    for f in faces:
        e = embed(img, f)
        out.append(max(sim(e, r) for r in refs))
    return out
