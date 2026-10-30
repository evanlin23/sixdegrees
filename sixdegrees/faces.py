"""Face detection and recognition with OpenCV's YuNet + SFace models.

No TensorFlow/dlib needed: both models are small ONNX files that run on
opencv-python's DNN module and are downloaded once into the cache dir.
"""
from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .util import cache_dir, file_hash, http, load_image, log, read_json, write_json

MODELS = {
    "detector": (
        "face_detection_yunet_2023mar.onnx",
        "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
        [
            "https://huggingface.co/opencv/face_detection_yunet/resolve/main/face_detection_yunet_2023mar.onnx",
            "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
        ],
    ),
    "recognizer": (
        "face_recognition_sface_2021dec.onnx",
        "0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
        [
            "https://huggingface.co/opencv/face_recognition_sface/resolve/main/face_recognition_sface_2021dec.onnx",
            "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        ],
    ),
}

# Cosine similarity between SFace embeddings. OpenCV recommends 0.363 for
# "same person"; we match against several reference photos, so be a bit stricter.
DEFAULT_MATCH_THRESHOLD = 0.40
REFERENCE_CLUSTER_THRESHOLD = 0.42
FACE_CACHE_VERSION = "yunet2023mar-sface2021dec-v1"


def ensure_model(key: str) -> Path:
    name, sha256, urls = MODELS[key]
    path = cache_dir() / "models" / name
    if path.exists() and _sha256(path) == sha256:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    last_err: Optional[Exception] = None
    for url in urls:
        try:
            log.info("Downloading face model %s ...", name)
            resp = http().get(url, timeout=120)
            resp.raise_for_status()
            tmp = path.with_suffix(".part")
            tmp.write_bytes(resp.content)
            if _sha256(tmp) != sha256:
                raise ValueError(f"checksum mismatch for {url}")
            tmp.replace(path)
            return path
        except Exception as e:  # try the next mirror
            last_err = e
            log.warning("  could not fetch %s: %s", url, e)
    raise RuntimeError(f"Could not download face model {name}. Put it in {path.parent} manually.") from last_err


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Face:
    """A detected face in original-image pixel coordinates."""

    x: float
    y: float
    w: float
    h: float
    score: float
    # YuNet order: right eye, left eye, nose tip, right mouth corner, left mouth corner
    # ("right" = the subject's right, i.e. image-left).
    landmarks: list = field(default_factory=list)
    embedding: Optional[np.ndarray] = field(default=None, repr=False, compare=False)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x + self.w / 2, self.y + self.h / 2)

    @property
    def size(self) -> float:
        return max(self.w, self.h)

    @property
    def roll(self) -> float:
        """Angle of the eye line in radians (0 = level)."""
        if len(self.landmarks) < 2:
            return 0.0
        (x1, y1), (x2, y2) = self.landmarks[0], self.landmarks[1]
        return math.atan2(y2 - y1, x2 - x1)

    def to_dict(self, with_embedding: bool = False) -> dict:
        d = {"box": [round(v, 1) for v in (self.x, self.y, self.w, self.h)], "score": round(float(self.score), 3),
             "landmarks": [[round(float(a), 1), round(float(b), 1)] for a, b in self.landmarks]}
        if with_embedding and self.embedding is not None:
            d["embedding"] = [round(float(v), 5) for v in self.embedding]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Face":
        x, y, w, h = d["box"]
        emb = d.get("embedding")
        return cls(x, y, w, h, d.get("score", 1.0), [list(p) for p in d.get("landmarks", [])],
                   np.asarray(emb, dtype=np.float32) if emb is not None else None)

    @classmethod
    def from_point(cls, x: float, y: float, size: float) -> "Face":
        """A pseudo-face for a hand-picked point (like clicking a point in the multi-animator)."""
        return cls(x - size / 2, y - size / 2, size, size, 1.0, [])


class FaceEngine:
    """Detect faces and compute 128-d SFace embeddings, with an on-disk cache per image."""

    def __init__(self, score_threshold: float = 0.7, max_side: int = 1600, min_face: int = 20):
        import cv2

        self.cv2 = cv2
        self.detector = cv2.FaceDetectorYN.create(str(ensure_model("detector")), "", (320, 320),
                                                  score_threshold, 0.3, 5000)
        self.recognizer = cv2.FaceRecognizerSF.create(str(ensure_model("recognizer")), "")
        self.max_side = max_side
        self.min_face = min_face
        self._memo: dict[str, list[Face]] = {}

    def detect(self, img: np.ndarray) -> list[Face]:
        cv2 = self.cv2
        h, w = img.shape[:2]
        long_side = max(h, w)
        scale = min(self.max_side / long_side, 2.0)
        if long_side * scale < 640:
            scale = min(640 / long_side, 3.0)
        if abs(scale - 1) > 1e-3:
            interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
            det = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=interp)
        else:
            det, scale = img, 1.0
        self.detector.setInputSize((det.shape[1], det.shape[0]))
        _, rows = self.detector.detect(det)
        faces: list[Face] = []
        for row in rows if rows is not None else []:
            if max(row[2], row[3]) / scale < self.min_face:
                continue
            aligned = self.recognizer.alignCrop(det, row)
            emb = self.recognizer.feature(aligned).flatten().astype(np.float32)
            emb /= np.linalg.norm(emb) + 1e-9
            lms = (row[4:14].reshape(5, 2) / scale).tolist()
            x, y, bw, bh = (row[:4] / scale).tolist()
            faces.append(Face(x, y, bw, bh, float(row[14]), lms, emb))
        faces.sort(key=lambda f: -f.size)
        return faces

    def faces_in(self, path: Path) -> list[Face]:
        """Faces in an image file (cached on disk by file content)."""
        path = Path(path)
        key = f"{file_hash(path)}-{FACE_CACHE_VERSION}"
        if key in self._memo:
            return self._memo[key]
        cache_path = cache_dir() / "faces" / f"{key}.json"
        cached = read_json(cache_path)
        if cached is not None:
            faces = [Face.from_dict(d) for d in cached]
        else:
            try:
                faces = self.detect(load_image(path))
            except Exception as e:
                log.debug("  could not read %s: %s", path, e)
                faces = []
            write_json(cache_path, [f.to_dict(with_embedding=True) for f in faces])
        self._memo[key] = faces
        return faces


@dataclass
class Identity:
    """What a person looks like: embeddings of their face from several reference photos."""

    name: str
    refs: np.ndarray  # (k, 128), L2-normalized
    sources: list = field(default_factory=list)
    confident: bool = True

    @property
    def ok(self) -> bool:
        return self.refs.shape[0] > 0

    def similarity(self, emb: Optional[np.ndarray]) -> float:
        if emb is None or not self.ok:
            return -1.0
        sims = np.sort(self.refs @ emb)[::-1]
        # Average the two best references once we have a few, so one odd reference can't carry a match.
        return float(sims[0] if len(sims) < 3 else sims[:2].mean())


def build_identity(name: str, images: Iterable[Path], engine: FaceEngine, anchor: Optional[Path] = None,
                   threshold: float = REFERENCE_CLUSTER_THRESHOLD) -> Identity:
    """Find the one face that recurs across reference photos (search results are noisy).

    Every face in every reference image votes for the faces it resembles in *other*
    images; the face with the most support is the person, and each image contributes
    at most its best match to that seed.
    """
    images = [Path(p) for p in images]
    entries: list[tuple[int, Face, bool]] = []  # (image index, face, is_anchor)
    for idx, path in enumerate(images):
        faces = [f for f in engine.faces_in(path) if f.size >= 40 and f.embedding is not None][:4]
        for rank, f in enumerate(faces):
            entries.append((idx, f, anchor is not None and path == Path(anchor)))
    if not entries:
        return Identity(name, np.zeros((0, 128), np.float32), [], False)

    embs = np.stack([e[1].embedding for e in entries])
    sims = embs @ embs.T
    img_ids = np.array([e[0] for e in entries])
    support = []
    for i in range(len(entries)):
        others = {img_ids[j] for j in np.nonzero(sims[i] >= threshold)[0] if img_ids[j] != img_ids[i]}
        support.append(len(others))
    largest_in_image = {}
    for i, (idx, f, _) in enumerate(entries):
        if idx not in largest_in_image or f.size > entries[largest_in_image[idx]][1].size:
            largest_in_image[idx] = i

    def rank(i: int):
        idx, f, is_anchor = entries[i]
        return (support[i], is_anchor, largest_in_image[idx] == i, f.size)

    seed = max(range(len(entries)), key=rank)
    confident = support[seed] >= 1 or len(images) == 1
    if support[seed] == 0:
        # Nobody agrees with anybody: trust the anchor (Wikipedia lead image) or the first portrait.
        anchored = [i for i, e in enumerate(entries) if e[2]]
        seed = max(anchored or [largest_in_image[min(largest_in_image)]], key=lambda i: entries[i][1].size)

    members = {}
    for i, (idx, f, _) in enumerate(entries):
        s = sims[seed, i]
        if (s >= threshold or i == seed) and (idx not in members or s > sims[seed, members[idx]]):
            members[idx] = i
    chosen = sorted(members.values())
    return Identity(name, embs[chosen], [str(images[entries[i][0]]) for i in chosen], confident)


@dataclass
class PairMatch:
    a: Face
    b: Face
    sim_a: float
    sim_b: float
    n_faces: int
    collage: bool = False

    def ok(self, threshold: float = DEFAULT_MATCH_THRESHOLD) -> bool:
        return self.sim_a >= threshold and self.sim_b >= threshold and not self.collage

    @property
    def score(self) -> float:
        """Rank candidate photos: confident identities first, then big faces, then uncluttered."""
        small_face = min(self.a.size, self.b.size)
        crowd_penalty = min(0.12, 0.02 * max(0, self.n_faces - 2))
        return (min(self.sim_a, self.sim_b) + 0.15 * min(1.0, small_face / 160.0)
                - crowd_penalty - (0.5 if self.collage else 0.0))


def match_pair(faces: list[Face], id_a: Optional[Identity], id_b: Optional[Identity]) -> Optional[PairMatch]:
    """Assign two *different* faces in a photo to person A and person B."""
    faces = [f for f in faces if f.embedding is not None]
    if len(faces) < 2:
        return None
    if id_a is None or id_b is None or not id_a.ok or not id_b.ok:
        # No way to tell who is who: take the two biggest faces, left to right.
        a, b = sorted(faces[:2], key=lambda f: f.center[0])
        return PairMatch(a, b, 0.0, 0.0, len(faces))
    sa = [id_a.similarity(f.embedding) for f in faces]
    sb = [id_b.similarity(f.embedding) for f in faces]
    best = None
    for i in range(len(faces)):
        for j in range(len(faces)):
            if i != j and (best is None or sa[i] + sb[j] > sa[best[0]] + sb[best[1]]):
                best = (i, j)
    i, j = best
    return PairMatch(faces[i], faces[j], sa[i], sb[j], len(faces))


def looks_like_collage(img: np.ndarray, a: Face, b: Face) -> bool:
    """Detect side-by-side / stacked composites: a straight seam running between the two faces.

    Search results for "A and B" are full of collages of two separate portraits, which
    pass a face check but aren't photos of the two people together.
    """
    import cv2

    h, w = img.shape[:2]
    s = min(1.0, 700.0 / max(h, w))
    small = cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else img
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ax, ay = a.center[0] * s, a.center[1] * s
    bx, by = b.center[0] * s, b.center[1] * s

    def seam_between(g: np.ndarray, lo: float, hi: float) -> bool:
        # g: rows = along the seam, cols = across it. Look for a column where the image
        # jumps on most rows (a cut between two photos) or a flat white/black divider strip.
        n = g.shape[1]
        lo, hi = int(max(2, lo)), int(min(n - 3, hi))
        if hi - lo < 3:
            return False
        jump = np.abs(g[:, 2:] - g[:, :-2]) > 28  # 2px step survives resampling blur
        frac = jump.mean(axis=0)
        base = float(np.median(frac))
        # A cut between photos runs edge to edge; poles and door frames usually don't.
        bands = np.array_split(jump, 10, axis=0)
        coverage = np.mean([band.mean(axis=0) >= 0.25 for band in bands], axis=0)
        if ((frac[lo:hi] >= max(0.45, 5 * base)) & (coverage[lo:hi] >= 0.9)).any():
            return True
        std, mean = g.std(axis=0), g.mean(axis=0)
        flat = (std < 3.0) & ((mean > 235) | (mean < 18))
        run = 0
        for x in range(lo, hi):
            run = run + 1 if flat[x] else 0
            if run >= 2:
                return True
        return False

    # Only test the gap between the two faces so ordinary vertical edges elsewhere don't count.
    if abs(ax - bx) > abs(ay - by):
        left, right = (a, b) if ax < bx else (b, a)
        return seam_between(gray, (left.x + left.w) * s, right.x * s)
    top, bottom = (a, b) if ay < by else (b, a)
    return seam_between(gray.T, (top.y + top.h) * s, bottom.y * s)
