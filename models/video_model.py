"""Video detector: samples frames, runs the face CNN on each, aggregates over the clip.
This is frame-level analysis plus temporal statistics; it does not model motion (no LSTM)."""
import time

import cv2
import numpy as np
from PIL import Image

from .common import LOW_CONF_NOTE, label_from_prob, low_confidence
from .errors import MediaError
from .face_utils import crop_face
from .image_model import load_detector, predict_fake_prob

N_FRAMES = 16


def _read_frames(cap, total):
    frames = []
    if total > 0:
        for idx in np.unique(np.linspace(0, total - 1, N_FRAMES).astype(int)):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, bgr = cap.read()
            if ok:
                frames.append((int(idx), bgr))
    if len(frames) < 3:  # seeking unsupported/unreliable for this file: read sequentially
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        every, i, frames = 15, 0, []
        while len(frames) < N_FRAMES * 2:
            ok, bgr = cap.read()
            if not ok:
                break
            if i % every == 0:
                frames.append((i, bgr))
            i += 1
        frames = frames[:N_FRAMES]
    return frames


def detect_video_deepfake(path: str) -> dict:
    t0 = time.time()
    load_detector()  # fail fast with a clear message if the model isn't trained
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise MediaError("Could not open this video file.")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frames = _read_frames(cap, total)
    cap.release()
    if not frames:
        raise MediaError("No frames could be read from this video.")

    crops, found = [], []
    for _, bgr in frames:
        pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        crop, ok = crop_face(pil)
        crops.append(crop)
        found.append(ok)

    probs = predict_fake_prob(crops)
    found = np.array(found)
    used = probs[found] if found.any() else probs  # ignore frames with no face when possible

    p_video = float(used.mean())
    label, conf = label_from_prob(p_video)
    diffs = np.abs(np.diff(used)) if len(used) > 1 else np.array([0.0])
    temporal = round(float(100 * (1 - min(1.0, diffs.mean()))), 1)  # 100 = very steady frame scores

    frame_results = []
    for (idx, _), p, ok in zip(frames, probs, found):
        frame_results.append({
            "frame": idx,
            "time": round(idx / fps, 2) if fps > 0 else None,
            "label": "FAKE" if p >= 0.5 else "REAL",
            "fake_probability": round(float(p) * 100, 1),
            "face_found": bool(ok),
        })

    n_fake = int((used >= 0.5).sum())
    details = (
        f"Analysed {len(frames)} frames ({int(found.sum())} with a detectable face). "
        f"{n_fake} of {len(used)} scored frames looked fake; mean P(fake) = {p_video * 100:.1f}%."
    )
    if not found.any():
        details += " No face was found in any frame, so whole frames were scored; treat this as unreliable."
    if low_confidence(conf):
        details += LOW_CONF_NOTE

    return {
        "label": label,
        "confidence": conf,
        "fake_probability": round(p_video * 100, 1),
        "low_confidence": low_confidence(conf),
        "frames_analyzed": len(frames),
        "faces_found": int(found.sum()),
        "temporal_score": temporal,
        "frame_results": frame_results,
        "details": details,
        "features_used": ["Frame sampling", "Face crop", "CNN per frame", "Temporal consistency"],
        "preprocessing": {
            "enhancement": f"{len(frames)} frames sampled + face crop",
            "processing_time": f"{time.time() - t0:.2f} s",
        },
        "video_info": f"{width} x {height}, {fps:.0f} fps" if fps else f"{width} x {height}",
    }
