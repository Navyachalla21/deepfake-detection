import cv2
import numpy as np
from PIL import Image

_cascade = None


def _get_cascade():
    global _cascade
    if _cascade is None:
        _cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        if _cascade.empty():
            raise RuntimeError("Could not load OpenCV's face cascade file.")
    return _cascade


def crop_face(img: Image.Image, margin: float = 0.4):
    """Crop the largest detected face (with margin). Returns (PIL image, face_found).
    If no face is found the whole image is returned unchanged."""
    img = img.convert("RGB")
    w, h = img.size
    gray = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2GRAY)

    scale = 1.0
    if max(w, h) > 640:
        scale = 640.0 / max(w, h)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)))

    min_face = max(30, int(0.08 * min(gray.shape)))
    faces = _get_cascade().detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_face, min_face)
    )
    if len(faces) == 0:
        return img, False

    x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
    x, y, fw, fh = x / scale, y / scale, fw / scale, fh / scale
    cx, cy = x + fw / 2, y + fh / 2
    side = max(fw, fh) * (1 + 2 * margin)
    box = (
        max(0, int(cx - side / 2)),
        max(0, int(cy - side / 2)),
        min(w, int(cx + side / 2)),
        min(h, int(cy + side / 2)),
    )
    return img.crop(box), True
