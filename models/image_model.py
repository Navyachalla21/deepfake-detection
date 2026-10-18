"""Image deepfake detector: MobileNetV2 fine-tuned on real vs fake faces (see train/train_image.py).
Class index 0 = REAL, 1 = FAKE."""
import time

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from torchvision import models as tvm
from torchvision import transforms

from .common import LOW_CONF_NOTE, label_from_prob, low_confidence
from .config import IMAGE_WEIGHTS
from .errors import MediaError, ModelNotTrainedError
from .face_utils import crop_face

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)


def build_model(pretrained: bool = False) -> nn.Module:
    weights = tvm.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    m = tvm.mobilenet_v2(weights=weights)
    m.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(m.last_channel, 2))
    return m


def make_eval_transform(img_size: int):
    return transforms.Compose(
        [transforms.Resize((img_size, img_size)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    )


_cache = {}


def load_detector():
    if "model" not in _cache:
        if not IMAGE_WEIGHTS.exists():
            raise ModelNotTrainedError(
                "The image/video model has not been trained yet. "
                "Put a dataset in kaggle_datasets/images and run: python -m train.train_image"
            )
        ckpt = torch.load(IMAGE_WEIGHTS, map_location="cpu", weights_only=True)
        model = build_model(pretrained=False)
        model.load_state_dict(ckpt["state_dict"])
        model.eval()
        _cache["model"] = model
        _cache["size"] = int(ckpt["img_size"])
    return _cache["model"], _cache["size"]


@torch.no_grad()
def predict_fake_prob(pil_images):
    """P(fake) for a list of RGB PIL images (already face-cropped). Returns a numpy array."""
    model, size = load_detector()
    tf = make_eval_transform(size)
    batch = torch.stack([tf(im) for im in pil_images])
    return torch.softmax(model(batch), dim=1)[:, 1].numpy()


def detect_image_deepfake(path: str) -> dict:
    t0 = time.time()
    load_detector()  # fail fast with a clear message if the model isn't trained
    try:
        img = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    except Exception:
        raise MediaError("Could not open this file as an image.")
    w, h = img.size
    if min(w, h) < 64:
        raise MediaError("Image is too small (minimum 64 x 64 pixels).")

    face, found = crop_face(img)
    # average the image with its mirror image: a cheap, safe accuracy boost
    p_fake = float(predict_fake_prob([face, ImageOps.mirror(face)]).mean())
    label, conf = label_from_prob(p_fake)

    where = "the detected face region" if found else "the whole image (no face was found)"
    details = f"A MobileNetV2 CNN analysed {where} and scored P(fake) = {p_fake * 100:.1f}%."
    if not found:
        details += " The model was trained on faces, so results on non-face images are less reliable."
    if low_confidence(conf):
        details += LOW_CONF_NOTE

    return {
        "label": label,
        "confidence": conf,
        "fake_probability": round(p_fake * 100, 1),
        "low_confidence": low_confidence(conf),
        "face_detected": found,
        "details": details,
        "features_used": ["Face crop", "CNN spatial features", "Mirror-averaged prediction"],
        "preprocessing": {
            "enhancement": "Face crop + ImageNet normalisation" if found else "Resize + ImageNet normalisation",
            "processing_time": f"{time.time() - t0:.2f} s",
        },
        "image_size": f"{w} x {h}",
    }
