"""Detection models. Heavy libraries (torch, librosa) are imported lazily so the web app
starts even before they are installed or the models are trained."""
from .config import AUDIO_WEIGHTS, IMAGE_WEIGHTS


def model_status():
    img = IMAGE_WEIGHTS.exists()
    return {"image": img, "video": img, "audio": AUDIO_WEIGHTS.exists()}


def analyze(path: str, media_type: str) -> dict:
    if media_type == "image":
        from .image_model import detect_image_deepfake
        return detect_image_deepfake(path)
    if media_type == "video":
        from .video_model import detect_video_deepfake
        return detect_video_deepfake(path)
    if media_type == "audio":
        from .audio_model import detect_audio_deepfake
        return detect_audio_deepfake(path)
    raise ValueError(f"Unknown media type: {media_type}")
