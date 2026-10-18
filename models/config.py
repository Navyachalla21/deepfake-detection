from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEIGHTS_DIR = ROOT / "weights"
IMAGE_WEIGHTS = WEIGHTS_DIR / "image_model.pt"
AUDIO_WEIGHTS = WEIGHTS_DIR / "audio_model.joblib"
DATASET_DIR = ROOT / "kaggle_datasets"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}
