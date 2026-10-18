"""Audio deepfake detector: MFCC + spectral features -> scikit-learn SVM (see train/train_audio.py).
Label 1 = FAKE, 0 = REAL."""
import base64
import io
import time

import joblib
import librosa
import numpy as np
from matplotlib.figure import Figure

from .common import LOW_CONF_NOTE, label_from_prob, low_confidence
from .config import AUDIO_WEIGHTS
from .errors import MediaError, ModelNotTrainedError

SR = 16000
WINDOW_S = 4.0
MAX_WINDOWS = 3
WIN = int(WINDOW_S * SR)


def load_audio(path: str, max_seconds: float = 60.0) -> np.ndarray:
    """Load mono 16 kHz audio and trim leading/trailing silence (same function used in training)."""
    try:
        y, _ = librosa.load(path, sr=SR, mono=True, duration=max_seconds)
    except Exception as e:
        raise MediaError(
            f"Could not decode this audio file ({e.__class__.__name__}). "
            "For MP3/M4A you may need ffmpeg installed, or convert the file to WAV."
        )
    trimmed, _ = librosa.effects.trim(y, top_db=30)
    if trimmed.size >= SR:
        y = trimmed
    if y.size < SR:
        raise MediaError("Audio is too short (need at least 1 second of sound).")
    return y


def fixed_window(y: np.ndarray, start: int = 0) -> np.ndarray:
    w = y[start:start + WIN]
    if w.size < WIN:
        w = np.pad(w, (0, WIN - w.size))
    return w


def extract_features(y: np.ndarray) -> np.ndarray:
    """172 numbers: mean and std over time of MFCCs, deltas and a few spectral descriptors."""
    y = y / (np.max(np.abs(y)) + 1e-9)  # peak-normalise so loudness doesn't decide the label
    S = np.abs(librosa.stft(y, n_fft=1024, hop_length=256))
    mfcc = librosa.feature.mfcc(y=y, sr=SR, n_mfcc=40, n_fft=1024, hop_length=256)
    parts = [
        mfcc,
        librosa.feature.delta(mfcc),
        librosa.feature.spectral_centroid(S=S, sr=SR),
        librosa.feature.spectral_bandwidth(S=S, sr=SR),
        librosa.feature.spectral_rolloff(S=S, sr=SR),
        librosa.feature.spectral_flatness(S=S),
        librosa.feature.zero_crossing_rate(y, frame_length=1024, hop_length=256),
        librosa.feature.rms(S=S, frame_length=1024),
    ]
    return np.concatenate([np.concatenate([p.mean(axis=1), p.std(axis=1)]) for p in parts]).astype(np.float32)


_cache = {}


def _load_bundle():
    if "bundle" not in _cache:
        if not AUDIO_WEIGHTS.exists():
            raise ModelNotTrainedError(
                "The audio model has not been trained yet. "
                "Put a dataset in kaggle_datasets/audio and run: python -m train.train_audio"
            )
        _cache["bundle"] = joblib.load(AUDIO_WEIGHTS)
    return _cache["bundle"]


def _spectrogram_png(y: np.ndarray) -> str:
    S = librosa.feature.melspectrogram(y=y[: SR * 10], sr=SR, n_mels=64, n_fft=1024, hop_length=256)
    S_db = librosa.power_to_db(S, ref=np.max)
    fig = Figure(figsize=(6, 2.2), dpi=110)
    ax = fig.add_subplot(111)
    ax.imshow(S_db, origin="lower", aspect="auto", cmap="magma")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def detect_audio_deepfake(path: str) -> dict:
    t0 = time.time()
    bundle = _load_bundle()
    pipe = bundle["pipeline"]
    y = load_audio(path)

    if y.size <= WIN:
        starts = [0]
    else:
        k = min(MAX_WINDOWS, max(1, y.size // WIN))
        starts = [int(s) for s in np.linspace(0, y.size - WIN, k)]
    X = np.stack([extract_features(fixed_window(y, s)) for s in starts])
    fake_col = list(pipe.classes_).index(1)
    p_fake = float(pipe.predict_proba(X)[:, fake_col].mean())
    label, conf = label_from_prob(p_fake)

    feature_analysis = {
        "Duration": f"{y.size / SR:.1f} s",
        "Windows scored": str(len(starts)),
        "Spectral centroid": f"{librosa.feature.spectral_centroid(y=y, sr=SR).mean():.0f} Hz",
        "Zero-crossing rate": f"{librosa.feature.zero_crossing_rate(y).mean():.3f}",
        "Spectral flatness": f"{librosa.feature.spectral_flatness(y=y).mean():.4f}",
        "RMS energy": f"{librosa.feature.rms(y=y).mean():.4f}",
    }
    details = (
        f"An SVM on MFCC and spectral features scored {len(starts)} window(s) of {WINDOW_S:.0f} s each; "
        f"average P(fake) = {p_fake * 100:.1f}%."
    )
    if low_confidence(conf):
        details += LOW_CONF_NOTE

    return {
        "label": label,
        "confidence": conf,
        "fake_probability": round(p_fake * 100, 1),
        "low_confidence": low_confidence(conf),
        "acoustic_score": round(p_fake * 100, 1),
        "features_used": ["MFCC (40) + deltas", "Spectral centroid/bandwidth/rolloff/flatness", "ZCR", "RMS"],
        "feature_analysis": feature_analysis,
        "spectrogram": _spectrogram_png(y),
        "details": details,
        "preprocessing": {
            "enhancement": "16 kHz mono, silence trimmed, peak-normalised",
            "processing_time": f"{time.time() - t0:.2f} s",
        },
    }
