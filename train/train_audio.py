"""Train the audio deepfake classifier (MFCC + spectral features -> SVM).

Data layout:
    kaggle_datasets/audio/real/...   (wav/mp3/flac/ogg/m4a)
    kaggle_datasets/audio/fake/...

Run from the project root:
    python -m train.train_audio --max-per-class 2000
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402
from sklearn.svm import SVC  # noqa: E402

from models.audio_model import SR, extract_features, fixed_window, load_audio  # noqa: E402
from models.config import AUDIO_EXTS, AUDIO_WEIGHTS, DATASET_DIR, WEIGHTS_DIR  # noqa: E402
from train.data_utils import balanced_sample, list_files  # noqa: E402


def featurize(path):
    try:
        return extract_features(fixed_window(load_audio(str(path), max_seconds=20), 0))
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATASET_DIR / "audio"))
    ap.add_argument("--max-per-class", type=int, default=2000)
    ap.add_argument("--jobs", type=int, default=-1, help="parallel workers (-1 = all cores)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = Path(args.data)
    real = list_files(data / "real", AUDIO_EXTS)
    fake = list_files(data / "fake", AUDIO_EXTS)
    print(f"Found {len(real)} real and {len(fake)} fake audio files in {data}")
    if min(len(real), len(fake)) < 30:
        sys.exit(f"Need at least 30 files per class in {data / 'real'} and {data / 'fake'}. See README, step 4.")
    real, fake = balanced_sample(real, fake, args.max_per_class, args.seed)

    paths = real + fake
    labels = [0] * len(real) + [1] * len(fake)
    print(f"Extracting features from {len(paths)} files (this is the slow part)...")
    feats = Parallel(n_jobs=args.jobs, verbose=5)(delayed(featurize)(p) for p in paths)
    keep = [i for i, f in enumerate(feats) if f is not None]
    if len(keep) < len(paths):
        print(f"Skipped {len(paths) - len(keep)} unreadable files")
    X = np.stack([feats[i] for i in keep])
    y = np.array([labels[i] for i in keep])

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=args.seed)
    pipe = make_pipeline(StandardScaler(), SVC(kernel="rbf", C=10, gamma="scale", probability=True, random_state=args.seed))
    print("Training SVM...")
    pipe.fit(X_tr, y_tr)

    pred = pipe.predict(X_te)
    report = classification_report(y_te, pred, target_names=["real", "fake"], output_dict=True, zero_division=0)
    print("\nHELD-OUT TEST RESULTS (files the model never saw)")
    print(classification_report(y_te, pred, target_names=["real", "fake"], zero_division=0))
    print("Confusion matrix [rows=true real/fake, cols=predicted real/fake]:")
    print(confusion_matrix(y_te, pred))

    metrics = {"test_accuracy": float(report["accuracy"]), "test_files": int(len(y_te)), "train_files": int(len(y_tr))}
    WEIGHTS_DIR.mkdir(exist_ok=True)
    joblib.dump({"pipeline": pipe, "sr": SR, "metrics": metrics}, AUDIO_WEIGHTS, compress=3)
    (WEIGHTS_DIR / "audio_metrics.json").write_text(json.dumps({**metrics, "report": report}, indent=2))
    print(f"\nSaved {AUDIO_WEIGHTS}  (test accuracy {metrics['test_accuracy']:.3f})")


if __name__ == "__main__":
    main()
