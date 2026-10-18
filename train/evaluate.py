"""Score the *deployed* detector on any folder with real/ and fake/ sub-folders.
Use files the model was NOT trained on (e.g. your own photos) to see how it really behaves.

    python -m train.evaluate --media image --data my_test_images --limit 200
    python -m train.evaluate --media audio --data my_test_audio
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from models import analyze  # noqa: E402
from models.config import AUDIO_EXTS, IMAGE_EXTS, VIDEO_EXTS  # noqa: E402
from train.data_utils import list_files  # noqa: E402

EXTS = {"image": IMAGE_EXTS, "video": VIDEO_EXTS, "audio": AUDIO_EXTS}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--media", choices=EXTS, required=True)
    ap.add_argument("--data", required=True, help="folder containing real/ and fake/")
    ap.add_argument("--limit", type=int, default=200, help="max files per class")
    args = ap.parse_args()

    cm = {("REAL", "REAL"): 0, ("REAL", "FAKE"): 0, ("FAKE", "REAL"): 0, ("FAKE", "FAKE"): 0}
    errors = 0
    for truth in ("REAL", "FAKE"):
        for f in list_files(Path(args.data) / truth.lower(), EXTS[args.media])[: args.limit]:
            try:
                pred = analyze(str(f), args.media)["label"]
            except Exception as e:  # noqa: BLE001
                errors += 1
                print(f"  skipped {f.name}: {e}")
                continue
            cm[(truth, pred)] += 1
            if pred != truth:
                print(f"  WRONG  {f.name}: truth {truth}, predicted {pred}")

    total = sum(cm.values())
    if not total:
        sys.exit("No files were scored. Check the folder layout (real/ and fake/ sub-folders).")
    print(f"\nScored {total} files ({errors} skipped)")
    print("                 pred REAL   pred FAKE")
    print(f"true REAL        {cm[('REAL', 'REAL')]:>9}   {cm[('REAL', 'FAKE')]:>9}")
    print(f"true FAKE        {cm[('FAKE', 'REAL')]:>9}   {cm[('FAKE', 'FAKE')]:>9}")
    print(f"Accuracy: {(cm[('REAL', 'REAL')] + cm[('FAKE', 'FAKE')]) / total:.1%}")


if __name__ == "__main__":
    main()
