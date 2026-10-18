"""Cut face crops out of real/fake videos so the image model can learn from video data.

    kaggle_datasets/videos/real/*.mp4
    kaggle_datasets/videos/fake/*.mp4
        -->  kaggle_datasets/images/real/vid_<name>_<k>.jpg  (and .../fake/)

Run from the project root, then retrain:
    python -m train.extract_frames --frames 8
    python -m train.train_image
"""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from models.config import DATASET_DIR, VIDEO_EXTS  # noqa: E402
from models.face_utils import crop_face  # noqa: E402
from train.data_utils import list_files  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(DATASET_DIR / "videos"))
    ap.add_argument("--dst", default=str(DATASET_DIR / "images"))
    ap.add_argument("--frames", type=int, default=8, help="frames per video")
    ap.add_argument("--max-videos", type=int, default=500, help="per class")
    args = ap.parse_args()

    for label in ("real", "fake"):
        out_dir = Path(args.dst) / label
        out_dir.mkdir(parents=True, exist_ok=True)
        videos = list_files(Path(args.src) / label, VIDEO_EXTS)[: args.max_videos]
        saved = 0
        for vi, video in enumerate(videos, 1):
            cap = cv2.VideoCapture(str(video))
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            for k, idx in enumerate(np.unique(np.linspace(0, max(total - 1, 0), args.frames).astype(int))):
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
                ok, bgr = cap.read()
                if not ok:
                    continue
                crop, found = crop_face(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)))
                if found:  # only keep frames where a face was detected
                    crop.resize((256, 256)).save(out_dir / f"vid_{video.stem}_{k}.jpg", quality=95)
                    saved += 1
            cap.release()
            if vi % 25 == 0:
                print(f"  {label}: {vi}/{len(videos)} videos, {saved} face crops", flush=True)
        print(f"{label}: {len(videos)} videos -> {saved} face crops in {out_dir}")


if __name__ == "__main__":
    main()
