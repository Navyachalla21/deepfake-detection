# Cloud-Based Deepfake Detection (Image, Video, Audio)

Flask web app with login, dashboard, scan history and three detectors:

| Media | Detector | Trained with |
|---|---|---|
| Image | MobileNetV2 CNN on a face crop (PyTorch) | `python -m train.train_image` |
| Video | Same CNN on 16 sampled frames, averaged, plus a temporal-consistency score | uses the image model |
| Audio | MFCC + spectral features into an SVM (librosa + scikit-learn) | `python -m train.train_audio` |

**The important difference from the first version:** the app never guesses. If a model has not been
trained, the dashboard says so and refuses to give a label. Every label comes from weights you trained
on real data, and each training script prints accuracy on files the model never saw.

## 1. Setup (VS Code, Windows)

Use **Python 3.11 or 3.12** (newer versions can lack wheels for librosa/numba).

```powershell
# in the VS Code terminal, inside this folder
python -m venv venv
.\venv\Scripts\Activate.ps1        # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt
copy .env.example .env
```

macOS/Linux: `python3 -m venv venv && source venv/bin/activate`, then the same `pip install`.
In VS Code press `Ctrl+Shift+P` -> **Python: Select Interpreter** -> pick the `venv` one.
For MP3/M4A audio, install ffmpeg (`winget install ffmpeg`); WAV works without it.

## 2. Run the app (works immediately, models untrained)

```powershell
python app.py
```
Open http://127.0.0.1:5000, register, log in. You will see a banner that the models are not trained.
That is correct. Do steps 3 to 5 next.

## 3. Get data

Everything goes in `kaggle_datasets/<type>/real` and `.../fake` (sub-folders are fine).

- **Images:** Kaggle `xhlulu/140k-real-and-fake-faces`. Real = FFHQ photos, fake = StyleGAN faces, 256 px.
  Copy `real_vs_fake/real-vs-fake/train/real` into `kaggle_datasets/images/real` and `.../train/fake`
  into `kaggle_datasets/images/fake`. You do not need all 100k; the trainer samples a balanced subset.
- **Audio:** Kaggle `birdy654/deep-voice-deepfake-voice-recognition`. Copy `AUDIO/REAL` into
  `kaggle_datasets/audio/real` and `AUDIO/FAKE` into `kaggle_datasets/audio/fake`.
- **Video (optional):** put real/fake face videos (FaceForensics++, a DFDC sample) in
  `kaggle_datasets/videos/real` and `.../fake`, then run
  `python -m train.extract_frames --frames 8` to turn them into face crops for the image model.

## 4. Train

```powershell
# smoke test first (a few minutes) to check everything runs
python -m train.train_image --epochs 2 --max-per-class 300
# real run (roughly 20-60 min on a laptop CPU, depends on the machine)
python -m train.train_image --epochs 5 --max-per-class 3000

python -m train.train_audio --max-per-class 2000
```
Each script ends with a held-out test report (accuracy, precision, recall, confusion matrix) and saves
weights to `weights/` plus `image_metrics.json` / `audio_metrics.json`. **Quote those numbers in your
report's Result Analysis**, not general claims.

Useful flags: `--face-crop` (if your images are full photos, not face crops), `--img-size 224`,
`--unfreeze-from 8` (train more layers; slower, often more accurate).

Restart `python app.py` after training. The banner disappears.

## 5. Check it on your own files

```powershell
python -m train.evaluate --media image --data my_test_images
```
`my_test_images` needs `real/` and `fake/` sub-folders. Wrong predictions are listed by file name.

## What to expect (read this before demoing)

- A model only knows the kind of fakes it was trained on. The 140k dataset teaches it StyleGAN faces vs
  FFHQ photos. Ordinary phone/WhatsApp photos, face-swap videos, or Midjourney/Stable Diffusion images are
  outside that distribution, and the model may call real photos fake. To improve this, add a few hundred
  of your own real photos to `images/real` and fakes from other generators to `images/fake`, then retrain.
- Confidence below 65% is flagged as inconclusive in the UI.
- Video uses frame-level analysis plus a temporal statistic. It does not model motion with an LSTM.
  `temporal_score` is informational and is not used in the decision.
- Audio: the DEEP-VOICE speakers overlap between train and test, so audio accuracy on unseen
  speakers or other voice cloners will be lower than the printed test number.

## Optional: MongoDB, S3

Without them the app stores users and scans in `instance/local_db.json` and uploads in `uploads/`.
To use MongoDB or S3: `pip install -r requirements-optional.txt` and fill the variables in `.env`.

## Deploying (Render) later

Train locally and commit `weights/image_model.pt` (about 9 MB) and `weights/audio_model.joblib`.
Set `SECRET_KEY` as an environment variable (with several workers a random per-process key breaks
logins). Start command: `gunicorn app:app --workers 1 --threads 2 --timeout 180`.
Render's free tier has 512 MB RAM, which is tight for PyTorch plus librosa; if the service restarts
while loading a model, use a larger instance.

## Project layout

```
app.py                Flask routes (auth, /upload, /api/history, /api/status)
models/               image_model.py, video_model.py, audio_model.py, face_utils.py
train/                train_image.py, train_audio.py, extract_frames.py, evaluate.py
utils/                db.py (MongoDB or local JSON), storage.py (local or S3)
templates/, static/   UI
weights/              trained models are saved here
kaggle_datasets/      put your data here
```

Left out versus the original report code: the dataset-browser page and fetching images from a URL
(the latter let the server download arbitrary links). Passwords are now hashed.
