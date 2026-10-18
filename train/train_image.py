"""Train the image/video face-forgery classifier (MobileNetV2, transfer learning).

Data layout (copy your downloaded dataset images here; sub-folders are fine):
    kaggle_datasets/images/real/...
    kaggle_datasets/images/fake/...

Run from the project root:
    python -m train.train_image --epochs 5 --max-per-class 3000
Quick smoke test:
    python -m train.train_image --epochs 2 --max-per-class 300
"""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from PIL import Image  # noqa: E402
from sklearn.metrics import classification_report, confusion_matrix  # noqa: E402
from torch.utils.data import DataLoader, Dataset  # noqa: E402
from torchvision import transforms  # noqa: E402

from models.config import DATASET_DIR, IMAGE_EXTS, IMAGE_WEIGHTS, WEIGHTS_DIR  # noqa: E402
from models.image_model import MEAN, STD, build_model  # noqa: E402
from train.data_utils import balanced_sample, list_files, split_by_group  # noqa: E402

REAL, FAKE = 0, 1


class FaceDataset(Dataset):
    def __init__(self, items, tf, face_crop):
        self.items, self.tf, self.face_crop = items, tf, face_crop
        if face_crop:
            from models.face_utils import crop_face
            self._crop = crop_face

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, y = self.items[i]
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            img = Image.new("RGB", (256, 256))
        if self.face_crop:
            img, _ = self._crop(img)
        return self.tf(img), y


def set_train_mode(model, unfreeze_from):
    model.train()
    for i, block in enumerate(model.features):
        if i < unfreeze_from:
            block.eval()  # keep frozen BatchNorm statistics fixed


def run_epoch(model, loader, device, crit, optimizer=None, unfreeze_from=0):
    training = optimizer is not None
    if training:
        set_train_mode(model, unfreeze_from)
    else:
        model.eval()
    loss_sum, correct, n = 0.0, 0, 0
    with torch.enable_grad() if training else torch.no_grad():
        for step, (x, y) in enumerate(loader, 1):
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = crit(out, y)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            n += len(y)
            if training and step % 20 == 0:
                print(f"    step {step}/{len(loader)}  loss {loss_sum / n:.4f}  acc {correct / n:.3f}", flush=True)
    return loss_sum / n, correct / n


def predict_all(model, loader, device):
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            ps += model(x.to(device)).argmax(1).cpu().tolist()
            ys += y.tolist()
    return ys, ps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(DATASET_DIR / "images"))
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--max-per-class", type=int, default=3000)
    ap.add_argument("--img-size", type=int, default=160)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--unfreeze-from", type=int, default=12, help="train MobileNet blocks >= this index (0-18)")
    ap.add_argument("--face-crop", action="store_true",
                    help="crop faces with OpenCV while training (use if your images are full photos, not face crops)")
    ap.add_argument("--workers", type=int, default=0, help="keep 0 on Windows")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = Path(args.data)
    real = list_files(data / "real", IMAGE_EXTS)
    fake = list_files(data / "fake", IMAGE_EXTS)
    print(f"Found {len(real)} real and {len(fake)} fake images in {data}")
    if min(len(real), len(fake)) < 50:
        sys.exit(f"Need at least 50 images per class in {data / 'real'} and {data / 'fake'}. See README, step 4.")

    real, fake = balanced_sample(real, fake, args.max_per_class, args.seed)
    splits = {}
    for name, a, b in zip(("train", "val", "test"), split_by_group(real, args.seed), split_by_group(fake, args.seed)):
        splits[name] = [(p, REAL) for p in a] + [(p, FAKE) for p in b]
    print({k: len(v) for k, v in splits.items()}, "(real+fake images per split)")

    size = args.img_size
    train_tf = transforms.Compose([
        transforms.Resize((size, size)), transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.15, 0.15, 0.1), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    eval_tf = transforms.Compose([
        transforms.Resize((size, size)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    def loader(name, tf, shuffle):
        ds = FaceDataset(splits[name], tf, args.face_crop)
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle, num_workers=args.workers)

    train_dl, val_dl, test_dl = loader("train", train_tf, True), loader("val", eval_tf, False), loader("test", eval_tf, False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    torch.manual_seed(args.seed)
    model = build_model(pretrained=True).to(device)  # downloads ImageNet weights the first time
    for i, block in enumerate(model.features):
        for p in block.parameters():
            p.requires_grad = i >= args.unfreeze_from
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    crit = nn.CrossEntropyLoss()

    best_acc, best_state = -1.0, None
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        print(f"Epoch {epoch}/{args.epochs}", flush=True)
        tl, ta = run_epoch(model, train_dl, device, crit, opt, args.unfreeze_from)
        vl, va = run_epoch(model, val_dl, device, crit)
        sched.step()
        print(f"  train loss {tl:.4f} acc {ta:.3f} | val loss {vl:.4f} acc {va:.3f} | {time.time() - t0:.0f}s", flush=True)
        if va > best_acc:
            best_acc, best_state = va, copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})

    model.load_state_dict(best_state)
    ys, ps = predict_all(model, test_dl, device)
    report = classification_report(ys, ps, target_names=["real", "fake"], output_dict=True, zero_division=0)
    print("\nHELD-OUT TEST RESULTS (images the model never saw)")
    print(classification_report(ys, ps, target_names=["real", "fake"], zero_division=0))
    print("Confusion matrix [rows=true real/fake, cols=predicted real/fake]:")
    print(confusion_matrix(ys, ps))

    metrics = {"val_accuracy": float(best_acc), "test_accuracy": float(report["accuracy"]),
               "test_images": len(ys), "epochs": args.epochs}
    WEIGHTS_DIR.mkdir(exist_ok=True)
    torch.save({"state_dict": best_state, "img_size": size, "classes": ["real", "fake"], "metrics": metrics}, IMAGE_WEIGHTS)
    (WEIGHTS_DIR / "image_metrics.json").write_text(json.dumps({**metrics, "report": report}, indent=2))
    print(f"\nSaved {IMAGE_WEIGHTS}  (test accuracy {metrics['test_accuracy']:.3f})")


if __name__ == "__main__":
    main()
