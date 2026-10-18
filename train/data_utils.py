"""Dataset helpers with no heavy dependencies (shared by the training scripts)."""
import random
from pathlib import Path


def list_files(folder, exts):
    folder = Path(folder)
    if not folder.exists():
        return []
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def group_key(path: Path) -> str:
    """Frames cut from one video (vid_<name>_<k>.jpg) share a group so they never straddle
    the train/test boundary, which would inflate accuracy."""
    stem = Path(path).stem
    return stem.rsplit("_", 1)[0] if stem.startswith("vid_") else str(path)


def balanced_sample(real, fake, max_per_class, seed=42):
    rng = random.Random(seed)
    real, fake = list(real), list(fake)
    rng.shuffle(real)
    rng.shuffle(fake)
    n = min(len(real), len(fake), max_per_class)
    return real[:n], fake[:n]


def split_by_group(paths, seed=42, train=0.70, val=0.15):
    """Split a list of paths into (train, val, test) keeping groups together."""
    groups = {}
    for p in paths:
        groups.setdefault(group_key(p), []).append(p)
    keys = sorted(groups)
    random.Random(seed).shuffle(keys)
    a, b = int(train * len(keys)), int((train + val) * len(keys))
    pick = lambda ks: [p for k in ks for p in groups[k]]  # noqa: E731
    return pick(keys[:a]), pick(keys[a:b]), pick(keys[b:])
