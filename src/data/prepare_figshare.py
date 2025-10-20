# src/data/prepare_figshare.py
import os, csv, cv2, hashlib, random
from glob import glob
from pathlib import Path

random.seed(42)

# INPUT: images exported from .mat in class folders
IN_DIR   = Path("data/processed/figshare_images")   # meningioma/glioma/pituitary/*.png
# OUTPUT: standardized 224x224 JPGs (by class)
OUT_DIR  = Path("data/interim/figshare")
# MANIFEST: (path,label,split) for train/val/test
MANIFEST = Path("data/processed/figshare_manifest.csv")

CLASSES = ["glioma", "meningioma", "pituitary"]  # labels we expect

def to_square_224(img):
    """pad to square then resize to 224x224 (convert gray/alpha to RGB)."""
    if img is None:
        return None
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    h, w = img.shape[:2]
    m = max(h, w)
    top  = (m - h) // 2
    bottom = m - h - top
    left = (m - w) // 2
    right = m - w - left
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=0)
    img = cv2.resize(img, (224, 224), interpolation=cv2.INTER_AREA)
    return img

def md5_bytes(arr):
    return hashlib.md5(arr.tobytes()).hexdigest()

def main():
    assert IN_DIR.exists(), f"Input directory not found: {IN_DIR}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    seen = set()
    count = 0

    for cls in CLASSES:
        src_dir = IN_DIR / cls
        if not src_dir.exists():
            # skip missing class but keep going
            continue

        out_cls = OUT_DIR / cls
        out_cls.mkdir(parents=True, exist_ok=True)

        files = sorted(
            glob(str(src_dir / "*.*"))
        )
        for p in files:
            img = cv2.imread(p, cv2.IMREAD_UNCHANGED)
            if img is None:
                continue
            img = to_square_224(img)
            if img is None:
                continue

            # de-duplicate after standardization (pixel md5)
            h = md5_bytes(img)
            if h in seen:
                continue
            seen.add(h)

            # persistent name from hash (stable, no collisions in practice)
            out_name = f"{h}.jpg"
            out_path = out_cls / out_name
            ok = cv2.imwrite(str(out_path), img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
            if not ok:
                continue

            rows.append((str(out_path), cls))
            count += 1
            if count % 500 == 0:
                print(f"[prepare] wrote {count} images…")

    # split 70/15/15
    random.shuffle(rows)
    n = len(rows)
    n_tr = int(0.70 * n)
    n_val = int(0.15 * n)
    splits = ["train"] * n_tr + ["val"] * n_val + ["test"] * (n - n_tr - n_val)

    with open(MANIFEST, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "split"])
        for (row, split) in zip(rows, splits):
            w.writerow([row[0], row[1], split])

    print(f"[prepare] total kept: {len(rows)}")
    print(f"[prepare] wrote manifest -> {MANIFEST.resolve()}")

if __name__ == "__main__":
    main()
