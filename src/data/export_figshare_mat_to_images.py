from pathlib import Path
import numpy as np
import h5py
import imageio.v2 as imageio

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "data/raw/figshare"
DST = ROOT / "data/processed/figshare_images"

# 1: meningioma, 2: glioma, 3: pituitary (per Figshare readme)
LABELS = {1: "meningioma", 2: "glioma", 3: "pituitary"}

def to_uint8(img):
    img = np.asarray(img, dtype=np.float32)
    mn, mx = float(img.min()), float(img.max())
    if mx > mn:
        img = (img - mn) / (mx - mn)
    else:
        img = np.zeros_like(img)
    return (img * 255).astype(np.uint8)

def _deref_scalar(h5, ref_ds):
    """MATLAB v7.3 stores struct fields as object-ref datasets.
    This resolves a [1,1] reference dataset to a numpy array."""
    # ref_ds is usually a 2D array of object refs with shape (1, 1)
    ref = ref_ds[0, 0] if getattr(ref_ds, "shape", None) == (1, 1) else ref_ds[()]
    return h5[ref][()]

def load_mat_v73(path):
    """Returns (image2d, label_int) from a Figshare .mat file."""
    with h5py.File(path, "r") as h5:
        # 'cjdata' may be a group or an object-ref dataset
        if isinstance(h5["cjdata"], h5py.Group):
            cj = h5["cjdata"]
            # fields might be datasets or object-ref datasets
            img = cj["image"][()] if isinstance(cj["image"], h5py.Dataset) else _deref_scalar(h5, cj["image"])
            lab = cj["label"][()] if isinstance(cj["label"], h5py.Dataset) else _deref_scalar(h5, cj["label"])
        else:
            # cjdata itself is a [1,1] ref to a Group
            cj_group = h5[h5["cjdata"][0, 0]]
            img = _deref_scalar(h5, cj_group["image"])
            lab = _deref_scalar(h5, cj_group["label"])

    # squeeze & convert
    img = np.asarray(img).squeeze()
    lab = int(np.asarray(lab).squeeze())
    return img, lab

def main():
    DST.mkdir(parents=True, exist_ok=True)
    mats = sorted(p for p in SRC.rglob("*.mat") if p.name.lower() != "cvind.mat")
    if not mats:
        raise SystemExit(f"No .mat files found in {SRC}")

    count = 0
    for f in mats:
        try:
            img, label = load_mat_v73(f)
            img = to_uint8(img)
            cls = LABELS.get(label)
            if cls is None:
                continue
            out_dir = DST / cls
            out_dir.mkdir(parents=True, exist_ok=True)
            imageio.imwrite(out_dir / f"{f.stem}.png", img)
            count += 1
        except Exception as e:
            # skip corrupt/unexpected files but continue
            # print(f"Skip {f.name}: {e}")
            pass

    print(f"Export complete: {count} images -> {DST}")

if __name__ == "__main__":
    main()
