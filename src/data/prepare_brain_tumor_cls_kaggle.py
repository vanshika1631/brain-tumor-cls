import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
import os

# --- CONFIG ---
KAGGLE_DATASET_DIR = Path("data/brain_tumor_dataset")
OUTPUT_IMG_DIR = Path("data/processed/kaggle_images")
MANIFEST_PATH = Path("data/processed/kaggle_manifest.csv")
FIGSHARE_MANIFEST_PATH = Path("data/processed/figshare_manifest.csv")
COMBINED_MANIFEST_PATH = Path("data/processed/combined_manifest.csv")

# Create output directories
for cls in ["meningioma", "glioma", "pituitary"]:
    os.makedirs(OUTPUT_IMG_DIR / cls, exist_ok=True)

# --- LOAD DATA ---
print("[INFO] Loading .npy files...")
images = np.load(KAGGLE_DATASET_DIR / "images.npy", allow_pickle=True)  # allow_pickle=True is crucial
labels = np.load(KAGGLE_DATASET_DIR / "labels.npy", allow_pickle=True)

assert len(images) == len(labels), "Number of images and labels must match"

# Label mapping (same as Figshare dataset)
label_map = {1: "meningioma", 2: "glioma", 3: "pituitary"}

records = []

print("[INFO] Converting images and saving to disk...")
for i, (img_array, label_idx) in enumerate(zip(images, labels)):
    label = label_map[int(label_idx)]

    # Convert grayscale → RGB to match model input (e.g., ResNet expects 3 channels)
    img = Image.fromarray(np.uint8(img_array)).convert("RGB")
    img = img.resize((224, 224))  # standard input size for most CNNs

    filename = f"{label}_{i:05d}.png"
    img_path = OUTPUT_IMG_DIR / label / filename
    img.save(img_path)

    records.append((str(img_path), label))

# --- CREATE MANIFEST ---
df = pd.DataFrame(records, columns=["path", "label"])

# Shuffle before splitting
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Define split ratios
train_frac, val_frac = 0.8, 0.1
train_end = int(len(df) * train_frac)
val_end = int(len(df) * (train_frac + val_frac))

df.loc[:train_end - 1, "split"] = "train"
df.loc[train_end:val_end - 1, "split"] = "val"
df.loc[val_end:, "split"] = "test"

# Save Kaggle manifest
df.to_csv(MANIFEST_PATH, index=False)
print(f"✅ Kaggle manifest saved to {MANIFEST_PATH}, total samples = {len(df)}")

# --- OPTIONAL: Merge with Figshare manifest ---
if FIGSHARE_MANIFEST_PATH.exists():
    print("[INFO] Merging with Figshare manifest...")
    figshare_df = pd.read_csv(FIGSHARE_MANIFEST_PATH)
    merged_df = pd.concat([figshare_df, df]).reset_index(drop=True)
    merged_df.to_csv(COMBINED_MANIFEST_PATH, index=False)
    print(f"✅ Combined manifest saved to {COMBINED_MANIFEST_PATH}")
else:
    print(f"[WARN] Figshare manifest not found at {FIGSHARE_MANIFEST_PATH}, skipping merge.")
