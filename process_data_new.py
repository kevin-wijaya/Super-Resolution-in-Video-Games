import os
import cv2
import random
from tqdm import tqdm

# ─── CONFIGURATION ────────────────────────────────────────────────────────────

# Input root directory (contains 'lr/' and 'hr/' subfolders)
ROOT_DIR = "dataset/train"
LR_DIR = os.path.join(ROOT_DIR, "lr")
HR_DIR = os.path.join(ROOT_DIR, "hr")

# Output base directory for paired patches
OUTPUT_BASE = "training_dataset"
TRAIN_LR_OUT = os.path.join(OUTPUT_BASE, "train", "LR")
TRAIN_HR_OUT = os.path.join(OUTPUT_BASE, "train", "HR")
VAL_LR_OUT = os.path.join(OUTPUT_BASE, "val", "LR")
VAL_HR_OUT = os.path.join(OUTPUT_BASE, "val", "HR")

# Train/val split ratio for LR images
TRAIN_RATIO = 0.9

# Fixed seed for reproducibility
SEED = 42

# Patch sizes
LR_PATCH_SIZE = 64
HR_PATCH_SIZE = LR_PATCH_SIZE * 4  # 256

# ─── PREPARE OUTPUT DIRECTORIES ───────────────────────────────────────────────

for path in [TRAIN_LR_OUT, TRAIN_HR_OUT, VAL_LR_OUT, VAL_HR_OUT]:
    os.makedirs(path, exist_ok=True)

# ─── GATHER LR FILENAMES AND PRECOMPUTE DIMENSIONS ─────────────────────────────

lr_filenames = [
    f for f in os.listdir(LR_DIR)
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
]
assert lr_filenames, f"No LR images found in {LR_DIR}"

# Precompute dimensions for each LR filename: (w_lr, h_lr)
dims = {}
for fname in lr_filenames:
    lr_path = os.path.join(LR_DIR, fname)
    lr_img = cv2.imread(lr_path)
    if lr_img is None:
        continue
    h_lr, w_lr = lr_img.shape[:2]
    dims[fname] = (w_lr, h_lr)

# ─── FUNCTION TO GENERATE NON-OVERLAPPING PATCHES FOR ONE IMAGE ───────────────

def generate_crops_for_image(lr_fname: str, split: str):
    """
    Generate non-overlapping patches from a single LR/HR pair.
    Save patches into train or val folders based on image split.

    Returns saved_this_image.
    """
    w_lr, h_lr = dims[lr_fname]
    base_name = os.path.splitext(lr_fname)[0]  # e.g. "00011"
    
    lr_path = os.path.join(LR_DIR, lr_fname)
    hr_path = os.path.join(HR_DIR, lr_fname)  # assume same filename for HR

    lr_img = cv2.imread(lr_path)
    hr_img = cv2.imread(hr_path)

    saved_this_image = 0

    max_x_lr = (w_lr // LR_PATCH_SIZE) * LR_PATCH_SIZE
    max_y_lr = (h_lr // LR_PATCH_SIZE) * LR_PATCH_SIZE

    patch_idx = 0
    for y_lr in range(0, max_y_lr, LR_PATCH_SIZE):
        for x_lr in range(0, max_x_lr, LR_PATCH_SIZE):
            # Corresponding HR coordinates (scale factor = 4)
            x_hr = x_lr * 4
            y_hr = y_lr * 4

            # Crop LR and HR
            lr_patch = lr_img[y_lr : y_lr + LR_PATCH_SIZE,
                              x_lr : x_lr + LR_PATCH_SIZE]
            hr_patch = hr_img[y_hr : y_hr + HR_PATCH_SIZE,
                              x_hr : x_hr + HR_PATCH_SIZE]

            # Filenames: "<base_name>_p<idx>.png"
            out_lr_name = f"{base_name}_p{patch_idx}.png"
            out_hr_name = f"{base_name}_p{patch_idx}.png"

            if split == "train":
                cv2.imwrite(os.path.join(TRAIN_LR_OUT, out_lr_name), lr_patch)
                cv2.imwrite(os.path.join(TRAIN_HR_OUT, out_hr_name), hr_patch)
            else:
                cv2.imwrite(os.path.join(VAL_LR_OUT, out_lr_name), lr_patch)
                cv2.imwrite(os.path.join(VAL_HR_OUT, out_hr_name), hr_patch)

            patch_idx += 1
            saved_this_image += 1

    return saved_this_image

# ─── MAIN PATCH GENERATION LOOP ────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(SEED)

    # Shuffle LR filenames so cropping order is random
    random.shuffle(lr_filenames)

    split_index = int(len(lr_filenames) * TRAIN_RATIO)
    train_files = lr_filenames[:split_index]
    val_files = lr_filenames[split_index:]

    total_patches = 0
    for lr_fname in lr_filenames:
        w_lr, h_lr = dims[lr_fname]
        max_x_lr = (w_lr // LR_PATCH_SIZE)
        max_y_lr = (h_lr // LR_PATCH_SIZE)
        total_patches += max_x_lr * max_y_lr

    pbar = tqdm(total=total_patches, desc="Generating Patches")

    train_count = 0
    val_count = 0
    for lr_fname in train_files:
        saved = generate_crops_for_image(lr_fname, "train")
        train_count += saved
        pbar.update(saved)

    for lr_fname in val_files:
        saved = generate_crops_for_image(lr_fname, "val")
        val_count += saved
        pbar.update(saved)

    pbar.close()
    print(f"Done! Generated {train_count} train and {val_count} val paired patches.")