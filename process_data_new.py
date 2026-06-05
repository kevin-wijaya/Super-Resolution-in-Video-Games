import os
import cv2
import random
import numpy as np
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

# Number of validation images
NUM_VAL_IMAGES = 1000

# Fixed seed for reproducibility
SEED = 42

# Patch sizes
LR_PATCH_SIZE = 64
HR_PATCH_SIZE = LR_PATCH_SIZE * 4  # 256

# PSNR threshold for saving patches
PSNR_THRESHOLD = 18.0

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
for fname in tqdm(lr_filenames):
    lr_path_tmp = os.path.join(LR_DIR, fname)
    lr_img_tmp = cv2.imread(lr_path_tmp)
    if lr_img_tmp is None:
        continue
    h_lr_tmp, w_lr_tmp = lr_img_tmp.shape[:2]
    dims[fname] = (w_lr_tmp, h_lr_tmp)

# ─── FUNCTION TO GENERATE NON-OVERLAPPING PATCHES FOR ONE IMAGE ───────────────
def calc_psnr(img1, img2):
    mse = ((img1.astype("float32") - img2.astype("float32")) ** 2).mean()
    if mse == 0:
        return float("inf")
    return 20 * (np.log10(255.0) - 0.5 * np.log10(mse))

def generate_crops_for_image(lr_fname: str, split: str):
    """
    Generate non-overlapping patches from a single LR/HR pair.
    Save patches into train or val folders based on image split.

    Returns saved_this_image.
    """
    w_lr_img, h_lr_img = dims[lr_fname]
    base_name = os.path.splitext(lr_fname)[0]  # e.g. "00011"
    
    lr_path = os.path.join(LR_DIR, lr_fname)
    hr_path = os.path.join(HR_DIR, lr_fname)  # assume same filename for HR

    lr_img = cv2.imread(lr_path)
    hr_img = cv2.imread(hr_path)

    saved_this_image = 0

    max_x_lr = (w_lr_img // LR_PATCH_SIZE) * LR_PATCH_SIZE
    max_y_lr = (h_lr_img // LR_PATCH_SIZE) * LR_PATCH_SIZE

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

            # Bicubic upscale LR patch to HR size for PSNR check
            lr_up = cv2.resize(
                lr_patch,
                (HR_PATCH_SIZE, HR_PATCH_SIZE),
                interpolation=cv2.INTER_CUBIC,
            )
            psnr = calc_psnr(lr_up, hr_patch)
            if psnr < PSNR_THRESHOLD:
                patch_idx += 1
                continue

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


def generate_random_patch_for_val(lr_fname: str):
    """
    Generate a single random 64x64 patch for validation.
    Returns 1 if saved, else 0.
    """
    w_lr_img, h_lr_img = dims[lr_fname]
    base_name = os.path.splitext(lr_fname)[0]

    lr_path = os.path.join(LR_DIR, lr_fname)
    hr_path = os.path.join(HR_DIR, lr_fname)

    lr_img = cv2.imread(lr_path)
    hr_img = cv2.imread(hr_path)

    max_x_lr = w_lr_img - LR_PATCH_SIZE
    max_y_lr = h_lr_img - LR_PATCH_SIZE

    if max_x_lr < 0 or max_y_lr < 0:
        return 0

    for _ in range(10):
        x_lr = random.randint(0, max_x_lr)
        y_lr = random.randint(0, max_y_lr)

        x_hr = x_lr * 4
        y_hr = y_lr * 4

        lr_patch = lr_img[y_lr : y_lr + LR_PATCH_SIZE,
                          x_lr : x_lr + LR_PATCH_SIZE]
        hr_patch = hr_img[y_hr : y_hr + HR_PATCH_SIZE,
                          x_hr : x_hr + HR_PATCH_SIZE]

        lr_up = cv2.resize(
            lr_patch,
            (HR_PATCH_SIZE, HR_PATCH_SIZE),
            interpolation=cv2.INTER_CUBIC,
        )
        psnr = calc_psnr(lr_up, hr_patch)
        if psnr < PSNR_THRESHOLD:
            continue

        out_lr_name = f"{base_name}_p0.png"
        out_hr_name = f"{base_name}_p0.png"

        cv2.imwrite(os.path.join(VAL_LR_OUT, out_lr_name), lr_patch)
        cv2.imwrite(os.path.join(VAL_HR_OUT, out_hr_name), hr_patch)
        return 1

    return 0


def write_meta_info(gt_dir: str, save_path: str):
    exts = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
    with open(save_path, "w", encoding="utf-8") as f:
        count = 0
        for root, _, files in os.walk(gt_dir):
            for name in sorted(files):
                if name.lower().endswith(exts):
                    rel_path = os.path.relpath(os.path.join(root, name), gt_dir)
                    f.write(rel_path + "\n")
                    count += 1
    return count


# ─── MAIN PATCH GENERATION LOOP ────────────────────────────────────────────────

if __name__ == "__main__":
    random.seed(SEED)

    # Shuffle LR filenames so cropping order is random
    random.shuffle(lr_filenames)

    val_count_images = min(NUM_VAL_IMAGES, len(lr_filenames))
    val_files = lr_filenames[:val_count_images]
    train_files = lr_filenames[val_count_images:]

    total_patches = 0
    for lr_file in lr_filenames:
        w_lr, h_lr = dims[lr_file]
        patch_cols = (w_lr // LR_PATCH_SIZE)
        patch_rows = (h_lr // LR_PATCH_SIZE)
        if lr_file in train_files:
            total_patches += patch_cols * patch_rows
        else:
            total_patches += 1

    pbar = tqdm(total=total_patches, desc="Generating Patches")

    train_count = 0
    val_count = 0
    for lr_file in train_files:
        saved = generate_crops_for_image(lr_file, "train")
        train_count += saved
        pbar.update(saved)

    for lr_file in val_files:
        saved = generate_random_patch_for_val(lr_file)
        val_count += saved
        pbar.update(saved)

    pbar.close()
    print(f"Done! Generated {train_count} train and {val_count} val paired patches.")

    train_meta = write_meta_info(TRAIN_HR_OUT, os.path.join(OUTPUT_BASE, "meta_info_train.txt"))
    val_meta = write_meta_info(VAL_HR_OUT, os.path.join(OUTPUT_BASE, "meta_info_val.txt"))
    print(f"Meta info saved. Train: {train_meta}, Val: {val_meta}")