import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Average ensemble two folders of super-resolved images."
    )

    parser.add_argument(
        "--folder1",
        type=str,
        required=True,
        help="Path to the first folder containing inferred images."
    )

    parser.add_argument(
        "--folder2",
        type=str,
        required=True,
        help="Path to the second folder containing inferred images."
    )

    parser.add_argument(
        "--out_folder",
        type=str,
        required=True,
        help="Path to save ensembled output images."
    )

    return parser.parse_args()


def main():
    args = parse_args()

    folder1 = args.folder1
    folder2 = args.folder2
    output_folder = args.out_folder

    os.makedirs(output_folder, exist_ok=True)

    # Supported image formats
    valid_exts = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")

    # Get image names from the first folder
    image_names = [
        f for f in os.listdir(folder1)
        if f.lower().endswith(valid_exts)
    ]

    if len(image_names) == 0:
        print(f"No valid images found in {folder1}")
        return

    for name in tqdm(image_names, desc="Ensembling images"):
        img1_path = os.path.join(folder1, name)
        img2_path = os.path.join(folder2, name)

        # Skip if the second folder does not contain the image with the same name
        if not os.path.exists(img2_path):
            print(f"Skip: {name} not found in folder2")
            continue

        img1 = cv2.imread(img1_path, cv2.IMREAD_UNCHANGED)
        img2 = cv2.imread(img2_path, cv2.IMREAD_UNCHANGED)

        # Skip unreadable images
        if img1 is None or img2 is None:
            print(f"Skip: cannot read {name}")
            continue

        # Skip images with different shapes
        if img1.shape != img2.shape:
            print(f"Skip: shape mismatch {name}: {img1.shape} vs {img2.shape}")
            continue

        # Average the two images
        avg_img = (img1.astype(np.float32) + img2.astype(np.float32)) / 2.0

        # Convert back to uint8 before saving
        avg_img = np.clip(avg_img, 0, 255).astype(np.uint8)

        output_path = os.path.join(output_folder, name)
        cv2.imwrite(output_path, avg_img)

    print(f"Done! Ensembled images are saved to: {output_folder}")


if __name__ == "__main__":
    main()