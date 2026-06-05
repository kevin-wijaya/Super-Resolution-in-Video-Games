import os
import os.path as osp
import argparse
from PIL import Image
import numpy as np
from tqdm import tqdm
from torchvision import transforms as tv_transforms


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-e", "--experiments",
        type=str,
        nargs='+',
        required=True,
        help="List of experiment folder names (e.g., exp0 exp1 exp2)"
    )
    parser.add_argument(
        "-o", "--out_path",
        type=str,
        required=True,
        help="Output folder path to save ensembled images"
    )
    args = parser.parse_args()
    return args


def ensemble_images(exp_names, out_path):
    """
    Load images from results/{exp_name}/images for each experiment,
    average them, and save to output folder.
    """
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    # Collect all image filenames from the first experiment
    first_exp_path = osp.join("results", exp_names[0], "images")
    if not os.path.isdir(first_exp_path):
        raise ValueError(f"Path does not exist: {first_exp_path}")

    image_files = sorted([
        f for f in os.listdir(first_exp_path)
        if f.lower().endswith(('.png', '.jpg', '.jpeg'))
    ])

    print(f"Found {len(image_files)} images to ensemble from {len(exp_names)} experiments")

    # Process each image
    for image_file in tqdm(image_files, desc="Ensembling"):
        image_arrays = []

        # Load image from each experiment
        for exp_name in exp_names:
            image_path = osp.join("results", exp_name, "images", image_file)
            if not os.path.exists(image_path):
                print(f"Warning: {image_path} not found, skipping")
                continue

            img = Image.open(image_path).convert('RGB')
            arr = np.array(img)
            # detect whether image is already in [0,1]
            if np.issubdtype(arr.dtype, np.floating):
                maxv = float(arr.max()) if arr.size else 0.0
                if maxv <= 1.0:
                    img_f = arr.astype(np.float32)
                else:
                    img_f = arr.astype(np.float32) / 255.0
            else:
                img_f = arr.astype(np.float32) / 255.0

            image_arrays.append(img_f)

        if not image_arrays:
            print(f"Warning: No valid images found for {image_file}")
            continue

        # Average the images (they are in range [0,1])
        avg_array = np.mean(image_arrays, axis=0)
        avg_array = np.clip(avg_array, 0.0, 1.0)

        # Save using ToPILImage so we preserve the 0-1 semantics before conversion
        output_image = tv_transforms.ToPILImage()(avg_array)
        output_path = osp.join(out_path, image_file)
        output_image.save(output_path)

    print(f"Ensembled images saved to {out_path}")


def main():
    args = get_parser()
    ensemble_images(args.experiments, args.out_path)


if __name__ == "__main__":
    main()
