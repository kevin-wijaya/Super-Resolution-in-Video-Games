import os
import os.path as osp
import argparse
import subprocess
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
    out_arg = args.out_path
    
    # If user provided a simple experiment name (like 'ensemble_exp'),
    # save into results/{out_arg}/images to match inference scripts.
    if (os.path.sep not in out_arg) and (not out_arg.startswith('results')) and (not out_arg.endswith('images')):
        out_dir = osp.join('results', out_arg, 'images')
        exp_name = out_arg
    else:
        # Ensure path ends with /images
        if out_arg.endswith('images'):
            out_dir = out_arg
        else:
            out_dir = osp.join(out_arg, 'images')
        
        # Extract experiment name from path for CSV naming
        if 'results' in out_dir:
            exp_name = out_dir.split(osp.sep)[-2]
        else:
            exp_name = 'ensemble'
    
    ensemble_images(args.experiments, out_dir)
    
    # Generate CSV submission
    csv_path = osp.join('results', exp_name, f'{exp_name}.csv')
    try:
        print(f"Generating CSV submission to {csv_path}")
        os.makedirs(osp.dirname(csv_path), exist_ok=True)
        subprocess.run([
            'python', 'gen_submission.py',
            '--folder', out_dir,
            '--save-path', csv_path,
            '--public-size', '1'
        ], check=True)
        print(f"CSV submission saved to {csv_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error generating CSV: {e}")
    except FileNotFoundError:
        print("gen_submission.py not found. Skipping CSV generation.")


if __name__ == "__main__":
    main()
