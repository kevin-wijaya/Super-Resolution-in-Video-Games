import torch
import os
import os.path as osp
import sys
import argparse

from tqdm import tqdm
from PIL import Image
from torchvision import transforms
from basicsr.archs.pft_arch import PFT

model_path = {
    "classical": {
        "2": "experiments/pretrained_models/001_PFT_SRx2_scratch.pth",
        "3": "experiments/pretrained_models/002_PFT_SRx3_finetune.pth",
        "4": "experiments/pretrained_models/003_PFT_SRx4_finetune.pth",
    },
    "lightweight": {
        "2": "experiments/pretrained_models/101_PFT_light_SRx2_scratch.pth",
        "3": "experiments/pretrained_models/102_PFT_light_SRx3_finetune.pth",
        "4": "experiments/pretrained_models/103_PFT_light_SRx4_finetune.pth",
    }
}

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument("-i", "--in_path", type=str, default="", help="Input directory path containing LR images.")
    parser.add_argument("-o", "--out_path", type=str, default="results/test/", help="Output directory path.")
    parser.add_argument("--scale", type=int, default=4, help="Scale factor for SR.")
    parser.add_argument(
            "--checkpoint",
            type=str,
            default="",
            help="Path to checkpoint. If provided, this overrides predefined model weights."
            )
    parser.add_argument(
            "--task",
            type=str,
            default="classical",
            choices=['classical', 'lightweight'],
            help="Task for the model. classical: for classical SR models. lightweight: for lightweight models."
            )
    args = parser.parse_args()

    return args


def process_image(image_input_path, image_output_path, model, device):
    with torch.no_grad():
        image = Image.open(image_input_path).convert('RGB')
        img_t = transforms.ToTensor()(image)

        # Define augmentations as tensor operations and their inverse
        aug_pairs = [
            # identity
            (lambda x: x,
            lambda x: x),

            # rot90
            (lambda x: torch.rot90(x, 1, [1, 2]),
            lambda x: torch.rot90(x, 3, [1, 2])),

            # rot180
            (lambda x: torch.rot90(x, 2, [1, 2]),
            lambda x: torch.rot90(x, 2, [1, 2])),

            # rot270
            (lambda x: torch.rot90(x, 3, [1, 2]),
            lambda x: torch.rot90(x, 1, [1, 2])),

            # hflip
            (lambda x: torch.flip(x, [2]),
            lambda x: torch.flip(x, [2])),

            # hflip + rot90
            (lambda x: torch.rot90(torch.flip(x, [2]), 1, [1, 2]),
            lambda x: torch.flip(torch.rot90(x, 3, [1, 2]), [2])),

            # hflip + rot180
            (lambda x: torch.rot90(torch.flip(x, [2]), 2, [1, 2]),
            lambda x: torch.flip(torch.rot90(x, 2, [1, 2]), [2])),

            # hflip + rot270
            (lambda x: torch.rot90(torch.flip(x, [2]), 3, [1, 2]),
            lambda x: torch.flip(torch.rot90(x, 1, [1, 2]), [2])),
        ]

        outputs = []
        for aug_fn, inv_fn in aug_pairs:
            inp = aug_fn(img_t)
            inp_batch = inp.unsqueeze(0).to(device)
            out = model(inp_batch).clamp(0.0, 1.0)[0]
            # inverse transform the output (keep on device for speed)
            out_inv = inv_fn(out)
            outputs.append(out_inv.cpu())

        # Average outputs
        avg = torch.stack(outputs, dim=0).mean(dim=0).clamp(0.0, 1.0)
        image_output = transforms.ToPILImage()(avg)
        image_output.save(image_output_path)

def main():
    args = get_parser()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if args.task == 'classical':
        model = PFT(upscale=args.scale,
                    embed_dim=240,
                    depths=[4, 4, 4, 6, 6, 6],
                    num_heads=6,
                    num_topk=[1024, 1024, 1024, 1024,
                             256, 256, 256, 256,
                             128, 128, 128, 128,
                             64, 64, 64, 64, 64, 64,
                             32, 32, 32, 32, 32, 32,
                             16, 16, 16, 16, 16, 16, ],
                    window_size=32,
                    convffn_kernel_size=7,
                    mlp_ratio=2,
                    upsampler='pixelshuffle',
                    use_checkpoint=False,
                    )  
    elif args.task == 'lightweight':
        model = PFT(upscale=args.scale,
                    embed_dim=52,
                    depths=[2, 4, 6, 6, 6],
                    num_heads=4,
                    num_topk=[1024, 1024,
                              256, 256, 256, 256,
                              128, 128, 128, 128, 128, 128,
                              64, 64, 64, 64, 64, 64,
                              32, 32, 32, 32, 32, 32],
                    window_size=32,
                    convffn_kernel_size=7,
                    mlp_ratio=1,
                    upsampler='pixelshuffledirect',
                    use_checkpoint=False,
                    )

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=device)
        if isinstance(ckpt, dict) and 'params_ema' in ckpt:
            state_dict = ckpt['params_ema']
        elif isinstance(ckpt, dict) and 'state_dict' in ckpt:
            state_dict = ckpt['state_dict']
        else:
            state_dict = ckpt
    else:
        state_dict = torch.load(model_path[args.task][str(args.scale)], map_location=device)['params_ema']
    model.load_state_dict(state_dict, strict=True)
    model = model.to(device)
    model.eval()

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    if not os.path.isdir(args.in_path):
        raise ValueError("in_path must be a directory containing LR images.")

    for file in tqdm(sorted(os.listdir(args.in_path)), total=len(os.listdir(args.in_path))):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_input_path = osp.join(args.in_path, file)
            image_output_path = os.path.join(args.out_path, file)
            process_image(image_input_path, image_output_path, model, device)


if __name__ == "__main__":
    main()
