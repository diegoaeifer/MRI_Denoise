import torch
import os
import yaml
import argparse

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
try:
    from pipeline import DenoisePipeline
except ImportError:
    from .pipeline import DenoisePipeline


def main(args):
    # Load Configs
    root_conf = "configs"
    config = {}
    for cfg_name in ["config_train.yaml", "config_data.yaml", "config_model.yaml"]:
        path = os.path.join(root_conf, cfg_name)
        if os.path.exists(path):
            with open(path) as f:
                config.update(yaml.safe_load(f))

    # Determine Checkpoint
    checkpoint_path = args.checkpoint
    if not checkpoint_path:
        # Try to find best model from experiments
        checkpoint_path = "experiments/checkpoints/best_model.pth"

    # Initialize Pipeline
    pipeline = DenoisePipeline(
        model_name=args.model,
        config=config,
        checkpoint_path=checkpoint_path,
        device=args.device
    )


    # Determine sigmas to run
    sigmas_to_run = []
    if args.sigmas:
        sigmas_to_run = args.sigmas
    elif args.sigma is not None:
        sigmas_to_run = [args.sigma]
    elif args.estimate_noise != 'none':
        # Pipeline will handle the estimation, just pass a dummy value to trigger the loop
        sigmas_to_run = [0.0]
    else:
        sigmas_to_run = [0.05] # default

    # Process
    if os.path.isdir(args.input):
        for s in sigmas_to_run:
            out_folder = args.output
            if len(sigmas_to_run) > 1:
                out_folder = f"{args.output}_sigma_{s}"

            est_flag = 'mad' if args.estimate_noise == 'mad' else None
            pipeline.process_folder(args.input, out_folder, sigma=s, estimate_noise=est_flag)
    else:
        # Single file
        for s in sigmas_to_run:
            out_file = args.output
            if len(sigmas_to_run) > 1:
                base, ext = os.path.splitext(args.output)
                out_file = f"{base}_sigma_{s}{ext}"

            est_flag = 'mad' if args.estimate_noise == 'mad' else None
            pipeline.process_dicom(args.input, out_file, sigma=s, estimate_noise=est_flag)
            if est_flag:
                print(f"Processed {args.input} -> {out_file} with estimated noise ({est_flag})")
            else:
                print(f"Processed {args.input} -> {out_file} with sigma {s}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="MRI Denoising Inference Script")
    parser.add_argument('--input', type=str, required=True, help='Input DICOM file or folder')
    parser.add_argument('--output', type=str, required=True, help='Output DICOM file or folder')
    parser.add_argument('--model', type=str, default='drunet', help='Model architecture')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to checkpoint pth file')
    parser.add_argument('--sigma', type=float, default=None, help='Single noise level sigma for model conditioning')
    parser.add_argument('--sigmas', type=float, nargs='+', default=None, help='Multiple noise level sigmas (e.g., 1.0 2.0 3.0). Overrides --sigma.')
    parser.add_argument('--estimate_noise', type=str, choices=['mad', 'none'], default='none', help='Method to estimate noise from the image itself (e.g., mad)')
    parser.add_argument('--device', type=str, default='cuda', help='Device to run on (cuda/cpu)')

    
    args = parser.parse_args()
    main(args)
