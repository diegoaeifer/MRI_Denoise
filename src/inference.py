import os
import yaml
import argparse
from pipeline import DenoisePipeline


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
        device=args.device,
    )

    # Process
    if os.path.isdir(args.input):
        pipeline.process_folder(args.input, args.output, sigma=args.sigma)
    else:
        # Single file
        pipeline.process_dicom(args.input, args.output, sigma=args.sigma)
        print(f"Processed {args.input} -> {args.output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MRI Denoising Inference Script")
    parser.add_argument(
        "--input", type=str, required=True, help="Input DICOM file or folder"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output DICOM file or folder"
    )
    parser.add_argument(
        "--model", type=str, default="drunet", help="Model architecture"
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="Path to checkpoint pth file"
    )
    parser.add_argument(
        "--sigma",
        type=float,
        default=0.05,
        help="Noise level sigma for model conditioning",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to run on (cuda/cpu)"
    )

    args = parser.parse_args()
    main(args)
