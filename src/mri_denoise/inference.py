import os
import argparse
import torch
from pathlib import Path
from monai.transforms import (
    Compose,
    LoadImaged,
    EnsureChannelFirstd,
    ScaleIntensityRangePercentilesd,
    SaveImaged,
    ScaleIntensityRanged,
    ConcatItemsd,
)
from monai.inferers import SlidingWindowInferer
from monai.data import DataLoader, Dataset
from mri_denoise.networks.registry import get_network
from mri_denoise.data.transforms.noise import SpatiallyVaryingNoised


def build_inference_transforms(cfg):
    spatial_dims = cfg.get("spatial_dims", 2)
    return Compose(
        [
            LoadImaged(keys="image", reader="ITKReader"),
            EnsureChannelFirstd(keys="image"),
            ScaleIntensityRangePercentilesd(
                keys="image", lower=0.05, upper=99.5, b_min=0.0, b_max=1.0, clip=True
            ),
            SpatiallyVaryingNoised(
                keys="image",
                spatial_dims=spatial_dims,
                sigma_range=(0.05, 0.05),
                multiplier_range=(1.0, 1.0),
            ),
            ScaleIntensityRanged(
                keys="image", a_min=0.0, a_max=1.0, b_min=0.0, b_max=1.0, clip=True
            ),
            ConcatItemsd(keys=["image", "sigma"], name="image", dim=0),
        ]
    )


def main(args):
    device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")

    # Load Network
    model_cfg = {
        "in_channels": 2,
        "out_channels": 1,
        "spatial_dims": args.spatial_dims,
        "width": 64,
        "enc_blk_nums": [2, 2, 4, 8],
        "middle_blk_num": 12,
        "dec_blk_nums": [2, 2, 2, 2],
    }
    network = get_network(args.network, **model_cfg).to(device)

    if args.ckpt:
        checkpoint = torch.load(args.ckpt, map_location=device, weights_only=True)
        # Extract state dict (MONAI CheckpointSaver uses 'net')
        state_dict = checkpoint.get("net", checkpoint)
        network.load_state_dict(state_dict)

    network.eval()

    # Datalist
    files = list(Path(args.input_dir).rglob("*.*"))
    # Filter valid extensions loosely
    files = [str(f) for f in files if f.suffix.lower() in [".dcm", ".nii", ".nii.gz"]]
    datalist = [{"image": f} for f in files]

    # Transforms
    cfg = {"spatial_dims": args.spatial_dims}
    transforms = build_inference_transforms(cfg)

    dataset = Dataset(data=datalist, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=2)

    inferer = SlidingWindowInferer(
        roi_size=args.roi_size, sw_batch_size=4, overlap=0.25, mode="gaussian"
    )

    saver = SaveImaged(
        keys="pred",
        output_dir=args.output_dir,
        output_postfix="denoised",
        resample=False,
        print_log=True,
    )

    with torch.no_grad():
        for batch_data in dataloader:
            inputs = batch_data["image"].to(device)
            # Sliding window expects (B, C, H, W, [D])
            with torch.amp.autocast("cuda", enabled=True):
                batch_data["pred"] = inferer(inputs, network)

            # Move back to CPU for saving
            batch_data["pred"] = batch_data["pred"].cpu()
            saver(batch_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, type=str)
    parser.add_argument("--output_dir", required=True, type=str)
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--network", type=str, default="nafnet")
    parser.add_argument("--spatial_dims", type=int, default=2)
    parser.add_argument(
        "--roi_size",
        nargs="+",
        type=int,
        default=[256, 256],
        help="ROI size for sliding window",
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()

    # Ensure roi_size length matches spatial_dims
    if len(args.roi_size) != args.spatial_dims:
        if args.spatial_dims == 3 and len(args.roi_size) == 2:
            args.roi_size.append(32)  # default Z dim
        else:
            args.roi_size = args.roi_size[: args.spatial_dims]

    main(args)
