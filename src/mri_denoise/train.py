import argparse
import torch
import os
from monai.bundle import ConfigParser
from monai.data import CacheDataset, DataLoader, pad_list_data_collate
from mri_denoise.data.datalist import build_datalist
from mri_denoise.data.transforms import build_train_transforms, build_val_transforms
from mri_denoise.engines import build_trainer, build_evaluator

def main(cfg_path: str):
    parser = ConfigParser()
    parser.read_config(cfg_path)
    cfg = parser.get_parsed_content()

    device = torch.device(f"cuda:{cfg.get('gpu_id', 0)}" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    os.makedirs(cfg["training"]["log_dir"], exist_ok=True)
    os.makedirs(cfg["training"]["ckpt_dir"], exist_ok=True)

    datalist = build_datalist(cfg["data"])
    train_tf = build_train_transforms(cfg)
    val_tf = build_val_transforms(cfg)

    train_ds = CacheDataset(
        data=datalist["train"],
        transform=train_tf,
        cache_rate=cfg["data"]["cache_rate"],
        num_workers=cfg["data"]["num_workers"]
    )
    val_ds = CacheDataset(
        data=datalist["val"],
        transform=val_tf,
        cache_rate=cfg["data"]["cache_rate"],
        num_workers=cfg["data"]["num_workers"]
    )

    train_loader = DataLoader(
        train_ds, batch_size=cfg["training"]["batch_size"],
        shuffle=True, num_workers=cfg["data"]["num_workers"],
        pin_memory=True, collate_fn=pad_list_data_collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, num_workers=cfg["data"]["num_workers"], pin_memory=True
    )

    network = cfg["network"].to(device)
    loss_fn = cfg["losses"].to(device)

    optimizer = torch.optim.AdamW(network.parameters(), lr=cfg["training"]["lr"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["training"]["epochs"])

    # Merge cfg params for evaluator/trainer into one dict
    run_cfg = {
        "spatial_dims": cfg["spatial_dims"],
        "use_amp": cfg["training"]["use_amp"],
        "log_dir": cfg["training"]["log_dir"],
        "ckpt_dir": cfg["training"]["ckpt_dir"],
        "epochs": cfg["training"]["epochs"],
        "val_interval": cfg["training"]["val_interval"],
        "save_interval": cfg["training"]["save_interval"],
    }

    evaluator = build_evaluator(network, val_loader, device, run_cfg)
    trainer = build_trainer(network, loss_fn, optimizer, scheduler, train_loader, evaluator, device, run_cfg)

    trainer.run()

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    main(args.config)
