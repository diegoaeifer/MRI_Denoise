import torch
from monai.engines import SupervisedTrainer, SupervisedEvaluator
from monai.handlers import (
    StatsHandler,
    TensorBoardStatsHandler,
    TensorBoardImageHandler,
    ValidationHandler,
    CheckpointSaver,
    LrScheduleHandler,
    MeanSquaredError,
    PeakSignalToNoiseRatio,
    SSIMHandler,
    from_engine,
)
from .handlers.divergence import DivergenceStopHandler


def build_evaluator(network, val_loader, device, cfg):
    val_handlers = [
        StatsHandler(output_transform=lambda x: None),
        TensorBoardStatsHandler(
            log_dir=cfg.get("log_dir", "logs"), output_transform=lambda x: None
        ),
        CheckpointSaver(
            save_dir=cfg.get("ckpt_dir", "checkpoints"),
            save_dict={"net": network},
            save_key_metric=True,
            key_metric_name="val_psnr",
        ),
    ]

    return SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=network,
        key_val_metric={
            "val_psnr": PeakSignalToNoiseRatio(
                max_val=1.0, output_transform=from_engine(["pred", "label"])
            )
        },
        additional_metrics={
            "val_mse": MeanSquaredError(
                output_transform=from_engine(["pred", "label"])
            ),
            "val_ssim": SSIMHandler(
                spatial_dims=cfg.get("spatial_dims", 2),
                data_range=1.0,
                output_transform=from_engine(["pred", "label"]),
            ),
        },
        val_handlers=val_handlers,
        amp=cfg.get("use_amp", True),
        amp_kwargs={"dtype": torch.float16},
    )


def build_trainer(
    network, loss_fn, optimizer, scheduler, train_loader, evaluator, device, cfg
):
    train_handlers = [
        LrScheduleHandler(scheduler, print_lr=True),
        ValidationHandler(
            validator=evaluator, interval=cfg.get("val_interval", 1), epoch_level=True
        ),
        StatsHandler(
            tag_name="train_loss", output_transform=from_engine(["loss"], first=True)
        ),
        TensorBoardStatsHandler(
            log_dir=cfg.get("log_dir", "logs"),
            tag_name="train_loss",
            output_transform=from_engine(["loss"], first=True),
        ),
        CheckpointSaver(
            save_dir=cfg.get("ckpt_dir", "checkpoints"),
            save_dict={"net": network, "opt": optimizer, "sched": scheduler},
            save_interval=cfg.get("save_interval", 10),
            n_saved=3,
        ),
    ]

    divergence_handler = DivergenceStopHandler(
        evaluator=evaluator, metric_name="val_psnr", threshold=0.0, patience=3
    )
    # We will attach it after instantiation in the main train.py or we can attach here if we had the trainer obj.
    # The clean way is to return the handler or instantiate the trainer then attach.

    trainer = SupervisedTrainer(
        device=device,
        max_epochs=cfg.get("epochs", 100),
        train_data_loader=train_loader,
        network=network,
        optimizer=optimizer,
        loss_function=loss_fn,
        train_handlers=train_handlers,
        amp=cfg.get("use_amp", True),
        amp_kwargs={"dtype": torch.float16},
    )

    divergence_handler.attach(trainer)

    return trainer
