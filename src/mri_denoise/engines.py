"""
MONAI-native training and evaluation engines for MRI denoising.

Replaces src/trainer.py. Uses ignite-based SupervisedTrainer / SupervisedEvaluator
from MONAI with custom handlers for divergence detection and gradient norm logging.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from monai.engines import SupervisedEvaluator, SupervisedTrainer
from monai.handlers import (
    CheckpointSaver,
    MeanDice,
    StatsHandler,
    TensorBoardStatsHandler,
    ValidationHandler,
)
from monai.inferers import SimpleInferer
from monai.transforms import Activations, AsDiscrete, Compose
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Custom handlers
# ──────────────────────────────────────────────────────────────────────────────


class DivergenceStopHandler:
    """
    Attaches to the evaluator's EPOCH_COMPLETED event.
    Aborts training if val PSNR stays negative for `threshold` consecutive epochs.
    """

    def __init__(self, trainer: Engine, threshold: int = 3, psnr_key: str = "psnr"):
        self._trainer = trainer
        self._threshold = threshold
        self._psnr_key = psnr_key
        self._neg_count = 0

    def attach(self, evaluator: Engine) -> None:
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, self)

    def __call__(self, evaluator: Engine) -> None:
        metrics = evaluator.state.metrics or {}
        psnr = metrics.get(self._psnr_key, 1.0)  # default positive → no abort
        if psnr < 0:
            self._neg_count += 1
            if self._neg_count >= self._threshold:
                logger.error(
                    "Training ABORTED: PSNR remained negative for "
                    f"{self._neg_count} consecutive epochs — likely divergence."
                )
                self._trainer.terminate()
        else:
            self._neg_count = 0


class GradNormHandler:
    """
    Clips gradients and logs the norm to TensorBoard after each iteration.
    Attaches to the trainer's ITERATION_COMPLETED event.
    """

    def __init__(
        self,
        model: nn.Module,
        writer: Any,  # SummaryWriter
        max_norm: float = 1.0,
        tag: str = "Stability/GradNorm",
    ):
        self._model = model
        self._writer = writer
        self._max_norm = max_norm
        self._tag = tag

    def attach(self, trainer: Engine) -> None:
        trainer.add_event_handler(Events.ITERATION_COMPLETED, self)

    def __call__(self, trainer: Engine) -> None:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            self._model.parameters(), max_norm=self._max_norm
        )
        global_step = trainer.state.iteration
        if self._writer is not None:
            self._writer.add_scalar(self._tag, grad_norm, global_step)


class EncoderFreezeHandler:
    """
    Calls model.set_epoch(epoch) at the start of every training epoch.
    Used by SwinUNETRDenoising to unfreeze the encoder after N epochs.
    """

    def __init__(self, model: nn.Module):
        self._model = model

    def attach(self, trainer: Engine) -> None:
        trainer.add_event_handler(Events.EPOCH_STARTED, self)

    def __call__(self, trainer: Engine) -> None:
        epoch = trainer.state.epoch - 1  # ignite epochs are 1-indexed
        if hasattr(self._model, "set_epoch"):
            self._model.set_epoch(epoch)


# ──────────────────────────────────────────────────────────────────────────────
# Prepare-batch helpers
# ──────────────────────────────────────────────────────────────────────────────


def _prepare_batch(
    batch: Dict[str, torch.Tensor],
    device: torch.device,
    non_blocking: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Standard prepare_batch for denoising:
    - batch["input"]  → model input  (B, 2, H, W[, D])
    - batch["target"] → clean image  (B, 1, H, W[, D])
    """
    inputs = batch["input"].to(device, non_blocking=non_blocking)
    targets = batch["target"].to(device, non_blocking=non_blocking)
    return inputs, targets


# ──────────────────────────────────────────────────────────────────────────────
# Engine factories
# ──────────────────────────────────────────────────────────────────────────────


def build_evaluator(
    network: nn.Module,
    val_loader: DataLoader,
    device: torch.device,
    cfg: Dict[str, Any],
    writer: Optional[Any] = None,
) -> SupervisedEvaluator:
    """Build a MONAI SupervisedEvaluator for the denoising pipeline."""
    use_amp = cfg.get("training", {}).get("use_amp", False)

    handlers = [
        StatsHandler(output_transform=lambda x: None, tag_name="val"),
    ]
    if writer is not None:
        handlers.append(
            TensorBoardStatsHandler(
                summary_writer=writer,
                output_transform=lambda x: None,
                tag_name="val",
            )
        )

    evaluator = SupervisedEvaluator(
        device=device,
        val_data_loader=val_loader,
        network=network,
        prepare_batch=_prepare_batch,
        inferer=SimpleInferer(),
        amp=use_amp,
        val_handlers=handlers,
    )
    return evaluator


def build_trainer(
    network: nn.Module,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    train_loader: DataLoader,
    evaluator: SupervisedEvaluator,
    device: torch.device,
    cfg: Dict[str, Any],
    scheduler: Optional[_LRScheduler] = None,
    writer: Optional[Any] = None,
    run_id: Optional[str] = None,
) -> SupervisedTrainer:
    """
    Build a MONAI SupervisedTrainer with:
    - GradNormHandler (gradient clipping + TensorBoard logging)
    - EncoderFreezeHandler (for SwinUNETRDenoising freeze schedule)
    - ValidationHandler (triggers evaluator every val_interval epochs)
    - CheckpointSaver (best model + periodic checkpoints)
    """
    training_cfg = cfg.get("training", {})
    use_amp = training_cfg.get("use_amp", False)
    val_interval = training_cfg.get("val_interval", 1)

    base_out = training_cfg.get("output_dir", "experiments")
    import datetime
    run_id = run_id or f"run_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = os.path.join(base_out, "checkpoints", run_id)
    os.makedirs(save_dir, exist_ok=True)

    # Loss wrapper: MONAI SupervisedTrainer expects loss_fn(pred, target) → scalar
    def _loss_fn(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss, _ = loss_fn(pred, target)
        return loss

    train_handlers: list[Any] = [
        ValidationHandler(validator=evaluator, interval=val_interval, epoch_level=True),
        StatsHandler(tag_name="train", output_transform=lambda x: x),
        CheckpointSaver(
            save_dir=save_dir,
            save_dict={"network": network, "optimizer": optimizer},
            save_interval=training_cfg.get("checkpoint_interval", 10),
            save_final=True,
        ),
    ]

    if writer is not None:
        train_handlers.append(
            TensorBoardStatsHandler(
                summary_writer=writer,
                tag_name="train",
                output_transform=lambda x: x,
            )
        )

    if scheduler is not None:
        from monai.handlers import LrScheduleHandler

        train_handlers.append(LrScheduleHandler(lr_scheduler=scheduler, print_lr=True))

    trainer = SupervisedTrainer(
        device=device,
        max_epochs=training_cfg.get("epochs", 100),
        train_data_loader=train_loader,
        network=network,
        optimizer=optimizer,
        loss_function=_loss_fn,
        prepare_batch=_prepare_batch,
        inferer=SimpleInferer(),
        amp=use_amp,
        train_handlers=train_handlers,
    )

    # Attach custom handlers
    GradNormHandler(network, writer).attach(trainer)
    EncoderFreezeHandler(network).attach(trainer)

    return trainer


def run_training(
    network: nn.Module,
    loss_fn: nn.Module,
    optimizer: Optimizer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: Dict[str, Any],
    scheduler: Optional[_LRScheduler] = None,
    writer: Optional[Any] = None,
    run_id: Optional[str] = None,
) -> None:
    """Top-level entry point: builds engines, attaches divergence handler, runs training."""
    evaluator = build_evaluator(network, val_loader, device, cfg, writer)
    trainer = build_trainer(
        network, loss_fn, optimizer, train_loader, evaluator,
        device, cfg, scheduler, writer, run_id,
    )

    # Divergence guard: abort if val PSNR stays negative for 3 epochs
    divergence_handler = DivergenceStopHandler(trainer, threshold=3)
    divergence_handler.attach(evaluator)

    trainer.run()
