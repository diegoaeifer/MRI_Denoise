"""
Ignite handler that terminates training when validation PSNR stays
negative for `threshold` consecutive epochs (indicates divergence).
"""

from __future__ import annotations

import logging
from typing import Optional

from ignite.engine import Engine, Events

logger = logging.getLogger(__name__)


class DivergenceStopHandler:
    """
    Attaches to an evaluator engine. Calls trainer.terminate() when
    val PSNR is negative for `threshold` consecutive complete evaluations.

    Usage:
        handler = DivergenceStopHandler(trainer, threshold=3)
        handler.attach(evaluator)
    """

    def __init__(
        self,
        trainer: Engine,
        threshold: int = 3,
        psnr_key: str = "psnr",
    ) -> None:
        self._trainer = trainer
        self._threshold = threshold
        self._psnr_key = psnr_key
        self._neg_count: int = 0

    def attach(self, evaluator: Engine) -> None:
        evaluator.add_event_handler(Events.EPOCH_COMPLETED, self)

    def __call__(self, evaluator: Engine) -> None:
        metrics: dict = evaluator.state.metrics or {}
        psnr: float = metrics.get(self._psnr_key, 1.0)

        if psnr < 0:
            self._neg_count += 1
            logger.warning(
                f"Negative val PSNR ({psnr:.2f} dB) — "
                f"divergence counter {self._neg_count}/{self._threshold}"
            )
            if self._neg_count >= self._threshold:
                logger.error(
                    "Training ABORTED: PSNR remained negative for "
                    f"{self._neg_count} consecutive epochs."
                )
                self._trainer.terminate()
        else:
            self._neg_count = 0
