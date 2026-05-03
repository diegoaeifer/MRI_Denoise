from ignite.engine import Events, Engine
import logging


class DivergenceStopHandler:
    """
    Early stopping handler that halts training if the validation metric (e.g., PSNR)
    drops below a certain threshold for a specified number of consecutive epochs.
    """

    def __init__(
        self,
        evaluator: Engine,
        metric_name: str = "val_psnr",
        threshold: float = 0.0,
        patience: int = 3,
    ):
        self.evaluator = evaluator
        self.metric_name = metric_name
        self.threshold = threshold
        self.patience = patience
        self.bad_epochs = 0
        self.logger = logging.getLogger(__name__)

    def attach(self, engine: Engine) -> None:
        engine.add_event_handler(Events.EPOCH_COMPLETED, self)

    def __call__(self, engine: Engine) -> None:
        metrics = self.evaluator.state.metrics
        if self.metric_name in metrics:
            val = metrics[self.metric_name]
            if val <= self.threshold:
                self.bad_epochs += 1
                self.logger.warning(
                    f"Divergence detected: {self.metric_name} = {val:.4f} <= {self.threshold}. "
                    f"Bad epochs: {self.bad_epochs}/{self.patience}"
                )
                if self.bad_epochs >= self.patience:
                    self.logger.error(
                        f"Stopping training due to divergence ({self.patience} bad epochs)."
                    )
                    engine.terminate()
            else:
                self.bad_epochs = 0
