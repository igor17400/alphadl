from typing import Any, Dict, List, Tuple, Optional
import torch
import torch.nn.functional as F
from torchmetrics import MetricCollection, MeanSquaredError, MeanAbsoluteError
from src.metrics.rank_ic import RankIC
from src.metrics.sharpe_ratio import SharpeRatio
from src.models.abstract_forecast import AbstractForecast
from src.models.core.stockmixer import StockMixer
from src.losses.loss_functions import stock_mixer_loss
import lightning as L
import wandb
from collections import defaultdict


class StockMixerForecastModule(AbstractForecast):
    def __init__(
        self,
        stocks: int,
        time_steps: int,
        channels: int,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        loss: str = "mse_loss",
        market: int = 20,
        scale: int = 8,
        alpha: float = 0.1,
        outputs: Dict[str, List[str]] = None,
        *args,
        **kwargs,
    ) -> None:
        if outputs is None:
            outputs = {
                "train": ["predictions", "targets", "loss"],
                "val": ["predictions", "targets", "loss"],
                "test": ["predictions", "targets", "loss", "sharpe_ratio", "rank_ic"],
            }

        super().__init__(
            outputs=outputs,
            optimizer=optimizer,
            scheduler=scheduler,
            loss=loss,
            *args,
            **kwargs,
        )

        self.save_hyperparameters(logger=False)

        # Initialize the core StockMixer model
        self.model = StockMixer(
            stocks=stocks,
            time_steps=time_steps,
            channels=channels,
            market=market,
            scale=scale,
        )

        # Metrics
        metrics = MetricCollection(
            {
                "mse": MeanSquaredError(),
                "mae": MeanAbsoluteError(),
                "rank_ic": RankIC(),
                "sharpe_ratio": SharpeRatio(),
            }
        )
        self.train_metrics = metrics.clone(prefix="train/")
        self.val_metrics = metrics.clone(prefix="val/")
        self.test_metrics = metrics.clone(prefix="test/")

        # Initialize step outputs
        self.training_step_outputs = {key: [] for key in outputs.get("train", [])}
        self.val_step_outputs = {key: [] for key in outputs.get("val", [])}
        self.test_step_outputs = {key: [] for key in outputs.get("test", [])}

        # Custom loss parameters
        self.alpha = alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x, y, base_price, mask = batch
        y_hat = self(x)

        # Use custom loss function
        loss, reg_loss, rank_loss, return_ratio = _stock_mixer_loss(
            y_hat, y, base_price, mask, x.size(0)
        )

        return loss, y_hat, y

    def training_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss, y_hat, y = self.model_step(batch)

        # Update metrics
        for name, metric in self.train_metrics.items():
            if name in ["mse", "mae"]:
                metric.update(y_hat, y)
            else:
                metric.update(y_hat.squeeze(-1), y.squeeze(-1))

        # Log loss
        self.log("train/loss", loss, prog_bar=True)
        self.train_loss(loss)

        # Store outputs
        self.training_step_outputs["preds"].append(y_hat)
        self.training_step_outputs["targets"].append(y)
        self.training_step_outputs["loss"].append(loss)

        return loss

    def validation_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss, y_hat, y = self.model_step(batch)

        # Handle NaN values
        if torch.isnan(y_hat).any() or torch.isnan(y).any():
            return loss

        # Update metrics
        for name, metric in self.val_metrics.items():
            if name in ["mse", "mae"]:
                metric.update(y_hat, y)
            else:
                metric.update(y_hat.squeeze(-1), y.squeeze(-1))

        # Log loss
        self.log("val/loss", loss, prog_bar=True)
        self.val_loss_best(loss)

        # Log best validation loss
        self.log("val/loss_best", self.val_loss_best.compute(), prog_bar=True)

        # Store outputs
        self.val_step_outputs["preds"].append(y_hat)
        self.val_step_outputs["targets"].append(y)
        self.val_step_outputs["loss"].append(loss)

        return loss

    def test_step(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        loss, y_hat, y = self.model_step(batch)

        # Update metrics
        for name, metric in self.test_metrics.items():
            if name in ["mse", "mae"]:
                metric.update(y_hat, y)
            else:
                metric.update(y_hat.squeeze(-1), y.squeeze(-1))

        # Log loss
        self.log("test/loss", loss, prog_bar=True)
        self.test_loss(loss)

        # Store outputs
        self.test_step_outputs["preds"].append(y_hat)
        self.test_step_outputs["targets"].append(y)
        self.test_step_outputs["loss"].append(loss)

        return loss

    def on_train_epoch_end(self) -> None:
        # Get predictions and targets
        preds = self._gather_step_outputs(self.training_step_outputs, "preds")
        targets = self._gather_step_outputs(self.training_step_outputs, "targets")

        # Only compute and log metrics if we have data
        if len(preds) > 0 and len(targets) > 0:
            # Log metrics without progress bar
            for name, metric in self.train_metrics.items():
                if metric._update_called:  # Only compute if metric has been updated
                    self.log(f"train/{name}", metric.compute(), prog_bar=False)
                    metric.reset()

        # Log predictions to wandb if enabled
        if isinstance(self.logger, L.pytorch.loggers.WandbLogger):
            self._log_predictions(preds, targets, "train")

        # Clear training step outputs
        self.training_step_outputs = {"preds": [], "targets": [], "loss": []}

    def on_validation_epoch_end(self) -> None:
        # Get predictions and targets
        preds = self._gather_step_outputs(self.val_step_outputs, "preds")
        targets = self._gather_step_outputs(self.val_step_outputs, "targets")

        # Log metrics
        for name, metric in self.val_metrics.items():
            self.log(f"val/{name}", metric.compute(), prog_bar=True)
            metric.reset()

        # Log predictions to wandb if enabled
        if isinstance(self.logger, L.pytorch.loggers.WandbLogger):
            self._log_predictions(preds, targets, "val")

        # Clear validation step outputs
        self.val_step_outputs = {"preds": [], "targets": [], "loss": []}

    def on_test_epoch_end(self) -> None:
        # Get predictions and targets
        preds = self._gather_step_outputs(self.test_step_outputs, "preds")
        targets = self._gather_step_outputs(self.test_step_outputs, "targets")

        # Only compute and log metrics if we have data
        if len(preds) > 0 and len(targets) > 0:
            # Log metrics without progress bar
            for name, metric in self.test_metrics.items():
                if metric._update_called:  # Only compute if metric has been updated
                    self.log(f"test/{name}", metric.compute(), prog_bar=False)
                    metric.reset()

        # Log predictions to wandb if enabled
        if isinstance(self.logger, L.pytorch.loggers.WandbLogger):
            self._log_predictions(preds, targets, "test")

        # Clear test step outputs
        self.test_step_outputs = {"preds": [], "targets": [], "loss": []}

    def _log_predictions(self, y: torch.Tensor, y_hat: torch.Tensor, stage: str):
        """Helper method to log predictions to wandb."""
        if isinstance(self.logger, L.pytorch.loggers.WandbLogger):
            self.logger.experiment.log(
                {
                    f"{stage}/predictions": wandb.plot.line_series(
                        xs=range(len(y[0])),
                        ys=[
                            y[0].detach().cpu().numpy(),
                            y_hat[0].detach().cpu().numpy(),
                        ],
                        keys=["Ground Truth", "Prediction"],
                        title="Stock Price Prediction",
                    )
                }
            )
