from typing import Dict, List, Tuple, Optional
import torch
from torchmetrics import MetricCollection, MeanSquaredError, MeanAbsoluteError
from src.metrics.rank_ic import RankIC
from src.metrics.sharpe_ratio import SharpeRatio
from src.models.abstract_forecast import AbstractForecast
from src.models.core.lstm import LSTM
from collections import defaultdict
import pandas as pd
from pathlib import Path


class LSTMForecastModule(AbstractForecast):
    """Lightning module for LSTM forecasting.

    This class implements the Lightning-specific logic for training, validation,
    and testing the LSTM model.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        outputs: Dict[str, List[str]],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        loss: str = "mse_loss",
        *args,
        **kwargs,
    ) -> None:
        # Initialize parent class
        super().__init__(
            outputs=outputs,
            optimizer=optimizer,
            scheduler=scheduler,
            loss=loss,
            *args,
            **kwargs,
        )

        self.save_hyperparameters(logger=False)

        # Initialize the core LSTM model
        self.model = LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
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

        # Initialize step outputs for predictions
        self.training_step_outputs = {key: [] for key in outputs.get("train", [])}
        self.val_step_outputs = {key: [] for key in outputs.get("val", [])}
        self.test_step_outputs = {key: [] for key in outputs.get("test", [])}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.

        Args:
            x: Input tensor of shape [batch_size, sequence_length, input_size]

        Returns:
            Output tensor of shape [batch_size, prediction_length, 1]
        """
        return self.model(x)

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        """Common step for training, validation, and testing.

        Args:
            batch: Tuple of (input, target) tensors

        Returns:
            Tuple of (loss, predictions, targets)
        """
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        return loss, y_hat, y

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss, y_hat, y = self.model_step(batch)

        # Update metrics
        self.train_metrics.update(y_hat, y)

        # Log loss for progress bar
        self.log("train/loss", loss, prog_bar=True, sync_dist=True)

        # Store predictions and targets
        self.training_step_outputs["predictions"].append(y_hat)
        self.training_step_outputs["targets"].append(y)

        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        loss, y_hat, y = self.model_step(batch)

        # Handle NaN values
        if torch.isnan(y_hat).any() or torch.isnan(y).any():
            return loss

        # Update metrics
        self.val_metrics.update(y_hat, y)

        # Store predictions and targets
        self.val_step_outputs["predictions"].append(y_hat)
        self.val_step_outputs["targets"].append(y)

        # Log loss for progress bar
        self.log("val/loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        """Perform test step and update metrics."""
        loss, y_hat, y = self.model_step(batch)

        # Update metrics
        self.test_metrics.update(y_hat, y)

        # Store predictions and targets
        self.test_step_outputs["predictions"].append(y_hat)
        self.test_step_outputs["targets"].append(y)

        # Log loss
        self.log("test/loss", loss, prog_bar=True, sync_dist=True)

        return loss

    def on_train_epoch_start(self):
        """Called at the start of each training epoch."""
        # Initialize outputs dictionary
        self.training_step_outputs = defaultdict(list)

    def on_train_epoch_end(self) -> None:
        """Called at the end of each training epoch."""
        # Log metrics
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

        # Clear outputs
        self.training_step_outputs = {key: [] for key in self.training_step_outputs}

    def on_validation_epoch_start(self):
        """Called at the start of each validation epoch."""
        # Initialize outputs dictionary
        self.val_step_outputs = defaultdict(list)

    def on_validation_epoch_end(self):
        """Called at the end of each validation epoch."""
        # Log metrics
        metrics = self.val_metrics.compute()
        self.log_dict(metrics)
        self.val_metrics.reset()

        # Get validation loss from the logged value
        val_loss = self.trainer.callback_metrics.get("val/loss")
        if val_loss is not None:
            if not hasattr(self, "best_val_loss") or val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self._save_metrics_to_csv(metrics, "val")

        # Clear outputs
        self.val_step_outputs = {key: [] for key in self.val_step_outputs}

    def on_test_epoch_start(self):
        """Called at the start of each test epoch."""
        # Initialize outputs dictionary
        self.test_step_outputs = defaultdict(list)

    def on_test_epoch_end(self) -> None:
        """Called at the end of test epoch."""
        # Log metrics
        metrics = self.test_metrics.compute()
        self.log_dict(metrics)
        self.test_metrics.reset()

        # Save final metrics
        self._save_metrics_to_csv(metrics, "test")

        # Clear outputs
        self.test_step_outputs = {key: [] for key in self.test_step_outputs}

    def _save_metrics_to_csv(self, metrics: dict, stage: str) -> None:
        """Save metrics and predictions to CSV files."""
        output_dir = Path(self.trainer.log_dir)

        # Save metrics
        metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Value"])
        metrics_file = output_dir / f"{stage}_metrics.csv"
        metrics_df.to_csv(metrics_file, index=False)
        print(f"\n{stage.capitalize()} metrics saved to: {metrics_file}")
        print(metrics_df.to_string())

        # Save predictions if available
        step_outputs = getattr(self, f"{stage}_step_outputs")
        if step_outputs["predictions"] and step_outputs["targets"]:
            preds = torch.cat(step_outputs["predictions"]).cpu().numpy()
            targs = torch.cat(step_outputs["targets"]).cpu().numpy()

            preds_df = pd.DataFrame(
                {"predictions": preds.squeeze(), "targets": targs.squeeze()}
            )
            preds_df.to_csv(output_dir / f"{stage}_predictions.csv", index=False)
            print(f"{stage.capitalize()} predictions saved to {stage}_predictions.csv")
