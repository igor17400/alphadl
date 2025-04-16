from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from lightning import LightningModule
from torch.nn import MSELoss, L1Loss, HuberLoss
from torchmetrics import MeanMetric, MinMetric, MeanSquaredError, MeanAbsoluteError


class AbstractForecast(LightningModule):
    """Base class for all stock market forecasting models.

    Implements common functionalities shared by all forecasting models.

    Attributes:
        outputs:
            A dictionary of user-defined attributes needed for metric calculation at the end of each `*_step` of the pipeline.
        optimizer:
            Optimizer used for model training.
        scheduler:
            Learning rate scheduler.
    """

    def __init__(
        self,
        outputs: Dict[str, List[str]],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        loss: str = "mse_loss",
        *args,
        **kwargs,
    ) -> None:
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()

        # for tracking best so far validation loss
        self.val_loss_best = MinMetric()

        # Initialize metrics
        self.train_metrics = torch.nn.ModuleDict(
            {"mse": MeanSquaredError(), "mae": MeanAbsoluteError()}
        )
        self.val_metrics = torch.nn.ModuleDict(
            {"mse": MeanSquaredError(), "mae": MeanAbsoluteError()}
        )
        self.test_metrics = torch.nn.ModuleDict(
            {"mse": MeanSquaredError(), "mae": MeanAbsoluteError()}
        )

        # Initialize step outputs for each stage
        self.train_step_outputs = {key: [] for key in outputs.get("train", [])}
        self.val_step_outputs = {key: [] for key in outputs.get("val", [])}
        self.test_step_outputs = {key: [] for key in outputs.get("test", [])}

        # Initialize loss function
        self.criterion = self._get_loss(self.hparams.loss)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_loss_best.reset()

    def model_step(self, batch: Tuple[torch.Tensor, torch.Tensor]):
        raise NotImplementedError

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        raise NotImplementedError

    def on_train_epoch_end(self) -> None:
        raise NotImplementedError

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        raise NotImplementedError

    def on_validation_epoch_end(self) -> None:
        raise NotImplementedError

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int):
        raise NotImplementedError

    def on_test_epoch_end(self) -> None:
        raise NotImplementedError

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

    def _init_embedding(self, filepath: str) -> torch.Tensor:
        return torch.from_numpy(np.load(filepath)).float().to(self.device)

    def _get_loss(self, criterion: str) -> Callable:
        """Returns an instantiated loss object based on the specified criterion."""
        if criterion == "mse_loss":
            return MSELoss()
        elif criterion == "mae_loss":
            return L1Loss()
        elif criterion == "huber_loss":
            return HuberLoss()
        else:
            raise ValueError(f"Loss not defined: {criterion}")

    def _collect_model_outputs(
        self, vector: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """Concatenates model outputs for metric computation."""
        model_output = torch.cat(
            [vector[n][mask[n]] for n in range(mask.shape[0])], dim=0
        )
        return model_output

    def _collect_step_outputs(
        self,
        outputs_dict: Dict[str, List[torch.Tensor]],
        local_vars: Dict[str, torch.Tensor],
    ) -> Dict[str, List[torch.Tensor]]:
        """Collects user-defined attributes of outputs at the end of a `*_step` in dict."""
        for key in outputs_dict.keys():
            if key in local_vars:
                outputs_dict[key].append(local_vars[key])
        return outputs_dict

    def _gather_step_outputs(
        self, outputs_dict: Dict[str, List[torch.Tensor]], key: str
    ) -> torch.Tensor:
        """Gathers and concatenates outputs for a specific key."""
        if key not in outputs_dict:
            raise AttributeError(f"{key} not in {outputs_dict}")

        if not outputs_dict[key]:
            raise ValueError(f"No outputs collected for key: {key}")

        # Ensure all tensors are on the same device
        tensors = [t.to(self.device) for t in outputs_dict[key]]
        return torch.cat(tensors)

    def _clear_epoch_outputs(
        self, outputs_dict: Dict[str, List[torch.Tensor]]
    ) -> Dict[str, List[torch.Tensor]]:
        """Clears the outputs collected during each epoch."""
        return {key: [] for key in outputs_dict}
