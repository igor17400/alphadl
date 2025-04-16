from typing import Any, Optional
import torch
from torchmetrics import Metric


class SharpeRatio(Metric):
    """Implementation of the Sharpe Ratio (SR) metric."""

    is_differentiable: bool = False
    higher_is_better: bool = True
    full_state_update: bool = False

    def __init__(self, risk_free_rate: float = 0.0, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.risk_free_rate = risk_free_rate
        self.add_state("returns", default=[], dist_reduce_fx="cat")

    def update(self, preds: torch.Tensor, targets: torch.Tensor) -> None:
        """Update state with predictions and targets."""
        # Calculate returns from predictions and targets
        returns = (preds - targets) / targets
        self.returns.append(returns)

    def compute(self) -> torch.Tensor:
        """Compute the Sharpe ratio."""
        if not self.returns:
            return torch.tensor(0.0, device=self.device)
            
        returns = torch.cat(self.returns)
        if len(returns) < 2:
            return torch.tensor(0.0, device=self.device)
            
        # Calculate mean and standard deviation of returns
        mean_return = torch.mean(returns)
        std_return = torch.std(returns)
        
        # Avoid division by zero
        if std_return == 0:
            return torch.tensor(0.0, device=self.device)
            
        # Calculate Sharpe ratio
        sharpe_ratio = (mean_return - self.risk_free_rate) / std_return
        return sharpe_ratio
