import torch
import torch.nn.functional as F


def stock_mixer_loss(prediction, ground_truth, base_price, mask, batch_size, alpha):
    """Custom loss function for StockMixer model combining regression and ranking losses.
    
    This loss function uses a combination of:
    1. Pointwise regression loss (MSE) on return ratios
    2. Pairwise ranking-aware loss that maintains relative order of stock returns
    
    The total loss is computed as: L = L_MSE + α * L_rank
    where:
    - L_MSE: Mean squared error between predicted and actual return ratios
    - L_rank: Ranking loss that penalizes incorrect ordering of stock returns
    - α: Weighting parameter for the ranking loss component
    
    Args:
        prediction (torch.Tensor): Model's predicted stock prices
        ground_truth (torch.Tensor): Actual stock prices (target values)
        base_price (torch.Tensor): Base prices used for return ratio calculation
        mask (torch.Tensor): Mask tensor for handling missing/invalid data
        batch_size (int): Size of the batch
        alpha (float): Weight factor for the ranking loss component
        
    Returns:
        tuple: Contains:
            - loss (torch.Tensor): Combined loss value
            - reg_loss (torch.Tensor): Regression loss component
            - rank_loss (torch.Tensor): Ranking loss component
            - return_ratio (torch.Tensor): Calculated return ratios
    """
    device = prediction.device
    # Create a tensor of ones for pairwise difference calculations
    all_one = torch.ones(batch_size, 1, dtype=torch.float32).to(device)
    
    # Calculate return ratios: (price_t - price_{t-1}) / price_{t-1}
    return_ratio = (prediction - base_price) / base_price
    
    # Compute masked MSE loss on return ratios
    reg_loss = F.mse_loss(return_ratio * mask, ground_truth * mask)
    
    # Calculate pairwise differences for predictions and ground truth
    # Shape: [batch_size, batch_size]
    pre_pw_dif = return_ratio @ all_one.t() - all_one @ return_ratio.t()
    gt_pw_dif = all_one @ ground_truth.t() - ground_truth @ all_one.t()
    
    # Create pairwise mask for valid comparisons
    mask_pw = mask @ mask.t()
    
    # Compute ranking loss using ReLU to penalize only incorrect orderings
    rank_loss = torch.mean(F.relu(pre_pw_dif * gt_pw_dif * mask_pw))
    
    # Combine regression and ranking losses
    loss = reg_loss + alpha * rank_loss
    
    return loss, reg_loss, rank_loss, return_ratio
